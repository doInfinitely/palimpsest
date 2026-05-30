#!/usr/bin/env python3
"""Render a contact sheet of words with original vs ink-density-refined
dividers.

Refinement logic per divider:
  action=ok                   → no change
  action=right                → shift LEFT (search ±W left for local ink-min)
  action=left                 → shift RIGHT (search ±W right for local ink-min)
  action=split                → snap to nearest local ink-min within ±W
  action=left_or_split        → snap to nearest local ink-min within ±W
  action=right_or_split       → snap to nearest local ink-min within ±W
  action=unsure               → no change

Usage:
    python3 divider_contact_sheet.py \\
        --bbox-jsonl runs/letter_bboxes_v2.jsonl \\
        --recommendations runs/divider_recommendations.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out eval_output/divider_contact.png \\
        --n 24 --seed 0 \\
        --search-window 15
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d


def col_ink_density(arr_gray: np.ndarray, ink_thresh: float = 0.5) -> np.ndarray:
    """Per-column count of ink pixels (1=ink). arr_gray in [0,1] (1=white)."""
    ink = (arr_gray < ink_thresh).astype(np.float32)
    col = ink.sum(axis=0)
    # Smooth a bit so we find broad valleys, not single-column noise.
    return gaussian_filter1d(col, sigma=1.0)


def find_local_ink_min(col_ink: np.ndarray, x_center: float, half_window: int,
                       direction: str = "any") -> int:
    """Find the x within [x_center - half_window, x_center + half_window]
    that minimizes ink density. direction = 'left' (only x <= center),
    'right' (only x >= center), or 'any'."""
    W = len(col_ink)
    lo = max(0, int(round(x_center - half_window)))
    hi = min(W, int(round(x_center + half_window + 1)))
    if direction == "left":
        hi = min(hi, int(round(x_center)) + 1)
    elif direction == "right":
        lo = max(lo, int(round(x_center)))
    if hi <= lo:
        return int(round(x_center))
    window = col_ink[lo:hi]
    return int(lo + np.argmin(window))


def refine_divider(action: str, current_x: float, col_ink: np.ndarray,
                   half_window: int) -> int:
    if action == "ok" or action == "unsure":
        return int(round(current_x))
    if action == "right":
        # divider too far right → shift LEFT
        return find_local_ink_min(col_ink, current_x, half_window, "left")
    if action == "left":
        # divider too far left → shift RIGHT
        return find_local_ink_min(col_ink, current_x, half_window, "right")
    # split / left_or_split / right_or_split → snap globally within window
    return find_local_ink_min(col_ink, current_x, half_window, "any")


def enforce_monotonic(orig: List[float], proposed: List[int],
                      min_spacing: int = 8, W: int = 1000000) -> List[int]:
    """Project proposed divider positions to satisfy:
       new[i+1] - new[i] >= min_spacing.
    Strategy: greedy left-to-right; if a proposal would crowd its left
    neighbor, fall back to the original position (or push to min_spacing
    past the left neighbor, whichever is farther right).
    """
    out: List[int] = []
    for i, p in enumerate(proposed):
        if i == 0:
            out.append(int(round(p)))
            continue
        floor = out[-1] + min_spacing
        if p >= floor:
            out.append(int(round(p)))
        else:
            # Crowded: try original; if still too tight, snap to floor.
            if int(round(orig[i])) >= floor:
                out.append(int(round(orig[i])))
            else:
                out.append(min(W - 1, floor))
    return out


def draw_word(img: Image.Image, divs_orig: List[float], divs_new: List[float],
              labels: List[Dict], scale: int = 4) -> Image.Image:
    """Render the word image scaled, with original (red) + refined (green)
    dividers overlaid as vertical lines, plus a header with action labels."""
    W, H = img.size
    base = img.convert("RGB").resize((W * scale, H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)
    for x in divs_orig:
        x_px = x * scale
        for y in range(0, H * scale, 6):
            draw.line([(x_px, y), (x_px, y + 3)], fill=(220, 0, 0), width=2)
    for x in divs_new:
        x_px = x * scale
        draw.line([(x_px, 0), (x_px, H * scale)], fill=(0, 160, 0), width=2)
    # Header strip with action labels (one per divider).
    header_h = 14
    canvas = Image.new("RGB", (base.width, base.height + header_h),
                       (255, 255, 255))
    d2 = ImageDraw.Draw(canvas)
    for i, lab in enumerate(labels):
        x = int((divs_orig[i] + divs_new[i]) / 2 * scale)
        action = lab["action"]
        conf = lab.get("combined_confidence", 0)
        if action == "ok":
            color = (0, 130, 0)
        elif action in ("left", "right"):
            color = (200, 100, 0)
        elif action == "split":
            color = (140, 0, 140)
        elif action == "unsure":
            color = (120, 120, 120)
        else:
            color = (180, 80, 0)
        text = action[:4]
        d2.text((max(2, x - 10), 1), text, fill=color)
    canvas.paste(base, (0, header_h))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--recommendations", required=True,
                    help="Output of analyze_divider_audit.py")
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24,
                    help="Number of words on the contact sheet.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-window", type=int, default=15,
                    help="Half-window for ink-density refinement (in original-image px).")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--min-spacing", type=int, default=8,
                    help="Minimum pixel spacing between adjacent refined "
                         "dividers (prevents collapse).")
    args = ap.parse_args()

    # Load audited bbox records.
    bbox_records: Dict[str, Dict] = {}
    with open(args.bbox_jsonl) as f:
        for raw in f:
            raw = raw.strip()
            if not raw: continue
            r = json.loads(raw)
            bbox_records[r["word_id"]] = r

    # Load recommendations grouped by word_id.
    recs_by_word: Dict[str, List[Dict]] = {}
    with open(args.recommendations) as f:
        for raw in f:
            raw = raw.strip()
            if not raw: continue
            r = json.loads(raw)
            recs_by_word.setdefault(r["word_id"], []).append(r)

    audited = list(recs_by_word.keys())
    rng = random.Random(args.seed)
    rng.shuffle(audited)
    audited = audited[: args.n]
    print(f"Sampling {len(audited)} words from {len(recs_by_word)} audited.")

    panels: List[Image.Image] = []
    for wid in audited:
        rec = bbox_records.get(wid)
        if rec is None: continue
        img_path = (Path(args.words_dir) / rec["form"] / rec["line"]
                    / f"{wid}.png")
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue
        arr = np.asarray(img, dtype=np.float32) / 255.0
        scale_x = img.size[0] / rec["target_w"]
        # Original dividers (between consecutive letters), in original-image px.
        letters = rec["letters"]
        divs_orig = [
            0.5 * (letters[i]["x2"] + letters[i + 1]["x1"]) * scale_x
            for i in range(len(letters) - 1)
        ]
        col_ink = col_ink_density(arr)
        # Build refinement using recommendations sorted by divider_idx.
        recs = sorted(recs_by_word[wid], key=lambda r: r["divider_idx"])
        proposed: List[int] = []
        labels = []
        for i, d in enumerate(divs_orig):
            rec_r = next((r for r in recs if r["divider_idx"] == i + 1), None)
            if rec_r is None:
                proposed.append(int(round(d)))
                labels.append({"action": "unsure", "combined_confidence": 0})
                continue
            new_x = refine_divider(rec_r["action"], d, col_ink, args.search_window)
            proposed.append(int(new_x))
            labels.append(rec_r)
        # Enforce monotonicity + min spacing so dividers can't collapse
        # onto each other.
        divs_new = [float(x) for x in enforce_monotonic(
            divs_orig, proposed, min_spacing=args.min_spacing, W=img.size[0])]
        panel = draw_word(img, divs_orig, divs_new, labels)
        # Add a small caption above with word_id + text.
        cap_h = 14
        capped = Image.new("RGB", (panel.width, panel.height + cap_h),
                           (255, 255, 255))
        d3 = ImageDraw.Draw(capped)
        d3.text((4, 1), f"{wid}  '{rec['text']}'", fill=(0, 0, 0))
        capped.paste(panel, (0, cap_h))
        panels.append(capped)

    if not panels:
        raise SystemExit("No panels rendered.")

    # Lay out on a grid. Each row contains `cols` panels, padded to the
    # widest panel in that row.
    cols = max(1, args.cols)
    rows = math.ceil(len(panels) / cols)
    cell_pad = 8
    row_imgs: List[Image.Image] = []
    for r in range(rows):
        cells = panels[r * cols:(r + 1) * cols]
        # Pad cells to same width.
        cell_w = max(c.width for c in cells)
        cell_h = max(c.height for c in cells)
        row = Image.new("RGB",
                        (cell_w * cols + cell_pad * (cols - 1) if len(cells) > 0 else 1,
                         cell_h),
                        (255, 255, 255))
        x = 0
        for c in cells:
            padded = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
            padded.paste(c, (0, 0))
            row.paste(padded, (x, 0))
            x += cell_w + cell_pad
        row_imgs.append(row)

    sheet_w = max(r.width for r in row_imgs)
    sheet_h = sum(r.height for r in row_imgs) + cell_pad * (rows - 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    y = 0
    for r in row_imgs:
        sheet.paste(r, (0, y))
        y += r.height + cell_pad

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Saved contact sheet → {out}  ({sheet.size})")


if __name__ == "__main__":
    main()
