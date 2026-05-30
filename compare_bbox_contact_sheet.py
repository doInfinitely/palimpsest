#!/usr/bin/env python3
"""Render a contact sheet comparing original (init) vs refined bboxes.

For each sampled word, draw the word image scaled up, with:
  - Original bboxes (red dashed rectangles + red dashed dividers)
  - Refined bboxes (green solid rectangles + green solid dividers)

Optionally also overlays GPT-audit action labels per divider so we can
visually check whether the refiner agreed with the audit.

Usage:
    python3 compare_bbox_contact_sheet.py \\
        --init-bbox-jsonl runs/letter_bboxes_v2.jsonl \\
        --refined-bbox-jsonl runs/bbox_refiner_v2/letter_bboxes_refined.jsonl \\
        --recommendations runs/divider_recommendations_v2.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out eval_output/bbox_compare.png \\
        --n 24 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw


def load_bbox_jsonl(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            out[r["word_id"]] = r
    return out


def load_recs(path: Path) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            out.setdefault(r["word_id"], []).append(r)
    return out


def dividers_from_letters(letters: List[Dict], scale_x: float) -> List[float]:
    return [
        0.5 * (letters[i]["x2"] + letters[i + 1]["x1"]) * scale_x
        for i in range(len(letters) - 1)
    ]


def draw_compare(img: Image.Image, init_rec: Dict, refined_rec: Dict,
                 recs: List[Dict], scale: int = 4) -> Image.Image:
    W, H = img.size
    scale_x = W / init_rec["target_w"]
    scale_y = H / init_rec.get("target_h", 64)
    base = img.convert("RGB").resize((W * scale, H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)

    def to_px(L):
        return (L["x1"] * scale_x * scale, L["y1"] * scale_y * scale,
                L["x2"] * scale_x * scale, L["y2"] * scale_y * scale)

    # Original bboxes (red dashed).
    for L in init_rec["letters"]:
        x1, y1, x2, y2 = to_px(L)
        for x in range(int(x1), int(x2), 6):
            draw.line([(x, y1), (min(x + 3, x2), y1)], fill=(220, 0, 0), width=1)
            draw.line([(x, y2), (min(x + 3, x2), y2)], fill=(220, 0, 0), width=1)
        for y in range(int(y1), int(y2), 6):
            draw.line([(x1, y), (x1, min(y + 3, y2))], fill=(220, 0, 0), width=1)
            draw.line([(x2, y), (x2, min(y + 3, y2))], fill=(220, 0, 0), width=1)

    # Refined bboxes (green solid). Some can have inverted edges if the
    # refiner pushed x1>x2 / y1>y2; sort so PIL accepts them.
    for L in refined_rec["letters"]:
        x1, y1, x2, y2 = to_px(L)
        xa, xb = sorted([x1, x2])
        ya, yb = sorted([y1, y2])
        draw.rectangle([xa, ya, xb, yb], outline=(0, 160, 0), width=2)

    # Dividers as vertical guide lines (between bboxes).
    init_divs = dividers_from_letters(init_rec["letters"], scale_x)
    ref_divs = dividers_from_letters(refined_rec["letters"], scale_x)
    for x in init_divs:
        x_px = x * scale
        for y in range(0, H * scale, 8):
            draw.line([(x_px, y), (x_px, y + 4)], fill=(220, 0, 0), width=1)
    for x in ref_divs:
        x_px = x * scale
        draw.line([(x_px, 0), (x_px, H * scale)], fill=(0, 160, 0), width=1)

    # Header with action labels keyed by divider_idx (1..N-1).
    header_h = 16
    canvas = Image.new("RGB", (base.width, base.height + header_h),
                       (255, 255, 255))
    d2 = ImageDraw.Draw(canvas)
    by_idx = {r["divider_idx"]: r for r in recs}
    for i in range(len(init_divs)):
        rec = by_idx.get(i + 1)
        x_mid = (init_divs[i] + ref_divs[i]) / 2 * scale
        if rec is None:
            d2.text((max(2, int(x_mid) - 12), 1), "?", fill=(120, 120, 120))
            continue
        action = rec["action"]
        if action == "ok":
            color = (0, 130, 0)
        elif action in ("left", "right"):
            color = (200, 100, 0)
        elif action == "split":
            color = (140, 0, 140)
        else:
            color = (120, 120, 120)
        # Arrow showing the GPT suggestion + actual refiner movement.
        delta_px = ref_divs[i] - init_divs[i]
        sign = "+" if delta_px > 0 else ""
        text = f"{action[:4]} {sign}{delta_px:.1f}"
        d2.text((max(2, int(x_mid) - 24), 1), text, fill=color)
    canvas.paste(base, (0, header_h))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-bbox-jsonl", required=True)
    ap.add_argument("--refined-bbox-jsonl", required=True)
    ap.add_argument("--recommendations", default=None,
                    help="Optional: GPT-audit recommendations for header labels.")
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--min-letters", type=int, default=3,
                    help="Sample only words with at least this many letters.")
    ap.add_argument("--require-recs", action="store_true",
                    help="Sample only words that have recommendations.")
    args = ap.parse_args()

    init_by_word = load_bbox_jsonl(Path(args.init_bbox_jsonl))
    refined_by_word = load_bbox_jsonl(Path(args.refined_bbox_jsonl))
    recs_by_word: Dict[str, List[Dict]] = {}
    if args.recommendations:
        recs_by_word = load_recs(Path(args.recommendations))

    # Sample word_ids that have both init+refined and (optionally) recs.
    candidates = [
        wid for wid in init_by_word
        if wid in refined_by_word
        and len(init_by_word[wid]["letters"]) >= args.min_letters
        and (not args.require_recs or wid in recs_by_word)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sampled = candidates[: args.n]
    print(f"Sampling {len(sampled)} from {len(candidates)} candidates "
          f"(init total={len(init_by_word)}, refined={len(refined_by_word)})")

    panels: List[Image.Image] = []
    for wid in sampled:
        init = init_by_word[wid]
        refined = refined_by_word[wid]
        img_path = (Path(args.words_dir) / init["form"] / init["line"]
                    / f"{wid}.png")
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue
        recs = sorted(recs_by_word.get(wid, []), key=lambda r: r["divider_idx"])
        panel = draw_compare(img, init, refined, recs)
        cap_h = 14
        capped = Image.new("RGB", (panel.width, panel.height + cap_h),
                           (255, 255, 255))
        d3 = ImageDraw.Draw(capped)
        d3.text((4, 1), f"{wid}  '{init['text']}'", fill=(0, 0, 0))
        capped.paste(panel, (0, cap_h))
        panels.append(capped)

    if not panels:
        raise SystemExit("No panels rendered.")
    cols = max(1, args.cols)
    rows = math.ceil(len(panels) / cols)
    cell_pad = 8
    row_imgs: List[Image.Image] = []
    for r in range(rows):
        cells = panels[r * cols:(r + 1) * cols]
        cell_w = max(c.width for c in cells)
        cell_h = max(c.height for c in cells)
        row = Image.new("RGB",
                        (cell_w * cols + cell_pad * (cols - 1) if cells else 1,
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
    print(f"Saved → {out}  ({sheet.size})")


if __name__ == "__main__":
    main()
