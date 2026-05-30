#!/usr/bin/env python3
"""Recover per-character bounding boxes via occlusion-based saliency.

For each input word image:
  1. Build an image pyramid (multiple scales).
  2. Enumerate candidate boxes at aspect ratios {0.5, 0.75, 1.0, 1.5, 2.0}
     across dense (x, y) positions within the word.
  3. For each candidate: white out everything outside the box, run the
     bag-of-chars recognizer on the word pyramid, record per-class scores.
  4. Per character: NMS using score/area as priority (biases toward small
     tight boxes that contain just one instance of the character).

Outputs an annotated contact sheet for visual inspection.

Usage:
    python3 extract_char_bboxes.py \
        --recognizer runs/char_recog_v1/best.pt \
        --words-txt data/iam_words/iam_words/words.txt \
        --words-dir data/iam_words/iam_words/words \
        --out-dir eval_output/char_bbox_v1 \
        --num-samples 24
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from train_char_recognizer import (
    LocalCharRecognizer, parse_words_txt, letterbox, VOCAB, CHAR_TO_IDX,
)


TARGET_H, TARGET_W = 64, 256


def build_candidate_boxes(
    word_h: int, word_w: int,
    aspect_ratios: List[float],
    height_steps: int = 6,
    min_height: int = 12,
    stride_frac: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """Return candidate (x1, y1, x2, y2) boxes within the word.

    For each aspect ratio, enumerate boxes at multiple heights from min_height
    up to word_h (height_steps steps), and slide each (h, w) box densely in
    both x and y across the word. Stride is proportional to box dimensions.
    """
    boxes = []
    if word_h < min_height or word_w < min_height:
        return boxes
    heights = np.linspace(min_height, word_h, height_steps).astype(int)
    seen = set()
    for h in heights:
        h = int(h)
        for ar in aspect_ratios:
            w = max(min_height, int(h * ar))
            if w > word_w:
                continue
            sx = max(2, int(w * stride_frac))
            sy = max(2, int(h * stride_frac))
            for y1 in range(0, word_h - h + 1, sy):
                for x1 in range(0, word_w - w + 1, sx):
                    box = (x1, y1, x1 + w, y1 + h)
                    if box in seen:
                        continue
                    seen.add(box)
                    boxes.append(box)
    return boxes


def build_candidate_boxes_exhaustive(
    word_h: int, word_w: int,
    aspect_ratios: List[float],
    min_height: int = 12,
) -> List[Tuple[int, int, int, int]]:
    """Truly exhaustive: every integer (x1, y1, h) × every aspect ratio."""
    boxes = []
    if word_h < min_height or word_w < min_height:
        return boxes
    seen = set()
    for h in range(min_height, word_h + 1):
        for ar in aspect_ratios:
            w = int(round(h * ar))
            if w < min_height or w > word_w:
                continue
            for y1 in range(0, word_h - h + 1):
                for x1 in range(0, word_w - w + 1):
                    box = (x1, y1, x1 + w, y1 + h)
                    if box in seen:
                        continue
                    seen.add(box)
                    boxes.append(box)
    return boxes


def filter_boxes_by_edge_ink(
    word_arr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    ink_threshold: float = 0.95,
    max_edge_ink_frac: float = 0.05,
) -> List[Tuple[Tuple[int, int, int, int]]]:
    """Drop boxes where ink passes through the top or bottom edge row.

    word_arr is [H, W] in [0, 1] with 1=white, 0=ink. A pixel counts as ink
    if value < ink_threshold. Box is rejected if either the top row or the
    bottom row contains more than max_edge_ink_frac ink pixels.
    """
    kept = []
    for x1, y1, x2, y2 in boxes:
        top_ink = (word_arr[y1, x1:x2] < ink_threshold).mean()
        bot_ink = (word_arr[y2 - 1, x1:x2] < ink_threshold).mean()
        if top_ink > max_edge_ink_frac or bot_ink > max_edge_ink_frac:
            continue
        kept.append((x1, y1, x2, y2))
    return kept


def score_boxes_chunked(
    model: LocalCharRecognizer,
    word_arr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    device: torch.device,
    pyramid_scales: List[float],
    box_chunk_size: int = 4096,
) -> np.ndarray:
    """Memory-friendly wrapper: scores boxes in chunks of `box_chunk_size`."""
    out = np.zeros((len(boxes), len(VOCAB), len(pyramid_scales)), dtype=np.float32)
    for i in range(0, len(boxes), box_chunk_size):
        chunk = boxes[i:i + box_chunk_size]
        out[i:i + len(chunk)] = score_boxes_per_scale(
            model, word_arr, chunk, device, pyramid_scales,
        )
    return out


def make_pyramid(arr: np.ndarray, scales: List[float]) -> List[np.ndarray]:
    """Resize letterboxed image (H, W) to multiple scales while keeping HxW."""
    out = []
    for s in scales:
        if s == 1.0:
            out.append(arr)
            continue
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        new_w, new_h = max(8, int(img.width * s)), max(8, int(img.height * s))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (arr.shape[1], arr.shape[0]), 255)
        canvas.paste(img, ((arr.shape[1] - new_w) // 2, (arr.shape[0] - new_h) // 2))
        out.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    return out


@torch.no_grad()
def score_boxes_per_scale(
    model: LocalCharRecognizer,
    word_arr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    device: torch.device,
    pyramid_scales: List[float],
    batch_size: int = 64,
) -> np.ndarray:
    """Return [N, NUM_CLASSES, S] sigmoid scores: per-box, per-class, per-scale.

    For each scale, the masked box content is shifted so its center is at the
    image center, then resampled at that scale via affine grid_sample. The
    recognizer is translation-invariant (global pool), so centering the
    content keeps it visible regardless of where the original box sat in the
    image.
    """
    H, W = word_arr.shape
    masked_imgs = []
    centers = []
    for x1, y1, x2, y2 in boxes:
        masked = np.ones_like(word_arr)
        masked[y1:y2, x1:x2] = word_arr[y1:y2, x1:x2]
        masked_imgs.append(masked)
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    masked_stack = np.stack(masked_imgs, axis=0)
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)  # [N, 2]

    N = len(boxes)
    out = np.zeros((N, len(VOCAB), len(pyramid_scales)), dtype=np.float32)

    for sidx, scale in enumerate(pyramid_scales):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            t = torch.from_numpy(masked_stack[start:end]).unsqueeze(1).float().to(device)
            B = t.shape[0]
            cx_norm = (centers_t[start:end, 0] / W) * 2 - 1
            cy_norm = (centers_t[start:end, 1] / H) * 2 - 1
            theta = torch.zeros(B, 2, 3, device=device)
            theta[:, 0, 0] = 1.0 / scale
            theta[:, 1, 1] = 1.0 / scale
            theta[:, 0, 2] = cx_norm
            theta[:, 1, 2] = cy_norm
            grid = F.affine_grid(theta, t.shape, align_corners=False)
            # Convert to ink-space first so out-of-bounds 0-padding = white.
            ink = 1.0 - t
            ink_warped = F.grid_sample(
                ink, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            logits = model(ink_warped)
            out[start:end, :, sidx] = torch.sigmoid(logits).cpu().numpy()
    return out


def effective_box(box: Tuple[int, int, int, int], scale: float) -> Tuple[float, float, float, float]:
    """Effective box for a (box, scale) detection: centered on box center,
    dimensions divided by scale."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) / scale, (y2 - y1) / scale
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def nms_for_class_multiscale(
    boxes: List[Tuple[int, int, int, int]],
    scores_per_scale: np.ndarray,  # [N, S] for one class
    pyramid_scales: List[float],
    score_threshold: float,
    iou_threshold: float = 0.4,
    top_k: int = 16,
    priority_exponent: float = 1.0,
    overlap_metric: str = "iou",
) -> List[Tuple[Tuple[float, float, float, float], float, float]]:
    """NMS over (box, scale) candidates using effective boxes.

    Returns list of (effective_box, score, scale). Effective box = nominal
    box centered on its center, dimensions scaled by 1/scale; effective area
    used for both NMS priority and IoU/IoM overlap.
    """
    cands = []
    N, S = scores_per_scale.shape
    for i in range(N):
        for s_idx in range(S):
            sc = float(scores_per_scale[i, s_idx])
            if sc <= score_threshold:
                continue
            scale = pyramid_scales[s_idx]
            ebox = effective_box(boxes[i], scale)
            ew, eh = ebox[2] - ebox[0], ebox[3] - ebox[1]
            earea = max(1.0, ew * eh)
            priority = sc / earea ** priority_exponent
            cands.append((ebox, sc, scale, earea, priority))

    if not cands:
        return []
    cands.sort(key=lambda c: c[4], reverse=True)

    selected = []
    for ebox, sc, scale, earea, _ in cands:
        x1, y1, x2, y2 = ebox
        suppressed = False
        for kbox, _, _, karea in selected:
            kx1, ky1, kx2, ky2 = kbox
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if overlap_metric == "iom":
                overlap = inter / max(1.0, min(earea, karea))
            else:
                overlap = inter / (earea + karea - inter)
            if overlap > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            selected.append((ebox, sc, scale, earea))
        if len(selected) >= top_k:
            break
    # Drop the area from the returned tuples
    return [(b, s, sc) for b, s, sc, _ in selected]


def render_word_with_boxes(
    word_arr: np.ndarray,
    detections: Dict[str, List[Tuple[Tuple[float, float, float, float], float, float]]],
    transcription: str,
    cell_size: int = 384,
) -> Image.Image:
    """Render the word with effective boxes per character class.

    Each detection is (effective_box, score, scale). Drawn boxes are the
    effective ones (smaller for upscale, larger for downscale).
    """
    base = Image.fromarray((word_arr * 255).astype(np.uint8), mode="L").convert("RGB")
    base = base.resize((cell_size * 2, cell_size // 2), Image.BILINEAR)
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    sx = (cell_size * 2) / word_arr.shape[1]
    sy = (cell_size // 2) / word_arr.shape[0]
    palette = [
        (255, 0, 0), (0, 180, 0), (0, 0, 255), (200, 100, 0),
        (180, 0, 180), (0, 180, 180), (120, 60, 0), (60, 60, 60),
    ]
    color_for = {}
    color_idx = 0
    for ch in sorted(detections.keys()):
        if ch not in color_for:
            color_for[ch] = palette[color_idx % len(palette)]
            color_idx += 1
        for ebox, score, scale in detections[ch]:
            x1, y1, x2, y2 = ebox
            r = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            draw.rectangle(r, outline=color_for[ch], width=2)
            draw.text((r[0] + 1, r[1] + 1), f"{ch}@{scale:g}", fill=color_for[ch])
    label_h = 18
    out = Image.new("RGB", (overlay.width, overlay.height + label_h), (255, 255, 255))
    d2 = ImageDraw.Draw(out)
    d2.text((4, 2), f'"{transcription}"', fill=(0, 0, 0))
    out.paste(overlay, (0, label_h))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recognizer", required=True)
    p.add_argument("--words-txt", required=True)
    p.add_argument("--words-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.4)
    p.add_argument("--priority-exponent", type=float, default=1.0,
                   help="NMS priority = score / area^exponent (1.0 = pure score/area)")
    p.add_argument("--overlap-metric", choices=["iou", "iom"], default="iou",
                   help="iou = intersection/union, iom = intersection/min-area")
    p.add_argument("--pyramid-scales", type=str, default="1.0",
                   help="Comma-separated scales (e.g. '0.75,1.0,1.25')")
    p.add_argument("--exhaustive", action="store_true",
                   help="Fully enumerate every integer (x,y,h) × aspect ratio")
    p.add_argument("--only-word-id", type=str, default=None,
                   help="Process only the word with this record id")
    p.add_argument("--edge-ink-filter", action="store_true",
                   help="Drop candidate boxes whose top/bottom edge intersects ink")
    p.add_argument("--max-edge-ink-frac", type=float, default=0.05,
                   help="Max fraction of top/bottom edge pixels allowed to be ink")
    p.add_argument("--ink-threshold", type=float, default=0.95,
                   help="Pixel value < this counts as ink (IAM scans are anti-aliased)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading recognizer: {args.recognizer}")
    ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    model = LocalCharRecognizer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Parsing words from {args.words_txt}...")
    records = parse_words_txt(Path(args.words_txt))
    if args.only_word_id:
        samples = [r for r in records if r["word_id"] == args.only_word_id]
        if not samples:
            print(f"  No record matches word_id={args.only_word_id}")
            return
    else:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        samples = [r for r in records if 3 <= len(r["text"]) <= 8][:args.num_samples]
    print(f"  Sampled {len(samples)} words")

    aspect_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
    pyramid_scales = [float(s) for s in args.pyramid_scales.split(",") if s.strip()]

    panels = []
    for rec in samples:
        word_path = Path(args.words_dir) / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
        img = Image.open(word_path).convert("L")
        word_arr = letterbox(img, TARGET_H, TARGET_W)

        # Find the actual content extent (tight bbox of the letterboxed word)
        # so candidate boxes don't waste evaluations on white padding.
        ink = (word_arr < 0.7)
        cols = ink.any(axis=0)
        if not cols.any():
            continue
        x_min = int(np.argmax(cols))
        x_max = TARGET_W - int(np.argmax(cols[::-1]))
        rows = ink.any(axis=1)
        y_min = int(np.argmax(rows))
        y_max = TARGET_H - int(np.argmax(rows[::-1]))
        word_w = max(16, x_max - x_min)
        word_h = max(16, y_max - y_min)

        if args.exhaustive:
            boxes_local = build_candidate_boxes_exhaustive(word_h, word_w, aspect_ratios)
        else:
            boxes_local = build_candidate_boxes(word_h, word_w, aspect_ratios)
        # Translate to image coordinates
        boxes = [(x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min)
                 for x1, y1, x2, y2 in boxes_local]
        if not boxes:
            continue
        n_before = len(boxes)
        if args.edge_ink_filter:
            boxes = filter_boxes_by_edge_ink(
                word_arr, boxes,
                ink_threshold=args.ink_threshold,
                max_edge_ink_frac=args.max_edge_ink_frac,
            )
            print(f"    {n_before} → {len(boxes)} candidates after edge-ink filter")
        else:
            print(f"    {len(boxes)} candidate boxes")
        if not boxes:
            continue

        scores = score_boxes_chunked(
            model, word_arr, boxes, device, pyramid_scales,
        )  # [N, NUM_CLASSES, S]

        # Per-class multiscale NMS, but only for chars in the transcription
        present_chars = set(rec["text"])
        detections: Dict[str, List] = {}
        for ch in present_chars:
            ci = CHAR_TO_IDX[ch]
            sel = nms_for_class_multiscale(
                boxes, scores[:, ci, :], pyramid_scales,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
                top_k=rec["text"].count(ch) + 1,
                priority_exponent=args.priority_exponent,
                overlap_metric=args.overlap_metric,
            )
            if sel:
                detections[ch] = sel

        panel = render_word_with_boxes(word_arr, detections, rec["text"])
        panels.append(panel)
        print(f"  '{rec['text']}' ({rec['word_id']}): {sum(len(v) for v in detections.values())} dets across {len(detections)} chars")

    if not panels:
        print("No panels produced.")
        return

    pw, ph = panels[0].size
    pad = 4
    sheet = Image.new("RGB", (pw + 2 * pad, len(panels) * (ph + pad) + pad), (200, 200, 200))
    y = pad
    for p_img in panels:
        sheet.paste(p_img, (pad, y))
        y += ph + pad
    out_path = out_dir / "char_bbox_sheet.png"
    sheet.save(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
