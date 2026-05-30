#!/usr/bin/env python3
"""Run the frozen letter classifier on each letter of a rendered word.

Pipeline: text → bbox predictor → predict per-letter boxes → infill model
to render → for each letter, crop the rendered image at that bbox →
letterbox to retina → classifier → top-k softmax.

Usage:
    python3 classify_rendered.py \\
        --bbox-ckpt runs/bbox_predictor_v2/best.pt \\
        --infill-ckpt runs/letter_gan_v9/last.pt \\
        --classifier-ckpt runs/letter_clf_v2/best.pt \\
        --texts "shabby" "hello world" "memory common" \\
        --style-seeds alpha beta gamma \\
        --out eval_output/v9_classified.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from train_bbox_predictor import BboxPredictor
from train_char_recognizer import VOCAB
from train_fractal_infill import FractalInfiller
from train_letter_classifier import LetterClassifier, pad_square_letterbox
from infer_scribe import (
    predict_boxes_px,
    render_word,
    style_from_seed,
    DEFAULT_LH_NORM,
    WORD_H,
    CANVAS,
    V_OFFSET,
)


def crop_letter_from_word(word_arr: np.ndarray, x1: float, x2: float,
                          y1: float, y2: float, retina: int = 64) -> np.ndarray:
    """word_arr is the [H, W] letterbox strip in [0,1] (1=white).
    x1/x2/y1/y2 are in word_arr's local coords.
    Returns retina×retina float in [0,1].
    """
    H, W = word_arr.shape
    ix1 = max(0, int(np.floor(x1)))
    iy1 = max(0, int(np.floor(y1)))
    ix2 = min(W, int(np.ceil(x2)))
    iy2 = min(H, int(np.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.ones((retina, retina), dtype=np.float32)
    crop = word_arr[iy1:iy2, ix1:ix2]
    return pad_square_letterbox(crop, retina)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-ckpt", required=True)
    ap.add_argument("--infill-ckpt", required=True)
    ap.add_argument("--classifier-ckpt", required=True)
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--style-seeds", nargs="+", default=["alpha"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bb_ck = torch.load(args.bbox_ckpt, map_location=device, weights_only=False)
    bb_args = bb_ck.get("args", {})
    predictor = BboxPredictor(
        d_model=bb_args.get("d_model", 128),
        nhead=bb_args.get("nhead", 4),
        layers=bb_args.get("layers", 3),
        max_len=bb_args.get("max_len", 24),
    ).to(device)
    predictor.load_state_dict(bb_ck["model_state_dict"])
    predictor.eval()
    lh_norm = float(bb_args.get("line_height", DEFAULT_LH_NORM))
    max_len = int(bb_args.get("max_len", 24))

    inf_ck = torch.load(args.infill_ckpt, map_location=device, weights_only=False)
    inf_args = inf_ck.get("args", {}) or {}
    noise_dim = int(inf_args.get("noise_dim", 0))
    gen = FractalInfiller(noise_dim=noise_dim).to(device)
    gen.load_state_dict(inf_ck["gen_state_dict"])
    gen.eval()

    clf_ck = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    cargs = clf_ck.get("args", {}) or {}
    classifier = LetterClassifier(
        retina=cargs.get("retina", 64),
        base_ch=cargs.get("base_ch", 64),
    ).to(device)
    classifier.load_state_dict(clf_ck["model_state_dict"])
    classifier.eval()
    retina = cargs.get("retina", 64)

    rows: List[Image.Image] = []

    for text in args.texts:
        for seed in args.style_seeds:
            style_idx = style_from_seed(seed)
            filtered = "".join(c for c in text if c in VOCAB or c == " ")
            words = filtered.split()
            row_panels: List[Image.Image] = []
            for w in words:
                if not w:
                    continue
                if len(w) > max_len:
                    w = w[:max_len]
                boxes_px = predict_boxes_px(
                    w, predictor, device, line_height=lh_norm, max_len=max_len
                )
                with torch.no_grad():
                    word_arr, _word_bbox = render_word(
                        w, boxes_px, gen, device, style_index=style_idx
                    )

                # Crop letters at predicted bboxes (in letterbox coords).
                # Account for the same x_margin used in render_word.
                word_x_min = min(b[0] for b in boxes_px)
                word_x_max = max(b[2] for b in boxes_px)
                word_width = word_x_max - word_x_min
                x_margin = max(0.0, (CANVAS - word_width) / 2.0 - word_x_min)

                # word_arr is the 64-tall letterbox strip starting at x=0.
                # Letter boxes are in word-relative coords; add x_margin for
                # canvas coords, but render_word's word_arr is the 64×CANVAS
                # strip extracted from the canvas, so absolute coords apply.
                W = word_arr.shape[1]
                letter_results = []
                for ch, b in zip(w, boxes_px):
                    lx1 = b[0] + x_margin
                    lx2 = b[2] + x_margin
                    ly1, ly2 = b[1], b[3]
                    crop = crop_letter_from_word(word_arr, lx1, lx2, ly1, ly2, retina)
                    crop_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = classifier(crop_t)
                        probs = F.softmax(logits, dim=-1)[0]
                        top_p, top_i = probs.topk(args.top_k)
                    letter_results.append({
                        "char": ch,
                        "top": [(VOCAB[int(i)], float(p)) for p, i in zip(top_p, top_i)],
                    })

                # Render the word strip with classification annotations beneath.
                strip_h = WORD_H
                ann_h = 16 + 12 * args.top_k
                # Crop the word strip to the (canvas-coord) bbox span so we
                # show only the letters (not the whole 256-px canvas).
                strip_x1 = max(0, int(np.floor(word_x_min + x_margin)))
                strip_x2 = min(word_arr.shape[1],
                               int(np.ceil(word_x_max + x_margin)))
                strip = (word_arr[:, strip_x1:strip_x2] * 255).astype(np.uint8)
                strip_rgb = np.stack([strip] * 3, axis=-1)
                # Build annotation strip beneath: each letter labeled with top-k predictions
                ann = Image.new("RGB", (strip_rgb.shape[1], ann_h), (255, 255, 255))
                d = ImageDraw.Draw(ann)
                # Map each letter's center x in word-strip coords for label placement.
                for ch_info, b in zip(letter_results, boxes_px):
                    # Boxes are in word-relative coords; strip starts at word_x_min
                    # in those coords, so subtract word_x_min for local x.
                    cx_word = 0.5 * (b[0] + b[2]) - word_x_min  # local x in strip
                    pred0 = ch_info["top"][0]
                    color_ok = (0, 130, 0) if pred0[0] == ch_info["char"] else (200, 0, 0)
                    d.text((max(0, int(cx_word) - 8), 0), ch_info["char"],
                           fill=(0, 0, 0))
                    for i, (lab, p) in enumerate(ch_info["top"]):
                        line_y = 14 + i * 12
                        d.text((max(0, int(cx_word) - 14), line_y),
                               f"{lab}:{p:.2f}",
                               fill=color_ok if i == 0 else (80, 80, 80))
                # Stack strip on top of annotations
                tile = Image.new("RGB", (strip_rgb.shape[1], strip_h + ann_h),
                                 (255, 255, 255))
                tile.paste(Image.fromarray(strip_rgb), (0, 0))
                tile.paste(ann, (0, strip_h))
                row_panels.append(tile)

            # Combine words side by side with a gap.
            if not row_panels:
                continue
            gap = 16
            total_w = sum(p.width for p in row_panels) + gap * (len(row_panels) - 1)
            total_h = max(p.height for p in row_panels)
            combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
            x = 0
            for p in row_panels:
                combined.paste(p, (x, 0))
                x += p.width + gap

            # Header for this row.
            hdr = Image.new("RGB", (combined.width, 18), (255, 255, 255))
            d = ImageDraw.Draw(hdr)
            d.text((4, 2), f"{text!r}  style={seed}", fill=(0, 0, 0))
            full = Image.new("RGB", (combined.width, hdr.height + combined.height),
                             (255, 255, 255))
            full.paste(hdr, (0, 0))
            full.paste(combined, (0, hdr.height))
            rows.append(full)

    if not rows:
        raise SystemExit("Nothing rendered")
    max_w = max(r.width for r in rows)
    sep_h = 4
    total_h = sum(r.height for r in rows) + sep_h * (len(rows) - 1)
    final = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for r in rows:
        final.paste(r, (0, y))
        y += r.height + sep_h

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save(out)
    print(f"saved {out}  ({final.size})")


if __name__ == "__main__":
    main()
