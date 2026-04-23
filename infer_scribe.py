#!/usr/bin/env python3
"""End-to-end scribe inference: text → handwritten image.

For each word:
  1. Autoregressively predict per-letter bboxes from the char sequence
     using the trained bbox predictor.
  2. In a 256×256 "word canvas" matching the infill model's training
     geometry, loop letters left-to-right; each step masks the next
     letter's bbox and runs the infill generator to fill it.
  3. Extract the 64-tall letterbox from the word canvas and paste into
     a longer line canvas, advancing the cursor.

Line spacing between words is algorithmic (fixed gap). This version
writes a single line.

Usage:
    python3 infer_scribe.py \\
        --bbox-ckpt runs/bbox_predictor_v1/best.pt \\
        --infill-ckpt runs/letter_gan_v1/best.pt \\
        --text "hello world" \\
        --out eval_output/scribe_hello.png
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from train_bbox_predictor import BboxPredictor, char_to_tok, PAD
from train_char_recognizer import VOCAB
from train_fractal_infill import FractalInfiller


WORD_H, WORD_W = 64, 256
CANVAS = 256
V_OFFSET = (CANVAS - WORD_H) // 2  # 96
DEFAULT_LH_NORM = 40.0  # must match train_bbox_predictor.py --line-height


# ------------------------------------------------------------------
# Bbox prediction (autoregressive)
# ------------------------------------------------------------------

@torch.no_grad()
def predict_boxes_px(
    chars: str,
    predictor: BboxPredictor,
    device: torch.device,
    line_height: float = DEFAULT_LH_NORM,
    max_len: int = 24,
) -> List[Tuple[float, float, float, float]]:
    """Predict per-letter boxes in letterbox-pixel coords (0..WORD_W, 0..WORD_H).

    Boxes are relative to word-start (word_x1=0, word_y_top=0).
    """
    N = len(chars)
    chars_t = torch.full((1, max_len), PAD, dtype=torch.long, device=device)
    for i, c in enumerate(chars):
        chars_t[0, i] = char_to_tok(c)
    pad_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    pad_mask[0, N:] = True  # True where padded

    prev_boxes = torch.zeros(1, max_len, 4, device=device, dtype=torch.float32)
    out = []
    for i in range(N):
        pred = predictor(chars_t, prev_boxes, pad_mask)  # [1, L, 4]
        box = pred[0, i].cpu().numpy()
        out.append(tuple(float(v) * line_height for v in box))
        if i + 1 < max_len:
            prev_boxes[0, i + 1] = pred[0, i]
    # Clamp to 64-tall letterbox bounds (y dims).
    clipped = []
    for (x1, y1, x2, y2) in out:
        clipped.append((x1, max(0.0, y1), x2, min(float(WORD_H), y2)))
    return clipped


# ------------------------------------------------------------------
# Infill inference for one word
# ------------------------------------------------------------------

@torch.no_grad()
def render_word(
    text: str,
    boxes_px: List[Tuple[float, float, float, float]],
    gen: FractalInfiller,
    device: torch.device,
    style_index: int = 0,
    x_margin: int | None = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Render one word into a 64×256 letterbox. Returns (word_arr, word_bbox_in_letterbox).

    word_arr: [64, 256] float in [0, 1], 1=white.
    word_bbox_in_letterbox: (x1, y1, x2, y2) spanning all letters.

    If x_margin is None, it is auto-set to center the word in the 256-wide
    canvas (so wide words stay inside; narrow words get more padding).
    """
    # Build 256×256 word canvas; word sits in V_OFFSET..V_OFFSET+64 rows.
    canvas = np.ones((CANVAS, CANVAS), dtype=np.float32)
    N = len(text)
    canvas_t = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).to(device)

    style_t = torch.tensor([style_index], dtype=torch.long, device=device)

    # Decide horizontal margin: center the word if possible, else pin at 0.
    word_x_min = min(b[0] for b in boxes_px)
    word_x_max = max(b[2] for b in boxes_px)
    word_width = word_x_max - word_x_min
    if x_margin is None:
        x_margin = max(0.0, (CANVAS - word_width) / 2.0 - word_x_min)

    # Track letters' absolute boxes in 256 canvas coords.
    canvas_boxes = []
    for i, (lx1, ly1, lx2, ly2) in enumerate(boxes_px):
        canvas_boxes.append((
            lx1 + x_margin,
            ly1 + V_OFFSET,
            lx2 + x_margin,
            ly2 + V_OFFSET,
        ))

    for i, ch in enumerate(text):
        bx1, by1, bx2, by2 = canvas_boxes[i]
        # Clamp & snap to pixel grid for mask drawing.
        mx1 = int(np.clip(np.floor(bx1), 0, CANVAS))
        my1 = int(np.clip(np.floor(by1), 0, CANVAS))
        mx2 = int(np.clip(np.ceil(bx2), 0, CANVAS))
        my2 = int(np.clip(np.ceil(by2), 0, CANVAS))
        if mx2 <= mx1 or my2 <= my1:
            continue

        # Build bbox_mask.
        bm = torch.zeros(1, 1, CANVAS, CANVAS, device=device)
        bm[0, 0, my1:my2, mx1:mx2] = 1.0

        # Normalized (cx, cy, w, h) bbox for FiLM conditioning.
        cx = 0.5 * (bx1 + bx2) / CANVAS
        cy = 0.5 * (by1 + by2) / CANVAS
        bw = (bx2 - bx1) / CANVAS
        bh = (by2 - by1) / CANVAS
        bbox_t = torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32, device=device)

        # char_tokens: prev + target + next as UTF-8 bytes (PAD=0 at edges).
        prev_c = text[i - 1] if i > 0 else None
        next_c = text[i + 1] if i < N - 1 else None
        toks = []
        for c in (prev_c, ch, next_c):
            if c is None:
                toks.append(0)
            else:
                b = c.encode("utf-8")
                toks.append(b[0] if len(b) == 1 else 0)
        char_tokens = torch.tensor([toks], dtype=torch.long, device=device)
        char_lengths = torch.tensor([len(toks)], dtype=torch.long, device=device)

        # Before = current canvas with the target region painted white (already is).
        before = canvas_t

        pred_delta = gen.forward_infill(
            before, bm, char_tokens, char_lengths, style_t, bbox_t,
        )
        pred_after = (before + pred_delta).clamp(0, 1)
        # Only commit pixels inside the mask (keep outside untouched).
        canvas_t = torch.where(bm > 0.5, pred_after, canvas_t)

    # Extract the 64×256 word strip.
    word_arr = canvas_t[0, 0, V_OFFSET:V_OFFSET + WORD_H].cpu().numpy()
    # Word bbox across all letters (in letterbox coords).
    lx1s = [b[0] for b in boxes_px]
    lx2s = [b[2] for b in boxes_px]
    ly1s = [b[1] for b in boxes_px]
    ly2s = [b[3] for b in boxes_px]
    word_bbox = (
        max(0.0, min(lx1s) + x_margin),
        max(0.0, min(ly1s)),
        min(float(CANVAS), max(lx2s) + x_margin),
        min(float(WORD_H), max(ly2s)),
    )
    return word_arr, word_bbox


def style_from_seed(seed: str, num_styles: int = 64) -> int:
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)
    return h % num_styles


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-ckpt", required=True)
    ap.add_argument("--infill-ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style-seed", default="default",
                    help="String hashed into a style_index for the infill model.")
    ap.add_argument("--word-gap", type=int, default=16,
                    help="Pixel gap between words on the output line.")
    ap.add_argument("--left-margin", type=int, default=24)
    ap.add_argument("--right-margin", type=int, default=24)
    ap.add_argument("--top-margin", type=int, default=8)
    ap.add_argument("--bottom-margin", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load bbox predictor.
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

    # Load infill generator.
    inf_ck = torch.load(args.infill_ckpt, map_location=device, weights_only=False)
    gen = FractalInfiller().to(device)
    gen.load_state_dict(inf_ck["gen_state_dict"])
    gen.eval()

    style_index = style_from_seed(args.style_seed)
    print(f"Using style_index={style_index} (seed={args.style_seed!r})")

    words = args.text.split()
    if not words:
        raise SystemExit("No words in --text")

    # Render each word.
    strips: List[np.ndarray] = []
    for w in words:
        # Drop chars not in the bbox predictor's vocab (letterbox coords can't
        # place them); leave a visible gap by padding a blank strip.
        filtered = "".join(c for c in w if c in VOCAB)
        if not filtered:
            # Whole word unsupported — inject an empty gap.
            strips.append(np.ones((WORD_H, 32), dtype=np.float32))
            continue
        if len(filtered) > max_len:
            print(f"  WARN: '{w}' has {len(filtered)} chars > max_len={max_len}, truncating")
            filtered = filtered[:max_len]

        boxes_px = predict_boxes_px(filtered, predictor, device,
                                    line_height=lh_norm, max_len=max_len)
        word_arr, word_bbox = render_word(
            filtered, boxes_px, gen, device, style_index=style_index,
        )
        # Crop the word strip to tight horizontal bounds (word_bbox x coords),
        # trimmed to the 256-wide letterbox.
        x_lo = max(0, int(np.floor(word_bbox[0])))
        x_hi = min(WORD_W, int(np.ceil(word_bbox[2])))
        strips.append(word_arr[:, x_lo:x_hi])
        print(f"  '{w}' → {len(filtered)} chars, width={x_hi - x_lo} px")

    # Assemble line canvas.
    line_w = args.left_margin + args.right_margin + sum(s.shape[1] for s in strips) \
             + args.word_gap * max(0, len(strips) - 1)
    line_h = args.top_margin + args.bottom_margin + WORD_H
    line = np.ones((line_h, line_w), dtype=np.float32)
    x = args.left_margin
    for i, s in enumerate(strips):
        line[args.top_margin:args.top_margin + WORD_H, x:x + s.shape[1]] = s
        x += s.shape[1]
        if i < len(strips) - 1:
            x += args.word_gap

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(line, 0, 1) * 255).astype(np.uint8), mode="L").save(out_path)
    print(f"Saved: {out_path}  ({line_w}×{line_h})")


if __name__ == "__main__":
    main()
