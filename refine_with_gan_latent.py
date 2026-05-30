#!/usr/bin/env python3
"""Latent-space variant of refine_with_gan.py.

Instead of optimizing pixels (which finds adversarial noise), this script
optimizes the per-letter NOISE VECTORS that feed into v11's generator.
Each refinement step re-renders the whole word with the current noise
tensor, computes D's score, and backprops to the noise. The output stays
on the generator's natural-image manifold.

Reuses the trained discriminator from refine_with_gan.py (looks for
eval_output/refine/discriminator.pt) and its cached fakes.

Usage:
    python3 refine_with_gan_latent.py \\
        --infill-ckpt runs/letter_gan_v11_extended2/best.pt \\
        --bbox-ckpt runs/bbox_predictor_v2/best.pt \\
        --print-glyphs runs/print_glyphs.pt \\
        --disc-ckpt eval_output/refine/discriminator.pt \\
        --target-text "hello" --target-style alpha \\
        --out-dir eval_output/refine_latent \\
        --steps 80 --lr 0.05
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from train_char_recognizer import VOCAB
from train_fractal_infill import FractalInfiller
from train_letter_gan_infill import CANVAS, V_OFFSET, WORD_H, WORD_W
from infer_scribe import BboxPredictor, predict_boxes_px, style_from_seed
from refine_with_gan import CropD, make_gif


# ============================================================
# Differentiable autoregressive word rendering
# ============================================================

def render_word_diff(
    text: str,
    boxes_px: List[Tuple[float, float, float, float]],
    gen: FractalInfiller,
    device: torch.device,
    style_index: int,
    noises: torch.Tensor,                 # [N, noise_dim] one per letter
    print_glyphs: torch.Tensor | None = None,
    x_margin: float | None = None,
) -> torch.Tensor:
    """Mirror of infer_scribe.render_word but gradient-enabled, returning
    just the 64×256 word strip as a tensor [1, 1, H, W] in [0, 1]."""
    N = len(text)
    canvas = torch.ones(1, 1, CANVAS, CANVAS, device=device)

    word_x_min = min(b[0] for b in boxes_px)
    word_x_max = max(b[2] for b in boxes_px)
    if x_margin is None:
        x_margin = max(0.0, (CANVAS - (word_x_max - word_x_min)) / 2.0 - word_x_min)

    canvas_boxes = [
        (lx1 + x_margin, ly1 + V_OFFSET, lx2 + x_margin, ly2 + V_OFFSET)
        for (lx1, ly1, lx2, ly2) in boxes_px
    ]
    style_t = torch.tensor([style_index], dtype=torch.long, device=device)

    for i, ch in enumerate(text):
        bx1, by1, bx2, by2 = canvas_boxes[i]
        mx1 = int(np.clip(np.floor(bx1), 0, CANVAS))
        my1 = int(np.clip(np.floor(by1), 0, CANVAS))
        mx2 = int(np.clip(np.ceil(bx2), 0, CANVAS))
        my2 = int(np.clip(np.ceil(by2), 0, CANVAS))
        if mx2 <= mx1 or my2 <= my1:
            continue
        bm = torch.zeros(1, 1, CANVAS, CANVAS, device=device)
        bm[0, 0, my1:my2, mx1:mx2] = 1.0
        cx = 0.5 * (bx1 + bx2) / CANVAS
        cy = 0.5 * (by1 + by2) / CANVAS
        bw = (bx2 - bx1) / CANVAS
        bh = (by2 - by1) / CANVAS
        bbox_t = torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32, device=device)

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

        pg = None
        if print_glyphs is not None and ch in VOCAB:
            pg = print_glyphs[VOCAB.index(ch)].to(device).unsqueeze(0)

        pred_delta = gen.forward_infill(
            canvas, bm, char_tokens, char_lengths, style_t, bbox_t,
            noise=noises[i:i+1], print_glyph=pg,
        )
        pred_after = (canvas + pred_delta).clamp(0, 1)
        canvas = torch.where(bm > 0.5, pred_after, canvas)

    return canvas[:, :, V_OFFSET:V_OFFSET + WORD_H]   # [1, 1, 64, 256]


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infill-ckpt", required=True)
    ap.add_argument("--bbox-ckpt", required=True)
    ap.add_argument("--print-glyphs", default=None)
    ap.add_argument("--print-font-idx", type=int, default=0)
    ap.add_argument("--disc-ckpt", required=True,
                    help="Path to a trained CropD state_dict.")
    ap.add_argument("--target-text", default="hello")
    ap.add_argument("--target-style", default="alpha")
    ap.add_argument("--out-dir", default="eval_output/refine_latent")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--anchor", type=float, default=0.01,
                    help="L2 weight on (noise - noise_init) to prevent drift.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load models.
    bb_ck = torch.load(args.bbox_ckpt, map_location=device, weights_only=False)
    bb_args = bb_ck.get("args", {})
    predictor = BboxPredictor(
        d_model=bb_args.get("d_model", 128),
        nhead=bb_args.get("nhead", 4),
        layers=bb_args.get("layers", 3),
        max_len=bb_args.get("max_len", 24),
    ).to(device)
    predictor.load_state_dict(bb_ck["model_state_dict"]); predictor.eval()
    lh_norm = float(bb_args.get("line_height", 0.05))
    max_len_pred = int(bb_args.get("max_len", 24))

    inf_ck = torch.load(args.infill_ckpt, map_location=device, weights_only=False)
    inf_args = inf_ck.get("args", {}) or {}
    noise_dim = int(inf_args.get("noise_dim", 0))
    use_print = bool(inf_args.get("print_glyphs"))
    gen = FractalInfiller(noise_dim=noise_dim, use_print_cond=use_print).to(device)
    gen.load_state_dict(inf_ck["gen_state_dict"]); gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    D = CropD().to(device)
    D.load_state_dict(torch.load(args.disc_ckpt, map_location=device, weights_only=False))
    D.eval()
    for p in D.parameters():
        p.requires_grad = False

    pglyphs = None
    if args.print_glyphs and use_print:
        pc = torch.load(args.print_glyphs, weights_only=False)
        pglyphs = pc["glyphs"][args.print_font_idx].float().unsqueeze(1) / 255.0

    text = "".join(c for c in args.target_text if c in VOCAB)[:max_len_pred]
    print(f"Refining '{text}' for {args.steps} steps with lr={args.lr} anchor={args.anchor}")

    style_idx = style_from_seed(args.target_style)
    boxes_px = predict_boxes_px(text, predictor, device,
                                line_height=lh_norm, max_len=max_len_pred)

    # Initialize per-letter noise vectors. Track init for the anchor.
    N = len(text)
    if noise_dim == 0:
        raise SystemExit("Generator has noise_dim=0; no latent to optimize.")
    noise = torch.randn(N, noise_dim, device=device, requires_grad=True)
    noise_init = noise.detach().clone()

    opt = torch.optim.Adam([noise], lr=args.lr)
    frame_dir = out_dir / f"frames_{text}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Save frame 0.
    with torch.no_grad():
        img0 = render_word_diff(text, boxes_px, gen, device, style_idx,
                                noise, pglyphs)
    Image.fromarray((img0[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(
        frame_dir / f"step_{0:03d}.png")

    for s in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        img = render_word_diff(text, boxes_px, gen, device, style_idx,
                               noise, pglyphs)
        score = D(img).mean()
        loss = (score - 1.0).pow(2).mean()
        anchor_loss = args.anchor * (noise - noise_init).pow(2).mean()
        (loss + anchor_loss).backward()
        opt.step()
        # Save frame.
        Image.fromarray((img.detach()[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(
            frame_dir / f"step_{s:03d}.png")
        if s % 5 == 0 or s == args.steps:
            print(f"  step {s:>3}/{args.steps}  D(x)={float(score):.3f}  "
                  f"||Δnoise||={float((noise-noise_init).norm()):.2f}")

    out_gif = out_dir / f"refine_{text}.gif"
    make_gif(frame_dir, out_gif, duration_ms=80)


if __name__ == "__main__":
    main()
