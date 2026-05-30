#!/usr/bin/env python3
"""Gradient-ascent bounding box optimization for character localization.

Represents bboxes as continuous (cx, cy, w, h) parameters, applies a
differentiable soft rectangular mask to the (Gaussian-blurred) word image,
and gradient-ascends the sum of character scores so each box migrates
toward a region that looks like some character to the recognizer.

Only boxes whose top character score crosses `--threshold` are drawn in
each frame; the animation shows detections emerge as the optimization runs.

Usage:
    python3 gradient_ascent_bboxes.py \\
        --recognizer runs/char_recog_v2/best.pt \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --word-id a04-072-10-02 \\
        --out eval_output/grad_ascent_major.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from train_char_recognizer import (
    LocalCharRecognizer, letterbox, parse_words_txt, VOCAB,
)


TARGET_H, TARGET_W = 64, 256


def soft_mask(params: torch.Tensor, H: int, W: int, k: float = 2.0) -> torch.Tensor:
    """Differentiable rectangular mask from (cx, cy, w, h).

    params: [B, 4]. Returns [B, H, W] in (0, 1).
    k is edge sharpness: higher = crisper but less gradient info.
    """
    cx, cy, w, h = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    xs = torch.arange(W, device=params.device, dtype=params.dtype)
    ys = torch.arange(H, device=params.device, dtype=params.dtype)
    xl, xr = cx - w / 2, cx + w / 2
    yt, yb = cy - h / 2, cy + h / 2
    mx = (torch.sigmoid(k * (xs[None, :] - xl[:, None]))
          * torch.sigmoid(k * (xr[:, None] - xs[None, :])))
    my = (torch.sigmoid(k * (ys[None, :] - yt[:, None]))
          * torch.sigmoid(k * (yb[:, None] - ys[None, :])))
    return my[:, :, None] * mx[:, None, :]


def render_frame(
    word_arr: np.ndarray,
    params_np: np.ndarray,
    probs_np: np.ndarray,
    threshold: float,
    step: int,
    cell_scale: int = 4,
    only_char: str | None = None,
    targets_np: np.ndarray | None = None,
) -> Image.Image:
    """Draw word + bbox overlays. Filter score = raw prob."""
    H, W = word_arr.shape
    base = Image.fromarray((word_arr * 255).astype(np.uint8), mode="L").convert("RGB")
    base = base.resize((W * cell_scale, H * cell_scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)

    N = params_np.shape[0]
    if targets_np is not None:
        box_letter = targets_np
        box_score = probs_np[np.arange(N), targets_np]
    else:
        box_letter = probs_np.argmax(axis=1)
        box_score = probs_np.max(axis=1)

    palette = [
        (220, 20, 60), (0, 180, 60), (0, 80, 220), (220, 120, 0),
        (180, 0, 180), (0, 180, 180), (120, 60, 0), (80, 80, 80),
    ]
    only_idx = VOCAB.index(only_char) if only_char else None

    for i in range(N):
        if box_score[i] < threshold:
            continue
        if only_idx is not None and box_letter[i] != only_idx:
            continue
        cx, cy, w, h = params_np[i]
        x1, y1 = (cx - w / 2) * cell_scale, (cy - h / 2) * cell_scale
        x2, y2 = (cx + w / 2) * cell_scale, (cy + h / 2) * cell_scale
        ch = VOCAB[box_letter[i]]
        color = palette[box_letter[i] % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=1)
        draw.text((x1 + 1, y1 + 1), ch, fill=color)

    label_h = 22
    canvas = Image.new("RGB", (base.width, base.height + label_h), (255, 255, 255))
    d2 = ImageDraw.Draw(canvas)
    n_shown = int((box_score >= threshold).sum())
    if only_idx is not None:
        n_shown = int(((box_score >= threshold) & (box_letter == only_idx)).sum())
    mode = "targeted" if targets_np is not None else "top-letter"
    label = f"step={step:>3}  {mode} active={n_shown}/{N}  thresh={threshold}"
    if only_char:
        label += f"  only='{only_char}'"
    d2.text((4, 4), label, fill=(0, 0, 0))
    canvas.paste(base, (0, label_h))
    return canvas


def render_stacked_frame(
    word_arr: np.ndarray,
    params_np: np.ndarray,
    probs_np: np.ndarray,
    threshold: float,
    step: int,
    letters: List[str],
    targets_np: np.ndarray | None,
    cell_scale: int = 4,
) -> Image.Image:
    """One word-panel per letter in `letters`, stacked vertically."""
    panels = []
    for ch in letters:
        panels.append(render_frame(
            word_arr, params_np, probs_np, threshold, step,
            cell_scale=cell_scale, only_char=ch, targets_np=targets_np,
        ))
    w = max(p.width for p in panels)
    total_h = sum(p.height for p in panels) + 2 * (len(panels) - 1)
    out = Image.new("RGB", (w, total_h), (235, 235, 235))
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height + 2
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recognizer", required=True)
    p.add_argument("--words-txt", required=True)
    p.add_argument("--words-dir", required=True)
    p.add_argument("--word-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-boxes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--mask-sharpness", type=float, default=0.5)
    p.add_argument("--image-blur-sigma", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-wh", type=float, default=8.0)
    p.add_argument("--max-wh", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-char", type=str, default=None,
                   help="Only draw boxes whose (target or top) letter equals this char")
    p.add_argument("--per-box-target", action="store_true",
                   help="Assign each box a random target letter and ascend only its probability")
    p.add_argument("--target-pool", type=str, default=None,
                   help="Characters to sample targets from (default: entire vocab)")
    p.add_argument("--forward-chunk", type=int, default=2000,
                   help="Chunk size for forward pass to limit memory")
    p.add_argument("--area-exponent", type=float, default=0.0,
                   help="Objective = prob / area^exponent. 0 = no normalization.")
    p.add_argument("--stacked-per-letter", action="store_true",
                   help="Stack one copy of the word per unique letter in the "
                        "transcription; each copy draws only its letter's bboxes. "
                        "Implies --per-box-target with pool = transcription letters.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    model = LocalCharRecognizer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    records = parse_words_txt(Path(args.words_txt))
    rec = next((r for r in records if r["word_id"] == args.word_id), None)
    if rec is None:
        print(f"No record with word_id={args.word_id}")
        return
    img_path = Path(args.words_dir) / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    img = Image.open(img_path).convert("L")
    word_arr = letterbox(img, TARGET_H, TARGET_W)

    # Blur (the "pixels as Gaussians" step): makes sub-pixel box moves smooth.
    word_arr_blur = gaussian_filter(word_arr, sigma=args.image_blur_sigma)

    # Ink-space image as a constant tensor
    ink_img = torch.from_numpy(1.0 - word_arr_blur).to(device).float()  # [H, W]
    ink_img = ink_img[None, None, :, :]  # [1, 1, H, W]

    # Ink bbox to seed initial box positions
    ink_mask = (word_arr < 0.95)
    cols = ink_mask.any(axis=0); rows = ink_mask.any(axis=1)
    x_min = int(np.argmax(cols)); x_max = TARGET_W - int(np.argmax(cols[::-1]))
    y_min = int(np.argmax(rows)); y_max = TARGET_H - int(np.argmax(rows[::-1]))

    N = args.num_boxes
    rng = np.random.default_rng(args.seed)
    init = np.stack([
        rng.uniform(x_min, x_max, N),
        rng.uniform(y_min, y_max, N),
        rng.uniform(args.min_wh, min(args.max_wh, x_max - x_min), N),
        rng.uniform(args.min_wh, min(args.max_wh, y_max - y_min), N),
    ], axis=1)
    params = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.Adam([params], lr=args.lr)

    stacked_letters: List[str] = []
    if args.stacked_per_letter:
        seen = []
        for c in rec["text"]:
            if c in VOCAB and c not in seen:
                seen.append(c)
        stacked_letters = seen
        args.per_box_target = True
        if not args.target_pool:
            args.target_pool = "".join(stacked_letters)
        print(f"  Stacked mode: one panel per letter in {stacked_letters}")

    targets = None
    targets_np = None
    if args.per_box_target:
        pool = list(args.target_pool) if args.target_pool else VOCAB
        pool_indices = np.array([VOCAB.index(c) for c in pool])
        targets_np = rng.choice(pool_indices, size=N)
        targets = torch.from_numpy(targets_np).long().to(device)
        print(f"  Per-box targets sampled from pool of {len(pool)} chars")

    frames: List[Image.Image] = []
    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        probs_chunks = []
        for c_start in range(0, N, args.forward_chunk):
            c_end = min(c_start + args.forward_chunk, N)
            chunk_params = params[c_start:c_end]
            mask = soft_mask(chunk_params, TARGET_H, TARGET_W, k=args.mask_sharpness)
            masked_ink = ink_img * mask[:, None, :, :]
            logits = model(masked_ink)
            probs = torch.sigmoid(logits)

            if step < args.steps:
                if targets is not None:
                    chunk_targets = targets[c_start:c_end]
                    score = torch.gather(
                        probs, 1, chunk_targets.unsqueeze(1),
                    ).squeeze(1)
                else:
                    score = probs.sum(dim=1)
                if args.area_exponent != 0.0:
                    area = chunk_params[:, 2] * chunk_params[:, 3]
                    score = score / area.clamp_min(1.0) ** args.area_exponent
                loss = -score.sum()
                loss.backward()

            probs_chunks.append(probs.detach())

        probs_all = torch.cat(probs_chunks, dim=0)

        if step < args.steps:
            opt.step()
            with torch.no_grad():
                params[:, 0].clamp_(0, TARGET_W - 1)
                params[:, 1].clamp_(0, TARGET_H - 1)
                params[:, 2].clamp_(args.min_wh, args.max_wh)
                params[:, 3].clamp_(args.min_wh, args.max_wh)

        if step % args.save_every == 0:
            p_np = params.detach().cpu().numpy()
            probs_np = probs_all.cpu().numpy()
            if stacked_letters:
                frames.append(render_stacked_frame(
                    word_arr, p_np, probs_np, args.threshold, step,
                    stacked_letters, targets_np=targets_np,
                ))
            else:
                frames.append(render_frame(
                    word_arr, p_np, probs_np, args.threshold, step,
                    only_char=args.only_char, targets_np=targets_np,
                ))
            if targets_np is not None:
                target_probs_np = probs_np[np.arange(N), targets_np]
                n_active = int((target_probs_np >= args.threshold).sum())
                obj = float(target_probs_np.sum())
            else:
                n_active = int((probs_np.max(axis=1) >= args.threshold).sum())
                obj = float(probs_np.sum())
            print(f"  step {step:>3}  active={n_active}/{N}  obj={obj:.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=120, loop=0,
    )
    print(f"\nSaved: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
