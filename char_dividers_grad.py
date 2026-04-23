#!/usr/bin/env python3
"""Character segmentation by gradient-descent on N-1 x-dividers.

Given a word of N letters, place N-1 dividers between the word's left and
right ink edges. Each letter i gets a box spanning from divider i-1 to
divider i (with the word edges acting as divider 0 and divider N). Only the
divider x-positions are learnable; y is fixed to the word's vertical
extent.

Loss = -sum_i prob(letter_i | box_i), computed with a soft rectangular
mask so gradients flow to the dividers.

Usage:
    python3 char_dividers_grad.py \\
        --recognizer runs/char_recog_v2/best.pt \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --word-id c03-007-02-07 \\
        --out eval_output/dividers_shabby.gif
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

from train_char_recognizer import LocalCharRecognizer, letterbox, VOCAB, parse_words_txt


TARGET_H, TARGET_W = 64, 256


def soft_mask_from_edges(
    x1: torch.Tensor, x2: torch.Tensor,
    y1: torch.Tensor, y2: torch.Tensor,
    H: int, W: int, k: float = 0.5,
) -> torch.Tensor:
    """Differentiable rectangular mask from edge coords. All inputs are [B]."""
    xs = torch.arange(W, device=x1.device, dtype=x1.dtype)
    ys = torch.arange(H, device=x1.device, dtype=x1.dtype)
    mx = (torch.sigmoid(k * (xs[None, :] - x1[:, None]))
          * torch.sigmoid(k * (x2[:, None] - xs[None, :])))
    my = (torch.sigmoid(k * (ys[None, :] - y1[:, None]))
          * torch.sigmoid(k * (y2[:, None] - ys[None, :])))
    return my[:, :, None] * mx[:, None, :]


def render_frame(
    word_arr: np.ndarray,
    top_positions: np.ndarray,
    bot_positions: np.ndarray,
    y_top, y_bot,  # scalar or per-letter array (len N)
    transcription: str,
    target_probs_np: np.ndarray,
    step,  # int or string (e.g. "refined")
    cell_scale: int = 6,
) -> Image.Image:
    H, W = word_arr.shape
    base = Image.fromarray((word_arr * 255).astype(np.uint8), mode="L").convert("RGB")
    base = base.resize((W * cell_scale, H * cell_scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)

    palette = [
        (220, 20, 60), (0, 150, 60), (0, 80, 220), (200, 110, 0),
        (160, 0, 160), (0, 150, 150), (120, 60, 0), (80, 80, 80),
    ]
    N = len(transcription)
    yt_arr = np.broadcast_to(np.asarray(y_top, dtype=np.float32), (N,)) * cell_scale
    yb_arr = np.broadcast_to(np.asarray(y_bot, dtype=np.float32), (N,)) * cell_scale
    for i in range(N):
        tx1 = top_positions[i] * cell_scale
        tx2 = top_positions[i + 1] * cell_scale
        bx1 = bot_positions[i] * cell_scale
        bx2 = bot_positions[i + 1] * cell_scale
        yt = float(yt_arr[i])
        yb = float(yb_arr[i])
        color = palette[i % len(palette)]
        draw.polygon([(tx1, yt), (tx2, yt), (bx2, yb), (bx1, yb)],
                     outline=color)
        label = f"{transcription[i]}:{target_probs_np[i]:.2f}"
        draw.text((tx1 + 2, yt + 2), label, fill=color)

    # Header with per-letter prob list
    label_h = 22
    canvas = Image.new("RGB", (base.width, base.height + label_h), (255, 255, 255))
    d2 = ImageDraw.Draw(canvas)
    obj = float(target_probs_np.sum())
    step_str = f"{step:>3}" if isinstance(step, int) else f"{step:>8}"
    d2.text((4, 4), f"step={step_str}  sum-prob={obj:.3f}  '{transcription}'",
            fill=(0, 0, 0))
    canvas.paste(base, (0, label_h))
    return canvas


def run_word(args, model, device, rec, words_dir: Path) -> List[Image.Image]:
    transcription = rec["text"]
    if not all(c in VOCAB for c in transcription):
        raise SystemExit(f"Transcription '{transcription}' has chars not in vocab")
    N = len(transcription)
    if N < 2:
        raise SystemExit("Need at least 2 letters for dividers to make sense")

    img_path = words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    img = Image.open(img_path).convert("L")
    word_arr = letterbox(img, TARGET_H, TARGET_W)
    word_arr_blur = gaussian_filter(word_arr, sigma=args.image_blur_sigma)

    ink_img = torch.from_numpy(1.0 - word_arr_blur).to(device).float()[None, None]

    # Ink bbox → word_x1, word_x2, y_top, y_bot
    ink_mask = (word_arr < 0.95)
    cols = ink_mask.any(axis=0); rows = ink_mask.any(axis=1)
    word_x1 = float(np.argmax(cols))
    word_x2 = float(TARGET_W - np.argmax(cols[::-1]))
    y_top = float(np.argmax(rows))
    y_bot = float(TARGET_H - np.argmax(rows[::-1]))
    print(f"[{rec['word_id']} '{transcription}'] ink bbox: x=[{word_x1}, {word_x2}]  "
          f"y=[{y_top}, {y_bot}]")

    # Parameterize N box widths via softmax over N logits (sum to word_width).
    # If --slant, also learn one global slant offset (in pixels): every divider
    # tilts the same amount, so boxes become parallelograms with a uniform
    # shear — matches how human handwriting slant actually works.
    width_logits = torch.zeros(N, dtype=torch.float32,
                               device=device, requires_grad=True)
    slant_shift = torch.zeros((), dtype=torch.float32,
                              device=device, requires_grad=args.slant)
    params = [width_logits]
    if args.slant:
        params.append(slant_shift)

    word_width = float(word_x2 - word_x1)
    wx1_t = torch.tensor(word_x1, device=device, dtype=torch.float32)

    targets = torch.tensor([VOCAB.index(c) for c in transcription],
                           device=device, dtype=torch.long)

    opt = torch.optim.Adam(params, lr=args.lr)

    frames: List[Image.Image] = []
    slack = word_width - N * args.min_width
    if slack <= 0:
        raise SystemExit(f"min_width={args.min_width} × N={N} > word_width={word_width}")

    # Precompute the per-location feature-map logits for the full word.
    # Model is frozen, input is fixed → one forward suffices.
    with torch.no_grad():
        fmap = model.feature_map(ink_img)  # [1, C, H', W']
    _, C, Hp, Wp = fmap.shape
    stride_w = TARGET_W // Wp  # 16
    stride_h = TARGET_H // Hp  # 16
    # Full 2D per-cell target-letter logits: [N, Hp, Wp]
    tgt_logits_2d = fmap[0][targets]

    # Chain of 4 stride-2 kernel-3 padding-1 convs gives RF=31, stride=16 in input
    # pixels. Cell c has center at input pixel 16·c.
    rf_half = 15
    cell_centers = (stride_w * torch.arange(Wp, device=device, dtype=torch.float32))
    row_centers = (stride_h * torch.arange(Hp, device=device, dtype=torch.float32))
    if args.rf_gaussian_sigma is not None:
        sigma = args.rf_gaussian_sigma
        pix = torch.arange(TARGET_W, device=device, dtype=torch.float32)
        rf_matrix = torch.exp(-0.5 * ((pix[None, :] - cell_centers[:, None]) / sigma) ** 2)
        rf_matrix = rf_matrix / rf_matrix.sum(dim=1, keepdim=True)
    else:
        rf_matrix = torch.zeros(Wp, TARGET_W, device=device, dtype=torch.float32)
        for c in range(Wp):
            lo = max(0, stride_w * c - rf_half)
            hi = min(TARGET_W, stride_w * c + rf_half + 1)
            rf_matrix[c, lo:hi] = 1.0 / (hi - lo)

    # Fraction of word height at each feature-row center, clamped to [0, 1].
    y_range = max(1.0, y_bot - y_top)
    row_frac = ((row_centers - y_top) / y_range).clamp(0.0, 1.0)  # [Hp]

    pixels = torch.arange(TARGET_W, device=device, dtype=torch.float32)  # [W]

    for step in range(args.steps + 1):
        widths = args.min_width + torch.softmax(width_logits, dim=0) * slack
        cum = torch.cumsum(widths, dim=0)
        pos = torch.cat([wx1_t[None], wx1_t + cum])  # [N+1]
        half = 0.5 * slant_shift
        top_pos = pos - half
        bot_pos = pos + half
        top_x1, top_x2 = top_pos[:-1], top_pos[1:]
        bot_x1, bot_x2 = bot_pos[:-1], bot_pos[1:]

        # Per-row x-bounds: linear interp between top and bot edges.
        # left/right: [N, Hp]
        left = top_x1[:, None] * (1 - row_frac[None, :]) + bot_x1[:, None] * row_frac[None, :]
        right = top_x2[:, None] * (1 - row_frac[None, :]) + bot_x2[:, None] * row_frac[None, :]

        # Pixel-level x-membership per feature row: [N, Hp, W]
        k = args.mask_sharpness
        pixel_w = (torch.sigmoid(k * (pixels[None, None, :] - left[:, :, None]))
                   * torch.sigmoid(k * (right[:, :, None] - pixels[None, None, :])))
        # Aggregate horizontally to cell columns: [N, Hp, Wp]
        col_w = pixel_w @ rf_matrix.t()

        # Soft max over all 2D cells, gated by RF-weighted membership.
        gated = tgt_logits_2d + torch.log(col_w.clamp(min=1e-8))  # [N, Hp, Wp]
        score = torch.logsumexp(gated.view(N, -1), dim=1)  # [N]
        target_probs = torch.sigmoid(score)

        if step < args.steps:
            loss = -target_probs.sum()
            if args.slant and args.slant_reg > 0:
                loss = loss + args.slant_reg * slant_shift ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if step % args.save_every == 0:
            top_np = top_pos.detach().cpu().numpy()
            bot_np = bot_pos.detach().cpu().numpy()
            tp_np = target_probs.detach().cpu().numpy()
            frames.append(render_frame(
                word_arr, top_np, bot_np, y_top, y_bot,
                transcription, tp_np, step,
            ))
            if step == 0 or step == args.steps:
                divs = np.round(pos.detach().cpu().numpy()[1:-1], 1).tolist()
                s = float(slant_shift.detach().cpu())
                print(f"  step {step:>3}  sum-prob={float(target_probs.sum()):.3f}  "
                      f"divs={divs}  slant_shift={s:+.2f}px")

    if args.refine_y:
        top_np = top_pos.detach().cpu().numpy()
        bot_np = bot_pos.detach().cpu().numpy()
        tp_np = target_probs.detach().cpu().numpy()
        # Tighter ink mask on the blurred image: kills speckle that the
        # permissive word-level mask picks up.
        refine_mask = word_arr_blur < args.refine_y_threshold
        y_top_per = np.empty(N, dtype=np.float32)
        y_bot_per = np.empty(N, dtype=np.float32)
        for i in range(N):
            mid_x1 = 0.5 * (top_np[i] + bot_np[i])
            mid_x2 = 0.5 * (top_np[i + 1] + bot_np[i + 1])
            c1 = int(np.clip(np.floor(mid_x1), 0, TARGET_W))
            c2 = int(np.clip(np.ceil(mid_x2), 0, TARGET_W))
            if c2 <= c1:
                y_top_per[i] = y_top
                y_bot_per[i] = y_bot
                continue
            col_slice = refine_mask[:, c1:c2]
            rows = np.where(col_slice.any(axis=1))[0]
            if rows.size == 0:
                y_top_per[i] = y_top
                y_bot_per[i] = y_bot
            else:
                y_top_per[i] = float(rows.min())
                y_bot_per[i] = float(rows.max()) + 1.0
        m = args.refine_y_margin
        y_top_per = np.clip(y_top_per - m, 0, TARGET_H)
        y_bot_per = np.clip(y_bot_per + m, 0, TARGET_H)
        print(f"  refined y: "
              + ", ".join(f"{transcription[i]}=[{y_top_per[i]:.1f},{y_bot_per[i]:.1f}]"
                          for i in range(N)))
        frames.append(render_frame(
            word_arr, top_np, bot_np, y_top_per, y_bot_per,
            transcription, tp_np, "refined",
        ))

    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recognizer", required=True)
    p.add_argument("--words-txt", required=True)
    p.add_argument("--words-dir", required=True)
    p.add_argument("--word-id", action="append", default=[],
                   help="Word ID to run. Repeat to stack multiple words in one gif.")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--mask-sharpness", type=float, default=0.5)
    p.add_argument("--image-blur-sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-width", type=float, default=6.0,
                   help="Floor on per-box width (pixels) to prevent collapse")
    p.add_argument("--rf-gaussian-sigma", type=float, default=None,
                   help="If set, weight each cell's RF with a Gaussian of this "
                        "std (pixels) centered on the cell — models effective "
                        "RF. If unset, uses uniform weights over the theoretical "
                        "RF (31 px).")
    p.add_argument("--slant", action="store_true",
                   help="Let top-edge and bottom-edge dividers move independently "
                        "so boxes become parallelograms — handles slanted writing.")
    p.add_argument("--slant-reg", type=float, default=0.005,
                   help="L2 penalty on the global slant_shift (in pixels, "
                        "top↔bot offset). Only active with --slant. 0 disables.")
    p.add_argument("--refine-y", action="store_true",
                   help="After opt, replace the shared word-level y-extent with "
                        "per-letter y-extents derived from ink pixels within each "
                        "letter's x-range. Appends one final 'refined' frame.")
    p.add_argument("--refine-y-margin", type=float, default=1.0,
                   help="Pixel margin added above/below each letter's ink extent "
                        "when --refine-y is on.")
    p.add_argument("--refine-y-threshold", type=float, default=0.5,
                   help="Threshold on the blurred image (0=black, 1=white) used "
                        "to detect ink during --refine-y. Lower = stricter.")
    args = p.parse_args()

    if not args.word_id:
        raise SystemExit("Pass at least one --word-id")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch = ckpt_args.get("base_ch", 32)
    final_stride = ckpt_args.get("final_stride", 2)
    model = LocalCharRecognizer(base_ch=base_ch, final_stride=final_stride).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    records = parse_words_txt(Path(args.words_txt))
    words_dir = Path(args.words_dir)

    per_word_frames: List[List[Image.Image]] = []
    for wid in args.word_id:
        rec = next((r for r in records if r["word_id"] == wid), None)
        if rec is None:
            raise SystemExit(f"No record with word_id={wid}")
        per_word_frames.append(run_word(args, model, device, rec, words_dir))

    n_frames = min(len(f) for f in per_word_frames)
    max_w = max(f[0].width for f in per_word_frames)
    row_h = per_word_frames[0][0].height
    total_h = row_h * len(per_word_frames)
    combined: List[Image.Image] = []
    for i in range(n_frames):
        canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        for row_idx, frames in enumerate(per_word_frames):
            f = frames[i]
            canvas.paste(f, ((max_w - f.width) // 2, row_idx * row_h))
        combined.append(canvas)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined[0].save(
        out_path, save_all=True, append_images=combined[1:],
        duration=120, loop=0,
    )
    print(f"\nSaved: {out_path}  ({len(combined)} frames, {len(per_word_frames)} words)")


if __name__ == "__main__":
    main()
