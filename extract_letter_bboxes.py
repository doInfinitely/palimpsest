#!/usr/bin/env python3
"""Run the gradient divider over the entire IAM ok-word corpus.

Batched version of char_dividers_grad.py. Processes B words in parallel:
one recognizer forward per batch, one Adam optimizer over a padded
[B, N_max] tensor of width logits. Per-word scalars (word_x1, y_top, ...)
become length-B vectors. Padded letter slots are masked in softmax, cum,
scoring, and loss so they don't perturb real slots.

After the opt loop, per-letter y-extents are refined by scanning ink in
each letter's x-range on the blurred image.

Output: one JSONL line per successfully-processed word.

Usage:
    python3 extract_letter_bboxes.py \\
        --recognizer runs/char_recog_v4/best.pt \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --out runs/letter_bboxes_v1.jsonl \\
        --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from train_char_recognizer import LocalCharRecognizer, letterbox, VOCAB, parse_words_txt


TARGET_H, TARGET_W = 64, 256


def prepare_word(rec: Dict, words_dir: Path, min_width: float, blur_sigma: float):
    """Returns a dict of arrays/scalars needed for the batched divider, or None to skip."""
    text = rec["text"]
    N = len(text)
    if N < 2:
        return None

    img_path = words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    try:
        img = Image.open(img_path).convert("L")
    except (FileNotFoundError, OSError):
        return None
    word_arr = letterbox(img, TARGET_H, TARGET_W)
    word_arr_blur = gaussian_filter(word_arr, sigma=blur_sigma)

    ink_mask = word_arr < 0.95
    cols = ink_mask.any(axis=0)
    rows = ink_mask.any(axis=1)
    if not cols.any() or not rows.any():
        return None
    word_x1 = float(np.argmax(cols))
    word_x2 = float(TARGET_W - np.argmax(cols[::-1]))
    y_top = float(np.argmax(rows))
    y_bot = float(TARGET_H - np.argmax(rows[::-1]))

    word_width = word_x2 - word_x1
    slack = word_width - N * min_width
    if slack <= 0:
        return None

    return {
        "rec": rec,
        "N": N,
        "targets": np.array([VOCAB.index(c) for c in text], dtype=np.int64),
        "word_arr": word_arr,
        "word_arr_blur": word_arr_blur,
        "word_x1": word_x1,
        "word_x2": word_x2,
        "y_top": y_top,
        "y_bot": y_bot,
        "slack": float(slack),
    }


def build_rf_matrix(Wp: int, stride_w: int, device, dtype=torch.float32) -> torch.Tensor:
    """Uniform weights over the theoretical RF (31 px, as in char_dividers_grad.py)."""
    rf_half = 15
    rf_matrix = torch.zeros(Wp, TARGET_W, device=device, dtype=dtype)
    for c in range(Wp):
        lo = max(0, stride_w * c - rf_half)
        hi = min(TARGET_W, stride_w * c + rf_half + 1)
        rf_matrix[c, lo:hi] = 1.0 / (hi - lo)
    return rf_matrix


def run_batch(prepared: List[Dict], model, device, args, rf_matrix, pixels):
    """Run the batched opt loop over `prepared` words. Returns list of result dicts."""
    B = len(prepared)
    N_max = max(p["N"] for p in prepared)

    # Stack images into [B, 1, H, W].
    ink_np = np.stack([1.0 - p["word_arr_blur"] for p in prepared], axis=0)[:, None]
    ink_img = torch.from_numpy(ink_np).to(device).float()

    # Per-word scalars → [B] tensors.
    wx1 = torch.tensor([p["word_x1"] for p in prepared], device=device, dtype=torch.float32)
    y_top = torch.tensor([p["y_top"] for p in prepared], device=device, dtype=torch.float32)
    y_bot = torch.tensor([p["y_bot"] for p in prepared], device=device, dtype=torch.float32)
    slack = torch.tensor([p["slack"] for p in prepared], device=device, dtype=torch.float32)
    N_vec = torch.tensor([p["N"] for p in prepared], device=device, dtype=torch.long)

    # Padded targets [B, N_max], padded positions get 0 (dummy, masked in loss).
    targets = torch.zeros(B, N_max, device=device, dtype=torch.long)
    mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)
    for b, p in enumerate(prepared):
        targets[b, :p["N"]] = torch.from_numpy(p["targets"]).to(device)
        mask[b, :p["N"]] = True
    mask_f = mask.float()

    # One recognizer forward for the whole batch.
    with torch.no_grad():
        fmap = model.feature_map(ink_img)  # [B, C, Hp, Wp]
    _, C, Hp, Wp = fmap.shape
    stride_h = TARGET_H // Hp

    # Gather per-letter target-class feature maps: [B, N_max, Hp, Wp].
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N_max)
    tgt_logits_2d = fmap[batch_idx, targets]  # [B, N_max, Hp, Wp]

    # row_frac: [B, Hp]
    row_centers = stride_h * torch.arange(Hp, device=device, dtype=torch.float32)
    y_range = torch.clamp(y_bot - y_top, min=1.0)  # [B]
    row_frac = ((row_centers[None, :] - y_top[:, None]) / y_range[:, None]).clamp(0.0, 1.0)

    # Learnable padded logits. -inf at padded positions so softmax is exactly
    # distributed over the real N letters.
    width_logits = torch.zeros(B, N_max, device=device, dtype=torch.float32,
                               requires_grad=True)
    neg_inf = torch.full_like(width_logits, float("-inf"))
    opt = torch.optim.Adam([width_logits], lr=args.lr)

    for step in range(args.steps + 1):
        logits_masked = torch.where(mask, width_logits, neg_inf)
        # softmax over real N only → padded slots get 0 weight.
        weights = torch.softmax(logits_masked, dim=1)  # [B, N_max]
        widths = args.min_width + weights * slack[:, None]  # [B, N_max]
        widths = widths * mask_f  # zero out padded widths
        cum = torch.cumsum(widths, dim=1)  # [B, N_max]
        pos = torch.cat([wx1[:, None], wx1[:, None] + cum], dim=1)  # [B, N_max+1]

        # Box edges: [B, N_max]
        x1 = pos[:, :-1]
        x2 = pos[:, 1:]

        # Per-row bounds (slant=0): [B, N_max, Hp] — with slant=0, top/bot edges
        # are identical, so left/right don't depend on row.
        left = x1[:, :, None].expand(B, N_max, Hp)
        right = x2[:, :, None].expand(B, N_max, Hp)

        # Pixel membership: [B, N_max, Hp, W]
        k = args.mask_sharpness
        pixel_w = (torch.sigmoid(k * (pixels[None, None, None, :] - left[:, :, :, None]))
                   * torch.sigmoid(k * (right[:, :, :, None] - pixels[None, None, None, :])))
        # Aggregate to cell columns: [B, N_max, Hp, Wp]
        col_w = pixel_w @ rf_matrix.t()

        # Soft max over 2D cells.
        gated = tgt_logits_2d + torch.log(col_w.clamp(min=1e-8))  # [B, N_max, Hp, Wp]
        score = torch.logsumexp(gated.reshape(B, N_max, -1), dim=2)  # [B, N_max]
        target_probs = torch.sigmoid(score)

        if step < args.steps:
            # Only real positions contribute.
            loss = -(target_probs * mask_f).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Final positions & probs.
    with torch.no_grad():
        logits_masked = torch.where(mask, width_logits, neg_inf)
        weights = torch.softmax(logits_masked, dim=1)
        widths = (args.min_width + weights * slack[:, None]) * mask_f
        cum = torch.cumsum(widths, dim=1)
        pos = torch.cat([wx1[:, None], wx1[:, None] + cum], dim=1)
        x1 = pos[:, :-1]
        x2 = pos[:, 1:]
        left = x1[:, :, None].expand(B, N_max, Hp)
        right = x2[:, :, None].expand(B, N_max, Hp)
        k = args.mask_sharpness
        pixel_w = (torch.sigmoid(k * (pixels[None, None, None, :] - left[:, :, :, None]))
                   * torch.sigmoid(k * (right[:, :, :, None] - pixels[None, None, None, :])))
        col_w = pixel_w @ rf_matrix.t()
        gated = tgt_logits_2d + torch.log(col_w.clamp(min=1e-8))
        score = torch.logsumexp(gated.reshape(B, N_max, -1), dim=2)
        final_probs = torch.sigmoid(score)  # [B, N_max]

    pos_np = pos.detach().cpu().numpy()  # [B, N_max+1]
    probs_np = final_probs.detach().cpu().numpy()  # [B, N_max]

    # Per-word refinement on the original blurred image.
    results = []
    for b, p in enumerate(prepared):
        N = p["N"]
        word_pos = pos_np[b, :N + 1]  # [N+1]
        word_probs = probs_np[b, :N]
        refine_mask = p["word_arr_blur"] < args.refine_y_threshold
        letters = []
        for i in range(N):
            lx1 = float(word_pos[i])
            lx2 = float(word_pos[i + 1])
            c1 = int(np.clip(np.floor(lx1), 0, TARGET_W))
            c2 = int(np.clip(np.ceil(lx2), 0, TARGET_W))
            if c2 <= c1:
                y1 = float(p["y_top"])
                y2 = float(p["y_bot"])
            else:
                col_slice = refine_mask[:, c1:c2]
                rows = np.where(col_slice.any(axis=1))[0]
                if rows.size == 0:
                    y1 = float(p["y_top"])
                    y2 = float(p["y_bot"])
                else:
                    y1 = float(rows.min())
                    y2 = float(rows.max()) + 1.0
            y1 = float(np.clip(y1 - args.refine_y_margin, 0, TARGET_H))
            y2 = float(np.clip(y2 + args.refine_y_margin, 0, TARGET_H))
            letters.append({
                "char": p["rec"]["text"][i],
                "x1": round(lx1, 2),
                "y1": round(y1, 2),
                "x2": round(lx2, 2),
                "y2": round(y2, 2),
                "prob": round(float(word_probs[i]), 4),
            })
        r = p["rec"]
        results.append({
            "word_id": r["word_id"],
            "form": r["form"],
            "line": r["line"],
            "text": r["text"],
            "target_h": TARGET_H,
            "target_w": TARGET_W,
            "word_x1": round(p["word_x1"], 2),
            "word_x2": round(p["word_x2"], 2),
            "word_y_top": round(p["y_top"], 2),
            "word_y_bot": round(p["y_bot"], 2),
            "letters": letters,
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--words-txt", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--mask-sharpness", type=float, default=0.5)
    ap.add_argument("--image-blur-sigma", type=float, default=1.0)
    ap.add_argument("--min-width", type=float, default=6.0)
    ap.add_argument("--refine-y-margin", type=float, default=1.0)
    ap.add_argument("--refine-y-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records (debug)")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch = ckpt_args.get("base_ch", 32)
    final_stride = ckpt_args.get("final_stride", 2)
    model = LocalCharRecognizer(base_ch=base_ch, final_stride=final_stride).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    records = parse_words_txt(Path(args.words_txt), words_dir=Path(args.words_dir))
    # Drop words too short for dividers to make sense.
    records = [r for r in records if len(r["text"]) >= 2]
    # Sort by N (letter count) to minimize padding waste within batches.
    records.sort(key=lambda r: len(r["text"]))
    if args.limit is not None:
        records = records[:args.limit]
    print(f"Total candidate records: {len(records)}")

    # Precompute rf_matrix and pixels once.
    with torch.no_grad():
        dummy = torch.zeros(1, 1, TARGET_H, TARGET_W, device=device)
        fmap = model.feature_map(dummy)
    _, _, Hp, Wp = fmap.shape
    stride_w = TARGET_W // Wp
    rf_matrix = build_rf_matrix(Wp, stride_w, device)
    pixels = torch.arange(TARGET_W, device=device, dtype=torch.float32)
    print(f"Feature map: Hp={Hp}, Wp={Wp}, stride_w={stride_w}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    with open(out_path, "w") as fout:
        buf: List[Dict] = []
        def flush():
            nonlocal n_done
            if not buf:
                return
            results = run_batch(buf, model, device, args, rf_matrix, pixels)
            for r in results:
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            n_done += len(results)
            buf.clear()

        for i, rec in enumerate(records):
            p = prepare_word(rec, Path(args.words_dir), args.min_width,
                             args.image_blur_sigma)
            if p is None:
                n_skipped += 1
                continue
            buf.append(p)
            if len(buf) >= args.batch_size:
                flush()
            if (i + 1) % args.log_every == 0 or i == len(records) - 1:
                dt = time.time() - t0
                rate = n_done / dt if dt > 0 else 0
                print(f"  [{i + 1:>6}/{len(records)}] done={n_done} skipped={n_skipped} "
                      f"elapsed={dt:.1f}s rate={rate:.1f} words/s")
        flush()

    dt = time.time() - t0
    print(f"\nSaved {n_done} records to {out_path} ({n_skipped} skipped) in {dt:.1f}s")


if __name__ == "__main__":
    main()
