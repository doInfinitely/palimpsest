#!/usr/bin/env python3
"""Skeletonize an IAM word image and render the skeleton overlaid on the
original.

Usage:
    python3 skeletonize_word.py \\
        --word-id c03-007-02-07 \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --out eval_output/skel_shabby.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize

from train_char_recognizer import parse_words_txt


def load_word_array(rec, words_dir: Path, blur_sigma: float = 0.6) -> np.ndarray:
    """Returns a float32 array in [0, 1] at the IMAGE'S NATIVE resolution
    (no letterboxing). 1=white, 0=ink."""
    p = words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    img = Image.open(p).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if blur_sigma > 0:
        arr = gaussian_filter(arr, sigma=blur_sigma)
    return arr


def compute_skeleton(arr: np.ndarray, ink_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Returns (binary_ink_mask, skeleton_mask)."""
    ink = arr < ink_threshold
    skel = skeletonize(ink)
    return ink, skel


def render_overlay(arr: np.ndarray, skel: np.ndarray, scale: int = 2) -> Image.Image:
    """Show original (top), skeleton-on-white (middle), overlay (bottom).
    Scaled up by `scale` for visibility."""
    H, W = arr.shape
    base = (arr * 255).astype(np.uint8)
    base_rgb = np.stack([base] * 3, axis=-1)

    # Skeleton on white background.
    skel_only = np.full((H, W, 3), 255, dtype=np.uint8)
    skel_only[skel] = (220, 20, 60)

    # Overlay: original + red skeleton.
    overlay = base_rgb.copy()
    overlay[skel] = (220, 20, 60)

    sep = np.full((4, W, 3), 220, dtype=np.uint8)
    stack = np.concatenate([base_rgb, sep, skel_only, sep, overlay], axis=0)

    img = Image.fromarray(stack)
    if scale != 1:
        img = img.resize((W * scale, stack.shape[0] * scale), Image.NEAREST)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-id", required=True)
    ap.add_argument("--words-txt", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ink-threshold", type=float, default=0.5)
    ap.add_argument("--blur-sigma", type=float, default=0.6)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    records = parse_words_txt(Path(args.words_txt), words_dir=Path(args.words_dir))
    rec = next((r for r in records if r["word_id"] == args.word_id), None)
    if rec is None:
        raise SystemExit(f"No record for word_id={args.word_id}")
    print(f"word_id={args.word_id}  text={rec['text']!r}")

    arr = load_word_array(rec, Path(args.words_dir), blur_sigma=args.blur_sigma)
    ink, skel = compute_skeleton(arr, ink_threshold=args.ink_threshold)
    print(f"image: {arr.shape}  ink pixels: {int(ink.sum())}  skeleton pixels: {int(skel.sum())}")

    img = render_overlay(arr, skel, scale=args.scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
