#!/usr/bin/env python3
"""Pre-render printed glyphs for every (font, char) into a single tensor
cache. Used by the print-glyph conditioning path in FractalInfiller.

Output: a torch .pt file with
    "glyphs": uint8 tensor [num_fonts, num_classes, size, size]  (255=white, 0=ink)
    "font_paths": list of font paths
    "vocab": list of chars

Usage:
    python3 render_print_glyphs.py \\
        --font-list runs/print_font_list.json \\
        --out runs/print_glyphs.pt \\
        --size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from train_char_recognizer import VOCAB


def render_glyph(font: ImageFont.FreeTypeFont, ch: str, size: int) -> np.ndarray:
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    d.text((x, y), ch, font=font, fill=0)
    return np.asarray(img, dtype=np.uint8)


def best_pt_for_font(font_path: Path, size: int) -> int | None:
    """Pick the largest pt that doesn't overflow the cell for any VOCAB char."""
    for pt in (28, 24, 22, 20, 18, 16):
        try:
            font = ImageFont.truetype(str(font_path), pt)
        except Exception:
            return None
        ok = True
        for ch in VOCAB:
            try:
                bbox = ImageDraw.Draw(Image.new("L", (size, size))).textbbox(
                    (0, 0), ch, font=font)
            except Exception:
                ok = False; break
            w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
            if w > size or h > size:
                ok = False; break
        if ok:
            return pt
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()

    cfg = json.load(open(args.font_list))
    font_paths = [Path(k["path"]) for k in cfg["kept"]]
    print(f"Rendering {len(font_paths)} fonts × {len(VOCAB)} chars @ {args.size}px")

    glyphs = np.full(
        (len(font_paths), len(VOCAB), args.size, args.size), 255, dtype=np.uint8)
    kept_paths: list[str] = []
    kept_idx = 0
    for fi, fp in enumerate(font_paths):
        pt = best_pt_for_font(fp, args.size)
        if pt is None:
            continue
        try:
            font = ImageFont.truetype(str(fp), pt)
        except Exception:
            continue
        for ci, ch in enumerate(VOCAB):
            try:
                glyphs[kept_idx, ci] = render_glyph(font, ch, args.size)
            except Exception:
                glyphs[kept_idx, ci] = 255  # blank on render fail
        kept_paths.append(str(fp))
        kept_idx += 1
        if kept_idx % 20 == 0:
            print(f"  {kept_idx}/{len(font_paths)} rendered")

    # Trim to actually-rendered fonts.
    glyphs = glyphs[:kept_idx]
    print(f"Final: {kept_idx} fonts × {len(VOCAB)} chars")
    out = {
        "glyphs": torch.from_numpy(glyphs),
        "font_paths": kept_paths,
        "vocab": VOCAB,
        "size": args.size,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Saved → {args.out}  ({glyphs.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
