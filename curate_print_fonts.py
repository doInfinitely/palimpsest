#!/usr/bin/env python3
"""Scan a fonts directory and keep only fonts that can cleanly render
every char in VOCAB. Writes a JSON list of font paths.

Reject criteria:
  - Missing glyph for any VOCAB char (renders to all-blank).
  - All-full square (some symbol fonts fill the cell entirely).
  - Glyph too small (< 20% of the cell area in ink) — usually means
    the font is at a wrong size or rendering broke.

Usage:
    python3 curate_print_fonts.py \\
        --fonts-dir ../tiny-tessarachnid/fonts \\
        --out runs/print_font_list.json \\
        --size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train_char_recognizer import VOCAB


def render_glyph(font: ImageFont.FreeTypeFont, ch: str, size: int = 32
                 ) -> np.ndarray | None:
    """Render `ch` centered on a `size`×`size` canvas, white bg, black ink.
    Returns float32 array in [0, 1] with 1=white, or None on failure."""
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    try:
        bbox = d.textbbox((0, 0), ch, font=font)
    except Exception:
        return None
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    # Center.
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    try:
        d.text((x, y), ch, font=font, fill=0)
    except Exception:
        return None
    return np.asarray(img, dtype=np.float32) / 255.0


def ink_fraction(arr: np.ndarray) -> float:
    return float((arr < 0.5).mean())


def font_works(font_path: Path, size: int) -> tuple[bool, str]:
    try:
        # Try multiple point sizes to find one that fits.
        for pt in (28, 24, 22, 20, 18, 16):
            font = ImageFont.truetype(str(font_path), pt)
            arrs = []
            for ch in VOCAB:
                arr = render_glyph(font, ch, size)
                if arr is None:
                    return False, f"render-fail on {ch!r}"
                arrs.append(arr)
            # Check ink fractions per glyph (sanity).
            inks = np.array([ink_fraction(a) for a in arrs])
            mean_ink = float(inks.mean())
            zero_ink = int((inks < 0.005).sum())
            full_ink = int((inks > 0.85).sum())
            if mean_ink < 0.04:
                continue  # font too small at this pt size
            if zero_ink > 2:
                return False, f"zero-ink glyphs at pt={pt}: {zero_ink}/{len(VOCAB)}"
            if full_ink > 1:
                return False, f"full-ink glyphs at pt={pt}: {full_ink}/{len(VOCAB)}"
            # Detect notdef-substitution: many distinct chars rendering to the
            # SAME glyph (e.g., the empty-rectangle placeholder). Sample a few
            # chars that are visually distinct in real fonts and check.
            sample = ["a", "m", "z", "A", "M", "Z", "3", "7", ".", "?"]
            ref = {}
            for ch in sample:
                if ch in VOCAB:
                    ref[ch] = arrs[VOCAB.index(ch)]
            # Count unique glyphs (allowing tiny noise).
            keys = [tuple(np.round(a * 10).astype(int).flatten()) for a in ref.values()]
            unique = len(set(keys))
            if unique < max(3, len(ref) // 2):
                return False, f"only {unique}/{len(ref)} unique glyphs at pt={pt} (notdef?)"
            # OK.
            return True, f"pt={pt} mean_ink={mean_ink:.2%} unique={unique}/{len(ref)}"
        return False, "no usable point size"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()

    fonts_dir = Path(args.fonts_dir)
    all_fonts = sorted(p for p in fonts_dir.iterdir() if p.suffix.lower() in (".ttf", ".otf"))
    print(f"Found {len(all_fonts)} fonts in {fonts_dir}")

    kept: list[dict] = []
    rejected: list[dict] = []
    for i, fp in enumerate(all_fonts):
        ok, info = font_works(fp, args.size)
        if ok:
            kept.append({"path": str(fp), "info": info})
        else:
            rejected.append({"path": str(fp), "reason": info})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_fonts)} scanned, kept={len(kept)}")
    print(f"Kept {len(kept)} / {len(all_fonts)} fonts")
    print(f"First 10 rejected reasons:")
    for r in rejected[:10]:
        print(f"  {Path(r['path']).name}: {r['reason']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "kept": kept,
            "rejected": rejected,
            "vocab": VOCAB,
            "size": args.size,
        }, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
