#!/usr/bin/env python3
"""Demo: use the fractal infill model to write words character-by-character.

Starts with a blank 256x256 canvas, places characters left-to-right,
and calls the model to infill each one sequentially.

Usage:
    python demo_write_word.py \
        --checkpoint runs/infill_v4/best.pt \
        --words "hello" "world" "cat" \
        --out-dir eval_output/write_demo
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from train_fractal_infill import FractalInfiller, RETINA_SIZE
from train_character_infill import text_to_byte_tensor, pad_1d_long


def write_word(model, word: str, style_index: int, device: torch.device) -> Image.Image:
    """Sequentially infill each character of a word onto a blank canvas."""
    S = RETINA_SIZE  # 256

    # Start with a white canvas
    canvas = torch.ones(1, 1, S, S, device=device)

    # Simple monospaced layout: divide canvas width evenly
    n = len(word)
    if n == 0:
        arr = (canvas[0, 0].cpu().clamp(0, 1) * 255).byte().numpy()
        return Image.fromarray(arr, mode="L")

    char_w = min(S // n, S // 2)  # width per character
    total_w = char_w * n
    x_start = (S - total_w) // 2  # centre the word

    # Vertical region: occupy middle 60% of canvas
    y_top = int(S * 0.2)
    y_bot = int(S * 0.8)

    for i, ch in enumerate(word):
        cx1 = x_start + i * char_w
        cx2 = cx1 + char_w

        # Build bbox mask
        bbox_mask = torch.zeros(1, 1, S, S, device=device)
        bbox_mask[:, :, y_top:y_bot, cx1:cx2] = 1.0

        # Normalised bbox (cx, cy, w, h) in [0, 1]
        cx_norm = (cx1 + cx2) / 2.0 / S
        cy_norm = (y_top + y_bot) / 2.0 / S
        w_norm = char_w / S
        h_norm = (y_bot - y_top) / S
        bbox = torch.tensor([[cx_norm, cy_norm, w_norm, h_norm]], dtype=torch.float32, device=device)

        # Character tokens
        char_tokens_raw = text_to_byte_tensor(ch)
        char_tokens = pad_1d_long([char_tokens_raw]).to(device)
        char_lengths = torch.tensor([len(char_tokens_raw)], dtype=torch.long, device=device)

        style = torch.tensor([style_index], dtype=torch.long, device=device)

        with torch.no_grad():
            pred_delta = model.forward_infill(
                canvas, bbox_mask,
                char_tokens, char_lengths,
                style, bbox,
            )
            canvas = (canvas + pred_delta).clamp(0, 1)

    arr = (canvas[0, 0].cpu() * 255).byte().numpy()
    return Image.fromarray(arr, mode="L")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/infill_v4/best.pt")
    p.add_argument("--words", nargs="+", default=["hello", "world", "the", "cat", "ABC"])
    p.add_argument("--styles", nargs="*", type=int, default=[0],
                   help="Style indices to try (0 = unknown)")
    p.add_argument("--out-dir", default="eval_output/write_demo")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]

    ckpt_dir = Path(args.checkpoint).parent
    style_path = ckpt_dir / "style_to_index.json"
    style_to_index = {"__unk__": 0}
    if style_path.exists():
        with style_path.open() as f:
            style_to_index = json.load(f)

    model = FractalInfiller(
        in_ch=2, out_ch=1,
        base_ch=int(model_args.get("base_ch", 96)),
        cond_dim=int(model_args.get("cond_dim", 128)),
        vocab_size=256,
        num_styles=len(style_to_index),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded model ({sum(pp.numel() for pp in model.parameters()):,} params)")

    # Generate words
    images = []
    labels = []
    for style_idx in args.styles:
        for word in args.words:
            print(f"  Writing '{word}' (style={style_idx})")
            img = write_word(model, word, style_idx, device)
            images.append(img)
            labels.append(f"'{word}' s={style_idx}")

    # Build contact sheet
    pad = 4
    label_h = 16
    cell_w = RETINA_SIZE
    cell_h = RETINA_SIZE + label_h
    cols = min(len(images), 4)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("L", (cols * (cell_w + pad) + pad, rows * (cell_h + pad) + pad), 200)
    draw = ImageDraw.Draw(sheet)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = c * (cell_w + pad) + pad
        y = r * (cell_h + pad) + pad
        draw.text((x + 2, y + 1), lbl, fill=0)
        sheet.paste(img, (x, y + label_h))

    sheet_path = out_dir / "word_sheet.png"
    sheet.save(sheet_path)
    print(f"\nSaved: {sheet_path}")


if __name__ == "__main__":
    main()
