#!/usr/bin/env python3
"""Demo: infill characters into real document context using the fractal model.

Takes actual before-patches from the validation set and infills with both
the correct character and alternative characters to see how the model responds.

Usage:
    python demo_infill_context.py \
        --checkpoint runs/infill_v4/best.pt \
        --val-jsonl data/synth_v1/infill_fractal/fractal_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir eval_output/infill_context_demo
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
from train_character_infill import (
    read_jsonl, load_gray_image, load_mask_image,
    text_to_byte_tensor, pad_1d_long,
)


def infill_one(model, before, bbox_mask, bbox, char_text, style_idx, device):
    """Run model on a single sample, return predicted image."""
    before_t = before.unsqueeze(0).to(device)
    mask_t = bbox_mask.unsqueeze(0).to(device)
    bbox_t = bbox.unsqueeze(0).to(device)

    char_tokens_raw = text_to_byte_tensor(char_text)
    char_tokens = pad_1d_long([char_tokens_raw]).to(device)
    char_lengths = torch.tensor([len(char_tokens_raw)], dtype=torch.long, device=device)
    style = torch.tensor([style_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        pred_delta = model.forward_infill(
            before_t, mask_t, char_tokens, char_lengths, style, bbox_t,
        )
        pred = (before_t + pred_delta).clamp(0, 1)

    return pred[0]


def tensor_to_pil(t, size=192):
    t = t.detach().cpu().clamp(0, 1)
    if t.shape[1] != size or t.shape[2] != size:
        t = F.interpolate(t.unsqueeze(0), size=(size, size),
                          mode="bilinear", align_corners=False)[0]
    arr = (t[0] * 255).byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/infill_v4/best.pt")
    p.add_argument("--val-jsonl", default="data/synth_v1/infill_fractal/fractal_infill_val.jsonl")
    p.add_argument("--image-root", default="data/synth_v1")
    p.add_argument("--out-dir", default="eval_output/infill_context_demo")
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--alt-chars", default="AXm7@",
                   help="Alternative characters to infill in each slot")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    ckpt_dir = Path(args.checkpoint).parent
    style_to_index = {"__unk__": 0}
    style_path = ckpt_dir / "style_to_index.json"
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

    # Load samples (character-level only for cleaner demo)
    records = read_jsonl(args.val_jsonl)
    char_recs = [r for r in records if r.get("level") == "character"][:args.num_samples]
    print(f"Using {len(char_recs)} character-level samples")

    image_root = Path(args.image_root)
    alt_chars = list(args.alt_chars)
    cell_size = 192
    pad = 3
    label_h = 16

    panels = []
    for rec in char_recs:
        before = load_gray_image(str(image_root / rec["before_patch_ref"]), RETINA_SIZE)
        after_gt = load_gray_image(str(image_root / rec["after_patch_ref"]), RETINA_SIZE)
        bbox_mask = load_mask_image(str(image_root / rec["bbox_mask_ref"]), RETINA_SIZE)
        bbox = torch.tensor(rec["target_bbox_parent_norm_cxcywh"], dtype=torch.float32)
        style_id = rec.get("style_id", "__unk__")
        style_idx = style_to_index.get(style_id, 0)
        gt_char = rec.get("char_text", "?")

        # Column images: before | mask | GT char infill | alt chars...
        cols = []
        cols.append(("before", tensor_to_pil(before, cell_size)))
        cols.append(("mask", tensor_to_pil(bbox_mask, cell_size)))

        # Ground truth
        cols.append(("GT", tensor_to_pil(after_gt, cell_size)))

        # Model prediction with correct char
        pred = infill_one(model, before, bbox_mask, bbox, gt_char, style_idx, device)
        cols.append((f"pred '{gt_char}'", tensor_to_pil(pred, cell_size)))

        # Model prediction with alternative chars
        for ac in alt_chars:
            pred_alt = infill_one(model, before, bbox_mask, bbox, ac, style_idx, device)
            cols.append((f"'{ac}'", tensor_to_pil(pred_alt, cell_size)))

        # Assemble row
        row_w = len(cols) * (cell_size + pad) + pad
        row_h = cell_size + label_h + pad
        row_img = Image.new("RGB", (row_w, row_h), (200, 200, 200))
        draw = ImageDraw.Draw(row_img)
        for ci, (label, img) in enumerate(cols):
            x = ci * (cell_size + pad) + pad
            draw.text((x + 2, 1), label, fill=(0, 0, 0))
            row_img.paste(img, (x, label_h))
        panels.append(row_img)

        print(f"  '{gt_char}' done")

    # Stack rows vertically
    total_w = max(p.width for p in panels)
    total_h = sum(p.height + pad for p in panels) + pad
    sheet = Image.new("RGB", (total_w, total_h), (180, 180, 180))
    y = pad
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height + pad

    sheet_path = out_dir / "context_infill_sheet.png"
    sheet.save(sheet_path)
    print(f"\nSaved: {sheet_path}")


if __name__ == "__main__":
    main()
