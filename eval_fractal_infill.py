#!/usr/bin/env python3
"""Evaluate single-pass character infill model.

Produces a contact sheet: before | mask | predicted | GT | error

Usage:
    python eval_fractal_infill.py \
        --checkpoint runs/infill_v4/best.pt \
        --val-jsonl data/synth_v1/infill_fractal/fractal_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir eval_output/infill_v4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from train_fractal_infill import FractalInfiller, RETINA_SIZE
from train_character_infill import (
    CharacterInfillDataset, collate_infill, read_jsonl,
)


def tensor_to_gray_pil(t: Tensor, size: int = 128) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    if t.shape[1] != size or t.shape[2] != size:
        t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    arr = (t[0] * 255).byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def tensor_to_error_pil(pred: Tensor, target: Tensor, mask: Tensor, size: int = 128) -> Image.Image:
    err = torch.abs(pred - target).detach().cpu().clamp(0, 1)
    if err.shape[1] != size or err.shape[2] != size:
        err = F.interpolate(err.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask.unsqueeze(0).cpu().float(), size=(size, size), mode="nearest")[0]
    err_np = (err[0] * 255).byte().numpy()
    mask_np = (mask[0].cpu() > 0.5).numpy()
    r = err_np.copy()
    g = np.where(mask_np, 0, err_np // 3).astype(np.uint8)
    b = np.zeros_like(err_np)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def add_label(img: Image.Image, text: str) -> Image.Image:
    pad = 15
    out = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((2, 1), text, fill=(0, 0, 0))
    out.paste(img, (0, pad))
    return out


def make_contact_sheet(panels, cols=3, pad=3):
    if not panels:
        return Image.new("RGB", (100, 100))
    pw = max(p.width for p in panels)
    ph = max(p.height for p in panels)
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * (pw + pad) + pad, rows * (ph + pad) + pad), (230, 230, 230))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        sheet.paste(p, (c * (pw + pad) + pad, r * (ph + pad) + pad))
    return sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--cell-size", type=int, default=128)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    epoch = ckpt.get("epoch", "?")
    print(f"  Epoch {epoch}")

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
    state = ckpt.get("model_state_dict", ckpt.get("gen_state_dict"))
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"  {sum(pp.numel() for pp in model.parameters()):,} params")

    ds = CharacterInfillDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
    )
    end = min(len(ds), args.offset + args.num_samples)
    print(f"Evaluating samples {args.offset}..{end} of {len(ds)}")

    subset = Subset(ds, list(range(args.offset, end)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_infill)

    panels = []
    all_l1 = []
    results = []

    sample_idx = 0
    for batch in loader:
        batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            pred_delta = model.forward_infill(
                batch_d["before"], batch_d["bbox_mask"],
                batch_d["char_tokens"], batch_d["char_lengths"],
                batch_d["style_index"], batch_d["bbox"],
            )
            pred_after = (batch_d["before"] + pred_delta).clamp(0, 1)

        bs = batch_d["before"].shape[0]
        for j in range(bs):
            inside = batch_d["bbox_mask"][j]
            gt_after = batch_d["after"][j]

            err = torch.abs(pred_after[j] - gt_after) * inside
            l1 = float(err.sum() / inside.sum().clamp_min(1))
            all_l1.append(l1)

            rec_idx = args.offset + sample_idx
            rec = ds.records[rec_idx]
            char_text = rec.get("char_text", "?")
            record_id = batch["record_id"][j]

            cs = args.cell_size
            cols_imgs = [
                add_label(tensor_to_gray_pil(batch_d["before"][j].cpu(), cs), "before"),
                add_label(tensor_to_gray_pil(batch_d["bbox_mask"][j].cpu(), cs), "mask"),
                add_label(tensor_to_gray_pil(pred_after[j].cpu(), cs), f"pred L1={l1:.4f}"),
                add_label(tensor_to_gray_pil(gt_after.cpu(), cs), "ground truth"),
                add_label(tensor_to_error_pil(pred_after[j].cpu(), gt_after.cpu(),
                                              inside.cpu(), cs), "error"),
            ]

            header_h = 14
            total_w = sum(c.width for c in cols_imgs) + 5 * 2
            total_h = cols_imgs[0].height + header_h
            panel = Image.new("RGB", (total_w, total_h), (255, 255, 255))
            draw = ImageDraw.Draw(panel)
            draw.text((2, 0), f'"{char_text}"  {record_id[:40]}', fill=(60, 60, 60))
            x = 0
            for c in cols_imgs:
                panel.paste(c, (x, header_h))
                x += c.width + 2
            panels.append(panel)

            results.append({
                "index": sample_idx, "record_id": record_id,
                "char_text": char_text, "inside_l1": l1,
            })
            sample_idx += 1

        print(f"  Processed {sample_idx}/{end - args.offset}")

    sheet = make_contact_sheet(panels, cols=args.cols)
    sheet.save(out_dir / "contact_sheet.png")

    avg_l1 = sum(all_l1) / max(1, len(all_l1))

    summary = {
        "checkpoint": args.checkpoint, "epoch": epoch,
        "num_samples": len(results),
        "avg_inside_l1": avg_l1, "samples": results,
    }
    with (out_dir / "eval_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAvg inside L1:  {avg_l1:.5f}")
    print(f"Contact sheet: {out_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
