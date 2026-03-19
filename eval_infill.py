#!/usr/bin/env python3
"""Evaluate a trained character infill checkpoint.

Loads best.pt, runs inference on validation samples, produces a contact
sheet with columns: before | mask | predicted | ground-truth | delta-error,
and reports inside/outside L1 metrics.

Usage:
    python eval_infill.py \
        --checkpoint runs/infill_v1/best.pt \
        --val-jsonl data/synth_v1/infill/character_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir eval_output/infill_v1 \
        --num-samples 40
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

from train_character_infill import (
    ConditionalUNet,
    CharacterInfillDataset,
    collate_infill,
    read_jsonl,
)


# ============================================================
# Rendering helpers
# ============================================================

def tensor_to_gray_pil(t: Tensor, size: int = 128) -> Image.Image:
    """[1,H,W] float tensor → grayscale PIL, resized."""
    t = t.detach().cpu().clamp(0, 1)
    if t.shape[1] != size or t.shape[2] != size:
        t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    arr = (t[0] * 255).byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def tensor_to_error_pil(pred: Tensor, target: Tensor, mask: Tensor, size: int = 128) -> Image.Image:
    """Render |pred-target| as a red-channel heatmap, brighter inside mask."""
    err = torch.abs(pred - target).detach().cpu()
    err = err.clamp(0, 1)
    if err.shape[1] != size or err.shape[2] != size:
        err = F.interpolate(err.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask.unsqueeze(0).cpu().float(), size=(size, size), mode="nearest")[0]
    err_np = (err[0] * 255).byte().numpy()
    mask_np = (mask[0].cpu() > 0.5).numpy()
    r = err_np.copy()
    g = np.where(mask_np, 0, err_np // 3).astype(np.uint8)
    b = np.zeros_like(err_np)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def add_label(img: Image.Image, text: str, font_size: int = 11) -> Image.Image:
    pad = font_size + 4
    out = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((2, 1), text, fill=(0, 0, 0))
    out.paste(img, (0, pad))
    return out


def make_contact_sheet(
    panels: List[Image.Image],
    cols: int = 5,
    pad: int = 3,
) -> Image.Image:
    if not panels:
        return Image.new("RGB", (100, 100), (255, 255, 255))
    pw = max(p.width for p in panels)
    ph = max(p.height for p in panels)
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * (pw + pad) + pad, rows * (ph + pad) + pad), (230, 230, 230))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        sheet.paste(p, (c * (pw + pad) + pad, r * (ph + pad) + pad))
    return sheet


# ============================================================
# Per-sample panel: before | mask | predicted | GT after | error
# ============================================================

def build_panel(
    before: Tensor,     # [1,H,W]
    bbox_mask: Tensor,  # [1,H,W]
    pred_after: Tensor, # [1,H,W]
    gt_after: Tensor,   # [1,H,W]
    char_text: str,
    record_id: str,
    inside_l1: float,
    cell_size: int = 128,
) -> Image.Image:
    cols = [
        add_label(tensor_to_gray_pil(before, cell_size), "before"),
        add_label(tensor_to_gray_pil(bbox_mask, cell_size), "mask"),
        add_label(tensor_to_gray_pil(pred_after, cell_size), "predicted"),
        add_label(tensor_to_gray_pil(gt_after, cell_size), "ground truth"),
        add_label(tensor_to_error_pil(pred_after, gt_after, bbox_mask, cell_size), f"err L1={inside_l1:.4f}"),
    ]
    # row header
    header_h = 14
    total_w = sum(c.width for c in cols) + 4 * 2
    total_h = cols[0].height + header_h
    panel = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    label = f'"{char_text}"  {record_id[:40]}'
    draw.text((2, 0), label, fill=(60, 60, 60))
    x = 0
    for c in cols:
        panel.paste(c, (x, header_h))
        x += c.width + 2
    return panel


# ============================================================
# Main
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate character infill model")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--val-jsonl", required=True, help="Validation JSONL")
    p.add_argument("--image-root", default=".", help="Root for image paths")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--cell-size", type=int, default=128, help="Render cell size")
    p.add_argument("--cols", type=int, default=4, help="Contact sheet columns")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    epoch = ckpt.get("epoch", "?")
    val_metrics = ckpt.get("val_metrics", {})
    print(f"  Epoch {epoch}, val_loss={val_metrics.get('total', '?')}")

    # Load style vocab
    ckpt_dir = Path(args.checkpoint).parent
    style_path = ckpt_dir / "style_to_index.json"
    if style_path.exists():
        with style_path.open() as f:
            style_to_index = json.load(f)
    else:
        style_to_index = {"__unk__": 0}
    num_styles = len(style_to_index)

    # Build model
    model = ConditionalUNet(
        in_ch=2, out_ch=1,
        base_ch=int(model_args.get("base_ch", 64)),
        cond_dim=int(model_args.get("cond_dim", 128)),
        vocab_size=256,
        num_styles=num_styles,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters, {num_styles} styles")

    # Dataset
    ds = CharacterInfillDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
    )
    end = min(len(ds), args.offset + args.num_samples)
    print(f"Evaluating samples {args.offset}..{end} of {len(ds)} total")

    # Run inference
    panels: List[Image.Image] = []
    results: List[Dict[str, Any]] = []
    all_inside_l1 = []
    all_outside_l1 = []

    from torch.utils.data import Subset, DataLoader

    subset = Subset(ds, list(range(args.offset, end)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_infill)

    sample_idx = 0
    for batch in loader:
        # Move to device
        before = batch["before"].to(device)
        bbox_mask = batch["bbox_mask"].to(device)
        char_tokens = batch["char_tokens"].to(device)
        char_lengths = batch["char_lengths"].to(device)
        style_index = batch["style_index"].to(device)
        bbox = batch["bbox"].to(device)
        gt_delta = batch["delta"].to(device)
        gt_after = batch["after"].to(device)

        with torch.no_grad():
            pred_delta = model(before, bbox_mask, char_tokens, char_lengths, style_index, bbox)

        pred_after = before + pred_delta

        # Per-sample metrics and panels
        bs = before.shape[0]
        for j in range(bs):
            inside = bbox_mask[j]
            outside = 1.0 - inside
            err = torch.abs(pred_delta[j] - gt_delta[j])
            in_l1 = float((err * inside).sum() / inside.sum().clamp_min(1))
            out_l1 = float((err * outside).sum() / outside.sum().clamp_min(1))
            all_inside_l1.append(in_l1)
            all_outside_l1.append(out_l1)

            # Lookup char text from dataset record
            rec_idx = args.offset + sample_idx
            rec = ds.records[rec_idx]
            char_text = rec.get("char_text", "?")
            record_id = batch["record_id"][j]

            panel = build_panel(
                before=before[j].cpu(),
                bbox_mask=bbox_mask[j].cpu(),
                pred_after=pred_after[j].cpu(),
                gt_after=gt_after[j].cpu(),
                char_text=char_text,
                record_id=record_id,
                inside_l1=in_l1,
                cell_size=args.cell_size,
            )
            panels.append(panel)

            results.append({
                "index": sample_idx,
                "record_id": record_id,
                "char_text": char_text,
                "inside_l1": in_l1,
                "outside_l1": out_l1,
            })

            sample_idx += 1

        print(f"  Processed {sample_idx}/{end - args.offset} samples")

    # Contact sheet
    sheet = make_contact_sheet(panels, cols=args.cols)
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\nContact sheet: {sheet_path}")

    # Summary
    avg_in = sum(all_inside_l1) / max(1, len(all_inside_l1))
    avg_out = sum(all_outside_l1) / max(1, len(all_outside_l1))
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "num_samples": len(results),
        "avg_inside_l1": avg_in,
        "avg_outside_l1": avg_out,
        "samples": results,
    }
    summary_path = out_dir / "eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Avg inside L1:  {avg_in:.5f}")
    print(f"Avg outside L1: {avg_out:.5f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
