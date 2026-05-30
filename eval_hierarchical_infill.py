#!/usr/bin/env python3
"""Evaluate hierarchical character infill model.

Produces a contact sheet with columns:
  before | mask | coarse (upsampled) | refined | ground-truth | error

Usage:
    python eval_hierarchical_infill.py \
        --checkpoint runs/hier_v1/best.pt \
        --val-jsonl data/synth_v1/infill/character_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir eval_output/hier_v1 \
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
from torch.utils.data import DataLoader, Subset

from train_hierarchical_infill import (
    HierarchicalInfiller,
    HierarchicalInfillDataset,
    collate_hier,
)
from train_character_infill import read_jsonl


# ============================================================
# Rendering helpers
# ============================================================

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


def add_label(img: Image.Image, text: str, font_size: int = 11) -> Image.Image:
    pad = font_size + 4
    out = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((2, 1), text, fill=(0, 0, 0))
    out.paste(img, (0, pad))
    return out


def make_contact_sheet(
    panels: List[Image.Image], cols: int = 3, pad: int = 3,
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


def build_panel(
    before: Tensor,
    bbox_mask: Tensor,
    coarse_up: Tensor,
    pred_after: Tensor,
    gt_after: Tensor,
    char_text: str,
    record_id: str,
    refine_l1: float,
    coarse_l1: float,
    cell_size: int = 128,
) -> Image.Image:
    cols = [
        add_label(tensor_to_gray_pil(before, cell_size), "before"),
        add_label(tensor_to_gray_pil(bbox_mask, cell_size), "mask"),
        add_label(tensor_to_gray_pil((before + coarse_up).clamp(0, 1), cell_size), f"coarse L1={coarse_l1:.4f}"),
        add_label(tensor_to_gray_pil(pred_after, cell_size), f"refined L1={refine_l1:.4f}"),
        add_label(tensor_to_gray_pil(gt_after, cell_size), "ground truth"),
        add_label(tensor_to_error_pil(pred_after, gt_after, bbox_mask, cell_size), "error"),
    ]
    header_h = 14
    total_w = sum(c.width for c in cols) + 5 * 2
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
    p = argparse.ArgumentParser(description="Evaluate hierarchical infill model")
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
    val_metrics = ckpt.get("val_metrics", {})
    print(f"  Epoch {epoch}, val_loss={val_metrics.get('total', '?')}")

    # Style vocab
    ckpt_dir = Path(args.checkpoint).parent
    style_path = ckpt_dir / "style_to_index.json"
    if style_path.exists():
        with style_path.open() as f:
            style_to_index = json.load(f)
    else:
        style_to_index = {"__unk__": 0}
    num_styles = len(style_to_index)

    coarse_size = int(model_args.get("coarse_size", 64))
    full_size = int(model_args.get("patch_size", 256))

    model = HierarchicalInfiller(
        cond_dim=int(model_args.get("cond_dim", 128)),
        coarse_ch=int(model_args.get("coarse_ch", 64)),
        refine_ch=int(model_args.get("refine_ch", 64)),
        vocab_size=256,
        num_styles=num_styles,
        coarse_size=coarse_size,
        full_size=full_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"  {n_params:,} parameters, {num_styles} styles")

    ds = HierarchicalInfillDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        patch_size=full_size,
        coarse_size=coarse_size,
    )
    end = min(len(ds), args.offset + args.num_samples)
    print(f"Evaluating samples {args.offset}..{end} of {len(ds)} total")

    subset = Subset(ds, list(range(args.offset, end)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_hier)

    panels: List[Image.Image] = []
    results: List[Dict[str, Any]] = []
    all_coarse_l1 = []
    all_refine_l1 = []
    all_outside_l1 = []

    sample_idx = 0
    for batch in loader:
        batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            coarse_delta, refine_delta = model(
                before=batch_d["before"],
                bbox_mask=batch_d["bbox_mask"],
                before_c=batch_d["before_c"],
                bbox_mask_c=batch_d["bbox_mask_c"],
                char_tokens=batch_d["char_tokens"],
                char_lengths=batch_d["char_lengths"],
                style_index=batch_d["style_index"],
                bbox=batch_d["bbox"],
            )

        coarse_up = F.interpolate(coarse_delta, size=full_size, mode="bilinear", align_corners=False)
        pred_after = batch_d["before"] + refine_delta

        bs = batch_d["before"].shape[0]
        for j in range(bs):
            inside = batch_d["bbox_mask"][j]
            outside = 1.0 - inside
            # Refine metrics
            err_r = torch.abs(refine_delta[j] - batch_d["delta"][j])
            r_in = float((err_r * inside).sum() / inside.sum().clamp_min(1))
            r_out = float((err_r * outside).sum() / outside.sum().clamp_min(1))
            # Coarse metrics (at full res for comparison)
            err_c = torch.abs(coarse_up[j] - batch_d["delta"][j])
            c_in = float((err_c * inside).sum() / inside.sum().clamp_min(1))

            all_coarse_l1.append(c_in)
            all_refine_l1.append(r_in)
            all_outside_l1.append(r_out)

            rec_idx = args.offset + sample_idx
            rec = ds.records[rec_idx]
            char_text = rec.get("char_text", "?")
            record_id = batch["record_id"][j]

            panel = build_panel(
                before=batch_d["before"][j].cpu(),
                bbox_mask=batch_d["bbox_mask"][j].cpu(),
                coarse_up=coarse_up[j].cpu(),
                pred_after=pred_after[j].cpu(),
                gt_after=batch_d["after"][j].cpu(),
                char_text=char_text,
                record_id=record_id,
                refine_l1=r_in,
                coarse_l1=c_in,
                cell_size=args.cell_size,
            )
            panels.append(panel)

            results.append({
                "index": sample_idx,
                "record_id": record_id,
                "char_text": char_text,
                "coarse_inside_l1": c_in,
                "refine_inside_l1": r_in,
                "outside_l1": r_out,
            })
            sample_idx += 1

        print(f"  Processed {sample_idx}/{end - args.offset} samples")

    sheet = make_contact_sheet(panels, cols=args.cols)
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\nContact sheet: {sheet_path}")

    avg_c = sum(all_coarse_l1) / max(1, len(all_coarse_l1))
    avg_r = sum(all_refine_l1) / max(1, len(all_refine_l1))
    avg_o = sum(all_outside_l1) / max(1, len(all_outside_l1))
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "num_samples": len(results),
        "avg_coarse_inside_l1": avg_c,
        "avg_refine_inside_l1": avg_r,
        "avg_outside_l1": avg_o,
        "samples": results,
    }
    with (out_dir / "eval_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Avg coarse inside L1:  {avg_c:.5f}")
    print(f"Avg refine inside L1:  {avg_r:.5f}")
    print(f"Avg outside L1:        {avg_o:.5f}")
    print(f"Summary: {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
