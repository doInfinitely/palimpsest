#!/usr/bin/env python3
"""Evaluate a trained OnlineWordScribe checkpoint.

Loads a checkpoint + JSONL data, runs greedy decode on N samples,
renders side-by-side gold vs predicted stroke paths, and produces
a contact sheet PNG + per-sample raster comparisons.

Usage:
    python eval_scribe.py \
        --checkpoint runs/combined_v1/best.pt \
        --jsonl data/iam_processed/val_word_routes.jsonl \
        --out-dir eval_output/combined_v1 \
        --num-samples 20
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from train_online_scribe import (
    OnlineWordScribe,
    greedy_decode,
    read_jsonl,
    soft_rasterize_segments,
)


# ============================================================
# Drawing helpers
# ============================================================

def events_to_absolute(start_xy: Tuple[float, float], events: Tensor) -> List[Tuple[float, float, float]]:
    """Convert (dx,dy,pen_down,...) events to list of (x, y, pen_down)."""
    x, y = start_xy
    pts = []
    for i in range(events.shape[0]):
        dx, dy = float(events[i, 0]), float(events[i, 1])
        pen = float(events[i, 2])
        x += dx
        y += dy
        pts.append((x, y, pen))
    return pts


def draw_strokes(
    size: int,
    pts: List[Tuple[float, float, float]],
    bg_color: Tuple[int, ...] = (255, 255, 255),
    ink_color: Tuple[int, ...] = (0, 0, 200),
    travel_color: Tuple[int, ...] = (200, 200, 200),
    line_width: int = 2,
) -> Image.Image:
    """Draw pen strokes on an image, scaling points to fit."""
    if not pts:
        return Image.new("RGB", (size, size), bg_color)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    margin = 0.1
    scale = (1 - 2 * margin) * size / max(span_x, span_y)
    off_x = margin * size - min_x * scale + (max(span_x, span_y) - span_x) * scale / 2
    off_y = margin * size - min_y * scale + (max(span_x, span_y) - span_y) * scale / 2

    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    for i in range(1, len(pts)):
        x0 = pts[i - 1][0] * scale + off_x
        y0 = pts[i - 1][1] * scale + off_y
        x1 = pts[i][0] * scale + off_x
        y1 = pts[i][1] * scale + off_y
        pen = pts[i][2]
        color = ink_color if pen > 0.5 else travel_color
        w = line_width if pen > 0.5 else 1
        draw.line([(x0, y0), (x1, y1)], fill=color, width=w)

    # start dot
    sx = pts[0][0] * scale + off_x
    sy = pts[0][1] * scale + off_y
    r = 3
    draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=(0, 180, 0))

    return img


def add_label(img: Image.Image, text: str, font_size: int = 12) -> Image.Image:
    """Add a label above the image."""
    pad = font_size + 6
    out = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), text, fill=(0, 0, 0))
    out.paste(img, (0, pad))
    return out


def make_contact_sheet(
    panels: List[Image.Image],
    cols: int = 4,
    pad: int = 4,
) -> Image.Image:
    """Arrange panels into a grid."""
    if not panels:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    pw = max(p.width for p in panels)
    ph = max(p.height for p in panels)
    rows = math.ceil(len(panels) / cols)

    sheet = Image.new("RGB", (cols * (pw + pad) + pad, rows * (ph + pad) + pad), (240, 240, 240))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        sheet.paste(p, (c * (pw + pad) + pad, r * (ph + pad) + pad))
    return sheet


def tensor_to_heatmap(t: Tensor, size: int = 128) -> Image.Image:
    """Convert [1,H,W] tensor to RGB heatmap."""
    t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    t = t.clamp(0, 1)
    arr = (t * 255).byte().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


# ============================================================
# Main
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate OnlineWordScribe checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--jsonl", required=True, help="JSONL file with word routes")
    p.add_argument("--out-dir", required=True, help="Output directory for eval results")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-decode-steps", type=int, default=512)
    p.add_argument("--stop-threshold", type=float, default=0.3)
    p.add_argument("--render-size", type=int, default=192)
    p.add_argument("--raster-size", type=int, default=64)
    p.add_argument("--raster-sigma-px", type=float, default=1.5)
    p.add_argument("--cols", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    style_to_index: Dict[str, int] = ckpt["style_to_index"]
    model_args = ckpt["args"]

    model = OnlineWordScribe(
        num_styles=len(style_to_index),
        d_model=int(model_args.get("d_model", 256)),
        nhead=int(model_args.get("nhead", 8)),
        text_layers=int(model_args.get("text_layers", 2)),
        dec_layers=int(model_args.get("dec_layers", 6)),
        dxdy_scale=float(model_args.get("dxdy_scale", 1.0)),
        canvas_size=int(model_args.get("canvas_size", 64)),
        snapshot_interval=int(model_args.get("snapshot_interval", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_metrics = ckpt.get("val_metrics", {})
    print(f"  Epoch {epoch}, val_loss={val_metrics.get('total', '?'):.4f}")
    print(f"  {len(style_to_index)} styles, d_model={model_args.get('d_model', 256)}")

    # Load data
    records = read_jsonl(args.jsonl)
    end = min(len(records), args.offset + args.num_samples)
    chosen = records[args.offset:end]
    print(f"Evaluating {len(chosen)} samples...")

    panels: List[Image.Image] = []
    results: List[Dict[str, Any]] = []

    for i, rec in enumerate(chosen):
        word_text = rec["word_text"]
        word_id = rec.get("word_id", f"sample_{i}")

        # Prepare inputs
        state_before = torch.ones(1, 1, 256, 256, device=device)
        text_tokens = torch.tensor(
            [list(word_text.encode("utf-8"))[:64]], dtype=torch.long, device=device
        )
        text_lengths = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)
        bbox = torch.tensor(
            [rec.get("target_bbox_parent_norm_cxcywh", [0.5, 0.5, 0.8, 0.3])],
            dtype=torch.float32, device=device,
        )
        style_id = str(rec.get("style_id", rec.get("writer_id", "__unk__")))
        si = style_to_index.get(style_id, style_to_index.get("__unk__", 0))
        style_index = torch.tensor([si], dtype=torch.long, device=device)

        # Greedy decode
        route_sp = rec.get("route_start_point", [0.0, 0.0])
        route_start_t = torch.tensor([route_sp], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_events = greedy_decode(
                model=model,
                state_before=state_before,
                text_tokens=text_tokens,
                text_lengths=text_lengths,
                bbox=bbox,
                style_index=style_index,
                route_start_point=route_start_t,
                max_steps=args.max_decode_steps,
                stop_threshold=args.stop_threshold,
            )

        # Gold events
        gold_rows = []
        for e in rec["events"]:
            gold_rows.append([
                float(e["dx"]), float(e["dy"]),
                float(e["pen_down"]), float(e["stroke_end"]),
                float(e["char_end"]), float(e["word_end"]),
                float(e["seq_end"]),
            ])
        gold_events = torch.tensor(gold_rows, dtype=torch.float32) if gold_rows else torch.zeros(0, 7)

        # Get start point
        start = rec.get("route_start_point", [0.0, 0.0])

        # Draw stroke paths
        gold_pts = events_to_absolute(tuple(start), gold_events)
        pred_pts = events_to_absolute(tuple(start), pred_events)

        gold_img = draw_strokes(args.render_size, gold_pts, ink_color=(0, 0, 180))
        pred_img = draw_strokes(args.render_size, pred_pts, ink_color=(180, 0, 0))

        # Rasterize for comparison
        if start != [0.0, 0.0] and len(gold_events) > 0 and len(pred_events) > 0:
            start_t = torch.tensor([start], dtype=torch.float32, device=device)
            with torch.no_grad():
                gold_raster = soft_rasterize_segments(
                    start_points=start_t,
                    dxdy=gold_events[:, 0:2].unsqueeze(0).to(device),
                    pen_probs=gold_events[:, 2].unsqueeze(0).to(device),
                    raster_size=args.raster_size,
                    sigma_px=args.raster_sigma_px,
                )[0].cpu()
                pred_raster = soft_rasterize_segments(
                    start_points=start_t,
                    dxdy=pred_events[:, 0:2].unsqueeze(0).to(device),
                    pen_probs=pred_events[:, 2].unsqueeze(0).to(device),
                    raster_size=args.raster_size,
                    sigma_px=args.raster_sigma_px,
                )[0].cpu()
            raster_l1 = float(torch.abs(gold_raster - pred_raster).mean())
        else:
            raster_l1 = float("nan")

        # Build side-by-side panel
        label = f'"{word_text}" ({word_id})'
        gold_labeled = add_label(gold_img, f"gold: {len(gold_events)}ev")
        pred_labeled = add_label(pred_img, f"pred: {len(pred_events)}ev")

        pair = Image.new("RGB", (gold_labeled.width + pred_labeled.width + 4, gold_labeled.height + 18), (255, 255, 255))
        draw = ImageDraw.Draw(pair)
        draw.text((4, 2), label[:40], fill=(0, 0, 0))
        pair.paste(gold_labeled, (0, 18))
        pair.paste(pred_labeled, (gold_labeled.width + 4, 18))

        panels.append(pair)

        # Save individual panel
        pair.save(out_dir / f"{i:03d}_{word_id}.png")

        results.append({
            "index": i,
            "word_id": word_id,
            "word_text": word_text,
            "gold_events": len(gold_events),
            "pred_events": len(pred_events),
            "raster_l1": raster_l1,
        })

        print(f"  [{i+1}/{len(chosen)}] '{word_text}' gold={len(gold_events)} pred={len(pred_events)} raster_l1={raster_l1:.4f}")

    # Contact sheet
    sheet = make_contact_sheet(panels, cols=args.cols)
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\nContact sheet: {sheet_path}")

    # Summary
    summary = {
        "checkpoint": args.checkpoint,
        "jsonl": args.jsonl,
        "epoch": epoch,
        "num_samples": len(results),
        "avg_raster_l1": sum(r["raster_l1"] for r in results if not math.isnan(r["raster_l1"])) / max(1, sum(1 for r in results if not math.isnan(r["raster_l1"]))),
        "samples": results,
    }
    summary_path = out_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")
    print(f"Avg raster L1: {summary['avg_raster_l1']:.4f}")


if __name__ == "__main__":
    main()
