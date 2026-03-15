#!/usr/bin/env python3
"""
Debug visualization for stroke routes and predicted pen events.

Renders word routes from JSONL data as SVG or PNG for visual inspection.

Usage:
    python debug_render_strokes.py --jsonl data/hw_v1/train_word_routes.jsonl --num 20 --out-dir debug_vis/
    python debug_render_strokes.py --jsonl data/hw_v1/train_word_routes.jsonl --index 42 --out-dir debug_vis/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw


def events_to_points(events: Sequence[Dict[str, Any]]) -> List[dict]:
    """Convert event dicts to list of absolute points with metadata."""
    points = []
    x, y = 0.0, 0.0
    for e in events:
        x += float(e["dx"])
        y += float(e["dy"])
        points.append({
            "x": x, "y": y,
            "pen_down": float(e["pen_down"]) > 0.5,
            "stroke_end": float(e["stroke_end"]) > 0.5,
            "char_end": float(e["char_end"]) > 0.5,
            "word_end": float(e["word_end"]) > 0.5,
        })
    return points


def render_word_route_png(
    record: Dict[str, Any],
    canvas_size: int = 512,
    line_width: int = 2,
    show_pen_up: bool = True,
) -> Image.Image:
    """Render a word route record as a PNG image."""
    events = record["events"]
    points = events_to_points(events)

    if not points:
        return Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

    # Find bounding box
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max(max_x - min_x, 0.001)
    range_y = max(max_y - min_y, 0.001)
    margin = 0.1
    scale = canvas_size * (1 - 2 * margin) / max(range_x, range_y)
    offset_x = canvas_size * margin - min_x * scale + (max(range_x, range_y) - range_x) * scale / 2
    offset_y = canvas_size * margin - min_y * scale + (max(range_x, range_y) - range_y) * scale / 2

    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Color palette for different strokes
    colors = [
        (30, 30, 180), (180, 30, 30), (30, 150, 30), (150, 30, 150),
        (30, 150, 150), (180, 120, 30), (100, 100, 100), (180, 30, 120),
    ]
    pen_up_color = (200, 200, 200)

    stroke_idx = 0
    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]

        x0 = p0["x"] * scale + offset_x
        y0 = p0["y"] * scale + offset_y
        x1 = p1["x"] * scale + offset_x
        y1 = p1["y"] * scale + offset_y

        if p1["pen_down"]:
            color = colors[stroke_idx % len(colors)]
            draw.line([(x0, y0), (x1, y1)], fill=color, width=line_width)
        elif show_pen_up:
            draw.line([(x0, y0), (x1, y1)], fill=pen_up_color, width=1)

        if p0.get("stroke_end", False):
            stroke_idx += 1

    # Draw start point
    if points:
        sx = points[0]["x"] * scale + offset_x
        sy = points[0]["y"] * scale + offset_y
        r = 4
        draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=(0, 200, 0))

    # Draw end point
    if len(points) > 1:
        ex = points[-1]["x"] * scale + offset_x
        ey = points[-1]["y"] * scale + offset_y
        r = 4
        draw.ellipse([(ex - r, ey - r), (ex + r, ey + r)], fill=(200, 0, 0))

    # Add text label
    word_text = record.get("word_text", "?")
    draw.text((5, 5), f'"{word_text}"', fill=(0, 0, 0))
    draw.text((5, 20), f'{len(events)} events, {stroke_idx + 1} strokes', fill=(80, 80, 80))

    return img


def render_grid(
    records: Sequence[Dict[str, Any]],
    cols: int = 5,
    cell_size: int = 256,
) -> Image.Image:
    """Render a grid of word routes."""
    n = len(records)
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), (240, 240, 240))

    for i, rec in enumerate(records):
        img = render_word_route_png(rec, canvas_size=cell_size)
        row, col = divmod(i, cols)
        grid.paste(img, (col * cell_size, row * cell_size))

    return grid


def main():
    parser = argparse.ArgumentParser(description="Debug visualize word stroke routes")
    parser.add_argument("--jsonl", required=True, help="Path to word_routes.jsonl")
    parser.add_argument("--num", type=int, default=25, help="Number of samples to render")
    parser.add_argument("--index", type=int, default=None, help="Specific record index to render")
    parser.add_argument("--out-dir", type=str, default="debug_vis")
    parser.add_argument("--cell-size", type=int, default=256)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pen-up", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.jsonl}...")
    records = []
    with open(args.jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  {len(records)} records")

    if args.index is not None:
        rec = records[args.index]
        img = render_word_route_png(rec, canvas_size=512, show_pen_up=not args.no_pen_up)
        path = out_dir / f"route_{args.index:05d}.png"
        img.save(str(path))
        print(f"  Saved: {path}")
        return

    # Sample and render grid
    indices = list(range(len(records)))
    if len(indices) > args.num:
        indices = random.sample(indices, args.num)
    selected = [records[i] for i in sorted(indices)]

    grid = render_grid(selected, cols=args.cols, cell_size=args.cell_size)
    grid_path = out_dir / "route_grid.png"
    grid.save(str(grid_path))
    print(f"  Saved grid: {grid_path} ({len(selected)} samples)")

    # Also save individual images
    for i, rec in enumerate(selected):
        img = render_word_route_png(rec, canvas_size=args.cell_size, show_pen_up=not args.no_pen_up)
        path = out_dir / f"route_{i:03d}_{rec.get('word_text', 'unk')}.png"
        img.save(str(path))

    print(f"  Saved {len(selected)} individual images to {out_dir}/")


if __name__ == "__main__":
    main()
