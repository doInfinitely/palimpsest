#!/usr/bin/env python3
"""
Generate multi-scale training data for the fractal infill model.

Reads existing tree annotations and page images to produce infill samples
at multiple hierarchy levels, all scaled/padded to a 256×256 retina:

  - Line-level:  line crop → retina, child_bboxes = word regions
  - Word-level:  word crop → retina, child_bboxes = character regions
  - Character-level: char crop within word → retina, child_bboxes = [] (leaf)

Each sample records the child bboxes in retina-normalised [cx, cy, w, h]
so the fractal model learns when to recurse and where.

Usage:
    python generate_fractal_data.py \
        --data-dir data/synth_v1 \
        --out-dir data/synth_v1 \
        --val-fraction 0.08
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

RETINA_SIZE = 256


def scale_and_pad(
    img: Image.Image,
    retina_size: int = RETINA_SIZE,
    pad_val: int = 255,
) -> Tuple[Image.Image, float, int, int]:
    """Scale image to fit retina, centre on canvas.

    Returns (padded_image, scale, offset_x, offset_y).
    """
    w, h = img.size
    if w == 0 or h == 0:
        canvas = Image.new(img.mode, (retina_size, retina_size), pad_val)
        return canvas, 1.0, 0, 0
    scale = min(retina_size / w, retina_size / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new(img.mode, (retina_size, retina_size), pad_val)
    ox = (retina_size - nw) // 2
    oy = (retina_size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def abs_bbox_to_retina_norm(
    child_bbox_abs: Tuple[int, int, int, int],
    parent_bbox_abs: Tuple[int, int, int, int],
    scale: float,
    ox: int,
    oy: int,
    retina_size: int = RETINA_SIZE,
) -> List[float]:
    """Convert a child's absolute bbox to retina-normalised [cx, cy, w, h].

    The parent has been scaled/padded to the retina with (scale, ox, oy).
    """
    px1, py1, px2, py2 = parent_bbox_abs
    cx1, cy1, cx2, cy2 = child_bbox_abs

    # Child coords relative to parent origin, then scaled to retina
    rx1 = (cx1 - px1) * scale + ox
    ry1 = (cy1 - py1) * scale + oy
    rx2 = (cx2 - px1) * scale + ox
    ry2 = (cy2 - py1) * scale + oy

    # Normalise to [0, 1]
    cx = (rx1 + rx2) / 2.0 / retina_size
    cy = (ry1 + ry2) / 2.0 / retina_size
    w = (rx2 - rx1) / retina_size
    h = (ry2 - ry1) / retina_size

    return [round(cx, 6), round(cy, 6), round(max(w, 0.001), 6), round(max(h, 0.001), 6)]


def compute_mask(
    page_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    threshold: int = 30,
) -> Image.Image:
    """Binary mask of non-background pixels inside bbox."""
    crop = page_img.crop(bbox)
    arr = np.array(crop.convert("L"), dtype=np.float32)
    bg_val = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    mask = (np.abs(arr - bg_val) > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def erase_region(
    page_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Erase a bbox region by filling with background color."""
    img = page_img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(list(bbox), fill=bg_color)
    return img


def erase_children(
    page_img: Image.Image,
    child_bboxes: List[Tuple[int, int, int, int]],
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Erase all child regions from page image."""
    img = page_img.copy()
    draw = ImageDraw.Draw(img)
    for bbox in child_bboxes:
        draw.rectangle(list(bbox), fill=bg_color)
    return img


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_gray(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("L").save(str(path))


def generate_fractal_samples(
    tree: Dict[str, Any],
    page_img: Image.Image,
    out_dir: Path,
    retina_size: int = RETINA_SIZE,
) -> List[Dict[str, Any]]:
    """Generate multi-scale fractal infill samples from one document."""
    doc_id = tree["document_id"]
    style_id = tree.get("style_id", f"style_{doc_id}")
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}

    bg_color = (255, 255, 255)
    # Try to detect background color from page corners
    arr = np.array(page_img)
    if arr.ndim == 3:
        corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
        bg_color = tuple(int(np.median([c[i] for c in corners])) for i in range(3))

    records = []
    infill_dir = out_dir / "infill_fractal"

    # ------------------------------------------------------------------
    # Word-level samples: word crop, child_bboxes = character bboxes
    # ------------------------------------------------------------------
    for node in tree["nodes"]:
        if node["level"] != "word":
            continue

        word_bbox = tuple(node["bbox_abs_xyxy"])
        wx1, wy1, wx2, wy2 = word_bbox
        if wx2 - wx1 < 4 or wy2 - wy1 < 4:
            continue

        char_nodes = [nodes_by_id[cid] for cid in node["child_ids"]
                      if cid in nodes_by_id]
        if not char_nodes:
            continue

        char_bboxes_abs = [tuple(cn["bbox_abs_xyxy"]) for cn in char_nodes]

        # "after" = word crop with all characters present
        after_crop = page_img.crop(word_bbox)

        # "before" = word crop with all characters erased
        before_page = erase_children(page_img, char_bboxes_abs, bg_color)
        before_crop = before_page.crop(word_bbox)

        # Full-word bbox mask (entire word region is the target)
        word_mask = Image.new("L", (wx2 - wx1, wy2 - wy1), 255)

        # Scale to retina
        after_retina, sc, ox, oy = scale_and_pad(after_crop, retina_size,
                                                  pad_val=int(np.mean(bg_color)))
        before_retina, _, _, _ = scale_and_pad(before_crop, retina_size,
                                                pad_val=int(np.mean(bg_color)))
        mask_retina, _, _, _ = scale_and_pad(word_mask, retina_size, pad_val=0)

        # Child bboxes in retina-normalised coords
        child_bboxes = []
        for cb in char_bboxes_abs:
            child_bboxes.append(abs_bbox_to_retina_norm(cb, word_bbox, sc, ox, oy, retina_size))

        word_text = node.get("text", "")
        record_id = f"fractal_word_{node['id']}"

        before_path = f"infill_fractal/{record_id}_before.png"
        after_path = f"infill_fractal/{record_id}_after.png"
        mask_path = f"infill_fractal/{record_id}_mask.png"

        save_gray(before_retina, out_dir / before_path)
        save_gray(after_retina, out_dir / after_path)
        save_gray(mask_retina, out_dir / mask_path)

        records.append({
            "schema_version": "2.0",
            "record_id": record_id,
            "document_id": doc_id,
            "level": "word",
            "node_id": node["id"],
            "retina_size": retina_size,
            "before_patch_ref": before_path,
            "after_patch_ref": after_path,
            "bbox_mask_ref": mask_path,
            "char_text": word_text,
            "style_id": style_id,
            "target_bbox_parent_norm_cxcywh": node.get("bbox_parent_norm_cxcywh", [0.5, 0.5, 1.0, 1.0]),
            "child_bboxes": child_bboxes,
            "confidence": 1.0,
        })

    # ------------------------------------------------------------------
    # Character-level samples: char within word crop, child_bboxes = []
    # ------------------------------------------------------------------
    for node in tree["nodes"]:
        if node["level"] != "character":
            continue

        char_node = node
        word_node = nodes_by_id.get(char_node["parent_id"])
        if word_node is None:
            continue

        word_bbox = tuple(word_node["bbox_abs_xyxy"])
        char_bbox = tuple(char_node["bbox_abs_xyxy"])

        wx1, wy1, wx2, wy2 = word_bbox
        if wx2 - wx1 < 4 or wy2 - wy1 < 4:
            continue

        # "after" = word crop with character present
        after_crop = page_img.crop(word_bbox)

        # "before" = word crop with this character erased
        before_page = erase_region(page_img, char_bbox, bg_color)
        before_crop = before_page.crop(word_bbox)

        # Bbox mask: character region within word
        cx1 = max(0, char_bbox[0] - wx1)
        cy1 = max(0, char_bbox[1] - wy1)
        cx2 = min(wx2 - wx1, char_bbox[2] - wx1)
        cy2 = min(wy2 - wy1, char_bbox[3] - wy1)
        char_mask = Image.new("L", (wx2 - wx1, wy2 - wy1), 0)
        ImageDraw.Draw(char_mask).rectangle([cx1, cy1, cx2, cy2], fill=255)

        # Scale to retina
        after_retina, sc, ox, oy = scale_and_pad(after_crop, retina_size,
                                                  pad_val=int(np.mean(bg_color)))
        before_retina, _, _, _ = scale_and_pad(before_crop, retina_size,
                                                pad_val=int(np.mean(bg_color)))
        mask_retina, _, _, _ = scale_and_pad(char_mask, retina_size, pad_val=0)

        char_text = char_node.get("text", "")
        record_id = f"fractal_char_{char_node['id']}"

        before_path = f"infill_fractal/{record_id}_before.png"
        after_path = f"infill_fractal/{record_id}_after.png"
        mask_path = f"infill_fractal/{record_id}_mask.png"

        save_gray(before_retina, out_dir / before_path)
        save_gray(after_retina, out_dir / after_path)
        save_gray(mask_retina, out_dir / mask_path)

        records.append({
            "schema_version": "2.0",
            "record_id": record_id,
            "document_id": doc_id,
            "level": "character",
            "node_id": char_node["id"],
            "retina_size": retina_size,
            "before_patch_ref": before_path,
            "after_patch_ref": after_path,
            "bbox_mask_ref": mask_path,
            "char_text": char_text,
            "style_id": style_id,
            "target_bbox_parent_norm_cxcywh": char_node.get(
                "bbox_parent_norm_cxcywh", [0.5, 0.5, 0.5, 0.5]),
            "child_bboxes": [],  # leaf node — stop signal
            "confidence": 1.0,
        })

    # ------------------------------------------------------------------
    # Line-level samples: line crop, child_bboxes = word bboxes
    # ------------------------------------------------------------------
    for node in tree["nodes"]:
        if node["level"] != "line":
            continue

        line_bbox = tuple(node["bbox_abs_xyxy"])
        lx1, ly1, lx2, ly2 = line_bbox
        if lx2 - lx1 < 8 or ly2 - ly1 < 4:
            continue

        word_nodes = [nodes_by_id[cid] for cid in node["child_ids"]
                      if cid in nodes_by_id]
        if not word_nodes:
            continue

        word_bboxes_abs = [tuple(wn["bbox_abs_xyxy"]) for wn in word_nodes]

        after_crop = page_img.crop(line_bbox)
        before_page = erase_children(page_img, word_bboxes_abs, bg_color)
        before_crop = before_page.crop(line_bbox)
        line_mask = Image.new("L", (lx2 - lx1, ly2 - ly1), 255)

        after_retina, sc, ox, oy = scale_and_pad(after_crop, retina_size,
                                                  pad_val=int(np.mean(bg_color)))
        before_retina, _, _, _ = scale_and_pad(before_crop, retina_size,
                                                pad_val=int(np.mean(bg_color)))
        mask_retina, _, _, _ = scale_and_pad(line_mask, retina_size, pad_val=0)

        child_bboxes = []
        for wb in word_bboxes_abs:
            child_bboxes.append(abs_bbox_to_retina_norm(wb, line_bbox, sc, ox, oy, retina_size))

        line_text = node.get("text", "")
        record_id = f"fractal_line_{node['id']}"

        before_path = f"infill_fractal/{record_id}_before.png"
        after_path = f"infill_fractal/{record_id}_after.png"
        mask_path = f"infill_fractal/{record_id}_mask.png"

        save_gray(before_retina, out_dir / before_path)
        save_gray(after_retina, out_dir / after_path)
        save_gray(mask_retina, out_dir / mask_path)

        records.append({
            "schema_version": "2.0",
            "record_id": record_id,
            "document_id": doc_id,
            "level": "line",
            "node_id": node["id"],
            "retina_size": retina_size,
            "before_patch_ref": before_path,
            "after_patch_ref": after_path,
            "bbox_mask_ref": mask_path,
            "char_text": line_text,
            "style_id": style_id,
            "target_bbox_parent_norm_cxcywh": node.get(
                "bbox_parent_norm_cxcywh", [0.5, 0.5, 1.0, 1.0]),
            "child_bboxes": child_bboxes,
            "confidence": 1.0,
        })

    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Generate fractal infill training data")
    p.add_argument("--data-dir", required=True, help="Directory with documents/ and images/")
    p.add_argument("--out-dir", required=True, help="Output directory for patches and JSONL")
    p.add_argument("--val-fraction", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retina-size", type=int, default=RETINA_SIZE)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all tree annotation files
    doc_dir = data_dir / "documents"
    doc_files = sorted(doc_dir.glob("*.json"))
    print(f"Found {len(doc_files)} documents in {doc_dir}")

    all_records: List[Dict[str, Any]] = []

    for i, doc_path in enumerate(doc_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(doc_files)}...")

        with doc_path.open() as f:
            tree = json.load(f)

        doc_id = tree["document_id"]
        img_path = data_dir / "images" / f"{doc_id}.png"
        if not img_path.exists():
            continue

        page_img = Image.open(img_path).convert("RGB")

        try:
            records = generate_fractal_samples(
                tree, page_img, out_dir, retina_size=args.retina_size)
            all_records.extend(records)
        except Exception as e:
            print(f"  WARNING: Failed on {doc_id}: {e}")
            continue

    # Split train/val
    random.shuffle(all_records)
    n_val = max(1, int(len(all_records) * args.val_fraction))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    # Write JSONL
    infill_dir = out_dir / "infill_fractal"
    write_jsonl(infill_dir / "fractal_infill.jsonl", train_records)
    write_jsonl(infill_dir / "fractal_infill_val.jsonl", val_records)

    # Stats
    from collections import Counter
    train_levels = Counter(r["level"] for r in train_records)
    val_levels = Counter(r["level"] for r in val_records)
    avg_children = np.mean([len(r["child_bboxes"]) for r in all_records])

    print(f"\nGenerated {len(all_records)} total fractal infill samples")
    print(f"  Train: {len(train_records)} — {dict(train_levels)}")
    print(f"  Val:   {len(val_records)} — {dict(val_levels)}")
    print(f"  Avg child bboxes per sample: {avg_children:.1f}")
    print(f"Output: {infill_dir}/")


if __name__ == "__main__":
    main()
