#!/usr/bin/env python3
"""
Generate word-level infill training data from IAM handwriting dataset.

Reconstructs line images by compositing word crops at their XML-annotated
positions, then creates before/after/mask triplets for each word.

Inputs:
  - data/iam_ondb/*.xml          — line/word hierarchy with absolute bboxes
  - data/iam_words/iam_words/words/  — word image crops (grayscale PNGs)
  - data/iam_full/style_to_index.json — writer→style index mapping
  - data/iam_full/writer_meta.json    — writer train/val split

Output:
  - data/iam_full/infill/           — before/after/mask PNGs + JSONL

Usage:
    python generate_iam_infill.py \
        --xml-dir data/iam_ondb \
        --words-dir data/iam_words/iam_words/words \
        --style-json data/iam_full/style_to_index.json \
        --writer-meta data/iam_full/writer_meta.json \
        --out-dir data/iam_full/infill \
        --retina-size 256
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

RETINA_SIZE = 256


# ============================================================
# XML parsing
# ============================================================

def parse_form_xml(xml_path: str) -> dict:
    """Parse an IAM form XML into a structured dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    form_id = root.get("id")
    writer_id = root.get("writer-id")  # e.g. "000"

    lines = []
    hw_part = root.find("handwritten-part")
    if hw_part is None:
        return {"form_id": form_id, "writer_id": writer_id, "lines": []}

    for line_el in hw_part.iter("line"):
        line_id = line_el.get("id")
        line_text = line_el.get("text", "")
        seg = line_el.get("segmentation", "")

        words = []
        for word_el in line_el.iter("word"):
            word_id = word_el.get("id")
            word_text = word_el.get("text", "")

            # Get word bbox from component elements
            cmps = list(word_el.iter("cmp"))
            if not cmps:
                continue

            xs = [int(c.get("x")) for c in cmps]
            ys = [int(c.get("y")) for c in cmps]
            ws = [int(c.get("width")) for c in cmps]
            hs = [int(c.get("height")) for c in cmps]

            x1 = min(xs)
            y1 = min(ys)
            x2 = max(x + w for x, w in zip(xs, ws))
            y2 = max(y + h for y, h in zip(ys, hs))

            words.append({
                "word_id": word_id,
                "word_text": word_text,
                "bbox_abs": (x1, y1, x2, y2),
            })

        if words:
            lines.append({
                "line_id": line_id,
                "line_text": line_text,
                "segmentation": seg,
                "words": words,
            })

    return {"form_id": form_id, "writer_id": writer_id, "lines": lines}


# ============================================================
# Image compositing
# ============================================================

def word_image_path(words_dir: str, word_id: str) -> str:
    """Construct path to word image: words/{prefix}/{doc}/{word_id}.png"""
    parts = word_id.split("-")
    prefix = parts[0]
    doc = f"{parts[0]}-{parts[1]}"
    return os.path.join(words_dir, prefix, doc, f"{word_id}.png")


def composite_line(
    words: List[dict],
    words_dir: str,
    bg_val: int = 255,
    pad_px: int = 10,
) -> Tuple[Optional[Image.Image], List[Tuple[int, int, int, int]]]:
    """Composite word crops into a reconstructed line image.

    Returns (line_image, word_bboxes_in_line_coords) or (None, []) on failure.
    """
    # Collect word images and their absolute positions
    word_imgs = []
    for w in words:
        img_path = word_image_path(words_dir, w["word_id"])
        if not os.path.exists(img_path):
            return None, []
        try:
            img = Image.open(img_path).convert("L")
            img.load()  # force decode
        except Exception:
            return None, []
        word_imgs.append(img)

    # Compute line bounding box from all word bboxes
    all_x1 = min(w["bbox_abs"][0] for w in words)
    all_y1 = min(w["bbox_abs"][1] for w in words)
    all_x2 = max(w["bbox_abs"][2] for w in words)
    all_y2 = max(w["bbox_abs"][3] for w in words)

    # Add padding
    all_x1 -= pad_px
    all_y1 -= pad_px
    all_x2 += pad_px
    all_y2 += pad_px

    line_w = all_x2 - all_x1
    line_h = all_y2 - all_y1

    if line_w <= 0 or line_h <= 0:
        return None, []

    # Create canvas
    canvas = Image.new("L", (line_w, line_h), bg_val)

    # Place each word image at its position relative to line origin
    word_bboxes_local = []
    for w, img in zip(words, word_imgs):
        x1, y1, x2, y2 = w["bbox_abs"]
        bw = x2 - x1
        bh = y2 - y1

        # Resize word image to match its annotated bbox size
        if bw > 0 and bh > 0:
            img_resized = img.resize((bw, bh), Image.LANCZOS)
        else:
            img_resized = img

        # Position in local (line) coords
        lx = x1 - all_x1
        ly = y1 - all_y1
        canvas.paste(img_resized, (lx, ly))

        word_bboxes_local.append((lx, ly, lx + bw, ly + bh))

    return canvas, word_bboxes_local


# ============================================================
# Patch generation
# ============================================================

def scale_and_pad(
    img: Image.Image,
    retina_size: int = RETINA_SIZE,
    pad_val: int = 255,
) -> Tuple[Image.Image, float, int, int]:
    """Scale image to fit retina, centre on canvas."""
    w, h = img.size
    if w == 0 or h == 0:
        canvas = Image.new("L", (retina_size, retina_size), pad_val)
        return canvas, 1.0, 0, 0
    scale = min(retina_size / w, retina_size / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (retina_size, retina_size), pad_val)
    ox = (retina_size - nw) // 2
    oy = (retina_size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def generate_infill_samples(
    form: dict,
    words_dir: str,
    out_dir: str,
    style_to_index: Dict[str, int],
    retina_size: int = RETINA_SIZE,
) -> List[dict]:
    """Generate infill samples for all words in all lines of a form."""
    writer_id = form["writer_id"]
    style_id = f"iam_writer_{writer_id}"
    style_idx = style_to_index.get(style_id, 0)
    form_id = form["form_id"]

    records = []

    for line in form["lines"]:
        line_id = line["line_id"]
        words = line["words"]

        if len(words) < 2:
            # Need at least 2 words for meaningful context
            continue

        # Composite line image
        line_img, word_bboxes = composite_line(words, words_dir)
        if line_img is None:
            continue

        line_w, line_h = line_img.size

        # For each word, create before/after/mask
        for wi, (w, bbox) in enumerate(zip(words, word_bboxes)):
            word_id = w["word_id"]
            word_text = w["word_text"]
            x1, y1, x2, y2 = bbox

            # "after" = full line (scaled to retina)
            after_img, scale, ox, oy = scale_and_pad(line_img, retina_size)

            # "before" = line with this word erased
            before_canvas = line_img.copy()
            # Erase word region (fill with white)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(before_canvas)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            before_img, _, _, _ = scale_and_pad(before_canvas, retina_size)

            # "mask" = bbox region in retina coords
            mask_canvas = Image.new("L", (retina_size, retina_size), 0)
            mask_draw = ImageDraw.Draw(mask_canvas)
            # Transform bbox to retina coords
            rx1 = int(x1 * scale + ox)
            ry1 = int(y1 * scale + oy)
            rx2 = int(x2 * scale + ox)
            ry2 = int(y2 * scale + oy)
            mask_draw.rectangle([rx1, ry1, rx2, ry2], fill=255)

            # Normalized bbox (cx, cy, w, h) in [0, 1]
            cx_norm = (rx1 + rx2) / 2.0 / retina_size
            cy_norm = (ry1 + ry2) / 2.0 / retina_size
            w_norm = (rx2 - rx1) / retina_size
            h_norm = (ry2 - ry1) / retina_size

            # Save images
            prefix = f"iam_{line_id}_{wi}"
            before_path = f"{prefix}_before.png"
            after_path = f"{prefix}_after.png"
            mask_path = f"{prefix}_mask.png"

            before_img.save(os.path.join(out_dir, before_path))
            after_img.save(os.path.join(out_dir, after_path))
            mask_canvas.save(os.path.join(out_dir, mask_path))

            records.append({
                "schema_version": "2.0",
                "record_id": prefix,
                "document_id": form_id,
                "level": "word",
                "word_id": word_id,
                "retina_size": retina_size,
                "before_patch_ref": f"infill/{before_path}",
                "after_patch_ref": f"infill/{after_path}",
                "bbox_mask_ref": f"infill/{mask_path}",
                "char_text": word_text,
                "style_id": style_id,
                "target_bbox_parent_norm_cxcywh": [
                    round(cx_norm, 6),
                    round(cy_norm, 6),
                    round(w_norm, 6),
                    round(h_norm, 6),
                ],
                "child_bboxes": [],
                "confidence": 1.0,
            })

    return records


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Generate IAM word-level infill data")
    p.add_argument("--xml-dir", default="data/iam_ondb")
    p.add_argument("--words-dir", default="data/iam_words/iam_words/words")
    p.add_argument("--style-json", default="data/iam_full/style_to_index.json")
    p.add_argument("--writer-meta", default="data/iam_full/writer_meta.json")
    p.add_argument("--out-dir", default="data/iam_full/infill")
    p.add_argument("--retina-size", type=int, default=256)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load style mapping
    with open(args.style_json) as f:
        style_to_index = json.load(f)
    print(f"Loaded {len(style_to_index)} styles")

    # Load writer metadata for train/val split
    with open(args.writer_meta) as f:
        writer_meta = json.load(f)

    # Parse all XMLs
    xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} form XMLs")

    train_records = []
    val_records = []
    skipped_forms = 0

    for xi, xml_path in enumerate(xml_files):
        form = parse_form_xml(str(xml_path))
        if not form["lines"]:
            skipped_forms += 1
            continue

        writer_key = f"iam_writer_{form['writer_id']}"
        split = writer_meta.get(writer_key, {}).get("split", "train")

        records = generate_infill_samples(
            form, args.words_dir, str(out_dir),
            style_to_index, args.retina_size,
        )

        if split == "val":
            val_records.extend(records)
        else:
            train_records.extend(records)

        if (xi + 1) % 100 == 0:
            print(f"  Processed {xi + 1}/{len(xml_files)} forms, "
                  f"train={len(train_records)} val={len(val_records)}")

    # Write JSONL files
    train_path = out_dir.parent / "infill_train.jsonl"
    val_path = out_dir.parent / "infill_val.jsonl"

    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone!")
    print(f"  Forms: {len(xml_files)} ({skipped_forms} skipped)")
    print(f"  Train: {len(train_records)} records → {train_path}")
    print(f"  Val:   {len(val_records)} records → {val_path}")

    # Summary
    writers = set()
    for r in train_records + val_records:
        writers.add(r["style_id"])

    summary = {
        "num_forms": len(xml_files) - skipped_forms,
        "num_writers": len(writers),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "retina_size": args.retina_size,
        "level": "word",
    }
    with open(out_dir.parent / "infill_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {out_dir.parent / 'infill_summary.json'}")


if __name__ == "__main__":
    main()
