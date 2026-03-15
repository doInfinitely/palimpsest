#!/usr/bin/env python3
"""
Ingest IAM datasets into palimpsest training format.

Reads:
  - IAM-OnDB form XMLs (hierarchy, writer IDs, transcriptions, word bboxes)
  - IAM word images (offline word crops)
  - IAM-OnDB lineStrokes XMLs (true online pen data, if available)

Produces:
  - Word route JSONL (pseudo-online from skeletonized word images)
  - Line bridge JSONL
  - Writer/style metadata

Usage:
    python ingest_iam.py \
        --xml-dir data/iam_ondb \
        --words-dir data/iam_words/iam_words/words \
        --words-txt data/iam_words/words_new.txt \
        --out-dir data/iam_processed \
        --max-forms 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stroke_utils import (
    PrimitiveStroke,
    RouteWeights,
    extract_strokes_from_binary,
    route_word_strokes,
    route_to_events,
)


# ============================================================
# Parse IAM word labels
# ============================================================

def parse_words_txt(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Parse IAM words.txt into {word_id: {text, status, bbox, ...}}."""
    words = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            word_id = parts[0]
            status = parts[1]
            gray = int(parts[2])
            x, y, w, h = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            tag = parts[7]
            text = parts[8] if len(parts) == 9 else " ".join(parts[8:])
            words[word_id] = {
                "word_id": word_id,
                "text": text,
                "status": status,
                "gray_level": gray,
                "bbox": (x, y, w, h),
                "tag": tag,
            }
    return words


# ============================================================
# Parse IAM-OnDB form XMLs
# ============================================================

def parse_form_xml(path: str | Path) -> Dict[str, Any]:
    """Parse a single IAM-OnDB form XML."""
    tree = ET.parse(path)
    root = tree.getroot()

    form_id = root.attrib.get("id", "")
    writer_id = root.attrib.get("writer-id", "unknown")
    width = int(root.attrib.get("width", 0))
    height = int(root.attrib.get("height", 0))

    lines = []
    hw_part = root.find("handwritten-part")
    if hw_part is None:
        return {"form_id": form_id, "writer_id": writer_id, "lines": [], "width": width, "height": height}

    for line_elem in hw_part.findall("line"):
        line_id = line_elem.attrib.get("id", "")
        line_text = line_elem.attrib.get("text", "")
        seg_status = line_elem.attrib.get("segmentation", "")

        words = []
        for word_elem in line_elem.findall("word"):
            word_id = word_elem.attrib.get("id", "")
            word_text = word_elem.attrib.get("text", "")
            tag = word_elem.attrib.get("tag", "")

            # Component bboxes
            cmps = []
            for cmp in word_elem.findall("cmp"):
                cmps.append({
                    "x": int(cmp.attrib["x"]),
                    "y": int(cmp.attrib["y"]),
                    "w": int(cmp.attrib["width"]),
                    "h": int(cmp.attrib["height"]),
                })

            words.append({
                "word_id": word_id,
                "text": word_text,
                "tag": tag,
                "components": cmps,
            })

        lines.append({
            "line_id": line_id,
            "text": line_text,
            "segmentation": seg_status,
            "words": words,
        })

    return {
        "form_id": form_id,
        "writer_id": writer_id,
        "width": width,
        "height": height,
        "lines": lines,
    }


# ============================================================
# Parse IAM-OnDB lineStrokes XML (true online data)
# ============================================================

def parse_linestroke_xml(path: str | Path) -> List[List[Tuple[int, int, float]]]:
    """Parse a lineStrokes XML into list of strokes, each a list of (x, y, time) tuples."""
    strokes = []
    current_stroke = []

    with open(path, "r") as f:
        for line in f:
            if "<Point" in line:
                # Extract x, y, time
                x_start = line.index('x="') + 3
                x_end = line.index('"', x_start)
                y_start = line.index('y="') + 3
                y_end = line.index('"', y_start)
                t_start = line.index('time="') + 6
                t_end = line.index('"', t_start)
                x = int(line[x_start:x_end])
                y = int(line[y_start:y_end])
                t = float(line[t_start:t_end])
                current_stroke.append((x, y, t))
            elif "</Stroke>" in line:
                if current_stroke:
                    strokes.append(current_stroke)
                    current_stroke = []

    return strokes


def linestroke_to_events(
    strokes: List[List[Tuple[int, int, float]]],
    normalize_size: float = 1.0,
) -> List[Dict[str, Any]]:
    """Convert parsed lineStrokes into our unified event format."""
    if not strokes:
        return []

    # Find bounding box for normalization
    all_pts = [p for s in strokes for p in s]
    if not all_pts:
        return []
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1)
    scale = normalize_size / span

    events = []
    prev_x, prev_y = 0.0, 0.0

    for si, stroke in enumerate(strokes):
        is_last_stroke = si == len(strokes) - 1

        for pi, (px, py, _t) in enumerate(stroke):
            nx = (px - min_x) * scale
            ny = (py - min_y) * scale
            dx = round(nx - prev_x, 6)
            dy = round(ny - prev_y, 6)

            is_first = pi == 0
            is_last = pi == len(stroke) - 1

            if si == 0 and pi == 0:
                events.append({
                    "dx": 0.0, "dy": 0.0,
                    "pen_down": 1, "stroke_end": 0,
                    "char_end": 0, "word_end": 0, "seq_end": 0,
                })
            elif is_first:
                # Pen-up move to stroke start
                events.append({
                    "dx": dx, "dy": dy,
                    "pen_down": 0, "stroke_end": 0,
                    "char_end": 0, "word_end": 0, "seq_end": 0,
                })
            else:
                events.append({
                    "dx": dx, "dy": dy,
                    "pen_down": 1,
                    "stroke_end": int(is_last),
                    "char_end": 0,
                    "word_end": int(is_last and is_last_stroke),
                    "seq_end": int(is_last and is_last_stroke),
                })

            prev_x, prev_y = nx, ny

    return events


# ============================================================
# Pseudo-online from word images
# ============================================================

def word_image_to_route(
    img_path: str | Path,
    word_id: str,
    word_text: str,
    gray_level: int,
    weights: RouteWeights,
) -> Optional[Dict[str, Any]]:
    """Convert an IAM word image to a pseudo-online word route record."""
    try:
        img = Image.open(img_path).convert("L")
    except Exception:
        return None

    arr = np.array(img)
    if arr.size == 0:
        return None

    # Binarize using the provided gray level
    binary = arr < gray_level

    if not binary.any():
        # Fallback: Otsu-like threshold
        binary = arr < np.median(arr) - 20
        if not binary.any():
            return None

    h, w = binary.shape
    word_size = (w, h)
    char_bbox = (0, 0, w, h)

    strokes = extract_strokes_from_binary(
        binary, word_id, 0, char_bbox, word_size, min_stroke_len=3,
    )

    if not strokes:
        return None

    route = route_word_strokes(strokes, weights)
    events = route_to_events(route, strokes)
    if not events:
        return None

    # Build record
    prim_strokes_data = []
    for s in strokes:
        prim_strokes_data.append({
            "stroke_id": f"{word_id}_s{s.stroke_id}",
            "char_id": s.char_id,
            "char_index": s.char_index,
            "points_fwd": [list(p) for p in s.points_fwd],
            "arc_len": round(s.arc_len, 6),
        })

    selected_seq = []
    for idx, d in route.ordered:
        selected_seq.append({
            "stroke_id": f"{word_id}_s{strokes[idx].stroke_id}",
            "orientation": "fwd" if d == 0 else "rev",
        })

    return {
        "schema_version": "1.1",
        "record_id": f"iam_route_{word_id}",
        "document_id": f"iam_{word_id.rsplit('-', 2)[0]}",
        "line_id": f"iam_{'-'.join(word_id.split('-')[:-1])}",
        "word_id": word_id,
        "word_index": int(word_id.split("-")[-1]),
        "word_text": word_text,
        "target_bbox_parent_norm_cxcywh": [0.5, 0.5, 1.0, 1.0],
        "route_start_point": list(route.start_point),
        "route_end_point": list(route.end_point),
        "primitive_strokes": prim_strokes_data,
        "selected_sequence": selected_seq,
        "events": events,
        "plan_cost": round(route.total_cost, 6),
        "source_type": "pseudo_online_iam",
        "style_id": None,  # filled in later from form XML
        "source_weight": 1.0,
        "confidence": 0.9,
    }


# ============================================================
# Main pipeline
# ============================================================

def find_word_image(words_dir: Path, word_id: str) -> Optional[Path]:
    """Find the image file for a word ID like a01-000u-00-00."""
    parts = word_id.split("-")
    if len(parts) < 4:
        return None
    writer_dir = parts[0]
    form_dir = f"{parts[0]}-{parts[1]}"
    filename = f"{word_id}.png"
    path = words_dir / writer_dir / form_dir / filename
    if path.exists():
        return path
    return None


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest IAM data into palimpsest format")
    parser.add_argument("--xml-dir", type=str, default="data/iam_ondb",
                        help="Directory with IAM-OnDB form XMLs")
    parser.add_argument("--words-dir", type=str, default="data/iam_words/iam_words/words",
                        help="Directory with IAM word images")
    parser.add_argument("--words-txt", type=str, default="data/iam_words/words_new.txt",
                        help="IAM words.txt label file")
    parser.add_argument("--linestrokes-dir", type=str, default=None,
                        help="Directory with lineStrokes XMLs (optional, for true online)")
    parser.add_argument("--out-dir", type=str, default="data/iam_processed")
    parser.add_argument("--max-forms", type=int, default=0,
                        help="Max forms to process (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_dir = Path(args.xml_dir)
    words_dir = Path(args.words_dir)

    # Parse word labels
    print("Parsing word labels...")
    word_labels = parse_words_txt(args.words_txt)
    print(f"  {len(word_labels)} word labels")

    # Parse form XMLs for writer IDs and hierarchy
    print("Parsing form XMLs...")
    xml_files = sorted(xml_dir.glob("*.xml"))
    if args.max_forms > 0:
        xml_files = xml_files[:args.max_forms]

    forms = []
    writer_to_words = {}
    word_to_writer = {}

    for xf in xml_files:
        form = parse_form_xml(xf)
        forms.append(form)
        wid = form["writer_id"]
        for line in form["lines"]:
            for word in line["words"]:
                word_to_writer[word["word_id"]] = wid
                writer_to_words.setdefault(wid, []).append(word["word_id"])

    print(f"  {len(forms)} forms, {len(word_to_writer)} writer-word mappings")
    print(f"  {len(writer_to_words)} unique writers")

    # Process word images → pseudo-online routes
    print("Processing word images → pseudo-online routes...")
    weights = RouteWeights()
    all_routes = []
    all_bridges = []
    skipped = 0
    processed = 0

    for fi, form in enumerate(forms):
        if (fi + 1) % 100 == 0 or fi == 0:
            print(f"  Form {fi+1}/{len(forms)} ({form['form_id']})...")

        writer_id = f"iam_writer_{form['writer_id']}"

        for line in form["lines"]:
            if line["segmentation"] != "ok":
                continue

            line_routes = []
            for word in line["words"]:
                wid = word["word_id"]
                wlabel = word_labels.get(wid)
                if wlabel is None or wlabel["status"] != "ok":
                    skipped += 1
                    continue

                img_path = find_word_image(words_dir, wid)
                if img_path is None:
                    skipped += 1
                    continue

                route = word_image_to_route(
                    img_path, wid, wlabel["text"],
                    wlabel["gray_level"], weights,
                )
                if route is None:
                    skipped += 1
                    continue

                route["style_id"] = writer_id
                all_routes.append(route)
                line_routes.append(route)
                processed += 1

            # Build bridges between consecutive words in line
            line_id = f"iam_{line['line_id']}"
            for i in range(len(line_routes) - 1):
                prev_end = line_routes[i]["route_end_point"]
                next_start = line_routes[i + 1]["route_start_point"]
                dx = next_start[0] - prev_end[0]
                dy = next_start[1] - prev_end[1]
                bridge = {
                    "record_id": f"bridge_{line_routes[i]['word_id']}_{line_routes[i+1]['word_id']}",
                    "document_id": line_routes[i]["document_id"],
                    "line_id": line_id,
                    "prev_word_id": line_routes[i]["word_id"],
                    "next_word_id": line_routes[i + 1]["word_id"],
                    "prev_word_text": line_routes[i]["word_text"],
                    "next_word_text": line_routes[i + 1]["word_text"],
                    "prev_word_end_point": prev_end,
                    "next_word_start_point": next_start,
                    "bridge_events": [{"dx": round(dx, 6), "dy": round(dy, 6),
                                       "pen_down": 0, "stroke_end": 0,
                                       "char_end": 0, "word_end": 0, "seq_end": 0}],
                    "bridge_cost": round(math.hypot(dx, dy), 6),
                    "confidence": 0.9,
                }
                all_bridges.append(bridge)

    print(f"  Processed: {processed}, Skipped: {skipped}")

    # Writer-disjoint train/val split
    writer_ids = sorted(set(r["style_id"] for r in all_routes))
    n_val = max(1, len(writer_ids) // 10)
    val_writers = set(writer_ids[-n_val:])
    train_writers = set(writer_ids) - val_writers

    train_routes = [r for r in all_routes if r["style_id"] in train_writers]
    val_routes = [r for r in all_routes if r["style_id"] in val_writers]
    train_word_ids = {r["word_id"] for r in train_routes}
    val_word_ids = {r["word_id"] for r in val_routes}
    train_bridges = [b for b in all_bridges if b["prev_word_id"] in train_word_ids]
    val_bridges = [b for b in all_bridges if b["prev_word_id"] in val_word_ids]

    # Write outputs
    write_jsonl(out_dir / "train_word_routes.jsonl", train_routes)
    write_jsonl(out_dir / "val_word_routes.jsonl", val_routes)
    write_jsonl(out_dir / "train_line_bridges.jsonl", train_bridges)
    write_jsonl(out_dir / "val_line_bridges.jsonl", val_bridges)

    # Style vocab
    style_to_index = {"__unk__": 0}
    for i, wid in enumerate(sorted(writer_ids)):
        style_to_index[wid] = i + 1
    with (out_dir / "style_to_index.json").open("w") as f:
        json.dump(style_to_index, f, indent=2)

    # Writer metadata
    writer_meta = {}
    for wid in writer_ids:
        wid_words = [r for r in all_routes if r["style_id"] == wid]
        writer_meta[wid] = {
            "num_words": len(wid_words),
            "split": "val" if wid in val_writers else "train",
        }
    with (out_dir / "writer_meta.json").open("w") as f:
        json.dump(writer_meta, f, indent=2)

    summary = {
        "num_forms": len(forms),
        "num_writers": len(writer_ids),
        "train_writers": len(train_writers),
        "val_writers": len(val_writers),
        "train_routes": len(train_routes),
        "val_routes": len(val_routes),
        "train_bridges": len(train_bridges),
        "val_bridges": len(val_bridges),
        "total_events": sum(len(r["events"]) for r in all_routes),
        "processed": processed,
        "skipped": skipped,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Output in {out_dir}/")
    print(f"  {len(writer_ids)} writers ({len(train_writers)} train, {len(val_writers)} val)")
    print(f"  Train: {len(train_routes)} routes, {len(train_bridges)} bridges")
    print(f"  Val:   {len(val_routes)} routes, {len(val_bridges)} bridges")
    print(f"  Total events: {summary['total_events']}")


if __name__ == "__main__":
    main()
