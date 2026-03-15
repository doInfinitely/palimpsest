#!/usr/bin/env python3
"""
Synthetic data generator for the palimpsest handwriting pipeline.

Generates training artifacts from synthetic pages rendered with fonts:
  1. Tree annotation JSONs (one per document)
  2. Level trajectory JSONLs (teacher-forced placement actions per hierarchy level)
  3. Character infill pairs (before/after patches for the character renderer)
  4. Pseudo-online stroke sequences (skeletonized pen events)

Outputs data in the canonical formats from the palimpsest spec, consumable
by tiny-tessarachnid training scripts.

Usage:
    python generate_synth_data.py --num-pages 100 --out-dir data/synth_v1
    python generate_synth_data.py --num-pages 500 --out-dir data/synth_v1 --skip-strokes
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add tiny-tessarachnid to path so we can import its utilities
_TESSARACHNID_DIR = str(Path(__file__).resolve().parent.parent / "tiny-tessarachnid")
if _TESSARACHNID_DIR not in sys.path:
    sys.path.insert(0, _TESSARACHNID_DIR)

from generate_training_data import (
    SyntheticPage,
    discover_fonts,
    scale_and_pad,
    bbox_to_retina,
    char_to_class,
    RETINA_SIZE,
    CLASS_NONE,
    CLASS_PAGE,
    CLASS_PARAGRAPH,
    CLASS_LINE,
    CLASS_WORD,
    CHAR_CLASS_OFFSET,
    PRINTABLE_CHARS,
    CHAR_TO_CLASS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_id() -> str:
    return uuid.uuid4().hex[:12]


def _bbox_to_parent_norm_cxcywh(
    bbox_abs: Tuple[int, int, int, int],
    parent_abs: Tuple[int, int, int, int],
) -> List[float]:
    """Convert absolute bbox to parent-normalized [cx, cy, w, h]."""
    px1, py1, px2, py2 = parent_abs
    pw = max(px2 - px1, 1)
    ph = max(py2 - py1, 1)
    x1, y1, x2, y2 = bbox_abs
    cx = ((x1 + x2) / 2.0 - px1) / pw
    cy = ((y1 + y2) / 2.0 - py1) / ph
    w = (x2 - x1) / pw
    h = (y2 - y1) / ph
    return [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_gray_patch(img: Image.Image, path: Path) -> None:
    """Save a grayscale version of an RGB image patch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("L").save(str(path))


# ---------------------------------------------------------------------------
# Tree annotation builder
# ---------------------------------------------------------------------------

def build_tree_annotation(
    page: SyntheticPage,
    doc_id: str,
    page_index: int = 0,
) -> Dict[str, Any]:
    """Build the canonical tree annotation JSON for one synthetic page.

    Hierarchy: document -> page -> paragraph -> line -> word -> character
    """
    nodes: List[Dict[str, Any]] = []

    # Collect full text
    full_text_parts = []
    for para in page.paragraphs:
        para_text_parts = []
        for line in para["lines"]:
            line_text = ""
            for word in line["words"]:
                word_text = "".join(c["char"] for c in word["characters"])
                line_text += (" " if line_text else "") + word_text
            para_text_parts.append(line_text)
        full_text_parts.append("\n".join(para_text_parts))
    full_text = "\n\n".join(full_text_parts)

    doc_node_id = f"doc_{doc_id}"
    page_node_id = f"page_{doc_id}_{page_index}"

    page_bbox = page.page_bbox
    if page_bbox[2] <= page_bbox[0] or page_bbox[3] <= page_bbox[1]:
        page_bbox = (0, 0, page.width, page.height)

    doc_bbox = (0, 0, page.width, page.height)

    # Document node
    nodes.append({
        "id": doc_node_id,
        "level": "document",
        "parent_id": None,
        "child_ids": [page_node_id],
        "order_index": 0,
        "bbox_abs_xyxy": list(doc_bbox),
        "bbox_parent_norm_cxcywh": [0.5, 0.5, 1.0, 1.0],
        "rotation_deg": 0.0,
        "text": full_text,
        "reading_order_key": [0],
        "label_source": "synthetic",
        "confidence": 1.0,
    })

    # Page node
    para_ids = []
    for pi in range(len(page.paragraphs)):
        para_ids.append(f"para_{doc_id}_{pi}")

    nodes.append({
        "id": page_node_id,
        "level": "page",
        "parent_id": doc_node_id,
        "child_ids": para_ids,
        "order_index": 0,
        "bbox_abs_xyxy": list(page_bbox),
        "bbox_parent_norm_cxcywh": _bbox_to_parent_norm_cxcywh(page_bbox, doc_bbox),
        "rotation_deg": 0.0,
        "text": full_text,
        "reading_order_key": [0, 0],
        "label_source": "synthetic",
        "confidence": 1.0,
    })

    # Paragraph, line, word, character nodes
    for pi, para in enumerate(page.paragraphs):
        para_id = f"para_{doc_id}_{pi}"
        para_bbox_abs = para["bbox"]
        is_hw = para.get("is_handwritten", False)

        line_ids = []
        para_text_parts = []
        for li, line in enumerate(para["lines"]):
            line_id = f"line_{doc_id}_{pi}_{li}"
            line_ids.append(line_id)

            word_ids = []
            line_text_parts = []
            for wi, word in enumerate(line["words"]):
                word_id = f"word_{doc_id}_{pi}_{li}_{wi}"
                word_ids.append(word_id)

                char_ids = []
                word_text = ""
                for ci, ch_data in enumerate(word["characters"]):
                    char_id = f"char_{doc_id}_{pi}_{li}_{wi}_{ci}"
                    char_ids.append(char_id)
                    word_text += ch_data["char"]

                    nodes.append({
                        "id": char_id,
                        "level": "character",
                        "parent_id": word_id,
                        "child_ids": [],
                        "order_index": ci,
                        "bbox_abs_xyxy": list(ch_data["bbox"]),
                        "bbox_parent_norm_cxcywh": _bbox_to_parent_norm_cxcywh(
                            ch_data["bbox"], word["bbox"]),
                        "rotation_deg": 0.0,
                        "text": ch_data["char"],
                        "reading_order_key": [0, 0, pi, li, wi, ci],
                        "label_source": "synthetic",
                        "confidence": 1.0,
                        "is_handwritten": is_hw,
                    })

                line_text_parts.append(word_text)
                nodes.append({
                    "id": word_id,
                    "level": "word",
                    "parent_id": line_id,
                    "child_ids": char_ids,
                    "order_index": wi,
                    "bbox_abs_xyxy": list(word["bbox"]),
                    "bbox_parent_norm_cxcywh": _bbox_to_parent_norm_cxcywh(
                        word["bbox"], line["bbox"]),
                    "rotation_deg": 0.0,
                    "text": word_text,
                    "reading_order_key": [0, 0, pi, li, wi],
                    "label_source": "synthetic",
                    "confidence": 1.0,
                    "is_handwritten": is_hw,
                })

            line_text = " ".join(line_text_parts)
            para_text_parts.append(line_text)
            nodes.append({
                "id": line_id,
                "level": "line",
                "parent_id": para_id,
                "child_ids": word_ids,
                "order_index": li,
                "bbox_abs_xyxy": list(line["bbox"]),
                "bbox_parent_norm_cxcywh": _bbox_to_parent_norm_cxcywh(
                    line["bbox"], para_bbox_abs),
                "rotation_deg": 0.0,
                "text": line_text,
                "reading_order_key": [0, 0, pi, li],
                "label_source": "synthetic",
                "confidence": 1.0,
                "is_handwritten": is_hw,
            })

        para_text = "\n".join(para_text_parts)
        nodes.append({
            "id": para_id,
            "level": "paragraph",
            "parent_id": page_node_id,
            "child_ids": line_ids,
            "order_index": pi,
            "bbox_abs_xyxy": list(para_bbox_abs),
            "bbox_parent_norm_cxcywh": _bbox_to_parent_norm_cxcywh(
                para_bbox_abs, page_bbox),
            "rotation_deg": 0.0,
            "text": para_text,
            "reading_order_key": [0, 0, pi],
            "label_source": "synthetic",
            "confidence": 1.0,
            "is_handwritten": is_hw,
        })

    return {
        "schema_version": "1.0",
        "document_id": doc_id,
        "dataset": "synthetic",
        "language": "en",
        "writer_id": None,
        "style_id": f"style_synth_{doc_id}",
        "source_image": {
            "width_px": page.width,
            "height_px": page.height,
            "dpi": 300,
            "deskew_angle_deg": 0.0,
        },
        "text": {"full_text": full_text},
        "root_id": doc_node_id,
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------

def _retina_size_for_level(level: str) -> int:
    return {
        "document": 1024,
        "page": 1024,
        "paragraph": 512,
        "line": 512,
        "word": 256,
        "character": 256,
    }.get(level, 256)


def _level_to_child_level(level: str) -> str:
    return {
        "document": "page",
        "page": "paragraph",
        "paragraph": "line",
        "line": "word",
        "word": "character",
    }[level]


def build_trajectories(
    tree: Dict[str, Any],
    page: SyntheticPage,
    out_dir: Path,
    save_states: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build teacher-forced trajectory records for all 6 hierarchy levels.

    Returns dict mapping level name -> list of trajectory action records.
    """
    doc_id = tree["document_id"]
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}

    trajectories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # For each non-leaf node, produce a trajectory of its children
    for node in tree["nodes"]:
        if node["level"] == "character":
            continue  # leaf

        parent_id = node["id"]
        child_ids = node["child_ids"]
        child_level = _level_to_child_level(node["level"])
        retina_size = _retina_size_for_level(child_level)
        parent_bbox = tuple(node["bbox_abs_xyxy"])

        traj_id = f"traj_{parent_id}"

        # Produce one action per child + final stop action
        for step_idx in range(len(child_ids) + 1):
            is_stop = step_idx >= len(child_ids)

            if is_stop:
                target_node_id = None
                semantic_text = None
                target_bbox = None
                previous_siblings = child_ids
            else:
                child_node = nodes_by_id[child_ids[step_idx]]
                target_node_id = child_node["id"]
                semantic_text = child_node.get("text")
                target_bbox = child_node["bbox_parent_norm_cxcywh"]
                previous_siblings = child_ids[:step_idx]

            # Build state reference (we save actual images below)
            state_ref = f"states/{parent_id}/step_{step_idx:03d}.png"

            if save_states:
                _save_state_image(
                    page, parent_bbox, retina_size,
                    out_dir / state_ref,
                )

            rec = {
                "schema_version": "1.0",
                "trajectory_id": traj_id,
                "document_id": doc_id,
                "level": child_level,
                "step_index": step_idx,
                "parent_id": parent_id,
                "target_node_id": target_node_id,
                "state_before_ref": state_ref,
                "parent_retina_size": retina_size,
                "parent_bbox_abs_xyxy": list(parent_bbox),
                "previous_sibling_ids": previous_siblings,
                "semantic_condition": {
                    "type": child_level,
                    "text": semantic_text,
                    "tokens": list(semantic_text.encode("utf-8"))[:256] if semantic_text else [],
                },
                "style_id": tree.get("style_id"),
                "target": {
                    "stop": is_stop,
                    "bbox_parent_norm_cxcywh": target_bbox,
                    "rotation_deg": 0.0 if not is_stop else None,
                },
                "confidence": 1.0,
            }
            trajectories[child_level].append(rec)

    return dict(trajectories)


def _save_state_image(
    page: SyntheticPage,
    parent_bbox: Tuple[int, int, int, int],
    retina_size: int,
    path: Path,
) -> None:
    """Crop the parent region from the page and save as grayscale retina."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = parent_bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(page.width, x2)
    y2 = min(page.height, y2)
    if x2 <= x1 or y2 <= y1:
        img = Image.new("L", (retina_size, retina_size), 255)
        img.save(str(path))
        return

    crop = page.image.crop((x1, y1, x2, y2))
    # Scale preserving aspect ratio, pad to square
    w, h = crop.size
    scale = min(retina_size / w, retina_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (retina_size, retina_size), page.bg_color)
    ox = (retina_size - new_w) // 2
    oy = (retina_size - new_h) // 2
    canvas.paste(resized, (ox, oy))
    canvas.convert("L").save(str(path))


# ---------------------------------------------------------------------------
# Character infill pair builder
# ---------------------------------------------------------------------------

def build_infill_pairs(
    tree: Dict[str, Any],
    page: SyntheticPage,
    out_dir: Path,
    retina_size: int = 256,
) -> List[Dict[str, Any]]:
    """Generate before/after infill pairs for each character.

    For synthetic data: 'before' is the word region with the character erased,
    'after' is the word region with the character present.
    """
    from generate_training_data import _compute_mask, erase_region

    doc_id = tree["document_id"]
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    records = []

    for node in tree["nodes"]:
        if node["level"] != "character":
            continue

        char_node = node
        word_node = nodes_by_id[char_node["parent_id"]]
        word_bbox = tuple(word_node["bbox_abs_xyxy"])
        char_bbox = tuple(char_node["bbox_abs_xyxy"])

        wx1, wy1, wx2, wy2 = word_bbox
        if wx2 <= wx1 or wy2 <= wy1:
            continue

        # "after" = word crop as-is
        after_crop = page.image.crop((wx1, wy1, wx2, wy2))

        # "before" = word crop with this character erased
        char_mask = _compute_mask(page.image, char_bbox, page.bg_color)
        before_page = erase_region(page.image, char_bbox, char_mask, page.bg_color)
        before_crop = before_page.crop((wx1, wy1, wx2, wy2))

        # bbox mask: binary mask showing where the character bbox is within
        # the word crop
        cx1 = max(0, char_bbox[0] - wx1)
        cy1 = max(0, char_bbox[1] - wy1)
        cx2 = min(wx2 - wx1, char_bbox[2] - wx1)
        cy2 = min(wy2 - wy1, char_bbox[3] - wy1)
        bbox_mask = Image.new("L", (wx2 - wx1, wy2 - wy1), 0)
        ImageDraw.Draw(bbox_mask).rectangle([cx1, cy1, cx2, cy2], fill=255)

        # Scale to retina
        def _to_retina(img):
            w, h = img.size
            s = min(retina_size / max(w, 1), retina_size / max(h, 1))
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            r = img.resize((nw, nh), Image.LANCZOS)
            c = Image.new(img.mode, (retina_size, retina_size),
                          0 if img.mode == "L" else page.bg_color)
            c.paste(r, ((retina_size - nw) // 2, (retina_size - nh) // 2))
            return c

        record_id = f"infill_{char_node['id']}"
        before_path = f"infill/{record_id}_before.png"
        after_path = f"infill/{record_id}_after.png"
        mask_path = f"infill/{record_id}_mask.png"

        for p in [before_path, after_path, mask_path]:
            (out_dir / p).parent.mkdir(parents=True, exist_ok=True)

        _to_retina(before_crop).convert("L").save(
            str(out_dir / before_path))
        _to_retina(after_crop).convert("L").save(
            str(out_dir / after_path))
        _to_retina(bbox_mask).save(str(out_dir / mask_path))

        records.append({
            "schema_version": "1.0",
            "record_id": record_id,
            "document_id": doc_id,
            "parent_level": "word",
            "parent_id": word_node["id"],
            "target_node_id": char_node["id"],
            "retina_size": retina_size,
            "before_patch_ref": before_path,
            "after_patch_ref": after_path,
            "bbox_mask_ref": mask_path,
            "neighbor_mask_ref": mask_path,
            "char_text": char_node["text"],
            "char_tokens": list(char_node["text"].encode("utf-8")),
            "style_id": tree.get("style_id"),
            "target_bbox_parent_norm_cxcywh": char_node["bbox_parent_norm_cxcywh"],
            "confidence": 1.0,
        })

    return records


# ---------------------------------------------------------------------------
# Pseudo-online stroke builder (skeleton-based)
# ---------------------------------------------------------------------------

def build_stroke_sequences(
    tree: Dict[str, Any],
    page: SyntheticPage,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Build pseudo-online stroke sequences from skeletonized character crops.

    For each word, collects character strokes, plans a route minimizing
    pen-up travel, and emits an event stream.
    """
    try:
        from skimage.morphology import skeletonize
        from skimage.graph import pixel_graph
    except ImportError:
        print("WARNING: scikit-image not available, skipping stroke generation")
        return []

    from generate_training_data import _compute_mask

    doc_id = tree["document_id"]
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    records = []

    for node in tree["nodes"]:
        if node["level"] != "word":
            continue

        word_node = node
        word_bbox = tuple(word_node["bbox_abs_xyxy"])
        wx1, wy1, wx2, wy2 = word_bbox
        if wx2 <= wx1 or wy2 <= wy1:
            continue

        word_text = word_node.get("text", "")
        char_nodes = [nodes_by_id[cid] for cid in word_node["child_ids"]]
        all_strokes = []

        for char_node in char_nodes:
            cb = tuple(char_node["bbox_abs_xyxy"])
            if cb[2] <= cb[0] or cb[3] <= cb[1]:
                continue

            # Extract and skeletonize character
            char_crop = page.image.crop(cb).convert("L")
            arr = np.array(char_crop)
            # Binarize: assume dark text on light background
            bg_val = np.median(arr)
            if bg_val > 128:
                binary = arr < (bg_val - 30)
            else:
                binary = arr > (bg_val + 30)

            if not binary.any():
                continue

            skel = skeletonize(binary)
            points = np.argwhere(skel)  # (row, col)
            if len(points) < 2:
                continue

            # Simple stroke extraction: trace connected skeleton pixels
            # by greedy nearest-neighbor walk
            strokes = _extract_strokes_greedy(points, cb, word_bbox)
            all_strokes.extend(strokes)

        if not all_strokes:
            continue

        # Route strokes to minimize pen-up travel (greedy nearest-neighbor)
        routed = _route_strokes_greedy(all_strokes)

        # Convert to event stream
        events = _strokes_to_events(routed, word_bbox)

        if not events:
            continue

        # Determine start and end points
        route_start = routed[0]["points_norm"][0] if routed else [0.0, 0.0]
        route_end = routed[-1]["points_norm"][-1] if routed else [0.0, 0.0]

        line_node = nodes_by_id.get(word_node["parent_id"])
        line_id = line_node["id"] if line_node else "unknown"

        record_id = f"word_route_{word_node['id']}"
        records.append({
            "schema_version": "1.1",
            "record_id": record_id,
            "document_id": doc_id,
            "line_id": line_id,
            "word_id": word_node["id"],
            "word_index": word_node["order_index"],
            "word_text": word_text,
            "target_bbox_parent_norm_cxcywh": word_node["bbox_parent_norm_cxcywh"],
            "route_start_point": list(route_start),
            "route_end_point": list(route_end),
            "events": events,
            "plan_cost": sum(s.get("arc_len", 0) for s in routed),
            "source_type": "pseudo_online",
            "source_weight": 0.7,
            "confidence": 0.8,
        })

    return records


def _extract_strokes_greedy(
    skel_points: np.ndarray,
    char_bbox: Tuple[int, int, int, int],
    word_bbox: Tuple[int, int, int, int],
) -> List[Dict[str, Any]]:
    """Extract stroke polylines from skeleton points via greedy walk.

    Returns strokes with points normalized to the word bbox [0,1].
    """
    if len(skel_points) == 0:
        return []

    cx1, cy1, cx2, cy2 = char_bbox
    wx1, wy1, wx2, wy2 = word_bbox
    ww = max(wx2 - wx1, 1)
    wh = max(wy2 - wy1, 1)

    # Build adjacency from skeleton using 8-connectivity
    visited = set()
    points_set = set(map(tuple, skel_points))
    strokes = []

    # Find endpoints (pixels with only 1 neighbor) as starting points
    def _neighbors(r, c):
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in points_set:
                    nbrs.append((nr, nc))
        return nbrs

    endpoints = []
    for r, c in skel_points:
        n = len(_neighbors(r, c))
        if n == 1:
            endpoints.append((r, c))

    # Start from endpoints, then any remaining unvisited points
    start_candidates = endpoints + [tuple(p) for p in skel_points]

    for start in start_candidates:
        if start in visited:
            continue

        stroke_pts = [start]
        visited.add(start)
        current = start

        while True:
            nbrs = [n for n in _neighbors(*current) if n not in visited]
            if not nbrs:
                break
            # Pick nearest unvisited neighbor
            current = min(nbrs, key=lambda n: (n[0] - current[0])**2 + (n[1] - current[1])**2)
            visited.add(current)
            stroke_pts.append(current)

        if len(stroke_pts) < 2:
            continue

        # Normalize points to word bbox [0,1]
        points_norm = []
        for r, c in stroke_pts:
            # r=row(y), c=col(x) in char crop space
            x_abs = cx1 + c
            y_abs = cy1 + r
            x_norm = (x_abs - wx1) / ww
            y_norm = (y_abs - wy1) / wh
            points_norm.append([round(x_norm, 6), round(y_norm, 6)])

        arc_len = 0.0
        for i in range(1, len(points_norm)):
            dx = points_norm[i][0] - points_norm[i-1][0]
            dy = points_norm[i][1] - points_norm[i-1][1]
            arc_len += math.hypot(dx, dy)

        strokes.append({
            "points_norm": points_norm,
            "arc_len": round(arc_len, 6),
        })

    return strokes


def _route_strokes_greedy(
    strokes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order strokes to minimize pen-up travel using greedy nearest-neighbor.

    Also considers reversing each stroke.
    """
    if not strokes:
        return []

    remaining = list(range(len(strokes)))
    # Start with the leftmost stroke start
    best_start = min(remaining, key=lambda i: strokes[i]["points_norm"][0][0])
    ordered = []

    current_end = strokes[best_start]["points_norm"][-1]
    ordered.append(strokes[best_start])
    remaining.remove(best_start)

    while remaining:
        best_idx = None
        best_dist = float("inf")
        best_reverse = False

        for idx in remaining:
            s = strokes[idx]
            fwd_start = s["points_norm"][0]
            rev_start = s["points_norm"][-1]

            d_fwd = math.hypot(fwd_start[0] - current_end[0],
                               fwd_start[1] - current_end[1])
            d_rev = math.hypot(rev_start[0] - current_end[0],
                               rev_start[1] - current_end[1])

            if d_fwd < best_dist:
                best_dist = d_fwd
                best_idx = idx
                best_reverse = False
            if d_rev < best_dist:
                best_dist = d_rev
                best_idx = idx
                best_reverse = True

        s = strokes[best_idx]
        if best_reverse:
            s = dict(s)
            s["points_norm"] = list(reversed(s["points_norm"]))
        ordered.append(s)
        current_end = s["points_norm"][-1]
        remaining.remove(best_idx)

    return ordered


def _strokes_to_events(
    strokes: List[Dict[str, Any]],
    word_bbox: Tuple[int, int, int, int],
) -> List[Dict[str, Any]]:
    """Convert ordered strokes to the unified pen event format."""
    events = []
    prev_x, prev_y = 0.0, 0.0

    for si, stroke in enumerate(strokes):
        pts = stroke["points_norm"]
        is_last_stroke = si == len(strokes) - 1

        for pi, (px, py) in enumerate(pts):
            is_first_point = pi == 0
            is_last_point = pi == len(pts) - 1

            if si == 0 and pi == 0:
                # First event: absolute start (dx=dy=0)
                events.append({
                    "dx": 0.0,
                    "dy": 0.0,
                    "pen_down": 1,
                    "stroke_end": 0,
                    "char_end": 0,
                    "word_end": 0,
                    "seq_end": 0,
                })
                prev_x, prev_y = px, py
                continue

            dx = round(px - prev_x, 6)
            dy = round(py - prev_y, 6)

            if is_first_point:
                # Pen-up move to stroke start
                events.append({
                    "dx": dx,
                    "dy": dy,
                    "pen_down": 0,
                    "stroke_end": 0,
                    "char_end": 0,
                    "word_end": 0,
                    "seq_end": 0,
                })
            else:
                events.append({
                    "dx": dx,
                    "dy": dy,
                    "pen_down": 1,
                    "stroke_end": int(is_last_point),
                    "char_end": 0,
                    "word_end": int(is_last_point and is_last_stroke),
                    "seq_end": int(is_last_point and is_last_stroke),
                })

            prev_x, prev_y = px, py

    return events


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_one_document(
    fonts: List[str],
    doc_index: int,
    out_dir: Path,
    save_states: bool = True,
    save_infill: bool = True,
    save_strokes: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, List], List, List]:
    """Generate one synthetic document with all training artifacts."""
    doc_id = f"synth_{doc_index:06d}"

    page = SyntheticPage(fonts, rotate_paragraphs=random.random() < 0.3)

    # 1. Tree annotation
    tree = build_tree_annotation(page, doc_id)

    # Save tree JSON
    tree_path = out_dir / "documents" / f"{doc_id}.json"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    # Save page image
    img_path = out_dir / "images" / f"{doc_id}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    page.image.save(str(img_path))

    # 2. Trajectories
    trajectories = build_trajectories(
        tree, page, out_dir, save_states=save_states)

    # 3. Infill pairs
    infill_records = []
    if save_infill:
        infill_records = build_infill_pairs(tree, page, out_dir)

    # 4. Stroke sequences
    stroke_records = []
    if save_strokes:
        stroke_records = build_stroke_sequences(tree, page, out_dir)

    return tree, trajectories, infill_records, stroke_records


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for palimpsest pipeline")
    parser.add_argument("--num-pages", type=int, default=100,
                        help="Number of synthetic documents to generate")
    parser.add_argument("--out-dir", type=str, default="data/synth_v1",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-states", action="store_true",
                        help="Skip saving state images (faster)")
    parser.add_argument("--skip-infill", action="store_true",
                        help="Skip generating infill pairs")
    parser.add_argument("--skip-strokes", action="store_true",
                        help="Skip generating stroke sequences")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering fonts...")
    fonts = discover_fonts()
    if not fonts:
        print("ERROR: No fonts found. Check tiny-tessarachnid/fonts/ directory.")
        sys.exit(1)
    print(f"Found {len(fonts)} fonts")

    # Accumulators for JSONL files
    all_trajectories: Dict[str, List] = defaultdict(list)
    all_infill: List[Dict] = []
    all_strokes: List[Dict] = []

    for i in range(args.num_pages):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Generating document {i+1}/{args.num_pages}...")

        try:
            tree, trajectories, infill_recs, stroke_recs = generate_one_document(
                fonts=fonts,
                doc_index=i,
                out_dir=out_dir,
                save_states=not args.skip_states,
                save_infill=not args.skip_infill,
                save_strokes=not args.skip_strokes,
            )

            for level, recs in trajectories.items():
                all_trajectories[level].extend(recs)
            all_infill.extend(infill_recs)
            all_strokes.extend(stroke_recs)

        except Exception as e:
            print(f"  WARNING: Failed on document {i}: {e}")
            continue

    # Write trajectory JSONL files
    traj_dir = out_dir / "trajectories"
    for level_name in ["page", "paragraph", "line", "word", "character"]:
        recs = all_trajectories.get(level_name, [])
        if recs:
            write_jsonl(traj_dir / f"{level_name}.jsonl", recs)
            print(f"  {level_name}: {len(recs)} actions")

    # Write infill JSONL
    if all_infill:
        write_jsonl(out_dir / "infill" / "character_infill.jsonl", all_infill)
        print(f"  infill: {len(all_infill)} pairs")

    # Write stroke JSONL
    if all_strokes:
        write_jsonl(out_dir / "strokes" / "word_routes.jsonl", all_strokes)
        print(f"  strokes: {len(all_strokes)} word routes")

    # Summary
    summary = {
        "num_documents": args.num_pages,
        "seed": args.seed,
        "trajectory_counts": {k: len(v) for k, v in all_trajectories.items()},
        "infill_count": len(all_infill),
        "stroke_count": len(all_strokes),
        "font_count": len(fonts),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Output written to {out_dir}/")
    print(f"  {args.num_pages} documents")
    print(f"  {sum(len(v) for v in all_trajectories.values())} total trajectory actions")
    print(f"  {len(all_infill)} infill pairs")
    print(f"  {len(all_strokes)} word stroke routes")


if __name__ == "__main__":
    main()
