#!/usr/bin/env python3
"""
Handwriting-focused synthetic data generator.

Generates training data specifically for the online handwriting scribe model:
  1. Word-level stroke routes from skeletonized rendered text
  2. Line-level bridge records between words
  3. Character infill pairs with handwriting-style fonts
  4. Full page annotations with writer/style IDs

Each "writer" is a (font, size, slant, color-tendency) tuple, producing
consistent style across a document.

Usage:
    python generate_handwriting_data.py --num-docs 200 --out-dir data/hw_v1
    python generate_handwriting_data.py --num-docs 50 --out-dir data/hw_v1 --writers 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_TESSARACHNID_DIR = str(Path(__file__).resolve().parent.parent / "tiny-tessarachnid")
if _TESSARACHNID_DIR not in sys.path:
    sys.path.insert(0, _TESSARACHNID_DIR)

from generate_training_data import (
    discover_fonts,
    _compute_mask,
    erase_region,
    _is_handwriting_font,
    _font_renders_latin,
    PRINTABLE_CHARS,
)
from stroke_utils import (
    PrimitiveStroke,
    RouteWeights,
    extract_strokes_from_binary,
    route_word_strokes,
    route_to_events,
    resample_polyline,
)


# ---------------------------------------------------------------------------
# Writer profiles
# ---------------------------------------------------------------------------

@dataclass
class WriterProfile:
    writer_id: str
    font_path: str
    font_size_range: Tuple[int, int]
    slant_deg: float  # rotation applied to each character
    spacing_factor: float  # inter-char spacing multiplier
    line_spacing_factor: float
    is_handwriting: bool
    bg_lum_range: Tuple[int, int]  # background luminance range
    ink_lum_range: Tuple[int, int]  # ink luminance range

    def sample_font_size(self) -> int:
        return random.randint(*self.font_size_range)

    def sample_bg_color(self) -> Tuple[int, int, int]:
        lum = random.randint(*self.bg_lum_range)
        # Slight color tint
        r = min(255, max(0, lum + random.randint(-15, 15)))
        g = min(255, max(0, lum + random.randint(-15, 15)))
        b = min(255, max(0, lum + random.randint(-10, 10)))
        return (r, g, b)

    def sample_ink_color(self, bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
        lum = random.randint(*self.ink_lum_range)
        r = min(255, max(0, lum + random.randint(-20, 20)))
        g = min(255, max(0, lum + random.randint(-20, 20)))
        b = min(255, max(0, lum + random.randint(-10, 10)))
        return (r, g, b)


def create_writer_profiles(fonts: List[str], num_writers: int = 50) -> List[WriterProfile]:
    """Create diverse writer profiles from available fonts."""
    profiles = []
    hw_fonts = [f for f in fonts if _is_handwriting_font(f)]
    all_fonts = fonts

    for i in range(num_writers):
        # Mix: 40% handwriting fonts, 60% any font
        if hw_fonts and random.random() < 0.4:
            font = random.choice(hw_fonts)
            is_hw = True
        else:
            font = random.choice(all_fonts)
            is_hw = _is_handwriting_font(font)

        size_base = random.randint(20, 48)
        size_var = random.randint(2, 6)

        profiles.append(WriterProfile(
            writer_id=f"writer_{i:04d}",
            font_path=font,
            font_size_range=(max(14, size_base - size_var), size_base + size_var),
            slant_deg=random.gauss(0, 3.0) if random.random() < 0.3 else 0.0,
            spacing_factor=random.uniform(0.8, 1.3),
            line_spacing_factor=random.uniform(1.0, 2.0),
            is_handwriting=is_hw,
            bg_lum_range=(200, 255) if random.random() < 0.7 else (30, 80),
            ink_lum_range=(0, 60) if random.random() < 0.7 else (180, 255),
        ))

    return profiles


# ---------------------------------------------------------------------------
# Text corpus (simple English word generator)
# ---------------------------------------------------------------------------

_COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "great", "between", "need", "large", "under",
    "never", "each", "much", "begin", "those", "being", "long", "make", "thing",
]


def random_sentence(min_words: int = 3, max_words: int = 12) -> str:
    n = random.randint(min_words, max_words)
    words = [random.choice(_COMMON_WORDS) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + random.choice([".", "!", "?", ",", ""])


def random_paragraph(min_lines: int = 2, max_lines: int = 6) -> List[str]:
    n = random.randint(min_lines, max_lines)
    return [random_sentence() for _ in range(n)]


# ---------------------------------------------------------------------------
# Handwriting page renderer
# ---------------------------------------------------------------------------

@dataclass
class RenderedChar:
    char: str
    bbox: Tuple[int, int, int, int]  # absolute on page
    char_index: int


@dataclass
class RenderedWord:
    text: str
    bbox: Tuple[int, int, int, int]
    characters: List[RenderedChar]
    word_index: int


@dataclass
class RenderedLine:
    text: str
    bbox: Tuple[int, int, int, int]
    words: List[RenderedWord]
    line_index: int


@dataclass
class RenderedParagraph:
    text: str
    bbox: Tuple[int, int, int, int]
    lines: List[RenderedLine]
    para_index: int
    is_handwritten: bool


@dataclass
class RenderedPage:
    image: Image.Image
    bg_color: Tuple[int, int, int]
    width: int
    height: int
    paragraphs: List[RenderedParagraph]
    writer: WriterProfile
    page_bbox: Tuple[int, int, int, int]


def render_handwriting_page(
    writer: WriterProfile,
    page_width: int = 2048,
    page_height: int = 2800,
    num_paragraphs: int = None,
) -> Optional[RenderedPage]:
    """Render a synthetic handwriting page with a specific writer profile."""
    if num_paragraphs is None:
        num_paragraphs = random.randint(1, 5)

    bg_color = writer.sample_bg_color()
    ink_color = writer.sample_ink_color(bg_color)

    image = Image.new("RGB", (page_width, page_height), bg_color)
    draw = ImageDraw.Draw(image)

    font_size = writer.sample_font_size()
    try:
        font = ImageFont.truetype(writer.font_path, font_size)
    except Exception:
        return None

    margin_left = random.randint(60, 150)
    margin_right = random.randint(60, 150)
    max_text_w = page_width - margin_left - margin_right
    y_cursor = random.randint(60, 150)

    paragraphs = []

    for pi in range(num_paragraphs):
        if y_cursor >= page_height - 100:
            break

        para_lines = random_paragraph()
        para_y_start = y_cursor
        rendered_lines = []

        line_spacing = int(font_size * writer.line_spacing_factor)

        for li, line_text in enumerate(para_lines):
            if y_cursor >= page_height - 60:
                break

            # Truncate to fit
            while len(line_text) > 1:
                try:
                    lw = font.getlength(line_text)
                except AttributeError:
                    lw = font.getsize(line_text)[0]
                if lw <= max_text_w:
                    break
                line_text = line_text.rsplit(" ", 1)[0]

            x_pos = margin_left
            # Slight indent for first line
            if li == 0 and random.random() < 0.3:
                x_pos += random.randint(20, 60)

            # Compute character positions
            words_in_line = line_text.split(" ")
            rendered_words = []
            char_global_idx = 0

            for wi, word_str in enumerate(words_in_line):
                word_chars = []
                word_x_start = x_pos

                for ci, ch in enumerate(word_str):
                    try:
                        cb = draw.textbbox((x_pos, y_cursor), ch, font=font)
                    except AttributeError:
                        cw, ch_h = font.getsize(ch)
                        cb = (x_pos, y_cursor, x_pos + cw, y_cursor + ch_h)

                    try:
                        adv = font.getlength(ch)
                    except AttributeError:
                        adv = font.getsize(ch)[0]

                    cb_int = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
                    if cb_int[2] > cb_int[0] and cb_int[3] > cb_int[1]:
                        word_chars.append(RenderedChar(
                            char=ch, bbox=cb_int, char_index=ci))

                    x_pos += adv * writer.spacing_factor

                if word_chars:
                    word_bbox = (
                        min(c.bbox[0] for c in word_chars),
                        min(c.bbox[1] for c in word_chars),
                        max(c.bbox[2] for c in word_chars),
                        max(c.bbox[3] for c in word_chars),
                    )
                    rendered_words.append(RenderedWord(
                        text=word_str, bbox=word_bbox,
                        characters=word_chars, word_index=wi))

                # Space between words
                try:
                    space_w = font.getlength(" ")
                except AttributeError:
                    space_w = font.getsize(" ")[0]
                x_pos += space_w * writer.spacing_factor

            # Draw the full line
            draw.text((margin_left + (random.randint(20, 60) if li == 0 and rendered_lines == [] and random.random() < 0.3 else 0),
                        y_cursor), line_text, fill=ink_color, font=font)

            if rendered_words:
                line_bbox = (
                    min(w.bbox[0] for w in rendered_words),
                    min(w.bbox[1] for w in rendered_words),
                    max(w.bbox[2] for w in rendered_words),
                    max(w.bbox[3] for w in rendered_words),
                )
                rendered_lines.append(RenderedLine(
                    text=line_text, bbox=line_bbox,
                    words=rendered_words, line_index=li))

            y_cursor += line_spacing

        if rendered_lines:
            para_bbox = (
                min(l.bbox[0] for l in rendered_lines),
                min(l.bbox[1] for l in rendered_lines),
                max(l.bbox[2] for l in rendered_lines),
                max(l.bbox[3] for l in rendered_lines),
            )
            paragraphs.append(RenderedParagraph(
                text="\n".join(l.text for l in rendered_lines),
                bbox=para_bbox, lines=rendered_lines,
                para_index=pi, is_handwritten=writer.is_handwriting))

        y_cursor += random.randint(20, 60)

    if not paragraphs:
        return None

    page_bbox = (
        min(p.bbox[0] for p in paragraphs),
        min(p.bbox[1] for p in paragraphs),
        max(p.bbox[2] for p in paragraphs),
        max(p.bbox[3] for p in paragraphs),
    )

    return RenderedPage(
        image=image, bg_color=bg_color,
        width=page_width, height=page_height,
        paragraphs=paragraphs, writer=writer,
        page_bbox=page_bbox,
    )


# ---------------------------------------------------------------------------
# Data extraction from rendered pages
# ---------------------------------------------------------------------------

def _bbox_to_parent_norm(
    bbox: Tuple[int, int, int, int],
    parent: Tuple[int, int, int, int],
) -> List[float]:
    px1, py1, px2, py2 = parent
    pw = max(px2 - px1, 1)
    ph = max(py2 - py1, 1)
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0 - px1) / pw
    cy = ((y1 + y2) / 2.0 - py1) / ph
    w = (x2 - x1) / pw
    h = (y2 - y1) / ph
    return [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]


def build_word_route_record(
    page: RenderedPage,
    word: RenderedWord,
    line: RenderedLine,
    para: RenderedParagraph,
    doc_id: str,
    out_dir: Path,
    weights: RouteWeights,
    save_images: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build a word route record with stroke extraction and routing."""
    word_bbox = word.bbox
    wx1, wy1, wx2, wy2 = word_bbox
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww <= 0 or wh <= 0:
        return None

    # Extract strokes for each character in the word
    all_strokes: List[PrimitiveStroke] = []
    for ci, ch_data in enumerate(word.characters):
        cb = ch_data.bbox
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            continue

        # Crop character from page
        char_crop = page.image.crop(cb).convert("L")
        arr = np.array(char_crop)

        # Binarize
        bg_val = np.median(arr)
        if bg_val > 128:
            binary = arr < (bg_val - 25)
        else:
            binary = arr > (bg_val + 25)

        if not binary.any():
            continue

        # Character bbox relative to word
        char_in_word = (cb[0] - wx1, cb[1] - wy1, cb[2] - wx1, cb[3] - wy1)
        char_id = f"char_{doc_id}_{para.para_index}_{line.line_index}_{word.word_index}_{ci}"

        strokes = extract_strokes_from_binary(
            binary, char_id, ci, char_in_word, (ww, wh))
        all_strokes.extend(strokes)

    if not all_strokes:
        return None

    # Route strokes
    route = route_word_strokes(all_strokes, weights)
    events = route_to_events(route, all_strokes)

    if not events:
        return None

    word_id = f"word_{doc_id}_{para.para_index}_{line.line_index}_{word.word_index}"
    line_id = f"line_{doc_id}_{para.para_index}_{line.line_index}"
    record_id = f"word_route_{word_id}"

    # Save state images
    state_before_ref = None
    state_after_ref = None
    if save_images:
        # State before = line crop (context for the word)
        lx1, ly1, lx2, ly2 = line.bbox
        line_crop = page.image.crop((lx1, ly1, lx2, ly2)).convert("L")
        # Scale to 256
        retina = 256
        lw, lh = line_crop.size
        scale = min(retina / max(lw, 1), retina / max(lh, 1))
        nw, nh = max(1, int(lw * scale)), max(1, int(lh * scale))
        resized = line_crop.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("L", (retina, retina), 255)
        canvas.paste(resized, ((retina - nw) // 2, (retina - nh) // 2))

        state_before_ref = f"states/{word_id}_state.png"
        state_path = out_dir / state_before_ref
        state_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(state_path))

    # Primitive stroke data for the record
    prim_strokes_data = []
    for s in all_strokes:
        prim_strokes_data.append({
            "stroke_id": f"{word_id}_s{s.stroke_id}",
            "char_id": s.char_id,
            "char_index": s.char_index,
            "points_fwd": [list(p) for p in s.points_fwd],
            "points_rev": [list(p) for p in s.points_rev],
            "arc_len": round(s.arc_len, 6),
        })

    selected_seq = []
    for idx, d in route.ordered:
        selected_seq.append({
            "stroke_id": f"{word_id}_s{all_strokes[idx].stroke_id}",
            "orientation": "fwd" if d == 0 else "rev",
        })

    return {
        "schema_version": "1.1",
        "record_id": record_id,
        "document_id": doc_id,
        "line_id": line_id,
        "word_id": word_id,
        "word_index": word.word_index,
        "word_text": word.text,
        "target_bbox_parent_norm_cxcywh": _bbox_to_parent_norm(word_bbox, line.bbox),
        "route_start_point": list(route.start_point),
        "route_end_point": list(route.end_point),
        "primitive_strokes": prim_strokes_data,
        "selected_sequence": selected_seq,
        "events": events,
        "plan_cost": round(route.total_cost, 6),
        "source_type": "pseudo_online",
        "style_id": page.writer.writer_id,
        "source_weight": 0.7,
        "confidence": 0.85,
        "state_before_ref": state_before_ref,
    }


def build_line_bridge_record(
    prev_route: Dict[str, Any],
    next_route: Dict[str, Any],
    doc_id: str,
    line_id: str,
) -> Dict[str, Any]:
    """Build a bridge record between two adjacent word routes in a line."""
    prev_end = prev_route["route_end_point"]
    next_start = next_route["route_start_point"]

    dx = next_start[0] - prev_end[0]
    dy = next_start[1] - prev_end[1]
    dist = math.hypot(dx, dy)

    bridge_events = [{
        "dx": round(dx, 6), "dy": round(dy, 6),
        "pen_down": 0, "stroke_end": 0,
        "char_end": 0, "word_end": 0, "seq_end": 0,
    }]

    return {
        "record_id": f"bridge_{prev_route['word_id']}_{next_route['word_id']}",
        "document_id": doc_id,
        "line_id": line_id,
        "prev_word_id": prev_route["word_id"],
        "next_word_id": next_route["word_id"],
        "prev_word_text": prev_route["word_text"],
        "next_word_text": next_route["word_text"],
        "prev_word_end_point": prev_end,
        "next_word_start_point": next_start,
        "bridge_events": bridge_events,
        "bridge_cost": round(dist, 6),
        "confidence": 0.85,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_one_hw_document(
    writer: WriterProfile,
    doc_index: int,
    out_dir: Path,
    weights: RouteWeights,
    save_images: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Generate one handwriting document with word routes and line bridges."""
    doc_id = f"hw_{doc_index:06d}"

    page = render_handwriting_page(writer)
    if page is None:
        return [], [], {}

    # Save page image
    if save_images:
        img_path = out_dir / "images" / f"{doc_id}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        page.image.save(str(img_path))

    word_routes = []
    line_bridges = []

    for para in page.paragraphs:
        for line in para.lines:
            line_routes = []
            for word in line.words:
                route_rec = build_word_route_record(
                    page, word, line, para, doc_id, out_dir,
                    weights, save_images=save_images,
                )
                if route_rec:
                    word_routes.append(route_rec)
                    line_routes.append(route_rec)

            # Build bridges between adjacent words
            line_id = f"line_{doc_id}_{para.para_index}_{line.line_index}"
            for i in range(len(line_routes) - 1):
                bridge = build_line_bridge_record(
                    line_routes[i], line_routes[i + 1], doc_id, line_id)
                line_bridges.append(bridge)

    # Document metadata
    meta = {
        "document_id": doc_id,
        "writer_id": writer.writer_id,
        "font": os.path.basename(writer.font_path),
        "is_handwriting": writer.is_handwriting,
        "num_paragraphs": len(page.paragraphs),
        "num_word_routes": len(word_routes),
        "num_bridges": len(line_bridges),
    }

    return word_routes, line_bridges, meta


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate handwriting-focused synthetic training data")
    parser.add_argument("--num-docs", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="data/hw_v1")
    parser.add_argument("--writers", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-images", action="store_true")
    # Route weights
    parser.add_argument("--lambda-up", type=float, default=1.0)
    parser.add_argument("--lambda-switch", type=float, default=0.3)
    parser.add_argument("--lambda-back", type=float, default=0.1)
    parser.add_argument("--beam-size", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering fonts...")
    fonts = discover_fonts()
    print(f"Found {len(fonts)} fonts")

    print(f"Creating {args.writers} writer profiles...")
    writers = create_writer_profiles(fonts, args.writers)

    weights = RouteWeights(
        lambda_up=args.lambda_up,
        lambda_switch=args.lambda_switch,
        lambda_back=args.lambda_back,
    )

    all_word_routes = []
    all_line_bridges = []
    all_meta = []

    for i in range(args.num_docs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Generating doc {i+1}/{args.num_docs}...")

        writer = writers[i % len(writers)]

        try:
            routes, bridges, meta = generate_one_hw_document(
                writer, i, out_dir, weights,
                save_images=not args.skip_images,
            )
            all_word_routes.extend(routes)
            all_line_bridges.extend(bridges)
            if meta:
                all_meta.append(meta)
        except Exception as e:
            print(f"  WARNING: doc {i} failed: {e}")
            continue

    # Split into train/val (90/10, writer-disjoint)
    writer_ids = sorted(set(m["writer_id"] for m in all_meta))
    n_val_writers = max(1, len(writer_ids) // 10)
    val_writers = set(writer_ids[-n_val_writers:])
    train_writers = set(writer_ids) - val_writers

    train_routes = [r for r in all_word_routes if r["style_id"] in train_writers]
    val_routes = [r for r in all_word_routes if r["style_id"] in val_writers]
    train_word_ids = {r["word_id"] for r in train_routes}
    val_word_ids = {r["word_id"] for r in val_routes}
    train_bridges = [b for b in all_line_bridges if b["prev_word_id"] in train_word_ids]
    val_bridges = [b for b in all_line_bridges if b["prev_word_id"] in val_word_ids]

    # Write outputs
    write_jsonl(out_dir / "train_word_routes.jsonl", train_routes)
    write_jsonl(out_dir / "val_word_routes.jsonl", val_routes)
    write_jsonl(out_dir / "train_line_bridges.jsonl", train_bridges)
    write_jsonl(out_dir / "val_line_bridges.jsonl", val_bridges)

    # Style vocab
    style_to_index = {w: i for i, w in enumerate(sorted(
        set(r["style_id"] for r in all_word_routes)))}
    with (out_dir / "style_to_index.json").open("w") as f:
        json.dump(style_to_index, f, indent=2)

    # Summary
    summary = {
        "num_documents": len(all_meta),
        "num_writers": len(writer_ids),
        "train_writers": len(train_writers),
        "val_writers": len(val_writers),
        "train_word_routes": len(train_routes),
        "val_word_routes": len(val_routes),
        "train_bridges": len(train_bridges),
        "val_bridges": len(val_bridges),
        "total_events": sum(len(r["events"]) for r in all_word_routes),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Output in {out_dir}/")
    print(f"  {len(all_meta)} documents, {len(writer_ids)} writers")
    print(f"  Train: {len(train_routes)} word routes, {len(train_bridges)} bridges")
    print(f"  Val:   {len(val_routes)} word routes, {len(val_bridges)} bridges")
    print(f"  Total events: {summary['total_events']}")


if __name__ == "__main__":
    main()
