#!/usr/bin/env python3
"""Topology-aware per-letter bbox extractor over the IAM ok-word corpus.

Mirrors extract_letter_bboxes.py but replaces the y-refinement step with the
topology pipeline (skeleton → trajectory plan → segment-to-letter assignment
→ Voronoi-fill → tight bbox per assigned ink pixels).

Pipeline per batch (B words):
  1. Prepare batch (image, ink masks, etc).
  2. Run the batched gradient divider → per-letter (x1, x2) for every word.
  3. For each word, sequentially run the topology refinement.
  4. Write JSONL entries.

Usage:
    python3 extract_letter_bboxes_topo.py \\
        --recognizer runs/char_recog_v4/best.pt \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --out runs/letter_bboxes_v2.jsonl \\
        --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize

from train_char_recognizer import LocalCharRecognizer, letterbox, parse_words_txt
from extract_letter_bboxes import (
    TARGET_H, TARGET_W, build_rf_matrix, prepare_word, run_batch,
)
from skeleton_plan import build_graph, prune_stubs, plan_exact
import skeleton_plan as SP

from extract_char_bboxes_topo import (
    extract_pen_segments, break_at_junctures, assign_segments_to_letters,
    voronoi_to_skeleton, build_skel_letter_map, refine_bboxes,
)


def topo_refine_one(prepared: Dict, letter_x_ranges, args) -> List[Dict]:
    """Run the topology refinement for one word, given divider x-ranges.
    Returns a list of letter dicts ready for JSONL."""
    word_arr = prepared["word_arr"]
    word_arr_blur = prepared["word_arr_blur"]
    text = prepared["rec"]["text"]
    N = prepared["N"]

    skel_mask = word_arr_blur < args.skel_threshold
    ink_mask = word_arr < args.ink_threshold
    skel = skeletonize(skel_mask)
    nodes, edges, comp_of_edge, n_comp = build_graph(skel)
    nodes, edges, comp_of_edge, n_comp = prune_stubs(
        nodes, edges, comp_of_edge, n_comp, args.stub_prune_threshold)

    if not edges:
        # No skeleton — fall back to divider x-ranges with default y.
        return [{
            "char": text[i],
            "x1": round(float(letter_x_ranges[i][0]), 2),
            "y1": 0.0,
            "x2": round(float(letter_x_ranges[i][1]), 2),
            "y2": float(TARGET_H),
            "prob": 0.0,
        } for i in range(N)]

    deg = {i: 0 for i in range(len(nodes))}
    for e in edges:
        deg[e["a"]] += 1
        if e["a"] != e["b"]:
            deg[e["b"]] += 1
    juncture_pixels = {tuple(nodes[i]) for i in range(len(nodes)) if deg[i] >= 3}

    SP.NODE_PX_TO_IDX = {tuple(p): i for i, p in enumerate(nodes)}
    SP.NODE_TO_PX_LOCAL = list(nodes)
    SP.EDGES = edges
    comp_min_x = {c: min(p[1] for eid, e in enumerate(edges)
                          if comp_of_edge[eid] == c for p in e["points"])
                  for c in range(n_comp)}
    components_lr = sorted(range(n_comp), key=lambda c: comp_min_x[c])
    word_left = min(p[1] for e in edges for p in e["points"])
    word_top = min(p[0] for e in edges for p in e["points"])
    word_bot = max(p[0] for e in edges for p in e["points"])
    seeds = [(int(y), max(0, int(word_left - 20)))
             for y in np.linspace(word_top, word_bot, 4).astype(int)]
    plan, _ = plan_exact(nodes, edges, comp_of_edge, components_lr,
                         seeds, args.penup_penalty)

    pen_segs = extract_pen_segments(plan.log, edges, nodes)
    sub_segs = break_at_junctures(pen_segs, juncture_pixels)
    assignments = assign_segments_to_letters(sub_segs, letter_x_ranges)
    nearest_idx = voronoi_to_skeleton(skel)
    s2l = build_skel_letter_map(skel.shape, sub_segs, assignments)
    labeled = s2l >= 0
    if labeled.any() and not labeled.all():
        ny, nx = distance_transform_edt(~labeled, return_indices=True)[1]
        s2l = s2l[ny, nx]
    H, W = skel.shape
    i2l = -np.ones((H, W), dtype=np.int32)
    ys, xs = np.where(ink_mask)
    i2l[ys, xs] = s2l[nearest_idx[0, ys, xs], nearest_idx[1, ys, xs]]
    bboxes = refine_bboxes(i2l, N, letter_x_ranges, margin=args.bbox_margin)

    return [{
        "char": text[i],
        "x1": round(float(bboxes[i][0]), 2),
        "y1": round(float(bboxes[i][1]), 2),
        "x2": round(float(bboxes[i][2]), 2),
        "y2": round(float(bboxes[i][3]), 2),
        "pixels": int((i2l == i).sum()),
    } for i in range(N)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--words-txt", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--mask-sharpness", type=float, default=0.5)
    ap.add_argument("--image-blur-sigma", type=float, default=1.0)
    ap.add_argument("--min-width", type=float, default=6.0)
    # Used by run_batch's divider but values won't propagate to topo.
    ap.add_argument("--refine-y-margin", type=float, default=1.0)
    ap.add_argument("--refine-y-threshold", type=float, default=0.5)
    # Topo controls.
    ap.add_argument("--skel-threshold", type=float, default=0.5)
    ap.add_argument("--ink-threshold", type=float, default=0.85)
    ap.add_argument("--stub-prune-threshold", type=float, default=3.0)
    ap.add_argument("--penup-penalty", type=float, default=2.0)
    ap.add_argument("--bbox-margin", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch = ckpt_args.get("base_ch", 32)
    final_stride = ckpt_args.get("final_stride", 2)
    model = LocalCharRecognizer(base_ch=base_ch, final_stride=final_stride).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    records = parse_words_txt(Path(args.words_txt), words_dir=Path(args.words_dir))
    records = [r for r in records if len(r["text"]) >= 2]
    records.sort(key=lambda r: len(r["text"]))
    if args.limit is not None:
        records = records[:args.limit]
    print(f"Total candidate records: {len(records)}")

    with torch.no_grad():
        dummy = torch.zeros(1, 1, TARGET_H, TARGET_W, device=device)
        fmap = model.feature_map(dummy)
    _, _, Hp, Wp = fmap.shape
    stride_w = TARGET_W // Wp
    rf_matrix = build_rf_matrix(Wp, stride_w, device)
    pixels = torch.arange(TARGET_W, device=device, dtype=torch.float32)
    print(f"Feature map: Hp={Hp}, Wp={Wp}, stride_w={stride_w}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    n_skipped = 0
    with open(out_path, "w") as fout:
        buf = []

        def flush():
            nonlocal n_done
            if not buf:
                return
            div_results = run_batch(buf, model, device, args, rf_matrix, pixels)
            for prepared, dr in zip(buf, div_results):
                x_ranges = [(L["x1"], L["x2"]) for L in dr["letters"]]
                topo_letters = topo_refine_one(prepared, x_ranges, args)
                rec = prepared["rec"]
                out_record = {
                    "word_id": rec["word_id"],
                    "form": rec["form"],
                    "line": rec["line"],
                    "text": rec["text"],
                    "target_h": TARGET_H,
                    "target_w": TARGET_W,
                    "word_x1": round(float(prepared["word_x1"]), 2),
                    "word_x2": round(float(prepared["word_x2"]), 2),
                    "word_y_top": round(float(prepared["y_top"]), 2),
                    "word_y_bot": round(float(prepared["y_bot"]), 2),
                    "letters": topo_letters,
                }
                fout.write(json.dumps(out_record) + "\n")
                n_done += 1
            fout.flush()
            buf.clear()

        for i, rec in enumerate(records):
            p = prepare_word(rec, Path(args.words_dir),
                             args.min_width, args.image_blur_sigma)
            if p is None:
                n_skipped += 1
                continue
            buf.append(p)
            if len(buf) >= args.batch_size:
                flush()
            if (i + 1) % args.log_every == 0 or i == len(records) - 1:
                dt = time.time() - t0
                rate = n_done / dt if dt > 0 else 0
                print(f"  [{i + 1:>6}/{len(records)}] done={n_done} skipped={n_skipped} "
                      f"elapsed={dt:.1f}s rate={rate:.1f} w/s")
        flush()

    dt = time.time() - t0
    print(f"\nSaved {n_done} records to {out_path} ({n_skipped} skipped) in {dt:.1f}s")


if __name__ == "__main__":
    main()
