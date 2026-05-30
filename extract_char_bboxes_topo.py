#!/usr/bin/env python3
"""Topology-aware per-letter bbox extraction.

Pipeline:
  1. Skeletonize the word.
  2. Plan a stroke trajectory exactly (open RPP per component).
  3. Extract pen-down → pen-up segments.
  4. Break each segment at skeleton junctures (degree ≥ 3 in pruned graph).
  5. Run the horizontal divider opt → per-letter x-range from the gradient
     model (no y-refinement at this point).
  6. Assign each sub-segment to the letter whose x-range it most overlaps.
  7. For every ink pixel (below threshold), find its nearest skeleton pixel
     (Voronoi). The pixel inherits its skeleton point's letter assignment.
  8. Refine each letter's bbox (both x and y) to the tight extent of its
     assigned ink pixels.
  9. Render two visualizations:
       (a) sub-segments + ink pixels color-coded by letter assignment.
       (b) final refined bboxes overlaid on the original word.

Usage:
    python3 extract_char_bboxes_topo.py \\
        --recognizer runs/char_recog_v4/best.pt \\
        --words-txt data/iam_words/iam_words/words.txt \\
        --words-dir data/iam_words/iam_words/words \\
        --word-id c03-007-02-07 \\
        --out-prefix eval_output/topo_shabby
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize

from train_char_recognizer import LocalCharRecognizer, letterbox, VOCAB, parse_words_txt
from extract_letter_bboxes import (
    TARGET_H, TARGET_W, build_rf_matrix, prepare_word,
)
from skeleton_plan import (
    build_graph, prune_stubs, plan_exact,
)


# ------------------------------------------------------------------
# Divider opt (single word) with optional x-center regularization
# ------------------------------------------------------------------

def divider_opt_single(model, device, prepared, args, rf_matrix, pixels,
                       target_x_centers=None, lambda_reg: float = 0.0):
    """Run the divider opt for one word. Returns list of (x1, x2) per letter.

    If target_x_centers is given (length N), adds an L2 penalty pulling
    each letter box's x-center toward the corresponding target. Used as
    soft feedback from the topology stage.
    """
    p = prepared
    N = p["N"]
    targets = torch.from_numpy(p["targets"]).to(device)
    word_x1 = float(p["word_x1"])
    y_top = float(p["y_top"])
    y_bot = float(p["y_bot"])
    slack = float(p["slack"])

    ink_img = torch.from_numpy(1.0 - p["word_arr_blur"])[None, None].to(device).float()
    with torch.no_grad():
        fmap = model.feature_map(ink_img)
    _, C, Hp, Wp = fmap.shape
    stride_h = TARGET_H // Hp
    tgt_logits_2d = fmap[0][targets]  # [N, Hp, Wp]

    row_centers = stride_h * torch.arange(Hp, device=device, dtype=torch.float32)
    y_range = max(1.0, y_bot - y_top)
    row_frac = ((row_centers - y_top) / y_range).clamp(0.0, 1.0)
    wx1_t = torch.tensor(word_x1, device=device, dtype=torch.float32)

    width_logits = torch.zeros(N, device=device, dtype=torch.float32,
                               requires_grad=True)
    opt = torch.optim.Adam([width_logits], lr=args.lr)

    target_t = None
    if target_x_centers is not None:
        target_t = torch.tensor(target_x_centers, device=device, dtype=torch.float32)

    for step in range(args.steps):
        widths = args.min_width + torch.softmax(width_logits, dim=0) * slack
        cum = torch.cumsum(widths, dim=0)
        pos = torch.cat([wx1_t[None], wx1_t + cum])  # [N+1]
        x1s, x2s = pos[:-1], pos[1:]
        left = x1s[:, None].expand(N, Hp)
        right = x2s[:, None].expand(N, Hp)
        k = args.mask_sharpness
        pixel_w = (torch.sigmoid(k * (pixels[None, None, :] - left[:, :, None]))
                   * torch.sigmoid(k * (right[:, :, None] - pixels[None, None, :])))
        col_w = pixel_w @ rf_matrix.t()
        gated = tgt_logits_2d + torch.log(col_w.clamp(min=1e-8))
        score = torch.logsumexp(gated.view(N, -1), dim=1)
        target_probs = torch.sigmoid(score)
        loss = -target_probs.sum()
        if target_t is not None and lambda_reg > 0:
            centers = 0.5 * (x1s + x2s)
            reg = ((centers - target_t) ** 2).mean()
            loss = loss + lambda_reg * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        widths = args.min_width + torch.softmax(width_logits, dim=0) * slack
        cum = torch.cumsum(widths, dim=0)
        pos = torch.cat([wx1_t[None], wx1_t + cum])
    pos_np = pos.detach().cpu().numpy()
    return [(float(pos_np[i]), float(pos_np[i + 1])) for i in range(N)]


# ------------------------------------------------------------------
# Trajectory → pen-segment extraction
# ------------------------------------------------------------------

def extract_pen_segments(action_log, edges, nodes):
    """Return a list of pen-down segments. Each segment is a list of (y, x)
    points (the trajectory while pen is down)."""
    segments: List[List[Tuple[int, int]]] = []
    cur_seg: List[Tuple[int, int]] = []
    cur_pos = None
    for a in action_log:
        if a[0] == "init":
            cur_pos = a[1]
        elif a[0] == "jump":
            if cur_seg:
                segments.append(cur_seg)
                cur_seg = []
            cur_pos = a[1]
        elif a[0] == "draw":
            pts = [tuple(p) for p in a[2]]
            if cur_seg and cur_seg[-1] == pts[0]:
                cur_seg.extend(pts[1:])
            else:
                cur_seg.extend(pts)
            cur_pos = pts[-1]
        elif a[0] == "retrace":
            eid = a[1]
            e = edges[eid]
            pa = tuple(nodes[e["a"]])
            pts = [tuple(p) for p in (e["points"] if tuple(cur_pos) == pa
                                      else list(reversed(e["points"])))]
            if cur_seg and cur_seg[-1] == pts[0]:
                cur_seg.extend(pts[1:])
            else:
                cur_seg.extend(pts)
            cur_pos = pts[-1]
    if cur_seg:
        segments.append(cur_seg)
    return segments


def break_at_junctures(pen_segments, juncture_pixels):
    """Split each pen-down segment whenever it passes through a juncture pixel."""
    out: List[List[Tuple[int, int]]] = []
    for seg in pen_segments:
        if len(seg) < 2:
            continue
        cur = [seg[0]]
        for p in seg[1:]:
            cur.append(p)
            if p in juncture_pixels and len(cur) >= 2:
                out.append(cur)
                cur = [p]
        if len(cur) >= 2:
            out.append(cur)
    return out


# ------------------------------------------------------------------
# Segment → letter assignment
# ------------------------------------------------------------------

def assign_segments_to_letters(sub_segments, letter_x_ranges):
    assignments = []
    for seg in sub_segments:
        scores = [sum(1 for (_, x) in seg if x1 <= x <= x2)
                  for (x1, x2) in letter_x_ranges]
        if max(scores) == 0:
            mean_x = sum(x for _, x in seg) / len(seg)
            d = [min(abs(mean_x - x1), abs(mean_x - x2))
                 for x1, x2 in letter_x_ranges]
            assignments.append(int(np.argmin(d)))
        else:
            assignments.append(int(np.argmax(scores)))
    return assignments


# ------------------------------------------------------------------
# Pixel → letter via nearest skeleton point
# ------------------------------------------------------------------

def voronoi_to_skeleton(skel: np.ndarray):
    return distance_transform_edt(~skel, return_indices=True)[1]


def build_skel_letter_map(skel_shape, sub_segments, assignments) -> np.ndarray:
    out = -np.ones(skel_shape, dtype=np.int32)
    for seg, lid in zip(sub_segments, assignments):
        for (y, x) in seg:
            out[y, x] = lid
    return out


# ------------------------------------------------------------------
# Bbox refinement
# ------------------------------------------------------------------

def refine_bboxes(ink_to_letter, n_letters, fallback_x_ranges,
                  default_y=(0.0, float(TARGET_H)), margin: float = 1.0):
    H, W = ink_to_letter.shape
    out = []
    for lid in range(n_letters):
        ys, xs = np.where(ink_to_letter == lid)
        if ys.size == 0:
            x1, x2 = fallback_x_ranges[lid]
            out.append((float(x1), float(default_y[0]),
                        float(x2), float(default_y[1])))
            continue
        x1 = float(max(0, xs.min() - margin))
        x2 = float(min(W, xs.max() + 1 + margin))
        y1 = float(max(0, ys.min() - margin))
        y2 = float(min(H, ys.max() + 1 + margin))
        out.append((x1, y1, x2, y2))
    return out


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

PALETTE = [
    (220, 20, 60), (0, 150, 60), (0, 80, 220), (200, 110, 0),
    (160, 0, 160), (0, 150, 150), (120, 60, 0), (80, 80, 80),
    (220, 130, 0), (50, 200, 100), (50, 100, 200), (200, 50, 200),
]


def render_assignment(word_arr, ink_to_letter, sub_segments, assignments,
                      transcription, out_path: Path, scale: int = 4):
    H, W = word_arr.shape
    base = (word_arr * 255).astype(np.uint8)
    rgb = np.stack([base] * 3, axis=-1)
    rgb = (rgb.astype(np.float32) * 0.5 + 127).clip(0, 255).astype(np.uint8)
    for lid in range(len(transcription)):
        mask = ink_to_letter == lid
        if mask.any():
            rgb[mask] = PALETTE[lid % len(PALETTE)]
    img = Image.fromarray(rgb).resize((W * scale, H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for seg, lid in zip(sub_segments, assignments):
        c = PALETTE[lid % len(PALETTE)]
        bright = tuple(min(255, v + 40) for v in c)
        pts = [(x * scale + scale // 2, y * scale + scale // 2) for (y, x) in seg]
        if len(pts) >= 2:
            draw.line(pts, fill=bright, width=max(1, scale // 2))
    written = set()
    for seg, lid in zip(sub_segments, assignments):
        if lid in written:
            continue
        written.add(lid)
        y, x = seg[0]
        ch = transcription[lid] if lid < len(transcription) else "?"
        draw.text((x * scale, max(0, y * scale - 14)),
                  f"{ch}:{lid}", fill=(0, 0, 0))
    img.save(out_path)


def render_final_bboxes(word_arr, bboxes, transcription, out_path: Path,
                        scale: int = 4):
    H, W = word_arr.shape
    base = (word_arr * 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB").resize(
        (W * scale, H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for lid, (x1, y1, x2, y2) in enumerate(bboxes):
        c = PALETTE[lid % len(PALETTE)]
        draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                       outline=c, width=max(1, scale // 2))
        ch = transcription[lid] if lid < len(transcription) else "?"
        draw.text((x1 * scale + 2, y1 * scale + 2), ch, fill=c)
    img.save(out_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--words-txt", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--word-id", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--skel-threshold", type=float, default=0.5,
                    help="Threshold on the BLURRED image for skeletonization "
                         "(tight; defines the centerline).")
    ap.add_argument("--ink-threshold", type=float, default=0.95,
                    help="Threshold on the UNBLURRED image for ink pixels to "
                         "associate with bboxes (permissive; matches the "
                         "original word-level ink mask threshold).")
    ap.add_argument("--blur-sigma", type=float, default=1.0)
    ap.add_argument("--stub-prune-threshold", type=float, default=3.0)
    ap.add_argument("--penup-penalty", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--mask-sharpness", type=float, default=0.5)
    ap.add_argument("--min-width", type=float, default=6.0)
    ap.add_argument("--refine-y-margin", type=float, default=1.0)
    ap.add_argument("--refine-y-threshold", type=float, default=0.5)
    ap.add_argument("--bbox-margin", type=float, default=1.0)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--feedback-iters", type=int, default=1,
                    help="After the initial divider, re-run the topology "
                         "pipeline → re-run the divider with x-center "
                         "regularization toward the topology bboxes; this many "
                         "additional rounds.")
    ap.add_argument("--feedback-lambda", type=float, default=0.005,
                    help="L2 weight on the divider's x-center regularization "
                         "during feedback iterations.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = parse_words_txt(Path(args.words_txt), words_dir=Path(args.words_dir))
    rec = next((r for r in records if r["word_id"] == args.word_id), None)
    if rec is None:
        raise SystemExit(f"No record for word_id={args.word_id}")
    transcription = rec["text"]
    if not all(c in VOCAB for c in transcription):
        raise SystemExit(f"Transcription {transcription!r} has out-of-vocab chars")
    print(f"word_id={args.word_id}  text={transcription!r}")

    img = Image.open(Path(args.words_dir) / rec["form"] / rec["line"]
                     / f"{rec['word_id']}.png").convert("L")
    word_arr = letterbox(img, TARGET_H, TARGET_W)
    word_arr_blur = gaussian_filter(word_arr, sigma=args.blur_sigma)
    skel_mask = word_arr_blur < args.skel_threshold
    ink_mask = word_arr < args.ink_threshold
    print(f"skel mask (blur σ={args.blur_sigma}, t={args.skel_threshold}): "
          f"{int(skel_mask.sum())}  |  "
          f"ink mask (raw, t={args.ink_threshold}): {int(ink_mask.sum())}")

    # Skeleton + graph (from the tight skel mask)
    skel = skeletonize(skel_mask)
    nodes, edges, comp_of_edge, n_comp = build_graph(skel)
    nodes, edges, comp_of_edge, n_comp = prune_stubs(
        nodes, edges, comp_of_edge, n_comp, args.stub_prune_threshold)
    print(f"skeleton: {int(skel.sum())} px  graph: {len(edges)} edges, {n_comp} comps")

    deg = {i: 0 for i in range(len(nodes))}
    for e in edges:
        deg[e["a"]] += 1
        if e["a"] != e["b"]:
            deg[e["b"]] += 1
    juncture_pixels = {tuple(nodes[i]) for i in range(len(nodes)) if deg[i] >= 3}
    print(f"junctures: {len(juncture_pixels)}")

    # Plan trajectory
    import skeleton_plan as SP
    SP.NODE_PX_TO_IDX = {tuple(p): i for i, p in enumerate(nodes)}
    SP.NODE_TO_PX_LOCAL = list(nodes)
    SP.EDGES = edges
    comp_min_x = {c: min(p[1] for eid, e in enumerate(edges) if comp_of_edge[eid] == c
                          for p in e["points"]) for c in range(n_comp)}
    components_lr = sorted(range(n_comp), key=lambda c: comp_min_x[c])
    word_left = min(p[1] for e in edges for p in e["points"])
    word_top = min(p[0] for e in edges for p in e["points"])
    word_bot = max(p[0] for e in edges for p in e["points"])
    seeds = [(int(y), max(0, int(word_left - 20)))
             for y in np.linspace(word_top, word_bot, 4).astype(int)]
    plan, _seed = plan_exact(nodes, edges, comp_of_edge, components_lr,
                             seeds, args.penup_penalty)
    print(f"plan: {len(plan.log)} actions  weighted cost {plan.dist:.1f}")

    pen_segments = extract_pen_segments(plan.log, edges, nodes)
    sub_segments = break_at_junctures(pen_segments, juncture_pixels)
    print(f"pen-down segments: {len(pen_segments)}  →  sub-segments: {len(sub_segments)}")

    # Horizontal divider
    ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    base_ch = ckpt_args.get("base_ch", 32)
    final_stride = ckpt_args.get("final_stride", 2)
    model = LocalCharRecognizer(base_ch=base_ch, final_stride=final_stride).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    div_args = SimpleNamespace(
        steps=args.steps, lr=args.lr, mask_sharpness=args.mask_sharpness,
        image_blur_sigma=args.blur_sigma, min_width=args.min_width,
        refine_y_margin=args.refine_y_margin,
        refine_y_threshold=args.refine_y_threshold,
    )
    prepared = prepare_word(rec, Path(args.words_dir),
                            div_args.min_width, div_args.image_blur_sigma)
    if prepared is None:
        raise SystemExit("prepare_word returned None")
    with torch.no_grad():
        dummy = torch.zeros(1, 1, TARGET_H, TARGET_W, device=device)
        fmap = model.feature_map(dummy)
    _, _, Hp, Wp = fmap.shape
    stride_w = TARGET_W // Wp
    rf_matrix = build_rf_matrix(Wp, stride_w, device)
    pixels = torch.arange(TARGET_W, device=device, dtype=torch.float32)
    # Pre-compute Voronoi (independent of divider state).
    nearest_idx = voronoi_to_skeleton(skel)  # [2, H, W]
    H, W = skel.shape

    letter_x_ranges = divider_opt_single(
        model, device, prepared, div_args, rf_matrix, pixels,
        target_x_centers=None, lambda_reg=0.0,
    )
    print("divider x-ranges (round 0): " + ", ".join(
        f"{transcription[i]}=[{x1:.1f},{x2:.1f}]"
        for i, (x1, x2) in enumerate(letter_x_ranges)))

    final_bboxes = None
    ink_to_letter = None
    assignments = None
    for round_i in range(args.feedback_iters + 1):
        assignments = assign_segments_to_letters(sub_segments, letter_x_ranges)
        skel_to_letter = build_skel_letter_map(skel.shape, sub_segments, assignments)
        # Propagate labels to ALL skeleton pixels (including pruned hairs and
        # other skel pixels not in any sub-segment) by Voronoi over the
        # already-labeled skeleton pixels. Without this, ink near a pruned hair
        # has nearest-skeleton-pixel = -1 and ends up unlabeled.
        labeled_mask = skel_to_letter >= 0
        if labeled_mask.any() and not labeled_mask.all():
            ny_idx, nx_idx = distance_transform_edt(
                ~labeled_mask, return_indices=True)[1]
            skel_to_letter = skel_to_letter[ny_idx, nx_idx]
        ink_to_letter = -np.ones((H, W), dtype=np.int32)
        ys, xs = np.where(ink_mask)
        ink_to_letter[ys, xs] = skel_to_letter[
            nearest_idx[0, ys, xs], nearest_idx[1, ys, xs]
        ]
        final_bboxes = refine_bboxes(
            ink_to_letter, len(transcription),
            fallback_x_ranges=letter_x_ranges,
            margin=args.bbox_margin,
        )
        print(f"--- topo round {round_i} ---")
        for i, (x1, y1, x2, y2) in enumerate(final_bboxes):
            n_pix = int((ink_to_letter == i).sum())
            print(f"  bbox[{i}] '{transcription[i]}': "
                  f"x=[{x1:.1f},{x2:.1f}] y=[{y1:.1f},{y2:.1f}]  pixels={n_pix}")
        if round_i < args.feedback_iters:
            target_centers = [0.5 * (b[0] + b[2]) for b in final_bboxes]
            letter_x_ranges = divider_opt_single(
                model, device, prepared, div_args, rf_matrix, pixels,
                target_x_centers=target_centers, lambda_reg=args.feedback_lambda,
            )
            print(f"divider x-ranges (round {round_i + 1}): " + ", ".join(
                f"{transcription[i]}=[{x1:.1f},{x2:.1f}]"
                for i, (x1, x2) in enumerate(letter_x_ranges)))

    out_assign = Path(f"{args.out_prefix}_assign.png")
    out_final = Path(f"{args.out_prefix}_final.png")
    out_assign.parent.mkdir(parents=True, exist_ok=True)
    render_assignment(word_arr, ink_to_letter, sub_segments, assignments,
                      transcription, out_assign, scale=args.scale)
    render_final_bboxes(word_arr, final_bboxes, transcription, out_final,
                        scale=args.scale)
    print(f"saved {out_assign}")
    print(f"saved {out_final}")


if __name__ == "__main__":
    main()
