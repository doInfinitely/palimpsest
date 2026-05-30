#!/usr/bin/env python3
"""Cross-reference the bbox refiner's per-divider movement with the GPT
audit's recommendations.

For each (word, divider) pair where the audit has signal (action != no_signal):
  - direction_gpt: -1 (left), 0 (ok), +1 (right)  -- the AUDIT's diagnosis
    of where the divider currently is, i.e. d_init is too-far-{left,right}.
    Action meanings (from analyze_divider_audit.py):
        "right"            → divider currently too far right → wants LEFT shift
        "left"             → divider currently too far left  → wants RIGHT shift
        "split"            → cut interior → no clean direction (snap)
        "ok"               → keep
  - direction_refined: sign(d_refined - d_init), bucketed by --tol-px.

Report:
  - Agreement rate on left/right-action dividers (refiner moved in the
    same direction the audit wanted, ignoring magnitude).
  - Mean / median refiner shift magnitude, bucketed by audit action.
  - Confusion-matrix-style table (gpt action × refiner direction).
  - Restricted views at confidence thresholds (>=0.50, >=0.85).

Usage:
    python3 compare_refiner_vs_audit.py \\
        --init-bbox-jsonl runs/letter_bboxes_v2.jsonl \\
        --refined-bbox-jsonl runs/bbox_refiner_v2/letter_bboxes_refined.jsonl \\
        --recommendations runs/divider_recommendations_v2.jsonl \\
        --tol-px 0.5
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_bbox_jsonl(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            out[r["word_id"]] = r
    return out


def load_recs(path: Path) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            out.setdefault(r["word_id"], []).append(r)
    return out


def dividers_pixels(letters: List[Dict], scale_x: float) -> List[float]:
    return [
        0.5 * (letters[i]["x2"] + letters[i + 1]["x1"]) * scale_x
        for i in range(len(letters) - 1)
    ]


def gpt_direction(action: str) -> int:
    """Direction the audit wants the divider to MOVE.
        -1 = left, +1 = right, 0 = ok, None = no direction (split, unsure).
    """
    if action == "ok":
        return 0
    if action == "right":   # currently too far right → move LEFT
        return -1
    if action == "left":    # currently too far left → move RIGHT
        return +1
    if action == "right_or_split":
        return -1
    if action == "left_or_split":
        return +1
    return None             # split / unsure / no_signal


def refined_direction(delta: float, tol: float) -> int:
    """Bucket the refiner shift into {-1, 0, +1} given a tolerance."""
    if delta > tol:
        return +1
    if delta < -tol:
        return -1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-bbox-jsonl", required=True)
    ap.add_argument("--refined-bbox-jsonl", required=True)
    ap.add_argument("--recommendations", required=True)
    ap.add_argument("--tol-px", type=float, default=0.5,
                    help="|delta| ≤ tol counted as 'no move' for the refiner.")
    ap.add_argument("--conf-bins", type=str, default="0.0,0.5,0.85",
                    help="Comma-separated min-confidence cutoffs to report.")
    args = ap.parse_args()

    init_by_word = load_bbox_jsonl(Path(args.init_bbox_jsonl))
    refined_by_word = load_bbox_jsonl(Path(args.refined_bbox_jsonl))
    recs_by_word = load_recs(Path(args.recommendations))

    # Rows we'll analyze: one per (word_id, divider_idx).
    rows: List[Tuple[str, int, float, float, str, int, int, float]] = []
    skipped_missing = 0
    for wid, recs in recs_by_word.items():
        if wid not in init_by_word or wid not in refined_by_word:
            skipped_missing += 1
            continue
        init = init_by_word[wid]
        refined = refined_by_word[wid]
        # Pixel scale: bboxes are in letterbox 64×256 coords; we want
        # *letterbox* pixel deltas (consistent with how the refiner moves).
        # That means we don't need to multiply by scale_x.
        init_letters = init["letters"]
        ref_letters = refined["letters"]
        if len(init_letters) != len(ref_letters):
            continue
        init_divs = dividers_pixels(init_letters, scale_x=1.0)
        ref_divs = dividers_pixels(ref_letters, scale_x=1.0)
        for r in recs:
            i = r["divider_idx"]
            if i - 1 < 0 or i - 1 >= len(init_divs):
                continue
            d_init = init_divs[i - 1]
            d_ref = ref_divs[i - 1]
            delta = d_ref - d_init
            rows.append((wid, i, d_init, d_ref, r["action"],
                         gpt_direction(r["action"]) if r["action"] not in ("split", "unsure") else None,
                         refined_direction(delta, args.tol_px),
                         float(r.get("combined_confidence", 0.0))))

    if skipped_missing:
        print(f"[note] skipped {skipped_missing} words missing from one or both bbox files")
    print(f"Analyzed {len(rows)} (word,divider) pairs\n")

    # ------------------------------------------------------------------
    # Per-action shift magnitudes
    # ------------------------------------------------------------------
    by_action: Dict[str, List[float]] = defaultdict(list)
    for _, _, di, dr, act, _, _, _ in rows:
        by_action[act].append(dr - di)

    print("Mean refiner shift (px) by audit action:")
    print(f"  {'action':<18} {'n':>6} {'mean':>8} {'median':>8} {'p95|':>10}")
    for act in sorted(by_action, key=lambda a: -len(by_action[a])):
        ds = by_action[act]
        if not ds:
            continue
        mean = sum(ds) / len(ds)
        med = statistics.median(ds)
        p95 = sorted(abs(d) for d in ds)[int(0.95 * len(ds))] if len(ds) >= 20 else float("nan")
        print(f"  {act:<18} {len(ds):>6} {mean:>+8.2f} {med:>+8.2f} {p95:>10.2f}")
    print()

    # ------------------------------------------------------------------
    # Confidence-bucketed agreement rate (only for "left"/"right" actions)
    # ------------------------------------------------------------------
    conf_cuts = [float(c) for c in args.conf_bins.split(",")]
    print("Agreement on direction (refiner moved the way GPT wanted):")
    print(f"  conf cut |  n_dir-actions  |  agree  disagree  no-move  |  rate")
    for cut in conf_cuts:
        agree = disagree = nomove = 0
        for _, _, _, _, act, gd, rd, conf in rows:
            if gd not in (-1, +1):
                continue
            if conf < cut:
                continue
            if rd == 0:
                nomove += 1
            elif rd == gd:
                agree += 1
            else:
                disagree += 1
        tot_dir = agree + disagree + nomove
        rate = agree / max(tot_dir - nomove, 1)
        print(f"  ≥{cut:>4.2f}   |  {tot_dir:>13}  | "
              f"{agree:>5}  {disagree:>8}  {nomove:>7}  |  "
              f"{rate:>5.1%}  (excl. no-move)")
    print()

    # ------------------------------------------------------------------
    # Confusion-style matrix (rows: gpt action; cols: refiner direction)
    # ------------------------------------------------------------------
    actions_in_order = ["ok", "left", "right", "split", "left_or_split",
                        "right_or_split", "unsure"]
    print("Confusion matrix (gpt action ↓  vs  refiner shift →):")
    print(f"  {'action':<18}  {'-1':>8} {'0':>8} {'+1':>8}  | {'n':>6}")
    refined_counts: Dict[str, Counter] = defaultdict(Counter)
    for _, _, _, _, act, _, rd, _ in rows:
        refined_counts[act][rd] += 1
    for act in actions_in_order:
        c = refined_counts.get(act, Counter())
        n = sum(c.values())
        if n == 0:
            continue
        print(f"  {act:<18}  {c[-1]:>8} {c[0]:>8} {c[+1]:>8}  | {n:>6}")
    print()

    # ------------------------------------------------------------------
    # Magnitude analysis: did refiner move BIG when GPT confidence was high?
    # ------------------------------------------------------------------
    print("Refiner |Δ| (px) on high-conf direction-actions:")
    for cut in conf_cuts:
        mags = [abs(dr - di) for _, _, di, dr, act, gd, _, conf in rows
                if gd in (-1, +1) and conf >= cut]
        if not mags:
            continue
        mags_sorted = sorted(mags)
        mean = sum(mags) / len(mags)
        med = mags_sorted[len(mags) // 2]
        p90 = mags_sorted[int(0.9 * len(mags))]
        print(f"  conf≥{cut:.2f}  n={len(mags):>5}  "
              f"mean={mean:.2f}  median={med:.2f}  p90={p90:.2f}")


if __name__ == "__main__":
    main()
