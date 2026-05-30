#!/usr/bin/env python3
"""Turn raw vision-LLM audit results into per-divider movement recommendations.

For each word with N letters and N-1 dividers (d_1, ..., d_{N-1}), each
divider d_i is involved in two crops from the audit:
  - front_i (cuts at d_i, keeping first i letters): drift > 0 means
    letter i was visible → d_i was too far right.
  - back_(N-i) (cuts at d_i, keeping last N-i letters): drift > 0 means
    letter i-1 was visible → d_i was too far left.

Combined diagnosis per divider:
  front>0  AND  back<0  → divider too far RIGHT (shift left)
  front<0  AND  back>0  → divider too far LEFT  (shift right)
  front>0  AND  back>0  → both adjacent letters leak (cut inside a
                          junction; snap to local ink minimum)
  front<0  AND  back<0  → both letters partially hidden (rare; cut into
                          mid-stroke of either)
  front=0  AND  back=0  → clean

Confidence is the geometric mean of the two crops' modal share (fraction
of n samples agreeing on the modal drift). Higher = more reliable signal.

Output JSONL: one record per (word_id, divider_idx) with recommendation
and confidence. Downstream refinement scripts can filter by confidence.

Usage:
    python3 analyze_divider_audit.py \\
        --audit-jsonl runs/divider_audit_v2.jsonl \\
        --out runs/divider_recommendations_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def modal_and_confidence(drift_distribution) -> Tuple[int, float, int]:
    """drift_distribution is a list of [drift, count] pairs (sorted)."""
    if not drift_distribution:
        return 0, 0.0, 0
    pairs = [tuple(p) for p in drift_distribution]
    total = sum(c for _, c in pairs)
    if total == 0:
        return 0, 0.0, 0
    drift, count = max(pairs, key=lambda kv: kv[1])
    return int(drift), count / total, total


def accuracy_aware_drift(reads: List[List[str]], crop_label: str,
                         expected_text: str) -> Tuple[Optional[int], float, int]:
    """Returns (modal_drift, confidence, n_valid_samples) using only the
    samples where the read actually contains the expected substring as a
    prefix (front) or suffix (back). Comparisons are case-insensitive.

    A read is "valid" if expected_text is a prefix/suffix of the read
    (joined as lowercase). Drift = len(read) - len(expected_text).
    Invalid samples are ignored — if all are invalid, returns (None, 0, 0).
    """
    if not reads:
        return None, 0.0, 0
    exp = expected_text.lower()
    valid_drifts = []
    for r in reads:
        joined = "".join(r).lower()
        if crop_label.startswith("front"):
            if joined.startswith(exp):
                valid_drifts.append(len(r) - len(exp))
        elif crop_label.startswith("back"):
            if joined.endswith(exp):
                valid_drifts.append(len(r) - len(exp))
        else:  # full
            valid_drifts.append(len(r) - len(exp))
    if not valid_drifts:
        return None, 0.0, 0
    counts: Dict[int, int] = {}
    for d in valid_drifts:
        counts[d] = counts.get(d, 0) + 1
    drift, c = max(counts.items(), key=lambda kv: kv[1])
    return int(drift), c / len(valid_drifts), len(valid_drifts)


def diagnose(front_drift: int, back_drift: int) -> str:
    if front_drift == 0 and back_drift == 0:
        return "ok"
    if front_drift > 0 and back_drift > 0:
        return "split"  # both adjacent letters leak — interior cut
    if front_drift > 0 and back_drift < 0:
        return "right"  # divider too far right; shift left
    if front_drift < 0 and back_drift > 0:
        return "left"   # divider too far left; shift right
    if front_drift > 0 and back_drift == 0:
        return "right"  # asymmetric: front leaks
    if front_drift < 0 and back_drift == 0:
        return "right_or_split"  # front under-reads but back ok
    if front_drift == 0 and back_drift > 0:
        return "left"
    if front_drift == 0 and back_drift < 0:
        return "left_or_split"
    return "unsure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_words = 0
    n_recs = 0
    counts: Dict[str, int] = {}
    confidences: List[float] = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    with open(args.audit_jsonl) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            a = json.loads(raw)
            n_words += 1
            text = a["text"]
            N = a["n_letters"]
            audits_by_label: Dict[str, dict] = {r["label"]: r for r in a["audits"]}

            # For each divider d_i (i=1..N-1), look up front_i and back_(N-i).
            for i in range(1, N):
                front = audits_by_label.get(f"front_{i}")
                back = audits_by_label.get(f"back_{N - i}")
                if not front or not back:
                    continue
                # Accuracy-aware drift: only count samples where the read
                # actually matches the expected substring.
                f_drift, f_conf, f_n = accuracy_aware_drift(
                    front.get("reads", []), front.get("label", ""),
                    front.get("expected_text", ""))
                b_drift, b_conf, b_n = accuracy_aware_drift(
                    back.get("reads", []), back.get("label", ""),
                    back.get("expected_text", ""))
                # Skip dividers with no valid signal on either side.
                if f_drift is None and b_drift is None:
                    counts["no_signal"] = counts.get("no_signal", 0) + 1
                    continue
                if f_drift is None: f_drift = 0
                if b_drift is None: b_drift = 0
                # Combined confidence: geometric mean.
                if f_conf > 0 and b_conf > 0:
                    combined_conf = (f_conf * b_conf) ** 0.5
                else:
                    combined_conf = 0.0
                action = diagnose(f_drift, b_drift)
                counts[action] = counts.get(action, 0) + 1
                confidences.append(combined_conf)

                rec = {
                    "word_id": a["word_id"],
                    "text": text,
                    "divider_idx": i,                 # divider between letter i-1 and letter i
                    "left_letter": text[i - 1] if i - 1 < len(text) else None,
                    "right_letter": text[i] if i < len(text) else None,
                    "front_modal_drift": f_drift,
                    "front_confidence": f_conf,
                    "front_samples": f_n,
                    "back_modal_drift": b_drift,
                    "back_confidence": b_conf,
                    "back_samples": b_n,
                    "combined_confidence": combined_conf,
                    "action": action,
                }
                fout.write(json.dumps(rec) + "\n")
                n_recs += 1

    fout.close()
    print(f"Processed {n_words} words → {n_recs} divider records")
    print(f"Saved to {out_path}\n")
    print("Action distribution:")
    for action in sorted(counts, key=lambda k: -counts[k]):
        c = counts[action]
        pct = c / max(n_recs, 1)
        print(f"  {action:>16}: {c:>5}  ({pct:.1%})")
    if confidences:
        import statistics
        print(f"\nCombined confidence:")
        print(f"  mean={statistics.mean(confidences):.2%}  "
              f"median={statistics.median(confidences):.2%}")
        for thr in (0.5, 0.7, 0.85, 1.0):
            share = sum(1 for c in confidences if c >= thr) / len(confidences)
            print(f"  ≥{thr:.2f}: {share:.1%}")


if __name__ == "__main__":
    main()
