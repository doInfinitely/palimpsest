#!/usr/bin/env python3
"""Inflate every per-letter bbox by a multiplicative factor (centered),
clipped to the canvas. Used to test whether the classifier was being
starved by tight bboxes.

Usage:
    python3 inflate_bboxes.py \\
        --in runs/letter_bboxes_v2.jsonl \\
        --out runs/letter_bboxes_v2_inflate1p33.jsonl \\
        --scale 1.33
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.33)
    ap.add_argument("--H", type=float, default=64.0)
    ap.add_argument("--W", type=float, default=256.0)
    args = ap.parse_args()

    s = args.scale
    half_extra = (s - 1.0) / 2.0
    n_clipped = 0
    n_letters = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            for L in r["letters"]:
                w = L["x2"] - L["x1"]
                h = L["y2"] - L["y1"]
                cx = 0.5 * (L["x1"] + L["x2"])
                cy = 0.5 * (L["y1"] + L["y2"])
                new_w = w * s
                new_h = h * s
                x1 = cx - new_w / 2.0
                x2 = cx + new_w / 2.0
                y1 = cy - new_h / 2.0
                y2 = cy + new_h / 2.0
                # Clip to canvas.
                if x1 < 0 or x2 > args.W or y1 < 0 or y2 > args.H:
                    n_clipped += 1
                L["x1"] = max(0.0, x1)
                L["y1"] = max(0.0, y1)
                L["x2"] = min(args.W, x2)
                L["y2"] = min(args.H, y2)
                n_letters += 1
            fout.write(json.dumps(r) + "\n")
    print(f"Inflated {n_letters} letter bboxes by ×{s} "
          f"(clipped {n_clipped} = {n_clipped/n_letters:.1%})")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
