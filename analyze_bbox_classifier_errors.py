#!/usr/bin/env python3
"""Diagnose the bbox-refiner classifier's errors on the val split.

Outputs:
  - per-class accuracy + support, sorted by support
  - top confusion pairs (true → predicted) over all misclassifications
  - a contact sheet of N random misclassified val instances showing the
    word image with the target letter's bbox highlighted, plus
    true → predicted

Usage:
    python3 analyze_bbox_classifier_errors.py \\
        --checkpoint runs/bbox_inflate_2p00_25ep/p1_best.pt \\
        --bbox-jsonl runs/letter_bboxes_v2_inflate2p00.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out-dir eval_output/bbox_clf_errors \\
        --n-mis 36 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from train_bbox_refiner import (
    LetterInstanceDataset, WordContextClassifier, collate,
    soft_bbox_mask, compose_masked, NUM_CLASSES,
)
from train_char_recognizer import VOCAB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-mis", type=int, default=36,
                    help="Number of misclassified samples to render.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--base-ch", type=int, default=32)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact same train/val split as training.
    full = LetterInstanceDataset(args.bbox_jsonl, args.words_dir)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    _, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=True)
    print(f"Val size: {len(val_ds)}")

    # Load model + bbox_table.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WordContextClassifier(base_ch=args.base_ch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    bbox_table = torch.nn.Embedding(len(full), 4).to(device)
    bbox_table.load_state_dict(ckpt["bbox_table_state_dict"])

    # Per-class counters + misclassification list (true, pred, inst_idx).
    class_correct = Counter()
    class_total = Counter()
    confusion: Dict[Tuple[int, int], int] = defaultdict(int)
    mis_list: List[Tuple[int, int, int]] = []  # (true, pred, instance_idx)
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            idxs = batch["instance_idx"]
            bbox = bbox_table(idxs)
            mask = soft_bbox_mask(bbox)
            masked = compose_masked(batch["image"], mask)
            logits = model(masked, mask)
            preds = logits.argmax(dim=1)
            labs = batch["label"]
            for inst_idx, t, p in zip(idxs.cpu().tolist(),
                                       labs.cpu().tolist(),
                                       preds.cpu().tolist()):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1
                else:
                    confusion[(t, p)] += 1
                    mis_list.append((t, p, inst_idx))

    # ------------------------------------------------------------------
    # Per-class accuracy table
    # ------------------------------------------------------------------
    print("\nPer-class accuracy (sorted by support):")
    print(f"  {'char':>5}  {'support':>8}  {'correct':>8}  {'acc':>6}")
    rows = []
    for c_idx in range(NUM_CLASSES):
        n = class_total.get(c_idx, 0)
        k = class_correct.get(c_idx, 0)
        acc = k / n if n else 0.0
        rows.append((c_idx, n, k, acc))
    rows.sort(key=lambda r: -r[1])
    for c_idx, n, k, acc in rows:
        if n == 0:
            continue
        marker = ""
        if n >= 20 and acc < 0.5:
            marker = "  ←low"
        print(f"  {VOCAB[c_idx]!r:>5}  {n:>8}  {k:>8}  {acc:>6.1%}{marker}")

    # Save per-class table to CSV too.
    csv_path = out_dir / "per_class_acc.csv"
    with open(csv_path, "w") as f:
        f.write("char,support,correct,acc\n")
        for c_idx, n, k, acc in rows:
            f.write(f"{VOCAB[c_idx]},{n},{k},{acc:.4f}\n")
    print(f"\nSaved per-class CSV → {csv_path}")

    # ------------------------------------------------------------------
    # Top confusion pairs (true → predicted)
    # ------------------------------------------------------------------
    print("\nTop confusion pairs (true → predicted):")
    print(f"  {'pair':>8}  {'count':>6}")
    top_conf = sorted(confusion.items(), key=lambda kv: -kv[1])[:30]
    for (t, p), c in top_conf:
        print(f"  {VOCAB[t]!r} → {VOCAB[p]!r}  {c:>6}")

    # ------------------------------------------------------------------
    # Contact sheet of N random misclassified instances
    # ------------------------------------------------------------------
    if args.n_mis > 0 and mis_list:
        rng = random.Random(args.seed)
        rng.shuffle(mis_list)
        chosen = mis_list[: args.n_mis]
        scale = 3
        panels: List[Image.Image] = []
        for t, p, inst_idx in chosen:
            rec, li = full.records[inst_idx]
            img_path = (Path(args.words_dir) / rec["form"] / rec["line"]
                        / f"{rec['word_id']}.png")
            try:
                img = Image.open(img_path).convert("L")
            except Exception:
                continue
            from train_char_recognizer import letterbox
            arr = letterbox(img, 64, 256)  # [H, W]
            arr_uint8 = (arr * 255).astype(np.uint8)
            box_img = Image.fromarray(arr_uint8).convert("RGB").resize(
                (256 * scale, 64 * scale), Image.NEAREST)
            d = ImageDraw.Draw(box_img)
            # Original (init) bbox in red.
            L = rec["letters"][li]
            x1, y1, x2, y2 = L["x1"]*scale, L["y1"]*scale, L["x2"]*scale, L["y2"]*scale
            xa, xb = sorted([x1, x2]); ya, yb = sorted([y1, y2])
            d.rectangle([xa, ya, xb, yb], outline=(220, 0, 0), width=2)
            # Header strip with true → pred + word text.
            header_h = 18
            canvas = Image.new("RGB", (box_img.width, box_img.height + header_h),
                               (255, 255, 255))
            d2 = ImageDraw.Draw(canvas)
            d2.text(
                (4, 2),
                f"{rec['word_id']}  '{rec['text']}'  letter#{li}  "
                f"true={VOCAB[t]!r}  pred={VOCAB[p]!r}",
                fill=(0, 0, 0))
            canvas.paste(box_img, (0, header_h))
            panels.append(canvas)

        # Stack vertically.
        cell_w = max(p.width for p in panels)
        cell_h = max(p.height for p in panels)
        cols = 1
        rows_n = math.ceil(len(panels) / cols)
        pad = 6
        sheet = Image.new(
            "RGB",
            (cell_w * cols + pad * (cols - 1),
             cell_h * rows_n + pad * (rows_n - 1)),
            (255, 255, 255))
        for i, p in enumerate(panels):
            r = i // cols; c = i % cols
            sheet.paste(p, (c * (cell_w + pad), r * (cell_h + pad)))
        out_path = out_dir / "misclassified_samples.png"
        sheet.save(out_path)
        print(f"\nSaved misclassified contact sheet → {out_path}  "
              f"({sheet.size})")


if __name__ == "__main__":
    main()
