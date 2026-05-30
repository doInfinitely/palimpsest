#!/usr/bin/env python3
"""Differentiable bbox refinement: jointly learn classification and per-letter
bounding-box coordinates.

Pipeline per letter instance:
  1. Look up the word's letterboxed image (1 ch, [H=64, W=256]).
  2. Look up the letter's current bbox params (x1, y1, x2, y2), trainable.
  3. Compute a soft mask over [H, W]: per-pixel fraction of pixel area
     inside the rectangle. Differentiable wrt bbox edges.
  4. Compose 2-channel input: [image, mask].
  5. CNN → 78-class softmax over VOCAB.

Phase 1: bbox params FROZEN at initial values from letter_bboxes_v2.jsonl.
  Train CNN until val acc >= --phase1-target-acc (default 0.70).

Phase 2: bbox params UNFROZEN. Joint training with regularization:
  - L2(bbox - bbox_init): keeps bbox close to where it started.
  - Min size penalty: width/height must remain >= --min-size px.

Output: a JSONL with refined per-letter bboxes plus the trained classifier
ckpt.

Usage:
    python3 train_bbox_refiner.py \\
        --bbox-jsonl runs/letter_bboxes_v2.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out-dir runs/bbox_refiner_v1 \\
        --phase1-epochs 30 --phase2-epochs 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_char_recognizer import VOCAB, letterbox


WORD_H, WORD_W = 64, 256
NUM_CLASSES = len(VOCAB)


# ------------------------------------------------------------------
# Differentiable soft bbox mask
# ------------------------------------------------------------------

def soft_bbox_mask(bbox: torch.Tensor, H: int = WORD_H, W: int = WORD_W
                   ) -> torch.Tensor:
    """Returns a [B, 1, H, W] mask; per-pixel fraction inside the bbox.
    bbox shape: [B, 4] = (x1, y1, x2, y2). Coords in pixel space.
    Differentiable wrt all four edges."""
    B = bbox.shape[0]
    x1, y1, x2, y2 = bbox.unbind(-1)
    # Ensure x2 > x1, y2 > y1; clamp by software (training is responsible
    # for keeping params sensible via reg + clipping).
    js = torch.arange(W, dtype=bbox.dtype, device=bbox.device)
    is_ = torch.arange(H, dtype=bbox.dtype, device=bbox.device)
    x_right = torch.minimum(js[None, :] + 1, x2[:, None])
    x_left = torch.maximum(js[None, :], x1[:, None])
    x_overlap = (x_right - x_left).clamp(0.0, 1.0)              # [B, W]
    y_bot = torch.minimum(is_[None, :] + 1, y2[:, None])
    y_top = torch.maximum(is_[None, :], y1[:, None])
    y_overlap = (y_bot - y_top).clamp(0.0, 1.0)                 # [B, H]
    mask = y_overlap[:, :, None] * x_overlap[:, None, :]        # [B, H, W]
    return mask.unsqueeze(1)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class LetterInstanceDataset(Dataset):
    """One example per (word, letter_idx). Returns the letterboxed word
    image, the instance index (for bbox lookup), and the letter label."""

    def __init__(self, jsonl_path: str | Path, words_dir: str | Path):
        self.records: List[Tuple[Dict, int]] = []
        self.words_dir = Path(words_dir)
        with open(jsonl_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw: continue
                r = json.loads(raw)
                for i, L in enumerate(r["letters"]):
                    if L["char"] in VOCAB:
                        self.records.append((r, i))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec, i = self.records[idx]
        p = self.words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
        img = Image.open(p).convert("L")
        arr = letterbox(img, WORD_H, WORD_W)  # [H, W] in [0, 1], 1=white
        L = rec["letters"][i]
        return {
            "instance_idx": idx,
            "image": torch.from_numpy(arr).unsqueeze(0),  # [1, H, W]
            "label": torch.tensor(VOCAB.index(L["char"]), dtype=torch.long),
            "bbox_init": torch.tensor(
                [L["x1"], L["y1"], L["x2"], L["y2"]], dtype=torch.float32),
        }


def collate(batch):
    return {
        "instance_idx": torch.tensor([b["instance_idx"] for b in batch], dtype=torch.long),
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "bbox_init": torch.stack([b["bbox_init"] for b in batch]),
    }


# ------------------------------------------------------------------
# Model: 2-channel CNN
# ------------------------------------------------------------------

class WordContextClassifier(nn.Module):
    """Takes [B, 2, H=64, W=256] (image + mask) → 78-class logits.

    Designed with enough downsampling to make the receptive field cover
    the 64×256 canvas without going to 1×1 too aggressively (keeping
    spatial info for when the masked region is small)."""

    def __init__(self, base_ch: int = 32, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        c = base_ch
        self.features = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1),                         # 64×256
            nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1),               # 32×128
            nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.GroupNorm(8, c * 2), nn.GELU(),
            nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1),       # 16×64
            nn.GroupNorm(8, c * 2), nn.GELU(),
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.GroupNorm(8, c * 4), nn.GELU(),
            nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1),       # 8×32
            nn.GroupNorm(8, c * 4), nn.GELU(),
            nn.Conv2d(c * 4, c * 8, 3, padding=1),
            nn.GroupNorm(8, c * 8), nn.GELU(),
            nn.Conv2d(c * 8, c * 8, 3, stride=2, padding=1),       # 4×16
            nn.GroupNorm(8, c * 8), nn.GELU(),
        )
        self.head = nn.Linear(c * 8, num_classes)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)                       # [B, 2, H, W]
        f = self.features(x)
        pooled = f.mean(dim=(2, 3))
        return self.head(pooled)


# ------------------------------------------------------------------
# Compose masked image
# ------------------------------------------------------------------

def compose_masked(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """image, mask: [B, 1, H, W]. image is in [0, 1] with 1=white.
    Outside the mask we want WHITE (matches inference where surrounding
    letters' bbox regions are visible but only the target letter is in
    its own bbox). masked = 1 - (1 - image) * mask = white outside, image
    inside."""
    return 1.0 - (1.0 - image) * mask


# ------------------------------------------------------------------
# Bbox regularization
# ------------------------------------------------------------------

def bbox_reg_loss(bbox: torch.Tensor, bbox_init: torch.Tensor,
                  min_size: float, max_drift: float) -> torch.Tensor:
    """L2 distance from initial position + soft min-size penalty."""
    drift = (bbox - bbox_init).abs().clamp(max=max_drift)
    drift_pen = (drift ** 2).mean()
    w = (bbox[:, 2] - bbox[:, 0])
    h = (bbox[:, 3] - bbox[:, 1])
    size_pen = F.relu(min_size - w).pow(2).mean() + F.relu(min_size - h).pow(2).mean()
    return drift_pen + size_pen


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def run_epoch(loader, model, bbox_table, opt_cnn, opt_bbox, device,
              freeze_bbox: bool, lambda_reg: float, min_size: float,
              max_drift: float, log_every: int = 200, train: bool = True,
              debug_first_batch: bool = False
              ) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    tot_loss = 0.0
    tot_correct = 0
    n = 0
    bbox_grad_sum = 0.0
    bbox_grad_n = 0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        idxs = batch["instance_idx"]
        if freeze_bbox:
            with torch.no_grad():
                bbox = bbox_table(idxs)
        else:
            bbox = bbox_table(idxs)
        mask = soft_bbox_mask(bbox)                                # [B,1,H,W]
        masked = compose_masked(batch["image"], mask)
        logits = model(masked, mask)
        ce = F.cross_entropy(logits, batch["label"], label_smoothing=0.1)

        if train:
            if not freeze_bbox and opt_bbox is not None:
                reg = bbox_reg_loss(bbox, batch["bbox_init"], min_size, max_drift)
                loss = ce + lambda_reg * reg
            else:
                loss = ce

            opt_cnn.zero_grad(set_to_none=True)
            if opt_bbox is not None:
                opt_bbox.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not freeze_bbox and opt_bbox is not None:
                # Log gradient on the rows that were just touched.
                with torch.no_grad():
                    g = bbox_table.weight.grad
                    if g is not None:
                        sel = g[idxs]                              # [B, 4]
                        bbox_grad_sum += float(sel.norm(dim=-1).sum())
                        bbox_grad_n += sel.shape[0]
                        if debug_first_batch and step == 0:
                            per_edge = sel.abs().mean(dim=0)
                            print(f"    [debug] bbox grad row-norm mean="
                                  f"{sel.norm(dim=-1).mean().item():.3e}  "
                                  f"per-edge |g| ="
                                  f" x1={per_edge[0].item():.3e}"
                                  f" y1={per_edge[1].item():.3e}"
                                  f" x2={per_edge[2].item():.3e}"
                                  f" y2={per_edge[3].item():.3e}")
            opt_cnn.step()
            if not freeze_bbox and opt_bbox is not None:
                opt_bbox.step()

        bs = idxs.shape[0]
        tot_loss += float(ce) * bs
        tot_correct += int((logits.argmax(dim=1) == batch["label"]).sum())
        n += bs
        if train and (step + 1) % log_every == 0:
            print(f"    step {step+1:>5}  ce={tot_loss/n:.4f}  acc={tot_correct/n:.4f}")
    out = {"loss": tot_loss / max(n, 1), "acc": tot_correct / max(n, 1)}
    if bbox_grad_n > 0:
        out["bbox_grad_norm"] = bbox_grad_sum / bbox_grad_n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--phase1-epochs", type=int, default=30)
    ap.add_argument("--phase2-epochs", type=int, default=30)
    ap.add_argument("--phase1-target-acc", type=float, default=0.70,
                    help="Stop phase 1 when val acc >= this (or after phase1-epochs).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-cnn", type=float, default=3e-4)
    ap.add_argument("--lr-cnn-phase2", type=float, default=1e-4,
                    help="CNN LR during phase 2 (lower than phase 1 to slow "
                         "memorization while bbox catches up).")
    ap.add_argument("--lr-bbox", type=float, default=0.1,
                    help="Adam LR on bbox params. With Adam, effective step is "
                         "~lr per visit regardless of grad magnitude.")
    ap.add_argument("--lambda-reg", type=float, default=0.001)
    ap.add_argument("--load-phase1", type=str, default=None,
                    help="Path to a phase1 checkpoint to skip phase 1 entirely "
                         "(e.g. runs/bbox_refiner_v1/p1_best.pt).")
    ap.add_argument("--debug-first-batch", action="store_true",
                    help="Print bbox gradient norms for the first batch of "
                         "each phase-2 epoch.")
    ap.add_argument("--min-size", type=float, default=4.0,
                    help="Minimum width/height for bbox in pixels.")
    ap.add_argument("--max-drift", type=float, default=20.0,
                    help="Cap on per-edge drift from init (penalty saturates beyond).")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = LetterInstanceDataset(args.bbox_jsonl, args.words_dir)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Dataset: {len(full)} letter instances → train={n_train} val={n_val}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=True)

    # Build a learnable bbox table: one (x1,y1,x2,y2) per letter instance.
    bbox_init_full = torch.zeros(len(full), 4, dtype=torch.float32)
    for i, (rec, li) in enumerate(full.records):
        L = rec["letters"][li]
        bbox_init_full[i] = torch.tensor([L["x1"], L["y1"], L["x2"], L["y2"]])
    bbox_table = nn.Embedding(len(full), 4).to(device)
    with torch.no_grad():
        bbox_table.weight.copy_(bbox_init_full)
    print(f"Bbox table: {len(full)} × 4 = {len(full)*4:,} params")

    model = WordContextClassifier(base_ch=args.base_ch).to(device)
    opt_cnn = torch.optim.AdamW(model.parameters(), lr=args.lr_cnn,
                                weight_decay=0.01)
    opt_bbox = torch.optim.Adam([bbox_table.weight], lr=args.lr_bbox,
                                betas=(0.9, 0.999), eps=1e-8)

    # Phase 1: bbox frozen, train CNN. Optionally skip if --load-phase1.
    best_acc = 0.0
    if args.load_phase1 is not None:
        print(f"\n=== Skipping Phase 1, loading {args.load_phase1} ===")
        ckpt = torch.load(args.load_phase1, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        bbox_table.load_state_dict(ckpt["bbox_table_state_dict"])
        best_acc = ckpt.get("val", {}).get("acc", 0.0)
        print(f"  loaded phase1 model (val acc={best_acc:.4f})")
    else:
        print("\n=== Phase 1: bbox frozen ===")
        for ep in range(args.phase1_epochs):
            t0 = time.time()
            train_m = run_epoch(train_loader, model, bbox_table, opt_cnn, None,
                                device, freeze_bbox=True, lambda_reg=0.0,
                                min_size=args.min_size, max_drift=args.max_drift,
                                train=True)
            val_m = run_epoch(val_loader, model, bbox_table, opt_cnn, None,
                              device, freeze_bbox=True, lambda_reg=0.0,
                              min_size=args.min_size, max_drift=args.max_drift,
                              train=False)
            dt = time.time() - t0
            print(f"P1 ep {ep+1}/{args.phase1_epochs}  "
                  f"train ce={train_m['loss']:.4f} acc={train_m['acc']:.4f}  "
                  f"val ce={val_m['loss']:.4f} acc={val_m['acc']:.4f}  ({dt:.1f}s)")
            ckpt = {"model_state_dict": model.state_dict(),
                    "bbox_table_state_dict": bbox_table.state_dict(),
                    "args": vars(args), "phase": 1, "epoch": ep + 1, "val": val_m}
            torch.save(ckpt, out_dir / "p1_last.pt")
            if val_m["acc"] > best_acc:
                best_acc = val_m["acc"]
                torch.save(ckpt, out_dir / "p1_best.pt")
            if val_m["acc"] >= args.phase1_target_acc:
                print(f"  → reached target acc {args.phase1_target_acc:.2f}; advancing to phase 2")
                break

    # Phase 2: unfreeze bbox params. Lower CNN lr so it doesn't run away.
    for g in opt_cnn.param_groups:
        g["lr"] = args.lr_cnn_phase2
    print(f"\n=== Phase 2: joint (bbox unfrozen)  "
          f"cnn_lr={args.lr_cnn_phase2}  bbox_lr={args.lr_bbox} (Adam) ===")
    best_acc2 = 0.0
    for ep in range(args.phase2_epochs):
        t0 = time.time()
        train_m = run_epoch(train_loader, model, bbox_table, opt_cnn, opt_bbox,
                            device, freeze_bbox=False, lambda_reg=args.lambda_reg,
                            min_size=args.min_size, max_drift=args.max_drift,
                            train=True,
                            debug_first_batch=args.debug_first_batch)
        val_m = run_epoch(val_loader, model, bbox_table, opt_cnn, opt_bbox,
                          device, freeze_bbox=False, lambda_reg=0.0,
                          min_size=args.min_size, max_drift=args.max_drift,
                          train=False)
        dt = time.time() - t0
        bgn = train_m.get("bbox_grad_norm", float("nan"))
        print(f"P2 ep {ep+1}/{args.phase2_epochs}  "
              f"train ce={train_m['loss']:.4f} acc={train_m['acc']:.4f}  "
              f"val ce={val_m['loss']:.4f} acc={val_m['acc']:.4f}  "
              f"bbox_grad={bgn:.3e}  ({dt:.1f}s)")
        # Track bbox drift from init: mean and 95th percentile (so a few
        # well-moved bboxes don't get drowned out by the mean).
        with torch.no_grad():
            d = (bbox_table.weight.cpu() - bbox_init_full).abs()
            drift_mean = d.mean().item()
            drift_p95 = d.quantile(0.95).item()
            drift_max = d.max().item()
        print(f"   bbox drift: mean={drift_mean:.2f}  "
              f"p95={drift_p95:.2f}  max={drift_max:.2f} px")
        ckpt = {"model_state_dict": model.state_dict(),
                "bbox_table_state_dict": bbox_table.state_dict(),
                "bbox_init": bbox_init_full,
                "args": vars(args), "phase": 2, "epoch": ep + 1, "val": val_m}
        torch.save(ckpt, out_dir / "p2_last.pt")
        if val_m["acc"] > best_acc2:
            best_acc2 = val_m["acc"]
            torch.save(ckpt, out_dir / "p2_best.pt")

    # Emit refined bboxes JSONL with same record structure as input but
    # with letters[i].x1/.y1/.x2/.y2 replaced by trained values.
    print(f"\nWriting refined bboxes...")
    refined_path = out_dir / "letter_bboxes_refined.jsonl"
    bboxes_final = bbox_table.weight.detach().cpu().numpy()
    by_word: Dict[str, Dict] = {}
    for inst_idx, (rec, li) in enumerate(full.records):
        wid = rec["word_id"]
        if wid not in by_word:
            by_word[wid] = json.loads(json.dumps(rec))  # deep copy
        bb = bboxes_final[inst_idx]
        by_word[wid]["letters"][li]["x1"] = float(bb[0])
        by_word[wid]["letters"][li]["y1"] = float(bb[1])
        by_word[wid]["letters"][li]["x2"] = float(bb[2])
        by_word[wid]["letters"][li]["y2"] = float(bb[3])
    with open(refined_path, "w") as f:
        for wid in by_word:
            f.write(json.dumps(by_word[wid]) + "\n")
    print(f"Saved refined bboxes: {refined_path}")
    print(f"Best phase1 val acc: {best_acc:.4f}")
    print(f"Best phase2 val acc: {best_acc2:.4f}")


if __name__ == "__main__":
    main()
