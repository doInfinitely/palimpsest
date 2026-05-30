#!/usr/bin/env python3
"""Train a single-letter classifier on IAM letter crops.

For each entry in the per-letter bbox JSONL, we load the letterboxed
64×256 word image, crop the letter's bbox, and letterbox it into a
square retina (default 64×64). The classifier is a small CNN with GAP
→ softmax over VOCAB.

Used both as a standalone evaluator of letter legibility AND as a
frozen cross-entropy signal during letter-infill finetuning.

Usage:
    python3 train_letter_classifier.py \\
        --bbox-jsonl runs/letter_bboxes_v1.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out-dir runs/letter_clf_v1
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


def pad_square_letterbox(arr: np.ndarray, retina: int) -> np.ndarray:
    """Letterbox a grayscale float array into a retina×retina white-padded
    square. Input values expected in [0, 1] with 1=white."""
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.ones((retina, retina), dtype=np.float32)
    scale = min(retina / h, retina / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (retina, retina), 255)
    canvas.paste(img, ((retina - nw) // 2, (retina - nh) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


class LetterCropDataset(Dataset):
    """Materializes one letter crop per __getitem__ by sampling a
    (word, letter_index) pair. Each word contributes N examples."""

    def __init__(
        self,
        jsonl_path: str | Path,
        words_dir: str | Path,
        retina: int = 64,
        augment: bool = False,
        bbox_pad: float = 2.0,
    ) -> None:
        self.retina = retina
        self.augment = augment
        self.bbox_pad = bbox_pad
        self.words_dir = Path(words_dir)
        self.index: List[Tuple[Dict[str, Any], int]] = []  # (word_rec, letter_idx)
        with open(jsonl_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                r = json.loads(raw)
                for i, L in enumerate(r["letters"]):
                    if L["char"] in VOCAB:
                        self.index.append((r, i))

    def __len__(self) -> int:
        return len(self.index)

    def _load_word(self, rec: Dict[str, Any]) -> np.ndarray:
        p = self.words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
        img = Image.open(p).convert("L")
        return letterbox(img, WORD_H, WORD_W)  # 1=white

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec, i = self.index[idx]
        word_arr = self._load_word(rec)
        L = rec["letters"][i]
        pad = self.bbox_pad
        if self.augment:
            # Stronger crop jitter: ±3 px in pad and ±3 px in offsets.
            pad = pad + random.uniform(-3.0, 3.0)
            jitter_x = random.uniform(-3.0, 3.0)
            jitter_y = random.uniform(-3.0, 3.0)
        else:
            jitter_x = jitter_y = 0.0
        x1 = max(0.0, L["x1"] - pad + jitter_x)
        y1 = max(0.0, L["y1"] - pad + jitter_y)
        x2 = min(float(WORD_W), L["x2"] + pad + jitter_x)
        y2 = min(float(WORD_H), L["y2"] + pad + jitter_y)
        ix1, iy1 = int(np.floor(x1)), int(np.floor(y1))
        ix2, iy2 = int(np.ceil(x2)), int(np.ceil(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            crop = np.ones((1, 1), dtype=np.float32)
        else:
            crop = word_arr[iy1:iy2, ix1:ix2]

        retina = pad_square_letterbox(crop, self.retina)

        if self.augment:
            # Brightness/contrast jitter (stronger).
            if random.random() < 0.7:
                b = random.uniform(-0.18, 0.18)
                c = 1.0 + random.uniform(-0.25, 0.25)
                retina = np.clip((retina - 0.5) * c + 0.5 + b, 0, 1)
            # Gaussian noise (stronger).
            if random.random() < 0.5:
                sigma = random.uniform(0.01, 0.05)
                retina = np.clip(retina + np.random.normal(0, sigma, retina.shape), 0, 1)
            # Small random patch erase (more frequent, larger).
            if random.random() < 0.4:
                ph = random.randint(4, 16)
                pw = random.randint(4, 16)
                py = random.randint(0, self.retina - ph)
                px = random.randint(0, self.retina - pw)
                retina[py:py + ph, px:px + pw] = random.uniform(0.0, 1.0)
            # Small rotation ±10° via PIL bilinear (white background fill).
            if random.random() < 0.5:
                angle = random.uniform(-10.0, 10.0)
                pil = Image.fromarray((retina * 255).astype(np.uint8), mode="L")
                pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
                retina = np.asarray(pil, dtype=np.float32) / 255.0

        label = VOCAB.index(L["char"])
        return {
            "img": torch.from_numpy(retina.astype(np.float32)).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate(batch):
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


class LetterClassifier(nn.Module):
    """Small CNN for retina-sized letter crops."""

    def __init__(self, retina: int = 64, base_ch: int = 32,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),                      # 64
            nn.GroupNorm(8, base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),      # 32
            nn.GroupNorm(8, base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2), nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1),  # 16
            nn.GroupNorm(8, base_ch * 2), nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4), nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),  # 8
            nn.GroupNorm(8, base_ch * 4), nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, padding=1),
            nn.GroupNorm(8, base_ch * 8), nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(base_ch * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        pooled = f.mean(dim=(2, 3))
        pooled = self.dropout(pooled)
        return self.head(pooled)


def train_epoch(model, loader, opt, device, log_every=200):
    model.train()
    tot_loss = 0.0
    tot_correct = 0
    n = 0
    for step, batch in enumerate(loader):
        img = batch["img"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)
        logits = model(img)
        loss = F.cross_entropy(logits, lab, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = img.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (logits.argmax(dim=1) == lab).sum().item()
        n += bs
        if (step + 1) % log_every == 0:
            print(f"    step {step+1:>5}  loss={tot_loss/n:.4f}  acc={tot_correct/n:.4f}")
    return {"loss": tot_loss / max(n, 1), "acc": tot_correct / max(n, 1)}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    tot_loss = 0.0
    tot_correct = 0
    n = 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)
        logits = model(img)
        loss = F.cross_entropy(logits, lab)
        bs = img.size(0)
        tot_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        tot_correct += (pred == lab).sum().item()
        n += bs
        # Per-class counts.
        for c in range(NUM_CLASSES):
            mask = lab == c
            if mask.any():
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (pred[mask] == c).sum().item()
    micro = tot_correct / max(n, 1)
    present = per_class_total > 0
    if present.any():
        per_class_acc = per_class_correct[present] / per_class_total[present]
        macro = float(per_class_acc.mean())
    else:
        macro = 0.0
    return {"loss": tot_loss / max(n, 1), "acc": micro, "macro_acc": macro}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--retina", type=int, default=64)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance-classes", action="store_true",
                    help="Use a WeightedRandomSampler that balances class "
                         "frequency in each epoch (rare classes oversampled).")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate before the classifier head.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = LetterCropDataset(args.bbox_jsonl, args.words_dir,
                             retina=args.retina, augment=True)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Eval dataset uses no augment.
    val_ds_noaug = LetterCropDataset(args.bbox_jsonl, args.words_dir,
                                     retina=args.retina, augment=False)
    val_ds_noaug.index = [val_ds_noaug.index[i] for i in val_ds.indices]

    print(f"Dataset: {len(full)} letters → train={n_train} val={n_val}")

    if args.balance_classes:
        # Class-balanced sampler: weight each train example by 1 / freq[label].
        # Compute class counts from the underlying dataset's index.
        train_indices = train_ds.indices
        labels = np.array([VOCAB.index(full.index[i][0]["letters"][full.index[i][1]]["char"])
                           for i in train_indices], dtype=np.int64)
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        # Avoid div-by-zero; classes with 0 examples don't contribute anyway.
        weights_per_class = np.where(counts > 0, 1.0 / counts, 0.0)
        sample_weights = weights_per_class[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_indices), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  collate_fn=collate, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        present = int((counts > 0).sum())
        print(f"Class balance: {present}/{NUM_CLASSES} classes present.  "
              f"Most freq class count={int(counts.max())}, least={int(counts[counts > 0].min())}")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds_noaug, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=True)

    model = LetterClassifier(retina=args.retina, base_ch=args.base_ch,
                             dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        print(f"\nEpoch {ep+1}/{args.epochs}")
        train_m = train_epoch(model, train_loader, opt, device)
        val_m = eval_epoch(model, val_loader, device)
        sched.step()
        dt = time.time() - t0
        print(f"  train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f}  "
              f"val loss={val_m['loss']:.4f} micro={val_m['acc']:.4f} "
              f"macro={val_m['macro_acc']:.4f}  ({dt:.1f}s)")
        ckpt = {"model_state_dict": model.state_dict(), "args": vars(args),
                "epoch": ep + 1, "val": val_m, "vocab": VOCAB}
        torch.save(ckpt, out_dir / "last.pt")
        # Best by macro-accuracy when class-balanced training is on; else micro.
        score = val_m["macro_acc"] if args.balance_classes else val_m["acc"]
        if score > best_acc:
            best_acc = score
            torch.save(ckpt, out_dir / "best.pt")

    print(f"\nBest val acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
