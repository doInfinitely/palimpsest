#!/usr/bin/env python3
"""Train a bag-of-chars recognizer on IAM word crops.

Used as the scoring head for occlusion-based character bbox recovery.
The recognizer is fully convolutional with a small receptive field so
that it produces locally-grounded character evidence; global-average
pooling aggregates spatial votes into a multi-label prediction.

Usage:
    python3 train_char_recognizer.py \
        --words-txt data/iam_words/iam_words/words.txt \
        --words-dir data/iam_words/iam_words/words \
        --out-dir runs/char_recog_v1
"""
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


VOCAB = list(string.ascii_letters + string.digits + ".,'\"-?!;():#&*+/")
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)


def parse_words_txt(path: Path, words_dir: Path | None = None) -> List[Dict]:
    """Returns list of {word_id, form, line, text}.

    If words_dir is given, drops records whose image file is missing or empty.
    """
    out = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 9:
                continue
            wid, status = parts[0], parts[1]
            if status != "ok":
                continue
            text = " ".join(parts[8:])
            if not text or any(c not in CHAR_TO_IDX for c in text):
                continue
            form = wid.split("-")[0]
            line_dir = "-".join(wid.split("-")[:2])
            if words_dir is not None:
                p = words_dir / form / line_dir / f"{wid}.png"
                try:
                    if p.stat().st_size == 0:
                        continue
                except FileNotFoundError:
                    continue
            out.append({"word_id": wid, "form": form, "line": line_dir, "text": text})
    return out


def letterbox(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """Aspect-preserving fit into target_h × target_w on white background."""
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (target_w, target_h), 255)
    canvas.paste(img, ((target_w - nw) // 2, (target_h - nh) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr  # 1.0 = white background, 0.0 = ink


class WordRecogDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        words_dir: Path,
        target_h: int = 64,
        target_w: int = 256,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.words_dir = words_dir
        self.target_h = target_h
        self.target_w = target_w
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _path_for(self, rec: Dict) -> Path:
        return self.words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img = Image.open(self._path_for(rec)).convert("L")
        arr = letterbox(img, self.target_h, self.target_w)

        if self.augment:
            if random.random() < 0.5:
                bright = random.uniform(-0.1, 0.1)
                contrast = random.uniform(0.85, 1.15)
                arr = ((arr - 0.5) * contrast + 0.5 + bright).clip(0, 1)

        ink = 1.0 - arr  # 1.0 = ink, 0.0 = background
        x = torch.from_numpy(ink).unsqueeze(0).float()  # [1, H, W]

        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for c in rec["text"]:
            target[CHAR_TO_IDX[c]] = 1.0

        return x, target


class LocalCharRecognizer(nn.Module):
    """Fully-conv CNN with ~31-pixel receptive field, global-avg-pool head.

    Input:  [B, 1, 64, 256]
    Output: [B, NUM_CLASSES] sigmoid logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base_ch: int = 32,
                 final_stride: int = 2) -> None:
        super().__init__()
        # final_stride=2 → total stride 16, feature map 4×16 (v1–v3).
        # final_stride=1 → total stride 8, feature map 8×32, same RF=31 (v4+).
        self.features = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, stride=2, padding=1),       # RF 3, 64→32
            nn.GroupNorm(8, base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),   # RF 7, 32→16
            nn.GroupNorm(8, base_ch * 2), nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),  # RF 15
            nn.GroupNorm(8, base_ch * 4), nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=final_stride, padding=1),  # RF 31
            nn.GroupNorm(8, base_ch * 4), nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 1),  # 1x1 mixing
            nn.GELU(),
        )
        self.head = nn.Conv2d(base_ch * 4, num_classes, 1)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.head(f)  # [B, C, H', W'] per-location logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.feature_map(x)
        return m.amax(dim=(2, 3))  # max-pool: a char is "present" if any location fires


def split_records(records: List[Dict], val_frac: float = 0.05, seed: int = 0) -> Tuple[List, List]:
    """Form-level split so the same writer isn't in train and val."""
    forms = sorted({r["form"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(forms)
    n_val = max(1, int(len(forms) * val_frac))
    val_forms = set(forms[:n_val])
    train = [r for r in records if r["form"] not in val_forms]
    val = [r for r in records if r["form"] in val_forms]
    return train, val


def evaluate(model: LocalCharRecognizer, loader: DataLoader, device) -> Dict:
    model.eval()
    bce_sum = 0.0
    tp = fp = fn = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            bce_sum += float(F.binary_cross_entropy_with_logits(logits, y, reduction="sum"))
            n += y.numel()
            pred = (torch.sigmoid(logits) > 0.5).float()
            tp += int((pred * y).sum())
            fp += int((pred * (1 - y)).sum())
            fn += int(((1 - pred) * y).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    return {"bce": bce_sum / max(1, n), "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--words-txt", required=True)
    p.add_argument("--words-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--final-stride", type=int, default=2,
                   help="Stride of the last stride-2 conv (2 for v1-v3, 1 for "
                        "v4+ which keeps RF=31 but doubles spatial resolution).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Parsing {args.words_txt}...")
    records = parse_words_txt(Path(args.words_txt), Path(args.words_dir))
    print(f"  {len(records)} OK words with in-vocab transcriptions and readable images")

    train_recs, val_recs = split_records(records)
    print(f"  Train: {len(train_recs)}  Val: {len(val_recs)}")

    words_dir = Path(args.words_dir)
    train_ds = WordRecogDataset(train_recs, words_dir, augment=True)
    val_ds = WordRecogDataset(val_recs, words_dir, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = LocalCharRecognizer(base_ch=args.base_ch,
                                final_stride=args.final_stride).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Recognizer: {n_params:,} params, {NUM_CLASSES} classes")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with (out_dir / "vocab.json").open("w") as f:
        json.dump({"vocab": VOCAB}, f)

    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * x.shape[0]
            n_seen += x.shape[0]
        sched.step()
        train_loss = loss_sum / max(1, n_seen)

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  train_bce={train_loss:.4f}  "
              f"val_bce={val_metrics['bce']:.4f}  P={val_metrics['precision']:.3f}  "
              f"R={val_metrics['recall']:.3f}  F1={val_metrics['f1']:.3f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        history.append({"epoch": epoch, "train_bce": train_loss, "val": val_metrics})
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "vocab": VOCAB,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ** new best F1: {best_f1:.3f}")

    print(f"\nDone. Best val F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()
