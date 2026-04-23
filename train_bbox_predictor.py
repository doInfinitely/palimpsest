#!/usr/bin/env python3
"""Train an autoregressive bbox predictor over letter sequences.

Given a word's character sequence and the boxes of preceding letters,
predict the next letter's box. Teacher-forced on the per-letter bbox
JSONL produced by extract_letter_bboxes.py.

Coords are normalized by a fixed reference line height (default 40 px)
so the model is scale-invariant; at inference the caller re-scales to
canvas pixels.

Box format (normalized):
    x1_rel = (x1 - word_x1) / line_height
    y1_rel = (y1 - word_y_top) / line_height
    x2_rel = (x2 - word_x1) / line_height
    y2_rel = (y2 - word_y_top) / line_height

(word_x1, word_y_top) is the per-word cursor anchor; at inference the
caller supplies its own anchor derived from the canvas cursor.

Usage:
    python3 train_bbox_predictor.py \\
        --bbox-jsonl runs/letter_bboxes_v1.jsonl \\
        --out-dir runs/bbox_predictor_v1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_char_recognizer import VOCAB


# Reserve index 0 for a padding / start-of-sequence token.
PAD = 0
SOS = 1
VOCAB_OFFSET = 2
TOKEN_COUNT = len(VOCAB) + VOCAB_OFFSET


def char_to_tok(c: str) -> int:
    return VOCAB.index(c) + VOCAB_OFFSET


class BboxDataset(Dataset):
    def __init__(self, jsonl_path: Path, line_height: float, max_len: int = 24):
        self.records: List[Dict] = []
        self.line_height = line_height
        self.max_len = max_len
        with open(jsonl_path) as f:
            for raw in f:
                r = json.loads(raw)
                N = len(r["letters"])
                if N < 2 or N > max_len:
                    continue
                self.records.append(r)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        N = len(r["letters"])
        chars = np.full(self.max_len, PAD, dtype=np.int64)
        boxes = np.zeros((self.max_len, 4), dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.bool_)
        x0 = r["word_x1"]
        y0 = r["word_y_top"]
        h = self.line_height
        for i, L in enumerate(r["letters"]):
            chars[i] = char_to_tok(L["char"])
            boxes[i, 0] = (L["x1"] - x0) / h
            boxes[i, 1] = (L["y1"] - y0) / h
            boxes[i, 2] = (L["x2"] - x0) / h
            boxes[i, 3] = (L["y2"] - y0) / h
            mask[i] = True
        return {"chars": chars, "boxes": boxes, "mask": mask, "N": N}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in ("chars", "boxes", "mask"):
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    out["N"] = torch.tensor([b["N"] for b in batch], dtype=torch.long)
    return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[None, : x.size(1)]


class BboxPredictor(nn.Module):
    """Bi-directional char encoder + causal decoder that emits boxes."""

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 3,
                 max_len: int = 24):
        super().__init__()
        self.max_len = max_len
        self.char_embed = nn.Embedding(TOKEN_COUNT, d_model, padding_idx=PAD)
        self.box_in = nn.Linear(4, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len + 1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.char_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.box_decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)

        self.head = nn.Linear(d_model, 4)

    def forward(self, chars: torch.Tensor, prev_boxes: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """chars [B, L], prev_boxes [B, L, 4], pad_mask [B, L] (True=pad).
        Returns predicted boxes [B, L, 4].
        """
        B, L = chars.shape
        char_feat = self.pos(self.char_embed(chars))
        char_feat = self.char_encoder(char_feat, src_key_padding_mask=pad_mask)

        dec_in = self.pos(self.box_in(prev_boxes))
        causal = torch.triu(torch.ones(L, L, device=chars.device, dtype=torch.bool),
                            diagonal=1)
        dec_out = self.box_decoder(
            tgt=dec_in, memory=char_feat,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask,
        )
        return self.head(dec_out)


def build_prev_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Shift boxes right by one; position 0 gets a learned zero vector
    representing the 'cursor' / SOS. Returns [B, L, 4]."""
    B, L, _ = boxes.shape
    prev = torch.zeros_like(boxes)
    prev[:, 1:] = boxes[:, :-1]
    return prev


def train_epoch(model, loader, opt, device, log_every: int = 50):
    model.train()
    tot = 0.0
    n = 0
    for step, batch in enumerate(loader):
        chars = batch["chars"].to(device)
        boxes = batch["boxes"].to(device)
        mask = batch["mask"].to(device)
        prev = build_prev_boxes(boxes)
        pad_mask = ~mask  # True where padded
        pred = model(chars, prev, pad_mask)
        diff = (pred - boxes).abs()
        loss = (diff * mask[:, :, None].float()).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * mask.sum().item()
        n += mask.sum().item()
        if (step + 1) % log_every == 0:
            print(f"    step {step+1:>5}  train L1={loss.item():.4f}")
    return tot / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        chars = batch["chars"].to(device)
        boxes = batch["boxes"].to(device)
        mask = batch["mask"].to(device)
        prev = build_prev_boxes(boxes)
        pad_mask = ~mask
        pred = model(chars, prev, pad_mask)
        diff = (pred - boxes).abs()
        tot += (diff.sum(dim=2) * mask.float()).sum().item()
        n += mask.sum().item() * 4
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--line-height", type=float, default=40.0,
                    help="Reference line height (px) for coord normalization.")
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BboxDataset(Path(args.bbox_jsonl), args.line_height, args.max_len)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    print(f"Dataset: {len(ds)} words → train={n_train} val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    model = BboxPredictor(d_model=args.d_model, nhead=args.nhead,
                          layers=args.layers, max_len=args.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for ep in range(args.epochs):
        print(f"\nEpoch {ep+1}/{args.epochs}")
        train_l = train_epoch(model, train_loader, opt, device)
        val_l = eval_loss(model, val_loader, device)
        sched.step()
        print(f"  train L1={train_l:.4f}  val L1={val_l:.4f}  "
              f"(~{val_l * args.line_height:.2f} px per coord)")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": ep + 1,
            "val_l1": val_l,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_l < best_val:
            best_val = val_l
            torch.save(ckpt, out_dir / "best.pt")

    print(f"\nBest val L1={best_val:.4f}  ({best_val * args.line_height:.2f} px)")


if __name__ == "__main__":
    main()
