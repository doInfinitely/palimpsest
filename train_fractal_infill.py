#!/usr/bin/env python3
"""
Train a single-pass character infill model (multi-scale).

UNet with a fixed 256x256 retina that predicts an infill delta in one pass.
Supports character, word, and line-level infill with balanced sampling.

Usage:
    python train_fractal_infill.py \
        --train-jsonl data/synth_v1/infill_fractal/fractal_infill.jsonl \
        --val-jsonl data/synth_v1/infill_fractal/fractal_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir runs/infill_v4
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_character_infill import (
    set_seed,
    read_jsonl,
    load_gray_image,
    load_mask_image,
    text_to_byte_tensor,
    pad_1d_long,
    ConvBlock,
    SelfAttention2d,
    FiLMModulation,
    CharacterInfillDataset,
    collate_infill,
    move_batch_to_device,
)

RETINA_SIZE = 256


# ============================================================
# Multi-scale dataset
# ============================================================

class FractalInfillDataset(Dataset):
    """Loads multi-scale infill samples (character, word, line)."""

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        style_to_index: Optional[Dict[str, int]] = None,
        patch_size: int = RETINA_SIZE,
        augment: bool = False,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index or {}
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _augment(
        self, before: Tensor, after: Tensor, bbox_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if random.random() < 0.5:
            brightness = 1.0 + random.uniform(-0.15, 0.15)
            contrast = 1.0 + random.uniform(-0.15, 0.15)
            mean_val = 0.5
            before = ((before - mean_val) * contrast + mean_val + (brightness - 1.0)).clamp(0, 1)
            after = ((after - mean_val) * contrast + mean_val + (brightness - 1.0)).clamp(0, 1)
        if random.random() < 0.5:
            before = before.flip(-1)
            after = after.flip(-1)
            bbox_mask = bbox_mask.flip(-1)
        return before, after, bbox_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        sz = self.patch_size

        before = load_gray_image(self.image_root / rec["before_patch_ref"], sz)
        after = load_gray_image(self.image_root / rec["after_patch_ref"], sz)
        bbox_mask = load_mask_image(self.image_root / rec["bbox_mask_ref"], sz)

        char_tokens = text_to_byte_tensor(rec.get("char_text"), max_len=16)
        style_id = rec.get("style_id", "__unk__")
        style_index = self.style_to_index.get(str(style_id), 0)

        target_bbox = rec.get("target_bbox_parent_norm_cxcywh", [0.5, 0.5, 0.5, 0.5])

        if self.augment:
            before, after, bbox_mask = self._augment(before, after, bbox_mask)

        return {
            "record_id": rec["record_id"],
            "level": rec.get("level", "character"),
            "before": before,
            "after": after,
            "bbox_mask": bbox_mask,
            "delta": after - before,
            "char_tokens": char_tokens,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "bbox": torch.tensor(target_bbox, dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_fractal(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_id": [b["record_id"] for b in batch],
        "level": [b["level"] for b in batch],
        "before": torch.stack([b["before"] for b in batch]),
        "after": torch.stack([b["after"] for b in batch]),
        "bbox_mask": torch.stack([b["bbox_mask"] for b in batch]),
        "delta": torch.stack([b["delta"] for b in batch]),
        "char_tokens": pad_1d_long([b["char_tokens"] for b in batch]),
        "char_lengths": torch.tensor([b["char_tokens"].numel() for b in batch], dtype=torch.long),
        "style_index": torch.stack([b["style_index"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "confidence": torch.stack([b["confidence"] for b in batch]),
    }


# ============================================================
# Infiller Model
# ============================================================

class FractalInfiller(nn.Module):
    """Single-pass infill UNet with FiLM conditioning."""

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 96,
        cond_dim: int = 128,
        vocab_size: int = 256,
        num_styles: int = 64,
    ) -> None:
        super().__init__()
        ch = base_ch

        # Conditioning
        self.char_embed = nn.Embedding(vocab_size, cond_dim)
        self.style_embed = nn.Embedding(num_styles, cond_dim)
        self.bbox_proj = nn.Linear(4, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 3, cond_dim)

        # Encoder
        self.enc1 = ConvBlock(in_ch, ch)
        self.film1 = FiLMModulation(cond_dim, ch)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(ch, ch * 2)
        self.film2 = FiLMModulation(cond_dim, ch * 2)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.film3 = FiLMModulation(cond_dim, ch * 4)
        self.down3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.film4 = FiLMModulation(cond_dim, ch * 8)
        self.down4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch * 8, ch * 8)
        self.film_bn = FiLMModulation(cond_dim, ch * 8)
        self.attn_bn = SelfAttention2d(ch * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 4)
        self.film_d4 = FiLMModulation(cond_dim, ch * 4)

        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 2)
        self.film_d3 = FiLMModulation(cond_dim, ch * 2)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch)
        self.film_d2 = FiLMModulation(cond_dim, ch)

        self.up1 = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)
        self.film_d1 = FiLMModulation(cond_dim, ch)

        self.out_conv = nn.Conv2d(ch, out_ch, 1)

    def _get_cond(self, char_tokens, char_lengths, style_index, bbox):
        if char_tokens.shape[1] > 0:
            char_emb = self.char_embed(char_tokens)
            mask = torch.arange(char_tokens.shape[1], device=char_tokens.device).unsqueeze(0) < char_lengths.unsqueeze(1)
            char_emb = (char_emb * mask.unsqueeze(-1).float()).sum(dim=1) / char_lengths.unsqueeze(-1).clamp_min(1).float()
        else:
            char_emb = torch.zeros(char_tokens.shape[0], self.char_embed.embedding_dim, device=char_tokens.device)
        style_emb = self.style_embed(style_index)
        bbox_emb = self.bbox_proj(bbox)
        return self.cond_proj(torch.cat([char_emb, style_emb, bbox_emb], dim=-1))

    def _encode(self, before, bbox_mask, cond):
        x = torch.cat([before, bbox_mask], dim=1)
        e1 = self.film1(self.enc1(x), cond)
        e2 = self.film2(self.enc2(self.down1(e1)), cond)
        e3 = self.film3(self.enc3(self.down2(e2)), cond)
        e4 = self.film4(self.enc4(self.down3(e3)), cond)
        bn = self.bottleneck(self.down4(e4))
        bn = self.film_bn(bn, cond)
        bn = self.attn_bn(bn)
        return e1, e2, e3, e4, bn

    def _decode(self, e1, e2, e3, e4, bn, cond):
        d4 = self.film_d4(self.dec4(torch.cat([self.up4(bn), e4], dim=1)), cond)
        d3 = self.film_d3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)), cond)
        d2 = self.film_d2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), cond)
        d1 = self.film_d1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), cond)
        return self.out_conv(d1)

    def forward_infill(self, before, bbox_mask, char_tokens, char_lengths, style_index, bbox):
        """Single-pass infill. Returns pred_delta."""
        cond = self._get_cond(char_tokens, char_lengths, style_index, bbox)
        e1, e2, e3, e4, bn = self._encode(before, bbox_mask, cond)
        return self._decode(e1, e2, e3, e4, bn, cond)


# ============================================================
# Loss
# ============================================================

def compute_infill_loss(
    pred_delta: Tensor,
    target_delta: Tensor,
    bbox_mask: Tensor,
    confidence: Tensor,
    halo_weight: float = 0.2,
) -> Tuple[Tensor, Dict[str, float]]:
    """Standard infill L1 loss."""
    inside = bbox_mask
    outside = 1.0 - bbox_mask
    l1 = torch.abs(pred_delta - target_delta)

    inside_count = inside.sum(dim=(1, 2, 3)).clamp_min(1.0)
    outside_count = outside.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_in = (l1 * inside).sum(dim=(1, 2, 3)) / inside_count
    per_out = (l1 * outside).sum(dim=(1, 2, 3)) / outside_count
    per_sample = per_in + halo_weight * per_out

    loss = (per_sample * confidence).sum() / confidence.sum().clamp_min(1.0)

    with torch.no_grad():
        inside_l1 = (l1 * inside).sum() / inside.sum().clamp_min(1.0)
        outside_l1 = (l1 * outside).sum() / outside.sum().clamp_min(1.0)

    return loss, {
        "inside_l1": float(inside_l1.cpu()),
        "outside_l1": float(outside_l1.cpu()),
    }


# ============================================================
# Training / validation steps
# ============================================================

def train_step(
    model: FractalInfiller,
    batch: Dict[str, Any],
    device: torch.device,
    halo_weight: float,
) -> Tuple[Tensor, Dict[str, float]]:
    pred_delta = model.forward_infill(
        batch["before"], batch["bbox_mask"],
        batch["char_tokens"], batch["char_lengths"],
        batch["style_index"], batch["bbox"],
    )
    loss, infill_metrics = compute_infill_loss(
        pred_delta, batch["delta"], batch["bbox_mask"],
        batch["confidence"], halo_weight,
    )
    metrics = {
        "total": float(loss.detach().cpu()),
        "infill_loss": float(loss.detach().cpu()),
        **infill_metrics,
    }
    return loss, metrics


@torch.no_grad()
def val_step(
    model: FractalInfiller,
    batch: Dict[str, Any],
    device: torch.device,
    halo_weight: float,
) -> Dict[str, float]:
    pred_delta = model.forward_infill(
        batch["before"], batch["bbox_mask"],
        batch["char_tokens"], batch["char_lengths"],
        batch["style_index"], batch["bbox"],
    )
    loss, infill_metrics = compute_infill_loss(
        pred_delta, batch["delta"], batch["bbox_mask"],
        batch["confidence"], halo_weight,
    )
    return {
        "total": float(loss.cpu()),
        "infill_loss": float(loss.cpu()),
        **infill_metrics,
    }


# ============================================================
# Epoch runners
# ============================================================

def run_epoch(
    model: FractalInfiller,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    halo_weight: float,
    scheduler: Optional[Any] = None,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    metric_keys = ["total", "infill_loss", "inside_l1", "outside_l1"]
    sums = {k: 0.0 for k in metric_keys}
    sums["count"] = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if train:
            loss, metrics = train_step(model, batch, device, halo_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            metrics = val_step(model, batch, device, halo_weight)

        bs = batch["before"].shape[0]
        for k in metric_keys:
            sums[k] += metrics.get(k, 0.0) * bs
        sums["count"] += bs

    if train and scheduler is not None:
        scheduler.step()

    count = max(1, sums["count"])
    return {k: (v / count if k != "count" else v) for k, v in sums.items()}


def save_checkpoint(
    out_dir: Path, epoch: int, model: FractalInfiller,
    optimizer: torch.optim.Optimizer, train_metrics, val_metrics,
    args: argparse.Namespace, is_best: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    ckpt_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(ckpt, ckpt_path)
    if is_best:
        torch.save(ckpt, out_dir / "best.pt")
    return ckpt_path


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train single-pass infill model")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--base-ch", type=int, default=96)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--num-styles", type=int, default=64)

    p.add_argument("--halo-weight", type=float, default=0.2)

    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt file to resume training from")

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {args.train_jsonl}...")
    train_records = read_jsonl(args.train_jsonl)
    print(f"Loading validation data from {args.val_jsonl}...")
    val_records = read_jsonl(args.val_jsonl)
    print(f"  Train: {len(train_records)} records, Val: {len(val_records)} records")

    # Build style vocab
    style_ids = {"__unk__"}
    for rec in train_records + val_records:
        sid = rec.get("style_id", "__unk__")
        style_ids.add(str(sid))
    style_to_index = {s: i for i, s in enumerate(sorted(style_ids))}
    num_styles = len(style_to_index)
    print(f"  {num_styles} styles")

    with (out_dir / "style_to_index.json").open("w") as f:
        json.dump(style_to_index, f, indent=2)

    # Detect data format
    has_levels = "level" in train_records[0] if train_records else False
    if has_levels:
        from collections import Counter
        level_counts = Counter(r.get("level", "character") for r in train_records)
        print(f"  Levels: {dict(level_counts)}")

        train_ds = FractalInfillDataset(
            jsonl_path=args.train_jsonl,
            image_root=args.image_root,
            style_to_index=style_to_index,
            augment=True,
        )
        val_ds = FractalInfillDataset(
            jsonl_path=args.val_jsonl,
            image_root=args.image_root,
            style_to_index=style_to_index,
        )
        collate_fn = collate_fractal

        # Balanced sampling: 50/50 character vs word, drop line
        n_char = level_counts.get("character", 0)
        n_word = level_counts.get("word", 0)
        print(f"  Balanced sampling: {n_char} char, {n_word} word (50/50 target)")

        weights = []
        for rec in train_records:
            lvl = rec.get("level", "character")
            if lvl == "line":
                weights.append(0.0)
            elif lvl == "word":
                weights.append(1.0 / max(n_word, 1))
            else:
                weights.append(1.0 / max(n_char, 1))
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )
    else:
        print("  Legacy character-only data")
        train_ds = CharacterInfillDataset(
            jsonl_path=args.train_jsonl,
            image_root=args.image_root,
            style_to_index=style_to_index,
            augment=True,
        )
        val_ds = CharacterInfillDataset(
            jsonl_path=args.val_jsonl,
            image_root=args.image_root,
            style_to_index=style_to_index,
        )
        collate_fn = collate_infill

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = FractalInfiller(
        in_ch=2, out_ch=1,
        base_ch=args.base_ch,
        cond_dim=args.cond_dim,
        vocab_size=256,
        num_styles=num_styles,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    best_val_infill = float("inf")
    start_epoch = 1
    history: List[Dict[str, Any]] = []
    log_path = out_dir / "train.log"

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Only load optimizer if architecture matches (fresh start for new base_ch)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            print("  Optimizer state mismatch — starting optimizer fresh")
        start_epoch = ckpt["epoch"] + 1
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with hist_path.open() as f:
                history = json.load(f)
            for h in history:
                vi = h.get("val", {}).get("infill_loss", float("inf"))
                if vi < best_val_infill:
                    best_val_infill = vi
        print(f"  Resuming at epoch {start_epoch}, best_val_infill={best_val_infill:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{args.epochs}  lr={lr_now:.2e}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
            scheduler=scheduler,
        )

        val_metrics = run_epoch(
            model, val_loader, None, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
        )

        is_best = val_metrics["infill_loss"] < best_val_infill
        if is_best:
            best_val_infill = val_metrics["infill_loss"]

        save_checkpoint(out_dir, epoch, model, optimizer,
                        train_metrics, val_metrics, args, is_best)

        line1 = (f"  Train: loss={train_metrics['total']:.4f}  "
                 f"in={train_metrics['inside_l1']:.4f}  "
                 f"out={train_metrics['outside_l1']:.4f}")
        line2 = (f"  Val:   loss={val_metrics['total']:.4f}  "
                 f"in={val_metrics['inside_l1']:.4f}  "
                 f"out={val_metrics['outside_l1']:.4f}")
        print(line1)
        print(line2)
        if is_best:
            print(f"  ** New best val infill: {best_val_infill:.4f}")

        history.append({
            "epoch": epoch,
            "train": train_metrics, "val": val_metrics,
        })
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        with log_path.open("a") as f:
            f.write(f"\nEpoch {epoch}/{args.epochs}  lr={lr_now:.2e}\n")
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            if is_best:
                f.write(f"  ** New best val infill: {best_val_infill:.4f}\n")

    print(f"\nTraining complete. Best val infill: {best_val_infill:.4f}")
    print(f"Checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
