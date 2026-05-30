#!/usr/bin/env python3
"""
Train a hierarchical character infill model (coarse → refine).

Two-stage approach inspired by mangaka's hierarchical infiller:
  Stage 1 (Coarse): 64×64 UNet predicts rough character shape/placement
  Stage 2 (Refine): 256×256 UNet sharpens strokes using coarse prediction

Both stages are conditioned on character identity, bbox, and style via FiLM.
Joint training with combined loss.

Usage:
    python train_hierarchical_infill.py \
        --train-jsonl data/synth_v1/infill/character_infill.jsonl \
        --val-jsonl data/synth_v1/infill/character_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir runs/hier_v1
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
from torch.utils.data import DataLoader, Dataset

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
    move_batch_to_device,
)


# ============================================================
# Dataset (adds coarse-res targets)
# ============================================================

class HierarchicalInfillDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        style_to_index: Optional[Dict[str, int]] = None,
        patch_size: int = 256,
        coarse_size: int = 64,
        augment: bool = False,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index or {}
        self.patch_size = patch_size
        self.coarse_size = coarse_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _augment(
        self, before: Tensor, after: Tensor, bbox_mask: Tensor, bbox: List[float],
    ) -> Tuple[Tensor, Tensor, Tensor, List[float]]:
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
            bbox = [1.0 - bbox[0], bbox[1], bbox[2], bbox[3]]
        return before, after, bbox_mask, bbox

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        sz = self.patch_size
        csz = self.coarse_size

        before = load_gray_image(self.image_root / rec["before_patch_ref"], sz)
        after = load_gray_image(self.image_root / rec["after_patch_ref"], sz)
        bbox_mask = load_mask_image(self.image_root / rec["bbox_mask_ref"], sz)

        char_tokens = text_to_byte_tensor(rec.get("char_text"), max_len=16)
        style_id = rec.get("style_id", "__unk__")
        style_index = self.style_to_index.get(str(style_id), 0)

        bbox = rec.get("target_bbox_parent_norm_cxcywh")
        if bbox is None:
            bbox = [0.5, 0.5, 0.5, 0.5]

        if self.augment:
            before, after, bbox_mask, bbox = self._augment(before, after, bbox_mask, bbox)

        delta = after - before

        # Coarse-resolution targets
        before_c = F.interpolate(before.unsqueeze(0), size=csz, mode="bilinear", align_corners=False)[0]
        after_c = F.interpolate(after.unsqueeze(0), size=csz, mode="bilinear", align_corners=False)[0]
        bbox_mask_c = F.interpolate(bbox_mask.unsqueeze(0), size=csz, mode="nearest")[0]
        delta_c = after_c - before_c

        return {
            "record_id": rec["record_id"],
            "before": before,
            "after": after,
            "bbox_mask": bbox_mask,
            "delta": delta,
            "before_c": before_c,
            "after_c": after_c,
            "bbox_mask_c": bbox_mask_c,
            "delta_c": delta_c,
            "char_tokens": char_tokens,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_hier(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_id": [b["record_id"] for b in batch],
        "before": torch.stack([b["before"] for b in batch]),
        "after": torch.stack([b["after"] for b in batch]),
        "bbox_mask": torch.stack([b["bbox_mask"] for b in batch]),
        "delta": torch.stack([b["delta"] for b in batch]),
        "before_c": torch.stack([b["before_c"] for b in batch]),
        "after_c": torch.stack([b["after_c"] for b in batch]),
        "bbox_mask_c": torch.stack([b["bbox_mask_c"] for b in batch]),
        "delta_c": torch.stack([b["delta_c"] for b in batch]),
        "char_tokens": pad_1d_long([b["char_tokens"] for b in batch]),
        "char_lengths": torch.tensor([b["char_tokens"].numel() for b in batch], dtype=torch.long),
        "style_index": torch.stack([b["style_index"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "confidence": torch.stack([b["confidence"] for b in batch]),
    }


# ============================================================
# Conditioning module (shared by both stages)
# ============================================================

class ConditioningEncoder(nn.Module):
    """Produce a conditioning vector from char, style, and bbox."""
    def __init__(self, cond_dim: int, vocab_size: int = 256, num_styles: int = 64) -> None:
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, cond_dim)
        self.style_embed = nn.Embedding(num_styles, cond_dim)
        self.bbox_proj = nn.Linear(4, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 3, cond_dim)

    def forward(
        self, char_tokens: Tensor, char_lengths: Tensor,
        style_index: Tensor, bbox: Tensor,
    ) -> Tensor:
        if char_tokens.shape[1] > 0:
            char_emb = self.char_embed(char_tokens)
            mask = torch.arange(char_tokens.shape[1], device=char_tokens.device).unsqueeze(0) < char_lengths.unsqueeze(1)
            char_emb = (char_emb * mask.unsqueeze(-1).float()).sum(dim=1) / char_lengths.unsqueeze(-1).clamp_min(1).float()
        else:
            char_emb = torch.zeros(char_tokens.shape[0], self.char_embed.embedding_dim, device=char_tokens.device)
        style_emb = self.style_embed(style_index)
        bbox_emb = self.bbox_proj(bbox)
        return self.cond_proj(torch.cat([char_emb, style_emb, bbox_emb], dim=-1))


# ============================================================
# Coarse UNet (64×64)
# ============================================================

class CoarseUNet(nn.Module):
    """
    Smaller UNet operating at 64×64.
    Input: before_c (1ch) + bbox_mask_c (1ch) = 2ch
    Output: coarse delta (1ch) at 64×64
    """
    def __init__(self, cond_dim: int = 128, base_ch: int = 64) -> None:
        super().__init__()
        ch = base_ch

        # Encoder: 64→32→16→8
        self.enc1 = ConvBlock(2, ch)
        self.film1 = FiLMModulation(cond_dim, ch)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(ch, ch * 2)
        self.film2 = FiLMModulation(cond_dim, ch * 2)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.film3 = FiLMModulation(cond_dim, ch * 4)
        self.down3 = nn.MaxPool2d(2)

        # Bottleneck at 8×8
        self.bottleneck = ConvBlock(ch * 4, ch * 4)
        self.film_bn = FiLMModulation(cond_dim, ch * 4)
        self.attn_bn = SelfAttention2d(ch * 4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 2)
        self.film_d3 = FiLMModulation(cond_dim, ch * 2)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch)
        self.film_d2 = FiLMModulation(cond_dim, ch)

        self.up1 = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)
        self.film_d1 = FiLMModulation(cond_dim, ch)

        self.out_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, before_c: Tensor, bbox_mask_c: Tensor, cond: Tensor) -> Tensor:
        x = torch.cat([before_c, bbox_mask_c], dim=1)

        e1 = self.film1(self.enc1(x), cond)
        e2 = self.film2(self.enc2(self.down1(e1)), cond)
        e3 = self.film3(self.enc3(self.down2(e2)), cond)

        bn = self.bottleneck(self.down3(e3))
        bn = self.film_bn(bn, cond)
        bn = self.attn_bn(bn)

        d3 = self.film_d3(self.dec3(torch.cat([self.up3(bn), e3], dim=1)), cond)
        d2 = self.film_d2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), cond)
        d1 = self.film_d1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), cond)

        return self.out_conv(d1)


# ============================================================
# Refine UNet (256×256)
# ============================================================

class RefineUNet(nn.Module):
    """
    Full-resolution UNet conditioned on the coarse prediction.
    Input: before (1ch) + bbox_mask (1ch) + upsampled_coarse_delta (1ch) = 3ch
    Output: refined delta (1ch) at 256×256
    """
    def __init__(self, cond_dim: int = 128, base_ch: int = 64) -> None:
        super().__init__()
        ch = base_ch

        # Encoder: 256→128→64→32→16
        self.enc1 = ConvBlock(3, ch)
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

        # Bottleneck at 16×16
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

        self.out_conv = nn.Conv2d(ch, 1, 1)

    def forward(
        self, before: Tensor, bbox_mask: Tensor,
        coarse_delta_up: Tensor, cond: Tensor,
    ) -> Tensor:
        x = torch.cat([before, bbox_mask, coarse_delta_up], dim=1)

        e1 = self.film1(self.enc1(x), cond)
        e2 = self.film2(self.enc2(self.down1(e1)), cond)
        e3 = self.film3(self.enc3(self.down2(e2)), cond)
        e4 = self.film4(self.enc4(self.down3(e3)), cond)

        bn = self.bottleneck(self.down4(e4))
        bn = self.film_bn(bn, cond)
        bn = self.attn_bn(bn)

        d4 = self.film_d4(self.dec4(torch.cat([self.up4(bn), e4], dim=1)), cond)
        d3 = self.film_d3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)), cond)
        d2 = self.film_d2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), cond)
        d1 = self.film_d1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), cond)

        return self.out_conv(d1)


# ============================================================
# Hierarchical model wrapper
# ============================================================

class HierarchicalInfiller(nn.Module):
    """Two-stage coarse→refine character infiller."""

    def __init__(
        self,
        cond_dim: int = 128,
        coarse_ch: int = 64,
        refine_ch: int = 64,
        vocab_size: int = 256,
        num_styles: int = 64,
        coarse_size: int = 64,
        full_size: int = 256,
    ) -> None:
        super().__init__()
        self.coarse_size = coarse_size
        self.full_size = full_size

        self.cond_encoder = ConditioningEncoder(cond_dim, vocab_size, num_styles)
        self.coarse = CoarseUNet(cond_dim=cond_dim, base_ch=coarse_ch)
        self.refine = RefineUNet(cond_dim=cond_dim, base_ch=refine_ch)

    def forward(
        self,
        before: Tensor,       # [B,1,256,256]
        bbox_mask: Tensor,    # [B,1,256,256]
        before_c: Tensor,     # [B,1,64,64]
        bbox_mask_c: Tensor,  # [B,1,64,64]
        char_tokens: Tensor,
        char_lengths: Tensor,
        style_index: Tensor,
        bbox: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        cond = self.cond_encoder(char_tokens, char_lengths, style_index, bbox)

        # Stage 1: coarse prediction at 64×64
        coarse_delta = self.coarse(before_c, bbox_mask_c, cond)

        # Upsample coarse to full resolution for refine stage
        coarse_delta_up = F.interpolate(
            coarse_delta, size=self.full_size, mode="bilinear", align_corners=False,
        )

        # Stage 2: refine at 256×256
        refine_delta = self.refine(before, bbox_mask, coarse_delta_up, cond)

        return coarse_delta, refine_delta


# ============================================================
# Loss
# ============================================================

def compute_hier_loss(
    coarse_delta: Tensor,     # [B,1,64,64]
    refine_delta: Tensor,     # [B,1,256,256]
    target_delta_c: Tensor,   # [B,1,64,64]
    target_delta: Tensor,     # [B,1,256,256]
    bbox_mask_c: Tensor,      # [B,1,64,64]
    bbox_mask: Tensor,        # [B,1,256,256]
    confidence: Tensor,       # [B]
    halo_weight: float = 0.2,
    coarse_weight: float = 1.0,
    refine_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    def _stage_loss(
        pred: Tensor, target: Tensor, mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        inside = mask
        outside = 1.0 - mask
        l1 = torch.abs(pred - target)
        in_count = inside.sum(dim=(1, 2, 3)).clamp_min(1.0)
        out_count = outside.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_in = (l1 * inside).sum(dim=(1, 2, 3)) / in_count
        per_out = (l1 * outside).sum(dim=(1, 2, 3)) / out_count
        per_sample = per_in + halo_weight * per_out
        with torch.no_grad():
            avg_in = (l1 * inside).sum() / inside.sum().clamp_min(1.0)
            avg_out = (l1 * outside).sum() / outside.sum().clamp_min(1.0)
        return per_sample, avg_in, avg_out

    c_loss, c_in, c_out = _stage_loss(coarse_delta, target_delta_c, bbox_mask_c)
    r_loss, r_in, r_out = _stage_loss(refine_delta, target_delta, bbox_mask)

    per_sample = coarse_weight * c_loss + refine_weight * r_loss
    loss = (per_sample * confidence).sum() / confidence.sum().clamp_min(1.0)

    return loss, {
        "total": float(loss.detach().cpu()),
        "coarse_inside_l1": float(c_in.cpu()),
        "coarse_outside_l1": float(c_out.cpu()),
        "refine_inside_l1": float(r_in.cpu()),
        "refine_outside_l1": float(r_out.cpu()),
    }


# ============================================================
# Training
# ============================================================

def run_epoch(
    model: HierarchicalInfiller,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    halo_weight: float,
    coarse_weight: float,
    refine_weight: float,
    scheduler: Optional[Any] = None,
    freeze_refine: bool = False,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    # Optionally freeze refine stage during early epochs
    if freeze_refine and train:
        for p in model.refine.parameters():
            p.requires_grad_(False)
    elif train:
        for p in model.refine.parameters():
            p.requires_grad_(True)

    metric_keys = ["total", "coarse_inside_l1", "coarse_outside_l1",
                   "refine_inside_l1", "refine_outside_l1"]
    sums = {k: 0.0 for k in metric_keys}
    sums["count"] = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        coarse_delta, refine_delta = model(
            before=batch["before"],
            bbox_mask=batch["bbox_mask"],
            before_c=batch["before_c"],
            bbox_mask_c=batch["bbox_mask_c"],
            char_tokens=batch["char_tokens"],
            char_lengths=batch["char_lengths"],
            style_index=batch["style_index"],
            bbox=batch["bbox"],
        )

        loss, metrics = compute_hier_loss(
            coarse_delta=coarse_delta,
            refine_delta=refine_delta,
            target_delta_c=batch["delta_c"],
            target_delta=batch["delta"],
            bbox_mask_c=batch["bbox_mask_c"],
            bbox_mask=batch["bbox_mask"],
            confidence=batch["confidence"],
            halo_weight=halo_weight,
            coarse_weight=coarse_weight,
            refine_weight=refine_weight,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch["before"].shape[0]
        for k in metric_keys:
            sums[k] += metrics[k] * bs
        sums["count"] += bs

    if train and scheduler is not None:
        scheduler.step()

    count = max(1, sums["count"])
    return {k: (v / count if k != "count" else v) for k, v in sums.items()}


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: HierarchicalInfiller,
    optimizer: torch.optim.Optimizer,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    args: argparse.Namespace,
    is_best: bool = False,
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
    p = argparse.ArgumentParser(description="Train hierarchical character infill model")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--coarse-ch", type=int, default=64)
    p.add_argument("--refine-ch", type=int, default=64)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--num-styles", type=int, default=64)

    p.add_argument("--coarse-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=256)

    p.add_argument("--halo-weight", type=float, default=0.2)
    p.add_argument("--coarse-weight", type=float, default=1.0)
    p.add_argument("--refine-weight", type=float, default=1.0)

    # Phase 1: train coarse only for N epochs before joint training
    p.add_argument("--coarse-only-epochs", type=int, default=10,
                   help="Train coarse stage alone for N epochs before joint training")

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

    train_ds = HierarchicalInfillDataset(
        jsonl_path=args.train_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        patch_size=args.patch_size,
        coarse_size=args.coarse_size,
        augment=True,
    )
    val_ds = HierarchicalInfillDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        patch_size=args.patch_size,
        coarse_size=args.coarse_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_hier,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_hier,
    )

    model = HierarchicalInfiller(
        cond_dim=args.cond_dim,
        coarse_ch=args.coarse_ch,
        refine_ch=args.refine_ch,
        vocab_size=256,
        num_styles=num_styles,
        coarse_size=args.coarse_size,
        full_size=args.patch_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_coarse = sum(p.numel() for p in model.coarse.parameters())
    n_refine = sum(p.numel() for p in model.refine.parameters())
    n_cond = sum(p.numel() for p in model.cond_encoder.parameters())
    print(f"Model: {n_params:,} trainable parameters")
    print(f"  Conditioning: {n_cond:,}  Coarse: {n_coarse:,}  Refine: {n_refine:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    log_path = out_dir / "train.log"

    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        freeze_refine = epoch <= args.coarse_only_epochs
        phase = "coarse-only" if freeze_refine else "joint"
        print(f"\nEpoch {epoch}/{args.epochs}  lr={lr_now:.2e}  phase={phase}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
            coarse_weight=args.coarse_weight, refine_weight=args.refine_weight,
            scheduler=scheduler, freeze_refine=freeze_refine,
        )

        val_metrics = run_epoch(
            model, val_loader, None, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
            coarse_weight=args.coarse_weight, refine_weight=args.refine_weight,
        )

        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]

        save_checkpoint(out_dir, epoch, model, optimizer, train_metrics, val_metrics, args, is_best)

        line1 = (f"  Train: loss={train_metrics['total']:.4f}  "
                 f"c_in={train_metrics['coarse_inside_l1']:.4f}  c_out={train_metrics['coarse_outside_l1']:.4f}  "
                 f"r_in={train_metrics['refine_inside_l1']:.4f}  r_out={train_metrics['refine_outside_l1']:.4f}")
        line2 = (f"  Val:   loss={val_metrics['total']:.4f}  "
                 f"c_in={val_metrics['coarse_inside_l1']:.4f}  c_out={val_metrics['coarse_outside_l1']:.4f}  "
                 f"r_in={val_metrics['refine_inside_l1']:.4f}  r_out={val_metrics['refine_outside_l1']:.4f}")
        print(line1)
        print(line2)
        if is_best:
            print(f"  ** New best val loss: {best_val:.4f}")

        history.append({
            "epoch": epoch,
            "phase": phase,
            "train": train_metrics,
            "val": val_metrics,
        })
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        # Append to log file
        with log_path.open("a") as f:
            f.write(f"\nEpoch {epoch}/{args.epochs}  lr={lr_now:.2e}  phase={phase}\n")
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            if is_best:
                f.write(f"  ** New best val loss: {best_val:.4f}\n")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
