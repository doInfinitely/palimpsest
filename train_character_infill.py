#!/usr/bin/env python3
"""
Train a character infill model (conditional UNet).

Transforms local 256x256 patches from "before character" to "after character",
conditioned on character identity, bbox mask, and style.

Architecture: Conditional UNet with:
  - Before-patch + bbox mask as spatial input (2ch → encoder)
  - Character token embedding added via cross-attention or FiLM
  - Style embedding added via FiLM conditioning
  - Predicts residual delta_patch (direct regression, no diffusion for v1)

Loss:
  - L1 on delta inside bbox region
  - L1 with lower weight on halo region
  - Optional perceptual loss

Usage:
    python train_character_infill.py \
        --train-jsonl data/synth_v1/infill/character_infill.jsonl \
        --val-jsonl data/synth_v1/infill/character_infill_val.jsonl \
        --image-root data/synth_v1 \
        --out-dir runs/infill_v1
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


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
    return records


def load_gray_image(path: str | Path, size: int = 256) -> Tensor:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (size, size):
            img = img.resize((size, size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_mask_image(path: str | Path, size: int = 256) -> Tensor:
    x = load_gray_image(path, size)
    return (x > 0.5).float()


def text_to_byte_tensor(text: Optional[str], max_len: int = 16) -> Tensor:
    if text is None:
        return torch.empty(0, dtype=torch.long)
    data = list(text.encode("utf-8"))[:max_len]
    return torch.tensor(data, dtype=torch.long)


def pad_1d_long(seqs: Sequence[Tensor], pad_value: int = 0) -> Tensor:
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s.numel() > 0:
            out[i, : s.numel()] = s
    return out


# ============================================================
# Dataset
# ============================================================

class CharacterInfillDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        style_to_index: Optional[Dict[str, int]] = None,
        patch_size: int = 256,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index or {}
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        sz = self.patch_size

        before = load_gray_image(self.image_root / rec["before_patch_ref"], sz)
        after = load_gray_image(self.image_root / rec["after_patch_ref"], sz)
        bbox_mask = load_mask_image(self.image_root / rec["bbox_mask_ref"], sz)

        char_tokens = text_to_byte_tensor(rec.get("char_text"), max_len=16)
        style_id = rec.get("style_id", "__unk__")
        style_index = self.style_to_index.get(str(style_id), 0)

        bbox = rec.get("target_bbox_parent_norm_cxcywh")
        if bbox is None:
            bbox = [0.5, 0.5, 0.5, 0.5]

        return {
            "record_id": rec["record_id"],
            "before": before,
            "after": after,
            "bbox_mask": bbox_mask,
            "delta": after - before,  # target residual
            "char_tokens": char_tokens,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_infill(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_id": [b["record_id"] for b in batch],
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
# Conditional UNet
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation: scale and shift feature maps."""
    def __init__(self, cond_dim: int, num_features: int) -> None:
        super().__init__()
        self.scale = nn.Linear(cond_dim, num_features)
        self.shift = nn.Linear(cond_dim, num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: [B,C,H,W], cond: [B,D]
        s = self.scale(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.shift(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b


class ConditionalUNet(nn.Module):
    """
    Simple UNet for character infill.

    Input: before_patch (1ch) + bbox_mask (1ch) = 2ch
    Conditioning: char embedding + style embedding via FiLM
    Output: delta_patch (1ch)
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 64,
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

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 4)

        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 2)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch)

        self.up1 = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)

        self.out_conv = nn.Conv2d(ch, out_ch, 1)

    def _get_cond(
        self,
        char_tokens: Tensor,
        char_lengths: Tensor,
        style_index: Tensor,
        bbox: Tensor,
    ) -> Tensor:
        # Average char embedding
        if char_tokens.shape[1] > 0:
            char_emb = self.char_embed(char_tokens)
            # Masked mean
            mask = torch.arange(char_tokens.shape[1], device=char_tokens.device).unsqueeze(0) < char_lengths.unsqueeze(1)
            char_emb = (char_emb * mask.unsqueeze(-1).float()).sum(dim=1) / char_lengths.unsqueeze(-1).clamp_min(1).float()
        else:
            char_emb = torch.zeros(char_tokens.shape[0], self.char_embed.embedding_dim, device=char_tokens.device)

        style_emb = self.style_embed(style_index)
        bbox_emb = self.bbox_proj(bbox)
        cond = self.cond_proj(torch.cat([char_emb, style_emb, bbox_emb], dim=-1))
        return cond

    def forward(
        self,
        before: Tensor,
        bbox_mask: Tensor,
        char_tokens: Tensor,
        char_lengths: Tensor,
        style_index: Tensor,
        bbox: Tensor,
    ) -> Tensor:
        x = torch.cat([before, bbox_mask], dim=1)
        cond = self._get_cond(char_tokens, char_lengths, style_index, bbox)

        # Encoder
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond)

        e2 = self.enc2(self.down1(e1))
        e2 = self.film2(e2, cond)

        e3 = self.enc3(self.down2(e2))
        e3 = self.film3(e3, cond)

        e4 = self.enc4(self.down3(e3))
        e4 = self.film4(e4, cond)

        # Bottleneck
        bn = self.bottleneck(self.down4(e4))
        bn = self.film_bn(bn, cond)

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # [B,1,H,W] — predicted delta


# ============================================================
# Loss
# ============================================================

def compute_infill_loss(
    pred_delta: Tensor,    # [B,1,H,W]
    target_delta: Tensor,  # [B,1,H,W]
    bbox_mask: Tensor,     # [B,1,H,W]
    confidence: Tensor,    # [B]
    halo_weight: float = 0.2,
) -> Tuple[Tensor, Dict[str, float]]:
    # Inside bbox: full weight
    inside = bbox_mask
    # Outside bbox (halo): lower weight
    outside = 1.0 - bbox_mask
    weight_map = inside + halo_weight * outside

    l1 = torch.abs(pred_delta - target_delta)
    weighted_l1 = (l1 * weight_map).mean(dim=(1, 2, 3))

    loss = (weighted_l1 * confidence).sum() / confidence.sum().clamp_min(1.0)

    # Metrics
    with torch.no_grad():
        inside_l1 = (l1 * inside).sum() / inside.sum().clamp_min(1.0)
        outside_l1 = (l1 * outside).sum() / outside.sum().clamp_min(1.0)

    return loss, {
        "total": float(loss.detach().cpu()),
        "inside_l1": float(inside_l1.cpu()),
        "outside_l1": float(outside_l1.cpu()),
    }


# ============================================================
# Training
# ============================================================

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_epoch(
    model: ConditionalUNet,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    halo_weight: float,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    sums = {"total": 0.0, "inside_l1": 0.0, "outside_l1": 0.0, "count": 0}

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred_delta = model(
            before=batch["before"],
            bbox_mask=batch["bbox_mask"],
            char_tokens=batch["char_tokens"],
            char_lengths=batch["char_lengths"],
            style_index=batch["style_index"],
            bbox=batch["bbox"],
        )

        loss, metrics = compute_infill_loss(
            pred_delta=pred_delta,
            target_delta=batch["delta"],
            bbox_mask=batch["bbox_mask"],
            confidence=batch["confidence"],
            halo_weight=halo_weight,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch["before"].shape[0]
        sums["total"] += metrics["total"] * bs
        sums["inside_l1"] += metrics["inside_l1"] * bs
        sums["outside_l1"] += metrics["outside_l1"] * bs
        sums["count"] += bs

    count = max(1, sums["count"])
    return {k: (v / count if k != "count" else v) for k, v in sums.items()}


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: ConditionalUNet,
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
    p = argparse.ArgumentParser(description="Train character infill model")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--num-styles", type=int, default=64)

    p.add_argument("--halo-weight", type=float, default=0.2)

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

    train_ds = CharacterInfillDataset(
        jsonl_path=args.train_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
    )
    val_ds = CharacterInfillDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_infill,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_infill,
    )

    model = ConditionalUNet(
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

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
        )

        val_metrics = run_epoch(
            model, val_loader, None, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
        )

        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]

        save_checkpoint(out_dir, epoch, model, optimizer, train_metrics, val_metrics, args, is_best)

        print(f"  Train: loss={train_metrics['total']:.4f}  in={train_metrics['inside_l1']:.4f}  out={train_metrics['outside_l1']:.4f}")
        print(f"  Val:   loss={val_metrics['total']:.4f}  in={val_metrics['inside_l1']:.4f}  out={val_metrics['outside_l1']:.4f}")
        if is_best:
            print(f"  ** New best val loss: {best_val:.4f}")

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
