#!/usr/bin/env python3
"""
Train a word-level infill model with GAN + perceptual loss on IAM handwriting.

Architecture:
  Generator: FractalInfiller (conditional UNet, predicts residual delta)
  Discriminator: Conditional PatchGAN with FiLM conditioning
  Losses: L1 + VGG perceptual + LSGAN adversarial

Usage:
    python train_gan_infill.py \
        --train-jsonl data/iam_full/infill_train.jsonl \
        --val-jsonl data/iam_full/infill_val.jsonl \
        --image-root data/iam_full \
        --out-dir runs/gan_v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import torchvision.models as tv_models

from train_character_infill import (
    set_seed,
    read_jsonl,
    CharacterInfillDataset,
    collate_infill,
    move_batch_to_device,
    text_to_byte_tensor,
    pad_1d_long,
    ConvBlock,
    FiLMModulation,
)
from train_fractal_infill import (
    FractalInfiller,
    FractalInfillDataset,
    collate_fractal,
    compute_infill_loss,
    RETINA_SIZE,
)


# ============================================================
# PatchGAN Discriminator
# ============================================================

class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator with FiLM conditioning.

    Input: [before, target_or_pred, bbox_mask] concatenated → 3 channels
    Output: patch map of real/fake scores
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 64,
        n_layers: int = 4,
        cond_dim: int = 128,
        vocab_size: int = 256,
        num_styles: int = 64,
    ) -> None:
        super().__init__()

        # Conditioning (same structure as generator)
        self.char_embed = nn.Embedding(vocab_size, cond_dim)
        self.style_embed = nn.Embedding(num_styles, cond_dim)
        self.bbox_proj = nn.Linear(4, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 3, cond_dim)

        layers = []
        films = nn.ModuleList()

        # First layer (no normalization)
        ch_in = in_ch
        ch_out = base_ch
        layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        films.append(FiLMModulation(cond_dim, ch_out))

        # Middle layers
        for i in range(1, n_layers):
            ch_in = ch_out
            ch_out = min(ch_in * 2, base_ch * 8)
            stride = 2 if i < n_layers - 1 else 1
            layers.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 4, stride=stride, padding=1)),
                nn.GroupNorm(min(8, ch_out), ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            films.append(FiLMModulation(cond_dim, ch_out))

        # Final 1x1 → scalar patch
        layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch_out, 1, 4, stride=1, padding=1)),
        ))

        self.layers = nn.ModuleList(layers)
        self.films = films

    def _get_cond(self, char_tokens, char_lengths, style_index, bbox):
        if char_tokens.shape[1] > 0:
            char_emb = self.char_embed(char_tokens)
            mask = (torch.arange(char_tokens.shape[1], device=char_tokens.device).unsqueeze(0)
                    < char_lengths.unsqueeze(1))
            char_emb = ((char_emb * mask.unsqueeze(-1).float()).sum(dim=1)
                        / char_lengths.unsqueeze(-1).clamp_min(1).float())
        else:
            char_emb = torch.zeros(
                char_tokens.shape[0], self.char_embed.embedding_dim,
                device=char_tokens.device,
            )
        style_emb = self.style_embed(style_index)
        bbox_emb = self.bbox_proj(bbox)
        return self.cond_proj(torch.cat([char_emb, style_emb, bbox_emb], dim=-1))

    def forward(
        self,
        before: Tensor,
        target_or_pred: Tensor,
        bbox_mask: Tensor,
        char_tokens: Tensor,
        char_lengths: Tensor,
        style_index: Tensor,
        bbox: Tensor,
    ) -> Tensor:
        x = torch.cat([before, target_or_pred, bbox_mask], dim=1)
        cond = self._get_cond(char_tokens, char_lengths, style_index, bbox)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.films):
                x = self.films[i](x, cond)

        return x  # [B, 1, H', W'] patch scores


# ============================================================
# VGG Perceptual Loss
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss comparing feature activations."""

    def __init__(self, layer_weights: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        features = vgg.features

        # Extract features at specific layers
        self.slices = nn.ModuleDict()
        # relu1_2 = features[:4], relu2_2 = features[:9], relu3_3 = features[:16]
        self.slices["relu1_2"] = nn.Sequential(*list(features[:4]))
        self.slices["relu2_2"] = nn.Sequential(*list(features[4:9]))
        self.slices["relu3_3"] = nn.Sequential(*list(features[9:16]))

        self.layer_weights = layer_weights or {
            "relu1_2": 1.0,
            "relu2_2": 1.0,
            "relu3_3": 1.0,
        }

        # Freeze VGG
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: Tensor) -> Tensor:
        """Convert grayscale [B,1,H,W] to normalized RGB [B,3,H,W]."""
        x = x.repeat(1, 3, 1, 1)  # grayscale → RGB
        return (x - self.mean) / self.std

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_rgb = self._normalize(pred)
        target_rgb = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        p, t = pred_rgb, target_rgb
        for name, slice_mod in self.slices.items():
            p = slice_mod(p)
            t = slice_mod(t)
            w = self.layer_weights.get(name, 1.0)
            loss = loss + w * F.l1_loss(p, t)

        return loss


# ============================================================
# Combined losses
# ============================================================

def discriminator_loss(
    d_real: Tensor, d_fake: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """LSGAN discriminator loss."""
    loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
    loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
    loss = 0.5 * (loss_real + loss_fake)
    return loss, {
        "d_loss": float(loss.detach()),
        "d_real": float(d_real.detach().mean()),
        "d_fake": float(d_fake.detach().mean()),
    }


def generator_loss(
    d_fake: Tensor,
    pred_delta: Tensor,
    target_delta: Tensor,
    bbox_mask: Tensor,
    confidence: Tensor,
    pred_after: Tensor,
    target_after: Tensor,
    vgg_loss_fn: VGGPerceptualLoss,
    halo_weight: float = 0.2,
    lambda_adv: float = 0.01,
    lambda_perc: float = 0.1,
) -> Tuple[Tensor, Dict[str, float]]:
    """Combined generator loss: L1 + perceptual + adversarial."""
    # L1 infill loss
    l1_loss, l1_metrics = compute_infill_loss(
        pred_delta, target_delta, bbox_mask, confidence, halo_weight,
    )

    # Adversarial loss (generator wants D to say "real")
    adv_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))

    # Perceptual loss (on full reconstructed images)
    perc_loss = vgg_loss_fn(pred_after, target_after)

    total = l1_loss + lambda_adv * adv_loss + lambda_perc * perc_loss

    metrics = {
        "g_total": float(total.detach()),
        "g_l1": float(l1_loss.detach()),
        "g_adv": float(adv_loss.detach()),
        "g_perc": float(perc_loss.detach()),
        **l1_metrics,
    }
    return total, metrics


# ============================================================
# Training loop
# ============================================================

def train_epoch(
    gen: FractalInfiller,
    disc: PatchDiscriminator,
    loader: DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    vgg_loss_fn: VGGPerceptualLoss,
    device: torch.device,
    grad_clip: float,
    halo_weight: float,
    lambda_adv: float,
    lambda_perc: float,
) -> Dict[str, float]:
    gen.train()
    disc.train()

    metric_keys = [
        "g_total", "g_l1", "g_adv", "g_perc",
        "d_loss", "d_real", "d_fake",
        "inside_l1", "outside_l1",
    ]
    sums = {k: 0.0 for k in metric_keys}
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bs = batch["before"].shape[0]

        before = batch["before"]
        bbox_mask = batch["bbox_mask"]
        target_delta = batch["delta"]
        target_after = batch["after"]
        char_tokens = batch["char_tokens"]
        char_lengths = batch["char_lengths"]
        style_index = batch["style_index"]
        bbox = batch["bbox"]
        confidence = batch["confidence"]

        # ---------- Generator forward ----------
        pred_delta = gen.forward_infill(
            before, bbox_mask, char_tokens, char_lengths, style_index, bbox,
        )
        pred_after = (before + pred_delta).clamp(0, 1)

        # ---------- Update Discriminator ----------
        opt_d.zero_grad(set_to_none=True)

        d_real = disc(before, target_after, bbox_mask,
                      char_tokens, char_lengths, style_index, bbox)
        d_fake = disc(before, pred_after.detach(), bbox_mask,
                      char_tokens, char_lengths, style_index, bbox)

        d_loss, d_metrics = discriminator_loss(d_real, d_fake)
        d_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        opt_d.step()

        # ---------- Update Generator ----------
        opt_g.zero_grad(set_to_none=True)

        # Re-run discriminator on fake (with gradients to generator)
        d_fake_for_g = disc(before, pred_after, bbox_mask,
                            char_tokens, char_lengths, style_index, bbox)

        g_loss, g_metrics = generator_loss(
            d_fake_for_g, pred_delta, target_delta, bbox_mask, confidence,
            pred_after, target_after, vgg_loss_fn,
            halo_weight, lambda_adv, lambda_perc,
        )
        g_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
        opt_g.step()

        # Accumulate metrics
        all_metrics = {**g_metrics, **d_metrics}
        for k in metric_keys:
            sums[k] += all_metrics.get(k, 0.0) * bs
        count += bs

    count = max(1, count)
    return {k: v / count for k, v in sums.items()}


@torch.no_grad()
def val_epoch(
    gen: FractalInfiller,
    loader: DataLoader,
    device: torch.device,
    halo_weight: float,
) -> Dict[str, float]:
    """Validation: only L1 loss (no GAN/perceptual for stability)."""
    gen.eval()

    metric_keys = ["total", "inside_l1", "outside_l1"]
    sums = {k: 0.0 for k in metric_keys}
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bs = batch["before"].shape[0]

        pred_delta = gen.forward_infill(
            batch["before"], batch["bbox_mask"],
            batch["char_tokens"], batch["char_lengths"],
            batch["style_index"], batch["bbox"],
        )
        loss, infill_metrics = compute_infill_loss(
            pred_delta, batch["delta"], batch["bbox_mask"],
            batch["confidence"], halo_weight,
        )

        sums["total"] += float(loss.cpu()) * bs
        for k in ["inside_l1", "outside_l1"]:
            sums[k] += infill_metrics.get(k, 0.0) * bs
        count += bs

    count = max(1, count)
    return {k: v / count for k, v in sums.items()}


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    out_dir: Path, epoch: int,
    gen: FractalInfiller, disc: PatchDiscriminator,
    opt_g, opt_d,
    train_metrics, val_metrics,
    args: argparse.Namespace,
    is_best: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_g_state_dict": opt_g.state_dict(),
        "opt_d_state_dict": opt_d.state_dict(),
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
    p = argparse.ArgumentParser(description="Train GAN + perceptual infill model")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--base-ch", type=int, default=96,
                   help="Base channels for generator UNet")
    p.add_argument("--disc-ch", type=int, default=64,
                   help="Base channels for discriminator")
    p.add_argument("--cond-dim", type=int, default=128)

    p.add_argument("--halo-weight", type=float, default=0.2)
    p.add_argument("--lambda-adv", type=float, default=0.01)
    p.add_argument("--lambda-perc", type=float, default=0.1)

    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="L1-only warmup before enabling GAN + perceptual")

    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--pretrained-gen", type=str, default=None,
                   help="Path to pretrained generator checkpoint for warm start")

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading training data from {args.train_jsonl}...")
    train_records = read_jsonl(args.train_jsonl)
    print(f"Loading validation data from {args.val_jsonl}...")
    val_records = read_jsonl(args.val_jsonl)
    print(f"  Train: {len(train_records)} records, Val: {len(val_records)} records")

    # Build style vocab
    style_ids = {"__unk__"}
    for rec in train_records + val_records:
        style_ids.add(str(rec.get("style_id", "__unk__")))
    style_to_index = {s: i for i, s in enumerate(sorted(style_ids))}
    num_styles = len(style_to_index)
    print(f"  {num_styles} styles")

    with (out_dir / "style_to_index.json").open("w") as f:
        json.dump(style_to_index, f, indent=2)

    # Detect data format (fractal multi-level vs character-only)
    has_levels = "level" in train_records[0] if train_records else False
    if has_levels:
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
    else:
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

    # Build models
    gen = FractalInfiller(
        in_ch=2, out_ch=1,
        base_ch=args.base_ch,
        cond_dim=args.cond_dim,
        vocab_size=256,
        num_styles=num_styles,
    ).to(device)

    disc = PatchDiscriminator(
        in_ch=3,
        base_ch=args.disc_ch,
        n_layers=4,
        cond_dim=args.cond_dim,
        vocab_size=256,
        num_styles=num_styles,
    ).to(device)

    vgg_loss_fn = VGGPerceptualLoss().to(device)

    # Optional warm start from pretrained generator
    if args.pretrained_gen:
        print(f"Loading pretrained generator: {args.pretrained_gen}")
        ckpt = torch.load(args.pretrained_gen, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("gen_state_dict", {}))
        gen.load_state_dict(state, strict=False)

    n_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    n_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator: {n_gen:,} params, Discriminator: {n_disc:,} params")

    opt_g = torch.optim.AdamW(
        gen.parameters(), lr=args.lr_g, weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.AdamW(
        disc.parameters(), lr=args.lr_d, weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.epochs, eta_min=args.lr_g * 0.01,
    )
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=args.epochs, eta_min=args.lr_d * 0.01,
    )

    best_val_infill = float("inf")
    start_epoch = 1
    history: List[Dict[str, Any]] = []

    # Resume
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["gen_state_dict"], strict=False)
        disc.load_state_dict(ckpt["disc_state_dict"], strict=False)
        try:
            opt_g.load_state_dict(ckpt["opt_g_state_dict"])
            opt_d.load_state_dict(ckpt["opt_d_state_dict"])
        except (ValueError, KeyError):
            print("  Optimizer state mismatch — starting optimizers fresh")
        start_epoch = ckpt["epoch"] + 1
        for _ in range(ckpt["epoch"]):
            sched_g.step()
            sched_d.step()
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with hist_path.open() as f:
                history = json.load(f)
            for h in history:
                vi = h.get("val", {}).get("inside_l1", float("inf"))
                if vi < best_val_infill:
                    best_val_infill = vi
        print(f"  Resuming at epoch {start_epoch}, best_val={best_val_infill:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        lr_g = opt_g.param_groups[0]["lr"]
        lr_d = opt_d.param_groups[0]["lr"]

        # Warmup: L1 only (no GAN/perceptual)
        in_warmup = epoch <= args.warmup_epochs
        eff_lambda_adv = 0.0 if in_warmup else args.lambda_adv
        eff_lambda_perc = 0.0 if in_warmup else args.lambda_perc

        phase = "warmup" if in_warmup else "GAN"
        print(f"\nEpoch {epoch}/{args.epochs}  lr_g={lr_g:.2e} lr_d={lr_d:.2e}  [{phase}]")

        train_metrics = train_epoch(
            gen, disc, train_loader, opt_g, opt_d, vgg_loss_fn,
            device, args.grad_clip, args.halo_weight,
            eff_lambda_adv, eff_lambda_perc,
        )

        sched_g.step()
        sched_d.step()

        val_metrics = val_epoch(gen, val_loader, device, args.halo_weight)

        # Log
        print(f"  Train: g_total={train_metrics['g_total']:.4f}  "
              f"g_l1={train_metrics['g_l1']:.4f}  "
              f"g_adv={train_metrics['g_adv']:.4f}  "
              f"g_perc={train_metrics['g_perc']:.4f}  "
              f"d_loss={train_metrics['d_loss']:.4f}")
        print(f"  Val:   loss={val_metrics['total']:.4f}  "
              f"in={val_metrics['inside_l1']:.4f}  "
              f"out={val_metrics['outside_l1']:.4f}")

        is_best = val_metrics["inside_l1"] < best_val_infill
        if is_best:
            best_val_infill = val_metrics["inside_l1"]
            print(f"  ** New best val infill: {best_val_infill:.4f}")

        save_checkpoint(
            out_dir, epoch, gen, disc, opt_g, opt_d,
            train_metrics, val_metrics, args, is_best,
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val infill: {best_val_infill:.4f}")
    print(f"Checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
