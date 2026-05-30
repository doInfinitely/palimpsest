#!/usr/bin/env python3
"""
Train a diffusion-based word infill model on IAM handwriting.

Architecture:
  Conditional DDPM with UNet backbone.
  Input: noisy delta + before image + bbox mask → 3 channels
  Conditioning: timestep + character text + style + bbox via FiLM
  Predicts noise (epsilon-parameterization)

Inference: DDIM sampling for fast generation.

Usage:
    python train_diffusion_infill.py \
        --train-jsonl data/iam_full/infill_train.jsonl \
        --val-jsonl data/iam_full/infill_val.jsonl \
        --image-root data/iam_full \
        --out-dir runs/diff_v1
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

from train_character_infill import (
    set_seed,
    read_jsonl,
    CharacterInfillDataset,
    collate_infill,
    move_batch_to_device,
    ConvBlock,
    SelfAttention2d,
    FiLMModulation,
)
from train_fractal_infill import (
    FractalInfillDataset,
    collate_fractal,
    RETINA_SIZE,
)


# ============================================================
# Diffusion schedule
# ============================================================

class DiffusionSchedule:
    """Linear beta schedule for DDPM."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_var
        self.posterior_log_variance = torch.log(posterior_var.clamp(min=1e-20))
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0(self, xt: Tensor, t: Tensor, noise_pred: Tensor) -> Tensor:
        """Predict x0 from xt and predicted noise."""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t][:, None, None, None]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None]
        return sqrt_recip * xt - sqrt_recipm1 * noise_pred

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: Tuple[int, ...],
        condition: Dict[str, Tensor],
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        """DDIM sampling for fast inference."""
        device = self.betas.device
        b = shape[0]

        # Sub-select timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        xt = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(xt, t, **condition)

            # Predict x0
            x0_pred = self.predict_x0(xt, t, noise_pred)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t_val]
                alpha_next = self.alphas_cumprod[t_next]

                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_next) / (1 - alpha_t))
                    * torch.sqrt(1 - alpha_t / alpha_next)
                )

                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred
                noise = torch.randn_like(xt) if sigma > 0 else torch.zeros_like(xt)
                xt = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise
            else:
                xt = x0_pred

        return xt


# ============================================================
# Timestep embedding
# ============================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ============================================================
# Diffusion UNet
# ============================================================

class DiffusionUNet(nn.Module):
    """Conditional UNet for diffusion-based infill.

    Input channels: noisy_delta (1) + before (1) + bbox_mask (1) = 3
    Conditioning: timestep + char + style + bbox → FiLM
    Output: predicted noise (1 channel)
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base_ch: int = 96,
        cond_dim: int = 128,
        vocab_size: int = 256,
        num_styles: int = 64,
    ) -> None:
        super().__init__()
        ch = base_ch

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(cond_dim)

        # Semantic conditioning (char, style, bbox)
        self.char_embed = nn.Embedding(vocab_size, cond_dim)
        self.style_embed = nn.Embedding(num_styles, cond_dim)
        self.bbox_proj = nn.Linear(4, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 4, cond_dim)  # time + char + style + bbox

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

    def _get_cond(
        self, t: Tensor,
        char_tokens: Tensor, char_lengths: Tensor,
        style_index: Tensor, bbox: Tensor,
    ) -> Tensor:
        # Timestep
        time_emb = self.time_embed(t)

        # Character (mean-pooled)
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

        return self.cond_proj(torch.cat([time_emb, char_emb, style_emb, bbox_emb], dim=-1))

    def forward(
        self,
        noisy_delta: Tensor,
        t: Tensor,
        before: Tensor,
        bbox_mask: Tensor,
        char_tokens: Tensor,
        char_lengths: Tensor,
        style_index: Tensor,
        bbox: Tensor,
    ) -> Tensor:
        """Predict noise given noisy delta and conditions."""
        x = torch.cat([noisy_delta, before, bbox_mask], dim=1)
        cond = self._get_cond(t, char_tokens, char_lengths, style_index, bbox)

        # Encoder
        e1 = self.film1(self.enc1(x), cond)
        e2 = self.film2(self.enc2(self.down1(e1)), cond)
        e3 = self.film3(self.enc3(self.down2(e2)), cond)
        e4 = self.film4(self.enc4(self.down3(e3)), cond)

        bn = self.bottleneck(self.down4(e4))
        bn = self.film_bn(bn, cond)
        bn = self.attn_bn(bn)

        # Decoder
        d4 = self.film_d4(self.dec4(torch.cat([self.up4(bn), e4], dim=1)), cond)
        d3 = self.film_d3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)), cond)
        d2 = self.film_d2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), cond)
        d1 = self.film_d1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), cond)

        return self.out_conv(d1)


# ============================================================
# Training
# ============================================================

def train_epoch(
    model: DiffusionUNet,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bs = batch["before"].shape[0]

        # Target is the delta (after - before)
        x0 = batch["delta"]

        # Sample random timesteps
        t = torch.randint(0, schedule.num_timesteps, (bs,), device=device)

        # Add noise
        noise = torch.randn_like(x0)
        xt = schedule.q_sample(x0, t, noise)

        # Predict noise
        noise_pred = model(
            xt, t,
            before=batch["before"],
            bbox_mask=batch["bbox_mask"],
            char_tokens=batch["char_tokens"],
            char_lengths=batch["char_lengths"],
            style_index=batch["style_index"],
            bbox=batch["bbox"],
        )

        # MSE loss on noise (weighted by bbox mask for focus)
        mask = batch["bbox_mask"]
        outside = 1.0 - mask
        err = (noise_pred - noise) ** 2

        # Weighted: full weight inside mask, 0.2 outside
        weighted_err = err * mask + 0.2 * err * outside
        loss = weighted_err.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach()) * bs
        count += bs

    return {"mse_loss": total_loss / max(1, count)}


@torch.no_grad()
def val_epoch(
    model: DiffusionUNet,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bs = batch["before"].shape[0]

        x0 = batch["delta"]
        t = torch.randint(0, schedule.num_timesteps, (bs,), device=device)
        noise = torch.randn_like(x0)
        xt = schedule.q_sample(x0, t, noise)

        noise_pred = model(
            xt, t,
            before=batch["before"],
            bbox_mask=batch["bbox_mask"],
            char_tokens=batch["char_tokens"],
            char_lengths=batch["char_lengths"],
            style_index=batch["style_index"],
            bbox=batch["bbox"],
        )

        mask = batch["bbox_mask"]
        outside = 1.0 - mask
        err = (noise_pred - noise) ** 2
        weighted_err = err * mask + 0.2 * err * outside
        loss = weighted_err.mean()

        total_loss += float(loss) * bs
        count += bs

    return {"mse_loss": total_loss / max(1, count)}


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    out_dir: Path, epoch: int,
    model: DiffusionUNet,
    optimizer: torch.optim.Optimizer,
    train_metrics, val_metrics,
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
    p = argparse.ArgumentParser(description="Train diffusion infill model")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--base-ch", type=int, default=96)
    p.add_argument("--cond-dim", type=int, default=128)

    p.add_argument("--num-timesteps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--resume", type=str, default=None)
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

    # Detect data format
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

    # Build model and schedule
    model = DiffusionUNet(
        in_ch=3, out_ch=1,
        base_ch=args.base_ch,
        cond_dim=args.cond_dim,
        vocab_size=256,
        num_styles=num_styles,
    ).to(device)

    schedule = DiffusionSchedule(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    best_val_loss = float("inf")
    start_epoch = 1
    history: List[Dict[str, Any]] = []

    # Resume
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            print("  Optimizer state mismatch — starting fresh")
        start_epoch = ckpt["epoch"] + 1
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with hist_path.open() as f:
                history = json.load(f)
            for h in history:
                vl = h.get("val", {}).get("mse_loss", float("inf"))
                if vl < best_val_loss:
                    best_val_loss = vl
        print(f"  Resuming at epoch {start_epoch}, best_val={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs}  lr={lr_now:.2e}")

        train_metrics = train_epoch(
            model, schedule, train_loader, optimizer, device, args.grad_clip,
        )
        scheduler.step()

        val_metrics = val_epoch(model, schedule, val_loader, device)

        print(f"  Train: mse={train_metrics['mse_loss']:.6f}")
        print(f"  Val:   mse={val_metrics['mse_loss']:.6f}")

        is_best = val_metrics["mse_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["mse_loss"]
            print(f"  ** New best val: {best_val_loss:.6f}")

        if epoch % 5 == 0 or is_best or epoch == args.epochs:
            save_checkpoint(
                out_dir, epoch, model, optimizer,
                train_metrics, val_metrics, args, is_best,
            )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
