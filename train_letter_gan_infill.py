#!/usr/bin/env python3
"""
Per-letter GAN + perceptual infill trainer for IAM handwriting.

Training task: given a word image with some suffix of letters erased, the
model infills the NEXT letter (the one immediately after the visible
prefix). This mirrors inference: left-to-right autoregressive scribing.

Data source: per-letter bbox JSONL from extract_letter_bboxes.py. Each
word is a 64×256 letterboxed image; we pad vertically to 256×256 so the
existing FractalInfiller (RETINA_SIZE=256) can be reused unchanged.

For each sampled letter index i in a word:
  before = image with letters i..N-1 painted white
  after  = image with letters i+1..N-1 painted white (letter i visible)
  mask   = letter i's bbox, in 256×256 canvas coords
  delta  = after - before
  char_tokens = UTF-8 bytes of "prev + target + next" (PAD=0 at boundaries)
  bbox   = letter i's cx,cy,w,h normalized to 256

Usage:
    python3 train_letter_gan_infill.py \\
        --bbox-jsonl runs/letter_bboxes_v1.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out-dir runs/letter_gan_v1 \\
        --epochs 40 --batch-size 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from train_char_recognizer import letterbox, VOCAB
from train_character_infill import set_seed, pad_1d_long, move_batch_to_device
from train_fractal_infill import FractalInfiller, RETINA_SIZE
from train_gan_infill import (
    PatchDiscriminator,
    VGGPerceptualLoss,
    discriminator_loss,
    generator_loss,
)
from train_letter_classifier import LetterClassifier


WORD_H, WORD_W = 64, 256
CANVAS = RETINA_SIZE  # 256
V_OFFSET = (CANVAS - WORD_H) // 2  # 96


# ============================================================
# Differentiable letterbox for classifier CE loss
# ============================================================

class CropDiscriminator(nn.Module):
    """PatchGAN-style discriminator on retina-sized letter crops (1 ch).

    Unconditional realism discriminator — letter identity is shaped by
    the classifier CE loss, so this network's job is just to push for
    sharp, non-averaged strokes at letter scale.
    """

    def __init__(self, retina: int = 64, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []
        ch_in = 1
        ch_out = base_ch
        layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        for i in range(1, n_layers):
            ch_in = ch_out
            ch_out = min(ch_in * 2, base_ch * 4)
            stride = 2 if i < n_layers - 1 else 1
            layers.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 4, stride=stride, padding=1)),
                nn.GroupNorm(min(8, ch_out), ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch_out, 1, 4, stride=1, padding=1)),
        ))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def differentiable_letterbox(
    img: Tensor, bbox_mask: Tensor,
    bx1: Tensor, by1: Tensor, bx2: Tensor, by2: Tensor,
    retina: int,
) -> Tensor:
    """Letterbox each sample's letter crop into retina×retina.

    img:        [B, 1, H, W] — source canvas in [0,1] (1=white).
    bbox_mask:  [B, 1, H, W] — 1 inside the letter's bbox, 0 outside.
    bx1..by2:   [B] — pixel coords of each letter's bbox corners.
    retina:     output side length.

    Uses grid_sample over (1 - img * mask) so that pixels outside the
    mask OR outside the canvas sample to 0 (→ white after inversion),
    giving a clean white letterbox background.
    """
    B, _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    bw = (bx2 - bx1).clamp(min=1.0)
    bh = (by2 - by1).clamp(min=1.0)
    scale = torch.minimum(retina / bh, retina / bw)  # [B]
    nw = bw * scale  # [B]
    nh = bh * scale  # [B]

    # Output retina grid
    ys = torch.arange(retina, device=device, dtype=dtype)
    xs = torch.arange(retina, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [R, R]

    # Map output (rx, ry) → input (in_x, in_y) in canvas pixel coords.
    in_x = bx1[:, None, None] + (grid_x[None] - (retina - nw[:, None, None]) / 2) / scale[:, None, None]
    in_y = by1[:, None, None] + (grid_y[None] - (retina - nh[:, None, None]) / 2) / scale[:, None, None]

    norm_x = 2 * in_x / (W - 1) - 1
    norm_y = 2 * in_y / (H - 1) - 1
    grid = torch.stack([norm_x, norm_y], dim=-1)  # [B, R, R, 2]

    # Inverse-masked image: ink>0 inside bbox, 0 outside bbox or canvas.
    inv = (1.0 - img) * bbox_mask
    sampled = F.grid_sample(inv, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
    return 1.0 - sampled  # white outside letter, letter pixels inside


# ============================================================
# Dataset
# ============================================================

class LetterInfillDataset(Dataset):
    """Per-letter infill dataset backed by the bbox JSONL + raw IAM images."""

    def __init__(
        self,
        jsonl_path: str | Path,
        words_dir: str | Path,
        num_styles: int = 64,
        augment: bool = False,
    ) -> None:
        self.records = self._load(jsonl_path)
        self.words_dir = Path(words_dir)
        self.num_styles = num_styles
        self.augment = augment

    @staticmethod
    def _load(path: str | Path) -> List[Dict[str, Any]]:
        recs = []
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                r = json.loads(raw)
                if len(r["letters"]) < 2:
                    continue
                recs.append(r)
        return recs

    def __len__(self) -> int:
        return len(self.records)

    def _style_index(self, form: str) -> int:
        h = int(hashlib.md5(form.encode("utf-8")).hexdigest()[:8], 16)
        return h % self.num_styles

    def _load_word_image(self, rec: Dict[str, Any]) -> np.ndarray:
        p = self.words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
        img = Image.open(p).convert("L")
        arr = letterbox(img, WORD_H, WORD_W)  # [64, 256], 1=white, 0=ink
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        letters = rec["letters"]
        N = len(letters)
        i = random.randrange(N)  # which letter to infill

        arr = self._load_word_image(rec)

        # Build `after`: erase letters i+1..N-1 → letter i visible.
        arr_after = arr.copy()
        for j in range(i + 1, N):
            L = letters[j]
            x1, x2 = int(max(0, np.floor(L["x1"]))), int(min(WORD_W, np.ceil(L["x2"])))
            y1, y2 = int(max(0, np.floor(L["y1"]))), int(min(WORD_H, np.ceil(L["y2"])))
            if x2 > x1 and y2 > y1:
                arr_after[y1:y2, x1:x2] = 1.0  # white

        # Build `before`: erase letters i..N-1 → prefix only.
        arr_before = arr_after.copy()
        L = letters[i]
        x1, x2 = int(max(0, np.floor(L["x1"]))), int(min(WORD_W, np.ceil(L["x2"])))
        y1, y2 = int(max(0, np.floor(L["y1"]))), int(min(WORD_H, np.ceil(L["y2"])))
        if x2 > x1 and y2 > y1:
            arr_before[y1:y2, x1:x2] = 1.0

        # Pad vertically to 256×256 canvas (center).
        before = np.ones((CANVAS, CANVAS), dtype=np.float32)
        after = np.ones((CANVAS, CANVAS), dtype=np.float32)
        before[V_OFFSET:V_OFFSET + WORD_H] = arr_before
        after[V_OFFSET:V_OFFSET + WORD_H] = arr_after

        # Letter bbox in canvas coords.
        bx1 = float(L["x1"])
        bx2 = float(L["x2"])
        by1 = float(L["y1"]) + V_OFFSET
        by2 = float(L["y2"]) + V_OFFSET

        bbox_mask = np.zeros((CANVAS, CANVAS), dtype=np.float32)
        mx1, mx2 = int(max(0, np.floor(bx1))), int(min(CANVAS, np.ceil(bx2)))
        my1, my2 = int(max(0, np.floor(by1))), int(min(CANVAS, np.ceil(by2)))
        bbox_mask[my1:my2, mx1:mx2] = 1.0

        # Normalized (cx, cy, w, h) in [0,1].
        cx = 0.5 * (bx1 + bx2) / CANVAS
        cy = 0.5 * (by1 + by2) / CANVAS
        bw = (bx2 - bx1) / CANVAS
        bh = (by2 - by1) / CANVAS
        bbox = [cx, cy, bw, bh]

        if self.augment:
            # Brightness/contrast jitter (identical to before+after).
            if random.random() < 0.5:
                b = 1.0 + random.uniform(-0.15, 0.15)
                c = 1.0 + random.uniform(-0.15, 0.15)
                before = np.clip((before - 0.5) * c + 0.5 + (b - 1.0), 0, 1)
                after = np.clip((after - 0.5) * c + 0.5 + (b - 1.0), 0, 1)
            # NOTE: no horizontal flip — flipping breaks left-to-right
            # autoregressive inference semantics.

        # Char context: prev + target + next (UTF-8 bytes, PAD=0 at edges).
        prev_c = letters[i - 1]["char"] if i > 0 else None
        cur_c = letters[i]["char"]
        next_c = letters[i + 1]["char"] if i < N - 1 else None
        toks: List[int] = []
        for c in (prev_c, cur_c, next_c):
            if c is None:
                toks.append(0)
            else:
                b = c.encode("utf-8")
                toks.extend(list(b)) if len(b) == 1 else toks.append(0)
        char_tokens = torch.tensor(toks, dtype=torch.long)

        style_idx = self._style_index(rec["form"])
        conf = float(letters[i].get("prob", 1.0))
        target_label = VOCAB.index(cur_c) if cur_c in VOCAB else 0

        return {
            "record_id": f"{rec['word_id']}:{i}",
            "before": torch.from_numpy(before).unsqueeze(0),
            "after": torch.from_numpy(after).unsqueeze(0),
            "bbox_mask": torch.from_numpy(bbox_mask).unsqueeze(0),
            "delta": torch.from_numpy(after - before).unsqueeze(0),
            "char_tokens": char_tokens,
            "style_index": torch.tensor(style_idx, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "confidence": torch.tensor(conf, dtype=torch.float32),
            # Absolute pixel bbox in 256×256 canvas coords + target class.
            "letter_box_px": torch.tensor([bx1, by1, bx2, by2], dtype=torch.float32),
            "target_label": torch.tensor(target_label, dtype=torch.long),
        }


def collate_infill(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_id": [b["record_id"] for b in batch],
        "before": torch.stack([b["before"] for b in batch]),
        "after": torch.stack([b["after"] for b in batch]),
        "bbox_mask": torch.stack([b["bbox_mask"] for b in batch]),
        "delta": torch.stack([b["delta"] for b in batch]),
        "char_tokens": pad_1d_long([b["char_tokens"] for b in batch]),
        "char_lengths": torch.tensor(
            [b["char_tokens"].numel() for b in batch], dtype=torch.long),
        "style_index": torch.stack([b["style_index"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "confidence": torch.stack([b["confidence"] for b in batch]),
        "letter_box_px": torch.stack([b["letter_box_px"] for b in batch]),
        "target_label": torch.stack([b["target_label"] for b in batch]),
    }


# ============================================================
# Training loop
# ============================================================

def train_epoch(gen, disc, loader, opt_g, opt_d, vgg_loss_fn, device,
                grad_clip, halo_weight, lambda_adv, lambda_perc,
                classifier=None, lambda_ce=0.0, clf_retina=64,
                crop_disc=None, opt_crop_d=None, lambda_crop_adv=0.0,
                log_every=100):
    gen.train()
    disc.train()
    sums: Dict[str, float] = {}
    count = 0

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        before = batch["before"]
        bbox_mask = batch["bbox_mask"]
        target_delta = batch["delta"]
        target_after = batch["after"]
        char_tokens = batch["char_tokens"]
        char_lengths = batch["char_lengths"]
        style_index = batch["style_index"]
        bbox = batch["bbox"]
        confidence = batch["confidence"]
        letter_box_px = batch["letter_box_px"]
        target_label = batch["target_label"]

        # ---------- Generator forward ----------
        pred_delta = gen.forward_infill(
            before, bbox_mask, char_tokens, char_lengths, style_index, bbox,
        )
        pred_after = (before + pred_delta).clamp(0, 1)

        # ---------- Discriminator step ----------
        opt_d.zero_grad(set_to_none=True)
        d_real = disc(before, target_after, bbox_mask,
                      char_tokens, char_lengths, style_index, bbox)
        d_fake = disc(before, pred_after.detach(), bbox_mask,
                      char_tokens, char_lengths, style_index, bbox)
        d_loss, d_metrics = discriminator_loss(d_real, d_fake)
        d_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        opt_d.step()

        # ---------- Generator step ----------
        opt_g.zero_grad(set_to_none=True)
        d_fake_g = disc(before, pred_after, bbox_mask,
                        char_tokens, char_lengths, style_index, bbox)
        g_loss, g_metrics = generator_loss(
            d_fake=d_fake_g, pred_delta=pred_delta, target_delta=target_delta,
            bbox_mask=bbox_mask, confidence=confidence,
            pred_after=pred_after, target_after=target_after,
            vgg_loss_fn=vgg_loss_fn,
            halo_weight=halo_weight, lambda_adv=lambda_adv, lambda_perc=lambda_perc,
        )
        ce_loss_val = 0.0
        ce_acc_val = 0.0
        if classifier is not None and lambda_ce > 0:
            bx1, by1, bx2, by2 = letter_box_px.unbind(1)
            retina_img = differentiable_letterbox(
                pred_after, bbox_mask, bx1, by1, bx2, by2, clf_retina,
            )
            logits = classifier(retina_img)
            ce = F.cross_entropy(logits, target_label)
            g_loss = g_loss + lambda_ce * ce
            ce_loss_val = float(ce.detach())
            ce_acc_val = float((logits.detach().argmax(dim=1) == target_label).float().mean())
        g_metrics["g_ce"] = ce_loss_val
        g_metrics["g_ce_acc"] = ce_acc_val

        # ---------- Crop-level discriminator ----------
        crop_g_adv_val = 0.0
        crop_d_loss_val = 0.0
        if crop_disc is not None and opt_crop_d is not None and lambda_crop_adv > 0:
            bx1_c, by1_c, bx2_c, by2_c = letter_box_px.unbind(1)
            real_crop = differentiable_letterbox(
                target_after, bbox_mask, bx1_c, by1_c, bx2_c, by2_c, clf_retina,
            )
            fake_crop = differentiable_letterbox(
                pred_after.detach(), bbox_mask, bx1_c, by1_c, bx2_c, by2_c, clf_retina,
            )
            opt_crop_d.zero_grad(set_to_none=True)
            d_real_c = crop_disc(real_crop)
            d_fake_c = crop_disc(fake_crop)
            crop_d_loss = 0.5 * (
                F.mse_loss(d_real_c, torch.ones_like(d_real_c))
                + F.mse_loss(d_fake_c, torch.zeros_like(d_fake_c))
            )
            crop_d_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(crop_disc.parameters(), grad_clip)
            opt_crop_d.step()
            crop_d_loss_val = float(crop_d_loss.detach())

            # Generator adversarial term from crop_disc.
            fake_crop_g = differentiable_letterbox(
                pred_after, bbox_mask, bx1_c, by1_c, bx2_c, by2_c, clf_retina,
            )
            d_fake_g_c = crop_disc(fake_crop_g)
            crop_g_adv = F.mse_loss(d_fake_g_c, torch.ones_like(d_fake_g_c))
            g_loss = g_loss + lambda_crop_adv * crop_g_adv
            crop_g_adv_val = float(crop_g_adv.detach())
        g_metrics["crop_d_loss"] = crop_d_loss_val
        g_metrics["crop_g_adv"] = crop_g_adv_val

        g_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
        opt_g.step()

        metrics = {**d_metrics, **g_metrics}
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        count += 1

        if (step + 1) % log_every == 0:
            avg = {k: v / count for k, v in sums.items()}
            ce_bit = f"  g_ce={avg['g_ce']:.3f} acc={avg['g_ce_acc']:.3f}" \
                if classifier is not None and lambda_ce > 0 else ""
            crop_bit = f"  crop_d={avg['crop_d_loss']:.3f} crop_g={avg['crop_g_adv']:.3f}" \
                if crop_disc is not None and lambda_crop_adv > 0 else ""
            print(f"    step {step+1:>5}  "
                  f"g_total={avg['g_total']:.3f}  g_l1={avg['g_l1']:.3f}  "
                  f"g_adv={avg['g_adv']:.3f}  g_perc={avg['g_perc']:.3f}  "
                  f"d_loss={avg['d_loss']:.3f}{ce_bit}{crop_bit}")

    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def eval_epoch(gen, loader, vgg_loss_fn, device, halo_weight):
    gen.eval()
    from train_fractal_infill import compute_infill_loss
    tot_l1 = 0.0
    tot_inside = 0.0
    tot_outside = 0.0
    tot_perc = 0.0
    n = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_delta = gen.forward_infill(
            batch["before"], batch["bbox_mask"],
            batch["char_tokens"], batch["char_lengths"],
            batch["style_index"], batch["bbox"],
        )
        pred_after = (batch["before"] + pred_delta).clamp(0, 1)
        l1, m = compute_infill_loss(
            pred_delta, batch["delta"], batch["bbox_mask"],
            batch["confidence"], halo_weight,
        )
        perc = vgg_loss_fn(pred_after, batch["after"])
        bs = batch["before"].shape[0]
        tot_l1 += float(l1) * bs
        tot_inside += m["inside_l1"] * bs
        tot_outside += m["outside_l1"] * bs
        tot_perc += float(perc) * bs
        n += bs
    return {
        "val_l1": tot_l1 / max(n, 1),
        "val_inside_l1": tot_inside / max(n, 1),
        "val_outside_l1": tot_outside / max(n, 1),
        "val_perc": tot_perc / max(n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--num-styles", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="L1-only epochs before GAN+perceptual kick in.")
    ap.add_argument("--lambda-adv", type=float, default=0.01)
    ap.add_argument("--lambda-perc", type=float, default=0.1)
    ap.add_argument("--halo-weight", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to an existing infill checkpoint to resume from.")
    ap.add_argument("--classifier-ckpt", type=str, default=None,
                    help="Path to a trained letter classifier. If set, its "
                         "cross-entropy over pred_after is added as a signal.")
    ap.add_argument("--lambda-ce", type=float, default=0.1)
    ap.add_argument("--ce-start-epoch", type=int, default=0,
                    help="Epoch (0-indexed within this run) before which the CE "
                         "weight is zero. After this, weight = --lambda-ce.")
    ap.add_argument("--crop-disc", action="store_true",
                    help="Add a second letter-crop discriminator pushing realism "
                         "on the retina-sized crop used by the classifier.")
    ap.add_argument("--lambda-crop-adv", type=float, default=0.05)
    ap.add_argument("--crop-disc-base-ch", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = LetterInfillDataset(
        args.bbox_jsonl, args.words_dir,
        num_styles=args.num_styles, augment=True,
    )
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Dataset: {len(full_ds)} words → train={n_train} val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_infill, num_workers=args.num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_infill, num_workers=args.num_workers,
                            pin_memory=True)

    gen = FractalInfiller().to(device)
    disc = PatchDiscriminator(num_styles=args.num_styles).to(device)
    vgg_loss_fn = VGGPerceptualLoss().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        gen.load_state_dict(ck["gen_state_dict"])
        disc.load_state_dict(ck["disc_state_dict"])
        print(f"Resumed gen+disc from {args.resume_from} (prior epoch {ck.get('epoch')}, "
              f"val inside={ck.get('val', {}).get('val_inside_l1')})")

    classifier = None
    clf_retina = 64
    if args.classifier_ckpt:
        clf_ck = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
        clf_retina = clf_ck["args"].get("retina", 64)
        base_ch = clf_ck["args"].get("base_ch", 32)
        classifier = LetterClassifier(retina=clf_retina, base_ch=base_ch).to(device)
        classifier.load_state_dict(clf_ck["model_state_dict"])
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False
        print(f"Loaded classifier (retina={clf_retina}, val acc="
              f"{clf_ck.get('val', {}).get('acc'):.4f}) from {args.classifier_ckpt}")

    crop_disc = None
    opt_crop_d = None
    if args.crop_disc:
        crop_disc = CropDiscriminator(retina=clf_retina,
                                      base_ch=args.crop_disc_base_ch).to(device)
        opt_crop_d = torch.optim.Adam(crop_disc.parameters(), lr=args.lr_d,
                                      betas=(0.5, 0.999))
        print(f"Crop discriminator enabled (retina={clf_retina}, "
              f"lambda_crop_adv={args.lambda_crop_adv})")

    best_val = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        is_warmup = ep < args.warmup_epochs
        lam_adv = 0.0 if is_warmup else args.lambda_adv
        lam_perc = 0.0 if is_warmup else args.lambda_perc
        lam_ce = args.lambda_ce if (classifier is not None and ep >= args.ce_start_epoch) else 0.0
        phase = "warmup" if is_warmup else ("gan+ce" if lam_ce > 0 else "gan")
        print(f"\nEpoch {ep+1}/{args.epochs} [{phase}]")
        train_m = train_epoch(
            gen, disc, train_loader, opt_g, opt_d, vgg_loss_fn, device,
            grad_clip=args.grad_clip, halo_weight=args.halo_weight,
            lambda_adv=lam_adv, lambda_perc=lam_perc,
            classifier=classifier, lambda_ce=lam_ce, clf_retina=clf_retina,
            crop_disc=crop_disc, opt_crop_d=opt_crop_d,
            lambda_crop_adv=(args.lambda_crop_adv if not is_warmup else 0.0),
        )
        val_m = eval_epoch(gen, val_loader, vgg_loss_fn, device, args.halo_weight)
        dt = time.time() - t0
        print(f"  train: g_total={train_m['g_total']:.3f} g_l1={train_m['g_l1']:.3f}  "
              f"val: l1={val_m['val_l1']:.4f} in={val_m['val_inside_l1']:.4f} "
              f"perc={val_m['val_perc']:.4f}  ({dt:.1f}s)")

        ckpt = {
            "gen_state_dict": gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "args": vars(args),
            "epoch": ep + 1,
            "val": val_m,
        }
        if crop_disc is not None:
            ckpt["crop_disc_state_dict"] = crop_disc.state_dict()
        torch.save(ckpt, out_dir / "last.pt")
        if val_m["val_inside_l1"] < best_val:
            best_val = val_m["val_inside_l1"]
            torch.save(ckpt, out_dir / "best.pt")

    print(f"\nBest val inside L1={best_val:.4f}")


if __name__ == "__main__":
    main()
