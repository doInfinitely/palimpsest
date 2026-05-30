#!/usr/bin/env python3
"""Train a discriminator on real IAM word crops vs v11 generator outputs,
then gradient-descend a generated crop's pixels to fool the discriminator.
Save every refinement step so the whole transformation can be made into a
gif.

Usage:
    python3 refine_with_gan.py \\
        --infill-ckpt runs/letter_gan_v11_extended2/best.pt \\
        --bbox-ckpt runs/bbox_predictor_v2/best.pt \\
        --print-glyphs runs/print_glyphs.pt \\
        --words-dir data/iam_words/iam_words/words \\
        --bbox-jsonl runs/letter_bboxes_v2.jsonl \\
        --target-text "hello" \\
        --out-dir eval_output/refine \\
        --n-real 2000 --n-fake 2000 \\
        --disc-epochs 8 --refine-steps 80 --refine-lr 1e-2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Project imports.
from train_char_recognizer import VOCAB, letterbox
from train_fractal_infill import FractalInfiller
from train_letter_gan_infill import CANVAS, V_OFFSET, WORD_H, WORD_W
from infer_scribe import BboxPredictor, predict_boxes_px, style_from_seed, render_word


# ============================================================
# Word-crop discriminator (64×256 → real/fake score)
# ============================================================

class CropD(nn.Module):
    """Small PatchGAN-style discriminator on 64×256 word crops."""
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.body = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1),        # 32×128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c*2, 4, stride=2, padding=1),      # 16×64
            nn.GroupNorm(8, c*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*2, c*4, 4, stride=2, padding=1),    # 8×32
            nn.GroupNorm(8, c*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*4, c*8, 4, stride=2, padding=1),    # 4×16
            nn.GroupNorm(8, c*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*8, 1, 4, stride=1, padding=0),      # 1×13 logits
        )

    def forward(self, x):
        # x: [B, 1, 64, 256] in [0, 1]. Center on 0 for the model.
        return self.body(x * 2.0 - 1.0)  # [B, 1, 1, 13]


# ============================================================
# Real / fake crop datasets
# ============================================================

class RealCropDataset(Dataset):
    """64×256 letterboxed IAM word crops from the existing bbox JSONL."""
    def __init__(self, jsonl_path, words_dir, n=2000, seed=0,
                 min_len=3, max_len=10):
        self.recs = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if min_len <= len(r["text"]) <= max_len:
                    self.recs.append(r)
        rng = random.Random(seed)
        rng.shuffle(self.recs)
        self.recs = self.recs[:n]
        self.words_dir = Path(words_dir)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        p = self.words_dir / r["form"] / r["line"] / f"{r['word_id']}.png"
        arr = letterbox(Image.open(p).convert("L"), WORD_H, WORD_W)  # [H,W] 1=white
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


def gen_fake_crops(args, n: int, device) -> torch.Tensor:
    """Render n fake word crops with the v11 model. Returns [n, 1, 64, 256]."""
    # Source: short transcriptions from the bbox jsonl. We don't need bboxes
    # to predict — infer_scribe's BboxPredictor handles that.
    texts = []
    with open(args.bbox_jsonl) as f:
        for line in f:
            r = json.loads(line)
            t = "".join(c for c in r["text"] if c in VOCAB)
            if 3 <= len(t) <= 10:
                texts.append(t)
    rng = random.Random(1)
    rng.shuffle(texts)
    texts = texts[:n]

    # Load bbox predictor + infill gen + optional print glyphs.
    bb_ck = torch.load(args.bbox_ckpt, map_location=device, weights_only=False)
    bb_args = bb_ck.get("args", {})
    predictor = BboxPredictor(
        d_model=bb_args.get("d_model", 128),
        nhead=bb_args.get("nhead", 4),
        layers=bb_args.get("layers", 3),
        max_len=bb_args.get("max_len", 24),
    ).to(device)
    predictor.load_state_dict(bb_ck["model_state_dict"])
    predictor.eval()
    lh_norm = float(bb_args.get("line_height", 0.05))
    max_len_pred = int(bb_args.get("max_len", 24))

    inf_ck = torch.load(args.infill_ckpt, map_location=device, weights_only=False)
    inf_args = inf_ck.get("args", {}) or {}
    noise_dim = int(inf_args.get("noise_dim", 0))
    use_print = bool(inf_args.get("print_glyphs"))
    gen = FractalInfiller(noise_dim=noise_dim, use_print_cond=use_print).to(device)
    gen.load_state_dict(inf_ck["gen_state_dict"])
    gen.eval()

    pglyphs = None
    if args.print_glyphs and use_print:
        pc = torch.load(args.print_glyphs, weights_only=False)
        pglyphs = pc["glyphs"][args.print_font_idx].float().unsqueeze(1) / 255.0

    out = torch.empty(n, 1, WORD_H, WORD_W, dtype=torch.float32)
    rng2 = random.Random(2)
    for i, t in enumerate(texts):
        t = t[:max_len_pred]
        style = style_from_seed(rng2.choice(["a","b","c","d","e","f","g","h"]))
        boxes_px = predict_boxes_px(t, predictor, device,
                                    line_height=lh_norm, max_len=max_len_pred)
        word_arr, _ = render_word(t, boxes_px, gen, device,
                                  style_index=style, print_glyphs=pglyphs)
        out[i, 0] = torch.from_numpy(word_arr)
        if (i+1) % 100 == 0:
            print(f"  fake {i+1}/{n}")
    return out


class FakeCropDataset(Dataset):
    def __init__(self, tensor):
        self.t = tensor
    def __len__(self): return self.t.shape[0]
    def __getitem__(self, i): return self.t[i]


# ============================================================
# Train discriminator
# ============================================================

def train_disc(real_ds, fake_ds, epochs, batch_size, lr, device):
    D = CropD().to(device)
    opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=True)
    fake_loader = DataLoader(fake_ds, batch_size=batch_size, shuffle=True,
                             num_workers=2, drop_last=True)
    for ep in range(epochs):
        D.train()
        n_steps = min(len(real_loader), len(fake_loader))
        ri = iter(real_loader); fi = iter(fake_loader)
        tot_loss = 0.0; tot_real = 0.0; tot_fake = 0.0
        for step in range(n_steps):
            real = next(ri).to(device)
            fake = next(fi).to(device)
            opt.zero_grad(set_to_none=True)
            r_logit = D(real).mean(dim=(1,2,3))
            f_logit = D(fake).mean(dim=(1,2,3))
            # LSGAN: real→1, fake→0
            loss = 0.5*((r_logit - 1.0)**2).mean() + 0.5*(f_logit**2).mean()
            loss.backward()
            opt.step()
            tot_loss += float(loss); tot_real += float(r_logit.mean()); tot_fake += float(f_logit.mean())
        print(f"D ep {ep+1}/{epochs}  loss={tot_loss/n_steps:.3f}  "
              f"r={tot_real/n_steps:.2f}  f={tot_fake/n_steps:.2f}")
    D.eval()
    return D


# ============================================================
# Refinement: gradient-descend pixels to fool D
# ============================================================

def refine_pixels(x0: torch.Tensor, D: nn.Module, steps: int, lr: float,
                  device, save_dir: Path, mask=None):
    """x0: [1,1,H,W] initial crop. Run gradient ascent on D(x) to maximize
    perceived realism. Save the image at each step."""
    save_dir.mkdir(parents=True, exist_ok=True)
    x = x0.clone().to(device).requires_grad_(True)
    # Save initial frame.
    Image.fromarray((x.detach().cpu().numpy()[0,0] * 255).astype(np.uint8)).save(
        save_dir / f"step_{0:03d}.png")
    opt = torch.optim.Adam([x], lr=lr)
    init = x0.clone().to(device)
    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        score = D(x).mean()
        loss = (score - 1.0).pow(2).mean()
        # Light L2 anchor to init so it doesn't drift wildly off.
        anchor = 0.02 * (x - init).pow(2).mean()
        (loss + anchor).backward()
        opt.step()
        x.data.clamp_(0.0, 1.0)
        Image.fromarray((x.detach().cpu().numpy()[0,0] * 255).astype(np.uint8)).save(
            save_dir / f"step_{s:03d}.png")
        if s % 10 == 0:
            print(f"  refine {s}/{steps}  D(x)={float(score):.3f}  ||Δ||={float((x-init).norm()):.2f}")
    return x.detach().cpu()


def make_gif(frame_dir: Path, out_path: Path, duration_ms: int = 80):
    frames = sorted(frame_dir.glob("step_*.png"))
    if not frames:
        raise FileNotFoundError(frame_dir)
    imgs = [Image.open(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=True)
    print(f"Saved gif: {out_path}  ({len(imgs)} frames)")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infill-ckpt", required=True)
    ap.add_argument("--bbox-ckpt", required=True)
    ap.add_argument("--print-glyphs", default=None)
    ap.add_argument("--print-font-idx", type=int, default=0)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--target-text", default="hello")
    ap.add_argument("--target-style", default="alpha")
    ap.add_argument("--out-dir", default="eval_output/refine")
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--n-fake", type=int, default=2000)
    ap.add_argument("--disc-epochs", type=int, default=8)
    ap.add_argument("--disc-batch", type=int, default=32)
    ap.add_argument("--disc-lr", type=float, default=2e-4)
    ap.add_argument("--refine-steps", type=int, default=80)
    ap.add_argument("--refine-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. Build datasets -----
    print(f"Loading {args.n_real} real word crops…")
    real_ds = RealCropDataset(args.bbox_jsonl, args.words_dir, n=args.n_real,
                              seed=args.seed)
    fake_path = out_dir / "fake_crops.pt"
    if fake_path.exists():
        print(f"Loading cached fake crops from {fake_path}")
        fake_t = torch.load(fake_path, weights_only=False)
    else:
        print(f"Generating {args.n_fake} fake word crops with v11…")
        fake_t = gen_fake_crops(args, args.n_fake, device)
        torch.save(fake_t, fake_path)
    fake_ds = FakeCropDataset(fake_t)
    print(f"Datasets: real={len(real_ds)}  fake={len(fake_ds)}")

    # ----- 2. Train D -----
    D = train_disc(real_ds, fake_ds, args.disc_epochs, args.disc_batch,
                   args.disc_lr, device)
    torch.save(D.state_dict(), out_dir / "discriminator.pt")

    # ----- 3. Render the target word with v11 -----
    bb_ck = torch.load(args.bbox_ckpt, map_location=device, weights_only=False)
    bb_args = bb_ck.get("args", {})
    predictor = BboxPredictor(
        d_model=bb_args.get("d_model", 128), nhead=bb_args.get("nhead", 4),
        layers=bb_args.get("layers", 3), max_len=bb_args.get("max_len", 24),
    ).to(device)
    predictor.load_state_dict(bb_ck["model_state_dict"]); predictor.eval()
    lh_norm = float(bb_args.get("line_height", 0.05))
    max_len_pred = int(bb_args.get("max_len", 24))

    inf_ck = torch.load(args.infill_ckpt, map_location=device, weights_only=False)
    inf_args = inf_ck.get("args", {}) or {}
    noise_dim = int(inf_args.get("noise_dim", 0))
    use_print = bool(inf_args.get("print_glyphs"))
    gen = FractalInfiller(noise_dim=noise_dim, use_print_cond=use_print).to(device)
    gen.load_state_dict(inf_ck["gen_state_dict"]); gen.eval()
    pglyphs = None
    if args.print_glyphs and use_print:
        pc = torch.load(args.print_glyphs, weights_only=False)
        pglyphs = pc["glyphs"][args.print_font_idx].float().unsqueeze(1) / 255.0

    text = "".join(c for c in args.target_text if c in VOCAB)[:max_len_pred]
    style = style_from_seed(args.target_style)
    boxes_px = predict_boxes_px(text, predictor, device,
                                line_height=lh_norm, max_len=max_len_pred)
    word_arr, _ = render_word(text, boxes_px, gen, device,
                              style_index=style, print_glyphs=pglyphs)
    x0 = torch.from_numpy(word_arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # ----- 4. Refine pixels under D's score -----
    frame_dir = out_dir / f"frames_{text}"
    print(f"Refining '{text}' for {args.refine_steps} steps…")
    refine_pixels(x0, D, args.refine_steps, args.refine_lr, device, frame_dir)

    # ----- 5. Compose gif -----
    make_gif(frame_dir, out_dir / f"refine_{text}.gif")


if __name__ == "__main__":
    main()
