#!/usr/bin/env python3
"""
Train an online word-level handwriting scribe.

Architecture: OnlineWordScribe
  - SmallCNNStateEncoder: grayscale page-state → spatial features
  - TextEncoder: byte-level transformer encoder for target word
  - StyleEmbedding: per-writer learned embedding
  - EventDecoder: transformer decoder producing (dx,dy) + 5 binary flags

Loss: lambda_dxdy * SmoothL1(dxdy) + lambda_flags * BCE(flags) + lambda_raster * raster

Features:
  - Scheduled sampling (teacher forcing → model sampling)
  - Differentiable point-splat rasterizer for pixel-level supervision
  - Writer-disjoint train/val splits
  - Per-example confidence and source weighting

Usage:
    python train_online_scribe.py \
        --train-jsonl data/hw_v1/train_word_routes.jsonl \
        --val-jsonl data/hw_v1/val_word_routes.jsonl \
        --image-root data/hw_v1 \
        --out-dir runs/scribe_v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


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


def load_gray_image(path: str | Path) -> Tensor:
    with Image.open(path) as img:
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]


def pad_1d_long(seqs: Sequence[Tensor], pad_value: int = 0) -> Tensor:
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s.numel() > 0:
            out[i, : s.numel()] = s
    return out


def pad_events(seqs: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    if not seqs:
        return torch.empty(0, 0, 0), torch.empty(0, 0, dtype=torch.bool)
    feat_dim = seqs[0].shape[-1]
    max_len = max(s.shape[0] for s in seqs)
    batch = len(seqs)
    padded = torch.zeros(batch, max_len, feat_dim, dtype=seqs[0].dtype)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        t = s.shape[0]
        padded[i, :t] = s
        mask[i, :t] = True
    return padded, mask


def build_style_vocab(*record_lists: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    ids = {"__unk__"}
    for records in record_lists:
        for rec in records:
            sid = rec.get("style_id", rec.get("writer_id", "__unk__"))
            if sid is None:
                sid = "__unk__"
            ids.add(str(sid))
    ids_sorted = sorted(ids)
    return {sid: i for i, sid in enumerate(ids_sorted)}


# ============================================================
# Token-budget batch sampler
# ============================================================

class TokenBudgetBatchSampler(Sampler):
    """Yields batches where max_seq_len * batch_size <= token_budget.

    Sorts by sequence length within random chunks so that sequences of
    similar length are grouped together, minimising padding waste while
    preserving approximate randomness across epochs.
    """

    def __init__(
        self,
        seq_lengths: Sequence[int],
        token_budget: int,
        max_batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.seq_lengths = list(seq_lengths)
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        n = len(self.seq_lengths)
        indices = list(range(n))

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            # Shuffle in megabatch chunks, then sort each chunk by length
            # to group similar lengths while keeping global order random.
            mega = self.max_batch_size * 50
            rng.shuffle(indices)
            sorted_chunks = []
            for i in range(0, n, mega):
                chunk = indices[i : i + mega]
                chunk.sort(key=lambda idx: self.seq_lengths[idx])
                sorted_chunks.extend(chunk)
            indices = sorted_chunks

        batch: List[int] = []
        batch_max_len = 0
        for idx in indices:
            slen = self.seq_lengths[idx]
            new_max = max(batch_max_len, slen)
            new_cost = new_max * (len(batch) + 1)
            if batch and (new_cost > self.token_budget or len(batch) >= self.max_batch_size):
                yield batch
                batch = [idx]
                batch_max_len = slen
            else:
                batch.append(idx)
                batch_max_len = new_max
        if batch:
            yield batch

    def __len__(self):
        # Approximate — exact count varies per epoch due to shuffling.
        total_tokens = sum(self.seq_lengths)
        avg_len = total_tokens / max(1, len(self.seq_lengths))
        avg_batch = min(self.max_batch_size, max(1, int(self.token_budget / avg_len)))
        return math.ceil(len(self.seq_lengths) / avg_batch)


# ============================================================
# Dataset
# ============================================================

class WordRouteDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path | None,
        image_root: str | Path,
        style_to_index: Dict[str, int],
        max_text_len: int = 64,
        require_state_after: bool = False,
        preloaded_records: List[Dict] | None = None,
    ) -> None:
        self.records = preloaded_records if preloaded_records is not None else read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index
        self.max_text_len = max_text_len
        self.require_state_after = require_state_after

    def __len__(self) -> int:
        return len(self.records)

    def seq_lengths(self) -> List[int]:
        return [len(rec["events"]) for rec in self.records]

    def _text_to_bytes(self, text: str) -> Tensor:
        return torch.tensor(list(text.encode("utf-8"))[: self.max_text_len], dtype=torch.long)

    def _events_to_tensor(self, events: Sequence[Dict[str, Any]]) -> Tensor:
        rows = []
        for e in events:
            rows.append([
                float(e["dx"]),
                float(e["dy"]),
                float(e["pen_down"]),
                float(e["stroke_end"]),
                float(e["char_end"]),
                float(e["word_end"]),
                float(e["seq_end"]),
            ])
        if not rows:
            return torch.zeros(0, 7, dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        # Load state image or create blank
        state_before_ref = rec.get("state_before_ref")
        if state_before_ref is not None:
            state_path = self.image_root / state_before_ref
            if state_path.exists():
                state_before = load_gray_image(state_path)
            else:
                state_before = torch.ones(1, 256, 256, dtype=torch.float32)
        else:
            state_before = torch.ones(1, 256, 256, dtype=torch.float32)

        state_after = None
        has_state_after = False
        state_after_ref = rec.get("state_after_ref")
        if state_after_ref is not None:
            state_path = self.image_root / state_after_ref
            if state_path.exists():
                state_after = load_gray_image(state_path)
                has_state_after = True

        if self.require_state_after and not has_state_after:
            raise ValueError(f"Missing state_after for record {rec.get('record_id')}")

        style_id = rec.get("style_id", rec.get("writer_id", "__unk__"))
        if style_id is None:
            style_id = "__unk__"
        style_index = self.style_to_index.get(str(style_id), self.style_to_index["__unk__"])

        route_start = rec.get("route_start_point")
        if route_start is None:
            route_start_point = torch.zeros(2, dtype=torch.float32)
            route_start_valid = torch.tensor(0.0, dtype=torch.float32)
        else:
            route_start_point = torch.tensor(route_start, dtype=torch.float32)
            route_start_valid = torch.tensor(1.0, dtype=torch.float32)

        return {
            "record_id": rec["record_id"],
            "document_id": rec["document_id"],
            "line_id": rec["line_id"],
            "word_id": rec["word_id"],
            "word_index": int(rec["word_index"]),
            "word_text": rec["word_text"],
            "text_tokens": self._text_to_bytes(rec["word_text"]),
            "bbox": torch.tensor(rec["target_bbox_parent_norm_cxcywh"], dtype=torch.float32),
            "state_before": state_before,
            "state_after": state_after,
            "has_state_after": torch.tensor(float(has_state_after), dtype=torch.float32),
            "route_start_point": route_start_point,
            "route_start_valid": route_start_valid,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "events": self._events_to_tensor(rec["events"]),
            "plan_cost": torch.tensor(float(rec.get("plan_cost", 0.0)), dtype=torch.float32),
            "source_weight": torch.tensor(float(rec.get("source_weight", 1.0)), dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
            "source_type": rec.get("source_type", "pseudo_online"),
        }


def collate_word_routes(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    events, event_mask = pad_events([b["events"] for b in batch])

    out: Dict[str, Any] = {
        "record_id": [b["record_id"] for b in batch],
        "document_id": [b["document_id"] for b in batch],
        "line_id": [b["line_id"] for b in batch],
        "word_id": [b["word_id"] for b in batch],
        "word_index": torch.tensor([b["word_index"] for b in batch], dtype=torch.long),
        "word_text": [b["word_text"] for b in batch],
        "text_tokens": pad_1d_long([b["text_tokens"] for b in batch], pad_value=0),
        "text_lengths": torch.tensor([b["text_tokens"].numel() for b in batch], dtype=torch.long),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "state_before": torch.stack([b["state_before"] for b in batch], dim=0),
        "has_state_after": torch.stack([b["has_state_after"] for b in batch], dim=0),
        "route_start_point": torch.stack([b["route_start_point"] for b in batch], dim=0),
        "route_start_valid": torch.stack([b["route_start_valid"] for b in batch], dim=0),
        "style_index": torch.stack([b["style_index"] for b in batch], dim=0),
        "events": events,
        "event_mask": event_mask,
        "plan_cost": torch.stack([b["plan_cost"] for b in batch], dim=0),
        "source_weight": torch.stack([b["source_weight"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
        "source_type": [b["source_type"] for b in batch],
    }

    if all(b["state_after"] is not None for b in batch):
        out["state_after"] = torch.stack([b["state_after"] for b in batch], dim=0)

    return out


# ============================================================
# Model
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.shape[1]]


class SmallCNNStateEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, d_model: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.net(x)                       # [B,D,H',W']
        feat = feat.flatten(2).transpose(1, 2)   # [B,S,D]
        return feat


class CanvasFeedbackEncoder(nn.Module):
    """Encodes a raster canvas of strokes-so-far into a d_model vector."""

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [N, 1, H, W] → [N, d_model]"""
        return self.net(x)


@torch.no_grad()
def rasterize_strokes_to_canvas(
    abs_pos: Tensor,
    pen_down: Tensor,
    canvas_size: int = 64,
) -> Tensor:
    """Non-differentiable point-splat rasterizer.

    abs_pos: [T, 2] in normalized coords
    pen_down: [T] binary
    Returns: [1, canvas_size, canvas_size] canvas image
    """
    canvas = torch.zeros(canvas_size, canvas_size, device=abs_pos.device)
    if abs_pos.shape[0] == 0:
        return canvas.unsqueeze(0)
    mask = pen_down > 0.5
    if mask.sum() == 0:
        return canvas.unsqueeze(0)
    pts = abs_pos[mask]
    px = (pts[:, 0] * (canvas_size - 1)).long().clamp(0, canvas_size - 1)
    py = (pts[:, 1] * (canvas_size - 1)).long().clamp(0, canvas_size - 1)
    canvas[py, px] = 1.0
    return canvas.unsqueeze(0)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 256, nhead: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        x = self.embed(tokens)
        x = self.pos(x)
        max_len = tokens.shape[1]
        ar = torch.arange(max_len, device=tokens.device).unsqueeze(0)
        key_padding_mask = ar >= lengths.unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


class EventDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        event_dim: int = 7,
    ) -> None:
        super().__init__()
        self.event_in = nn.Linear(event_dim, d_model)
        self.bbox_in = nn.Linear(4, d_model)
        self.style_in = nn.Linear(d_model, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)

        # Absolute pen position conditioning (2D: x, y)
        self.abs_pos_in = nn.Linear(2, d_model)
        # Canvas feedback conditioning
        self.canvas_in = nn.Linear(d_model, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.dxdy_head = nn.Linear(d_model, 2)
        self.flag_head = nn.Linear(d_model, 4)  # pen_down, stroke_end, char_end, word_end

        # Separate seq_end head: local hidden + cumulative mean summary → MLP
        self.seq_end_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def causal_mask(self, t: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        prev_events: Tensor,   # [B,T,7]
        bbox: Tensor,          # [B,4]
        style_vec: Tensor,     # [B,D]
        memory: Tensor,        # [B,S,D]
        abs_pos: Tensor,       # [B,T,2] — absolute pen position
        canvas_feat: Tensor,   # [B,T,D] — canvas feedback features
    ) -> Dict[str, Tensor]:
        x = self.event_in(prev_events)
        x = x + self.bbox_in(bbox).unsqueeze(1)
        x = x + self.style_in(style_vec).unsqueeze(1)
        x = x + self.abs_pos_in(abs_pos)
        x = x + self.canvas_in(canvas_feat)
        x = self.pos(x)

        tgt_mask = self.causal_mask(prev_events.shape[1], prev_events.device)
        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        y = self.norm(y)

        # Causal cumulative mean: summary[t] = mean(y[0..t])
        cum_sum = torch.cumsum(y, dim=1)  # [B, T, D]
        step_idx = torch.arange(1, y.shape[1] + 1, device=y.device, dtype=y.dtype)
        cum_mean = cum_sum / step_idx.view(1, -1, 1)  # [B, T, D]

        seq_end_in = torch.cat([y, cum_mean], dim=-1)  # [B, T, 2*D]
        seq_end_logit = self.seq_end_head(seq_end_in)  # [B, T, 1]

        return {
            "dxdy": self.dxdy_head(y),
            "flags": self.flag_head(y),           # [B, T, 4]
            "seq_end": seq_end_logit.squeeze(-1),  # [B, T]
        }


class OnlineWordScribe(nn.Module):
    def __init__(
        self,
        num_styles: int,
        d_model: int = 256,
        nhead: int = 8,
        text_layers: int = 2,
        dec_layers: int = 6,
        dxdy_scale: float = 100.0,
        canvas_size: int = 64,
        snapshot_interval: int = 16,
    ) -> None:
        super().__init__()
        self.dxdy_scale = dxdy_scale
        self.canvas_size = canvas_size
        self.snapshot_interval = snapshot_interval
        self.state_encoder = SmallCNNStateEncoder(in_ch=1, d_model=d_model)
        self.text_encoder = TextEncoder(vocab_size=256, d_model=d_model, nhead=nhead, num_layers=text_layers)
        self.style_embed = nn.Embedding(num_styles, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.decoder = EventDecoder(d_model=d_model, nhead=nhead, num_layers=dec_layers, event_dim=7)
        self.canvas_encoder = CanvasFeedbackEncoder(d_model=d_model)

    def encode_context(
        self,
        state_before: Tensor,
        text_tokens: Tensor,
        text_lengths: Tensor,
        style_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        state_mem = self.state_encoder(state_before)
        text_mem = self.text_encoder(text_tokens, text_lengths)
        style_vec = self.style_embed(style_index)
        style_token = style_vec.unsqueeze(1)
        memory = torch.cat([style_token, state_mem, text_mem], dim=1)
        memory = self.memory_proj(memory)
        return memory, style_vec

    def _scale_events_in(self, events: Tensor) -> Tensor:
        """Scale dx/dy up for internal representation."""
        scaled = events.clone()
        scaled[:, :, 0:2] = scaled[:, :, 0:2] * self.dxdy_scale
        return scaled

    def _compute_abs_pos(self, prev_events: Tensor, start_point: Tensor) -> Tensor:
        """Compute absolute pen position for each timestep.

        prev_events: [B, T, 7] in original (unscaled) coordinates
        start_point: [B, 2] route start point in normalized coords
        Returns: [B, T, 2] absolute pen positions
        """
        dxdy = prev_events[:, :, 0:2]  # [B, T, 2]
        cum_dxdy = torch.cumsum(dxdy, dim=1)  # [B, T, 2]
        abs_pos = start_point.unsqueeze(1) + cum_dxdy  # [B, T, 2]
        return abs_pos

    def _compute_canvas_features(
        self,
        abs_pos: Tensor,      # [B, T, 2]
        pen_down: Tensor,     # [B, T]
    ) -> Tensor:
        """Compute canvas feedback features via periodic snapshots.

        Returns: [B, T, D] canvas features for each timestep.
        """
        B, T, _ = abs_pos.shape
        device = abs_pos.device
        K = self.snapshot_interval

        snap_times = list(range(0, T + 1, K))
        if snap_times[-1] > T:
            snap_times[-1] = T
        if snap_times[-1] < T:
            snap_times.append(T)
        S = len(snap_times)

        # Rasterize prefix at each snapshot (non-differentiable)
        with torch.no_grad():
            canvases = []
            for b in range(B):
                for t in snap_times:
                    if t == 0:
                        canvas = torch.zeros(
                            1, self.canvas_size, self.canvas_size, device=device
                        )
                    else:
                        canvas = rasterize_strokes_to_canvas(
                            abs_pos[b, :t], pen_down[b, :t], self.canvas_size
                        )
                    canvases.append(canvas)
            canvases = torch.stack(canvases)  # [B*S, 1, H, W]

        # Encode with CNN (differentiable — gradients flow to canvas_encoder)
        features = self.canvas_encoder(canvases)  # [B*S, D]
        D = features.shape[-1]
        features = features.view(B, S, D)

        # Assign features to timesteps
        out_parts = []
        for i in range(S - 1):
            dur = snap_times[i + 1] - snap_times[i]
            if dur > 0:
                out_parts.append(features[:, i : i + 1].expand(-1, dur, -1))
        # Handle remainder if snap_times[-1] < T
        remaining = T - snap_times[-2] if len(snap_times) > 1 else T
        if remaining > 0 and len(out_parts) > 0:
            pass  # already handled by the loop above
        elif remaining > 0:
            out_parts.append(features[:, -1:].expand(-1, remaining, -1))

        out = torch.cat(out_parts, dim=1)  # [B, T, D]
        return out

    def compute_canvas_features(
        self,
        events: Tensor,        # [B, T, 7]
        start_point: Tensor,   # [B, 2]
    ) -> Tensor:
        """Compute canvas feedback from a trajectory. Returns [B, T, D]."""
        abs_pos = self._compute_abs_pos(events, start_point)
        pen_down = events[:, :, 2]
        return self._compute_canvas_features(abs_pos, pen_down)

    def decode(
        self,
        prev_events: Tensor,
        bbox: Tensor,
        style_vec: Tensor,
        memory: Tensor,
        start_point: Tensor,
        canvas_feat: Tensor,
    ) -> Dict[str, Tensor]:
        abs_pos = self._compute_abs_pos(prev_events, start_point)
        out = self.decoder(
            prev_events=self._scale_events_in(prev_events),
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
            abs_pos=abs_pos,
            canvas_feat=canvas_feat,
        )
        # dxdy stays in scaled space — loss computed in scaled space,
        # caller must divide by dxdy_scale for real coordinates
        return out


# ============================================================
# Differentiable segment rasterizer
# ============================================================

def soft_rasterize_segments(
    start_points: Tensor,   # [B,2], normalized [0,1]
    dxdy: Tensor,           # [B,T,2]
    pen_probs: Tensor,      # [B,T], in [0,1]
    raster_size: int = 64,
    sigma_px: float = 1.25,
    mode: str = "max",
) -> Tensor:
    """
    Differentiable line-segment rasterizer.

    Instead of splatting Gaussians at individual points, computes the
    minimum distance from each raster pixel to each line segment between
    consecutive pen-down positions.  This produces continuous ink traces
    for long pen-down strokes.
    """
    _, _, _ = dxdy.shape
    device = dxdy.device
    eps = 1e-6

    pos = start_points.unsqueeze(1) + torch.cumsum(dxdy, dim=1)  # [B,T,2]

    seg_start = torch.cat([start_points.unsqueeze(1), pos[:, :-1, :]], dim=1)  # [B,T,2]
    seg_end = pos                                                               # [B,T,2]

    scale = float(raster_size - 1)
    a = seg_start * scale
    bpt = seg_end * scale

    grid_y, grid_x = torch.meshgrid(
        torch.arange(raster_size, device=device, dtype=torch.float32),
        torch.arange(raster_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    p = torch.stack([grid_x, grid_y], dim=-1)              # [R,R,2]
    p = p.view(1, 1, raster_size, raster_size, 2)          # [1,1,R,R,2]

    a = a.unsqueeze(2).unsqueeze(2)                        # [B,T,1,1,2]
    bpt = bpt.unsqueeze(2).unsqueeze(2)                    # [B,T,1,1,2]
    ab = bpt - a                                           # [B,T,1,1,2]
    ap = p - a                                             # [B,T,R,R,2]

    ab_len2 = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(eps)
    tau = (ap * ab).sum(dim=-1, keepdim=True) / ab_len2
    tau = tau.clamp(0.0, 1.0)

    closest = a + tau * ab
    dist2 = ((p - closest) ** 2).sum(dim=-1)              # [B,T,R,R]

    gauss = torch.exp(-dist2 / (2.0 * sigma_px * sigma_px))
    contrib = pen_probs.unsqueeze(-1).unsqueeze(-1) * gauss

    if mode == "max":
        raster = torch.amax(contrib, dim=1, keepdim=True)
    elif mode == "alpha":
        contrib = contrib.clamp(0.0, 1.0 - 1e-5)
        raster = 1.0 - torch.prod(1.0 - contrib, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown rasterization mode: {mode}")

    return raster.clamp(0.0, 1.0)


# ============================================================
# Loss
# ============================================================

def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Focal loss for binary classification (per-element, unreduced)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return focal_weight * bce


def compute_sequence_loss(
    pred_dxdy: Tensor,
    pred_flags_logits: Tensor,
    pred_seq_end_logits: Tensor,
    target_events: Tensor,
    event_mask: Tensor,
    example_weights: Tensor,
    seq_end_pos_weight: float = 1.0,
    dxdy_scale: float = 1.0,
    seq_end_focal: bool = True,
    dx_weight: float = 1.0,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    dxdy_tgt = target_events[:, :, 0:2] * dxdy_scale
    flags_tgt = target_events[:, :, 2:6]  # pen_down, stroke_end, char_end, word_end
    seq_end_tgt = target_events[:, :, 6]  # seq_end

    mask = event_mask.float()
    wt = mask * example_weights.unsqueeze(1)

    # Separate x/y huber with dx upweighting
    huber_xy = F.smooth_l1_loss(pred_dxdy, dxdy_tgt, reduction="none")  # [B, T, 2]
    huber_weighted = huber_xy[:, :, 0] * dx_weight + huber_xy[:, :, 1]
    dxdy_loss = (huber_weighted * wt).sum() / wt.sum().clamp_min(1.0)

    # 4 structural flags: standard BCE
    bce = F.binary_cross_entropy_with_logits(
        pred_flags_logits, flags_tgt, reduction="none",
    )
    per_flag = []
    for i in range(4):
        li = (bce[:, :, i] * wt).sum() / wt.sum().clamp_min(1.0)
        per_flag.append(li)

    # seq_end: focal loss (handles extreme imbalance better than pos_weight)
    if seq_end_focal:
        se_loss_raw = focal_bce_with_logits(
            pred_seq_end_logits, seq_end_tgt,
            alpha=0.75,   # upweight the rare positive class
            gamma=2.0,
        )
    else:
        pw = torch.tensor(seq_end_pos_weight, device=pred_seq_end_logits.device)
        se_loss_raw = F.binary_cross_entropy_with_logits(
            pred_seq_end_logits, seq_end_tgt, reduction="none",
            pos_weight=pw,
        )
    se_loss = (se_loss_raw * wt).sum() / wt.sum().clamp_min(1.0)
    per_flag.append(se_loss)

    flags_loss = sum(per_flag) / 5.0
    return dxdy_loss, flags_loss, per_flag


def compute_raster_loss(
    pred_dxdy: Tensor,
    pred_flags_logits: Tensor,
    target_events: Tensor,
    route_start_point: Tensor,
    route_start_valid: Tensor,
    example_weights: Tensor,
    raster_size: int,
    sigma_px: float,
    dxdy_scale: float = 1.0,
) -> Tensor:
    """Raster loss: compare predicted vs gold stroke rasterizations."""
    valid = route_start_valid > 0.5
    if valid.sum() == 0:
        return pred_dxdy.new_tensor(0.0)

    pred_dxdy_v = pred_dxdy[valid] / dxdy_scale
    pred_pen_v = torch.sigmoid(pred_flags_logits[valid, :, 0])
    gold_dxdy_v = target_events[valid, :, 0:2]
    gold_pen_v = target_events[valid, :, 2]
    start_v = route_start_point[valid]
    ex_w = example_weights[valid]

    pred_raster = soft_rasterize_segments(
        start_points=start_v,
        dxdy=pred_dxdy_v,
        pen_probs=pred_pen_v,
        raster_size=raster_size,
        sigma_px=sigma_px,
    )

    with torch.no_grad():
        gold_raster = soft_rasterize_segments(
            start_points=start_v,
            dxdy=gold_dxdy_v,
            pen_probs=gold_pen_v,
            raster_size=raster_size,
            sigma_px=sigma_px,
        )

    per_example = torch.abs(pred_raster - gold_raster).mean(dim=(1, 2, 3))
    raster_loss = (per_example * ex_w).sum() / ex_w.sum().clamp_min(1.0)
    return raster_loss


# ============================================================
# Autoregressive rollout with scheduled sampling
# ============================================================

def rollout_autoregressive(
    model: OnlineWordScribe,
    state_before: Tensor,
    text_tokens: Tensor,
    text_lengths: Tensor,
    bbox: Tensor,
    style_index: Tensor,
    route_start_point: Tensor,
    target_events: Tensor,
    teacher_prob: float,
    sample_model_inputs: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    b, t, _ = target_events.shape
    device = target_events.device

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    # Compute canvas features from gold trajectory (teacher-forced visual context)
    canvas_feat = model.compute_canvas_features(target_events, route_start_point)

    # Build teacher-forced input: [zero_start, gold[0], gold[1], ..., gold[T-2]]
    zero_start = torch.zeros(b, 1, 7, device=device)
    gold_input = torch.cat([zero_start, target_events[:, :-1, :]], dim=1)

    # Fast path: pure teacher forcing — single parallel forward pass
    if teacher_prob >= 1.0:
        out = model.decode(
            prev_events=gold_input,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
            start_point=route_start_point,
            canvas_feat=canvas_feat,
        )
        return out["dxdy"], out["flags"], out["seq_end"]

    # Two-pass scheduled sampling (memory-efficient):
    # Pass 1: detached forward to get model predictions
    with torch.no_grad():
        out_detached = model.decode(
            prev_events=gold_input,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
            start_point=route_start_point,
            canvas_feat=canvas_feat,
        )
        pred_events_detached = torch.cat([
            out_detached["dxdy"] / model.dxdy_scale,  # unscale to original coords
            torch.sigmoid(out_detached["flags"]),
            torch.sigmoid(out_detached["seq_end"].unsqueeze(-1)),
        ], dim=-1)  # [B, T, 7]

    # Build mixed input: per-timestep random choice of gold vs model prediction
    # (applied to the *input* sequence, shifted by 1)
    pred_input = torch.cat([zero_start, pred_events_detached[:, :-1, :]], dim=1)
    use_teacher = (torch.rand(b, t, 1, device=device) < teacher_prob).float()
    mixed_input = use_teacher * gold_input + (1.0 - use_teacher) * pred_input

    # Pass 2: forward with mixed input (with gradients)
    out = model.decode(
        prev_events=mixed_input,
        bbox=bbox,
        style_vec=style_vec,
        memory=memory,
        start_point=route_start_point,
        canvas_feat=canvas_feat,
    )
    return out["dxdy"], out["flags"], out["seq_end"]


@torch.no_grad()
def greedy_decode(
    model: OnlineWordScribe,
    state_before: Tensor,
    text_tokens: Tensor,
    text_lengths: Tensor,
    bbox: Tensor,
    style_index: Tensor,
    route_start_point: Tensor,
    max_steps: int = 4096,
    stop_threshold: float = 0.3,
) -> Tensor:
    model.eval()
    device = state_before.device
    cs = model.canvas_size

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    prev_events = torch.zeros(1, 1, 7, device=device)
    generated: List[Tensor] = []

    # Maintain running canvas and its CNN encoding
    canvas = torch.zeros(1, 1, cs, cs, device=device)
    canvas_feat_vec = model.canvas_encoder(canvas)  # [1, D]
    abs_x = float(route_start_point[0, 0])
    abs_y = float(route_start_point[0, 1])

    for step in range(max_steps):
        T = prev_events.shape[1]
        cf = canvas_feat_vec.unsqueeze(1).expand(1, T, -1)  # [1, T, D]

        out = model.decode(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
            start_point=route_start_point,
            canvas_feat=cf,
        )

        dxdy_scaled = out["dxdy"][:, -1, :]
        dxdy = dxdy_scaled / model.dxdy_scale  # unscale to original coords
        flags_logits = out["flags"][:, -1, :]
        flags = torch.sigmoid(flags_logits)
        seq_end_prob = torch.sigmoid(out["seq_end"][:, -1])

        # Reconstruct 7-dim event: [dx, dy, pen_down, stroke_end, char_end, word_end, seq_end]
        next_evt = torch.cat([dxdy, flags, seq_end_prob.unsqueeze(-1)], dim=-1)
        generated.append(next_evt[0].detach().cpu())

        prev_events = torch.cat([prev_events, next_evt.unsqueeze(1)], dim=1)

        # Update absolute position and canvas
        abs_x += float(dxdy[0, 0])
        abs_y += float(dxdy[0, 1])
        pen = float(flags[0, 0])
        if pen > 0.5:
            px = max(0, min(cs - 1, int(abs_x * (cs - 1))))
            py = max(0, min(cs - 1, int(abs_y * (cs - 1))))
            canvas[0, 0, py, px] = 1.0

        # Re-encode canvas periodically
        if (step + 1) % model.snapshot_interval == 0:
            canvas_feat_vec = model.canvas_encoder(canvas)

        if bool(seq_end_prob[0].item() > stop_threshold):
            break

    if not generated:
        return torch.zeros(0, 7)
    return torch.stack(generated, dim=0)


# ============================================================
# Training / Eval
# ============================================================

def teacher_prob_for_epoch(epoch: int, total_epochs: int, ss_start_epoch: int, ss_final_teacher_prob: float) -> float:
    if epoch < ss_start_epoch:
        return 1.0
    if total_epochs <= ss_start_epoch:
        return ss_final_teacher_prob
    progress = (epoch - ss_start_epoch) / max(1, total_epochs - ss_start_epoch)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 + progress * (ss_final_teacher_prob - 1.0)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_epoch(
    model: OnlineWordScribe,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    lambda_dxdy: float,
    lambda_flags: float,
    lambda_raster: float,
    teacher_prob: float,
    scheduled_sampling_active: bool,
    raster_size: int,
    raster_sigma_px: float,
    max_seq_len: int = 2048,
    seq_end_pos_weight: float = 1.0,
    seq_end_focal: bool = True,
    dx_weight: float = 1.0,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    sums = {
        "total": 0.0,
        "dxdy": 0.0,
        "flags": 0.0,
        "raster": 0.0,
        "pen_down": 0.0,
        "stroke_end": 0.0,
        "char_end": 0.0,
        "word_end": 0.0,
        "seq_end": 0.0,
        "count": 0,
    }

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        # Truncate very long sequences to avoid OOM
        events = batch["events"][:, :max_seq_len, :]
        event_mask = batch["event_mask"][:, :max_seq_len]
        batch["events"] = events
        batch["event_mask"] = event_mask

        if batch_idx == 0 and train:
            print(f"    First batch: bs={events.shape[0]} seq_len={events.shape[1]}")

        pred_dxdy, pred_flags_logits, pred_seq_end_logits = rollout_autoregressive(
            model=model,
            state_before=batch["state_before"],
            text_tokens=batch["text_tokens"],
            text_lengths=batch["text_lengths"],
            bbox=batch["bbox"],
            style_index=batch["style_index"],
            route_start_point=batch["route_start_point"],
            target_events=batch["events"],
            teacher_prob=teacher_prob,
            sample_model_inputs=scheduled_sampling_active and train,
        )

        example_weights = batch["source_weight"] * batch["confidence"]

        dxdy_loss, flags_loss, per_flag = compute_sequence_loss(
            pred_dxdy=pred_dxdy,
            pred_flags_logits=pred_flags_logits,
            pred_seq_end_logits=pred_seq_end_logits,
            target_events=batch["events"],
            event_mask=batch["event_mask"],
            example_weights=example_weights,
            seq_end_pos_weight=seq_end_pos_weight,
            dxdy_scale=model.dxdy_scale,
            seq_end_focal=seq_end_focal,
            dx_weight=dx_weight,
        )

        raster_loss = compute_raster_loss(
            pred_dxdy=pred_dxdy,
            pred_flags_logits=pred_flags_logits,
            target_events=batch["events"],
            route_start_point=batch["route_start_point"],
            route_start_valid=batch["route_start_valid"],
            example_weights=example_weights,
            raster_size=raster_size,
            sigma_px=raster_sigma_px,
            dxdy_scale=model.dxdy_scale,
        )

        total = (
            lambda_dxdy * dxdy_loss
            + lambda_flags * flags_loss
            + lambda_raster * raster_loss
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = events.shape[0]
        sums["total"] += float(total.detach().cpu()) * bs
        sums["dxdy"] += float(dxdy_loss.detach().cpu()) * bs
        sums["flags"] += float(flags_loss.detach().cpu()) * bs
        sums["raster"] += float(raster_loss.detach().cpu()) * bs
        sums["pen_down"] += float(per_flag[0].detach().cpu()) * bs
        sums["stroke_end"] += float(per_flag[1].detach().cpu()) * bs
        sums["char_end"] += float(per_flag[2].detach().cpu()) * bs
        sums["word_end"] += float(per_flag[3].detach().cpu()) * bs
        sums["seq_end"] += float(per_flag[4].detach().cpu()) * bs
        sums["count"] += bs

    count = max(1, sums["count"])
    return {k: (v / count if k != "count" else v) for k, v in sums.items()}


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: OnlineWordScribe,
    optimizer: torch.optim.Optimizer,
    style_to_index: Dict[str, int],
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
        "style_to_index": style_to_index,
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
    p = argparse.ArgumentParser(description="Train online word-level handwriting scribe")

    p.add_argument("--train-jsonl", required=True, nargs="+",
                    help="One or more JSONL files of training word routes")
    p.add_argument("--val-jsonl", required=True, nargs="+",
                    help="One or more JSONL files of validation word routes")
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--text-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=6)
    p.add_argument("--dxdy-scale", type=float, default=100.0,
                    help="Scale factor for dx/dy (original coords ~0.006 avg magnitude)")

    p.add_argument("--lambda-dxdy", type=float, default=1.0)
    p.add_argument("--lambda-flags", type=float, default=1.0)
    p.add_argument("--lambda-raster", type=float, default=0.5)

    p.add_argument("--raster-size", type=int, default=64)
    p.add_argument("--raster-sigma-px", type=float, default=1.5)

    p.add_argument("--ss-start-epoch", type=int, default=5)
    p.add_argument("--ss-final-teacher-prob", type=float, default=0.2)

    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--token-budget", type=int, default=32768,
                    help="Max total tokens (max_seq_in_batch * batch_size) per batch")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-decode-steps", type=int, default=4096)
    p.add_argument("--seq-end-pos-weight", type=float, default=100.0,
                    help="BCE pos_weight for seq_end flag (fallback if --no-seq-end-focal)")
    p.add_argument("--no-seq-end-focal", action="store_true",
                    help="Use pos_weight BCE instead of focal loss for seq_end")
    p.add_argument("--stop-threshold", type=float, default=0.3,
                    help="seq_end probability threshold for greedy decode stopping")

    p.add_argument("--dx-weight", type=float, default=2.0,
                    help="Loss weight multiplier for dx (horizontal) vs dy (vertical)")
    p.add_argument("--canvas-size", type=int, default=64,
                    help="Canvas resolution for stroke feedback rasterization")
    p.add_argument("--snapshot-interval", type=int, default=16,
                    help="Re-encode canvas every N timesteps")
    p.add_argument("--curriculum-end-epoch", type=int, default=10,
                    help="Epoch by which all word lengths are included (0=disabled)")
    p.add_argument("--curriculum-min-chars", type=int, default=3,
                    help="Max word length (chars) at epoch 1 of curriculum")

    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint .pt file to resume training from")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records: List[Dict] = []
    for p in args.train_jsonl:
        recs = read_jsonl(p)
        print(f"  Loaded {len(recs)} train records from {p}")
        train_records.extend(recs)
    val_records: List[Dict] = []
    for p in args.val_jsonl:
        recs = read_jsonl(p)
        print(f"  Loaded {len(recs)} val records from {p}")
        val_records.extend(recs)
    print(f"  Total: {len(train_records)} train, {len(val_records)} val")

    style_to_index = build_style_vocab(train_records, val_records)
    print(f"  {len(style_to_index)} styles (including __unk__)")

    with (out_dir / "style_to_index.json").open("w", encoding="utf-8") as f:
        json.dump(style_to_index, f, ensure_ascii=False, indent=2)

    train_ds = WordRouteDataset(
        jsonl_path=None,
        image_root=args.image_root,
        style_to_index=style_to_index,
        require_state_after=False,
        preloaded_records=train_records,
    )
    val_ds = WordRouteDataset(
        jsonl_path=None,
        image_root=args.image_root,
        style_to_index=style_to_index,
        require_state_after=False,
        preloaded_records=val_records,
    )

    train_sampler = TokenBudgetBatchSampler(
        seq_lengths=train_ds.seq_lengths(),
        token_budget=args.token_budget,
        max_batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_sampler = TokenBudgetBatchSampler(
        seq_lengths=val_ds.seq_lengths(),
        token_budget=args.token_budget,
        max_batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_word_routes,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_word_routes,
    )

    model = OnlineWordScribe(
        num_styles=len(style_to_index),
        d_model=args.d_model,
        nhead=args.nhead,
        text_layers=args.text_layers,
        dec_layers=args.dec_layers,
        dxdy_scale=args.dxdy_scale,
        canvas_size=args.canvas_size,
        snapshot_interval=args.snapshot_interval,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")
    print(f"Token budget: {args.token_budget}, max batch size: {args.batch_size}, ~{len(train_sampler)} train batches/epoch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Reload history if available
        history_path = out_dir / "history.json"
        if history_path.exists():
            with history_path.open() as f:
                history = json.load(f)
            best_val = min(h["val"]["total"] for h in history)
            print(f"  Restored {len(history)} history entries, best_val={best_val:.4f}")
        print(f"  Resuming from epoch {start_epoch}")

    # Pre-sort training records by word length for curriculum
    sorted_train_records = sorted(train_records, key=lambda r: len(r["word_text"]))
    max_word_len = max(len(r["word_text"]) for r in train_records)

    for epoch in range(start_epoch, args.epochs + 1):
        # Curriculum: gradually include longer words
        if args.curriculum_end_epoch > 0 and epoch <= args.curriculum_end_epoch:
            frac = epoch / args.curriculum_end_epoch
            cur_max_chars = int(args.curriculum_min_chars + frac * (max_word_len - args.curriculum_min_chars))
            cur_records = [r for r in sorted_train_records if len(r["word_text"]) <= cur_max_chars]
            if len(cur_records) < args.batch_size:
                cur_records = sorted_train_records[:args.batch_size]
        else:
            cur_max_chars = max_word_len
            cur_records = train_records

        cur_train_ds = WordRouteDataset(
            jsonl_path=None,
            image_root=args.image_root,
            style_to_index=style_to_index,
            require_state_after=False,
            preloaded_records=cur_records,
        )
        cur_train_sampler = TokenBudgetBatchSampler(
            seq_lengths=cur_train_ds.seq_lengths(),
            token_budget=args.token_budget,
            max_batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        )
        cur_train_loader = DataLoader(
            cur_train_ds,
            batch_sampler=cur_train_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_word_routes,
        )

        tp = teacher_prob_for_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            ss_start_epoch=args.ss_start_epoch,
            ss_final_teacher_prob=args.ss_final_teacher_prob,
        )

        print(f"\nEpoch {epoch}/{args.epochs}  teacher_prob={tp:.3f}  curriculum: {len(cur_records)} records (max {cur_max_chars if args.curriculum_end_epoch > 0 and epoch <= args.curriculum_end_epoch else max_word_len} chars)")
        print(f"    GPU allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

        seq_end_focal = not args.no_seq_end_focal

        train_metrics = run_epoch(
            model=model,
            loader=cur_train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            lambda_dxdy=args.lambda_dxdy,
            lambda_flags=args.lambda_flags,
            lambda_raster=args.lambda_raster,
            teacher_prob=tp,
            scheduled_sampling_active=True,
            raster_size=args.raster_size,
            raster_sigma_px=args.raster_sigma_px,
            max_seq_len=args.max_seq_len,
            seq_end_pos_weight=args.seq_end_pos_weight,
            seq_end_focal=seq_end_focal,
            dx_weight=args.dx_weight,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
            lambda_dxdy=args.lambda_dxdy,
            lambda_flags=args.lambda_flags,
            lambda_raster=args.lambda_raster,
            teacher_prob=1.0,
            scheduled_sampling_active=False,
            raster_size=args.raster_size,
            raster_sigma_px=args.raster_sigma_px,
            max_seq_len=args.max_seq_len,
            seq_end_pos_weight=args.seq_end_pos_weight,
            seq_end_focal=seq_end_focal,
            dx_weight=args.dx_weight,
        )

        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]

        ckpt_path = save_checkpoint(
            out_dir=out_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            style_to_index=style_to_index,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
            is_best=is_best,
        )

        print(f"  Train: total={train_metrics['total']:.4f}  dxdy={train_metrics['dxdy']:.4f}  flags={train_metrics['flags']:.4f}")
        print(f"  Val:   total={val_metrics['total']:.4f}  dxdy={val_metrics['dxdy']:.4f}  flags={val_metrics['flags']:.4f}")
        if is_best:
            print(f"  ** New best val loss: {best_val:.4f}")

        # Qualitative sample
        if len(val_ds) > 0:
            sample = val_ds[0]
            sample_state = sample["state_before"].unsqueeze(0).to(device)
            sample_text = sample["text_tokens"].unsqueeze(0).to(device)
            sample_text_len = torch.tensor([sample["text_tokens"].numel()], device=device)
            sample_bbox = sample["bbox"].unsqueeze(0).to(device)
            sample_style = sample["style_index"].unsqueeze(0).to(device)
            sample_start = sample["route_start_point"].unsqueeze(0).to(device)

            pred_events = greedy_decode(
                model=model,
                state_before=sample_state,
                text_tokens=sample_text,
                text_lengths=sample_text_len,
                bbox=sample_bbox,
                style_index=sample_style,
                route_start_point=sample_start,
                max_steps=args.max_decode_steps,
                stop_threshold=args.stop_threshold,
            )
            print(f"  Greedy decode: {pred_events.shape[0]} events for '{sample['word_text']}'")

        history.append({
            "epoch": epoch,
            "teacher_prob": tp,
            "train": train_metrics,
            "val": val_metrics,
        })

        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
