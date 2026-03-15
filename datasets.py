"""
PyTorch dataset classes for the palimpsest handwriting pipeline.

Covers all five dataset types from the design spec:
  1. HierarchyPlacementDataset  — placement actions at 6 levels
  2. CharacterInfillDataset     — local before/after raster patches
  3. OnlineSequenceDataset      — point-level online sequences (IAM-OnDB or pseudo)
  4. CharacterPlanDataset       — primitive stroke sets + candidate DP plans
  5. WordRouteDataset           — word-level stroke routes (in train_online_scribe.py)

The WordRouteDataset + collate_word_routes are in train_online_scribe.py since
they're tightly coupled to the training loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


# ============================================================
# Shared utilities
# ============================================================

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {e}") from e
    return records


def load_gray_image(path: str | Path) -> Tensor:
    path = Path(path)
    with Image.open(path) as img:
        img = img.convert("L")
        x = torch.from_numpy(np.array(img)).float() / 255.0
    return x.unsqueeze(0)  # [1,H,W]


def load_mask_image(path: str | Path) -> Tensor:
    x = load_gray_image(path)
    return (x > 0.5).float()


def text_to_byte_tensor(text: Optional[str], max_len: int = 256) -> Tensor:
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


# ============================================================
# 1. Hierarchy Placement Dataset
# ============================================================

class HierarchyPlacementDataset(Dataset):
    """
    Loads teacher-forced placement actions for one level.

    Expected schema fields:
      - trajectory_id
      - level
      - state_before_ref
      - semantic_condition.text
      - target.stop
      - target.bbox_parent_norm_cxcywh
      - target.rotation_deg (optional)
      - style_id (optional)
      - confidence (optional)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        style_to_index: Optional[Dict[str, int]] = None,
        image_transform: Optional[Callable[[Tensor], Tensor]] = None,
        max_text_len: int = 256,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index or {}
        self.image_transform = image_transform
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.records)

    def _style_index(self, style_id: Optional[str]) -> int:
        if style_id is None:
            return -1
        return self.style_to_index.get(style_id, -1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        state_ref = rec.get("state_before_ref")
        if state_ref is not None:
            state_path = self.image_root / state_ref
            if Path(state_path).exists():
                state = load_gray_image(state_path)
            else:
                state = torch.ones(1, 256, 256, dtype=torch.float32)
        else:
            state = torch.ones(1, 256, 256, dtype=torch.float32)

        if self.image_transform is not None:
            state = self.image_transform(state)

        sem = rec.get("semantic_condition", {})
        text = sem.get("text")
        text_tokens = text_to_byte_tensor(text, max_len=self.max_text_len)

        target = rec["target"]
        stop = torch.tensor(float(target["stop"]), dtype=torch.float32)

        bbox = target.get("bbox_parent_norm_cxcywh")
        if bbox is None:
            bbox_tensor = torch.full((4,), -1.0, dtype=torch.float32)
            bbox_valid = torch.tensor(0.0, dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
            bbox_valid = torch.tensor(1.0, dtype=torch.float32)

        rotation = target.get("rotation_deg")
        rotation_tensor = torch.tensor(
            -999.0 if rotation is None else float(rotation), dtype=torch.float32
        )
        rotation_valid = torch.tensor(0.0 if rotation is None else 1.0, dtype=torch.float32)

        return {
            "record_id": rec.get("trajectory_id", f"rec_{idx}"),
            "level": rec["level"],
            "state": state,
            "text_tokens": text_tokens,
            "style_index": torch.tensor(self._style_index(rec.get("style_id")), dtype=torch.long),
            "stop": stop,
            "bbox": bbox_tensor,
            "bbox_valid": bbox_valid,
            "rotation": rotation_tensor,
            "rotation_valid": rotation_valid,
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_hierarchy_placement(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    states = torch.stack([b["state"] for b in batch], dim=0)
    text_tokens = pad_1d_long([b["text_tokens"] for b in batch], pad_value=0)
    text_lengths = torch.tensor([b["text_tokens"].numel() for b in batch], dtype=torch.long)

    return {
        "record_id": [b["record_id"] for b in batch],
        "level": [b["level"] for b in batch],
        "state": states,
        "text_tokens": text_tokens,
        "text_lengths": text_lengths,
        "style_index": torch.stack([b["style_index"] for b in batch], dim=0),
        "stop": torch.stack([b["stop"] for b in batch], dim=0),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "bbox_valid": torch.stack([b["bbox_valid"] for b in batch], dim=0),
        "rotation": torch.stack([b["rotation"] for b in batch], dim=0),
        "rotation_valid": torch.stack([b["rotation_valid"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
    }


# ============================================================
# 2. Character Infill Dataset
# ============================================================

class CharacterInfillDataset(Dataset):
    """
    Loads local before/after raster records for the character infiller.

    Expected fields:
      - record_id
      - before_patch_ref
      - after_patch_ref
      - bbox_mask_ref
      - char_text
      - target_bbox_parent_norm_cxcywh
      - style_id (optional)
      - confidence (optional)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        style_to_index: Optional[Dict[str, int]] = None,
        image_transform: Optional[Callable[[Tensor], Tensor]] = None,
        max_text_len: int = 16,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index or {}
        self.image_transform = image_transform
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        before = load_gray_image(self.image_root / rec["before_patch_ref"])
        after = load_gray_image(self.image_root / rec["after_patch_ref"])
        bbox_mask = load_mask_image(self.image_root / rec["bbox_mask_ref"])

        # Optional neighbor mask
        neighbor_ref = rec.get("neighbor_mask_ref")
        if neighbor_ref is not None:
            neighbor_mask = load_mask_image(self.image_root / neighbor_ref)
        else:
            neighbor_mask = torch.zeros_like(bbox_mask)

        if self.image_transform is not None:
            before = self.image_transform(before)
            after = self.image_transform(after)

        char_tokens = text_to_byte_tensor(rec.get("char_text"), max_len=self.max_text_len)
        style_index = self.style_to_index.get(rec.get("style_id", "__unk__"), -1)

        return {
            "record_id": rec["record_id"],
            "before": before,
            "after": after,
            "bbox_mask": bbox_mask,
            "neighbor_mask": neighbor_mask,
            "char_tokens": char_tokens,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "bbox": torch.tensor(rec["target_bbox_parent_norm_cxcywh"], dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_character_infill(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_id": [b["record_id"] for b in batch],
        "before": torch.stack([b["before"] for b in batch], dim=0),
        "after": torch.stack([b["after"] for b in batch], dim=0),
        "bbox_mask": torch.stack([b["bbox_mask"] for b in batch], dim=0),
        "neighbor_mask": torch.stack([b["neighbor_mask"] for b in batch], dim=0),
        "char_tokens": pad_1d_long([b["char_tokens"] for b in batch], pad_value=0),
        "char_lengths": torch.tensor([b["char_tokens"].numel() for b in batch], dtype=torch.long),
        "style_index": torch.stack([b["style_index"] for b in batch], dim=0),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
    }


# ============================================================
# 3. Online Sequence Dataset
# ============================================================

class OnlineSequenceDataset(Dataset):
    """
    Point-level online sequences (true IAM-OnDB or pseudo-online).

    Expected fields:
      - record_id
      - events: [{dx, dy, pen_down, stroke_end, char_end, word_end, seq_end}, ...]
      - word_text
      - style_id (optional)
      - source_type (optional)
      - confidence (optional)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        style_to_index: Optional[Dict[str, int]] = None,
        max_text_len: int = 64,
        max_seq_len: int = 4096,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.style_to_index = style_to_index or {}
        self.max_text_len = max_text_len
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def _events_to_tensor(self, events: Sequence[Dict[str, Any]]) -> Tensor:
        rows = []
        for e in events[:self.max_seq_len]:
            rows.append([
                float(e["dx"]), float(e["dy"]),
                float(e["pen_down"]), float(e["stroke_end"]),
                float(e["char_end"]), float(e["word_end"]),
                float(e["seq_end"]),
            ])
        if not rows:
            return torch.zeros(0, 7, dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        text_tokens = text_to_byte_tensor(rec.get("word_text"), max_len=self.max_text_len)
        events = self._events_to_tensor(rec["events"])

        style_id = rec.get("style_id", rec.get("writer_id", "__unk__"))
        if style_id is None:
            style_id = "__unk__"
        style_index = self.style_to_index.get(str(style_id), self.style_to_index.get("__unk__", 0))

        return {
            "record_id": rec["record_id"],
            "word_text": rec.get("word_text", ""),
            "text_tokens": text_tokens,
            "events": events,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "source_type": rec.get("source_type", "pseudo_online"),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_online_sequences(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    events, event_mask = pad_events([b["events"] for b in batch])
    return {
        "record_id": [b["record_id"] for b in batch],
        "word_text": [b["word_text"] for b in batch],
        "text_tokens": pad_1d_long([b["text_tokens"] for b in batch], pad_value=0),
        "text_lengths": torch.tensor([b["text_tokens"].numel() for b in batch], dtype=torch.long),
        "events": events,
        "event_mask": event_mask,
        "style_index": torch.stack([b["style_index"] for b in batch], dim=0),
        "source_type": [b["source_type"] for b in batch],
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
    }


# ============================================================
# 4. Character Plan Dataset
# ============================================================

class CharacterPlanDataset(Dataset):
    """
    Loads primitive stroke sets and candidate route plans per character.

    Expected fields:
      - record_id
      - char_text
      - primitive_strokes: [{stroke_id, points_fwd, arc_len}, ...]
      - selected_sequence: [{stroke_id, orientation}, ...]
      - plan_cost
      - style_id (optional)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        style_to_index: Optional[Dict[str, int]] = None,
        max_strokes: int = 32,
        max_points_per_stroke: int = 256,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.style_to_index = style_to_index or {}
        self.max_strokes = max_strokes
        self.max_points = max_points_per_stroke

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        strokes = rec.get("primitive_strokes", [])[:self.max_strokes]
        n_strokes = len(strokes)

        # Encode strokes as padded point arrays
        all_points = []
        stroke_lengths = []
        for s in strokes:
            pts = s["points_fwd"][:self.max_points]
            stroke_lengths.append(len(pts))
            # Pad to max_points
            padded = pts + [[0.0, 0.0]] * (self.max_points - len(pts))
            all_points.append(padded)

        # Pad to max_strokes
        while len(all_points) < self.max_strokes:
            all_points.append([[0.0, 0.0]] * self.max_points)
            stroke_lengths.append(0)

        points_tensor = torch.tensor(all_points, dtype=torch.float32)  # [max_strokes, max_points, 2]
        lengths_tensor = torch.tensor(stroke_lengths, dtype=torch.long)  # [max_strokes]

        # Selected sequence as indices
        seq = rec.get("selected_sequence", [])
        stroke_id_to_idx = {s.get("stroke_id", i): i for i, s in enumerate(strokes)}
        seq_indices = []
        seq_dirs = []
        for item in seq:
            sid = item["stroke_id"]
            si = stroke_id_to_idx.get(sid, 0)
            seq_indices.append(si)
            seq_dirs.append(0 if item.get("orientation", "fwd") == "fwd" else 1)

        char_tokens = text_to_byte_tensor(rec.get("char_text", rec.get("word_text")), max_len=16)

        style_id = rec.get("style_id", "__unk__")
        style_index = self.style_to_index.get(str(style_id), 0)

        return {
            "record_id": rec["record_id"],
            "char_text": rec.get("char_text", rec.get("word_text", "")),
            "char_tokens": char_tokens,
            "stroke_points": points_tensor,
            "stroke_lengths": lengths_tensor,
            "n_strokes": torch.tensor(n_strokes, dtype=torch.long),
            "seq_indices": torch.tensor(seq_indices, dtype=torch.long),
            "seq_dirs": torch.tensor(seq_dirs, dtype=torch.long),
            "seq_len": torch.tensor(len(seq_indices), dtype=torch.long),
            "plan_cost": torch.tensor(float(rec.get("plan_cost", 0.0)), dtype=torch.float32),
            "style_index": torch.tensor(style_index, dtype=torch.long),
        }
