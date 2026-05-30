#!/usr/bin/env python3
"""Audit per-letter divider placement using a vision LLM.

For each word with bboxes from extract_letter_bboxes.py:
  - Crop the original word image from x=0 to each divider (front prefix)
  - Crop from each divider to x=W (back suffix)
  - Send each crop to a vision LLM ("read the handwritten letters")
  - Compare LLM reading to what we *expect* given the chop

Discrepancy per divider d_i (between letter i-1 and i):
  - Front crop of i letters: expected `i` letters; if LLM reads i → ok,
    i+1 → divider was too far right (cut into letter i, so its head stuck
    out and got read), i-1 → divider was too far left (already cut into
    letter i-1).
  - Back crop of N-i letters: symmetric.

Outputs per-word JSONL with: word_id, transcription, dividers, crops,
LLM readings, and divider quality scores.

Usage:
    OPENAI_API_KEY=sk-... python3 divider_audit_openai.py \\
        --bbox-jsonl runs/letter_bboxes_v2.jsonl \\
        --words-dir data/iam_words/iam_words/words \\
        --out runs/divider_audit_v2.jsonl \\
        --model gpt-4o-mini \\
        --limit 100 \\
        --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ------------------------------------------------------------------
# Crop construction
# ------------------------------------------------------------------

def expected_dividers_pixels(rec: Dict, scale: float) -> List[float]:
    """Return divider x-positions (in original-image pixels) between
    consecutive letters: [x_div_1, x_div_2, ..., x_div_{N-1}].
    The bboxes are in 64×256 letterbox coords; multiply by scale_x to
    map back to original-image pixels."""
    letters = rec["letters"]
    divs = []
    for i in range(len(letters) - 1):
        # Use the average of letter[i].x2 and letter[i+1].x1 as the divider.
        d = 0.5 * (letters[i]["x2"] + letters[i + 1]["x1"]) * scale
        divs.append(d)
    return divs


def make_crops(img: Image.Image, divs: List[float]) -> List[Tuple[str, Image.Image, int]]:
    """Return a list of (label, crop_image, expected_letter_count) tuples.

    label: 'front_k' = leftmost k letters by chopping at divider k-1
    label: 'back_k'  = rightmost k letters by chopping at divider N-k
    label: 'full'    = the whole word
    expected_letter_count: how many letters we expect to be visible
    """
    W, H = img.size
    N_letters = len(divs) + 1
    crops: List[Tuple[str, Image.Image, int]] = []
    crops.append(("full", img, N_letters))
    for k in range(1, N_letters):
        # front_k: crop x=0 to divider[k-1]
        x_right = max(1, int(round(divs[k - 1])))
        crops.append((f"front_{k}", img.crop((0, 0, x_right, H)), k))
    for k in range(1, N_letters):
        # back_k: crop x=divider[N-k-1] to x=W
        x_left = min(W - 1, int(round(divs[N_letters - k - 1])))
        crops.append((f"back_{k}", img.crop((x_left, 0, W, H)), k))
    return crops


# ------------------------------------------------------------------
# OpenAI call (vision)
# ------------------------------------------------------------------

PROMPT = (
    "This is a crop of a handwritten word from the IAM dataset. "
    "Read each visible letter from left to right and return JSON: "
    '{"letters": ["a", "b", ...]} containing only the letters you can clearly identify. '
    "If a letter is partially cut off and you cannot identify it, do NOT include it. "
    "Return only the JSON object."
)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def call_openai(client, model: str, img: Image.Image,
                      detail: str = "high", max_retries: int = 3,
                      n: int = 1, temperature: float = 0.0) -> List[List[str]]:
    """Returns a list of K letter-readings (one per sampled completion).
    Empty list on failure."""
    b64 = img_to_b64(img)
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": detail,
                                },
                            },
                        ],
                    },
                ],
                temperature=temperature,
                n=n,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            outs: List[List[str]] = []
            for choice in resp.choices:
                text = choice.message.content
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                    letters = obj.get("letters", [])
                    if isinstance(letters, list):
                        outs.append([str(x) for x in letters])
                except Exception:
                    continue
            return outs
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"   [warn] OpenAI call failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
                return []
            await asyncio.sleep(2 ** attempt)
    return []


# ------------------------------------------------------------------
# Per-word audit
# ------------------------------------------------------------------

def divider_score(transcription: str, crop_label: str, expected_count: int,
                  reads: List[List[str]]) -> Dict:
    """Compare K LLM readings to what we expect given the chop. Returns
    aggregated stats: drift distribution, modal drift, exact-match rate."""
    if crop_label.startswith("front_"):
        expected_text = transcription[:expected_count]
    elif crop_label.startswith("back_"):
        expected_text = transcription[-expected_count:]
    else:
        expected_text = transcription

    drifts = [len(r) - expected_count for r in reads]
    drift_counts: Dict[int, int] = {}
    for d in drifts:
        drift_counts[d] = drift_counts.get(d, 0) + 1
    # Convert to sorted list for stable JSON.
    drift_distribution = sorted(drift_counts.items())
    if drifts:
        modal_drift = max(drift_counts.items(), key=lambda kv: kv[1])[0]
    else:
        modal_drift = None
    n_exact = sum(1 for r in reads if r == list(expected_text))
    return {
        "label": crop_label,
        "expected_count": expected_count,
        "expected_text": expected_text,
        "reads": reads,
        "drift_distribution": drift_distribution,
        "modal_drift": modal_drift,
        "n_samples": len(reads),
        "exact_substring_rate": n_exact / max(len(reads), 1),
    }


async def audit_word(client, model: str, rec: Dict, words_dir: Path,
                     detail: str = "high", semaphore=None,
                     samples_n: int = 1, samples_temp: float = 0.0) -> Dict:
    img_path = words_dir / rec["form"] / rec["line"] / f"{rec['word_id']}.png"
    img = Image.open(img_path).convert("L")
    W_orig, H_orig = img.size
    # Bboxes are in 64×256 letterbox coords. We crop the ORIGINAL image,
    # not the letterbox, so we need scale factors.
    scale_x = W_orig / rec["target_w"]
    divs = expected_dividers_pixels(rec, scale_x)
    crops = make_crops(img, divs)

    async def call_one(label, crop_img, expected_count):
        if semaphore is not None:
            async with semaphore:
                reads = await call_openai(client, model, crop_img, detail=detail,
                                          n=samples_n, temperature=samples_temp)
        else:
            reads = await call_openai(client, model, crop_img, detail=detail,
                                      n=samples_n, temperature=samples_temp)
        return divider_score(rec["text"], label, expected_count, reads)

    results = await asyncio.gather(*[
        call_one(label, c, n) for label, c, n in crops
    ])

    return {
        "word_id": rec["word_id"],
        "text": rec["text"],
        "n_letters": len(rec["letters"]),
        "n_crops": len(crops),
        "audits": results,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main_async(args) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise SystemExit("Install openai: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY env var.")
    client = AsyncOpenAI(api_key=api_key)

    records: List[Dict] = []
    with open(args.bbox_jsonl) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            if len(r["letters"]) < 2:
                continue
            records.append(r)
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Auditing {len(records)} words")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume-safe: skip already-audited word_ids.
    done_ids = set()
    if out_path.exists() and args.resume:
        with open(out_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    done_ids.add(json.loads(raw)["word_id"])
                except Exception:
                    pass
        records = [r for r in records if r["word_id"] not in done_ids]
        print(f"  Resumed: {len(done_ids)} done, {len(records)} remaining")

    semaphore = asyncio.Semaphore(args.concurrency)
    fout = open(out_path, "a")

    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            result = await audit_word(client, args.model, rec,
                                      Path(args.words_dir),
                                      detail=args.detail,
                                      semaphore=semaphore,
                                      samples_n=args.samples,
                                      samples_temp=args.temperature)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(records) - i - 1) / max(rate, 1e-6)
            if (i + 1) % 5 == 0 or i == len(records) - 1:
                print(f"  [{i+1:>5}/{len(records)}] {rec['word_id']:>20} "
                      f"'{rec['text']!r}' done — {rate:.2f} w/s, ETA {eta/60:.1f} min")
        except Exception as e:
            print(f"  [{i+1}] {rec['word_id']} FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
    fout.close()
    print(f"\nWrote audits to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-jsonl", required=True)
    ap.add_argument("--words-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--detail", default="high", choices=["low", "high", "auto"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle records before applying --limit (with --seed).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="Skip word_ids already present in --out.")
    ap.add_argument("--samples", type=int, default=1,
                    help="Number of completions per crop (n parameter).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (use >0 with --samples > 1).")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
