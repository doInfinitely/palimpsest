# Session notes — print-glyph conditioning + supporting work

This document summarizes the experiments and changes from a single
extended work session on the palimpsest handwriting-generation pipeline.
The headline result is **print-glyph conditioning**, which became the
single largest quality lever we found.

## TL;DR

We added a small additional input to the GAN generator: a *printed*
rendering of the target letter (in a randomly-selected font, sampled
per training example), encoded by a small CNN and fed in additively
via a zero-initialized projection so existing checkpoints can resume
without disruption. This shape prior pushed classifier-acc-on-generated
letters from a ~0.67 plateau (v9b) to **0.9338** on the held-out
validation set (v11_ext2 best.pt), and substantially improved
visual handwriting quality on the prompts that previously failed.

Key checkpoint: `runs/letter_gan_v11_extended2/best.pt` (val_clf_acc =
0.9338, abs epoch 60 from the pixel-L1 warmup).

## Phase 1 — bbox refiner (didn't work; instructive failure)

**Files:** `train_bbox_refiner.py`

We tried to refine the per-letter bounding boxes using a differentiable
soft-mask + a 2-channel CNN classifier. The setup: per-instance learnable
`(x1, y1, x2, y2)` parameters stored in an `nn.Embedding(N_instances, 4)`,
with the CNN trained to classify the letter from `[image, soft_mask]`.
Phase 1 froze the bbox and trained the CNN to 70.2% val acc; phase 2
unfroze the bbox.

Result: **bboxes essentially didn't move** (mean drift 0.01 px over
30 epochs). Diagnosis: `nn.Embedding` gradients are sparse (only updated
rows that appear in a batch), and SGD with `lr=0.5` produced per-step
movement smaller than measurement precision.

Fix: swapped `SGD(lr=0.5)` for `Adam(lr=0.1)`, also lowered CNN lr in
phase 2 to prevent overfitting while the bbox catches up. Bboxes then
moved 3-44 px and the run produced refined coordinates.

**Cross-check vs GPT-audit recommendations**
(`compare_refiner_vs_audit.py`): the directions agreed on 56-58% of
dividers (vs 50% baseline). Modest signal — not transformative.

## Phase 2 — bbox inflation sweep (real but limited gain)

**Files:** `inflate_bboxes.py`, `run_inflation_sweep.sh`,
`run_inflation_extended.sh`

We tried inflating every bbox by a multiplicative factor (centered,
clipped) and re-training the classifier to see whether tight bboxes
were starving the classifier of context.

| scale | val acc | Δ vs baseline |
|---|---|---|
| 1.00 (tight) | 0.7020 | — |
| 1.15 | 0.7214 | +1.9 pp |
| 1.33 | 0.7284 | +2.6 pp |
| **2.00** | **0.7367** | **+3.5 pp** |
| 3.00 | 0.6856 | -1.6 pp |
| 10.0 (no mask) | 0.3660 | -33.6 pp |

Clear peak at scale 2.0. Past that, the mask becomes too coarse to
localize the target letter; below that, surrounding ink is being
masked out and hurts recognition. **At scale 2 the classifier hits a
ceiling around 73%** — well short of the ~95% that v11's outputs reach
when scored by the same classifier, suggesting the classifier wasn't
the bottleneck either.

## Phase 3 — classifier error diagnosis (data is fine, task is hard)

**Files:** `analyze_bbox_classifier_errors.py`

Per-class accuracy + top confusions on the val set at scale 2.0:

- **Long tail** (17 rare classes: `j q 0 U O J 2 3 5 9 z K 1 4 6 V Y`):
  all at 0% acc, accounting for ~5% of errors.
- **Cursive minim ambiguity** (95% of errors): top confusions are the
  classic `m↔n↔u↔i` set, `a↔o`, `r → e/o/n/t/i/a` (508 r samples but
  only 57.5% acc), `c↔e↔o`, `l↔e↔i`. These are genuinely ambiguous
  without word-level context.

Conclusion: more data won't fix this. The classifier ceiling reflects
real Bayes-floor for isolated cursive characters. We accepted ~73% as
"good enough supervision for the GAN" and moved on. **This was a
key calibration: the classifier doesn't need to be perfect — it just
needs informative gradients on the cases the generator can fix.**

## Phase 4 — divider audit via GPT-4o-mini (paused on quota)

**Files:** `divider_audit_openai.py`, `analyze_divider_audit.py`,
`divider_contact_sheet.py`

We built an OCR-based audit of per-letter bbox placement: for each word,
crop at each divider and ask GPT-4o-mini "what letters do you see?"
with n=5 samples at temperature 0.7. The drift between expected and
read letter counts tells us each divider's direction error.

Audited 27,481 of 33,258 words (~83%) before hitting OpenAI quota.
Recommendations cover 53k dividers; 52% want movement in some direction.

## Phase 5 — print-glyph conditioning (THE result)

**Files:** `curate_print_fonts.py`, `render_print_glyphs.py`,
`train_fractal_infill.py` (modified), `train_letter_gan_infill.py`
(modified), `infer_scribe.py` (modified).

### Hypothesis

The cleanest classifier confusions are minim ambiguities. The
classifier eventually fails on these because m/n/u/i can be locally
indistinguishable without context. But the **generator** doesn't have
that excuse — it knows which letter it's drawing. It struggled because
its char-identity signal (`char_tokens`, a UTF-8 byte embedding
mean-pooled across `prev+target+next`) is a 128-dim opaque vector
that the generator has to slowly learn to decode into letter shapes,
class by class, weighted by class frequency.

What if we gave it the literal printed shape of the target letter as
an additional conditioning signal?

### Implementation

1. **Curate fonts** (`curate_print_fonts.py`): scanned 204 fonts in
   `../tiny-tessarachnid/fonts`, kept 80 that render every VOCAB char
   distinctly. Rejected braille, symbols-only, "notdef-substituting"
   fonts that render every char as the same placeholder rectangle.
2. **Render glyph cache** (`render_print_glyphs.py`): produces a
   `[F, C, 32, 32]` uint8 tensor of every (font, char) pair, saved to
   `runs/print_glyphs.pt` (~6 MB).
3. **PrintEncoder** (`train_fractal_infill.py`): a small CNN
   (16→32→64 channels) mapping `[B, 1, 32, 32]` glyph images to
   `[B, cond_dim=128]`. The output is projected via a `Linear(cond_dim,
   cond_dim, bias=False)` whose weights are **zero-initialized** — so
   a checkpoint resumed without print conditioning behaves identically
   at step 0, and training gradually grows the projection's weights.
4. **Additive injection**: `cond = cond + self.print_proj(self.print_encoder(glyph))`.
   Mixed in alongside the existing conditioning (char_tokens + style +
   bbox + noise), not as a replacement.
5. **Random fonts per training sample**: the dataset picks a random
   font index per `__getitem__`, so the generator learns that letter
   identity is invariant to print font. (This was the crucial choice —
   single-font training would have caused the model to copy the chosen
   font's style.)
6. **Inference** (`infer_scribe.py`): picks one font (default index 0)
   consistently across all letters in the output. The grid of 8 print
   fonts × 4 styles confirmed the output is essentially invariant to
   the chosen print font — style is governed by `style_seed`, print is
   just a shape prior.

### Results

**v10** (resumed v9b's last.pt with print conditioning enabled, 7
epochs to best):

| metric | v9b | v10_best |
|---|---|---|
| val_inside_l1 | 0.168 | 0.1594 |
| classifier acc on generated letters | ~0.67 | 0.93 |

In a single epoch of v10 training, classifier acc jumped from v9b's
0.67 plateau to 0.81 — a +14 pp gain. The print signal's contribution
was so direct that the zero-init projection grew nonzero within
hundreds of gradient steps.

**v11** (from-scratch run, identical lambdas/ramps to v9, only
difference is print conditioning):

| epoch | v9 acc | v11 acc | delta |
|---|---|---|---|
| 15 (just past ce ramp start) | 0.508 | 0.792 | +28 pp |
| 18 | 0.586 | 0.853 | +27 pp |
| 21 | 0.628 | 0.887 | +26 pp |
| 30 | — (v9 paused) | 0.915 | |

The +25-28 pp gap held across the ce-active regime. This is the clean
attribution: same warmup baseline, same lambdas, same ramps, only
print conditioning is new. **By v11 ep22 we were already past v9's
full-training final accuracy.**

**v11_extended (+20 epochs)** and **v11_extended2 (+5 epochs with
val-loop classifier eval)**: final `val_clf_acc = 0.9338`, best at
ep4 of ext2.

### Visual quality

The best visual jump was on the hardest prompt — **"minimum number"**.
In v9b this rendered as a row of indistinguishable minim strokes.
In v10/v11 the minim counts are visibly correct enough that "minimum"
reads as a word. Other prompts (hello world, the quick brown, rabbit)
became cleanly readable in v10 and stayed that way through v11.

### Why it worked so fast

1. **Direct shape signal**. The print glyph is a literal 32×32 image
   of the target letter. Compare to a `char_embed` vector the generator
   had to slowly learn to decode through gradient.
2. **Class-imbalance bypass**. The print encoder is class-agnostic — Z
   gets the same quality shape signal as e, regardless of dataset
   frequency.
3. **Short gradient path**. `print_proj(print_encoder(glyph))` is two
   layers from the FiLM modulation. High SNR on its gradient.
4. **Adapter bootstrapping**. Even at random init the print_encoder
   extracts *some* features (edges, blobs). The FiLM layers can use
   them immediately; print_proj weights quickly grow nonzero; the rest
   compounds.

## Phase 6 — val-loop classifier eval (code hygiene fix)

**File:** `train_letter_gan_infill.py`

The training script saved `best.pt` keyed on `val_inside_l1` — a pixel
reconstruction metric. This rewards blurry mean-pixel outputs over
sharp letter-shaped ones, which is exactly the opposite of what we
want. Fixed: added classifier evaluation to the val loop, switched
best-tracking to `val_clf_acc` when a classifier is provided. v11_ext2
was the first run with proper best-tracking, and its best.pt
(val_clf_acc=0.9338) is the artifact.

## Phase 7 — GAN-refine experiments (negative results)

**Files:** `refine_with_gan.py`, `refine_with_gan_latent.py`

We trained a small word-crop discriminator on real IAM vs v11 outputs
and tried two refinement strategies:

1. **Pixel-space** (`refine_with_gan.py`): optimize the output crop's
   pixels to fool D. Found adversarial noise — D had learned coarse
   texture distractors and pixel gradients sprinkled them in without
   meaningful structural change. Lame.
2. **Latent-space** (`refine_with_gan_latent.py`): optimize the per-letter
   noise vectors that feed the generator, keeping outputs on the
   model's natural-image manifold. The structural changes were
   non-adversarial but mild — D's score climbed -0.13 → 0.43 (real ≈
   0.85), and the output didn't visibly improve. Plausibility actually
   went down: D was rewarding "looks like the kind of texture in real
   IAM scans" but didn't know about letter identity, so it pulled in
   directions that erode legibility.

Honest takeaway: a simple unconditional GAN-realism discriminator is
the wrong loss for refinement. The right one would be **text-conditioned**
("does this look like a real `hello`?") or a **CTC handwriting
recognizer's confidence** on the target text. Both are real-work
projects beyond this session.

## Code-level changes summary

### Modified
- `train_fractal_infill.py`: added `PrintEncoder` and the additive
  `use_print_cond` path with zero-init `print_proj`. `forward_infill`
  accepts an optional `print_glyph`.
- `train_letter_gan_infill.py`: dataset loads the glyph cache and
  picks a random font per sample. Training/val loops pass `print_glyph`
  to the generator. Resume uses `strict=False` to accept the new
  parameters. `eval_epoch` now optionally runs the classifier on the
  val set; `best.pt` is keyed on `val_clf_acc` when available.
- `infer_scribe.py`: loads the glyph cache, picks one font for the
  inference, threads `print_glyph` through `render_word`.
- `train_letter_classifier.py`: stronger augmentation and class
  balancing changes that produced v3 (72% val acc).

### New
- `curate_print_fonts.py` — font selection with notdef detection.
- `render_print_glyphs.py` — pre-render `(font, char)` cache.
- `inflate_bboxes.py` — multiplicative bbox inflation, used by the
  classifier sweep.
- `train_bbox_refiner.py` — per-instance learnable bbox + 2-channel CNN.
- `analyze_bbox_classifier_errors.py` — per-class acc + confusion table.
- `compare_bbox_contact_sheet.py` — init vs refined bbox visualization.
- `compare_refiner_vs_audit.py` — refiner-vs-GPT-audit agreement stats.
- `divider_audit_openai.py`, `analyze_divider_audit.py`,
  `divider_contact_sheet.py` — GPT-4o-mini-based divider audit pipeline.
- `refine_with_gan.py`, `refine_with_gan_latent.py` — pixel-space and
  latent-space GAN refinement experiments.
- `run_inflation_sweep.sh`, `run_inflation_extended.sh` — sweep drivers.

## Headline artifacts

- `runs/letter_gan_v11_extended2/best.pt` — best generator
  (val_clf_acc = 0.9338).
- `runs/bbox_predictor_v2/best.pt` — bbox predictor used at inference
  (unchanged from before this session).
- `runs/print_glyphs.pt` — print glyph cache (79 fonts × 78 chars
  × 32 × 32 uint8).
- `eval_output/final_v9b_v10_v11ext2.png` — three-way visual comparison.
- `eval_output/hello_font_style_grid.png` — invariance check (8 print
  fonts × 4 styles, "hello world").

## Open follow-ups

- **Text-conditioned discriminator** for actual refinement (replaces
  the simple-GAN approach that didn't work).
- **CTC-recognizer-loss refinement** as a cleaner alternative.
- **Inflated bboxes in the generator**: classifier ceiling sweep
  showed scale 2.0 helps recognition; doesn't follow that it helps
  generation, but worth a fresh GAN training run to test.
- **Resume the divider audit** when OpenAI credits are restored
  (~5.7k words remaining).
