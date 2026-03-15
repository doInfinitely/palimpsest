# ChatGPT Transcript

## User

I want to use the IAM Online Handwriting Database and other handwriting datasets to train a handwriting generator. Assume we have images of pages, lets first compute the image pyramid of bounding boxes for each page, and use pixel analysis to reject empty bounding boxes, do non-maximum suppression in local spatial and scale space, then use a siamese network with a pretrained vision backbone (maybe resnet) on all pairs of detections to tell if they are the same letter or not. Give the siamase network two retinas (dimension should be a power of two and we should try 2^5 up to 2^9), the scale each input preserving aspect ratio and pad to fill the retinas. Then we feed through the backbone and we connect the hidden layers to a MLP with a sigmoid activation at the end that outputs if the to items are the same. We pick a threshold for sameness, compute the connected components of the sameness graph, and then pick paragons from the components, send those to anthropic or open ai to find the identity of the cluster. Then we propagate back the identities of the clusters to the bounding boxes to produce annotations of the characters on the page. We scan an angle to the right of each bounding box, forming a directed chain when a letter lands in the preceding letter's scan angle and we conglomerate chains into lines, we do the same down the page to combine lines into paragraphs (this time with a wide scan angle), and paragraphs are combined into pages by proximity. At this point we should have a hierarchical annotation of the symbols on the page. Now, we need to construct trajectories of bounding boxes where each bounding box is basically an action in the trajectory, for each action we store the the preceding bounding boxes with their images (probably just references), the identity of the symbol in the bounding box as context and the symbol image as the action. We train an infill model with a retina (with dimensions of say 2^7 up to 2^9) on each action step and we zoom in our retina on the input page so that it perfectly contains our bounding box in width or height at each step. The infiller is conditioned on the character to draw and noise. We train the infiller to go from the page before the symbol was drawn to after the symbol was drawn. We can now infill any symbol. Last we need to train the scribe that starts with the current state of the page (which starts empty) and autoregressively detects a bounding box to infill conditioned until the stop token is produced. Our infiller retina is zoomed in on the bounding box, and we infill. The scribe should be trained with behavioral cloning via teacher forcing on the annotations we produced earlier. At this point we have both the symbol infiller, and a model to place the infills so we have a handwriting generator.

---

## ChatGPT

I’m checking the current dataset landscape and anchoring the pipeline against primary sources so I can give you a sharper architecture rather than just a hand-wavy yes/no.

---

## ChatGPT

One big structural issue is already clear: doing a learned same/different decision on all pairs of detections will blow up quadratically. A better version is “embed everything once, retrieve only near neighbors, then run the calibrated same/different head on those candidate pairs.”

---

## ChatGPT

Yes — this is a coherent research program. I’d describe it as:

pseudo-annotate page images into symbols + hierarchy,

learn a local symbol renderer/inserter,

learn a layout policy that decides where symbols go.

The main thing I’d change is how you get the pseudo-labels and how you scale the similarity step.

IAM is a strong offline starting point because it gives scanned handwritten English pages and extracted lines/words; the current IAM page says version 3.0 has 657 writers, 1,539 scanned pages, 13,353 labeled text lines, and 115,320 labeled words. IAM-OnDB is especially valuable for your later generator because it adds online forms stored in XML with writer IDs and transcriptions, plus extracted lines; the official page lists 221 writers, more than 1,700 forms, and 13,049 isolated labeled text lines. CVL is a good complementary offline page dataset because it has 311 writers, 300 dpi color page images, and XML ground truth with a hierarchy from page → text block → text line → word plus word bounding boxes. EMNIST is still useful, but mostly as an isolated-character pretraining source rather than a page-layout source. 
NIST
+3
fki.tic.heia-fr.ch
+3
fki.tic.heia-fr.ch
+3

The most important correction is this: do not run the Siamese model on all pairs of detections. If a page has 
𝑁
N candidate boxes, all-pairs comparison is 
𝑂
(
𝑁
2
)
O(N
2
), and it gets ugly fast. A better pipeline is:

detect candidate symbol boxes,

embed each box once with a shared encoder,

retrieve only a small candidate neighborhood in embedding space and local geometry,

then run the “same vs different” head only on those candidate pairs,

threshold the resulting graph and cluster with union-find / connected components.

That preserves your idea, but makes it practical.

1) Pseudo-annotation pipeline

Your page-analysis stage is reasonable, but I’d tighten it into this:

A. Proposal generation

deskew / normalize contrast

build an image pyramid

propose boxes from foreground regions plus sliding windows

reject empty or junk boxes with simple pixel tests:

foreground occupancy

connected-component count

entropy

stroke-width sanity

aspect-ratio priors

B. Local spatial+scale NMS
Do NMS in 
(
𝑥
,
𝑦
,
log
⁡
𝑠
)
(x,y,logs) rather than only image-plane overlap. That gives you the “same place at slightly different scales” suppression you want.

C. Siamese glyph clustering
Your Retina idea is good. For each crop:

preserve aspect ratio

scale to fit

pad to a square power-of-two retina

For the similarity model, I’d try:

64 and 128 first for the glyph encoder

keep 32 only as an ablation

use 256 only for messy cursive fragments / ligatures / punctuation-heavy crops

Architecture:

shared ResNet or ConvNeXt-style encoder

embeddings 
𝑒
1
,
𝑒
2
e
1
	​

,e
2
	​


MLP on features like 
[
𝑒
1
,
𝑒
2
,
∣
𝑒
1
−
𝑒
2
∣
,
𝑒
1
⊙
𝑒
2
]
[e
1
	​

,e
2
	​

,∣e
1
	​

−e
2
	​

∣,e
1
	​

⊙e
2
	​

]

sigmoid same/different score

Train with a mixture of:

BCE on same/different

contrastive or triplet loss on the embedding itself

That usually works better than pure BCE because the embedding space becomes usable for retrieval, not just classification.

2) Labeling clusters

Your “pick paragons, send to Anthropic/OpenAI, propagate labels back” idea is workable, but I would not make that the primary labeling path.

A stronger route is:

choose each cluster’s medoid/prototype

use known word/line transcripts from IAM/CVL to constrain cluster identity

solve cluster-to-character mapping with sequence consistency

only use a VLM/LLM on unresolved clusters, punctuation, rare ligatures, or ambiguous cases

Why this is better:

VLMs on tiny glyph crops are noisy

transcript constraints are cheap and global

you can use the known word text to disambiguate clusters without asking a model to “guess the letter”

So I’d treat the LLM/VLM as a backstop, not the main source of truth.

3) Hierarchy construction

Your chaining idea is solid, but I’d make it graph-based.

For each symbol box, predict or estimate:

center

height/width

local baseline angle

forward direction

Then create directed edges to candidate successors that lie:

in a rightward cone,

within a compatible baseline band,

at a plausible inter-character distance.

Extract lines with dynamic programming or shortest-path / best-successor decoding rather than greedy chaining.

Then merge lines into paragraphs using:

vertical spacing,

left-margin consistency,

overlap in horizontal extent,

shared baseline angle.

You do not need “paragraphs combined into pages by proximity” if the input unit is already a page. The page is the root. So the natural hierarchy is:

page → paragraph → line → symbol

CVL’s XML hierarchy is especially nice here because it gives you a target structure to validate against. 
cvl.tuwien.ac.at

4) The infiller

This part is promising, but there is one conceptual catch:

If you only have offline page images, then “before symbol was drawn” to “after symbol was drawn” is a valid supervision signal for symbol insertion, but it is not true stroke-order supervision. It teaches the model how to add a symbol-shaped patch to a page state; it does not tell the model the real pen trajectory inside the symbol.

That is still useful. It means your infiller becomes a conditional local compositor:

input: page patch before insertion, bbox mask, target character ID, writer/style embedding, noise

output: delta patch or after-patch

I would make it predict the delta inside the bbox rather than regenerate the whole crop.

For retina size:

128 for clean isolated symbols

256 if you include neighboring context and want the model to respect ligatures / overlaps

And I’d condition it on:

character token

writer/style embedding

optional local context embedding from recent symbols

5) The scribe

This is where I’d make the biggest modeling change.

A page-level autoregressive scribe that starts from an empty page and emits symbol boxes until stop is possible, but for v1 it is probably too hard. Recent handwriting generation work still spends a lot of effort at the word or text-line level; a 2025 ICCV paper explicitly argues that generating only at the word level loses vertical alignment and writer-specific spacing, and shows benefits from modeling complete text lines directly. 
CVF Open Access

So I would make the generator hierarchical:

line planner: where the next line lives

symbol placer within the line

infiller for the actual glyph rendering

Or even:

page planner → line planner → symbol placer → infiller

Also, the scribe must be conditioned on desired content and style. Without that, it is just an unconditional page imitator.

A clean formulation is:

At step 
𝑡
t, condition on:

current page state

target transcript

pointer into transcript / remaining characters

writer/style embedding

Predict:

next bbox

stop token

optionally confidence / layer order

Train with teacher forcing on your pseudo-annotations.

6) Where IAM-OnDB really helps

IAM-OnDB is the piece that can turn this from “page insertion model” into something closer to genuine handwriting generation, because the official dataset contains online forms acquired with the E-Beam system, stored in XML with writer IDs and transcriptions. That means you can derive actual temporal order and partial states rather than inventing them from final page images. 
fki.tic.heia-fr.ch

That matters because classic online handwriting synthesis work, like Graves, models handwriting as a sequence of real-valued pen events rather than a sequence of image insertions. More recent work also explicitly separates writer-level and character-level style factors instead of treating style as one undifferentiated latent. 
arXiv
+2
arXiv
+2

So I’d split the project in two tracks:

offline track: page/symbol insertion from scanned pages

online track: stroke-order supervision and style learning from IAM-OnDB

Then let the online track regularize the offline renderer.

7) What I would actually build first

I would sequence it like this:

Stage 1 — pseudo-labeler

proposals

empty-box rejection

local+scale NMS

embedding model

kNN candidate graph

calibrated same/different head

connected components

transcript-constrained cluster labeling

page/paragraph/line/symbol hierarchy

Stage 2 — symbol infiller

local conditional diffusion or rectified-flow model

before-patch + bbox + char + style → after-patch / delta

Stage 3 — line-level scribe

not full page yet

current line canvas + remaining text + style → next bbox + stop

call infiller after each step

Stage 4 — page planner

place lines/paragraphs on a blank page

then invoke the line scribe

That decomposition is much more likely to converge than “blank full page → entire symbol sequence” from day one.

Bottom line

Your overall idea is good. The version I’d endorse is:

keep the multi-scale detection → similarity graph → connected components → pseudo-label pipeline,

replace all-pairs Siamese with embedding retrieval + local pair scoring,

use transcript-constrained labeling before resorting to VLMs,

treat the infiller as a local conditional renderer,

and make the scribe hierarchical and text-conditioned, ideally starting at the line level rather than the full page.

The single biggest unlock is this:

Use IAM-OnDB for true temporal supervision, and use IAM/CVL for page/layout supervision. That gives you both the geometry and the writing dynamics. 
fki.tic.heia-fr.ch
+2
fki.tic.heia-fr.ch
+2

I can 
turn this into a concrete training spec next: datasets, model I/O shapes, losses, and the exact annotation JSON schema.

---

## User

No we need paragraphs combined into pages case we could in theory have multiple pages, just trust me on this it's an integration thing with existing work. For this we don't need stroke-order supervision but it would be interesting to do this with Japanese which has stroke order in the dictionary and train an online version. In fact let's do that as well as a side-project. 

<div class="small-12 columns">
            <h2>Stroke order</h2>
  <div class="stroke_order_diagram--outer_container">
    <svg class="stroke_order_diagram--svg_container_for_51866274d5dda796580000d7" style="height: 100px; width: 900px;" viewBox="0 0 900 100"><desc>Created with Snap</desc><defs></defs><line x1="1" x2="899" y1="1" y2="1" class="stroke_order_diagram--bounding_box"></line><line x1="1" x2="1" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><line x1="1" x2="899" y1="99" y2="99" class="stroke_order_diagram--bounding_box"></line><line x1="0" x2="900" y1="50" y2="50" class="stroke_order_diagram--guide_line"></line><line x1="50" x2="50" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="99" x2="99" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s1" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,-4,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82i" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,-4,-4)" class="stroke_order_diagram--current_path"></path><circle cx="52.97" cy="12.5" r="4" transform="matrix(1,0,0,1,-4,-4)" class="stroke_order_diagram--path_start"></circle><line x1="150" x2="150" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="199" x2="199" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82l" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,96,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s2" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,96,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82n" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,96,-4)" class="stroke_order_diagram--current_path"></path><circle cx="29.12" cy="28" r="4" transform="matrix(1,0,0,1,96,-4)" class="stroke_order_diagram--path_start"></circle><line x1="250" x2="250" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="299" x2="299" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82q" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,196,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82r" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,196,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s3" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,196,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82t" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,196,-4)" class="stroke_order_diagram--current_path"></path><circle cx="37.12" cy="35.38" r="4" transform="matrix(1,0,0,1,196,-4)" class="stroke_order_diagram--path_start"></circle><line x1="350" x2="350" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="399" x2="399" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82w" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,296,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82x" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,296,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o82y" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,296,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s4" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,296,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8210" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,296,-4)" class="stroke_order_diagram--current_path"></path><circle cx="67.79" cy="27.5" r="4" transform="matrix(1,0,0,1,296,-4)" class="stroke_order_diagram--path_start"></circle><line x1="450" x2="450" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="499" x2="499" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8213" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8214" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8215" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8216" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s5" kvg:type="㇐" d="M12.75,50.5c3.65,0.66,7.11,0.39,10.76-0.01c20.16-2.21,40.45-4.53,63.37-5.41c3.4-0.13,7.17-0.48,10.5,0.47" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8218" kvg:type="㇐" d="M12.75,50.5c3.65,0.66,7.11,0.39,10.76-0.01c20.16-2.21,40.45-4.53,63.37-5.41c3.4-0.13,7.17-0.48,10.5,0.47" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--current_path"></path><circle cx="12.75" cy="50.5" r="4" transform="matrix(1,0,0,1,396,-4)" class="stroke_order_diagram--path_start"></circle><line x1="550" x2="550" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="599" x2="599" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821b" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821c" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821d" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821e" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821f" kvg:type="㇐" d="M12.75,50.5c3.65,0.66,7.11,0.39,10.76-0.01c20.16-2.21,40.45-4.53,63.37-5.41c3.4-0.13,7.17-0.48,10.5,0.47" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s6" kvg:type="㇑" d="M34.55,59.11c0.58,0.74,1.08,2.24,1.08,3.25c0,1.02,0.12,28.46,0.12,28.97c0,0.51,0,2.04,0,3.54" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821h" kvg:type="㇑" d="M34.55,59.11c0.58,0.74,1.08,2.24,1.08,3.25c0,1.02,0.12,28.46,0.12,28.97c0,0.51,0,2.04,0,3.54" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--current_path"></path><circle cx="34.55" cy="59.11" r="4" transform="matrix(1,0,0,1,496,-4)" class="stroke_order_diagram--path_start"></circle><line x1="650" x2="650" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="699" x2="699" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821k" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821l" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821m" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821n" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821o" kvg:type="㇐" d="M12.75,50.5c3.65,0.66,7.11,0.39,10.76-0.01c20.16-2.21,40.45-4.53,63.37-5.41c3.4-0.13,7.17-0.48,10.5,0.47" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821p" kvg:type="㇑" d="M34.55,59.11c0.58,0.74,1.08,2.24,1.08,3.25c0,1.02,0.12,28.46,0.12,28.97c0,0.51,0,2.04,0,3.54" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s7" kvg:type="㇕a" d="M36.1,60.79c9.52-1.16,26.02-3.04,32.08-3.68c2.95-0.31,4.26,0.27,4.26,3.22c0,0.98-0.11,17.98-0.17,27.68c-0.02,3.04-0.03,5.36-0.03,6.25" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821r" kvg:type="㇕a" d="M36.1,60.79c9.52-1.16,26.02-3.04,32.08-3.68c2.95-0.31,4.26,0.27,4.26,3.22c0,0.98-0.11,17.98-0.17,27.68c-0.02,3.04-0.03,5.36-0.03,6.25" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--current_path"></path><circle cx="36.1" cy="60.79" r="4" transform="matrix(1,0,0,1,596,-4)" class="stroke_order_diagram--path_start"></circle><line x1="750" x2="750" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="799" x2="799" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821u" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821v" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821w" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821x" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821y" kvg:type="㇐" d="M12.75,50.5c3.65,0.66,7.11,0.39,10.76-0.01c20.16-2.21,40.45-4.53,63.37-5.41c3.4-0.13,7.17-0.48,10.5,0.47" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o821z" kvg:type="㇑" d="M34.55,59.11c0.58,0.74,1.08,2.24,1.08,3.25c0,1.02,0.12,28.46,0.12,28.97c0,0.51,0,2.04,0,3.54" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8220" kvg:type="㇕a" d="M36.1,60.79c9.52-1.16,26.02-3.04,32.08-3.68c2.95-0.31,4.26,0.27,4.26,3.22c0,0.98-0.11,17.98-0.17,27.68c-0.02,3.04-0.03,5.36-0.03,6.25" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s8" kvg:type="㇐a" d="M36.69,74.75c5.44-0.5,29.72-2.64,34.74-2.85" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8222" kvg:type="㇐a" d="M36.69,74.75c5.44-0.5,29.72-2.64,34.74-2.85" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--current_path"></path><circle cx="36.69" cy="74.75" r="4" transform="matrix(1,0,0,1,696,-4)" class="stroke_order_diagram--path_start"></circle><line x1="850" x2="850" y1="1" y2="99" class="stroke_order_diagram--guide_line"></line><line x1="899" x2="899" y1="1" y2="99" class="stroke_order_diagram--bounding_box"></line><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8225" kvg:type="㇑a" d="M52.97,12.5c0.77,0.77,1.62,2.12,1.62,3.64c0,3.86-0.09,4.61-0.09,9.17" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8226" kvg:type="㇐" d="M29.12,28c1.5,0.5,2.75,0.62,4.61,0.42c11.39-1.29,25.05-3.48,38.89-4.79c2.63-0.25,5.63-0.25,7.38,0.12" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8227" kvg:type="㇔" d="M37.12,35.38c4.15,3.73,5.01,7.31,5.6,9.93" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8228" kvg:type="㇒" d="M67.79,27.5c0.46,0.9,0.57,2.02,0.23,3.14c-1.33,4.34-3.46,10.57-4.64,12.99" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o8229" kvg:type="㇐" d="M12.75,50.5c3.65,0.66,7.11,0.39,10.76-0.01c20.16-2.21,40.45-4.53,63.37-5.41c3.4-0.13,7.17-0.48,10.5,0.47" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o822a" kvg:type="㇑" d="M34.55,59.11c0.58,0.74,1.08,2.24,1.08,3.25c0,1.02,0.12,28.46,0.12,28.97c0,0.51,0,2.04,0,3.54" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o822b" kvg:type="㇕a" d="M36.1,60.79c9.52-1.16,26.02-3.04,32.08-3.68c2.95-0.31,4.26,0.27,4.26,3.22c0,0.98-0.11,17.98-0.17,27.68c-0.02,3.04-0.03,5.36-0.03,6.25" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o822c" kvg:type="㇐a" d="M36.69,74.75c5.44-0.5,29.72-2.64,34.74-2.85" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--existing_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="kvg:097f3-s9" kvg:type="㇐a" d="M37.11,90.41c6.58-0.27,25.43-2.25,33.94-2.6" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--current_path"></path><path xmlns:kvg="http://kanjivg.tagaini.net" id="Smmgn2o822e" kvg:type="㇐a" d="M37.11,90.41c6.58-0.27,25.43-2.25,33.94-2.6" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--current_path"></path><circle cx="37.11" cy="90.41" r="4" transform="matrix(1,0,0,1,796,-4)" class="stroke_order_diagram--path_start"></circle></svg>
  </div>

        </div>

Jisho has the stroke order for all japanese symbols, here it is encoded as an svg. For each symbol extract the drawn paths as a sequence of successive points after from the path data from jisho, take our font dataset and for each glyph in each font generate the glyph, align the paths in the glyph with the strokes in the jisho stroke order and the points on both paired strokes by scaling both glyphs to be the same scale and overlaying them. You want to minimize the the sum distance between aligned strokes, which can be computed per stroke as the sum of the distances between the aligned points.

256 if you include neighboring context and want the model to respect ligatures / overlaps <-- i want this. 

Your're right, let's go hierarchical, so we detect where we want to place the page, then the paragraph, then the line, then the word, then the character, training is conditioned on each level on our past placements and the next element at each level (so the character at the character level or the word at the word level)

---

## ChatGPT

I’m tightening this into a single hierarchical training plan and checking the Japanese stroke-order source so I can ground the online side-project cleanly.

---

## ChatGPT

I’ve got the key adjustment: make “page” an explicit node above paragraph, and for Japanese treat the SVG stroke-order source as the canonical sequence while the font glyph provides the target geometry to align to.

---

## ChatGPT

Yes — let’s keep page as an explicit node and just put one more root above it when needed.

The clean hierarchy becomes:

document-set / canvas → page → paragraph → line → word → character
and for the Japanese online side-project we add
character → stroke → point.

That fits your integration constraint and also matches the kinds of supervision the datasets expose: IAM gives scanned handwritten forms/pages, IAM-OnDB gives online handwritten forms with XML and transcriptions, CVL gives page images with hierarchical XML, and IAMonDo is especially relevant if you want richer document-level layouts because it contains online handwritten documents with content placed arbitrarily on the document and ground truth down to the word level. 
fki.tic.heia-fr.ch
+3
fki.tic.heia-fr.ch
+3
fki.tic.heia-fr.ch
+3

Revised offline generator

I’d make the system explicitly hierarchical and train each level by behavioral cloning with teacher forcing, exactly as you described:

At level 
𝐿
L, the model sees:

the current rendered state of the parent region,

the already placed siblings at level 
𝐿
L,

the next semantic item at level 
𝐿
L as conditioning,

a style / writer embedding,

and outputs the next child bbox plus stop.

So concretely:

page placer: conditioned on document context and next page token, place the next page bbox or stop

paragraph placer: conditioned on page state and next paragraph token/content summary, place paragraph bbox or stop

line placer: conditioned on paragraph state and next line text, place line bbox or stop

word placer: conditioned on line state and next word string, place word bbox or stop

character placer: conditioned on word state and next character, place character bbox or stop

character infiller: conditioned on the local 256×256 retina, character identity, writer/style, and noise, transform “before” into “after”

That gives you a factorization where layout and rendering are separated, but still coupled through teacher-forced states.

Retina choice

For the infiller: 256 is the right default if you want neighboring context, ligature behavior, overlap handling, and local style consistency. I would keep:

128 as an ablation,

256 as the default,

512 only if you later find that word-level interactions are being clipped.

For the similarity encoder in the pseudo-labeling stage, I’d still start with 64 and 128 first, then compare against 256.

Training targets at each level

Each action in the trajectory should be an object like:

Plain text
{
  level: "character",
  parent_id: word_17,
  previous_sibling_ids: [...],
  next_semantic_token: "e",
  bbox_xyxy: [...],
  bbox_center_wh: [...],
  local_retina_ref: ...,
  style_id: ...,
  stop: false
}

And similarly for words/lines/paragraphs/pages.

I would make each level predict:

stop / continue

bbox center

bbox size

optional rotation / baseline angle

optional z-order or overwrite flag if you anticipate corrections/edits

Loss:

BCE / CE for stop

L1 or smooth-L1 for bbox params

IoU-style auxiliary loss for placement quality

optional ordering loss so siblings stay in plausible reading order

Pseudo-labeling remains the front end

Your earlier pseudo-annotation pipeline still stands, now just producing a deeper tree:

multi-scale proposals

empty-box rejection

local + scale-space NMS

embedding + same/different graph

connected components / cluster labels

rightward chaining into words/lines

vertical grouping into paragraphs

paragraph grouping into pages

page grouping into document/canvas

The only thing I still strongly recommend is avoiding all-pairs Siamese scoring. Embed every detection once, do nearest-neighbor retrieval, then run the calibrated pair head only on candidate pairs.

Japanese side-project: from SVG stroke order to pseudo-online trajectories

The SVG you pasted is effectively KanjiVG-style data: KanjiVG’s official format represents kanji strokes as ordered SVG paths, and the project explicitly states that each character file provides stroke shape, direction, and stroke order; the files are vector paths on a 109×109 area, with stroke types and variant forms available as metadata. 
KanjiVG
+4
KanjiVG
+4
KanjiVG
+4

So the Japanese side-project should be:

1) Parse the canonical stroke sequence

From the SVG:

extract the ordered stroke paths

parse each d string into Bézier/path primitives

resample each stroke by arc length into a fixed or adaptive sequence of points

For each character you get:

Plain text
stroke_1 = [p1, p2, ..., pk]
stroke_2 = [q1, q2, ..., qm]
...

This becomes the canonical temporal order.

2) Render each font glyph

For each font and glyph:

render at high resolution

binarize

skeletonize or compute a medial-axis-like centerline

decompose the centerline graph into candidate stroke segments

You do not want to align KanjiVG strokes directly to the filled outline; you want to align them to a stroke-like skeleton graph.

3) Align canonical strokes to font geometry

Your proposed objective is right, but I’d make it slightly more formal.

Let:

𝑆
𝑖
S
i
	​

 = canonical KanjiVG stroke polyline

𝐺
𝑗
G
j
	​

 = candidate font skeleton segment/polyline

Allow a global normalization first:

scale both to common box

translate to common centroid

optionally allow a small affine or thin-plate-spline warp

Then solve an order-preserving assignment:

usually one canonical stroke maps to one font stroke segment,

but allow split/merge penalties for tricky fonts.

Per matched pair, score with resampled point distance, for example:

𝑑
(
𝑆
𝑖
,
𝐺
𝑗
)
=
∑
𝑡
∥
𝑆
𝑖
(
𝑡
)
−
𝐺
𝑗
(
𝑡
)
∥
2
d(S
i
	​

,G
j
	​

)=
t
∑
	​

∥S
i
	​

(t)−G
j
	​

(t)∥
2
	​


where both are arc-length normalized and sampled at the same number of points.

In practice I’d use:

arc-length resampling,

DTW or soft-DTW for point alignment,

dynamic programming over the ordered stroke list,

penalties for skipped / merged segments.

So the total alignment objective becomes something like:

min
⁡
𝜋
∑
𝑖
𝑑
(
𝑆
𝑖
,
𝐺
𝜋
(
𝑖
)
)
+
𝜆
split
𝑁
split
+
𝜆
skip
𝑁
skip
π
min
	​

i
∑
	​

d(S
i
	​

,G
π(i)
	​

)+λ
split
	​

N
split
	​

+λ
skip
	​

N
skip
	​


with 
𝜋
π constrained to preserve stroke order.

4) Produce pseudo-online training data

Once aligned, each font glyph becomes a synthetic online example:

state before stroke 1

draw stroke 1

state before stroke 2

draw stroke 2

...

final character

That gives you stroke-conditioned online supervision without needing pen-capture data for every font.

5) Train the online Japanese writer

Then train a stroke-level model:

Input at step 
𝑡
t:

current character canvas

current stroke index

character identity

writer/style embedding

optional previous stroke embeddings

noise

Output:

next stroke polyline or stroke raster delta

optional stop within character

Then place those characters inside the higher-level hierarchy:
page → paragraph → line → word → character → stroke.

Best use of datasets

For the English/offline branch:

IAM for page/line/word handwritten scans,

CVL for additional page-level structure,

IAMonDo if you want more document-like, arbitrarily arranged online content and document-root supervision. 
fki.tic.heia-fr.ch
+2
cvl.tuwien.ac.at
+2

For the Japanese branch:

KanjiVG-style SVG stroke data as the canonical stroke-order source,

your font corpus for geometry/style transfer,

optionally Kuzushiji-Kanji as extra isolated-character image supervision; its official repo describes 3,832 kanji classes and 140,424 grayscale images. 
KanjiVG
+2
KanjiVG
+2

The most important design choice

I’d make the generator explicitly content-conditioned at every level:

page conditioned on next page content chunk

paragraph conditioned on next paragraph text

line conditioned on next line text

word conditioned on next word string

character conditioned on next character

stroke conditioned on next stroke index / character identity

That keeps the model from having to invent both what and where at once.

Recommended first build

I would implement in this order:

hierarchical pseudo-labeler producing
document -> page -> paragraph -> line -> word -> character

character infiller at 256×256

character placer inside words

word placer inside lines

line placer inside paragraphs

page placer inside document/canvas

Japanese stroke-alignment pipeline
KanjiVG path order + font skeleton -> pseudo-online strokes

online Japanese stroke writer

That gives you an offline page generator first, then a stroke-aware Japanese extension that plugs into the same hierarchy.

The next useful step is for me to write the exact annotation schema and model I/O for all six levels, plus the Japanese stroke-alignment data format.

---

## User

Your overall idea is good. The version I’d endorse is:

keep the multi-scale detection → similarity graph → connected components → pseudo-label pipeline,

replace all-pairs Siamese with embedding retrieval + local pair scoring,

use transcript-constrained labeling before resorting to VLMs,

treat the infiller as a local conditional renderer,

and make the scribe hierarchical and text-conditioned, ideally starting at the line level rather than the full page.

The single biggest unlock is this:

Use IAM-OnDB for true temporal supervision, and use IAM/CVL for page/layout supervision. That gives you both the geometry and the writing dynamics.

I can 
turn this into a concrete training spec next: datasets, model I/O shapes, losses, and the exact annotation JSON schema.

All this sound good, turn it into a concrete training spec.

The next useful step is for me to write the exact annotation schema and model I/O for all six levels, plus the Japanese stroke-alignment data format.

Do this as well

---

## ChatGPT

I’m turning this into an implementable spec with concrete tensors, losses, curriculum, and JSON formats. I’m also checking the official dataset docs so the supervision assumptions for IAM/IAM-OnDB/CVL line up cleanly.

---

## ChatGPT

I’ve got the supervision split pinned down. The spec will treat the hierarchy as document -> page -> paragraph -> line -> word -> character, then attach a separate Japanese stroke format under character so the offline and online branches share most of the stack.

---

## ChatGPT

Below is a concrete spec I’d actually build.

The hierarchy is:

document → page → paragraph → line → word → character

and the renderer is a separate character infiller. For Japanese, add a parallel stroke representation under character using the SVG stroke order data from your sample’s kvg namespace; KanjiVG’s official format is SVG with stroke shape, direction, and order, which is exactly what we want as the canonical sequence. 
KanjiVG

1. Dataset roles

Use the datasets like this:

IAM (offline) for page, line, and word supervision, plus writer/style learning from scanned handwritten pages. The official IAM page says v3.0 has 657 writers, 1,539 scanned pages, 13,353 labeled text lines, and 115,320 labeled words, all scanned at 300 dpi. 
fki.tic.heia-fr.ch

IAM-OnDB (online + paired offline) for temporal supervision, writer/style embeddings, and line-level before/after or stroke-derived targets. The official IAM-OnDB page says it has 221 writers, more than 1,700 acquired forms, 13,049 isolated labeled text lines in online and offline format, and downloadable XML forms plus lineStrokes and lineImages. The download docs also define the official IAM-OnDB-t1 split. 
fki.tic.heia-fr.ch
+2
fki.tic.heia-fr.ch
+2

CVL (offline, structured XML) for page/paragraph/line/word geometry. The official CVL page says the dataset has roughly 310 writers, 300 dpi RGB page images, and XML with word boxes; the CVL paper also describes an explicit hierarchy in the XML where AttrRegion values correspond to page, text block, text line, and word. 
cvl.tuwien.ac.at
+1

Optional add-on: IAMonDo if you want stronger document-root supervision for arbitrarily placed content and richer page composition. The official page says IAMonDo contains 941 online handwritten documents with content such as text blocks, lists, tables, formulas, diagrams, drawings, and markings placed arbitrarily on the document, with ground truth down to the word level. 
fki.tic.heia-fr.ch

Recommended split policy:

preserve the official IAM-OnDB-t1 split for any result you want to compare to prior IAM-OnDB work,

but for your integrated generator maintain writer-disjoint train/val/test splits across all datasets so style generalization is measured honestly. The official IAM-OnDB download docs explicitly provide the t1 split definition. 
fki.tic.heia-fr.ch

2. Training products to generate first

From the raw datasets, produce four canonical artifacts:

Hierarchy annotations
document -> page -> paragraph -> line -> word -> character

Action trajectories for each level
Each child placement becomes one teacher-forced action.

Character render pairs
(before_patch, bbox, char_id, style) -> after_patch

Japanese stroke-alignment records
canonical SVG stroke order + font glyph skeleton + matched point correspondences

3. Preprocessing and pseudo-label pipeline
3.1 Page normalization

For every page or form image:

grayscale normalize to [0,1]

keep original resolution metadata

estimate skew and deskew

estimate foreground mask

keep both original and deskewed versions

Store:

image_abs_path

height_px, width_px

dpi

deskew_angle_deg

3.2 Multi-scale proposal generation

Build an image pyramid over each page.

Default:

min box side: 8 px

max box side: 256 px

6 scales per octave

propose boxes by combining:

connected-component boxes

projection-profile windows

dense windows for missed punctuation/fragments

For each box compute:

foreground occupancy

number of foreground connected components

bounding-box fill ratio

entropy

aspect ratio

estimated stroke width stats

Reject as empty/junk if any of:

occupancy < 0.02

occupancy > 0.90

connected components = 0

mean stroke width outside a broad valid band

aspect ratio outside [0.1, 10]

3.3 NMS in local spatial + scale space

Represent each proposal as (cx, cy, log_w, log_h, pyramid_level).

Suppress if:

IoU > 0.5 in image plane, and

scale difference < 1 octave, and

one proposal has lower foreground score / detector prior

Keep top M proposals per page after suppression, for example M = 2,000.

3.4 Glyph embedding and pair scoring

Do not score all pairs.

For each kept proposal:

crop

preserve aspect ratio

fit into a square retina

pad with background

Retina sizes to try:

32, 64, 128, 256, 512

start with 64, 128, 256

use 256 as the main “same letter?” model if you want context/ligature tolerance

Encoder:

shared pretrained ResNet backbone

projection head to embedding size D = 256

For each proposal embedding:

retrieve top K = 32 nearest neighbors

optionally restrict by scale ratio in [0.5, 2.0]

then run pair scorer only on those candidate pairs

Pair scorer input:

e1

e2

|e1 - e2|

e1 * e2

optional geometry features (Δlog_w, Δlog_h, same_page_flag)

MLP:

1024 -> 256 -> 1

sigmoid output p_same

Add edge if p_same >= τ_same, where τ_same is calibrated on validation.

Cluster with connected components exactly as you proposed.

3.5 Transcript-constrained labeling

Before any VLM:

order character candidates within each word/line

align candidate sequence to known transcript with dynamic programming

infer a posterior over cluster-to-character assignments

propagate the highest-confidence label back to all members of the cluster

Use the VLM only for:

unresolved clusters

punctuation

ligatures

uncertain clusters below confidence threshold

Confidence fields to store:

cluster_confidence

char_label_confidence

hierarchy_confidence

Low-confidence nodes can still exist in the tree but are excluded from supervised training unless confidence >= threshold.

4. Canonical retina policy

Use a single crop function at every level:

take the parent region,

zoom so the parent fits exactly in width or height,

preserve aspect ratio,

pad to a square power-of-two retina,

transform all child boxes into parent-normalized coordinates.

Default retina sizes:

document placer: 1024

page placer: 1024

paragraph placer: 512

line placer: 512

word placer: 256

character placer: 256

character infiller: 256

That keeps layout models cheaper while giving the local renderer the context you explicitly wanted.

5. Shared conditioning

Every model gets:

state raster: grayscale canvas of current parent state

occupancy mask: binary mask of already placed children

recent sibling masks: last K child masks, zero-padded

text condition

style embedding

level embedding

Recommended defaults:

style embedding dim: 256

text encoder: UTF-8 byte tokens or SentencePiece

recent sibling count:

document/page: 8

paragraph/line: 16

word/character: 32

Writer/style encoder:

shared CNN over sampled words/characters/lines

supervised with writer ID where available

final embedding 256d

optionally contrastive across same writer

6. Model I/O for all six levels

I’d use the same model family for all six placers:

raster encoder on state/masks

text encoder on next semantic item

style MLP

fusion transformer or MLP

bbox head + stop head

Each level predicts:

stop_logit

cx, cy, log_w, log_h

optional rotation_deg

optional baseline_slope

Coordinates are normalized to the parent region:

cx, cy in [0,1]

w, h in (0,1]

6.1 Document placer

Purpose: place pages inside a document/canvas.

Input tensors:

state: [1, 1024, 1024]

occupancy: [1, 1024, 1024]

recent_pages: [8, 1024, 1024]

next_page_text_tokens: [T_doc]

style: [256]

Output:

stop: scalar

bbox: [4]

optional rotation: scalar

Condition text:

the full text chunk assigned to the next page, or a truncation window

Loss:

BCE(stop)

SmoothL1(cx,cy,log_w,log_h)

GIoU on normalized boxes

optional ordering loss on page index

6.2 Page placer

Purpose: place paragraphs on a page.

Input:

state: [1, 1024, 1024]

occupancy: [1, 1024, 1024]

recent_paragraphs: [16, 1024, 1024]

next_paragraph_tokens: [T_para]

style: [256]

Output:

stop

bbox

optional rotation

Condition text:

exact paragraph text

Loss:

same as above

optional margin/alignment regularizer

6.3 Paragraph placer

Purpose: place lines inside a paragraph.

Input:

state: [1, 512, 512]

occupancy: [1, 512, 512]

recent_lines: [16, 512, 512]

next_line_tokens: [T_line]

style: [256]

Output:

stop

bbox

baseline_slope

Loss:

stop BCE

bbox regression

baseline slope regression

line-overlap penalty

6.4 Line placer

Purpose: place words inside a line.

Input:

state: [1, 512, 512]

occupancy: [1, 512, 512]

recent_words: [16, 512, 512]

next_word_tokens: [T_word]

style: [256]

Output:

stop

bbox

optional baseline_offset

Loss:

stop BCE

bbox regression + GIoU

spacing regularizer against previous word

6.5 Word placer

Purpose: place characters inside a word.

Input:

state: [1, 256, 256]

occupancy: [1, 256, 256]

recent_characters: [32, 256, 256]

next_char_token: scalar or [T_char=1..4 bytes]

style: [256]

Output:

stop

bbox

optional slant

Loss:

stop BCE

bbox regression + GIoU

inter-character spacing regularizer

6.6 Character placer

Purpose: for complex scripts or decomposed symbols, place subparts or the character’s final bounding box before rendering. For English this can simply predict the final character bbox. For Japanese this can feed the stroke renderer.

Input:

state: [1, 256, 256]

occupancy: [1, 256, 256]

recent_subparts_or_chars: [32, 256, 256]

next_char_token

style: [256]

Output:

stop

bbox

Loss:

same as word placer

In the English branch, you can collapse “word placer” and “character placer” if needed. In the unified spec I’d keep both so the interface matches Japanese.

7. Character infiller

This is separate from the six placers.

Purpose:

transform a local patch from “before character” to “after character”

Input:

before_patch: [1, 256, 256]

bbox_mask: [1, 256, 256]

neighbor_context_mask: [1, 256, 256]

char_token

style_embedding: [256]

noise_timestep

optional writer_id

Output:

either predicted noise

or predicted residual patch delta_patch

I’d use:

conditional UNet diffusion or rectified-flow UNet

target is local delta inside the bbox plus a small context halo

Loss:

noise prediction loss

masked L1 on reconstructed patch

perceptual loss on the bbox halo

optional writer/style consistency loss

Teacher-forced training sample:

before_patch = parent state before this character action

after_patch = parent state after this character action

8. Training curriculum
Phase A — writer/style encoder

Train a writer-style encoder on IAM, IAM-OnDB, and CVL with writer ID supervision where available. IAM, IAM-OnDB, and CVL all provide writer IDs or writer-linked forms/pages. 
fki.tic.heia-fr.ch
+2
fki.tic.heia-fr.ch
+2

Phase B — proposal + similarity bootstrap

generate proposals

learn embedding model

retrieve candidates

train pair scorer

cluster proposals

label with transcript-constrained DP

Phase C — hierarchy induction

Build:

document/page/paragraph/line/word/character trees

reading-order indices

teacher-forced action trajectories

For CVL, the XML hierarchy described in the paper is especially useful because it already distinguishes page, text block, text line, and word. 
cvl.tuwien.ac.at

Phase D — bottom-up placer training

Train in this order:

character infiller

word placer

line placer

paragraph placer

page placer

document placer

Phase E — scheduled sampling fine-tune

After all placers train under teacher forcing:

feed sampled predictions back in with scheduled sampling

keep stop-loss weight high to prevent runaway generation

Phase F — full generation

Generation order:

document places page

page places paragraph

paragraph places line

line places word

word places character

character bbox sent to infiller

stop tokens terminate each level

9. Default losses

For all placers:

𝐿
=
𝜆
𝑠
𝑡
𝑜
𝑝
𝐿
𝑏
𝑐
𝑒
+
𝜆
𝑏
𝑜
𝑥
𝐿
𝑠
𝑚
𝑜
𝑜
𝑡
ℎ
𝐿
1
+
𝜆
𝑖
𝑜
𝑢
𝐿
𝑔
𝑖
𝑜
𝑢
+
𝜆
𝑜
𝑟
𝑑
𝐿
𝑜
𝑟
𝑑
𝑒
𝑟
L=λ
stop
	​

L
bce
	​

+λ
box
	​

L
smoothL1
	​

+λ
iou
	​

L
giou
	​

+λ
ord
	​

L
order
	​


Recommended starting weights:

λ_stop = 1.0

λ_box = 5.0

λ_iou = 2.0

λ_ord = 0.5

For infiller:

λ_noise = 1.0

λ_patch = 2.0

λ_perc = 0.1

λ_style = 0.25

For similarity:

BCE on same/different

plus contrastive loss on embeddings

10. Exact annotation schema

Use one canonical tree file per document and one JSONL per level for trajectories.

10.1 Tree annotation file
JSON
{
  "schema_version": "1.0",
  "document_id": "iam_000123",
  "dataset": "IAM",
  "language": "en",
  "writer_id": "w_0042",
  "style_id": "style_w_0042",
  "source_image": {
    "path": "images/iam/forms/a01-000u.png",
    "width_px": 2479,
    "height_px": 3508,
    "dpi": 300,
    "deskew_angle_deg": -1.23
  },
  "text": {
    "full_text": "....",
    "pages": ["....", "...."]
  },
  "root_id": "doc_iam_000123",
  "nodes": [
    {
      "id": "doc_iam_000123",
      "level": "document",
      "parent_id": null,
      "child_ids": ["page_0"],
      "order_index": 0,
      "bbox_abs_xyxy": [0, 0, 2479, 3508],
      "bbox_parent_norm_cxcywh": [0.5, 0.5, 1.0, 1.0],
      "rotation_deg": 0.0,
      "text": "....",
      "text_span": [0, 500],
      "reading_order_key": [0],
      "label_source": "dataset",
      "confidence": 1.0
    },
    {
      "id": "page_0",
      "level": "page",
      "parent_id": "doc_iam_000123",
      "child_ids": ["para_0", "para_1"],
      "order_index": 0,
      "bbox_abs_xyxy": [0, 0, 2479, 3508],
      "bbox_parent_norm_cxcywh": [0.5, 0.5, 1.0, 1.0],
      "rotation_deg": 0.0,
      "text": "....",
      "text_span": [0, 500],
      "reading_order_key": [0, 0],
      "label_source": "dataset",
      "confidence": 1.0
    },
    {
      "id": "para_0",
      "level": "paragraph",
      "parent_id": "page_0",
      "child_ids": ["line_0", "line_1"],
      "order_index": 0,
      "bbox_abs_xyxy": [110, 420, 2250, 980],
      "bbox_parent_norm_cxcywh": [0.476, 0.200, 0.863, 0.160],
      "rotation_deg": 0.0,
      "text": "First paragraph text ...",
      "text_span": [0, 120],
      "reading_order_key": [0, 0, 0],
      "label_source": "induced",
      "confidence": 0.94
    },
    {
      "id": "line_0",
      "level": "line",
      "parent_id": "para_0",
      "child_ids": ["word_0", "word_1"],
      "order_index": 0,
      "bbox_abs_xyxy": [130, 440, 2180, 520],
      "bbox_parent_norm_cxcywh": [0.496, 0.107, 0.958, 0.143],
      "rotation_deg": -0.5,
      "baseline_slope": -0.003,
      "text": "The quick brown fox",
      "text_span": [0, 19],
      "reading_order_key": [0, 0, 0, 0],
      "label_source": "dataset",
      "confidence": 0.98
    },
    {
      "id": "word_0",
      "level": "word",
      "parent_id": "line_0",
      "child_ids": ["char_0", "char_1", "char_2"],
      "order_index": 0,
      "bbox_abs_xyxy": [130, 445, 270, 515],
      "bbox_parent_norm_cxcywh": [0.034, 0.625, 0.068, 0.875],
      "rotation_deg": 0.0,
      "text": "The",
      "text_span": [0, 3],
      "reading_order_key": [0, 0, 0, 0, 0],
      "label_source": "dataset",
      "confidence": 0.99
    },
    {
      "id": "char_0",
      "level": "character",
      "parent_id": "word_0",
      "child_ids": [],
      "order_index": 0,
      "bbox_abs_xyxy": [132, 448, 165, 514],
      "bbox_parent_norm_cxcywh": [0.125, 0.507, 0.236, 0.943],
      "rotation_deg": 0.0,
      "text": "T",
      "text_span": [0, 1],
      "reading_order_key": [0, 0, 0, 0, 0, 0],
      "cluster_id": "cluster_1942",
      "label_source": "transcript_constrained",
      "confidence": 0.93,
      "image_ref": {
        "page_id": "page_0",
        "crop_path": "crops/char_0.png"
      }
    }
  ]
}
10.2 Level trajectory JSONL

One line per action.

JSON
{
  "schema_version": "1.0",
  "trajectory_id": "traj_word_0",
  "document_id": "iam_000123",
  "level": "character",
  "step_index": 0,
  "parent_id": "word_0",
  "target_node_id": "char_0",
  "state_before_ref": "states/word_0/step_000_before.png",
  "state_after_ref": "states/word_0/step_000_after.png",
  "parent_retina_size": 256,
  "parent_bbox_abs_xyxy": [130, 445, 270, 515],
  "previous_sibling_ids": [],
  "recent_sibling_mask_refs": [],
  "semantic_condition": {
    "type": "character",
    "text": "T",
    "tokens": [84]
  },
  "style_id": "style_w_0042",
  "target": {
    "stop": false,
    "bbox_parent_norm_cxcywh": [0.125, 0.507, 0.236, 0.943],
    "rotation_deg": 0.0
  },
  "confidence": 0.93
}

Final stop action:

JSON
{
  "schema_version": "1.0",
  "trajectory_id": "traj_word_0",
  "document_id": "iam_000123",
  "level": "character",
  "step_index": 3,
  "parent_id": "word_0",
  "target_node_id": null,
  "state_before_ref": "states/word_0/step_003_before.png",
  "state_after_ref": null,
  "parent_retina_size": 256,
  "previous_sibling_ids": ["char_0", "char_1", "char_2"],
  "semantic_condition": {
    "type": "character",
    "text": null,
    "tokens": []
  },
  "style_id": "style_w_0042",
  "target": {
    "stop": true,
    "bbox_parent_norm_cxcywh": null,
    "rotation_deg": null
  },
  "confidence": 1.0
}

Use the same trajectory schema for all six levels; only level, retina size, and semantic condition differ.

11. Character infiller training record
JSON
{
  "schema_version": "1.0",
  "record_id": "infill_char_0",
  "document_id": "iam_000123",
  "parent_level": "word",
  "parent_id": "word_0",
  "target_node_id": "char_0",
  "retina_size": 256,
  "before_patch_ref": "infill/char_0_before.png",
  "after_patch_ref": "infill/char_0_after.png",
  "bbox_mask_ref": "infill/char_0_bbox_mask.png",
  "neighbor_mask_ref": "infill/char_0_neighbor_mask.png",
  "char_text": "T",
  "char_tokens": [84],
  "style_id": "style_w_0042",
  "target_bbox_parent_norm_cxcywh": [0.125, 0.507, 0.236, 0.943],
  "confidence": 0.93
}
12. Japanese stroke-alignment data format

Treat the SVG stroke sequence as canonical and align each font glyph to it.

12.1 Canonical stroke extraction

For each Japanese character:

parse each ordered SVG path

resample by arc length to N_i points

store:

stroke index

path type

raw d

point list

start/end point

total arc length

KanjiVG officially provides per-character SVGs containing stroke shape, direction, order, and stroke metadata, which is why it is a good canonical source for this alignment stage. 
KanjiVG

12.2 Font glyph extraction

For each font and glyph:

render to high-res binary image

skeletonize

decompose skeleton graph into candidate stroke segments

resample each segment to points

12.3 Alignment objective

For each canonical stroke 
𝑆
𝑖
S
i
	​

 and candidate glyph segment 
𝐺
𝑗
G
j
	​

:

normalize both into a shared coordinate frame

optionally apply a global similarity transform

compute point alignment cost with arc-length matching or soft-DTW

Total alignment cost:

𝐿
align
=
∑
𝑖
𝑑
(
𝑆
𝑖
,
𝐺
𝜋
(
𝑖
)
)
+
𝜆
split
𝑁
split
+
𝜆
skip
𝑁
skip
L
align
	​

=
i
∑
	​

d(S
i
	​

,G
π(i)
	​

)+λ
split
	​

N
split
	​

+λ
skip
	​

N
skip
	​


with order-preserving assignment 
𝜋
π.

12.4 Stroke-alignment JSON
JSON
{
  "schema_version": "1.0",
  "record_id": "jp_align_U+98DF_NotoSerifJP",
  "language": "ja",
  "char_text": "食",
  "codepoint": "U+98DF",
  "canonical_source": {
    "type": "svg_kvg",
    "source_ref": "kvg:097f3",
    "viewbox": [0, 0, 109, 109]
  },
  "canonical_strokes": [
    {
      "stroke_index": 0,
      "stroke_type": "㇑a",
      "path_d": "M52.97,12.5c0.77...",
      "points": [[48.97, 8.50], [49.10, 8.95], [49.22, 9.41]],
      "arc_length": 14.28
    }
  ],
  "font_glyph": {
    "font_id": "NotoSerifJP-Regular",
    "render_ref": "fonts/NotoSerifJP-Regular/U+98DF.png",
    "skeleton_ref": "fonts/NotoSerifJP-Regular/U+98DF_skel.json",
    "segments": [
      {
        "segment_id": "seg_0",
        "points": [[49.2, 8.4], [49.4, 8.9], [49.5, 9.5]],
        "arc_length": 14.01
      }
    ]
  },
  "global_transform": {
    "scale": 0.982,
    "rotation_deg": -1.4,
    "translate_xy": [0.8, -1.2]
  },
  "stroke_matches": [
    {
      "canonical_stroke_index": 0,
      "glyph_segment_ids": ["seg_0"],
      "match_type": "one_to_one",
      "cost": 1.37,
      "point_correspondence": [[0, 0], [1, 1], [2, 2]]
    }
  ],
  "total_cost": 8.52,
  "alignment_ok": true
}
12.5 Pseudo-online character sequence JSON
JSON
{
  "schema_version": "1.0",
  "record_id": "jp_online_U+98DF_NotoSerifJP",
  "char_text": "食",
  "codepoint": "U+98DF",
  "font_id": "NotoSerifJP-Regular",
  "canvas_size": 256,
  "style_id": "font_NotoSerifJP-Regular",
  "steps": [
    {
      "step_index": 0,
      "stroke_index": 0,
      "before_ref": "jp_steps/000_before.png",
      "after_ref": "jp_steps/000_after.png",
      "delta_mask_ref": "jp_steps/000_delta.png",
      "canonical_points": [[48.97, 8.50], [49.10, 8.95], [49.22, 9.41]],
      "glyph_points": [[49.20, 8.40], [49.40, 8.90], [49.50, 9.50]]
    }
  ]
}

That record is the direct training target for the Japanese online writer.

13. Minimal file layout
Plain text
data/
  documents/
    iam_000123.json
  trajectories/
    document.jsonl
    page.jsonl
    paragraph.jsonl
    line.jsonl
    word.jsonl
    character.jsonl
  infill/
    character_infill.jsonl
  japanese/
    stroke_alignments.jsonl
    pseudo_online_steps.jsonl
  crops/
  states/
  fonts/
14. What I would train first

writer/style encoder

proposal + embedding + pair scorer

transcript-constrained pseudo-labeler

tree builder

character infiller at 256x256

word placer

line placer

paragraph placer

page placer

document placer

Japanese stroke-aligner

Japanese pseudo-online writer

15. Recommended v1 defaults

encoder: ResNet-50 shared for similarity and style bootstrap

embedding dim: 256

pair scorer threshold: calibrated on val

layout placers: ViT-small or CNN+Transformer hybrid

text tokenizer: bytes for unified English/Japanese handling

optimizer: AdamW

batch sizes:

similarity: 256 pairs

placers: 32 actions

infiller: 16 patches

training order: bottom-up first, then scheduled sampling

The next most useful deliverable is a 
formal JSON Schema plus 
PyTorch dataset class signatures for these files, so you can plug them straight into dataloaders.

---

## User

Let's train an online scribe for english as well, take the online data and enrich with the offline data, what we do for offline is we schedule the points along the symbol to minimize the total travel distance of the pen. We do this hierarchically as well, planning each letter then chaining to words, lines, paragraphs, and pages, we store the symbol plan as contiguous strokes and order the strokes when we hit the word level to minimize the travel distance objective. Consider the strokes in both directions but make sure you only include the stroke once (one of the directions).

---

## ChatGPT

I’m folding this into the spec as a unified online English branch: true online trajectories where available, and pseudo-online trajectories induced from offline pages by a stroke-routing objective where they aren’t.

---

## ChatGPT

Yes — I’d add a parallel online English branch and make the leaf renderer a stroke decoder instead of only a raster infiller.

The clean split is:

True online supervision from IAM-OnDB, which exposes online forms and line-level assets like lineStrokes, lineImages, XML forms, writer-linked data, and an official IAM-OnDB-t1 evaluation split. 
fki.tic.heia-fr.ch

Offline enrichment from IAM and CVL, where you induce pseudo-online trajectories from scanned page images. IAM’s offline corpus has 1,539 pages of modern English text written by 657 writers, which is exactly the scale you want for style and page/layout enrichment. CVL adds writer IDs, 300 dpi page images, and XML word boxes. 
fki.tic.heia-fr.ch
+1

A text-conditioned online handwriting decoder at the leaf, following the same general setup Graves used for real-valued pen trajectories conditioned on text, but now inserted into your hierarchical page/paragraph/line/word/character stack. 
arXiv

Updated model stack

Keep the same hierarchical placers:

document → page → paragraph → line → word → character

but replace the English leaf renderer with:

character → ordered stroke sequence → ordered point sequence

So generation becomes:

document places page

page places paragraph

paragraph places line

line places word

word places character box

character-level online scribe emits strokes/points inside that box

differentiably rasterize the emitted points back onto the page canvas

For English, I would keep the old 256×256 local retina as the conditioning window for the online character renderer, even though the output is now a point sequence instead of a raster patch.

Unified online point format

Use one event stream for both true-online and pseudo-online English:

JSON
{
  "dx": float,
  "dy": float,
  "pen_down": 0|1,
  "stroke_end": 0|1,
  "char_end": 0|1,
  "word_end": 0|1,
  "seq_end": 0|1
}

Interpretation:

dx, dy: offset from previous point in parent-normalized coordinates

pen_down=1: drawing

pen_down=0: pen-up travel

stroke_end=1: current contiguous stroke finished

char_end=1: current character finished

word_end=1: current word finished

seq_end=1: parent sequence finished

For true online data from IAM-OnDB, convert the native stroke data into this event representation. For offline induced trajectories, generate the same representation from skeletonized strokes so the model can train on a single format. 
fki.tic.heia-fr.ch
+1

Offline → pseudo-online induction

This is the key new piece.

1) Extract a skeleton graph

For each pseudo-labeled character crop:

binarize

skeletonize

build a graph whose nodes are endpoints and junctions

edges are centerline segments

Every maximal path between:

endpoint ↔ junction

endpoint ↔ endpoint

junction ↔ junction

becomes a contiguous primitive stroke candidate.

Each primitive stroke has:

polyline points

total arc length

start/end endpoints

local curvature stats

width profile

2) Create the two directed versions of each stroke

For each primitive stroke 
𝑠
𝑖
s
i
	​

, create:

𝑠
𝑖
+
s
i
+
	​

: original direction

𝑠
𝑖
−
s
i
−
	​

: reversed direction

but enforce a mutual exclusion constraint:

choose exactly one of 
𝑠
𝑖
+
s
i
+
	​

 or 
𝑠
𝑖
−
s
i
−
	​


never both

That directly implements your “consider both directions but include the stroke only once” rule.

3) Character-level candidate plans

At the character level, do not force a single order yet. Instead, produce a small set of admissible candidate plans per character.

A candidate character plan is:

a permutation of the character’s primitive strokes

an orientation choice for each primitive stroke

the ordered point sequence obtained by arc-length resampling along those oriented strokes

Score each candidate by:

𝐽
char
=
𝜆
down
∑
𝑖
arc_len
(
𝑠
𝑖
)
+
𝜆
up
∑
𝑖
∥
end
(
𝑠
𝜋
𝑖
𝑜
𝑖
)
−
start
(
𝑠
𝜋
𝑖
+
1
𝑜
𝑖
+
1
)
∥
2
+
𝜆
curv
∑
𝑖
curv_penalty
(
𝑠
𝜋
𝑖
𝑜
𝑖
)
+
𝜆
start
 
start_prior
+
𝜆
end
 
end_prior
J
char
	​

=λ
down
	​

i
∑
	​

arc_len(s
i
	​

)+λ
up
	​

i
∑
	​

∥end(s
π
i
	​

o
i
	​

	​

)−start(s
π
i+1
	​

o
i+1
	​

	​

)∥
2
	​

+λ
curv
	​

i
∑
	​

curv_penalty(s
π
i
	​

o
i
	​

	​

)+λ
start
	​

start_prior+λ
end
	​

end_prior

The pen-down arc-length term is mostly constant, so in practice the optimizer is dominated by:

pen-up travel between strokes

penalties for awkward starts/ends

smoothness penalties

For small characters, solve this exactly with DP over subsets. For larger skeletons, use beam search.

4) Word-level stitching

This is where I’d make the hierarchy do the real work.

For each character in transcript order:

generate top K_char candidate character plans from step 3

Then the word-level planner chooses one candidate plan per character and stitches them together to minimize total pen-up travel across the word.

For a word with character plans 
𝑃
1
,
𝑃
2
,
…
,
𝑃
𝑛
P
1
	​

,P
2
	​

,…,P
n
	​

:

𝐽
word
=
∑
𝑘
=
1
𝑛
𝐽
char
(
𝑃
𝑘
)
+
𝜆
bridge
∑
𝑘
=
1
𝑛
−
1
∥
end
(
𝑃
𝑘
)
−
start
(
𝑃
𝑘
+
1
)
∥
2
+
𝜆
lig
∑
𝑘
=
1
𝑛
−
1
ligature_penalty
(
𝑃
𝑘
,
𝑃
𝑘
+
1
)
J
word
	​

=
k=1
∑
n
	​

J
char
	​

(P
k
	​

)+λ
bridge
	​

k=1
∑
n−1
	​

∥end(P
k
	​

)−start(P
k+1
	​

)∥
2
	​

+λ
lig
	​

k=1
∑
n−1
	​

ligature_penalty(P
k
	​

,P
k+1
	​

)

Default policy:

preserve transcript character order

optimize internal stroke order/orientation via the selected character plans

optionally allow a ligature bridge between adjacent characters

That gives you the hierarchical behavior you wanted: letters are planned first, then the word level chooses how those plans compose.

5) Line / paragraph / page chaining

Do the same one level up.

At the line level:

choose word plans

add bridge costs between words

optionally model writer-specific spacing and long pen-up motions

At the paragraph level:

chain lines

add return-sweep motions if desired

At the page level:

chain paragraphs

optionally include page-global travel if you want a fully continuous sequence over the page

I would store both:

a continuous full-page point stream

and the hierarchical segmentation tags (char_end, word_end, etc.)

Point scheduling along a chosen stroke plan

Once the order and direction are chosen, schedule points like this:

resample each oriented stroke by arc length with spacing Δs

emit pen_down=1 along the stroke

between consecutive strokes, emit either:

a short straight pen-up bridge with pen_down=0, or

no intermediate points, only a jump marker if you want a sparse format

Recommended defaults:

character-local spacing: Δs = 1/128 of the character box diagonal

word/line-level bridge spacing: Δs = 1/256 of the parent box diagonal

That gives you a stable dense sequence for training.

English online scribe model

I’d use the same conditioning stack as before, but the leaf model now outputs pen events.

Character-level online renderer

Inputs:

local 256×256 retina of current word state

target character token

style embedding

optional recent stroke summary

optional predicted character bbox mask

noise token or timestep if you want stochasticity

Outputs:

next event stream for the character:

dx, dy

pen_down

stroke_end

char_end

Model choices:

transformer decoder over pen events

or continuous-event decoder with MDN-style head

I’d start with a transformer decoder plus:

Huber loss on dx, dy

BCE / CE on pen-state flags

raster consistency loss after differentiable rendering

Word-level bridge planner

Optional but useful:

condition on current word state and the selected sequence of target characters

predict inter-character bridge tendencies

regularize the character-level planner so its selected plans are compatible with the word-level travel objective

Full online page generation

The higher levels remain as before:

page/paragraph/line/word/character placers produce boxes

character leaf emits pen trajectories

differentiable rasterizer writes those trajectories back to the page

Training data mixture

Use three source types:

A) True-online English

From IAM-OnDB:

native point/stroke sequences

writer IDs

text labels

line images paired with stroke data where available 
fki.tic.heia-fr.ch

Weight:

source_weight = 1.0

B) Pseudo-online English from offline handwriting

From IAM/CVL:

pseudo-labeled character boxes

skeleton-induced primitive strokes

hierarchical travel-minimizing plans

Weight:

confidence-weighted, for example 0.2 to 0.8

Confidence can depend on:

skeleton quality

branch ambiguity

transcript alignment confidence

cluster confidence from pseudo-labeling

C) Synthetic augmentation

Render text with handwriting-like fonts, then apply the same pseudo-online induction pipeline.
Use lower weight unless it empirically helps.

Losses

For the online leaf decoder:

𝐿
online
=
𝜆
𝑥
𝑦
𝐿
Huber
(
Δ
𝑥
,
Δ
𝑦
)
+
𝜆
𝑝
𝑒
𝑛
𝐿
CE
(
pen_down
)
+
𝜆
𝑠
𝑡
𝑟
𝑜
𝑘
𝑒
𝐿
CE
(
stroke_end
)
+
𝜆
𝑐
ℎ
𝑎
𝑟
𝐿
CE
(
char_end
)
+
𝜆
𝑟
𝑎
𝑠
𝑡
𝐿
raster
+
𝜆
𝑏
𝑏
𝑜
𝑥
𝐿
containment
L
online
	​

=λ
xy
	​

L
Huber
	​

(Δx,Δy)+λ
pen
	​

L
CE
	​

(pen_down)+λ
stroke
	​

L
CE
	​

(stroke_end)+λ
char
	​

L
CE
	​

(char_end)+λ
rast
	​

L
raster
	​

+λ
bbox
	​

L
containment
	​


Where:

L_raster compares rendered output against the target image crop

L_containment penalizes strokes leaving the predicted character bbox

For pseudo-online examples, multiply the whole loss by source_weight.

I’d also add a travel regularizer during training:

𝐿
travel
=
∑
𝑡
1
[
pen_down
𝑡
=
0
]
⋅
∥
Δ
𝑝
𝑡
∥
2
L
travel
	​

=
t
∑
	​

1[pen_down
t
	​

=0]⋅∥Δp
t
	​

∥
2
	​


so the decoder doesn’t invent huge pen-up jumps.

Updated schemas
1) True/pseudo-online English sequence record
JSON
{
  "schema_version": "1.1",
  "record_id": "eng_online_word_000123",
  "source_type": "true_online",
  "dataset": "IAM-OnDB",
  "document_id": "ondb_doc_001",
  "writer_id": "w_0042",
  "level": "word",
  "parent_id": "line_17",
  "target_text": "brown",
  "style_id": "style_w_0042",
  "retina_size": 256,
  "state_before_ref": "states/line_17/word_03_before.png",
  "state_after_ref": "states/line_17/word_03_after.png",
  "target_bbox_parent_norm_cxcywh": [0.42, 0.51, 0.18, 0.22],
  "events": [
    {"dx": 0.0000, "dy": 0.0000, "pen_down": 0, "stroke_end": 0, "char_end": 0, "word_end": 0, "seq_end": 0},
    {"dx": 0.0121, "dy": -0.0042, "pen_down": 1, "stroke_end": 0, "char_end": 0, "word_end": 0, "seq_end": 0}
  ],
  "source_weight": 1.0,
  "confidence": 1.0
}

For offline-induced examples:

source_type = "pseudo_online"

add skeleton_ref

add plan_cost

add orientation_choices

2) Character plan record
JSON
{
  "schema_version": "1.1",
  "record_id": "char_plan_000987",
  "source_type": "pseudo_online",
  "document_id": "iam_000123",
  "writer_id": "w_0042",
  "char_text": "e",
  "char_bbox_abs_xyxy": [132, 448, 165, 514],
  "primitive_strokes": [
    {
      "stroke_id": "s0",
      "points_fwd": [[0.11, 0.24], [0.12, 0.25]],
      "points_rev": [[0.12, 0.25], [0.11, 0.24]],
      "arc_len": 0.083
    }
  ],
  "candidate_plans": [
    {
      "plan_id": "p0",
      "ordered_strokes": [
        {"stroke_id": "s0", "orientation": "fwd"}
      ],
      "total_cost": 0.083,
      "start_point": [0.11, 0.24],
      "end_point": [0.12, 0.25]
    }
  ],
  "confidence": 0.76
}
3) Word stitching record
JSON
{
  "schema_version": "1.1",
  "record_id": "word_stitch_001",
  "source_type": "pseudo_online",
  "document_id": "iam_000123",
  "word_text": "the",
  "character_order": ["t", "h", "e"],
  "selected_character_plans": [
    {"char_id": "char_0", "plan_id": "p2"},
    {"char_id": "char_1", "plan_id": "p1"},
    {"char_id": "char_2", "plan_id": "p0"}
  ],
  "bridge_costs": [0.021, 0.014],
  "total_cost": 0.417,
  "events_ref": "online/word_stitch_001.json"
}
Training order

I’d do this in this sequence:

train style encoder

build pseudo-label hierarchy

train character-level online decoder on true-online IAM-OnDB only

add pseudo-online offline examples with confidence weighting

train word-level stitch planner

fine-tune the online decoder on mixed true/pseudo data

plug the online decoder into the existing page/paragraph/line/word placers

fine-tune full-page generation with raster consistency loss

That gives you a real online English scribe, but it still benefits from the scale of offline IAM/CVL.

The next deliverable I’d write is 
the exact DP / beam-search algorithm for character-plan and word-stitch optimization, plus 
the PyTorch dataset classes for these new online JSON records.

---

## User

nice, give them to me

---

## ChatGPT

I’m writing the exact search procedures so they’re implementable without guessing, then I’ll give you PyTorch dataset skeletons that match the JSON formats from the spec.

---

## ChatGPT

Below are both deliverables.

1) Exact search procedures

The structure is:

character plan search: choose an order over a character’s primitive strokes and an orientation for each stroke

word stitch search: choose one candidate character plan per character, preserving transcript order, minimizing bridge cost

the exact same recurrence lifts to line / paragraph / page if you want continuous pen motion across larger units

1.1 Character-plan exact DP

Assume a character has m primitive strokes extracted from the skeleton graph.

For each primitive stroke i, we have two directed versions:

(i, 0) = forward

(i, 1) = reverse

But the stroke can only be used once, so the DP state tracks stroke IDs only, not directed variants separately.

Per-stroke data

For each oriented stroke (i, d) precompute:

pts[i][d]: resampled point list

start[i][d]

end[i][d]

draw_cost[i][d]

curv_cost[i][d]

optional priors:

start_prior[i][d]

end_prior[i][d]

A simple internal stroke cost is:

𝐶
internal
(
𝑖
,
𝑑
)
=
𝜆
down
⋅
arc_len
(
𝑖
,
𝑑
)
+
𝜆
curv
⋅
curv_cost
(
𝑖
,
𝑑
)
C
internal
	​

(i,d)=λ
down
	​

⋅arc_len(i,d)+λ
curv
	​

⋅curv_cost(i,d)

Transition cost from oriented stroke (i, d1) to (j, d2):

𝐶
trans
(
(
𝑖
,
𝑑
1
)
→
(
𝑗
,
𝑑
2
)
)
=
𝜆
up
⋅
∥
end
(
𝑖
,
𝑑
1
)
−
start
(
𝑗
,
𝑑
2
)
∥
2
+
𝜆
smooth
⋅
heading_penalty
C
trans
	​

((i,d1)→(j,d2))=λ
up
	​

⋅∥end(i,d1)−start(j,d2)∥
2
	​

+λ
smooth
	​

⋅heading_penalty

Start and end costs:

𝐶
start
(
𝑖
,
𝑑
)
=
𝜆
start
⋅
start_prior
(
𝑖
,
𝑑
)
C
start
	​

(i,d)=λ
start
	​

⋅start_prior(i,d)
𝐶
end
(
𝑖
,
𝑑
)
=
𝜆
end
⋅
end_prior
(
𝑖
,
𝑑
)
C
end
	​

(i,d)=λ
end
	​

⋅end_prior(i,d)
DP state
Plain text
dp[mask][i][d] = best cost of a plan that uses exactly the stroke IDs in `mask`
                 and ends on stroke i with orientation d

where:

mask is an m-bit bitmask over primitive stroke IDs

i is the last used stroke ID

d in {0,1} is the chosen direction of that last stroke

Initialization

For every stroke i and direction d:

Plain text
mask = 1 << i
dp[mask][i][d] = C_start(i,d) + C_internal(i,d)
Transition

For each reachable state (mask, i, d1), and each unused stroke j, and each direction d2:

Plain text
new_mask = mask | (1 << j)

candidate =
    dp[mask][i][d1]
    + C_trans((i,d1) -> (j,d2))
    + C_internal(j,d2)

dp[new_mask][j][d2] = min(dp[new_mask][j][d2], candidate)
Finalization

Let full = (1 << m) - 1.

Best final plan:

Plain text
best_cost = min over i,d of dp[full][i][d] + C_end(i,d)

Backtrack pointers recover the ordered list of (stroke_id, orientation).

Complexity

This is exact Held-Karp style DP:

states: O(2^m * m * 2)

transitions: O(2^m * m^2 * 4)

That is practical for small m, which is exactly the regime you want for a single character.

Good rule:

m <= 10: exact DP

m > 10: beam search

1.2 K-best character plans

You do not just want the single best plan. The word-level stitcher works better if each character exposes K_char candidate plans.

Two workable options:

Option A: K-best per state

Store the best K_state hypotheses for each (mask, i, d) and prune after every update.

Option B: exact-best + perturbations

Compute the best exact plan, then generate alternatives by:

flipping one stroke orientation

swapping adjacent strokes

re-solving on a constrained neighborhood

For v1, I’d use Option A.

1.3 Character-plan beam search

For ambiguous or highly branched characters, use beam search over partial plans.

Beam state

A partial hypothesis contains:

used_mask

last_stroke_id

last_dir

cost_so_far

ordered_strokes

Expansion

From each beam state:

choose an unused stroke j

choose d2 in {0,1}

add transition + internal cost

Scoring with lower bound

Use:

Plain text
score = cost_so_far + admissible_lower_bound(unused_strokes)

A simple lower bound is:

sum of internal costs of all unused strokes

plus a cheap optimistic MST-like pen-up bridge estimate over unused strokes

For v1, even just summing internal costs is enough.

Pruning

Keep the best B partial hypotheses after each expansion depth.

Good defaults:

B = 16 for character search

K_char = 4 or 8

1.4 Word-stitch DP

Now preserve transcript order of characters and choose one candidate plan per character.

Suppose the word has characters c_0, ..., c_{n-1}.

For each character c_t, the character planner has already produced candidate plans:

Plain text
plans[t] = [P_t^0, P_t^1, ..., P_t^{K_t-1}]

Each plan has:

internal_cost

start_point

end_point

optional metadata like exit_heading, entry_heading

Transition cost between character plans
𝐶
bridge
(
𝑃
𝑡
−
1
𝑎
,
𝑃
𝑡
𝑏
)
=
𝜆
bridge
⋅
∥
end
(
𝑃
𝑡
−
1
𝑎
)
−
start
(
𝑃
𝑡
𝑏
)
∥
2
+
𝜆
lig
⋅
ligature_penalty
C
bridge
	​

(P
t−1
a
	​

,P
t
b
	​

)=λ
bridge
	​

⋅∥end(P
t−1
a
	​

)−start(P
t
b
	​

)∥
2
	​

+λ
lig
	​

⋅ligature_penalty
DP state
Plain text
word_dp[t][k] = best cost for the prefix c_0...c_t
                if character c_t uses candidate plan k
Initialization
Plain text
word_dp[0][k] = plans[0][k].internal_cost + word_start_cost(plans[0][k])
Transition
Plain text
word_dp[t][k] =
    plans[t][k].internal_cost
    + min over j (
        word_dp[t-1][j]
        + C_bridge(plans[t-1][j], plans[t][k])
      )
Finalization
Plain text
best_word_cost =
    min over k (
        word_dp[n-1][k] + word_end_cost(plans[n-1][k])
    )

Backtrack to recover the selected character plan per character.

This is just Viterbi over character plans.

Complexity
𝑂
(
∑
𝑡
𝐾
𝑡
−
1
𝐾
𝑡
)
O(
t
∑
	​

K
t−1
	​

K
t
	​

)

If all characters expose K_char = K, then:

𝑂
(
𝑛
𝐾
2
)
O(nK
2
)

which is cheap.

1.5 Line / paragraph / page chaining

Exactly the same recurrence applies one level up.

line stitcher chooses one candidate word plan per word

paragraph stitcher chooses one candidate line plan per line

page stitcher chooses one candidate paragraph plan per paragraph

Each higher-level object just needs:

internal_cost

start_point

end_point

For lines, you may also add a return-sweep penalty:

end of line t to start of line t+1

For pages, you may choose between:

discontinuous page writing

continuous page-writing trajectory

I would store both.

1.6 Reference Python for the search
Python
Run
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import heapq


Point = Tuple[float, float]
OrientedStroke = Tuple[int, int]  # (stroke_id, dir), dir=0 fwd, 1 rev


@dataclass
class PrimitiveStroke:
    stroke_id: int
    points_fwd: List[Point]
    points_rev: List[Point]
    arc_len_fwd: float
    arc_len_rev: float
    curv_cost_fwd: float = 0.0
    curv_cost_rev: float = 0.0
    start_prior_fwd: float = 0.0
    start_prior_rev: float = 0.0
    end_prior_fwd: float = 0.0
    end_prior_rev: float = 0.0

    def points(self, direction: int) -> List[Point]:
        return self.points_fwd if direction == 0 else self.points_rev

    def start(self, direction: int) -> Point:
        pts = self.points(direction)
        return pts[0]

    def end(self, direction: int) -> Point:
        pts = self.points(direction)
        return pts[-1]

    def arc_len(self, direction: int) -> float:
        return self.arc_len_fwd if direction == 0 else self.arc_len_rev

    def curv_cost(self, direction: int) -> float:
        return self.curv_cost_fwd if direction == 0 else self.curv_cost_rev

    def start_prior(self, direction: int) -> float:
        return self.start_prior_fwd if direction == 0 else self.start_prior_rev

    def end_prior(self, direction: int) -> float:
        return self.end_prior_fwd if direction == 0 else self.end_prior_rev


@dataclass
class PlanHypothesis:
    total_cost: float
    ordered_strokes: List[OrientedStroke]
    start_point: Point
    end_point: Point


@dataclass
class StrokeCostWeights:
    lambda_down: float = 1.0
    lambda_up: float = 1.0
    lambda_curv: float = 0.1
    lambda_start: float = 0.1
    lambda_end: float = 0.1
    lambda_smooth: float = 0.0


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heading(points: Sequence[Point]) -> Optional[Point]:
    if len(points) < 2:
        return None
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return (dx / norm, dy / norm)


def heading_penalty(stroke_a: PrimitiveStroke, dir_a: int,
                    stroke_b: PrimitiveStroke, dir_b: int) -> float:
    ha = heading(stroke_a.points(dir_a))
    hb = heading(stroke_b.points(dir_b)[:2] if len(stroke_b.points(dir_b)) >= 2 else stroke_b.points(dir_b))
    if ha is None or hb is None:
        return 0.0
    dot = max(-1.0, min(1.0, ha[0] * hb[0] + ha[1] * hb[1]))
    return 1.0 - dot


def internal_cost(stroke: PrimitiveStroke, direction: int, w: StrokeCostWeights) -> float:
    return (
        w.lambda_down * stroke.arc_len(direction)
        + w.lambda_curv * stroke.curv_cost(direction)
    )


def start_cost(stroke: PrimitiveStroke, direction: int, w: StrokeCostWeights) -> float:
    return w.lambda_start * stroke.start_prior(direction)


def end_cost(stroke: PrimitiveStroke, direction: int, w: StrokeCostWeights) -> float:
    return w.lambda_end * stroke.end_prior(direction)


def trans_cost(a: PrimitiveStroke, dir_a: int,
               b: PrimitiveStroke, dir_b: int,
               w: StrokeCostWeights) -> float:
    bridge = euclidean(a.end(dir_a), b.start(dir_b))
    smooth = heading_penalty(a, dir_a, b, dir_b)
    return w.lambda_up * bridge + w.lambda_smooth * smooth


def exact_character_plan(
    strokes: Sequence[PrimitiveStroke],
    weights: StrokeCostWeights,
) -> PlanHypothesis:
    """
    Exact DP over subsets.
    Practical for small numbers of primitive strokes, e.g. <= 10.
    """
    m = len(strokes)
    if m == 0:
        raise ValueError("Character has no primitive strokes.")

    # dp[(mask, i, d)] = best cost
    dp: Dict[Tuple[int, int, int], float] = {}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}

    for i in range(m):
        for d in (0, 1):
            mask = 1 << i
            key = (mask, i, d)
            dp[key] = start_cost(strokes[i], d, weights) + internal_cost(strokes[i], d, weights)
            parent[key] = None

    for mask in range(1, 1 << m):
        for i in range(m):
            for d1 in (0, 1):
                key = (mask, i, d1)
                if key not in dp:
                    continue
                for j in range(m):
                    if mask & (1 << j):
                        continue
                    for d2 in (0, 1):
                        new_mask = mask | (1 << j)
                        cand = (
                            dp[key]
                            + trans_cost(strokes[i], d1, strokes[j], d2, weights)
                            + internal_cost(strokes[j], d2, weights)
                        )
                        new_key = (new_mask, j, d2)
                        if cand < dp.get(new_key, float("inf")):
                            dp[new_key] = cand
                            parent[new_key] = key

    full = (1 << m) - 1
    best_key = None
    best_cost = float("inf")
    for i in range(m):
        for d in (0, 1):
            key = (full, i, d)
            if key not in dp:
                continue
            cand = dp[key] + end_cost(strokes[i], d, weights)
            if cand < best_cost:
                best_cost = cand
                best_key = key

    if best_key is None:
        raise RuntimeError("Failed to recover a complete character plan.")

    ordered: List[OrientedStroke] = []
    cur = best_key
    while cur is not None:
        _, i, d = cur
        ordered.append((i, d))
        cur = parent[cur]
    ordered.reverse()

    first_i, first_d = ordered[0]
    last_i, last_d = ordered[-1]
    return PlanHypothesis(
        total_cost=best_cost,
        ordered_strokes=ordered,
        start_point=strokes[first_i].start(first_d),
        end_point=strokes[last_i].end(last_d),
    )


def beam_character_plans(
    strokes: Sequence[PrimitiveStroke],
    weights: StrokeCostWeights,
    beam_size: int = 16,
    top_k: int = 8,
) -> List[PlanHypothesis]:
    """
    Beam search version that returns multiple candidate plans.
    Good when the character has many primitive strokes.
    """
    m = len(strokes)
    if m == 0:
        return []

    # Each hypothesis: (score_so_far, used_mask, last_i, last_d, ordered_strokes)
    beam: List[Tuple[float, int, int, int, List[OrientedStroke]]] = []

    for i in range(m):
        for d in (0, 1):
            cost = start_cost(strokes[i], d, weights) + internal_cost(strokes[i], d, weights)
            beam.append((cost, 1 << i, i, d, [(i, d)]))

    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_size]

    for _depth in range(1, m):
        next_beam: List[Tuple[float, int, int, int, List[OrientedStroke]]] = []
        for cost_so_far, used_mask, last_i, last_d, ordered in beam:
            for j in range(m):
                if used_mask & (1 << j):
                    continue
                for d2 in (0, 1):
                    new_cost = (
                        cost_so_far
                        + trans_cost(strokes[last_i], last_d, strokes[j], d2, weights)
                        + internal_cost(strokes[j], d2, weights)
                    )
                    next_beam.append((
                        new_cost,
                        used_mask | (1 << j),
                        j,
                        d2,
                        ordered + [(j, d2)],
                    ))
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_size]

    plans: List[PlanHypothesis] = []
    for cost_so_far, used_mask, last_i, last_d, ordered in beam:
        total = cost_so_far + end_cost(strokes[last_i], last_d, weights)
        first_i, first_d = ordered[0]
        plans.append(
            PlanHypothesis(
                total_cost=total,
                ordered_strokes=ordered,
                start_point=strokes[first_i].start(first_d),
                end_point=strokes[last_i].end(last_d),
            )
        )

    plans.sort(key=lambda p: p.total_cost)
    return plans[:top_k]


def stitch_word_plans(
    per_char_plans: Sequence[Sequence[PlanHypothesis]],
    bridge_lambda: float = 1.0,
    ligature_lambda: float = 0.0,
) -> Tuple[float, List[int]]:
    """
    Viterbi-style DP over transcript-ordered characters.
    Returns (best_total_cost, chosen_plan_indices).
    """
    if not per_char_plans:
        return 0.0, []

    n = len(per_char_plans)
    dp: List[List[float]] = [[float("inf")] * len(per_char_plans[t]) for t in range(n)]
    parent: List[List[Optional[int]]] = [[None] * len(per_char_plans[t]) for t in range(n)]

    # init
    for k, plan in enumerate(per_char_plans[0]):
        dp[0][k] = plan.total_cost

    # recurrence
    for t in range(1, n):
        for k, cur in enumerate(per_char_plans[t]):
            cur_internal = cur.total_cost
            best_prev = None
            best_cost = float("inf")
            for j, prev in enumerate(per_char_plans[t - 1]):
                bridge = euclidean(prev.end_point, cur.start_point)
                ligature_penalty = 0.0  # plug in heuristic/model here
                cand = dp[t - 1][j] + cur_internal + bridge_lambda * bridge + ligature_lambda * ligature_penalty
                if cand < best_cost:
                    best_cost = cand
                    best_prev = j
            dp[t][k] = best_cost
            parent[t][k] = best_prev

    last_idx = min(range(len(dp[-1])), key=lambda k: dp[-1][k])
    best_total = dp[-1][last_idx]

    chosen = [last_idx]
    for t in range(n - 1, 0, -1):
        prev_idx = parent[t][chosen[-1]]
        if prev_idx is None:
            raise RuntimeError("Broken backpointer in word stitching.")
        chosen.append(prev_idx)
    chosen.reverse()

    return best_total, chosen
2) PyTorch dataset classes

I’d split the data layer into five pieces:

HierarchyPlacementDataset
For document/page/paragraph/line/word/character placement actions

CharacterInfillDataset
For the raster infiller

OnlineSequenceDataset
For true-online or pseudo-online point sequences

CharacterPlanDataset
For loading primitive stroke sets and candidate plans

WordStitchDataset
For training a learned stitch model later, or just for analysis/debugging

The code below is meant to match the JSON formats from the earlier spec.

2.1 Shared utilities
Python
Run
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


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
    """
    Returns float tensor in [0,1], shape [1,H,W].
    """
    path = Path(path)
    with Image.open(path) as img:
        img = img.convert("L")
        x = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0
    return x.unsqueeze(0)


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
    """
    seqs: list of [T_i, F]
    returns:
      padded: [B, T_max, F]
      mask:   [B, T_max] (1 where valid)
    """
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
2.2 Hierarchy placement dataset

This handles the six placement levels:

document

page

paragraph

line

word

character

It loads the teacher-forced action records from the trajectory JSONL files.

Python
Run
class HierarchyPlacementDataset(Dataset):
    """
    Loads teacher-forced placement actions for one level.
    Expected schema fields include:
      - state_before_ref
      - parent_retina_size
      - semantic_condition.text
      - target.stop
      - target.bbox_parent_norm_cxcywh
      - style_id
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

        state = load_gray_image(self.image_root / rec["state_before_ref"])
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
            "record_id": rec["trajectory_id"],
            "level": rec["level"],
            "state": state,  # [1,H,W]
            "text_tokens": text_tokens,  # [T]
            "style_index": torch.tensor(self._style_index(rec.get("style_id")), dtype=torch.long),
            "stop": stop,  # scalar
            "bbox": bbox_tensor,  # [4]
            "bbox_valid": bbox_valid,  # scalar
            "rotation": rotation_tensor,  # scalar
            "rotation_valid": rotation_valid,  # scalar
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_hierarchy_placement(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    states = torch.stack([b["state"] for b in batch], dim=0)
    text_tokens = pad_1d_long([b["text_tokens"] for b in batch], pad_value=0)
    text_lengths = torch.tensor([b["text_tokens"].numel() for b in batch], dtype=torch.long)

    return {
        "record_id": [b["record_id"] for b in batch],
        "level": [b["level"] for b in batch],
        "state": states,  # [B,1,H,W]
        "text_tokens": text_tokens,  # [B,T]
        "text_lengths": text_lengths,  # [B]
        "style_index": torch.stack([b["style_index"] for b in batch], dim=0),
        "stop": torch.stack([b["stop"] for b in batch], dim=0),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "bbox_valid": torch.stack([b["bbox_valid"] for b in batch], dim=0),
        "rotation": torch.stack([b["rotation"] for b in batch], dim=0),
        "rotation_valid": torch.stack([b["rotation_valid"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
    }
2.3 Character infill dataset
Python
Run
class CharacterInfillDataset(Dataset):
    """
    Loads local before/after raster records for the character infiller.
    Expected fields:
      - before_patch_ref
      - after_patch_ref
      - bbox_mask_ref
      - neighbor_mask_ref
      - char_text
      - style_id
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
        neighbor_mask = load_mask_image(self.image_root / rec["neighbor_mask_ref"])

        if self.image_transform is not None:
            before = self.image_transform(before)
            after = self.image_transform(after)

        char_tokens = text_to_byte_tensor(rec.get("char_text"), max_len=self.max_text_len)
        style_index = self.style_to_index.get(rec.get("style_id"), -1)

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
2.4 Online sequence dataset

This one covers:

true-online English from IAM-OnDB

pseudo-online English from offline induction

Japanese pseudo-online stroke sequences

The event format is standardized.

Python
Run
class OnlineSequenceDataset(Dataset):
    """
    Loads point-event sequence records.
    Expected event fields:
      dx, dy, pen_down, stroke_end, char_end, word_end, seq_end
    """

    EVENT_KEYS = ("dx", "dy", "pen_down", "stroke_end", "char_end", "word_end", "seq_end")

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        style_to_index: Optional[Dict[str, int]] = None,
        load_state_images: bool = True,
        max_text_len: int = 256,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index or {}
        self.load_state_images = load_state_images
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def _events_to_tensor(cls, events: Sequence[Dict[str, Any]]) -> Tensor:
        rows = []
        for e in events:
            rows.append([
                float(e.get("dx", 0.0)),
                float(e.get("dy", 0.0)),
                float(e.get("pen_down", 0.0)),
                float(e.get("stroke_end", 0.0)),
                float(e.get("char_end", 0.0)),
                float(e.get("word_end", 0.0)),
                float(e.get("seq_end", 0.0)),
            ])
        if not rows:
            return torch.zeros(0, len(cls.EVENT_KEYS), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        state_before = None
        state_after = None
        if self.load_state_images:
            if rec.get("state_before_ref") is not None:
                state_before = load_gray_image(self.image_root / rec["state_before_ref"])
            if rec.get("state_after_ref") is not None:
                state_after = load_gray_image(self.image_root / rec["state_after_ref"])

        events = self._events_to_tensor(rec["events"])
        text_tokens = text_to_byte_tensor(rec.get("target_text"), max_len=self.max_text_len)
        style_index = self.style_to_index.get(rec.get("style_id"), -1)

        bbox = rec.get("target_bbox_parent_norm_cxcywh")
        bbox_tensor = (
            torch.full((4,), -1.0, dtype=torch.float32)
            if bbox is None
            else torch.tensor(bbox, dtype=torch.float32)
        )

        return {
            "record_id": rec["record_id"],
            "source_type": rec.get("source_type", "unknown"),
            "level": rec.get("level", "unknown"),
            "state_before": state_before,
            "state_after": state_after,
            "events": events,  # [T,7]
            "text_tokens": text_tokens,
            "style_index": torch.tensor(style_index, dtype=torch.long),
            "bbox": bbox_tensor,
            "source_weight": torch.tensor(float(rec.get("source_weight", 1.0)), dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_online_sequences(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    events, event_mask = pad_events([b["events"] for b in batch])
    text_tokens = pad_1d_long([b["text_tokens"] for b in batch], pad_value=0)

    out: Dict[str, Any] = {
        "record_id": [b["record_id"] for b in batch],
        "source_type": [b["source_type"] for b in batch],
        "level": [b["level"] for b in batch],
        "events": events,  # [B,T,7]
        "event_mask": event_mask,  # [B,T]
        "text_tokens": text_tokens,  # [B,T_text]
        "text_lengths": torch.tensor([b["text_tokens"].numel() for b in batch], dtype=torch.long),
        "style_index": torch.stack([b["style_index"] for b in batch], dim=0),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "source_weight": torch.stack([b["source_weight"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
    }

    if batch[0]["state_before"] is not None:
        out["state_before"] = torch.stack([b["state_before"] for b in batch], dim=0)
    if batch[0]["state_after"] is not None:
        out["state_after"] = torch.stack([b["state_after"] for b in batch], dim=0)

    return out
2.5 Character plan dataset

This is for loading primitive strokes and candidate plans from the pseudo-online induction stage.

Python
Run
class CharacterPlanDataset(Dataset):
    """
    Loads character-level primitive strokes and candidate plans.
    Good for:
      - debugging induction quality
      - training a learned character planner later
      - offline evaluation
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.records = read_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        primitive_strokes = []
        for s in rec["primitive_strokes"]:
            primitive_strokes.append({
                "stroke_id": s["stroke_id"],
                "points_fwd": torch.tensor(s["points_fwd"], dtype=torch.float32),
                "points_rev": torch.tensor(s["points_rev"], dtype=torch.float32),
                "arc_len": torch.tensor(float(s["arc_len"]), dtype=torch.float32),
            })

        candidate_plans = []
        for p in rec.get("candidate_plans", []):
            ordered_strokes = []
            for item in p["ordered_strokes"]:
                orientation = 0 if item["orientation"] == "fwd" else 1
                ordered_strokes.append((item["stroke_id"], orientation))
            candidate_plans.append({
                "plan_id": p["plan_id"],
                "ordered_strokes": ordered_strokes,
                "total_cost": torch.tensor(float(p["total_cost"]), dtype=torch.float32),
                "start_point": torch.tensor(p["start_point"], dtype=torch.float32),
                "end_point": torch.tensor(p["end_point"], dtype=torch.float32),
            })

        return {
            "record_id": rec["record_id"],
            "char_text": rec["char_text"],
            "primitive_strokes": primitive_strokes,
            "candidate_plans": candidate_plans,
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }

This one is best used with batch_size=1 initially, because plan counts and point counts vary a lot.

2.6 Word stitch dataset
Python
Run
class WordStitchDataset(Dataset):
    """
    Loads word-level selections over candidate character plans.
    Useful if you later train a model to predict which character plan to pick.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.records = read_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        chosen = rec["selected_character_plans"]

        return {
            "record_id": rec["record_id"],
            "word_text": rec["word_text"],
            "char_order": rec["character_order"],
            "selected_char_ids": [x["char_id"] for x in chosen],
            "selected_plan_ids": [x["plan_id"] for x in chosen],
            "bridge_costs": torch.tensor(rec["bridge_costs"], dtype=torch.float32),
            "total_cost": torch.tensor(float(rec["total_cost"]), dtype=torch.float32),
        }
3) Recommended training usage

Use the datasets like this:

Placers

Train:

HierarchyPlacementDataset

one dataloader per level

Raster leaf

Train:

CharacterInfillDataset

Online leaf

Train:

OnlineSequenceDataset

First with:

true-online IAM-OnDB only

Then with:

true-online + pseudo-online mixed

Search / induction tooling

Use:

CharacterPlanDataset

WordStitchDataset

for offline evaluation, debugging, and later learned planners.

4) Suggested first-pass wiring

I’d wire the training entry points as:

Python
Run
train_document_placer.py
train_page_placer.py
train_paragraph_placer.py
train_line_placer.py
train_word_placer.py
train_character_placer.py
train_character_infill.py
train_online_scribe.py
build_character_plans.py
build_word_stitches.py

And the generation path as:

Plain text
document placer
  -> page placer
    -> paragraph placer
      -> line placer
        -> word placer
          -> character placer
            -> online scribe OR raster infiller

For English online mode, use the online scribe at the leaf.
For English raster mode, use the raster infiller.
For Japanese, use the same online leaf API but with stroke-conditioned data.

5) Two practical defaults I’d use immediately

For character search:

exact DP for m <= 10

beam search otherwise

keep K_char = 4

For word stitching:

Viterbi over those 4 candidates per character

transcript order fixed

ligature penalty off at first

bridge distance on

That gives you a clean first system that is already faithful to your spec.

Next I’d write the actual model skeletons for:

the placement network,

the online scribe transformer,

and a differentiable rasterizer hook for point sequences.

---

## User

You do not just want the single best plan. The word-level stitcher works better if each character exposes K_char candidate plans.

Don't collapse the strokes to plans at this level, instead minimize the objective across the whole document. When we go to plan the document it should go document -> page -> paragraph -> line -> word -> character then we plan the strokes for all the words in parallel from the aggregation of the character strokes in that word. Since most people finish one word before starting the next we can stop here and then just plan the word order

---

## ChatGPT

Agreed — that is a better factorization.

The correction is:

character level should expose primitive strokes, not collapsed candidate plans

word level should solve the first real routing problem

and because people usually finish one word before starting the next, the practical global objective can stop at within-word stroke planning + between-word ordering

So the updated stack is:

document → page → paragraph → line → word → character → primitive strokes

but the actual stroke-order optimization happens first at the word level, not the character level.

Revised planning split
What the hierarchy does

The hierarchy still predicts geometry:

document places pages

page places paragraphs

paragraph places lines

line places words

word places characters

That gives you:

word boxes

character boxes inside words

primitive stroke sets for each character

What the stroke planner does

Then, for each word:

collect all primitive strokes from all characters in the word

consider both directions for each primitive stroke

choose exactly one direction per primitive stroke

produce a single ordered stroke sequence for the whole word minimizing travel distance

Then, after all words have their internal stroke plans, do a higher-level ordering over words.

So the actual optimization is now:

inner problem: optimize a route over all primitive strokes in a word

outer problem: order words within line / paragraph / page / document

That matches your “plan the strokes for all the words in parallel from the aggregation of the character strokes in that word” idea.

1) Updated data model

Character nodes should no longer store K_char plans.

They should store only:

character identity

character bbox

primitive stroke decomposition

stroke membership in the character

So the leaf object becomes:

JSON
{
  "character_id": "char_17",
  "char_text": "e",
  "char_bbox_parent_norm_cxcywh": [0.41, 0.54, 0.08, 0.72],
  "primitive_strokes": [
    {
      "stroke_id": "char_17_s0",
      "points_fwd": [[0.11, 0.24], [0.12, 0.25]],
      "points_rev": [[0.12, 0.25], [0.11, 0.24]],
      "arc_len": 0.083
    }
  ]
}

Then the word record becomes the first place where a plan exists:

JSON
{
  "record_id": "word_plan_001",
  "word_id": "word_9",
  "word_text": "the",
  "character_ids": ["char_15", "char_16", "char_17"],
  "stroke_ids": [
    "char_15_s0",
    "char_15_s1",
    "char_16_s0",
    "char_17_s0",
    "char_17_s1"
  ],
  "selected_sequence": [
    {"stroke_id": "char_15_s0", "orientation": "fwd"},
    {"stroke_id": "char_15_s1", "orientation": "rev"},
    {"stroke_id": "char_16_s0", "orientation": "fwd"},
    {"stroke_id": "char_17_s0", "orientation": "fwd"},
    {"stroke_id": "char_17_s1", "orientation": "rev"}
  ],
  "total_cost": 0.417
}
2) Updated optimization problem
Word-level routing over aggregated character strokes

For a word 
𝑤
w, let its primitive stroke set be:

𝑆
(
𝑤
)
=
{
𝑠
1
,
𝑠
2
,
…
,
𝑠
𝑚
}
S(w)={s
1
	​

,s
2
	​

,…,s
m
	​

}

Each stroke 
𝑠
𝑖
s
i
	​

 has two directed variants:

𝑠
𝑖
+
s
i
+
	​


𝑠
𝑖
−
s
i
−
	​


Constraint:

choose exactly one of 
𝑠
𝑖
+
s
i
+
	​

 or 
𝑠
𝑖
−
s
i
−
	​


Decision variables:

orientation of each stroke

total ordering of all strokes in the word

Objective:

𝐽
word
=
∑
𝑘
=
1
𝑚
𝐶
internal
(
𝑠
𝜋
𝑘
𝑜
𝑘
)
+
∑
𝑘
=
1
𝑚
−
1
𝐶
bridge
(
𝑠
𝜋
𝑘
𝑜
𝑘
,
𝑠
𝜋
𝑘
+
1
𝑜
𝑘
+
1
)
+
𝐶
start
(
𝑠
𝜋
1
𝑜
1
)
+
𝐶
end
(
𝑠
𝜋
𝑚
𝑜
𝑚
)
J
word
	​

=
k=1
∑
m
	​

C
internal
	​

(s
π
k
	​

o
k
	​

	​

)+
k=1
∑
m−1
	​

C
bridge
	​

(s
π
k
	​

o
k
	​

	​

,s
π
k+1
	​

o
k+1
	​

	​

)+C
start
	​

(s
π
1
	​

o
1
	​

	​

)+C
end
	​

(s
π
m
	​

o
m
	​

	​

)

where:

𝜋
π is a permutation of the word’s stroke IDs

𝑜
𝑘
∈
{
+
,
−
}
o
k
	​

∈{+,−}

Important difference from before

Before, I had:

character plan first

then word stitch between character plans

Now it becomes:

characters contribute stroke inventories

word planner solves over the union of those strokes directly

That is the right change.

3) Hard vs soft character constraints

At the word level, you now have to decide how strongly to preserve character-local structure.

I’d recommend:

Hard constraints

Always enforce:

each primitive stroke used exactly once

only one direction per primitive stroke

Soft constraints

Add penalties, not hard rules, for:

leaving a character before covering most of its nearby strokes

jumping backward across the word too aggressively

strong interleaving between distant characters

So add:

𝐽
char_cohesion
=
𝜆
cohesion
∑
𝑘
char_switch_penalty
(
𝑠
𝜋
𝑘
,
𝑠
𝜋
𝑘
+
1
)
J
char_cohesion
	​

=λ
cohesion
	​

k
∑
	​

char_switch_penalty(s
π
k
	​

	​

,s
π
k+1
	​

	​

)

This gives you the behavior you want:

the solver is free to optimize across the whole word

but it is nudged toward human-like within-character / within-letter continuity

You do not need to force “finish character before next character” unless you want a stricter handwriting prior.

4) Exact DP is no longer the main solver

Once you aggregate all strokes in a word, exact subset DP becomes expensive fast.

If a word has 
𝑚
m primitive strokes, exact Held-Karp is:

𝑂
(
2
𝑚
𝑚
2
)
O(2
m
m
2
)

That is fine for small characters, but not for whole words.

So the updated recommendation is:

character level: no combinatorial planning

word level: use beam search or A* over partial stroke routes

optionally refine with local search afterward

5) Word-level beam search

Each beam state represents a partial route over the word’s primitive strokes.

State:

used_mask over the word’s stroke IDs

last_stroke_id

last_dir

cost_so_far

ordered_strokes

optional summary:

current character coverage counts

number of character switches

current x-position / y-position

Transition:

choose an unused stroke j

choose direction d in {fwd, rev}

append it

Transition cost:

𝐶
=
𝐶
internal
(
𝑗
,
𝑑
)
+
𝐶
bridge
(
(
𝑖
,
𝑑
𝑖
)
→
(
𝑗
,
𝑑
)
)
+
𝐶
cohesion
+
𝐶
reading_prior
C=C
internal
	​

(j,d)+C
bridge
	​

((i,d
i
	​

)→(j,d))+C
cohesion
	​

+C
reading_prior
	​


where:

bridge = pen-up travel

cohesion = penalty for awkward character switching

reading_prior = optional penalty for large backward motion against the word’s flow

Good default beam:

B = 32 or 64

Then optionally run local improvement:

swap adjacent route segments

flip single-stroke directions

2-opt over the word route

6) Whole-document factorization

You said “minimize the objective across the whole document,” but also “most people finish one word before starting the next, so we can stop here and then just plan the word order.”

That means the full objective should be written as:

𝐽
doc
=
∑
𝑤
∈
words
𝐽
word
(
𝑤
)
+
𝜆
word_bridge
⋅
𝐽
word-order
J
doc
	​

=
w∈words
∑
	​

J
word
	​

(w)+λ
word_bridge
	​

⋅J
word-order
	​


So the document objective decomposes into:

parallel word-internal optimization

ordering over words

That is a very good compromise:

it is much closer to real writing than isolated character planning

but still computationally manageable

7) Word-order planning

Once each word has:

an internal stroke route

a start point

an end point

an internal cost

then each word becomes a super-node.

For each word 
𝑤
w, define:

start(w)

end(w)

internal_cost(w)

Then order words within a line, paragraph, page, or document.

Because English is strongly ordered, I would start with:

Line-level hard order

preserve transcript word order within the line

Then the only thing to optimize is:

where the pen exits one word and enters the next

whether you add explicit pen-up bridges

For paragraph/page/document levels, you can similarly keep natural reading order:

line order top-to-bottom

paragraph order top-to-bottom

page order as given

So for English, the “word-order planner” is mostly:

a bridge planner

not a free permutation solver

If you truly want freer document-level routing later, you can lift that constraint, but for v1 I would keep English reading order fixed.

8) Updated online training target

Your online English target should now come from:

hierarchy places words and characters

primitive strokes extracted per character

word-level route optimizer produces a full ordered stroke sequence

that sequence is converted into point events

So the main supervision unit becomes:

JSON
{
  "record_id": "eng_online_word_000123",
  "source_type": "pseudo_online",
  "level": "word",
  "word_id": "word_9",
  "word_text": "the",
  "character_ids": ["char_15", "char_16", "char_17"],
  "state_before_ref": "states/line_17/word_03_before.png",
  "state_after_ref": "states/line_17/word_03_after.png",
  "target_bbox_parent_norm_cxcywh": [0.42, 0.51, 0.18, 0.22],
  "primitive_strokes": [
    {"stroke_id": "char_15_s0", "char_id": "char_15"},
    {"stroke_id": "char_15_s1", "char_id": "char_15"},
    {"stroke_id": "char_16_s0", "char_id": "char_16"}
  ],
  "selected_sequence": [
    {"stroke_id": "char_15_s0", "orientation": "fwd"},
    {"stroke_id": "char_15_s1", "orientation": "rev"},
    {"stroke_id": "char_16_s0", "orientation": "fwd"}
  ],
  "events": [
    {"dx": 0.0000, "dy": 0.0000, "pen_down": 0, "stroke_end": 0, "char_end": 0, "word_end": 0, "seq_end": 0}
  ],
  "plan_cost": 0.417,
  "source_weight": 0.74,
  "confidence": 0.81
}

This is better than supervising one character at a time, because now the model can learn:

cross-character pen motion

ligature-like transitions

writer-specific within-word continuity

9) Updated dataset recommendation

You no longer need:

CharacterPlanDataset as “candidate plans per character”

Instead you want:

CharacterStrokeDataset

Stores:

primitive strokes per character

no combinatorial plan

WordRouteDataset

Stores:

aggregated stroke set for the word

selected route over all strokes

emitted event sequence

That is the correct dataset split now.

10) Revised search code structure

The solver modules should now be:

Plain text
build_character_strokes.py
build_word_routes.py
build_line_bridges.py
build_paragraph_bridges.py
build_page_bridges.py

where:

build_character_strokes.py
extracts primitive strokes only

build_word_routes.py
solves the real combinatorial search over the aggregated word stroke set

build_line_bridges.py
connects word routes in fixed transcript order

And your online scribe should primarily train on word-level event sequences, not isolated character plans.

11) Best v1 policy

For English, I would now do this:

hierarchy predicts:

pages

paragraphs

lines

words

characters

character stage outputs:

primitive strokes only

word stage solves:

orientation + ordering over all primitive strokes in the word

line stage:

fixed transcript word order

bridge planning only

This matches your revised design and is better than the earlier K_char plan abstraction.

The next concrete thing to write is the exact word-level beam-search algorithm over aggregated character strokes and the replacement PyTorch datasets:

CharacterStrokeDataset

WordRouteDataset

LineBridgeDataset

---

## User

Do it

---

## ChatGPT

I’m replacing the old character-plan abstraction with a word-route solver and matching datasets, so the search and training code line up with your revised factorization.

---

## ChatGPT

Below is the revised version.

It does two things:

gives the exact word-level beam-search algorithm over the aggregated primitive strokes from all characters in a word

replaces the old datasets with:

CharacterStrokeDataset

WordRouteDataset

LineBridgeDataset

1) Revised planning stack

Geometry is still hierarchical:

document → page → paragraph → line → word → character

But combinatorial stroke planning now starts here:

word → aggregated character strokes → routed word stroke sequence

So:

character level outputs primitive strokes only

word level chooses stroke order + direction over the union of all character strokes in that word

line level keeps transcript word order fixed and only plans bridges

paragraph/page/document can do the same

That means the core optimization object is now the word route.

2) Word-level optimization problem

For a word 
𝑤
w, let its aggregated primitive stroke set be:

𝑆
(
𝑤
)
=
{
𝑠
1
,
…
,
𝑠
𝑚
}
S(w)={s
1
	​

,…,s
m
	​

}

Each primitive stroke has two directions:

𝑠
𝑖
+
s
i
+
	​


𝑠
𝑖
−
s
i
−
	​


Constraint:

choose exactly one direction for each stroke

use each stroke exactly once

A complete word route is:

𝑅
=
[
(
𝑠
𝜋
1
,
𝑜
1
)
,
(
𝑠
𝜋
2
,
𝑜
2
)
,
…
,
(
𝑠
𝜋
𝑚
,
𝑜
𝑚
)
]
R=[(s
π
1
	​

	​

,o
1
	​

),(s
π
2
	​

	​

,o
2
	​

),…,(s
π
m
	​

	​

,o
m
	​

)]

where:

𝜋
π is a permutation of stroke IDs

𝑜
𝑘
∈
{
+
,
−
}
o
k
	​

∈{+,−}

Objective:

𝐽
word
(
𝑅
)
=
𝐶
start
(
𝑠
𝜋
1
𝑜
1
)
+
∑
𝑘
=
1
𝑚
𝐶
internal
(
𝑠
𝜋
𝑘
𝑜
𝑘
)
+
∑
𝑘
=
1
𝑚
−
1
𝐶
bridge
(
𝑠
𝜋
𝑘
𝑜
𝑘
,
𝑠
𝜋
𝑘
+
1
𝑜
𝑘
+
1
)
+
𝐶
end
(
𝑠
𝜋
𝑚
𝑜
𝑚
)
+
∑
𝑘
=
1
𝑚
−
1
𝐶
char_cohesion
(
𝑠
𝜋
𝑘
,
𝑠
𝜋
𝑘
+
1
)
+
∑
𝑘
=
1
𝑚
−
1
𝐶
reading_prior
(
𝑠
𝜋
𝑘
,
𝑠
𝜋
𝑘
+
1
)
J
word
	​

(R)=C
start
	​

(s
π
1
	​

o
1
	​

	​

)+
k=1
∑
m
	​

C
internal
	​

(s
π
k
	​

o
k
	​

	​

)+
k=1
∑
m−1
	​

C
bridge
	​

(s
π
k
	​

o
k
	​

	​

,s
π
k+1
	​

o
k+1
	​

	​

)+C
end
	​

(s
π
m
	​

o
m
	​

	​

)+
k=1
∑
m−1
	​

C
char_cohesion
	​

(s
π
k
	​

	​

,s
π
k+1
	​

	​

)+
k=1
∑
m−1
	​

C
reading_prior
	​

(s
π
k
	​

	​

,s
π
k+1
	​

	​

)
Cost terms
Internal stroke cost

Usually almost constant, but keep it for completeness:

𝐶
internal
(
𝑠
𝑖
𝑜
)
=
𝜆
down
⋅
arc_len
(
𝑠
𝑖
)
+
𝜆
curv
⋅
curv_cost
(
𝑠
𝑖
𝑜
)
C
internal
	​

(s
i
o
	​

)=λ
down
	​

⋅arc_len(s
i
	​

)+λ
curv
	​

⋅curv_cost(s
i
o
	​

)
Bridge cost

Pen-up travel between consecutive strokes:

𝐶
bridge
(
𝑠
𝑖
𝑜
𝑖
,
𝑠
𝑗
𝑜
𝑗
)
=
𝜆
up
⋅
∥
end
(
𝑠
𝑖
𝑜
𝑖
)
−
start
(
𝑠
𝑗
𝑜
𝑗
)
∥
2
+
𝜆
smooth
⋅
heading_penalty
C
bridge
	​

(s
i
o
i
	​

	​

,s
j
o
j
	​

	​

)=λ
up
	​

⋅
	​

end(s
i
o
i
	​

	​

)−start(s
j
o
j
	​

	​

)
	​

2
	​

+λ
smooth
	​

⋅heading_penalty
Character cohesion penalty

Discourages pathological interleaving across characters:

𝐶
char_cohesion
(
𝑠
𝑖
,
𝑠
𝑗
)
=
{
0
	
if char
(
𝑠
𝑖
)
=
char
(
𝑠
𝑗
)


𝜆
switch
	
otherwise
C
char_cohesion
	​

(s
i
	​

,s
j
	​

)={
0
λ
switch
	​

	​

if char(s
i
	​

)=char(s
j
	​

)
otherwise
	​


Optionally scale by character distance in transcript order.

Reading prior

Discourages weird backward jumps inside the word:

𝐶
reading_prior
(
𝑠
𝑖
,
𝑠
𝑗
)
=
𝜆
back
⋅
max
⁡
(
0
,
𝑥
center
(
𝑠
𝑖
)
−
𝑥
center
(
𝑠
𝑗
)
)
C
reading_prior
	​

(s
i
	​

,s
j
	​

)=λ
back
	​

⋅max(0,x
center
	​

(s
i
	​

)−x
center
	​

(s
j
	​

))

Useful for English.

3) Word-level beam search

Because this is now a whole-word route, exact subset DP is usually too expensive. Beam search is the right default.

State

A beam state contains:

used_mask: which stroke IDs have already been used

last_stroke_idx

last_dir

cost_so_far

route: ordered list of (stroke_idx, dir)

char_coverage: how many strokes used from each character

switch_count

current_end_point

Optional:

lower_bound

score = cost_so_far + lower_bound

Initialization

For every stroke i and direction d:

route starts with (i,d)

cost is:

start_cost(i,d)

internal_cost(i,d)

Expansion

From a partial state ending at (i,d_prev):

choose any unused stroke j

choose orientation d_new in {0,1}

append (j,d_new)

New cost increment:

Δ
=
𝐶
internal
(
𝑗
𝑑
new
)
+
𝐶
bridge
(
𝑖
𝑑
prev
,
𝑗
𝑑
new
)
+
𝐶
char_cohesion
(
𝑖
,
𝑗
)
+
𝐶
reading_prior
(
𝑖
,
𝑗
)
Δ=C
internal
	​

(j
d
new
	​

)+C
bridge
	​

(i
d
prev
	​

,j
d
new
	​

)+C
char_cohesion
	​

(i,j)+C
reading_prior
	​

(i,j)
Lower bound

Use a cheap admissible-ish optimistic bound:

𝐿
𝐵
=
∑
𝑢
∈
unused
min
⁡
𝑑
∈
{
+
,
−
}
𝐶
internal
(
𝑢
𝑑
)
LB=
u∈unused
∑
	​

d∈{+,−}
min
	​

C
internal
	​

(u
d
)

Optionally add:

a cheap MST-style bridge bound over unused strokes

or nearest-neighbor approximation

For v1, internal-only lower bound is fine.

Pruning

After expansion:

sort by score = cost_so_far + lower_bound

keep top B

Good defaults:

B = 32 for short words

B = 64 for longer words

Finalization

When all strokes are used:

add end_cost(last_stroke,last_dir)

choose lowest-cost complete route

Optional local improvement

After beam search returns a complete route:

try flipping one stroke direction

try swapping adjacent route items

try short 2-opt segment reversals

accept improving moves greedily

That usually gives a noticeable improvement for cheap.

4) Reference Python: word-level beam search
Python
Run
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import heapq


Point = Tuple[float, float]


@dataclass
class PrimitiveStroke:
    stroke_id: str
    char_id: str
    char_index: int  # transcript order of the character within the word
    points_fwd: List[Point]
    points_rev: List[Point]
    arc_len: float
    curv_cost_fwd: float = 0.0
    curv_cost_rev: float = 0.0

    def points(self, direction: int) -> List[Point]:
        return self.points_fwd if direction == 0 else self.points_rev

    def start(self, direction: int) -> Point:
        pts = self.points(direction)
        return pts[0]

    def end(self, direction: int) -> Point:
        pts = self.points(direction)
        return pts[-1]

    def center_x(self) -> float:
        pts = self.points_fwd
        return sum(p[0] for p in pts) / max(1, len(pts))


@dataclass
class WordRouteWeights:
    lambda_down: float = 1.0
    lambda_up: float = 1.0
    lambda_curv: float = 0.05
    lambda_smooth: float = 0.0
    lambda_switch: float = 0.1
    lambda_back: float = 0.1
    lambda_start: float = 0.0
    lambda_end: float = 0.0


@dataclass(order=True)
class BeamState:
    score: float
    cost_so_far: float = field(compare=False)
    used_mask: int = field(compare=False)
    last_idx: int = field(compare=False)
    last_dir: int = field(compare=False)
    route: List[Tuple[int, int]] = field(compare=False, default_factory=list)
    switch_count: int = field(compare=False, default=0)


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heading(points: Sequence[Point]) -> Optional[Point]:
    if len(points) < 2:
        return None
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return (dx / norm, dy / norm)


def heading_penalty(a: PrimitiveStroke, dir_a: int, b: PrimitiveStroke, dir_b: int) -> float:
    pts_a = a.points(dir_a)
    pts_b = b.points(dir_b)
    if len(pts_a) < 2 or len(pts_b) < 2:
        return 0.0
    ha = heading(pts_a)
    hb = heading(pts_b[:2])
    if ha is None or hb is None:
        return 0.0
    dot = max(-1.0, min(1.0, ha[0] * hb[0] + ha[1] * hb[1]))
    return 1.0 - dot


def internal_cost(s: PrimitiveStroke, direction: int, w: WordRouteWeights) -> float:
    curv = s.curv_cost_fwd if direction == 0 else s.curv_cost_rev
    return w.lambda_down * s.arc_len + w.lambda_curv * curv


def start_cost(s: PrimitiveStroke, direction: int, w: WordRouteWeights) -> float:
    return w.lambda_start * 0.0


def end_cost(s: PrimitiveStroke, direction: int, w: WordRouteWeights) -> float:
    return w.lambda_end * 0.0


def bridge_cost(a: PrimitiveStroke, dir_a: int, b: PrimitiveStroke, dir_b: int, w: WordRouteWeights) -> float:
    bridge = euclidean(a.end(dir_a), b.start(dir_b))
    smooth = heading_penalty(a, dir_a, b, dir_b)
    return w.lambda_up * bridge + w.lambda_smooth * smooth


def char_cohesion_cost(a: PrimitiveStroke, b: PrimitiveStroke, w: WordRouteWeights) -> float:
    return 0.0 if a.char_id == b.char_id else w.lambda_switch


def reading_prior_cost(a: PrimitiveStroke, b: PrimitiveStroke, w: WordRouteWeights) -> float:
    # Penalize large backward moves in x for English-like scripts.
    dx_back = max(0.0, a.center_x() - b.center_x())
    # Also discourage big backward jumps in transcript character order.
    char_back = max(0, a.char_index - b.char_index)
    return w.lambda_back * (dx_back + float(char_back))


def lower_bound_unused(strokes: Sequence[PrimitiveStroke], used_mask: int, w: WordRouteWeights) -> float:
    lb = 0.0
    for i, s in enumerate(strokes):
        if not (used_mask & (1 << i)):
            lb += min(internal_cost(s, 0, w), internal_cost(s, 1, w))
    return lb


def build_word_route_beam(
    strokes: Sequence[PrimitiveStroke],
    weights: WordRouteWeights,
    beam_size: int = 64,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Returns:
      best_total_cost,
      best_route = [(stroke_idx, dir), ...]
    """
    m = len(strokes)
    if m == 0:
        return 0.0, []

    beam: List[BeamState] = []

    for i, s in enumerate(strokes):
        for d in (0, 1):
            used = 1 << i
            cost = start_cost(s, d, weights) + internal_cost(s, d, weights)
            lb = lower_bound_unused(strokes, used, weights)
            beam.append(
                BeamState(
                    score=cost + lb,
                    cost_so_far=cost,
                    used_mask=used,
                    last_idx=i,
                    last_dir=d,
                    route=[(i, d)],
                    switch_count=0,
                )
            )

    beam = heapq.nsmallest(beam_size, beam)

    for _depth in range(1, m):
        expanded: List[BeamState] = []

        for state in beam:
            last_s = strokes[state.last_idx]

            for j, s in enumerate(strokes):
                if state.used_mask & (1 << j):
                    continue

                for d in (0, 1):
                    delta = (
                        internal_cost(s, d, weights)
                        + bridge_cost(last_s, state.last_dir, s, d, weights)
                        + char_cohesion_cost(last_s, s, weights)
                        + reading_prior_cost(last_s, s, weights)
                    )

                    new_used = state.used_mask | (1 << j)
                    new_cost = state.cost_so_far + delta
                    lb = lower_bound_unused(strokes, new_used, weights)

                    expanded.append(
                        BeamState(
                            score=new_cost + lb,
                            cost_so_far=new_cost,
                            used_mask=new_used,
                            last_idx=j,
                            last_dir=d,
                            route=state.route + [(j, d)],
                            switch_count=state.switch_count + int(last_s.char_id != s.char_id),
                        )
                    )

        if not expanded:
            break

        beam = heapq.nsmallest(beam_size, expanded)

    best_total = float("inf")
    best_route: List[Tuple[int, int]] = []

    for state in beam:
        s = strokes[state.last_idx]
        total = state.cost_so_far + end_cost(s, state.last_dir, weights)
        if total < best_total:
            best_total = total
            best_route = state.route

    return best_total, best_route


def greedy_local_refine_word_route(
    strokes: Sequence[PrimitiveStroke],
    route: List[Tuple[int, int]],
    weights: WordRouteWeights,
    max_passes: int = 3,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Cheap local improvement:
      - flip one orientation
      - swap adjacent items
    """
    def total_cost(rt: List[Tuple[int, int]]) -> float:
        if not rt:
            return 0.0
        total = 0.0
        i0, d0 = rt[0]
        total += start_cost(strokes[i0], d0, weights) + internal_cost(strokes[i0], d0, weights)
        for t in range(1, len(rt)):
            i_prev, d_prev = rt[t - 1]
            i_cur, d_cur = rt[t]
            total += (
                internal_cost(strokes[i_cur], d_cur, weights)
                + bridge_cost(strokes[i_prev], d_prev, strokes[i_cur], d_cur, weights)
                + char_cohesion_cost(strokes[i_prev], strokes[i_cur], weights)
                + reading_prior_cost(strokes[i_prev], strokes[i_cur], weights)
            )
        i_last, d_last = rt[-1]
        total += end_cost(strokes[i_last], d_last, weights)
        return total

    best = list(route)
    best_cost = total_cost(best)

    for _ in range(max_passes):
        improved = False

        # Try direction flips
        for t in range(len(best)):
            cand = list(best)
            i, d = cand[t]
            cand[t] = (i, 1 - d)
            c = total_cost(cand)
            if c < best_cost:
                best, best_cost = cand, c
                improved = True

        # Try adjacent swaps
        for t in range(len(best) - 1):
            cand = list(best)
            cand[t], cand[t + 1] = cand[t + 1], cand[t]
            c = total_cost(cand)
            if c < best_cost:
                best, best_cost = cand, c
                improved = True

        if not improved:
            break

    return best_cost, best
5) Word-level event construction

Once the word route is chosen, convert the ordered strokes into pen events.

For each selected (stroke_idx, dir):

resample points along the oriented stroke

emit pen-up bridge from previous stroke end to current stroke start

emit pen-down points along the stroke

mark stroke_end=1 at the final point of that stroke

mark char_end=1 when the next stroke belongs to a different character

mark word_end=1 at the final point of the last stroke

Reference Python: route to events
Python
Run
from typing import Any


def resample_polyline(points: Sequence[Point], step: float) -> List[Point]:
    if len(points) <= 1:
        return list(points)

    out = [points[0]]
    acc = 0.0
    prev = points[0]

    for cur in points[1:]:
        seg_len = euclidean(prev, cur)
        if seg_len == 0:
            prev = cur
            continue

        direction = ((cur[0] - prev[0]) / seg_len, (cur[1] - prev[1]) / seg_len)
        remaining = seg_len
        p = prev

        while acc + remaining >= step:
            needed = step - acc
            new_p = (p[0] + direction[0] * needed, p[1] + direction[1] * needed)
            out.append(new_p)
            p = new_p
            remaining -= needed
            acc = 0.0

        acc += remaining
        prev = cur

    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def route_to_events(
    strokes: Sequence[PrimitiveStroke],
    route: List[Tuple[int, int]],
    stroke_step: float = 0.01,
    bridge_step: float = 0.01,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    def emit_delta(prev: Point, cur: Point, pen_down: int, stroke_end: int, char_end: int, word_end: int, seq_end: int):
        events.append({
            "dx": float(cur[0] - prev[0]),
            "dy": float(cur[1] - prev[1]),
            "pen_down": int(pen_down),
            "stroke_end": int(stroke_end),
            "char_end": int(char_end),
            "word_end": int(word_end),
            "seq_end": int(seq_end),
        })

    prev_point: Optional[Point] = None

    for t, (idx, direction) in enumerate(route):
        s = strokes[idx]
        pts = resample_polyline(s.points(direction), step=stroke_step)
        if not pts:
            continue

        # Pen-up bridge
        if prev_point is None:
            prev_point = pts[0]
            events.append({
                "dx": 0.0, "dy": 0.0,
                "pen_down": 0, "stroke_end": 0, "char_end": 0, "word_end": 0, "seq_end": 0
            })
        else:
            bridge_pts = resample_polyline([prev_point, pts[0]], step=bridge_step)
            for bp in bridge_pts[1:]:
                emit_delta(prev_point, bp, pen_down=0, stroke_end=0, char_end=0, word_end=0, seq_end=0)
                prev_point = bp

        # Pen-down along stroke
        for p in pts[1:]:
            next_char_end = 0
            next_word_end = 0
            next_seq_end = 0
            next_stroke_end = 0

            is_last_point_of_stroke = (p == pts[-1])
            if is_last_point_of_stroke:
                next_stroke_end = 1
                if t == len(route) - 1:
                    next_word_end = 1
                    next_seq_end = 1
                else:
                    cur_char = strokes[route[t][0]].char_id
                    next_char = strokes[route[t + 1][0]].char_id
                    if cur_char != next_char:
                        next_char_end = 1

            emit_delta(
                prev_point, p,
                pen_down=1,
                stroke_end=next_stroke_end,
                char_end=next_char_end,
                word_end=next_word_end,
                seq_end=next_seq_end,
            )
            prev_point = p

    return events
6) Replacement dataset classes

You asked for the replacements. These are the new three.

6.1 CharacterStrokeDataset

Purpose:

load character-level primitive strokes only

no candidate plans

this is the output of the offline pseudo-online induction stage

Expected JSONL schema
JSON
{
  "record_id": "char_strokes_0001",
  "document_id": "iam_000123",
  "word_id": "word_9",
  "char_id": "char_17",
  "char_index": 2,
  "char_text": "e",
  "bbox_parent_norm_cxcywh": [0.41, 0.54, 0.08, 0.72],
  "primitive_strokes": [
    {
      "stroke_id": "char_17_s0",
      "points_fwd": [[0.11, 0.24], [0.12, 0.25]],
      "points_rev": [[0.12, 0.25], [0.11, 0.24]],
      "arc_len": 0.083
    }
  ],
  "confidence": 0.81
}
Dataset code
Python
Run
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import torch
from torch.utils.data import Dataset


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class CharacterStrokeDataset(Dataset):
    """
    Character-level primitive strokes only.
    No combinatorial plans.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self.records = read_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        primitive_strokes = []
        for s in rec["primitive_strokes"]:
            primitive_strokes.append({
                "stroke_id": s["stroke_id"],
                "points_fwd": torch.tensor(s["points_fwd"], dtype=torch.float32),
                "points_rev": torch.tensor(s["points_rev"], dtype=torch.float32),
                "arc_len": torch.tensor(float(s["arc_len"]), dtype=torch.float32),
            })

        return {
            "record_id": rec["record_id"],
            "document_id": rec["document_id"],
            "word_id": rec["word_id"],
            "char_id": rec["char_id"],
            "char_index": int(rec["char_index"]),
            "char_text": rec["char_text"],
            "bbox": torch.tensor(rec["bbox_parent_norm_cxcywh"], dtype=torch.float32),
            "primitive_strokes": primitive_strokes,
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }

Use batch_size=1 or a custom collate that leaves stroke lists ragged.

6.2 WordRouteDataset

Purpose:

training the online word-level scribe

loading word-level optimized stroke routes and emitted point events

Expected JSONL schema
JSON
{
  "record_id": "word_route_001",
  "document_id": "iam_000123",
  "line_id": "line_17",
  "word_id": "word_9",
  "word_index": 3,
  "word_text": "the",
  "character_ids": ["char_15", "char_16", "char_17"],
  "state_before_ref": "states/line_17/word_03_before.png",
  "state_after_ref": "states/line_17/word_03_after.png",
  "target_bbox_parent_norm_cxcywh": [0.42, 0.51, 0.18, 0.22],
  "primitive_strokes": [
    {"stroke_id": "char_15_s0", "char_id": "char_15", "char_index": 0, "points_fwd": [[0.1,0.2]], "points_rev": [[0.1,0.2]], "arc_len": 0.1}
  ],
  "selected_sequence": [
    {"stroke_id": "char_15_s0", "orientation": "fwd"}
  ],
  "events": [
    {"dx": 0.0, "dy": 0.0, "pen_down": 0, "stroke_end": 0, "char_end": 0, "word_end": 0, "seq_end": 0}
  ],
  "plan_cost": 0.417,
  "source_type": "pseudo_online",
  "source_weight": 0.74,
  "confidence": 0.81
}
Dataset code
Python
Run
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import numpy as np


def load_gray_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class WordRouteDataset(Dataset):
    """
    Word-level routed stroke sequences.
    This is the main dataset for the online English word scribe.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_text_len: int = 64,
        load_images: bool = True,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.image_transform = image_transform
        self.max_text_len = max_text_len
        self.load_images = load_images

    def _text_to_bytes(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8"))[: self.max_text_len], dtype=torch.long)

    def _events_to_tensor(self, events: Sequence[Dict[str, Any]]) -> torch.Tensor:
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
        return torch.tensor(rows, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        state_before = None
        state_after = None
        if self.load_images:
            state_before = load_gray_image(self.image_root / rec["state_before_ref"])
            state_after = load_gray_image(self.image_root / rec["state_after_ref"])
            if self.image_transform is not None:
                state_before = self.image_transform(state_before)
                state_after = self.image_transform(state_after)

        primitive_strokes = []
        for s in rec["primitive_strokes"]:
            primitive_strokes.append({
                "stroke_id": s["stroke_id"],
                "char_id": s["char_id"],
                "char_index": int(s["char_index"]),
                "points_fwd": torch.tensor(s["points_fwd"], dtype=torch.float32),
                "points_rev": torch.tensor(s["points_rev"], dtype=torch.float32),
                "arc_len": torch.tensor(float(s["arc_len"]), dtype=torch.float32),
            })

        orientation_map = {
            item["stroke_id"]: 0 if item["orientation"] == "fwd" else 1
            for item in rec["selected_sequence"]
        }

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
            "primitive_strokes": primitive_strokes,
            "selected_sequence": rec["selected_sequence"],
            "stroke_orientation_map": orientation_map,
            "events": self._events_to_tensor(rec["events"]),
            "plan_cost": torch.tensor(float(rec["plan_cost"]), dtype=torch.float32),
            "source_weight": torch.tensor(float(rec.get("source_weight", 1.0)), dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
            "source_type": rec.get("source_type", "pseudo_online"),
        }


def pad_1d_long(seqs: Sequence[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def pad_events(seqs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(s.shape[0] for s in seqs)
    feat_dim = seqs[0].shape[1]
    batch = len(seqs)

    out = torch.zeros(batch, max_len, feat_dim, dtype=seqs[0].dtype)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)

    for i, s in enumerate(seqs):
        t = s.shape[0]
        out[i, :t] = s
        mask[i, :t] = True

    return out, mask


def collate_word_routes(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "record_id": [b["record_id"] for b in batch],
        "document_id": [b["document_id"] for b in batch],
        "line_id": [b["line_id"] for b in batch],
        "word_id": [b["word_id"] for b in batch],
        "word_index": torch.tensor([b["word_index"] for b in batch], dtype=torch.long),
        "word_text": [b["word_text"] for b in batch],
        "text_tokens": pad_1d_long([b["text_tokens"] for b in batch], pad_value=0),
        "text_lengths": torch.tensor([b["text_tokens"].numel() for b in batch], dtype=torch.long),
        "bbox": torch.stack([b["bbox"] for b in batch], dim=0),
        "events": pad_events([b["events"] for b in batch])[0],
        "event_mask": pad_events([b["events"] for b in batch])[1],
        "plan_cost": torch.stack([b["plan_cost"] for b in batch], dim=0),
        "source_weight": torch.stack([b["source_weight"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
        "source_type": [b["source_type"] for b in batch],
        "primitive_strokes": [b["primitive_strokes"] for b in batch],
        "selected_sequence": [b["selected_sequence"] for b in batch],
        "stroke_orientation_map": [b["stroke_orientation_map"] for b in batch],
    }

    if batch[0]["state_before"] is not None:
        out["state_before"] = torch.stack([b["state_before"] for b in batch], dim=0)
    if batch[0]["state_after"] is not None:
        out["state_after"] = torch.stack([b["state_after"] for b in batch], dim=0)

    return out
6.3 LineBridgeDataset

Purpose:

keep transcript word order fixed

train or evaluate line-level bridge prediction between already-routed words

Each item is a transition between neighboring words in a line.

Expected JSONL schema
JSON
{
  "record_id": "line_bridge_001",
  "document_id": "iam_000123",
  "line_id": "line_17",
  "prev_word_id": "word_9",
  "next_word_id": "word_10",
  "prev_word_text": "the",
  "next_word_text": "quick",
  "prev_word_end_point": [0.41, 0.52],
  "next_word_start_point": [0.45, 0.50],
  "bridge_events": [
    {"dx": 0.01, "dy": 0.00, "pen_down": 0, "stroke_end": 0, "char_end": 0, "word_end": 0, "seq_end": 0}
  ],
  "bridge_cost": 0.021,
  "confidence": 0.92
}
Dataset code
Python
Run
class LineBridgeDataset(Dataset):
    """
    Fixed transcript-order line bridge records.
    Each item is a bridge between adjacent routed words in a line.
    """

    def __init__(self, jsonl_path: str | Path, max_text_len: int = 64) -> None:
        self.records = read_jsonl(jsonl_path)
        self.max_text_len = max_text_len

    def _text_to_bytes(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8"))[: self.max_text_len], dtype=torch.long)

    def _events_to_tensor(self, events: Sequence[Dict[str, Any]]) -> torch.Tensor:
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
        return torch.tensor(rows, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "record_id": rec["record_id"],
            "document_id": rec["document_id"],
            "line_id": rec["line_id"],
            "prev_word_id": rec["prev_word_id"],
            "next_word_id": rec["next_word_id"],
            "prev_word_text": rec["prev_word_text"],
            "next_word_text": rec["next_word_text"],
            "prev_word_tokens": self._text_to_bytes(rec["prev_word_text"]),
            "next_word_tokens": self._text_to_bytes(rec["next_word_text"]),
            "prev_word_end_point": torch.tensor(rec["prev_word_end_point"], dtype=torch.float32),
            "next_word_start_point": torch.tensor(rec["next_word_start_point"], dtype=torch.float32),
            "bridge_events": self._events_to_tensor(rec["bridge_events"]),
            "bridge_cost": torch.tensor(float(rec["bridge_cost"]), dtype=torch.float32),
            "confidence": torch.tensor(float(rec.get("confidence", 1.0)), dtype=torch.float32),
        }


def collate_line_bridges(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bridge_events, bridge_mask = pad_events([b["bridge_events"] for b in batch])

    return {
        "record_id": [b["record_id"] for b in batch],
        "document_id": [b["document_id"] for b in batch],
        "line_id": [b["line_id"] for b in batch],
        "prev_word_id": [b["prev_word_id"] for b in batch],
        "next_word_id": [b["next_word_id"] for b in batch],
        "prev_word_text": [b["prev_word_text"] for b in batch],
        "next_word_text": [b["next_word_text"] for b in batch],
        "prev_word_tokens": pad_1d_long([b["prev_word_tokens"] for b in batch], pad_value=0),
        "prev_word_lengths": torch.tensor([b["prev_word_tokens"].numel() for b in batch], dtype=torch.long),
        "next_word_tokens": pad_1d_long([b["next_word_tokens"] for b in batch], pad_value=0),
        "next_word_lengths": torch.tensor([b["next_word_tokens"].numel() for b in batch], dtype=torch.long),
        "prev_word_end_point": torch.stack([b["prev_word_end_point"] for b in batch], dim=0),
        "next_word_start_point": torch.stack([b["next_word_start_point"] for b in batch], dim=0),
        "bridge_events": bridge_events,
        "bridge_mask": bridge_mask,
        "bridge_cost": torch.stack([b["bridge_cost"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
    }
7) Builder scripts you now want

The old character-plan builder should be replaced with:

Plain text
build_character_strokes.py
build_word_routes.py
build_line_bridges.py
build_character_strokes.py

Input:

pseudo-labeled character boxes

skeleton graphs

Output:

primitive stroke decomposition per character

no ordering

build_word_routes.py

Input:

word nodes

constituent character strokes

Process:

aggregate all primitive strokes in the word

run word-level beam search

optionally local refine

convert route to events

Output:

routed word records for WordRouteDataset

build_line_bridges.py

Input:

routed words in transcript order

Process:

connect end of word 
𝑖
i to start of word 
𝑖
+
1
i+1

emit pen-up bridge events

Output:

bridge records for LineBridgeDataset

8) Training usage
Online English word scribe

Main dataset:

WordRouteDataset

Target:

predict the word-level event stream conditioned on:

local state retina

word text

optional character boxes

style embedding

Character induction / debugging

Use:

CharacterStrokeDataset

Line bridge model

Optional:

LineBridgeDataset

Can be:

rule-based at first

learned later

9) Best immediate defaults

I’d actually start with these:

word beam size: 64

no free word permutation

fixed transcript order within line

reading prior on

character-switch penalty on, but small

local refinement on after beam search

That gives you a realistic first system:

global enough to optimize across the whole word

still tractable

matches how people usually write

The next thing to write is the actual builder scripts:

build_word_routes.py

build_line_bridges.py

with file I/O and JSONL emission so you can generate the datasets directly.

---

## User

sure

---

## ChatGPT

Here are the two builder scripts.

They assume:

character_strokes.jsonl contains one record per character with primitive strokes

word_metadata.jsonl contains one record per word with the word-level metadata needed to emit WordRouteDataset

word_routes.jsonl is the output of the first script and the input to the second

I also made the scripts conservative about schemas so they’re easy to adapt.

build_word_routes.py
Python
Run
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


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


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heading(points: Sequence[Point]) -> Optional[Point]:
    if len(points) < 2:
        return None
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return (dx / norm, dy / norm)


def resample_polyline(points: Sequence[Point], step: float) -> List[Point]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points[0]]

    out = [points[0]]
    carry = 0.0
    prev = points[0]

    for cur in points[1:]:
        seg_len = euclidean(prev, cur)
        if seg_len == 0:
            prev = cur
            continue

        dir_x = (cur[0] - prev[0]) / seg_len
        dir_y = (cur[1] - prev[1]) / seg_len
        remaining = seg_len
        cur_pt = prev

        while carry + remaining >= step:
            needed = step - carry
            new_pt = (cur_pt[0] + dir_x * needed, cur_pt[1] + dir_y * needed)
            out.append(new_pt)
            cur_pt = new_pt
            remaining -= needed
            carry = 0.0

        carry += remaining
        prev = cur

    if out[-1] != points[-1]:
        out.append(points[-1])

    return out


@dataclass
class PrimitiveStroke:
    stroke_id: str
    char_id: str
    char_index: int
    points_fwd: List[Point]
    points_rev: List[Point]
    arc_len: float
    curv_cost_fwd: float = 0.0
    curv_cost_rev: float = 0.0

    def points(self, direction: int) -> List[Point]:
        return self.points_fwd if direction == 0 else self.points_rev

    def start(self, direction: int) -> Point:
        pts = self.points(direction)
        return pts[0]

    def end(self, direction: int) -> Point:
        pts = self.points(direction)
        return pts[-1]

    def center_x(self) -> float:
        pts = self.points_fwd
        return sum(p[0] for p in pts) / max(1, len(pts))


@dataclass
class WordRouteWeights:
    lambda_down: float = 1.0
    lambda_up: float = 1.0
    lambda_curv: float = 0.05
    lambda_smooth: float = 0.0
    lambda_switch: float = 0.1
    lambda_back: float = 0.1
    lambda_start: float = 0.0
    lambda_end: float = 0.0


@dataclass(order=True)
class BeamState:
    score: float
    cost_so_far: float = field(compare=False)
    used_mask: int = field(compare=False)
    last_idx: int = field(compare=False)
    last_dir: int = field(compare=False)
    route: List[Tuple[int, int]] = field(compare=False, default_factory=list)
    switch_count: int = field(compare=False, default=0)


def heading_penalty(a: PrimitiveStroke, dir_a: int, b: PrimitiveStroke, dir_b: int) -> float:
    pts_a = a.points(dir_a)
    pts_b = b.points(dir_b)
    if len(pts_a) < 2 or len(pts_b) < 2:
        return 0.0
    ha = heading(pts_a)
    hb = heading(pts_b[:2])
    if ha is None or hb is None:
        return 0.0
    dot = max(-1.0, min(1.0, ha[0] * hb[0] + ha[1] * hb[1]))
    return 1.0 - dot


def internal_cost(s: PrimitiveStroke, direction: int, w: WordRouteWeights) -> float:
    curv = s.curv_cost_fwd if direction == 0 else s.curv_cost_rev
    return w.lambda_down * s.arc_len + w.lambda_curv * curv


def start_cost(_: PrimitiveStroke, __: int, w: WordRouteWeights) -> float:
    return w.lambda_start * 0.0


def end_cost(_: PrimitiveStroke, __: int, w: WordRouteWeights) -> float:
    return w.lambda_end * 0.0


def bridge_cost(a: PrimitiveStroke, dir_a: int, b: PrimitiveStroke, dir_b: int, w: WordRouteWeights) -> float:
    bridge = euclidean(a.end(dir_a), b.start(dir_b))
    smooth = heading_penalty(a, dir_a, b, dir_b)
    return w.lambda_up * bridge + w.lambda_smooth * smooth


def char_cohesion_cost(a: PrimitiveStroke, b: PrimitiveStroke, w: WordRouteWeights) -> float:
    return 0.0 if a.char_id == b.char_id else w.lambda_switch


def reading_prior_cost(a: PrimitiveStroke, b: PrimitiveStroke, w: WordRouteWeights) -> float:
    dx_back = max(0.0, a.center_x() - b.center_x())
    char_back = max(0, a.char_index - b.char_index)
    return w.lambda_back * (dx_back + float(char_back))


def lower_bound_unused(strokes: Sequence[PrimitiveStroke], used_mask: int, w: WordRouteWeights) -> float:
    lb = 0.0
    for i, s in enumerate(strokes):
        if not (used_mask & (1 << i)):
            lb += min(internal_cost(s, 0, w), internal_cost(s, 1, w))
    return lb


def build_word_route_beam(
    strokes: Sequence[PrimitiveStroke],
    weights: WordRouteWeights,
    beam_size: int = 64,
) -> Tuple[float, List[Tuple[int, int]]]:
    m = len(strokes)
    if m == 0:
        return 0.0, []

    if m > 62:
        raise ValueError(f"Word has {m} strokes; bitmask code currently supports up to 62 cleanly.")

    beam: List[BeamState] = []

    for i, s in enumerate(strokes):
        for d in (0, 1):
            used = 1 << i
            cost = start_cost(s, d, weights) + internal_cost(s, d, weights)
            lb = lower_bound_unused(strokes, used, weights)
            beam.append(
                BeamState(
                    score=cost + lb,
                    cost_so_far=cost,
                    used_mask=used,
                    last_idx=i,
                    last_dir=d,
                    route=[(i, d)],
                    switch_count=0,
                )
            )

    beam.sort()
    beam = beam[:beam_size]

    for _depth in range(1, m):
        expanded: List[BeamState] = []

        for state in beam:
            last_s = strokes[state.last_idx]

            for j, s in enumerate(strokes):
                if state.used_mask & (1 << j):
                    continue

                for d in (0, 1):
                    delta = (
                        internal_cost(s, d, weights)
                        + bridge_cost(last_s, state.last_dir, s, d, weights)
                        + char_cohesion_cost(last_s, s, weights)
                        + reading_prior_cost(last_s, s, weights)
                    )
                    new_used = state.used_mask | (1 << j)
                    new_cost = state.cost_so_far + delta
                    lb = lower_bound_unused(strokes, new_used, weights)

                    expanded.append(
                        BeamState(
                            score=new_cost + lb,
                            cost_so_far=new_cost,
                            used_mask=new_used,
                            last_idx=j,
                            last_dir=d,
                            route=state.route + [(j, d)],
                            switch_count=state.switch_count + int(last_s.char_id != s.char_id),
                        )
                    )

        if not expanded:
            break

        expanded.sort()
        beam = expanded[:beam_size]

    best_total = float("inf")
    best_route: List[Tuple[int, int]] = []

    for state in beam:
        s = strokes[state.last_idx]
        total = state.cost_so_far + end_cost(s, state.last_dir, weights)
        if total < best_total:
            best_total = total
            best_route = state.route

    return best_total, best_route


def total_route_cost(
    strokes: Sequence[PrimitiveStroke],
    route: List[Tuple[int, int]],
    weights: WordRouteWeights,
) -> float:
    if not route:
        return 0.0

    i0, d0 = route[0]
    total = start_cost(strokes[i0], d0, weights) + internal_cost(strokes[i0], d0, weights)

    for t in range(1, len(route)):
        i_prev, d_prev = route[t - 1]
        i_cur, d_cur = route[t]
        total += (
            internal_cost(strokes[i_cur], d_cur, weights)
            + bridge_cost(strokes[i_prev], d_prev, strokes[i_cur], d_cur, weights)
            + char_cohesion_cost(strokes[i_prev], strokes[i_cur], weights)
            + reading_prior_cost(strokes[i_prev], strokes[i_cur], weights)
        )

    i_last, d_last = route[-1]
    total += end_cost(strokes[i_last], d_last, weights)
    return total


def greedy_local_refine_word_route(
    strokes: Sequence[PrimitiveStroke],
    route: List[Tuple[int, int]],
    weights: WordRouteWeights,
    max_passes: int = 3,
) -> Tuple[float, List[Tuple[int, int]]]:
    best = list(route)
    best_cost = total_route_cost(strokes, best, weights)

    for _ in range(max_passes):
        improved = False

        for t in range(len(best)):
            cand = list(best)
            i, d = cand[t]
            cand[t] = (i, 1 - d)
            c = total_route_cost(strokes, cand, weights)
            if c < best_cost:
                best, best_cost = cand, c
                improved = True

        for t in range(len(best) - 1):
            cand = list(best)
            cand[t], cand[t + 1] = cand[t + 1], cand[t]
            c = total_route_cost(strokes, cand, weights)
            if c < best_cost:
                best, best_cost = cand, c
                improved = True

        if not improved:
            break

    return best_cost, best


def route_to_events(
    strokes: Sequence[PrimitiveStroke],
    route: List[Tuple[int, int]],
    stroke_step: float = 0.01,
    bridge_step: float = 0.01,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    def emit_delta(prev: Point, cur: Point, pen_down: int, stroke_end: int, char_end: int, word_end: int, seq_end: int) -> None:
        events.append({
            "dx": float(cur[0] - prev[0]),
            "dy": float(cur[1] - prev[1]),
            "pen_down": int(pen_down),
            "stroke_end": int(stroke_end),
            "char_end": int(char_end),
            "word_end": int(word_end),
            "seq_end": int(seq_end),
        })

    prev_point: Optional[Point] = None

    for t, (idx, direction) in enumerate(route):
        s = strokes[idx]
        pts = resample_polyline(s.points(direction), step=stroke_step)
        if not pts:
            continue

        if prev_point is None:
            prev_point = pts[0]
            events.append({
                "dx": 0.0,
                "dy": 0.0,
                "pen_down": 0,
                "stroke_end": 0,
                "char_end": 0,
                "word_end": 0,
                "seq_end": 0,
            })
        else:
            bridge_pts = resample_polyline([prev_point, pts[0]], step=bridge_step)
            for bp in bridge_pts[1:]:
                emit_delta(prev_point, bp, pen_down=0, stroke_end=0, char_end=0, word_end=0, seq_end=0)
                prev_point = bp

        for p in pts[1:]:
            next_stroke_end = 0
            next_char_end = 0
            next_word_end = 0
            next_seq_end = 0

            is_last_point_of_stroke = (p == pts[-1])
            if is_last_point_of_stroke:
                next_stroke_end = 1
                if t == len(route) - 1:
                    next_word_end = 1
                    next_seq_end = 1
                else:
                    cur_char = strokes[route[t][0]].char_id
                    next_char = strokes[route[t + 1][0]].char_id
                    if cur_char != next_char:
                        next_char_end = 1

            emit_delta(
                prev_point,
                p,
                pen_down=1,
                stroke_end=next_stroke_end,
                char_end=next_char_end,
                word_end=next_word_end,
                seq_end=next_seq_end,
            )
            prev_point = p

    return events


def route_start_end_points(strokes: Sequence[PrimitiveStroke], route: List[Tuple[int, int]]) -> Tuple[Optional[Point], Optional[Point]]:
    if not route:
        return None, None
    i0, d0 = route[0]
    i1, d1 = route[-1]
    return strokes[i0].start(d0), strokes[i1].end(d1)


def convert_char_record_to_strokes(rec: Dict[str, Any]) -> List[PrimitiveStroke]:
    out: List[PrimitiveStroke] = []
    for s in rec["primitive_strokes"]:
        out.append(
            PrimitiveStroke(
                stroke_id=s["stroke_id"],
                char_id=rec["char_id"],
                char_index=int(rec["char_index"]),
                points_fwd=[(float(x), float(y)) for x, y in s["points_fwd"]],
                points_rev=[(float(x), float(y)) for x, y in s["points_rev"]],
                arc_len=float(s["arc_len"]),
                curv_cost_fwd=float(s.get("curv_cost_fwd", 0.0)),
                curv_cost_rev=float(s.get("curv_cost_rev", 0.0)),
            )
        )
    return out


def group_character_strokes_by_word(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_word: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_word[rec["word_id"]].append(rec)
    for word_id in by_word:
        by_word[word_id].sort(key=lambda r: (int(r["char_index"]), r["char_id"]))
    return by_word


def index_word_metadata(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        out[rec["word_id"]] = rec
    return out


def build_word_route_record(
    word_meta: Dict[str, Any],
    char_recs: Sequence[Dict[str, Any]],
    weights: WordRouteWeights,
    beam_size: int,
    do_local_refine: bool,
    stroke_step: float,
    bridge_step: float,
) -> Dict[str, Any]:
    char_recs = sorted(char_recs, key=lambda r: (int(r["char_index"]), r["char_id"]))
    strokes: List[PrimitiveStroke] = []
    for rec in char_recs:
        strokes.extend(convert_char_record_to_strokes(rec))

    plan_cost, route = build_word_route_beam(strokes=strokes, weights=weights, beam_size=beam_size)

    if do_local_refine and route:
        plan_cost, route = greedy_local_refine_word_route(strokes=strokes, route=route, weights=weights)

    events = route_to_events(strokes=strokes, route=route, stroke_step=stroke_step, bridge_step=bridge_step)
    start_point, end_point = route_start_end_points(strokes=strokes, route=route)

    selected_sequence = []
    for idx, direction in route:
        selected_sequence.append({
            "stroke_id": strokes[idx].stroke_id,
            "orientation": "fwd" if direction == 0 else "rev",
        })

    primitive_strokes_out = []
    for s in strokes:
        primitive_strokes_out.append({
            "stroke_id": s.stroke_id,
            "char_id": s.char_id,
            "char_index": s.char_index,
            "points_fwd": [[float(x), float(y)] for x, y in s.points_fwd],
            "points_rev": [[float(x), float(y)] for x, y in s.points_rev],
            "arc_len": float(s.arc_len),
        })

    char_ids = [rec["char_id"] for rec in char_recs]
    confs = [float(rec.get("confidence", 1.0)) for rec in char_recs]
    confidence = sum(confs) / max(1, len(confs))

    return {
        "record_id": word_meta.get("record_id", f"word_route_{word_meta['word_id']}"),
        "document_id": word_meta["document_id"],
        "line_id": word_meta["line_id"],
        "word_id": word_meta["word_id"],
        "word_index": int(word_meta["word_index"]),
        "word_text": word_meta["word_text"],
        "character_ids": char_ids,
        "state_before_ref": word_meta["state_before_ref"],
        "state_after_ref": word_meta["state_after_ref"],
        "target_bbox_parent_norm_cxcywh": word_meta["target_bbox_parent_norm_cxcywh"],
        "primitive_strokes": primitive_strokes_out,
        "selected_sequence": selected_sequence,
        "events": events,
        "plan_cost": float(plan_cost),
        "route_start_point": None if start_point is None else [float(start_point[0]), float(start_point[1])],
        "route_end_point": None if end_point is None else [float(end_point[0]), float(end_point[1])],
        "source_type": word_meta.get("source_type", "pseudo_online"),
        "source_weight": float(word_meta.get("source_weight", confidence)),
        "confidence": float(word_meta.get("confidence", confidence)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build word-level stroke routes from character primitive strokes.")
    p.add_argument("--character-strokes-jsonl", required=True)
    p.add_argument("--word-metadata-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)

    p.add_argument("--beam-size", type=int, default=64)
    p.add_argument("--no-local-refine", action="store_true")
    p.add_argument("--stroke-step", type=float, default=0.01)
    p.add_argument("--bridge-step", type=float, default=0.01)

    p.add_argument("--lambda-down", type=float, default=1.0)
    p.add_argument("--lambda-up", type=float, default=1.0)
    p.add_argument("--lambda-curv", type=float, default=0.05)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--lambda-switch", type=float, default=0.1)
    p.add_argument("--lambda-back", type=float, default=0.1)
    p.add_argument("--lambda-start", type=float, default=0.0)
    p.add_argument("--lambda-end", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    char_recs = read_jsonl(args.character_strokes_jsonl)
    word_meta_recs = read_jsonl(args.word_metadata_jsonl)

    by_word = group_character_strokes_by_word(char_recs)
    word_meta = index_word_metadata(word_meta_recs)

    weights = WordRouteWeights(
        lambda_down=args.lambda_down,
        lambda_up=args.lambda_up,
        lambda_curv=args.lambda_curv,
        lambda_smooth=args.lambda_smooth,
        lambda_switch=args.lambda_switch,
        lambda_back=args.lambda_back,
        lambda_start=args.lambda_start,
        lambda_end=args.lambda_end,
    )

    out_recs: List[Dict[str, Any]] = []
    missing_words = []

    for word_id, chars in sorted(by_word.items()):
        meta = word_meta.get(word_id)
        if meta is None:
            missing_words.append(word_id)
            continue

        rec = build_word_route_record(
            word_meta=meta,
            char_recs=chars,
            weights=weights,
            beam_size=args.beam_size,
            do_local_refine=not args.no_local_refine,
            stroke_step=args.stroke_step,
            bridge_step=args.bridge_step,
        )
        out_recs.append(rec)

    write_jsonl(args.output_jsonl, out_recs)

    print(f"Wrote {len(out_recs)} word routes to {args.output_jsonl}")
    if missing_words:
        print(f"Skipped {len(missing_words)} words missing metadata.")
        print("First 20 missing word_ids:", missing_words[:20])


if __name__ == "__main__":
    main()
Expected word_metadata.jsonl

This script expects one record per word that looks like:

JSON
{
  "record_id": "word_route_word_9",
  "document_id": "iam_000123",
  "line_id": "line_17",
  "word_id": "word_9",
  "word_index": 3,
  "word_text": "the",
  "state_before_ref": "states/line_17/word_03_before.png",
  "state_after_ref": "states/line_17/word_03_after.png",
  "target_bbox_parent_norm_cxcywh": [0.42, 0.51, 0.18, 0.22],
  "source_type": "pseudo_online",
  "source_weight": 0.74,
  "confidence": 0.81
}
Example usage
Bash
python build_word_routes.py \
  --character-strokes-jsonl data/character_strokes.jsonl \
  --word-metadata-jsonl data/word_metadata.jsonl \
  --output-jsonl data/word_routes.jsonl \
  --beam-size 64
build_line_bridges.py
Python
Run
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


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


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def resample_polyline(points: Sequence[Point], step: float) -> List[Point]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points[0]]

    out = [points[0]]
    carry = 0.0
    prev = points[0]

    for cur in points[1:]:
        seg_len = euclidean(prev, cur)
        if seg_len == 0:
            prev = cur
            continue

        dir_x = (cur[0] - prev[0]) / seg_len
        dir_y = (cur[1] - prev[1]) / seg_len
        remaining = seg_len
        cur_pt = prev

        while carry + remaining >= step:
            needed = step - carry
            new_pt = (cur_pt[0] + dir_x * needed, cur_pt[1] + dir_y * needed)
            out.append(new_pt)
            cur_pt = new_pt
            remaining -= needed
            carry = 0.0

        carry += remaining
        prev = cur

    if out[-1] != points[-1]:
        out.append(points[-1])

    return out


def points_to_penup_events(points: Sequence[Point]) -> List[Dict[str, Any]]:
    if not points:
        return []

    events: List[Dict[str, Any]] = []
    prev = points[0]
    for cur in points[1:]:
        events.append({
            "dx": float(cur[0] - prev[0]),
            "dy": float(cur[1] - prev[1]),
            "pen_down": 0,
            "stroke_end": 0,
            "char_end": 0,
            "word_end": 0,
            "seq_end": 0,
        })
        prev = cur
    return events


def group_by_line(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        out[rec["line_id"]].append(rec)
    for line_id in out:
        out[line_id].sort(key=lambda r: (int(r["word_index"]), r["word_id"]))
    return out


def build_bridge_record(
    prev_word: Dict[str, Any],
    next_word: Dict[str, Any],
    bridge_step: float,
) -> Dict[str, Any]:
    prev_end = prev_word.get("route_end_point")
    next_start = next_word.get("route_start_point")

    if prev_end is None or next_start is None:
        bridge_points = []
        bridge_events = []
        bridge_cost = 0.0
    else:
        p0 = (float(prev_end[0]), float(prev_end[1]))
        p1 = (float(next_start[0]), float(next_start[1]))
        bridge_points = resample_polyline([p0, p1], step=bridge_step)
        bridge_events = points_to_penup_events(bridge_points)
        bridge_cost = euclidean(p0, p1)

    conf = min(float(prev_word.get("confidence", 1.0)), float(next_word.get("confidence", 1.0)))

    return {
        "record_id": f"line_bridge_{prev_word['word_id']}__{next_word['word_id']}",
        "document_id": prev_word["document_id"],
        "line_id": prev_word["line_id"],
        "prev_word_id": prev_word["word_id"],
        "next_word_id": next_word["word_id"],
        "prev_word_text": prev_word["word_text"],
        "next_word_text": next_word["word_text"],
        "prev_word_index": int(prev_word["word_index"]),
        "next_word_index": int(next_word["word_index"]),
        "prev_word_end_point": prev_end,
        "next_word_start_point": next_start,
        "bridge_points": bridge_points,
        "bridge_events": bridge_events,
        "bridge_cost": float(bridge_cost),
        "confidence": float(conf),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed-order line bridges between adjacent routed words.")
    p.add_argument("--word-routes-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--bridge-step", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    word_routes = read_jsonl(args.word_routes_jsonl)
    by_line = group_by_line(word_routes)

    out_recs: List[Dict[str, Any]] = []

    for line_id, words in sorted(by_line.items()):
        if len(words) < 2:
            continue
        for i in range(len(words) - 1):
            rec = build_bridge_record(
                prev_word=words[i],
                next_word=words[i + 1],
                bridge_step=args.bridge_step,
            )
            out_recs.append(rec)

    write_jsonl(args.output_jsonl, out_recs)
    print(f"Wrote {len(out_recs)} line bridges to {args.output_jsonl}")


if __name__ == "__main__":
    main()
Example usage
Bash
python build_line_bridges.py \
  --word-routes-jsonl data/word_routes.jsonl \
  --output-jsonl data/line_bridges.jsonl
Notes that matter

A couple of design choices here are the most important:

build_word_routes.py optimizes across the entire word stroke inventory, not per character.

build_line_bridges.py keeps word order fixed and only builds pen-up transitions between adjacent words.

The route_start_point and route_end_point fields emitted by build_word_routes.py are what make the line bridge builder simple.

The beam-search bitmask implementation caps cleanly at 62 strokes per word. That is usually fine for English words, but for pathological cases you’d want a set-based state or chunked planner.

The sharpest next step is a training script for the online word scribe that consumes WordRouteDataset and predicts the event sequence.

---

## User

sure

---

## ChatGPT

I’m giving you a runnable first-pass trainer rather than just pseudocode: model, masked losses, teacher forcing, and checkpointing, wired to the WordRouteDataset/collate_word_routes interface from above.

---

## ChatGPT

Here’s a solid first-pass train_online_scribe.py.

It trains a word-level online scribe on WordRouteDataset records. The model:

encodes the local state_before raster,

encodes the target word text,

autoregressively predicts the next pen event with teacher forcing.

It predicts:

dx, dy

pen_down

stroke_end

char_end

word_end

seq_end

and uses:

Huber loss for dx, dy

BCE-with-logits for the binary flags

masking over padded sequence positions

per-example weighting with source_weight * confidence

train_online_scribe.py
Python
Run
#!/usr/bin/env python3
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
    """
    seqs: list of [T_i, 7]
    returns:
      padded: [B, T_max, 7]
      mask:   [B, T_max] (True for valid positions)
    """
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
# Dataset
# ============================================================

class WordRouteDataset(Dataset):
    """
    Main dataset for the online English word scribe.

    Expected JSONL fields:
      - record_id
      - document_id
      - line_id
      - word_id
      - word_index
      - word_text
      - state_before_ref
      - state_after_ref   (optional for this trainer)
      - target_bbox_parent_norm_cxcywh
      - events
      - plan_cost
      - source_type
      - source_weight
      - confidence
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path = ".",
        max_text_len: int = 64,
        load_state_after: bool = False,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.max_text_len = max_text_len
        self.load_state_after = load_state_after

    def __len__(self) -> int:
        return len(self.records)

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

        state_before = load_gray_image(self.image_root / rec["state_before_ref"])
        state_after = None
        if self.load_state_after and rec.get("state_after_ref") is not None:
            state_after = load_gray_image(self.image_root / rec["state_after_ref"])

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
        "events": events,
        "event_mask": event_mask,
        "plan_cost": torch.stack([b["plan_cost"] for b in batch], dim=0),
        "source_weight": torch.stack([b["source_weight"] for b in batch], dim=0),
        "confidence": torch.stack([b["confidence"] for b in batch], dim=0),
        "source_type": [b["source_type"] for b in batch],
    }

    if batch[0]["state_after"] is not None:
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
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        t = x.shape[1]
        return x + self.pe[:, :t]


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
        # x: [B,1,H,W]
        feat = self.net(x)  # [B,D,H',W']
        b, d, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # [B, H'W', D]
        return feat


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
        # tokens: [B,T]
        x = self.embed(tokens)
        x = self.pos(x)

        key_padding_mask = None
        if tokens.numel() > 0:
            max_len = tokens.shape[1]
            device = tokens.device
            ar = torch.arange(max_len, device=device).unsqueeze(0)
            key_padding_mask = ar >= lengths.unsqueeze(1)  # [B,T], True=pad

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x  # [B,T,D]


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
        self.pos = SinusoidalPositionalEncoding(d_model)

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
        self.flag_head = nn.Linear(d_model, 5)  # pen_down, stroke_end, char_end, word_end, seq_end

    def causal_mask(self, t: int, device: torch.device) -> Tensor:
        # True entries are masked
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        prev_events: Tensor,     # [B,T,7]
        bbox: Tensor,            # [B,4]
        memory: Tensor,          # [B,S,D]
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        b, t, _ = prev_events.shape

        x = self.event_in(prev_events) + self.bbox_in(bbox).unsqueeze(1)
        x = self.pos(x)

        tgt_mask = self.causal_mask(t, prev_events.device)
        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        y = self.norm(y)

        dxdy = self.dxdy_head(y)        # [B,T,2]
        flags = self.flag_head(y)       # [B,T,5]
        return {"dxdy": dxdy, "flags": flags}


class OnlineWordScribe(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        text_layers: int = 2,
        dec_layers: int = 6,
    ) -> None:
        super().__init__()
        self.state_encoder = SmallCNNStateEncoder(in_ch=1, d_model=d_model)
        self.text_encoder = TextEncoder(vocab_size=256, d_model=d_model, nhead=nhead, num_layers=text_layers)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.decoder = EventDecoder(d_model=d_model, nhead=nhead, num_layers=dec_layers, event_dim=7)

    def forward(
        self,
        state_before: Tensor,        # [B,1,H,W]
        text_tokens: Tensor,         # [B,T_text]
        text_lengths: Tensor,        # [B]
        bbox: Tensor,                # [B,4]
        prev_events: Tensor,         # [B,T,7]
    ) -> Dict[str, Tensor]:
        state_mem = self.state_encoder(state_before)  # [B,S1,D]
        text_mem = self.text_encoder(text_tokens, text_lengths)  # [B,S2,D]

        memory = torch.cat([state_mem, text_mem], dim=1)
        memory = self.memory_proj(memory)

        return self.decoder(
            prev_events=prev_events,
            bbox=bbox,
            memory=memory,
            memory_key_padding_mask=None,
        )


# ============================================================
# Loss
# ============================================================

@dataclass
class LossBreakdown:
    total: Tensor
    dxdy: Tensor
    flags: Tensor
    pen_down: Tensor
    stroke_end: Tensor
    char_end: Tensor
    word_end: Tensor
    seq_end: Tensor


def build_teacher_forced_inputs(events: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Teacher forcing:
      prev_events[:, 0] = BOS = zeros
      prev_events[:, t] = gold event at t-1
      targets = gold events

    events: [B,T,7]
    returns:
      prev_events: [B,T,7]
      targets:     [B,T,7]
    """
    b, t, f = events.shape
    prev_events = torch.zeros_like(events)
    if t > 1:
        prev_events[:, 1:, :] = events[:, :-1, :]
    return prev_events, events


def sequence_loss(
    pred: Dict[str, Tensor],
    target_events: Tensor,       # [B,T,7]
    event_mask: Tensor,          # [B,T] bool
    example_weights: Tensor,     # [B]
    lambda_dxdy: float = 1.0,
    lambda_flags: float = 1.0,
) -> LossBreakdown:
    """
    target columns:
      0: dx
      1: dy
      2: pen_down
      3: stroke_end
      4: char_end
      5: word_end
      6: seq_end
    """
    dxdy_pred = pred["dxdy"]          # [B,T,2]
    flags_pred = pred["flags"]        # [B,T,5]

    dxdy_tgt = target_events[:, :, 0:2]
    flags_tgt = target_events[:, :, 2:7]

    mask = event_mask.float()  # [B,T]
    ex_w = example_weights.unsqueeze(1)  # [B,1]
    wt = mask * ex_w

    # Huber for dx/dy
    huber = F.smooth_l1_loss(dxdy_pred, dxdy_tgt, reduction="none").sum(dim=-1)  # [B,T]
    dxdy_loss = (huber * wt).sum() / wt.sum().clamp_min(1.0)

    # BCE per flag
    bce = F.binary_cross_entropy_with_logits(flags_pred, flags_tgt, reduction="none")  # [B,T,5]
    per_flag = []
    for i in range(5):
        li = (bce[:, :, i] * wt).sum() / wt.sum().clamp_min(1.0)
        per_flag.append(li)

    flags_loss = sum(per_flag) / 5.0
    total = lambda_dxdy * dxdy_loss + lambda_flags * flags_loss

    return LossBreakdown(
        total=total,
        dxdy=dxdy_loss,
        flags=flags_loss,
        pen_down=per_flag[0],
        stroke_end=per_flag[1],
        char_end=per_flag[2],
        word_end=per_flag[3],
        seq_end=per_flag[4],
    )


# ============================================================
# Training / Eval
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
    model: OnlineWordScribe,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    lambda_dxdy: float,
    lambda_flags: float,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    sums = {
        "total": 0.0,
        "dxdy": 0.0,
        "flags": 0.0,
        "pen_down": 0.0,
        "stroke_end": 0.0,
        "char_end": 0.0,
        "word_end": 0.0,
        "seq_end": 0.0,
        "count": 0,
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        prev_events, target_events = build_teacher_forced_inputs(batch["events"])

        pred = model(
            state_before=batch["state_before"],
            text_tokens=batch["text_tokens"],
            text_lengths=batch["text_lengths"],
            bbox=batch["bbox"],
            prev_events=prev_events,
        )

        example_weights = batch["source_weight"] * batch["confidence"]
        loss = sequence_loss(
            pred=pred,
            target_events=target_events,
            event_mask=batch["event_mask"],
            example_weights=example_weights,
            lambda_dxdy=lambda_dxdy,
            lambda_flags=lambda_flags,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch["events"].shape[0]
        sums["total"] += float(loss.total.detach().cpu()) * bs
        sums["dxdy"] += float(loss.dxdy.detach().cpu()) * bs
        sums["flags"] += float(loss.flags.detach().cpu()) * bs
        sums["pen_down"] += float(loss.pen_down.detach().cpu()) * bs
        sums["stroke_end"] += float(loss.stroke_end.detach().cpu()) * bs
        sums["char_end"] += float(loss.char_end.detach().cpu()) * bs
        sums["word_end"] += float(loss.word_end.detach().cpu()) * bs
        sums["seq_end"] += float(loss.seq_end.detach().cpu()) * bs
        sums["count"] += bs

    count = max(1, sums["count"])
    return {k: (v / count if k != "count" else v) for k, v in sums.items()}


@torch.no_grad()
def greedy_decode(
    model: OnlineWordScribe,
    state_before: Tensor,     # [1,1,H,W]
    text_tokens: Tensor,      # [1,T]
    text_lengths: Tensor,     # [1]
    bbox: Tensor,             # [1,4]
    max_steps: int = 512,
    stop_threshold: float = 0.5,
) -> Tensor:
    model.eval()
    device = state_before.device

    events = torch.zeros(1, 1, 7, device=device)  # BOS only
    generated: List[Tensor] = []

    for _ in range(max_steps):
        pred = model(
            state_before=state_before,
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            bbox=bbox,
            prev_events=events,
        )
        dxdy = pred["dxdy"][:, -1, :]         # [1,2]
        flags = torch.sigmoid(pred["flags"][:, -1, :])  # [1,5]

        next_evt = torch.zeros(1, 1, 7, device=device)
        next_evt[:, 0, 0:2] = dxdy
        next_evt[:, 0, 2:7] = (flags > stop_threshold).float()

        generated.append(next_evt[:, 0, :].detach().cpu())
        events = torch.cat([events, next_evt], dim=1)

        seq_end = bool(next_evt[0, 0, 6].item() > 0.5)
        if seq_end:
            break

    if not generated:
        return torch.zeros(0, 7)
    return torch.stack(generated, dim=1).squeeze(0)  # [T,7]


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: OnlineWordScribe,
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
        best_path = out_dir / "best.pt"
        torch.save(ckpt, best_path)

    return ckpt_path


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an online word-level handwriting scribe on WordRouteDataset.")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--text-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=6)

    p.add_argument("--lambda-dxdy", type=float, default=1.0)
    p.add_argument("--lambda-flags", type=float, default=1.0)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-decode-steps", type=int, default=512)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = WordRouteDataset(
        jsonl_path=args.train_jsonl,
        image_root=args.image_root,
        load_state_after=False,
    )
    val_ds = WordRouteDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        load_state_after=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_word_routes,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_word_routes,
    )

    model = OnlineWordScribe(
        d_model=args.d_model,
        nhead=args.nhead,
        text_layers=args.text_layers,
        dec_layers=args.dec_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            lambda_dxdy=args.lambda_dxdy,
            lambda_flags=args.lambda_flags,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
            lambda_dxdy=args.lambda_dxdy,
            lambda_flags=args.lambda_flags,
        )

        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]

        ckpt_path = save_checkpoint(
            out_dir=out_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
            is_best=is_best,
        )

        # quick qualitative decode on first val example
        sample = val_ds[0]
        sample_state = sample["state_before"].unsqueeze(0).to(device)
        sample_text = sample["text_tokens"].unsqueeze(0).to(device)
        sample_text_len = torch.tensor([sample["text_tokens"].numel()], device=device)
        sample_bbox = sample["bbox"].unsqueeze(0).to(device)

        pred_events = greedy_decode(
            model=model,
            state_before=sample_state,
            text_tokens=sample_text,
            text_lengths=sample_text_len,
            bbox=sample_bbox,
            max_steps=args.max_decode_steps,
        )

        sample_pred_path = out_dir / f"sample_decode_epoch_{epoch:03d}.json"
        with sample_pred_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "record_id": sample["record_id"],
                    "word_text": sample["word_text"],
                    "predicted_events": pred_events.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "checkpoint": str(ckpt_path),
            "sample_decode": str(sample_pred_path),
        }
        history.append(row)

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"[epoch {epoch:03d}] "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"train_dxdy={train_metrics['dxdy']:.4f} "
            f"val_dxdy={val_metrics['dxdy']:.4f}"
        )

    print(f"Done. Best val total loss: {best_val:.6f}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
How to run it
Bash
python train_online_scribe.py \
  --train-jsonl data/train_word_routes.jsonl \
  --val-jsonl data/val_word_routes.jsonl \
  --image-root . \
  --out-dir runs/online_word_scribe_v1 \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4
What this script is doing

The important training choice is:

teacher forcing on the gold event stream:

input at time t is the gold event at t-1

target at time t is the gold event at t

So if your route events are good, this gives you a strong first supervised model.

The context memory is the concatenation of:

CNN features from the local word-state raster

transformer features from the word text bytes

Then the decoder autoregressively predicts the event sequence.

What I would change next

For the next iteration, I’d upgrade three things:

Add style conditioning
by adding style_id / writer embedding to WordRouteDataset and injecting it into the decoder.

Add raster consistency
by differentiably rasterizing predicted pen events and comparing to state_after - state_before.

Scheduled sampling
after the teacher-forced model stabilizes.

The next most useful code artifact is a differentiable rasterizer module for point sequences, so the online scribe can get image-space supervision instead of only event-space supervision.

---

## User

Wait explain each and then implement.

---

## ChatGPT

I’m going to unpack the three upgrades in plain terms first, then give you a single revised trainer that includes all three so you can see how they fit together.

---

## ChatGPT

Yep — the three upgrades are:

1) Style conditioning

What it fixes:
Without style conditioning, the model only knows the target text and current page state. It does not know who is writing, so it tends to average across writers and loses things like:

slant

stroke thickness tendencies

spacing habits

loop size

ligature tendencies

What we do:
Give every example a style_id or writer_id, turn that into a learned embedding, and feed it into the model.

Where it enters the model:
Two places:

as a memory token alongside the state/image features and text features

as an additive conditioning vector inside the decoder

That makes it influence both the global plan and each emitted pen event.

2) Raster consistency

What it fixes:
Event-space loss alone says “match the sequence numerically,” but many event sequences can have similar local deltas while still drawing ugly or misplaced ink.

You want the model to care about what the trajectory looks like on the page, not just the raw dx,dy numbers.

What we do:
Take the predicted event stream, convert it into a soft image with a differentiable rasterizer, and compare that rendered ink to the target ink:

target_delta
=
max
⁡
(
0
,
state_after
−
state_before
)
target_delta=max(0,state_after−state_before)

So the model is trained both to:

emit the right pen events

and produce the right visible word image

Important requirement:
To rasterize correctly, you need an absolute anchor for the trajectory. That is why the dataset should include:

route_start_point

If it is missing, the script below simply skips raster loss for that example.

Implementation choice here:
I use a soft point-splat rasterizer on a downsampled grid. It is cheap and differentiable. It is not the fanciest line renderer, but it is the right first version.

3) Scheduled sampling

What it fixes:
Teacher forcing trains the model on gold previous events. At inference time, those gold events are gone, so errors compound. That is the classic exposure-bias problem.

What we do:
During training, after a warmup period, we sometimes feed the model’s own previous prediction back in instead of the gold previous event.

Early in training:

almost pure teacher forcing

Later:

a mixture of gold previous events and model-generated previous events

That makes inference-time behavior much more stable.

Implementation choice here:
I use a simple linear schedule:

teacher probability starts at 1.0

after --ss-start-epoch, it decays toward --ss-final-teacher-prob

This is slower than pure teacher forcing because it rolls out autoregressively during training, but it is the cleanest first implementation.

Drop-in revised trainer

This version adds all three:

style conditioning

raster consistency loss

scheduled sampling

It expects your WordRouteDataset JSONL to have, ideally:

style_id or writer_id

state_before_ref

state_after_ref

route_start_point

If route_start_point or state_after_ref is missing, raster loss is skipped for that example.

Python
Run
#!/usr/bin/env python3
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
    """
    seqs: list of [T_i, 7]
    returns:
      padded: [B, T_max, 7]
      mask:   [B, T_max] bool
    """
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
    """
    style_id preferred, then writer_id, else __unk__.
    Reserve 0 for __unk__.
    """
    ids = {"__unk__"}
    for records in record_lists:
        for rec in records:
            sid = rec.get("style_id", rec.get("writer_id", "__unk__"))
            if sid is None:
                sid = "__unk__"
            ids.add(str(sid))
    ids = sorted(ids)
    return {sid: i for i, sid in enumerate(ids)}


# ============================================================
# Dataset
# ============================================================

class WordRouteDataset(Dataset):
    """
    Word-level routed stroke sequences.

    Preferred JSONL fields:
      - record_id
      - document_id
      - line_id
      - word_id
      - word_index
      - word_text
      - state_before_ref
      - state_after_ref
      - target_bbox_parent_norm_cxcywh
      - route_start_point
      - style_id or writer_id
      - events
      - plan_cost
      - source_type
      - source_weight
      - confidence
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        style_to_index: Dict[str, int],
        max_text_len: int = 64,
        require_state_after: bool = False,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index
        self.max_text_len = max_text_len
        self.require_state_after = require_state_after

    def __len__(self) -> int:
        return len(self.records)

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

        state_before = load_gray_image(self.image_root / rec["state_before_ref"])

        state_after = None
        has_state_after = False
        if rec.get("state_after_ref") is not None:
            state_after = load_gray_image(self.image_root / rec["state_after_ref"])
            has_state_after = True
        elif self.require_state_after:
            raise ValueError(f"Missing state_after_ref for record {rec.get('record_id')}")

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
        feat = self.net(x)                  # [B,D,H',W']
        feat = feat.flatten(2).transpose(1, 2)  # [B,S,D]
        return feat


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
        self.flag_head = nn.Linear(d_model, 5)

    def causal_mask(self, t: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        prev_events: Tensor,   # [B,T,7]
        bbox: Tensor,          # [B,4]
        style_vec: Tensor,     # [B,D]
        memory: Tensor,        # [B,S,D]
    ) -> Dict[str, Tensor]:
        b, t, _ = prev_events.shape

        x = self.event_in(prev_events)
        x = x + self.bbox_in(bbox).unsqueeze(1)
        x = x + self.style_in(style_vec).unsqueeze(1)
        x = self.pos(x)

        tgt_mask = self.causal_mask(t, prev_events.device)
        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        y = self.norm(y)

        return {
            "dxdy": self.dxdy_head(y),
            "flags": self.flag_head(y),
        }


class OnlineWordScribe(nn.Module):
    def __init__(
        self,
        num_styles: int,
        d_model: int = 256,
        nhead: int = 8,
        text_layers: int = 2,
        dec_layers: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_encoder = SmallCNNStateEncoder(in_ch=1, d_model=d_model)
        self.text_encoder = TextEncoder(vocab_size=256, d_model=d_model, nhead=nhead, num_layers=text_layers)
        self.style_embed = nn.Embedding(num_styles, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.decoder = EventDecoder(d_model=d_model, nhead=nhead, num_layers=dec_layers, event_dim=7)

    def encode_context(
        self,
        state_before: Tensor,
        text_tokens: Tensor,
        text_lengths: Tensor,
        style_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        state_mem = self.state_encoder(state_before)                # [B,S1,D]
        text_mem = self.text_encoder(text_tokens, text_lengths)     # [B,S2,D]
        style_vec = self.style_embed(style_index)                   # [B,D]
        style_token = style_vec.unsqueeze(1)                        # [B,1,D]

        memory = torch.cat([style_token, state_mem, text_mem], dim=1)
        memory = self.memory_proj(memory)
        return memory, style_vec

    def decode(
        self,
        prev_events: Tensor,
        bbox: Tensor,
        style_vec: Tensor,
        memory: Tensor,
    ) -> Dict[str, Tensor]:
        return self.decoder(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
        )


# ============================================================
# Differentiable rasterizer
# ============================================================

def soft_rasterize_points(
    start_points: Tensor,   # [B,2], normalized [0,1]
    dxdy: Tensor,           # [B,T,2]
    pen_probs: Tensor,      # [B,T], in [0,1]
    raster_size: int = 64,
    sigma_px: float = 1.5,
) -> Tensor:
    """
    Point-splat rasterizer.
    Returns [B,1,R,R].
    """
    b, t, _ = dxdy.shape
    device = dxdy.device

    positions = start_points.unsqueeze(1) + torch.cumsum(dxdy, dim=1)  # [B,T,2]
    x = positions[:, :, 0].clamp(0, 1) * (raster_size - 1)
    y = positions[:, :, 1].clamp(0, 1) * (raster_size - 1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(raster_size, device=device, dtype=torch.float32),
        torch.arange(raster_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.view(1, 1, raster_size, raster_size)
    grid_y = grid_y.view(1, 1, raster_size, raster_size)

    x = x.unsqueeze(-1).unsqueeze(-1)   # [B,T,1,1]
    y = y.unsqueeze(-1).unsqueeze(-1)   # [B,T,1,1]
    w = pen_probs.unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1]

    dist2 = (grid_x - x) ** 2 + (grid_y - y) ** 2
    gauss = torch.exp(-dist2 / (2 * sigma_px * sigma_px))
    contrib = w * gauss                      # [B,T,R,R]
    raster = torch.amax(contrib, dim=1, keepdim=True)  # [B,1,R,R]
    return raster.clamp(0, 1)


# ============================================================
# Loss
# ============================================================

@dataclass
class LossBreakdown:
    total: Tensor
    dxdy: Tensor
    flags: Tensor
    raster: Tensor
    pen_down: Tensor
    stroke_end: Tensor
    char_end: Tensor
    word_end: Tensor
    seq_end: Tensor


def compute_sequence_loss(
    pred_dxdy: Tensor,          # [B,T,2]
    pred_flags_logits: Tensor,  # [B,T,5]
    target_events: Tensor,      # [B,T,7]
    event_mask: Tensor,         # [B,T] bool
    example_weights: Tensor,    # [B]
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    dxdy_tgt = target_events[:, :, 0:2]
    flags_tgt = target_events[:, :, 2:7]

    mask = event_mask.float()
    wt = mask * example_weights.unsqueeze(1)

    huber = F.smooth_l1_loss(pred_dxdy, dxdy_tgt, reduction="none").sum(dim=-1)
    dxdy_loss = (huber * wt).sum() / wt.sum().clamp_min(1.0)

    bce = F.binary_cross_entropy_with_logits(pred_flags_logits, flags_tgt, reduction="none")
    per_flag = []
    for i in range(5):
        li = (bce[:, :, i] * wt).sum() / wt.sum().clamp_min(1.0)
        per_flag.append(li)

    flags_loss = sum(per_flag) / 5.0
    return dxdy_loss, flags_loss, per_flag


def compute_raster_loss(
    pred_dxdy: Tensor,              # [B,T,2]
    pred_flags_logits: Tensor,      # [B,T,5]
    state_before: Tensor,           # [B,1,H,W]
    state_after: Optional[Tensor],  # [B,1,H,W] or None
    route_start_point: Tensor,      # [B,2]
    raster_valid_mask: Tensor,      # [B] float
    example_weights: Tensor,        # [B]
    raster_size: int,
    sigma_px: float,
) -> Tensor:
    if state_after is None:
        return pred_dxdy.new_tensor(0.0)

    valid = raster_valid_mask > 0.5
    if valid.sum() == 0:
        return pred_dxdy.new_tensor(0.0)

    pred_dxdy_v = pred_dxdy[valid]
    pen_probs_v = torch.sigmoid(pred_flags_logits[valid, :, 0])
    start_v = route_start_point[valid]
    before_v = state_before[valid]
    after_v = state_after[valid]
    ex_w = example_weights[valid]

    pred_raster = soft_rasterize_points(
        start_points=start_v,
        dxdy=pred_dxdy_v,
        pen_probs=pen_probs_v,
        raster_size=raster_size,
        sigma_px=sigma_px,
    )

    target_delta = torch.relu(after_v - before_v)
    target_delta = F.interpolate(target_delta, size=(raster_size, raster_size), mode="bilinear", align_corners=False)
    target_delta = target_delta.clamp(0, 1)

    per_example = torch.abs(pred_raster - target_delta).mean(dim=(1, 2, 3))
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
    target_events: Tensor,      # [B,T,7]
    teacher_prob: float,
    sample_model_inputs: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      pred_dxdy: [B,T,2]
      pred_flags_logits: [B,T,5]

    If sample_model_inputs=True, scheduled sampling is active.
    Otherwise teacher forcing with probability teacher_prob is used,
    and when teacher_prob==1.0 this reduces to pure teacher forcing rollout.
    """
    b, t, _ = target_events.shape
    device = target_events.device

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    prev_events = torch.zeros(b, 1, 7, device=device)
    pred_dxdy_list: List[Tensor] = []
    pred_flags_list: List[Tensor] = []

    for step in range(t):
        out = model.decode(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
        )

        dxdy_step = out["dxdy"][:, -1, :]         # [B,2]
        flags_step = out["flags"][:, -1, :]       # [B,5]

        pred_dxdy_list.append(dxdy_step)
        pred_flags_list.append(flags_step)

        pred_event = torch.cat([dxdy_step, torch.sigmoid(flags_step)], dim=-1)  # [B,7]
        gold_event = target_events[:, step, :]                                   # [B,7]

        if sample_model_inputs:
            use_teacher = (torch.rand(b, device=device) < teacher_prob).float().unsqueeze(1)
            next_input = use_teacher * gold_event + (1.0 - use_teacher) * pred_event.detach()
        else:
            if teacher_prob >= 1.0:
                next_input = gold_event
            else:
                use_teacher = (torch.rand(b, device=device) < teacher_prob).float().unsqueeze(1)
                next_input = use_teacher * gold_event + (1.0 - use_teacher) * pred_event.detach()

        prev_events = torch.cat([prev_events, next_input.unsqueeze(1)], dim=1)

    pred_dxdy = torch.stack(pred_dxdy_list, dim=1)          # [B,T,2]
    pred_flags_logits = torch.stack(pred_flags_list, dim=1) # [B,T,5]
    return pred_dxdy, pred_flags_logits


@torch.no_grad()
def greedy_decode(
    model: OnlineWordScribe,
    state_before: Tensor,     # [1,1,H,W]
    text_tokens: Tensor,      # [1,T]
    text_lengths: Tensor,     # [1]
    bbox: Tensor,             # [1,4]
    style_index: Tensor,      # [1]
    max_steps: int = 512,
    stop_threshold: float = 0.5,
) -> Tensor:
    model.eval()
    device = state_before.device

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    prev_events = torch.zeros(1, 1, 7, device=device)
    generated: List[Tensor] = []

    for _ in range(max_steps):
        out = model.decode(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
        )

        dxdy = out["dxdy"][:, -1, :]
        flags_logits = out["flags"][:, -1, :]
        flags = torch.sigmoid(flags_logits)

        next_evt = torch.cat([dxdy, flags], dim=-1)  # [1,7]
        generated.append(next_evt[0].detach().cpu())

        prev_events = torch.cat([prev_events, next_evt.unsqueeze(1)], dim=1)

        if bool(flags[0, 4].item() > stop_threshold):  # seq_end
            break

    if not generated:
        return torch.zeros(0, 7)
    return torch.stack(generated, dim=0)  # [T,7]


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

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred_dxdy, pred_flags_logits = rollout_autoregressive(
            model=model,
            state_before=batch["state_before"],
            text_tokens=batch["text_tokens"],
            text_lengths=batch["text_lengths"],
            bbox=batch["bbox"],
            style_index=batch["style_index"],
            target_events=batch["events"],
            teacher_prob=teacher_prob,
            sample_model_inputs=scheduled_sampling_active and train,
        )

        example_weights = batch["source_weight"] * batch["confidence"]

        dxdy_loss, flags_loss, per_flag = compute_sequence_loss(
            pred_dxdy=pred_dxdy,
            pred_flags_logits=pred_flags_logits,
            target_events=batch["events"],
            event_mask=batch["event_mask"],
            example_weights=example_weights,
        )

        state_after = batch.get("state_after", None)
        raster_valid_mask = batch["route_start_valid"] * batch["has_state_after"]
        raster_loss = compute_raster_loss(
            pred_dxdy=pred_dxdy,
            pred_flags_logits=pred_flags_logits,
            state_before=batch["state_before"],
            state_after=state_after,
            route_start_point=batch["route_start_point"],
            raster_valid_mask=raster_valid_mask,
            example_weights=example_weights,
            raster_size=raster_size,
            sigma_px=raster_sigma_px,
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

        bs = batch["events"].shape[0]
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
    p = argparse.ArgumentParser(description="Train an online word-level handwriting scribe with style conditioning, raster consistency, and scheduled sampling.")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
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

    p.add_argument("--lambda-dxdy", type=float, default=1.0)
    p.add_argument("--lambda-flags", type=float, default=1.0)
    p.add_argument("--lambda-raster", type=float, default=0.5)

    p.add_argument("--raster-size", type=int, default=64)
    p.add_argument("--raster-sigma-px", type=float, default=1.5)

    p.add_argument("--ss-start-epoch", type=int, default=5)
    p.add_argument("--ss-final-teacher-prob", type=float, default=0.2)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-decode-steps", type=int, default=512)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.train_jsonl)
    val_records = read_jsonl(args.val_jsonl)
    style_to_index = build_style_vocab(train_records, val_records)

    with (out_dir / "style_to_index.json").open("w", encoding="utf-8") as f:
        json.dump(style_to_index, f, ensure_ascii=False, indent=2)

    train_ds = WordRouteDataset(
        jsonl_path=args.train_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        require_state_after=False,
    )
    val_ds = WordRouteDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        require_state_after=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_word_routes,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        teacher_prob = teacher_prob_for_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            ss_start_epoch=args.ss_start_epoch,
            ss_final_teacher_prob=args.ss_final_teacher_prob,
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            lambda_dxdy=args.lambda_dxdy,
            lambda_flags=args.lambda_flags,
            lambda_raster=args.lambda_raster,
            teacher_prob=teacher_prob,
            scheduled_sampling_active=True,
            raster_size=args.raster_size,
            raster_sigma_px=args.raster_sigma_px,
        )

        # Validation stays on pure teacher forcing for stable metrics.
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

        # Qualitative sample decode
        sample = val_ds[0]
        sample_state = sample["state_before"].unsqueeze(0).to(device)
        sample_text = sample["text_tokens"].unsqueeze(0).to(device)
        sample_text_len = torch.tensor([sample["text_tokens"].numel()], device=device)
        sample_bbox = sample["bbox"].unsqueeze(0).to(device)
        sample_style = sample["style_index"].unsqueeze(0).to(device)

        pred_events = greedy_decode(
            model=model,
            state_before=sample_state,
            text_tokens=sample_text,
            text_lengths=sample_text_len,
            bbox=sample_bbox,
            style_index=sample_style,
            max_steps=args.max_decode_steps,
        )

        sample_pred_path = out_dir / f"sample_decode_epoch_{epoch:03d}.json"
        with sample_pred_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "record_id": sample["record_id"],
                    "word_text": sample["word_text"],
                    "predicted_events": pred_events.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        row = {
            "epoch": epoch,
            "teacher_prob": teacher_prob,
            "train": train_metrics,
            "val": val_metrics,
            "checkpoint": str(ckpt_path),
            "sample_decode": str(sample_pred_path),
        }
        history.append(row)

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"[epoch {epoch:03d}] "
            f"teacher_prob={teacher_prob:.3f} "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"train_raster={train_metrics['raster']:.4f} "
            f"val_raster={val_metrics['raster']:.4f}"
        )

    print(f"Done. Best val total loss: {best_val:.6f}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
How to run it
Bash
python train_online_scribe.py \
  --train-jsonl data/train_word_routes.jsonl \
  --val-jsonl data/val_word_routes.jsonl \
  --image-root . \
  --out-dir runs/online_word_scribe_v2 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --ss-start-epoch 5 \
  --ss-final-teacher-prob 0.2 \
  --lambda-raster 0.5
The main caveat

This version is intentionally clean, not maximally fast.

Because scheduled sampling is implemented with an autoregressive rollout inside training, it is slower than the earlier pure-teacher-forcing trainer. That is the right tradeoff for a first correct implementation.

The next thing I’d do is factor out a small differentiable line rasterizer instead of pure point splats, because that will make the raster loss sharper on long pen-down segments.

---

## User

Nice, do it

---

## ChatGPT

I’m swapping the point splats for segment rendering, so the raster loss follows the actual pen path between points instead of only the sampled vertices.

---

## ChatGPT

Here’s the upgrade.

The idea is simple:

the old rasterizer puts soft blobs at the predicted points

the new rasterizer renders line segments between consecutive points

so long pen-down strokes become continuous ink instead of dotted traces

That makes raster supervision much better for handwriting.

What changes

Replace:

soft_rasterize_points(...)

with:

soft_rasterize_segments(...)

and keep the rest of the trainer the same, except for the raster-loss call.

1) New differentiable line rasterizer

Drop this into your trainer, replacing the point rasterizer section.

Python
Run
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

    Interprets the event stream as:
      start_point -> start_point + dxdy[0] -> ... -> cumulative positions

    For each segment, compute the squared distance from every raster pixel
    to the closest point on that segment, then splat a Gaussian tube around it.

    Args:
      start_points: [B,2] normalized [0,1]
      dxdy:         [B,T,2]
      pen_probs:    [B,T]
      raster_size:  output side length
      sigma_px:     softness / stroke width in raster pixels
      mode:
        - "max":  per-pixel max over segment contributions
        - "alpha": alpha-compositing style accumulation

    Returns:
      raster: [B,1,R,R] in [0,1]
    """
    b, t, _ = dxdy.shape
    device = dxdy.device
    eps = 1e-6

    # Absolute positions after each event
    pos = start_points.unsqueeze(1) + torch.cumsum(dxdy, dim=1)  # [B,T,2]

    # Segment starts/ends in normalized coordinates
    seg_start = torch.cat([start_points.unsqueeze(1), pos[:, :-1, :]], dim=1)  # [B,T,2]
    seg_end = pos                                                               # [B,T,2]

    # Convert to pixel coordinates
    scale = float(raster_size - 1)
    a = seg_start * scale   # [B,T,2]
    bpt = seg_end * scale   # [B,T,2]

    # Raster grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(raster_size, device=device, dtype=torch.float32),
        torch.arange(raster_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    p = torch.stack([grid_x, grid_y], dim=-1)              # [R,R,2]
    p = p.view(1, 1, raster_size, raster_size, 2)          # [1,1,R,R,2]

    # Broadcast segment geometry
    a = a.unsqueeze(2).unsqueeze(2)                        # [B,T,1,1,2]
    bpt = bpt.unsqueeze(2).unsqueeze(2)                    # [B,T,1,1,2]
    ab = bpt - a                                           # [B,T,1,1,2]
    ap = p - a                                             # [B,T,R,R,2]

    ab_len2 = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(eps)  # [B,T,1,1,1]
    tau = (ap * ab).sum(dim=-1, keepdim=True) / ab_len2            # [B,T,R,R,1]
    tau = tau.clamp(0.0, 1.0)

    closest = a + tau * ab                                  # [B,T,R,R,2]
    dist2 = ((p - closest) ** 2).sum(dim=-1)                # [B,T,R,R]

    # Segment contribution
    gauss = torch.exp(-dist2 / (2.0 * sigma_px * sigma_px)) # [B,T,R,R]
    contrib = pen_probs.unsqueeze(-1).unsqueeze(-1) * gauss # [B,T,R,R]

    if mode == "max":
        raster = torch.amax(contrib, dim=1, keepdim=True)   # [B,1,R,R]
    elif mode == "alpha":
        # Alpha-style accumulation: 1 - Π(1 - a_i)
        contrib = contrib.clamp(0.0, 1.0 - 1e-5)
        raster = 1.0 - torch.prod(1.0 - contrib, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown rasterization mode: {mode}")

    return raster.clamp(0.0, 1.0)
2) Replace the raster-loss function

Use this version instead of the old point-raster loss.

Python
Run
def compute_raster_loss(
    pred_dxdy: Tensor,              # [B,T,2]
    pred_flags_logits: Tensor,      # [B,T,5]
    state_before: Tensor,           # [B,1,H,W]
    state_after: Optional[Tensor],  # [B,1,H,W] or None
    route_start_point: Tensor,      # [B,2]
    raster_valid_mask: Tensor,      # [B] float
    example_weights: Tensor,        # [B]
    raster_size: int,
    sigma_px: float,
    raster_mode: str = "max",
) -> Tensor:
    if state_after is None:
        return pred_dxdy.new_tensor(0.0)

    valid = raster_valid_mask > 0.5
    if valid.sum() == 0:
        return pred_dxdy.new_tensor(0.0)

    pred_dxdy_v = pred_dxdy[valid]
    pen_probs_v = torch.sigmoid(pred_flags_logits[valid, :, 0])  # pen_down
    start_v = route_start_point[valid]
    before_v = state_before[valid]
    after_v = state_after[valid]
    ex_w = example_weights[valid]

    pred_raster = soft_rasterize_segments(
        start_points=start_v,
        dxdy=pred_dxdy_v,
        pen_probs=pen_probs_v,
        raster_size=raster_size,
        sigma_px=sigma_px,
        mode=raster_mode,
    )

    # Target ink delta
    target_delta = torch.relu(after_v - before_v)
    target_delta = F.interpolate(
        target_delta,
        size=(raster_size, raster_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)

    # L1 image loss
    per_example_l1 = torch.abs(pred_raster - target_delta).mean(dim=(1, 2, 3))

    # Optional soft IoU-style overlap term
    inter = torch.minimum(pred_raster, target_delta).sum(dim=(1, 2, 3))
    union = torch.maximum(pred_raster, target_delta).sum(dim=(1, 2, 3)).clamp_min(1e-6)
    per_example_iou = 1.0 - inter / union

    per_example = per_example_l1 + 0.25 * per_example_iou
    raster_loss = (per_example * ex_w).sum() / ex_w.sum().clamp_min(1.0)
    return raster_loss
3) Add one CLI arg

In parse_args(), add:

Python
Run
p.add_argument("--raster-mode", choices=["max", "alpha"], default="max")

max is the safer first default.
alpha can look a bit more ink-like, but it is slightly easier to saturate.

4) Thread the new arg into the training loop

In both calls to run_epoch(...), pass:

Python
Run
raster_mode=args.raster_mode,

So for train:

Python
Run
train_metrics = run_epoch(
    model=model,
    loader=train_loader,
    optimizer=optimizer,
    device=device,
    grad_clip=args.grad_clip,
    lambda_dxdy=args.lambda_dxdy,
    lambda_flags=args.lambda_flags,
    lambda_raster=args.lambda_raster,
    teacher_prob=teacher_prob,
    scheduled_sampling_active=True,
    raster_size=args.raster_size,
    raster_sigma_px=args.raster_sigma_px,
    raster_mode=args.raster_mode,
)

And similarly for validation.

5) Update run_epoch signature

Change it to:

Python
Run
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
    raster_mode: str,
) -> Dict[str, float]:

Then pass raster_mode into compute_raster_loss(...):

Python
Run
raster_loss = compute_raster_loss(
    pred_dxdy=pred_dxdy,
    pred_flags_logits=pred_flags_logits,
    state_before=batch["state_before"],
    state_after=state_after,
    route_start_point=batch["route_start_point"],
    raster_valid_mask=raster_valid_mask,
    example_weights=example_weights,
    raster_size=raster_size,
    sigma_px=raster_sigma_px,
    raster_mode=raster_mode,
)
6) Recommended defaults

I’d start with:

Bash
--raster-size 64
--raster-sigma-px 1.25
--raster-mode max
--lambda-raster 0.5

Then try:

Bash
--raster-mode alpha
--raster-sigma-px 1.5

if you want slightly fuller strokes.

7) Why this is better

The old point rasterizer only says:

“put ink near these sampled vertices”

The new segment rasterizer says:

“put ink along the actual path traveled by the pen”

That matters because handwriting is mostly about continuous curves, not isolated points.

So this will usually improve:

stroke continuity

long downstrokes

loops

visible smoothness of letters

8) One more practical note

This rasterizer assumes your dx,dy events are already in the same normalized coordinate frame as route_start_point. That matches the spec you’ve been building, so it should slot in directly.

The next step after this is a curvature-aware segment rasterizer or a Bezier renderer so the raster loss follows smoother curves instead of piecewise linear segments.

---

## User

ok, do it

---

## ChatGPT

Here’s the fully integrated train_online_scribe.py with:

style conditioning

scheduled sampling

segment-based differentiable raster loss instead of point splats

Python
Run
#!/usr/bin/env python3
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
    """
    seqs: list of [T_i, 7]
    returns:
      padded: [B, T_max, 7]
      mask:   [B, T_max] bool
    """
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
    ids = sorted(ids)
    return {sid: i for i, sid in enumerate(ids)}


# ============================================================
# Dataset
# ============================================================

class WordRouteDataset(Dataset):
    """
    Expected JSONL fields:
      - record_id
      - document_id
      - line_id
      - word_id
      - word_index
      - word_text
      - state_before_ref
      - state_after_ref            (optional but needed for raster loss)
      - target_bbox_parent_norm_cxcywh
      - route_start_point          (optional but needed for raster loss)
      - style_id or writer_id      (optional)
      - events
      - source_weight
      - confidence
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        style_to_index: Dict[str, int],
        max_text_len: int = 64,
        require_state_after: bool = False,
    ) -> None:
        self.records = read_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.style_to_index = style_to_index
        self.max_text_len = max_text_len
        self.require_state_after = require_state_after

    def __len__(self) -> int:
        return len(self.records)

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

        state_before = load_gray_image(self.image_root / rec["state_before_ref"])

        state_after = None
        has_state_after = False
        if rec.get("state_after_ref") is not None:
            state_after = load_gray_image(self.image_root / rec["state_after_ref"])
            has_state_after = True
        elif self.require_state_after:
            raise ValueError(f"Missing state_after_ref for record {rec.get('record_id')}")

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
        feat = self.net(x)                  # [B,D,H',W']
        feat = feat.flatten(2).transpose(1, 2)  # [B,S,D]
        return feat


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
        self.flag_head = nn.Linear(d_model, 5)

    def causal_mask(self, t: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        prev_events: Tensor,   # [B,T,7]
        bbox: Tensor,          # [B,4]
        style_vec: Tensor,     # [B,D]
        memory: Tensor,        # [B,S,D]
    ) -> Dict[str, Tensor]:
        x = self.event_in(prev_events)
        x = x + self.bbox_in(bbox).unsqueeze(1)
        x = x + self.style_in(style_vec).unsqueeze(1)
        x = self.pos(x)

        tgt_mask = self.causal_mask(prev_events.shape[1], prev_events.device)
        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        y = self.norm(y)

        return {
            "dxdy": self.dxdy_head(y),
            "flags": self.flag_head(y),
        }


class OnlineWordScribe(nn.Module):
    def __init__(
        self,
        num_styles: int,
        d_model: int = 256,
        nhead: int = 8,
        text_layers: int = 2,
        dec_layers: int = 6,
    ) -> None:
        super().__init__()
        self.state_encoder = SmallCNNStateEncoder(in_ch=1, d_model=d_model)
        self.text_encoder = TextEncoder(vocab_size=256, d_model=d_model, nhead=nhead, num_layers=text_layers)
        self.style_embed = nn.Embedding(num_styles, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.decoder = EventDecoder(d_model=d_model, nhead=nhead, num_layers=dec_layers, event_dim=7)

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

    def decode(
        self,
        prev_events: Tensor,
        bbox: Tensor,
        style_vec: Tensor,
        memory: Tensor,
    ) -> Dict[str, Tensor]:
        return self.decoder(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
        )


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
# Losses
# ============================================================

@dataclass
class LossBreakdown:
    total: Tensor
    dxdy: Tensor
    flags: Tensor
    raster: Tensor
    pen_down: Tensor
    stroke_end: Tensor
    char_end: Tensor
    word_end: Tensor
    seq_end: Tensor


def compute_sequence_loss(
    pred_dxdy: Tensor,          # [B,T,2]
    pred_flags_logits: Tensor,  # [B,T,5]
    target_events: Tensor,      # [B,T,7]
    event_mask: Tensor,         # [B,T] bool
    example_weights: Tensor,    # [B]
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    dxdy_tgt = target_events[:, :, 0:2]
    flags_tgt = target_events[:, :, 2:7]

    mask = event_mask.float()
    wt = mask * example_weights.unsqueeze(1)

    huber = F.smooth_l1_loss(pred_dxdy, dxdy_tgt, reduction="none").sum(dim=-1)
    dxdy_loss = (huber * wt).sum() / wt.sum().clamp_min(1.0)

    bce = F.binary_cross_entropy_with_logits(pred_flags_logits, flags_tgt, reduction="none")
    per_flag = []
    for i in range(5):
        li = (bce[:, :, i] * wt).sum() / wt.sum().clamp_min(1.0)
        per_flag.append(li)

    flags_loss = sum(per_flag) / 5.0
    return dxdy_loss, flags_loss, per_flag


def compute_raster_loss(
    pred_dxdy: Tensor,              # [B,T,2]
    pred_flags_logits: Tensor,      # [B,T,5]
    state_before: Tensor,           # [B,1,H,W]
    state_after: Optional[Tensor],  # [B,1,H,W] or None
    route_start_point: Tensor,      # [B,2]
    raster_valid_mask: Tensor,      # [B] float
    example_weights: Tensor,        # [B]
    raster_size: int,
    sigma_px: float,
    raster_mode: str = "max",
) -> Tensor:
    if state_after is None:
        return pred_dxdy.new_tensor(0.0)

    valid = raster_valid_mask > 0.5
    if valid.sum() == 0:
        return pred_dxdy.new_tensor(0.0)

    pred_dxdy_v = pred_dxdy[valid]
    pen_probs_v = torch.sigmoid(pred_flags_logits[valid, :, 0])  # pen_down
    start_v = route_start_point[valid]
    before_v = state_before[valid]
    after_v = state_after[valid]
    ex_w = example_weights[valid]

    pred_raster = soft_rasterize_segments(
        start_points=start_v,
        dxdy=pred_dxdy_v,
        pen_probs=pen_probs_v,
        raster_size=raster_size,
        sigma_px=sigma_px,
        mode=raster_mode,
    )

    target_delta = torch.relu(after_v - before_v)
    target_delta = F.interpolate(
        target_delta,
        size=(raster_size, raster_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)

    per_example_l1 = torch.abs(pred_raster - target_delta).mean(dim=(1, 2, 3))

    inter = torch.minimum(pred_raster, target_delta).sum(dim=(1, 2, 3))
    union = torch.maximum(pred_raster, target_delta).sum(dim=(1, 2, 3)).clamp_min(1e-6)
    per_example_iou = 1.0 - inter / union

    per_example = per_example_l1 + 0.25 * per_example_iou
    raster_loss = (per_example * ex_w).sum() / ex_w.sum().clamp_min(1.0)
    return raster_loss


# ============================================================
# Rollout / scheduled sampling
# ============================================================

def rollout_autoregressive(
    model: OnlineWordScribe,
    state_before: Tensor,
    text_tokens: Tensor,
    text_lengths: Tensor,
    bbox: Tensor,
    style_index: Tensor,
    target_events: Tensor,      # [B,T,7]
    teacher_prob: float,
    sample_model_inputs: bool,
) -> Tuple[Tensor, Tensor]:
    b, t, _ = target_events.shape
    device = target_events.device

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    prev_events = torch.zeros(b, 1, 7, device=device)
    pred_dxdy_list: List[Tensor] = []
    pred_flags_list: List[Tensor] = []

    for step in range(t):
        out = model.decode(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
        )

        dxdy_step = out["dxdy"][:, -1, :]
        flags_step = out["flags"][:, -1, :]

        pred_dxdy_list.append(dxdy_step)
        pred_flags_list.append(flags_step)

        pred_event = torch.cat([dxdy_step, torch.sigmoid(flags_step)], dim=-1)
        gold_event = target_events[:, step, :]

        if sample_model_inputs:
            use_teacher = (torch.rand(b, device=device) < teacher_prob).float().unsqueeze(1)
            next_input = use_teacher * gold_event + (1.0 - use_teacher) * pred_event.detach()
        else:
            if teacher_prob >= 1.0:
                next_input = gold_event
            else:
                use_teacher = (torch.rand(b, device=device) < teacher_prob).float().unsqueeze(1)
                next_input = use_teacher * gold_event + (1.0 - use_teacher) * pred_event.detach()

        prev_events = torch.cat([prev_events, next_input.unsqueeze(1)], dim=1)

    pred_dxdy = torch.stack(pred_dxdy_list, dim=1)
    pred_flags_logits = torch.stack(pred_flags_list, dim=1)
    return pred_dxdy, pred_flags_logits


@torch.no_grad()
def greedy_decode(
    model: OnlineWordScribe,
    state_before: Tensor,     # [1,1,H,W]
    text_tokens: Tensor,      # [1,T]
    text_lengths: Tensor,     # [1]
    bbox: Tensor,             # [1,4]
    style_index: Tensor,      # [1]
    max_steps: int = 512,
    stop_threshold: float = 0.5,
) -> Tensor:
    model.eval()
    device = state_before.device

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    prev_events = torch.zeros(1, 1, 7, device=device)
    generated: List[Tensor] = []

    for _ in range(max_steps):
        out = model.decode(
            prev_events=prev_events,
            bbox=bbox,
            style_vec=style_vec,
            memory=memory,
        )

        dxdy = out["dxdy"][:, -1, :]
        flags_logits = out["flags"][:, -1, :]
        flags = torch.sigmoid(flags_logits)

        next_evt = torch.cat([dxdy, flags], dim=-1)
        generated.append(next_evt[0].detach().cpu())

        prev_events = torch.cat([prev_events, next_evt.unsqueeze(1)], dim=1)

        if bool(flags[0, 4].item() > stop_threshold):  # seq_end
            break

    if not generated:
        return torch.zeros(0, 7)
    return torch.stack(generated, dim=0)


# ============================================================
# Training / eval
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
    raster_mode: str,
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

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        pred_dxdy, pred_flags_logits = rollout_autoregressive(
            model=model,
            state_before=batch["state_before"],
            text_tokens=batch["text_tokens"],
            text_lengths=batch["text_lengths"],
            bbox=batch["bbox"],
            style_index=batch["style_index"],
            target_events=batch["events"],
            teacher_prob=teacher_prob,
            sample_model_inputs=scheduled_sampling_active and train,
        )

        example_weights = batch["source_weight"] * batch["confidence"]

        dxdy_loss, flags_loss, per_flag = compute_sequence_loss(
            pred_dxdy=pred_dxdy,
            pred_flags_logits=pred_flags_logits,
            target_events=batch["events"],
            event_mask=batch["event_mask"],
            example_weights=example_weights,
        )

        state_after = batch.get("state_after", None)
        raster_valid_mask = batch["route_start_valid"] * batch["has_state_after"]
        raster_loss = compute_raster_loss(
            pred_dxdy=pred_dxdy,
            pred_flags_logits=pred_flags_logits,
            state_before=batch["state_before"],
            state_after=state_after,
            route_start_point=batch["route_start_point"],
            raster_valid_mask=raster_valid_mask,
            example_weights=example_weights,
            raster_size=raster_size,
            sigma_px=raster_sigma_px,
            raster_mode=raster_mode,
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

        bs = batch["events"].shape[0]
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
    p = argparse.ArgumentParser(description="Train an online word-level handwriting scribe with style conditioning, scheduled sampling, and segment-based raster loss.")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
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

    p.add_argument("--lambda-dxdy", type=float, default=1.0)
    p.add_argument("--lambda-flags", type=float, default=1.0)
    p.add_argument("--lambda-raster", type=float, default=0.5)

    p.add_argument("--raster-size", type=int, default=64)
    p.add_argument("--raster-sigma-px", type=float, default=1.25)
    p.add_argument("--raster-mode", choices=["max", "alpha"], default="max")

    p.add_argument("--ss-start-epoch", type=int, default=5)
    p.add_argument("--ss-final-teacher-prob", type=float, default=0.2)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-decode-steps", type=int, default=512)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.train_jsonl)
    val_records = read_jsonl(args.val_jsonl)
    style_to_index = build_style_vocab(train_records, val_records)

    with (out_dir / "style_to_index.json").open("w", encoding="utf-8") as f:
        json.dump(style_to_index, f, ensure_ascii=False, indent=2)

    train_ds = WordRouteDataset(
        jsonl_path=args.train_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        require_state_after=False,
    )
    val_ds = WordRouteDataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        style_to_index=style_to_index,
        require_state_after=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_word_routes,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        teacher_prob = teacher_prob_for_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            ss_start_epoch=args.ss_start_epoch,
            ss_final_teacher_prob=args.ss_final_teacher_prob,
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            lambda_dxdy=args.lambda_dxdy,
            lambda_flags=args.lambda_flags,
            lambda_raster=args.lambda_raster,
            teacher_prob=teacher_prob,
            scheduled_sampling_active=True,
            raster_size=args.raster_size,
            raster_sigma_px=args.raster_sigma_px,
            raster_mode=args.raster_mode,
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
            raster_mode=args.raster_mode,
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

        sample = val_ds[0]
        sample_state = sample["state_before"].unsqueeze(0).to(device)
        sample_text = sample["text_tokens"].unsqueeze(0).to(device)
        sample_text_len = torch.tensor([sample["text_tokens"].numel()], device=device)
        sample_bbox = sample["bbox"].unsqueeze(0).to(device)
        sample_style = sample["style_index"].unsqueeze(0).to(device)

        pred_events = greedy_decode(
            model=model,
            state_before=sample_state,
            text_tokens=sample_text,
            text_lengths=sample_text_len,
            bbox=sample_bbox,
            style_index=sample_style,
            max_steps=args.max_decode_steps,
        )

        sample_pred_path = out_dir / f"sample_decode_epoch_{epoch:03d}.json"
        with sample_pred_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "record_id": sample["record_id"],
                    "word_text": sample["word_text"],
                    "predicted_events": pred_events.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        row = {
            "epoch": epoch,
            "teacher_prob": teacher_prob,
            "train": train_metrics,
            "val": val_metrics,
            "checkpoint": str(ckpt_path),
            "sample_decode": str(sample_pred_path),
        }
        history.append(row)

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"[epoch {epoch:03d}] "
            f"teacher_prob={teacher_prob:.3f} "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"train_raster={train_metrics['raster']:.4f} "
            f"val_raster={val_metrics['raster']:.4f}"
        )

    print(f"Done. Best val total loss: {best_val:.6f}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

Run it like this:

Bash
python train_online_scribe.py \
  --train-jsonl data/train_word_routes.jsonl \
  --val-jsonl data/val_word_routes.jsonl \
  --image-root . \
  --out-dir runs/online_word_scribe_v3 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --ss-start-epoch 5 \
  --ss-final-teacher-prob 0.2 \
  --lambda-raster 0.5 \
  --raster-size 64 \
  --raster-sigma-px 1.25 \
  --raster-mode max

The next useful upgrade is to 
add a tiny renderer debug script that writes predicted and target raster images side by side for a batch.

---

## User

okay

---

## ChatGPT

Good.

The next best add-on is a small debug script that renders:

predicted stroke raster

target delta raster

absolute trajectory overlay

for a few validation samples each epoch, so you can tell whether failures are coming from:

bad event geometry,

bad stop flags,

or bad raster alignment.

---

## User

you didn't do anything try again and do this next set of stuff

---

## ChatGPT

I’m fixing that by actually giving you the next implementation now: a debug renderer script plus the exact hook to call it from training.

---

## ChatGPT

You were right — I hadn’t actually implemented the next piece.

I’ve now created the debug renderer script here:

debug_render_online_scribe.py

What it does:

loads a checkpoint from train_online_scribe.py

runs greedy decoding on validation samples

renders side-by-side panels with:

state_before

state_after

target_delta

pred_raster

gold_raster

pred_path

gold_path

saves predicted event JSON for each sample

writes a manifest.json

Run it like this:

Bash
python /mnt/data/debug_render_online_scribe.py \
  --checkpoint runs/online_word_scribe_v3/best.pt \
  --jsonl data/val_word_routes.jsonl \
  --image-root . \
  --out-dir runs/online_word_scribe_v3/debug_epoch_best \
  --num-samples 8 \
  --raster-size 96 \
  --raster-sigma-px 1.25 \
  --raster-mode max

If you want it auto-run after every epoch inside the trainer, add this right after the sample decode block:

Python
Run
import subprocess

subprocess.run(
    [
        "python",
        "/mnt/data/debug_render_online_scribe.py",
        "--checkpoint",
        str(ckpt_path if not is_best else (out_dir / "best.pt")),
        "--jsonl",
        args.val_jsonl,
        "--image-root",
        args.image_root,
        "--out-dir",
        str(out_dir / f"debug_epoch_{epoch:03d}"),
        "--num-samples",
        "6",
        "--raster-size",
        str(args.raster_size),
        "--raster-sigma-px",
        str(args.raster_sigma_px),
        "--raster-mode",
        args.raster_mode,
    ],
    check=False,
)

The next thing to build is a small script that turns those debug folders into a contact sheet or animated GIF across epochs so you can see whether the handwriting is stabilizing or drifting.

---

## User

debug_render_online_scribe.py, include it in the chat directly

---

## ChatGPT

I’m pulling the exact script content so I can paste it directly here without drifting from what was generated.

---

## ChatGPT

Python
Run
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# I/O
# ============================================================

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


# ============================================================
# Model definitions (must match train_online_scribe.py)
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
        feat = self.net(x)
        return feat.flatten(2).transpose(1, 2)


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
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6, event_dim: int = 7) -> None:
        super().__init__()
        self.event_in = nn.Linear(event_dim, d_model)
        self.bbox_in = nn.Linear(4, d_model)
        self.style_in = nn.Linear(d_model, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
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
        self.flag_head = nn.Linear(d_model, 5)

    def causal_mask(self, t: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, prev_events: Tensor, bbox: Tensor, style_vec: Tensor, memory: Tensor) -> Dict[str, Tensor]:
        x = self.event_in(prev_events)
        x = x + self.bbox_in(bbox).unsqueeze(1)
        x = x + self.style_in(style_vec).unsqueeze(1)
        x = self.pos(x)
        tgt_mask = self.causal_mask(prev_events.shape[1], prev_events.device)
        y = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        y = self.norm(y)
        return {"dxdy": self.dxdy_head(y), "flags": self.flag_head(y)}


class OnlineWordScribe(nn.Module):
    def __init__(self, num_styles: int, d_model: int = 256, nhead: int = 8, text_layers: int = 2, dec_layers: int = 6) -> None:
        super().__init__()
        self.state_encoder = SmallCNNStateEncoder(in_ch=1, d_model=d_model)
        self.text_encoder = TextEncoder(vocab_size=256, d_model=d_model, nhead=nhead, num_layers=text_layers)
        self.style_embed = nn.Embedding(num_styles, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.decoder = EventDecoder(d_model=d_model, nhead=nhead, num_layers=dec_layers, event_dim=7)

    def encode_context(self, state_before: Tensor, text_tokens: Tensor, text_lengths: Tensor, style_index: Tensor) -> Tuple[Tensor, Tensor]:
        state_mem = self.state_encoder(state_before)
        text_mem = self.text_encoder(text_tokens, text_lengths)
        style_vec = self.style_embed(style_index)
        style_token = style_vec.unsqueeze(1)
        memory = torch.cat([style_token, state_mem, text_mem], dim=1)
        memory = self.memory_proj(memory)
        return memory, style_vec

    def decode(self, prev_events: Tensor, bbox: Tensor, style_vec: Tensor, memory: Tensor) -> Dict[str, Tensor]:
        return self.decoder(prev_events=prev_events, bbox=bbox, style_vec=style_vec, memory=memory)


# ============================================================
# Rendering
# ============================================================

def soft_rasterize_segments(
    start_points: Tensor,
    dxdy: Tensor,
    pen_probs: Tensor,
    raster_size: int = 64,
    sigma_px: float = 1.25,
    mode: str = "max",
) -> Tensor:
    device = dxdy.device
    eps = 1e-6
    pos = start_points.unsqueeze(1) + torch.cumsum(dxdy, dim=1)
    seg_start = torch.cat([start_points.unsqueeze(1), pos[:, :-1, :]], dim=1)
    seg_end = pos

    scale = float(raster_size - 1)
    a = seg_start * scale
    bpt = seg_end * scale

    grid_y, grid_x = torch.meshgrid(
        torch.arange(raster_size, device=device, dtype=torch.float32),
        torch.arange(raster_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    p = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, raster_size, raster_size, 2)

    a = a.unsqueeze(2).unsqueeze(2)
    bpt = bpt.unsqueeze(2).unsqueeze(2)
    ab = bpt - a
    ap = p - a

    ab_len2 = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(eps)
    tau = (ap * ab).sum(dim=-1, keepdim=True) / ab_len2
    tau = tau.clamp(0.0, 1.0)
    closest = a + tau * ab
    dist2 = ((p - closest) ** 2).sum(dim=-1)

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


@torch.no_grad()
def greedy_decode(
    model: OnlineWordScribe,
    state_before: Tensor,
    text_tokens: Tensor,
    text_lengths: Tensor,
    bbox: Tensor,
    style_index: Tensor,
    max_steps: int = 512,
    stop_threshold: float = 0.5,
) -> Tensor:
    model.eval()
    device = state_before.device

    memory, style_vec = model.encode_context(
        state_before=state_before,
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        style_index=style_index,
    )

    prev_events = torch.zeros(1, 1, 7, device=device)
    generated: List[Tensor] = []

    for _ in range(max_steps):
        out = model.decode(prev_events=prev_events, bbox=bbox, style_vec=style_vec, memory=memory)
        dxdy = out["dxdy"][:, -1, :]
        flags = torch.sigmoid(out["flags"][:, -1, :])
        next_evt = torch.cat([dxdy, flags], dim=-1)
        generated.append(next_evt[0].detach().cpu())
        prev_events = torch.cat([prev_events, next_evt.unsqueeze(1)], dim=1)
        if bool(flags[0, 4].item() > stop_threshold):
            break

    if not generated:
        return torch.zeros(0, 7)
    return torch.stack(generated, dim=0)


# ============================================================
# Visualization helpers
# ============================================================

def tensor_gray_to_pil(x: Tensor) -> Image.Image:
    # x: [1,H,W] or [H,W]
    if x.dim() == 3:
        x = x[0]
    arr = (x.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def tensor_heat_to_pil(x: Tensor) -> Image.Image:
    # simple blue->cyan->white heat mapping
    if x.dim() == 3:
        x = x[0]
    arr = x.detach().cpu().clamp(0, 1).numpy()
    r = (255.0 * arr).astype(np.uint8)
    g = (255.0 * np.sqrt(arr)).astype(np.uint8)
    b = np.full_like(r, 255, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def draw_absolute_path(size: int, start_point: Tensor, events: Tensor) -> Image.Image:
    """
    start_point: [2] normalized [0,1]
    events: [T,7]
    """
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if events.numel() == 0:
        return img

    start = start_point.detach().cpu().numpy().astype(np.float32)
    pos = start.copy()
    scale = float(size - 1)

    # draw start point
    sx, sy = int(round(pos[0] * scale)), int(round(pos[1] * scale))
    draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=(255, 0, 0))

    for evt in events.detach().cpu().numpy():
        dx, dy, pen_down, stroke_end, char_end, word_end, seq_end = evt.tolist()
        new_pos = pos + np.array([dx, dy], dtype=np.float32)
        x0, y0 = pos * scale
        x1, y1 = new_pos * scale
        color = (0, 0, 0) if pen_down >= 0.5 else (180, 180, 180)
        width = 2 if pen_down >= 0.5 else 1
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=color, width=width)
        if stroke_end >= 0.5:
            draw.ellipse((x1 - 2, y1 - 2, x1 + 2, y1 + 2), outline=(0, 100, 255), width=1)
        if char_end >= 0.5:
            draw.ellipse((x1 - 3, y1 - 3, x1 + 3, y1 + 3), outline=(255, 120, 0), width=1)
        pos = new_pos
        if seq_end >= 0.5:
            break

    return img


def add_label(img: Image.Image, text: str, pad_h: int = 22) -> Image.Image:
    labeled = Image.new("RGB", (img.width, img.height + pad_h), (255, 255, 255))
    labeled.paste(img.convert("RGB"), (0, pad_h))
    draw = ImageDraw.Draw(labeled)
    draw.text((4, 4), text, fill=(0, 0, 0), font=ImageFont.load_default())
    return labeled


def make_panel(images: List[Image.Image], cols: int = 3, bg=(245, 245, 245)) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    cell_w = max(widths)
    cell_h = max(heights)
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), bg)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * cell_w + (cell_w - im.width) // 2
        y = r * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


# ============================================================
# Main debug routine
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render predicted vs target debug panels for online word scribe checkpoints.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-decode-steps", type=int, default=512)
    p.add_argument("--raster-size", type=int, default=96)
    p.add_argument("--raster-sigma-px", type=float, default=1.25)
    p.add_argument("--raster-mode", choices=["max", "alpha"], default="max")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    style_to_index: Dict[str, int] = ckpt["style_to_index"]
    model_args = ckpt["args"]

    model = OnlineWordScribe(
        num_styles=len(style_to_index),
        d_model=int(model_args.get("d_model", 256)),
        nhead=int(model_args.get("nhead", 8)),
        text_layers=int(model_args.get("text_layers", 2)),
        dec_layers=int(model_args.get("dec_layers", 6)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    records = read_jsonl(args.jsonl)
    end = min(len(records), args.offset + args.num_samples)
    chosen = records[args.offset:end]

    manifest: List[Dict[str, Any]] = []

    for i, rec in enumerate(chosen):
        state_before = load_gray_image(Path(args.image_root) / rec["state_before_ref"]).unsqueeze(0).to(device)
        state_after = None
        if rec.get("state_after_ref") is not None:
            state_after = load_gray_image(Path(args.image_root) / rec["state_after_ref"]).unsqueeze(0).to(device)

        text_tokens = torch.tensor([list(rec["word_text"].encode("utf-8"))[:64]], dtype=torch.long, device=device)
        text_lengths = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)
        bbox = torch.tensor([rec["target_bbox_parent_norm_cxcywh"]], dtype=torch.float32, device=device)
        style_id = rec.get("style_id", rec.get("writer_id", "__unk__"))
        style_index = torch.tensor([style_to_index.get(str(style_id), style_to_index["__unk__"])], dtype=torch.long, device=device)

        pred_events = greedy_decode(
            model=model,
            state_before=state_before,
            text_tokens=text_tokens,
            text_lengths=text_lengths,
            bbox=bbox,
            style_index=style_index,
            max_steps=args.max_decode_steps,
        )

        gold_events_rows = []
        for e in rec["events"]:
            gold_events_rows.append([
                float(e["dx"]), float(e["dy"]), float(e["pen_down"]), float(e["stroke_end"]),
                float(e["char_end"]), float(e["word_end"]), float(e["seq_end"]),
            ])
        gold_events = torch.tensor(gold_events_rows, dtype=torch.float32)

        route_start = rec.get("route_start_point")
        route_start_valid = route_start is not None
        if route_start_valid:
            start_pt = torch.tensor([route_start], dtype=torch.float32, device=device)
            pred_raster = soft_rasterize_segments(
                start_points=start_pt,
                dxdy=pred_events[:, 0:2].unsqueeze(0).to(device),
                pen_probs=pred_events[:, 2].unsqueeze(0).to(device),
                raster_size=args.raster_size,
                sigma_px=args.raster_sigma_px,
                mode=args.raster_mode,
            )[0].cpu()

            gold_raster = soft_rasterize_segments(
                start_points=start_pt,
                dxdy=gold_events[:, 0:2].unsqueeze(0).to(device),
                pen_probs=gold_events[:, 2].unsqueeze(0).to(device),
                raster_size=args.raster_size,
                sigma_px=args.raster_sigma_px,
                mode=args.raster_mode,
            )[0].cpu()

            pred_path = draw_absolute_path(args.raster_size, start_pt[0].cpu(), pred_events)
            gold_path = draw_absolute_path(args.raster_size, start_pt[0].cpu(), gold_events)
        else:
            pred_raster = torch.zeros(1, args.raster_size, args.raster_size)
            gold_raster = torch.zeros(1, args.raster_size, args.raster_size)
            pred_path = Image.new("RGB", (args.raster_size, args.raster_size), (255, 255, 255))
            gold_path = Image.new("RGB", (args.raster_size, args.raster_size), (255, 255, 255))

        before_img = tensor_gray_to_pil(state_before[0].cpu())
        if state_after is not None:
            after_img = tensor_gray_to_pil(state_after[0].cpu())
            target_delta = torch.relu(state_after[0].cpu() - state_before[0].cpu())
            target_delta = F.interpolate(target_delta.unsqueeze(0), size=(args.raster_size, args.raster_size), mode="bilinear", align_corners=False)[0]
            target_delta_img = tensor_heat_to_pil(target_delta)
        else:
            after_img = Image.new("L", (before_img.width, before_img.height), 255)
            target_delta_img = Image.new("RGB", (args.raster_size, args.raster_size), (255, 255, 255))

        pred_raster_img = tensor_heat_to_pil(pred_raster)
        gold_raster_img = tensor_heat_to_pil(gold_raster)

        panel = make_panel([
            add_label(before_img.convert("RGB"), "state_before"),
            add_label(after_img.convert("RGB"), "state_after"),
            add_label(target_delta_img, "target_delta"),
            add_label(pred_raster_img, "pred_raster"),
            add_label(gold_raster_img, "gold_raster"),
            add_label(pred_path, "pred_path"),
            add_label(gold_path, "gold_path"),
        ], cols=3)

        stem = f"{i:03d}_{rec['word_id']}"
        panel_path = out_dir / f"{stem}.png"
        panel.save(panel_path)

        pred_events_path = out_dir / f"{stem}_pred_events.json"
        with pred_events_path.open("w", encoding="utf-8") as f:
            json.dump({
                "record_id": rec["record_id"],
                "word_id": rec["word_id"],
                "word_text": rec["word_text"],
                "predicted_events": pred_events.tolist(),
            }, f, ensure_ascii=False, indent=2)

        manifest.append({
            "record_id": rec["record_id"],
            "word_id": rec["word_id"],
            "word_text": rec["word_text"],
            "panel": str(panel_path),
            "pred_events": str(pred_events_path),
        })

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(manifest)} debug panels to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

---

## User

The next thing to build is a small script that turns those debug folders into a contact sheet or animated GIF across epochs so you can see whether the handwriting is stabilizing or drifting.

Do this

---

## ChatGPT

Python
Run
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Helpers
# ============================================================

EPOCH_RE = re.compile(r"(\d+)")


def natural_epoch_key(path: Path) -> Tuple[int, str]:
    m = EPOCH_RE.search(path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def add_header(img: Image.Image, title: str, subtitle: Optional[str] = None, pad: int = 30) -> Image.Image:
    img = img.convert("RGB")
    out = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    font = load_font()
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    if subtitle:
        draw.text((6, 16), subtitle, fill=(90, 90, 90), font=font)
    return out


def resize_to_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    new_h = max(1, int(round(img.height * width / img.width)))
    return img.resize((width, new_h), Image.BILINEAR)


def pad_to_same_size(images: Sequence[Image.Image], bg=(245, 245, 245)) -> List[Image.Image]:
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)
    out: List[Image.Image] = []
    for im in images:
        canvas = Image.new("RGB", (max_w, max_h), bg)
        x = (max_w - im.width) // 2
        y = (max_h - im.height) // 2
        canvas.paste(im.convert("RGB"), (x, y))
        out.append(canvas)
    return out


def make_grid(images: Sequence[Image.Image], cols: int = 3, bg=(245, 245, 245), pad: int = 8) -> Image.Image:
    if not images:
        return Image.new("RGB", (256, 256), (255, 255, 255))

    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols

    canvas = Image.new(
        "RGB",
        (cols * max_w + (cols + 1) * pad, rows * max_h + (rows + 1) * pad),
        bg,
    )

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + pad)
        canvas.paste(im.convert("RGB"), (x, y))
    return canvas


def save_gif(frames: Sequence[Image.Image], out_path: str | Path, duration_ms: int, loop: int = 0) -> None:
    frames = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    if not frames:
        return
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )


# ============================================================
# Indexing debug folders
# ============================================================

def find_debug_dirs(root: Path, glob_pat: str) -> List[Path]:
    dirs = [p for p in root.glob(glob_pat) if p.is_dir()]
    dirs.sort(key=natural_epoch_key)
    return dirs


def load_epoch_manifest(epoch_dir: Path) -> Tuple[str, List[Dict[str, Any]]]:
    manifest_path = epoch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {epoch_dir}")
    manifest = read_json(manifest_path)
    return epoch_dir.name, manifest


def resolve_panel_path(epoch_dir: Path, entry: Dict[str, Any]) -> Path:
    p = Path(entry["panel"])
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return epoch_dir / p.name


def build_index(debug_dirs: Sequence[Path]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Returns:
      epoch_names: ordered list of epoch dir names
      by_record: record_id -> epoch_name -> manifest entry
    """
    epoch_names: List[str] = []
    by_record: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for epoch_dir in debug_dirs:
        epoch_name, manifest = load_epoch_manifest(epoch_dir)
        epoch_names.append(epoch_name)
        for entry in manifest:
            rec_id = entry["record_id"]
            row = dict(entry)
            row["_epoch_dir"] = str(epoch_dir)
            by_record[rec_id][epoch_name] = row

    return epoch_names, by_record


# ============================================================
# Outputs
# ============================================================

def make_per_sample_gifs(
    epoch_names: Sequence[str],
    by_record: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
    duration_ms: int,
    min_epochs: int,
) -> List[Dict[str, Any]]:
    gif_dir = ensure_dir(out_dir / "sample_gifs")
    summary: List[Dict[str, Any]] = []

    for rec_id, epoch_map in sorted(by_record.items()):
        available = [ep for ep in epoch_names if ep in epoch_map]
        if len(available) < min_epochs:
            continue

        frames: List[Image.Image] = []
        word_text = None
        word_id = None

        for ep in available:
            entry = epoch_map[ep]
            epoch_dir = Path(entry["_epoch_dir"])
            panel_path = resolve_panel_path(epoch_dir, entry)
            if not panel_path.exists():
                continue

            img = Image.open(panel_path).convert("RGB")
            word_text = entry.get("word_text", word_text)
            word_id = entry.get("word_id", word_id)
            title = f"{ep}"
            subtitle = f"{word_text or ''} | {word_id or ''} | {rec_id}"
            frames.append(add_header(img, title=title, subtitle=subtitle))

        if len(frames) < min_epochs:
            continue

        frames = pad_to_same_size(frames)
        out_path = gif_dir / f"{rec_id}.gif"
        save_gif(frames, out_path, duration_ms=duration_ms)

        summary.append(
            {
                "record_id": rec_id,
                "word_text": word_text,
                "word_id": word_id,
                "epochs": available,
                "gif": str(out_path),
            }
        )

    return summary


def choose_sample_ids(
    epoch_names: Sequence[str],
    by_record: Dict[str, Dict[str, Dict[str, Any]]],
    max_samples: int,
    require_all_epochs: bool,
) -> List[str]:
    scored: List[Tuple[int, str]] = []
    for rec_id, epoch_map in by_record.items():
        coverage = sum(ep in epoch_map for ep in epoch_names)
        if require_all_epochs and coverage != len(epoch_names):
            continue
        scored.append((-coverage, rec_id))
    scored.sort()
    return [rec_id for _, rec_id in scored[:max_samples]]


def make_epoch_contact_sheets(
    epoch_names: Sequence[str],
    by_record: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
    sample_ids: Sequence[str],
    cols: int,
    thumb_width: int,
) -> List[Dict[str, Any]]:
    sheet_dir = ensure_dir(out_dir / "epoch_contact_sheets")
    rows: List[Dict[str, Any]] = []

    for ep in epoch_names:
        tiles: List[Image.Image] = []

        for rec_id in sample_ids:
            entry = by_record.get(rec_id, {}).get(ep)
            if entry is None:
                continue

            epoch_dir = Path(entry["_epoch_dir"])
            panel_path = resolve_panel_path(epoch_dir, entry)
            if not panel_path.exists():
                continue

            img = Image.open(panel_path).convert("RGB")
            img = resize_to_width(img, thumb_width)

            word_text = entry.get("word_text", "")
            word_id = entry.get("word_id", "")
            subtitle = f"{word_text} | {word_id}"
            img = add_header(img, title=rec_id, subtitle=subtitle)
            tiles.append(img)

        if not tiles:
            continue

        sheet = make_grid(tiles, cols=cols)
        sheet = add_header(sheet, title=ep, subtitle=f"{len(tiles)} samples")
        out_path = sheet_dir / f"{ep}.png"
        sheet.save(out_path)

        rows.append(
            {
                "epoch": ep,
                "sheet": str(out_path),
                "num_samples": len(tiles),
            }
        )

    return rows


def make_contact_sheet_gif(sheet_rows: Sequence[Dict[str, Any]], out_path: Path, duration_ms: int) -> None:
    frames: List[Image.Image] = []
    for row in sheet_rows:
        p = Path(row["sheet"])
        if p.exists():
            frames.append(Image.open(p).convert("RGB"))
    if frames:
        frames = pad_to_same_size(frames)
        save_gif(frames, out_path, duration_ms=duration_ms)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Turn debug_epoch_* folders into per-sample animated GIFs and per-epoch contact sheets."
    )
    p.add_argument("--debug-root", required=True, help="Root directory containing debug_epoch_* folders")
    p.add_argument("--glob", default="debug_epoch_*", help="Glob pattern for debug folders")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--gif-ms", type=int, default=700, help="Frame duration for GIFs in milliseconds")
    p.add_argument("--min-epochs", type=int, default=2, help="Minimum epochs required to build a sample GIF")
    p.add_argument("--max-samples", type=int, default=12, help="Max samples in epoch contact sheets")
    p.add_argument("--contact-cols", type=int, default=3, help="Columns in epoch contact sheets")
    p.add_argument("--thumb-width", type=int, default=420, help="Thumbnail width for contact sheets")
    p.add_argument(
        "--require-all-epochs",
        action="store_true",
        help="Only use samples present in every epoch for contact sheets",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    debug_root = Path(args.debug_root)
    out_dir = ensure_dir(args.out_dir)

    debug_dirs = find_debug_dirs(debug_root, args.glob)
    if not debug_dirs:
        raise FileNotFoundError(f"No debug directories found under {debug_root} with glob {args.glob!r}")

    epoch_names, by_record = build_index(debug_dirs)

    sample_gif_rows = make_per_sample_gifs(
        epoch_names=epoch_names,
        by_record=by_record,
        out_dir=out_dir,
        duration_ms=args.gif_ms,
        min_epochs=args.min_epochs,
    )

    chosen_sample_ids = choose_sample_ids(
        epoch_names=epoch_names,
        by_record=by_record,
        max_samples=args.max_samples,
        require_all_epochs=args.require_all_epochs,
    )

    sheet_rows = make_epoch_contact_sheets(
        epoch_names=epoch_names,
        by_record=by_record,
        out_dir=out_dir,
        sample_ids=chosen_sample_ids,
        cols=args.contact_cols,
        thumb_width=args.thumb_width,
    )

    contact_sheet_gif = out_dir / "epochs_contact_sheet.gif"
    make_contact_sheet_gif(
        sheet_rows=sheet_rows,
        out_path=contact_sheet_gif,
        duration_ms=args.gif_ms,
    )

    summary = {
        "debug_root": str(debug_root),
        "epochs": epoch_names,
        "num_records": len(by_record),
        "sample_gifs": sample_gif_rows,
        "contact_sheet_samples": chosen_sample_ids,
        "epoch_contact_sheets": sheet_rows,
        "contact_sheet_gif": str(contact_sheet_gif),
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Found {len(epoch_names)} epochs and {len(by_record)} unique records")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote per-sample GIFs to {out_dir / 'sample_gifs'}")
    print(f"Wrote per-epoch contact sheets to {out_dir / 'epoch_contact_sheets'}")
    print(f"Wrote contact-sheet GIF to {contact_sheet_gif}")


if __name__ == "__main__":
    main()

Run it like this:

Bash
python make_debug_timelapse.py \
  --debug-root runs/online_word_scribe_v3 \
  --out-dir runs/online_word_scribe_v3/timelapse \
  --glob "debug_epoch_*" \
  --gif-ms 700 \
  --max-samples 12 \
  --contact-cols 3

What it produces:

sample_gifs/<record_id>.gif → one animated GIF per sample across epochs

epoch_contact_sheets/debug_epoch_XXX.png → one contact sheet per epoch

epochs_contact_sheet.gif → an animated GIF of the contact sheets over epochs

summary.json → index of everything

If you want, I can also add a tiny helper that automatically runs this at the end of training whenever a new debug folder appears.

---

## User

sure

---

## ChatGPT

Add this helper script and call it right after each epoch’s debug folder is written.

auto_make_debug_timelapse.py
Python
Run
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Refresh timelapse artifacts from debug_epoch_* folders."
    )
    p.add_argument("--debug-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--glob", default="debug_epoch_*")
    p.add_argument("--gif-ms", type=int, default=700)
    p.add_argument("--max-samples", type=int, default=12)
    p.add_argument("--contact-cols", type=int, default=3)
    p.add_argument("--thumb-width", type=int, default=420)
    p.add_argument("--require-all-epochs", action="store_true")
    p.add_argument(
        "--timelapse-script",
        default="make_debug_timelapse.py",
        help="Path to make_debug_timelapse.py",
    )
    return p.parse_args()


def main():
    args = parse_args()

    debug_root = Path(args.debug_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(args.timelapse_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find timelapse script: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--debug-root",
        str(debug_root),
        "--out-dir",
        str(out_dir),
        "--glob",
        args.glob,
        "--gif-ms",
        str(args.gif_ms),
        "--max-samples",
        str(args.max_samples),
        "--contact-cols",
        str(args.contact_cols),
        "--thumb-width",
        str(args.thumb_width),
    ]

    if args.require_all_epochs:
        cmd.append("--require-all-epochs")

    print("Refreshing timelapse artifacts...")
    subprocess.run(cmd, check=True)
    print(f"Done. Timelapse artifacts updated at: {out_dir}")


if __name__ == "__main__":
    main()
Hook it into train_online_scribe.py

Right after the block where you run debug_render_online_scribe.py, add this second subprocess call:

Python
Run
import subprocess
import sys

# 1) render per-epoch debug folder
debug_dir = out_dir / f"debug_epoch_{epoch:03d}"
subprocess.run(
    [
        sys.executable,
        "debug_render_online_scribe.py",
        "--checkpoint",
        str(ckpt_path if not is_best else (out_dir / "best.pt")),
        "--jsonl",
        args.val_jsonl,
        "--image-root",
        args.image_root,
        "--out-dir",
        str(debug_dir),
        "--num-samples",
        "6",
        "--raster-size",
        str(args.raster_size),
        "--raster-sigma-px",
        str(args.raster_sigma_px),
        "--raster-mode",
        args.raster_mode,
    ],
    check=False,
)

# 2) refresh cross-epoch timelapse artifacts
subprocess.run(
    [
        sys.executable,
        "auto_make_debug_timelapse.py",
        "--debug-root",
        str(out_dir),
        "--out-dir",
        str(out_dir / "timelapse"),
        "--glob",
        "debug_epoch_*",
        "--gif-ms",
        "700",
        "--max-samples",
        "12",
        "--contact-cols",
        "3",
        "--thumb-width",
        "420",
        "--timelapse-script",
        "make_debug_timelapse.py",
    ],
    check=False,
)
Folder layout this assumes
Plain text
runs/online_word_scribe_v3/
  debug_epoch_001/
  debug_epoch_002/
  ...
  timelapse/
  debug_render_online_scribe.py
  make_debug_timelapse.py
  auto_make_debug_timelapse.py
What happens now

After every epoch:

the trainer writes debug_epoch_XXX/

the helper rescans all debug_epoch_* folders

it rebuilds:

per-sample GIFs

per-epoch contact sheets

the global epochs_contact_sheet.gif

So you always have an up-to-date timelapse without running a second manual command.

If you want the cleaner version next, I can fold both subprocess blocks into a single maybe_refresh_debug_artifacts(...) helper function for the trainer.

---

## User

Give me our entire conversation as markdown.

---

## ChatGPT

I’m packaging the visible conversation from this chat into a markdown transcript file so you can download it directly.

---

## User

Give me our entire conversation as markdown.

---

## ChatGPT

I’m generating the transcript as a markdown file now.

---

## User

Don't summarize anything! Give me the full transcript

---

## ChatGPT

I’m assembling the conversation into a markdown transcript file now, preserving the visible messages verbatim where I can.

---

## ChatGPT

Here it is: handwriting_conversation_transcript.md

---
