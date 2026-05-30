#!/usr/bin/env bash
set -u
# Extended/upper-bound inflation runs: 25 epochs each, scales 2.0, 3.0,
# and a no-mask baseline at scale=10.0 (every bbox clips to full canvas,
# so the mask is all-ones).

SCALES=(2.00 3.00 10.00)
BASE=runs/letter_bboxes_v2.jsonl
WORDS=data/iam_words/iam_words/words

for S in "${SCALES[@]}"; do
    TAG=${S/./p}
    INF=runs/letter_bboxes_v2_inflate${TAG}.jsonl
    OUT=runs/bbox_inflate_${TAG}_25ep
    LOG=bbox_inflate_${TAG}_25ep.log

    echo "=== Scale ${S} (25 epochs) ==="
    python3 inflate_bboxes.py --in $BASE --out $INF --scale $S
    python3 -u train_bbox_refiner.py \
        --bbox-jsonl $INF \
        --words-dir $WORDS \
        --out-dir $OUT \
        --phase1-epochs 25 \
        --phase2-epochs 0 \
        --phase1-target-acc 1.01 \
        --num-workers 4 \
        > $LOG 2>&1
    BEST=$(grep "Best phase1" $LOG | awk '{print $NF}')
    echo "SWEEP_RESULT scale=${S} best=${BEST}"
done
echo "=== Sweep done ==="
