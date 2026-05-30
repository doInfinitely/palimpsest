#!/usr/bin/env bash
set -u
# Sweep of bbox inflation factors. Phase 1 only, 12 epochs each.
# Outputs each run's best val acc on its own line for easy parsing.

SCALES=(1.15 1.25 1.50 2.00)
BASE=runs/letter_bboxes_v2.jsonl
WORDS=data/iam_words/iam_words/words

for S in "${SCALES[@]}"; do
    TAG=${S/./p}
    INF=runs/letter_bboxes_v2_inflate${TAG}.jsonl
    OUT=runs/bbox_inflate_${TAG}
    LOG=bbox_inflate_${TAG}.log

    echo "=== Scale ${S} ==="
    python3 inflate_bboxes.py --in $BASE --out $INF --scale $S
    python3 -u train_bbox_refiner.py \
        --bbox-jsonl $INF \
        --words-dir $WORDS \
        --out-dir $OUT \
        --phase1-epochs 12 \
        --phase2-epochs 0 \
        --phase1-target-acc 1.01 \
        --num-workers 4 \
        > $LOG 2>&1
    BEST=$(grep "Best phase1" $LOG | awk '{print $NF}')
    echo "SWEEP_RESULT scale=${S} best=${BEST}"
done
echo "=== Sweep done ==="
