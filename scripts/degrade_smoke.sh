#!/usr/bin/env bash
# Smoke-test the degradation pipeline on the first 6 images.
set -euo pipefail

PY=/data/hanchong/miniconda3/envs/dinov2/bin/python
INPUT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein/In-distribution_100perCat/train
OUTPUT=/tmp/degrade_smoke

"$PY" "$(dirname "$0")/degrade_dataset.py" \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --scale 1 \
    --seed  0 \
    --limit 6
