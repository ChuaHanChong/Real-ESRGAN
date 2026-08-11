#!/usr/bin/env bash
# Run the full Real-ESRGAN degradation pipeline over the maritime IR dataset.
set -euo pipefail

PY=/data/hanchong/miniconda3/envs/dinov2/bin/python
INPUT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein/In-distribution_100perCat/train
OUTPUT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein/In-distribution_100perCat_degraded/train
LOG=/tmp/degrade_full.log

# Defaults (scale=4, use_usm_on_gt=True) match finetune_realesrgan_x4plus.yml exactly.
"$PY" "$(dirname "$0")/degrade_dataset.py" \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --use-usm \
    --scale 4 \
    --seed  0 2>&1 | tee "$LOG"
