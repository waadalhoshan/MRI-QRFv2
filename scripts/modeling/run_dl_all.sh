#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MRI_QRF
mkdir -p logs
python -u run_dl_models_mri_qrf.py \
  --models SimpleCNN ResNet18 \
  --datasets OASIS_1 OASIS_2 \
  --seeds 13 47 101 \
  --num-workers 8 \
  2>&1 | tee logs/dl_training.log
