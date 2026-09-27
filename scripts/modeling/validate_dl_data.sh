#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MRI_QRF
python run_dl_models_mri_qrf.py --validate-only --num-workers 4
