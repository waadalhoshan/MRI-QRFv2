#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MRI_QRF
mkdir -p logs
if ! command -v tmux >/dev/null 2>&1; then
  apt-get update && apt-get install -y tmux
fi
if tmux has-session -t mriqrf 2>/dev/null; then
  echo "Session mriqrf already exists. Attach with: tmux attach -t mriqrf"
  exit 0
fi
tmux new-session -d -s mriqrf 'cd /workspace/MRI_QRF && bash run_dl_all.sh'
echo "Started training in tmux session mriqrf"
echo "Attach: tmux attach -t mriqrf"
echo "Detach: Ctrl+B then D"
echo "Log: /workspace/MRI_QRF/logs/dl_training.log"
