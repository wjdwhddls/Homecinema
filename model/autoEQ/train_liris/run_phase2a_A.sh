#!/usr/bin/env bash
# Phase 2a-A: Regularization 강화 sweep
#   a1: head_dropout 0.5, weight_decay 1e-4 (V2와 head만 ↑)
#   a2: head_dropout 0.3, weight_decay 1e-3 (V2와 wd만 ↑)
#   a3: head_dropout 0.5, weight_decay 1e-3 (둘 다 ↑)
# seeds: 42 123 2024

set -e
cd "$(dirname "$0")/../../.."

PY=venv/bin/python

run() {
  local name=$1 seed=$2 hd=$3 wd=$4
  echo ""
  echo "=== $name  seed=$seed  head_dropout=$hd  weight_decay=$wd ==="
  $PY -m model.autoEQ.train_liris.run_train \
    --run-name "$name" --seed "$seed" \
    --head-dropout "$hd" --weight-decay "$wd"
}

for seed in 42 123 2024; do
  run "A_a1_s${seed}" "$seed" 0.5 1e-4
  run "A_a2_s${seed}" "$seed" 0.3 1e-3
  run "A_a3_s${seed}" "$seed" 0.5 1e-3
done

echo ""
echo "=== all 9 runs done ==="
