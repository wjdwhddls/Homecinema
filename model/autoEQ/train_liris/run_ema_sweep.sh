#!/usr/bin/env bash
# EMA decay tuning: seed 42 × decay {0.99, 0.995, 0.998}  → pick best → 3-seed
#
# Rationale: ~38 steps/epoch × 20 epochs = ~760 total steps.
# decay=0.999 half-life=693 (too slow, EMA can't catch live model)
# decay=0.995 half-life=138
# decay=0.998 half-life=346

set -e
cd "$(dirname "$0")/../../.."

PY=venv/bin/python

run() {
  local name=$1 seed=$2 decay=$3
  echo ""
  echo "=== $name  seed=$seed  ema_decay=$decay ==="
  $PY -m model.autoEQ.train_liris.run_train \
    --run-name "$name" --seed "$seed" \
    --use-ema --ema-decay "$decay"
}

# Phase 1 — pick best decay on seed 42
run "EMA_d99_s42"  42 0.99
run "EMA_d995_s42" 42 0.995
run "EMA_d998_s42" 42 0.998

echo ""
echo "=== decay pick done — run aggregator to see winner ==="
