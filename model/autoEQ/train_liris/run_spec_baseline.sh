#!/usr/bin/env bash
# V5-FINAL §11 Phase 2a-0 Baseline — spec-compliant defaults, 3 seeds.
#
# All config comes from TrainLirisConfig defaults (restored to §9-1 line 237).
# Only --run-name and --seed are overridden.

set -e
cd "$(dirname "$0")/../../.."

PY=venv/bin/python

for seed in 42 123 2024; do
  echo ""
  echo "=== spec_baseline_s${seed}  (V5-FINAL §11 defaults) ==="
  $PY -m model.autoEQ.train_liris.run_train \
    --run-name "spec_baseline_s${seed}" \
    --seed "$seed"
done

echo ""
echo "=== 3-seed spec baseline done — run aggregate_spec_baseline.py ==="
