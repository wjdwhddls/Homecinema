"""LOMO orchestrator for the CLIP frame-mean ablation.

Delegates to ``model_base.run_lomo.main`` with this module's ``run_train.main``
injected. Per-fold output structure, gate evaluation, and report generation
are identical to the baseline — only the model / config classes differ.

Usage:
    python -m model.autoEQ.train_pseudo.model_clip_framemean.run_lomo \\
        --feature_dir data/features/ccmovies_clipimg --split_name ccmovies \\
        --movie_set ccmovies --epochs 40 \\
        --output_dir runs/ablation_clipimg_lomo9 \\
        --base_seed 42 \\
        --sigma_filter_threshold -1.0 \\
        --modality_dropout_p 0.05 --feature_noise_std 0.03 \\
        --mixup_prob 0.5 --mixup_alpha 0.4 \\
        --label_smooth_eps 0.05 --label_smooth_sigma_threshold 0.15 \\
        --lambda_mood 0.3
"""

from __future__ import annotations

import json

from ..model_base import run_lomo as _base_run_lomo
from . import run_train


def main(argv: list[str] | None = None) -> dict:
    return _base_run_lomo.main(argv, run_train_main=run_train.main)


if __name__ == "__main__":
    report = main()
    print(json.dumps(
        {"run_id": report["run_id"], "gate": report["gate"]},
        indent=2,
    ))
