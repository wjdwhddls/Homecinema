"""LOMO orchestrator for the AST ablation.

Delegates to ``model_base.run_lomo.main`` with this module's ``run_train.main``
injected. Per-fold output structure, gate evaluation, and report generation
are identical to the baseline — only the model / config classes differ.

Usage:
    python -m model.autoEQ.train_pseudo.model_ast.run_lomo \\
        --feature_dir data/features/ccmovies_ast --split_name ccmovies \\
        --movie_set ccmovies --epochs 30 \\
        --output_dir runs/ablation_ast_lomo9
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
