"""Single-fold training entrypoint for the AST ablation.

Thin wrapper over ``model_base.run_train`` that only swaps the model and
config classes. All CLI flags, augmentation logic, split handling, and
fold-mapping artefacts are identical to the baseline.

Usage:
    python -m model.autoEQ.train_pseudo.model_ast.run_train \\
        --feature_dir data/features/ccmovies_ast --split_name ccmovies \\
        --movie_set ccmovies --lomo_fold 0 --epochs 30
"""

from __future__ import annotations

import json

from ..model_base import run_train as _base_run_train
from .config import TrainCogConfigAST
from .model import AutoEQModelAST


def main(argv: list[str] | None = None) -> dict:
    return _base_run_train.main(
        argv,
        model_cls=AutoEQModelAST,
        config_cls=TrainCogConfigAST,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
