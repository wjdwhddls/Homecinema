"""Single-fold training entrypoint for the CLIP frame-mean ablation.

Thin wrapper over ``model_base.run_train`` that only swaps the model and
config classes. All CLI flags, augmentation logic, split handling, and
fold-mapping artefacts are identical to the baseline.

Usage:
    python -m model.autoEQ.train_pseudo.model_clip_framemean.run_train \\
        --feature_dir data/features/ccmovies_clipimg --split_name ccmovies \\
        --movie_set ccmovies --lomo_fold 0 --epochs 40
"""

from __future__ import annotations

import json

from ..model_base import run_train as _base_run_train
from .config import TrainCogConfigClipFrameMean
from .model import AutoEQModelClipFrameMean


def main(argv: list[str] | None = None) -> dict:
    return _base_run_train.main(
        argv,
        model_cls=AutoEQModelClipFrameMean,
        config_cls=TrainCogConfigClipFrameMean,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
