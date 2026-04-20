"""Single-fold training entrypoint for the simple-concat ablation.

Thin wrapper over ``model_base.run_train`` that only swaps the model and
config classes. All CLI flags, augmentation logic, split handling, and
fold-mapping artefacts are identical to the baseline.
"""

from __future__ import annotations

import json

from ..model_base import run_train as _base_run_train
from .config import TrainCogConfigConcat
from .model import AutoEQModelConcat


def main(argv: list[str] | None = None) -> dict:
    return _base_run_train.main(
        argv,
        model_cls=AutoEQModelConcat,
        config_cls=TrainCogConfigConcat,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
