"""Single-fold training entrypoint for the GMU fusion ablation."""

from __future__ import annotations

import json

from ..model_base import run_train as _base_run_train
from .config import TrainCogConfigGMU
from .model import AutoEQModelGMU


def main(argv: list[str] | None = None) -> dict:
    return _base_run_train.main(
        argv,
        model_cls=AutoEQModelGMU,
        config_cls=TrainCogConfigGMU,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
