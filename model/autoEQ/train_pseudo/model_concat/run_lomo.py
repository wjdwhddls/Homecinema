"""LOMO orchestrator for the simple-concat ablation."""

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
