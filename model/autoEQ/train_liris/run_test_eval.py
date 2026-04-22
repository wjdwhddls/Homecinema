"""Phase 3 — LIRIS Test Set Final Evaluation.

V5-FINAL §14-3 test access gate: unlocked only after Phase 2a completion.
This script runs the frozen Base Model (Phase 2a-7 rev.) against the 80-film
test split (4,900 clips) for the **final** generalization report.

Output:
    - per-seed 11-metric (compare across seeds 42/123/2024)
    - 3-seed aggregate (mean ± std)
    - 3-seed ensemble (va_pred averaged across seeds → metric)
    - val-test delta (generalization audit vs BASE_MODEL.md §4)

Runs `runs/phase2a/2a2_A_K7_s{seed}/best.pt` in eval mode using the same
evaluation pipeline (PrecomputedLirisDataset + MixupTargetShrinkageCollator
with active=False) Phase 2a used for val — guarantees comparability.

Usage:
    venv/bin/python -m model.autoEQ.train_liris.run_test_eval \
        --seeds 42 123 2024 \
        --output runs/phase3/test_final_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..train.utils import compute_mean_ccc, compute_va_regression_metrics
from .config import TrainLirisConfig
from .dataset import (
    MixupTargetShrinkageCollator,
    PrecomputedLirisDataset,
    official_split,
)
from .model import AutoEQModelLiris

METRIC_KEYS = [
    "mean_ccc", "ccc_v", "ccc_a",
    "mean_pearson", "pearson_valence", "pearson_arousal",
    "mean_mae", "mae_valence", "mae_arousal",
    "rmse_valence", "rmse_arousal",
]


def compute_11_metrics(preds: torch.Tensor, tgts: torch.Tensor) -> dict[str, float]:
    mean_ccc, ccc_v, ccc_a = compute_mean_ccc(preds, tgts)
    extra = compute_va_regression_metrics(preds, tgts)
    out = {
        "mean_ccc": float(mean_ccc.item()),
        "ccc_v": float(ccc_v.item()),
        "ccc_a": float(ccc_a.item()),
        "mean_pearson": (extra["pearson_valence"] + extra["pearson_arousal"]) / 2.0,
        "mean_mae": (extra["mae_valence"] + extra["mae_arousal"]) / 2.0,
        **{k: float(v) for k, v in extra.items()},
    }
    return {k: out[k] for k in METRIC_KEYS}


def forward_split(model: AutoEQModelLiris, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["visual"], batch["audio"])
            preds.append(out["va_pred"].cpu())
            tgts.append(batch["va_target"].cpu())
    return torch.cat(preds, dim=0), torch.cat(tgts, dim=0)


def agg_stats(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
    return {"mean": m, "std": math.sqrt(var), "per_seed": list(vals)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--ckpt-pattern", default="runs/phase2a/2a2_A_K7_s{seed}/best.pt")
    p.add_argument("--output", default="runs/phase3/test_final_metrics.json")
    p.add_argument("--feature-file", default=None)
    p.add_argument("--metadata-csv", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    cfg = TrainLirisConfig()
    if args.feature_file:
        cfg.feature_file = args.feature_file
    if args.metadata_csv:
        cfg.metadata_csv = args.metadata_csv
    if args.batch_size:
        cfg.batch_size = args.batch_size

    print(f"[cfg] feature_file={cfg.feature_file}")
    print(f"[cfg] metadata_csv={cfg.metadata_csv}")
    print(f"[cfg] va_norm={cfg.va_norm_strategy}  K={cfg.num_mood_classes}  bs={cfg.batch_size}")

    features = torch.load(cfg.feature_file, map_location="cpu", weights_only=False)
    splits = official_split(
        Path(cfg.metadata_csv),
        use_full_learning_set=cfg.use_full_learning_set,
        va_norm_strategy=cfg.va_norm_strategy,
    )
    test_df = splits["test"]
    print(f"[split] test: {len(test_df)} clips / {test_df['film_id'].nunique()} films")
    # Sanity: test must be disjoint from train+val
    learn_films = set(splits["train"]["film_id"]) | set(splits["val"]["film_id"])
    overlap = set(test_df["film_id"]) & learn_films
    assert not overlap, f"film-level overlap with learning set: {overlap}"

    ds_test = PrecomputedLirisDataset(test_df, features)
    eval_collate = MixupTargetShrinkageCollator(cfg, active=False)
    loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, collate_fn=eval_collate)

    # --- Per-seed forward ---
    per_seed: dict[int, dict] = {}
    seed_preds: dict[int, torch.Tensor] = {}
    tgts_ref: torch.Tensor | None = None
    for seed in args.seeds:
        ckpt_path = args.ckpt_pattern.format(seed=seed)
        print(f"\n[seed {seed}] loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = AutoEQModelLiris(cfg)
        model.load_state_dict(ckpt["model"])

        preds, tgts = forward_split(model, loader)
        if tgts_ref is None:
            tgts_ref = tgts
        else:
            # Target tensor must match across seeds (same data loader config)
            assert torch.equal(tgts_ref, tgts), "target tensor drifted across seeds"

        metrics = compute_11_metrics(preds, tgts)
        per_seed[seed] = metrics
        seed_preds[seed] = preds
        print(f"[seed {seed}] test mean_CCC = {metrics['mean_ccc']:+.4f}  "
              f"(V={metrics['ccc_v']:+.4f}, A={metrics['ccc_a']:+.4f})")

    # --- 3-seed aggregate (mean ± std per metric) ---
    aggregate = {
        k: agg_stats([per_seed[s][k] for s in args.seeds])
        for k in METRIC_KEYS
    }

    # --- 3-seed ensemble (average predictions) ---
    stacked = torch.stack([seed_preds[s] for s in args.seeds], dim=0)  # (S, N, 2)
    ensemble_pred = stacked.mean(dim=0)                                 # (N, 2)
    ensemble_metrics = compute_11_metrics(ensemble_pred, tgts_ref)
    print(f"\n[ensemble] test mean_CCC = {ensemble_metrics['mean_ccc']:+.4f}  "
          f"(V={ensemble_metrics['ccc_v']:+.4f}, A={ensemble_metrics['ccc_a']:+.4f})")

    # --- val-test comparison (from Phase 2a-2 summary.json) ---
    val_per_seed = {}
    for seed in args.seeds:
        summary_path = Path(f"runs/phase2a/2a2_A_K7_s{seed}/summary.json")
        if summary_path.exists():
            val_per_seed[seed] = json.loads(summary_path.read_text())["best_val"]
    val_mean_ccc = [val_per_seed[s]["mean_ccc"] for s in args.seeds if s in val_per_seed]
    val_agg = agg_stats(val_mean_ccc) if val_mean_ccc else None

    # --- Write JSON ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "phase": "3 — LIRIS test final evaluation",
        "split": {
            "n_clips": len(test_df),
            "n_films": int(test_df["film_id"].nunique()),
            "disjoint_from_learning_set": True,
        },
        "seeds": list(args.seeds),
        "ckpt_pattern": args.ckpt_pattern,
        "per_seed": {str(s): per_seed[s] for s in args.seeds},
        "aggregate_11_metric": aggregate,
        "ensemble_11_metric": ensemble_metrics,
        "val_comparison": {
            "val_mean_ccc_per_seed": {str(s): v for s, v in val_per_seed.items()},
            "val_aggregate_mean_ccc": val_agg,
            "test_aggregate_mean_ccc": aggregate["mean_ccc"],
            "val_test_delta_mean": (
                aggregate["mean_ccc"]["mean"] - val_agg["mean"] if val_agg else None
            ),
        },
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
