"""Phase 4-A Step 4 — COGNIMUSE OOD inference.

run_test_eval.py 의 fork. 변경점:
    1. feature file / metadata csv 를 COGNIMUSE 로 (CLI override 또는 default)
    2. split 처리: COGNIMUSE 는 모두 "test" 이므로 official_split 대신 직접 로드
    3. **Raw CCC + Per-film z-score CCC 두 전략 병행 보고**
    4. Per-film 11-metric breakdown 추가

BASE weights는 read-only. 출력은 runs/cognimuse/phase4a/ood_eval/ 하위.

Usage:
    venv/bin/python -m model.autoEQ.train_liris.model_cognimuse_ood.run_eval_ood \
        --seeds 42 123 2024 \
        --ckpt-pattern "runs/phase2a/2a2_A_K7_s{seed}/best.pt" \
        --feature-file data/features/cognimuse_panns_v5spec/features.pt \
        --metadata-csv dataset/autoEQ/cognimuse/cognimuse_metadata.csv \
        --output-dir runs/cognimuse/phase4a/ood_eval
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.autoEQ.train.utils import compute_mean_ccc, compute_va_regression_metrics
from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.dataset import (
    MixupTargetShrinkageCollator,
    PrecomputedLirisDataset,
)
from model.autoEQ.train_liris.model import AutoEQModelLiris

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


def zscore_per_film(
    preds: torch.Tensor,
    tgts: torch.Tensor,
    film_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """영화별로 pred/target 을 독립 z-normalize.

    Scale/shift bias를 제거하고 representation 의 "상대적 순서/변동" 품질만 측정.
    """
    import numpy as np

    p = preds.numpy().copy()
    t = tgts.numpy().copy()
    arr = np.array(film_ids)
    for fid in np.unique(arr):
        mask = arr == fid
        for ax in (0, 1):
            p_mu, p_sd = p[mask, ax].mean(), p[mask, ax].std()
            t_mu, t_sd = t[mask, ax].mean(), t[mask, ax].std()
            p[mask, ax] = (p[mask, ax] - p_mu) / max(p_sd, 1e-8)
            t[mask, ax] = (t[mask, ax] - t_mu) / max(t_sd, 1e-8)
    return torch.from_numpy(p).float(), torch.from_numpy(t).float()


def forward_split(
    model: AutoEQModelLiris, loader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    preds, tgts, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["visual"], batch["audio"])
            preds.append(out["va_pred"].cpu())
            tgts.append(batch["va_target"].cpu())
            names.extend(batch["name"])
    return torch.cat(preds, dim=0), torch.cat(tgts, dim=0), names


def agg_stats(vals: list[float]) -> dict:
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
    return {"mean": m, "std": math.sqrt(var), "per_seed": list(vals)}


def per_film_breakdown(
    preds: torch.Tensor,
    tgts: torch.Tensor,
    film_ids: list[int],
    film_names: list[str],
) -> list[dict]:
    import numpy as np

    arr = np.array(film_ids)
    names_arr = np.array(film_names)
    out = []
    for fid in np.unique(arr):
        mask = arr == fid
        fname = names_arr[mask][0]
        if mask.sum() < 5:
            continue
        p = preds[torch.from_numpy(mask)]
        t = tgts[torch.from_numpy(mask)]
        metrics = compute_11_metrics(p, t)
        out.append({"film_id": int(fid), "film_name": str(fname), "n_windows": int(mask.sum()), **metrics})
    return out


def run_eval(
    features: dict,
    metadata: pd.DataFrame,
    cfg: TrainLirisConfig,
    seeds: list[int],
    ckpt_pattern: str,
) -> dict:
    assert all(metadata["split"] == "test"), "COGNIMUSE metadata should be all 'test'"
    ds = PrecomputedLirisDataset(metadata, features)
    eval_collate = MixupTargetShrinkageCollator(cfg, active=False)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=eval_collate,
    )

    film_ids = metadata["film_id"].tolist()
    film_names = (
        metadata["film_name"].tolist() if "film_name" in metadata.columns
        else metadata["movie_code"].tolist()
    )

    per_seed: dict[int, dict] = {}
    per_seed_zscore: dict[int, dict] = {}
    per_seed_per_film: dict[int, list[dict]] = {}
    seed_preds: dict[int, torch.Tensor] = {}
    tgts_ref: torch.Tensor | None = None

    for seed in seeds:
        ckpt_path = ckpt_pattern.format(seed=seed)
        print(f"\n[seed {seed}] loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = AutoEQModelLiris(cfg)
        model.load_state_dict(ckpt["model"])

        preds, tgts, _names = forward_split(model, loader)
        if tgts_ref is None:
            tgts_ref = tgts
        else:
            assert torch.equal(tgts_ref, tgts), "target tensor drifted across seeds"

        raw_metrics = compute_11_metrics(preds, tgts)
        p_z, t_z = zscore_per_film(preds, tgts, film_ids)
        zscore_metrics = compute_11_metrics(p_z, t_z)

        per_seed[seed] = raw_metrics
        per_seed_zscore[seed] = zscore_metrics
        per_seed_per_film[seed] = per_film_breakdown(preds, tgts, film_ids, film_names)
        seed_preds[seed] = preds
        print(
            f"[seed {seed}] RAW mean_CCC={raw_metrics['mean_ccc']:+.4f} "
            f"(V={raw_metrics['ccc_v']:+.4f}, A={raw_metrics['ccc_a']:+.4f}) | "
            f"Z-SCORE mean_CCC={zscore_metrics['mean_ccc']:+.4f} "
            f"(V={zscore_metrics['ccc_v']:+.4f}, A={zscore_metrics['ccc_a']:+.4f})"
        )

    agg_raw = {k: agg_stats([per_seed[s][k] for s in seeds]) for k in METRIC_KEYS}
    agg_z = {k: agg_stats([per_seed_zscore[s][k] for s in seeds]) for k in METRIC_KEYS}

    # Ensemble: 3-seed va_pred average → 11-metric (raw + z-score)
    stacked = torch.stack([seed_preds[s] for s in seeds], dim=0)
    ensemble_pred = stacked.mean(dim=0)
    ens_raw = compute_11_metrics(ensemble_pred, tgts_ref)
    p_z, t_z = zscore_per_film(ensemble_pred, tgts_ref, film_ids)
    ens_z = compute_11_metrics(p_z, t_z)
    ens_per_film = per_film_breakdown(ensemble_pred, tgts_ref, film_ids, film_names)

    print(
        f"\n[ensemble] RAW mean_CCC={ens_raw['mean_ccc']:+.4f} "
        f"(V={ens_raw['ccc_v']:+.4f}, A={ens_raw['ccc_a']:+.4f}) | "
        f"Z-SCORE mean_CCC={ens_z['mean_ccc']:+.4f} "
        f"(V={ens_z['ccc_v']:+.4f}, A={ens_z['ccc_a']:+.4f})"
    )

    return {
        "seeds": seeds,
        "per_seed_raw": {str(s): per_seed[s] for s in seeds},
        "per_seed_zscore": {str(s): per_seed_zscore[s] for s in seeds},
        "per_seed_per_film": {str(s): per_seed_per_film[s] for s in seeds},
        "aggregate_raw": agg_raw,
        "aggregate_zscore": agg_z,
        "ensemble_raw": ens_raw,
        "ensemble_zscore": ens_z,
        "ensemble_per_film": ens_per_film,
    }


def load_liris_baseline() -> dict | None:
    """BASE 결과 (val / test) 를 리포트용으로 로드."""
    out: dict = {}
    test_path = Path("runs/phase3/test_final_metrics.json")
    if test_path.is_file():
        out["test"] = json.loads(test_path.read_text())
    return out or None


def write_report(output_dir: Path, results: dict, liris_baseline: dict | None) -> None:
    """사람이 읽는 report.md 작성."""
    lines: list[str] = []
    lines.append("# Phase 4-A — COGNIMUSE OOD Generalization Report\n")
    lines.append("**BASE weights**: `runs/phase2a/2a2_A_K7_s{42,123,2024}/best.pt` (read-only)")
    lines.append("**Metadata**: 2,197 windows × 12 Hollywood films, experienced median consensus\n")

    agg_raw = results["aggregate_raw"]
    agg_z = results["aggregate_zscore"]
    ens_raw = results["ensemble_raw"]
    ens_z = results["ensemble_zscore"]

    lines.append("## 1. Headline Numbers\n")
    lines.append("| Dataset | Split | n_clips | mean CCC | CCC_V | CCC_A |")
    lines.append("|---|---|---|---|---|---|")
    if liris_baseline and "test" in liris_baseline:
        tt = liris_baseline["test"]
        agg = tt["aggregate_11_metric"]
        ens = tt["ensemble_11_metric"]
        lines.append(
            f"| LIRIS | test (Phase 3, 3-seed agg) | {tt['split']['n_clips']} | "
            f"{agg['mean_ccc']['mean']:+.4f} ± {agg['mean_ccc']['std']:.4f} | "
            f"{agg['ccc_v']['mean']:+.4f} | {agg['ccc_a']['mean']:+.4f} |"
        )
        lines.append(
            f"| LIRIS | test ensemble | {tt['split']['n_clips']} | "
            f"{ens['mean_ccc']:+.4f} | {ens['ccc_v']:+.4f} | {ens['ccc_a']:+.4f} |"
        )
    n_clips = sum(results["per_seed_per_film"][str(results['seeds'][0])][i]["n_windows"]
                  for i in range(len(results["per_seed_per_film"][str(results['seeds'][0])])))
    lines.append(
        f"| COGNIMUSE | 12 films **RAW** (3-seed agg) | {n_clips} | "
        f"{agg_raw['mean_ccc']['mean']:+.4f} ± {agg_raw['mean_ccc']['std']:.4f} | "
        f"{agg_raw['ccc_v']['mean']:+.4f} | {agg_raw['ccc_a']['mean']:+.4f} |"
    )
    lines.append(
        f"| COGNIMUSE | 12 films **RAW** ensemble | {n_clips} | "
        f"{ens_raw['mean_ccc']:+.4f} | {ens_raw['ccc_v']:+.4f} | {ens_raw['ccc_a']:+.4f} |"
    )
    lines.append(
        f"| COGNIMUSE | 12 films **z-score** (3-seed agg) | {n_clips} | "
        f"{agg_z['mean_ccc']['mean']:+.4f} ± {agg_z['mean_ccc']['std']:.4f} | "
        f"{agg_z['ccc_v']['mean']:+.4f} | {agg_z['ccc_a']['mean']:+.4f} |"
    )
    lines.append(
        f"| COGNIMUSE | 12 films **z-score** ensemble | {n_clips} | "
        f"{ens_z['mean_ccc']:+.4f} | {ens_z['ccc_v']:+.4f} | {ens_z['ccc_a']:+.4f} |"
    )

    lines.append("\n## 2. Interpretation Guide\n")
    lines.append("| z-score CCC | 해석 | 후속 권장 |")
    lines.append("|---|---|---|")
    lines.append("| ≥ 0.30 | 강한 일반화 | Track B 선택사항 |")
    lines.append("| 0.15–0.30 | 부분 일반화 (표준) | Track B 권장 |")
    lines.append("| 0.05–0.15 | 약한 일반화 | Track B 필수 |")
    lines.append("| < 0.05 | 일반화 실패 | 아키텍처 재검토 |")

    zscore_mean = agg_z["mean_ccc"]["mean"]
    if zscore_mean >= 0.30:
        interp = "**강한 일반화** — Track B 선택사항"
    elif zscore_mean >= 0.15:
        interp = "**부분 일반화 (표준)** — Track B 권장"
    elif zscore_mean >= 0.05:
        interp = "**약한 일반화** — Track B 필수"
    else:
        interp = "**일반화 실패** — 아키텍처 재검토"
    lines.append(f"\n**현재 결과** (z-score mean CCC = {zscore_mean:+.4f}): {interp}\n")

    lines.append("## 3. Raw vs Z-score Delta\n")
    d_mean = agg_z["mean_ccc"]["mean"] - agg_raw["mean_ccc"]["mean"]
    d_v = agg_z["ccc_v"]["mean"] - agg_raw["ccc_v"]["mean"]
    d_a = agg_z["ccc_a"]["mean"] - agg_raw["ccc_a"]["mean"]
    lines.append(f"- Δ mean CCC (z − raw) = {d_mean:+.4f}")
    lines.append(f"- Δ CCC_V = {d_v:+.4f}, Δ CCC_A = {d_a:+.4f}\n")
    if d_mean > 0.05:
        lines.append(
            "- 해석: z-score > raw 크게 → **scale/shift mismatch 가 raw 점수를 끌어내림**.\n"
            "  예측의 방향성 (순서) 은 raw 점수가 시사하는 것보다 더 잘 보존됨.\n"
        )
    elif d_mean < -0.02:
        lines.append(
            "- 해석: z-score < raw → per-film 표준화가 오히려 예측과 타깃의 공통 scale bias 를 제거하면서\n"
            "  CCC 가 낮아짐. raw 가 더 맞는 지표.\n"
        )
    else:
        lines.append(
            "- 해석: raw ≈ z-score → scale 은 잘 맞추고 있음. 남은 차이는 representation 한계.\n"
        )

    lines.append("## 4. Per-Seed Breakdown\n")
    lines.append("### Raw\n")
    lines.append("| seed | mean CCC | CCC_V | CCC_A | Pearson | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for seed_str, m in results["per_seed_raw"].items():
        lines.append(
            f"| {seed_str} | {m['mean_ccc']:+.4f} | {m['ccc_v']:+.4f} | "
            f"{m['ccc_a']:+.4f} | {m['mean_pearson']:+.4f} | {m['mean_mae']:.4f} |"
        )
    lines.append("\n### Per-film z-score\n")
    lines.append("| seed | mean CCC | CCC_V | CCC_A | Pearson | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for seed_str, m in results["per_seed_zscore"].items():
        lines.append(
            f"| {seed_str} | {m['mean_ccc']:+.4f} | {m['ccc_v']:+.4f} | "
            f"{m['ccc_a']:+.4f} | {m['mean_pearson']:+.4f} | {m['mean_mae']:.4f} |"
        )

    lines.append("\n## 5. Per-Film Breakdown (Ensemble, Raw)\n")
    lines.append("| film | n | mean CCC | CCC_V | CCC_A | Pearson | MAE |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in sorted(results["ensemble_per_film"], key=lambda r: -r["mean_ccc"]):
        lines.append(
            f"| {p['film_name']} | {p['n_windows']} | "
            f"{p['mean_ccc']:+.4f} | {p['ccc_v']:+.4f} | {p['ccc_a']:+.4f} | "
            f"{p['mean_pearson']:+.4f} | {p['mean_mae']:.4f} |"
        )

    lines.append("\n## 6. LIRIS Val/Test Baseline (참고)\n")
    if liris_baseline and "test" in liris_baseline:
        t = liris_baseline["test"]
        agg = t["aggregate_11_metric"]
        ens = t["ensemble_11_metric"]
        lines.append("| metric | val (Phase 2a-2) | test (Phase 3 agg) | test ensemble |")
        lines.append("|---|---|---|---|")
        lines.append(f"| mean CCC | 0.3812 ± 0.024 | {agg['mean_ccc']['mean']:+.4f} | {ens['mean_ccc']:+.4f} |")
        lines.append(f"| CCC_V | 0.3623 | {agg['ccc_v']['mean']:+.4f} | {ens['ccc_v']:+.4f} |")
        lines.append(f"| CCC_A | 0.4001 | {agg['ccc_a']['mean']:+.4f} | {ens['ccc_a']:+.4f} |")

    lines.append("\n---\n")
    lines.append("*Phase 4-A OOD evaluation · BASE 3-seed ensemble · read-only*")

    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--ckpt-pattern", default="runs/phase2a/2a2_A_K7_s{seed}/best.pt")
    p.add_argument(
        "--feature-file",
        default="data/features/cognimuse_panns_v5spec/features.pt",
    )
    p.add_argument(
        "--metadata-csv",
        default="dataset/autoEQ/cognimuse/cognimuse_metadata.csv",
    )
    p.add_argument(
        "--output-dir",
        default="runs/cognimuse/phase4a/ood_eval",
    )
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    # Output dir 보호
    if not args.output_dir.startswith("runs/cognimuse/"):
        raise ValueError(
            f"output-dir must start with 'runs/cognimuse/' (BASE 자산 보호), got {args.output_dir}"
        )

    cfg = TrainLirisConfig()
    cfg.feature_file = args.feature_file
    cfg.metadata_csv = args.metadata_csv
    cfg.batch_size = args.batch_size
    # COGNIMUSE는 VA_norm IDENTITY 로 CSV 에 이미 저장됨 → va_norm_strategy 는 의미 없음
    # (official_split 을 호출하지 않기 때문)

    print(f"[cfg] feature_file = {cfg.feature_file}")
    print(f"[cfg] metadata_csv = {cfg.metadata_csv}")
    print(f"[cfg] seeds        = {args.seeds}")
    print(f"[cfg] ckpt_pattern = {args.ckpt_pattern}")

    features = torch.load(cfg.feature_file, map_location="cpu", weights_only=False)
    metadata = pd.read_csv(cfg.metadata_csv)
    print(f"[data] features: {len(features)} windows")
    print(f"[data] metadata: {len(metadata)} rows, {metadata['film_id'].nunique()} films")

    # Feature missing 검사
    missing = [n for n in metadata["name"] if n not in features]
    if missing:
        print(f"[warn] {len(missing)} windows have no feature; dropping "
              f"(first 5: {missing[:5]})")
        metadata = metadata[metadata["name"].isin(features)].reset_index(drop=True)

    results = run_eval(features, metadata, cfg, args.seeds, args.ckpt_pattern)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Primary JSON
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out_dir}/results.json")

    # Human-readable report
    liris_baseline = load_liris_baseline()
    write_report(out_dir, results, liris_baseline)
    print(f"[done] wrote {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
