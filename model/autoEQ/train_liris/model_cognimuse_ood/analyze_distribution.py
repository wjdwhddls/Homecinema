"""Phase 4-A Phase 0 Gate — LIRIS vs COGNIMUSE label distribution 비교.

Track A OOD eval 결과(Raw CCC vs per-film z-score CCC 차이)를 해석할
사전 근거를 생성한다.

출력:
    runs/cognimuse/phase4a/phase0/
      distribution_report.json   — 수치 요약
      va_histogram.png           — 2D scatter overlay
      mood_k7_distribution.json  — K=7 class 비율
      scale_shift_summary.md     — 사람이 읽는 요약

Gate 조건 (통과해야 Step 3 precompute 진입):
    1. COGNIMUSE v_norm, a_norm ∈ [-1, +1] 모두 만족
    2. Windows ≥ 1500 (분석 통계력)
    3. mood_k7 class 중 ≥ 4개가 ≥ 1% 비율
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MOOD_NAMES = [
    "Tension", "Sadness", "Peacefulness", "JoyfulActivation",
    "Tenderness", "Power", "Wonder",
]


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    summary = {
        "label": label,
        "n_clips": n,
        "n_films": int(df["film_id"].nunique()) if "film_id" in df.columns else None,
        "v_norm": {
            "mean": float(df["v_norm"].mean()),
            "std": float(df["v_norm"].std()),
            "min": float(df["v_norm"].min()),
            "max": float(df["v_norm"].max()),
            "p25": float(df["v_norm"].quantile(0.25)),
            "p50": float(df["v_norm"].quantile(0.50)),
            "p75": float(df["v_norm"].quantile(0.75)),
        },
        "a_norm": {
            "mean": float(df["a_norm"].mean()),
            "std": float(df["a_norm"].std()),
            "min": float(df["a_norm"].min()),
            "max": float(df["a_norm"].max()),
            "p25": float(df["a_norm"].quantile(0.25)),
            "p50": float(df["a_norm"].quantile(0.50)),
            "p75": float(df["a_norm"].quantile(0.75)),
        },
        "va_correlation": float(df[["v_norm", "a_norm"]].corr().iloc[0, 1]),
    }
    if "v_var" in df.columns:
        summary["v_var"] = {
            "mean": float(df["v_var"].mean()),
            "p75": float(df["v_var"].quantile(0.75)),
        }
    if "a_var" in df.columns:
        summary["a_var"] = {
            "mean": float(df["a_var"].mean()),
            "p75": float(df["a_var"].quantile(0.75)),
        }
    return summary


def mood_distribution(df: pd.DataFrame, label: str) -> dict:
    counts = df["mood_k7"].value_counts().sort_index().to_dict()
    total = len(df)
    dist = {}
    for k in range(7):
        c = int(counts.get(k, 0))
        dist[str(k)] = {
            "name": MOOD_NAMES[k],
            "count": c,
            "fraction": c / total if total else 0.0,
        }
    return {"label": label, "n_clips": total, "by_class": dist}


def quadrant_distribution(df: pd.DataFrame) -> dict:
    counts = df["quadrant_k4"].value_counts().sort_index().to_dict()
    total = len(df)
    names = {0: "HVHA", 1: "HVLA", 2: "LVHA", 3: "LVLA"}
    return {
        names[k]: {
            "count": int(counts.get(k, 0)),
            "fraction": counts.get(k, 0) / total if total else 0.0,
        }
        for k in range(4)
    }


def plot_va_overlay(
    liris_df: pd.DataFrame,
    cogn_df: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, dfs) in zip(
        axes,
        [
            ("LIRIS (train+val)", [("LIRIS", liris_df, "steelblue")]),
            ("COGNIMUSE (Hollywood 12)", [("COGNIMUSE", cogn_df, "crimson")]),
        ],
    ):
        for name, df, color in dfs:
            ax.scatter(df["v_norm"], df["a_norm"], s=3, alpha=0.25, color=color, label=name)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Valence (normalized)")
        ax.set_ylabel("Arousal (normalized)")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def apply_va_strategy_A(df: pd.DataFrame) -> pd.DataFrame:
    """LIRIS raw → normalized 을 Strategy A 로 재계산 (보조 측정용)."""
    out = df.copy()
    out["v_norm"] = (out["v_raw"] - 3.0) / 2.0
    out["a_norm"] = (out["a_raw"] - 3.0) / 2.0
    out["v_norm"] = out["v_norm"].clip(-1.0, 1.0)
    out["a_norm"] = out["a_norm"].clip(-1.0, 1.0)
    return out


def build_report(
    liris_csv: Path,
    cognimuse_csv: Path,
    output_dir: Path,
) -> dict:
    """Generates Phase 0 gate outputs. Returns dict with gate_pass bool."""
    liris = pd.read_csv(liris_csv)
    cogn = pd.read_csv(cognimuse_csv)

    # LIRIS 는 v_norm / a_norm 컬럼이 Strategy A 산출물 가정. 없으면 raw 로 계산
    if "v_norm" not in liris.columns and "v_raw" in liris.columns:
        liris = apply_va_strategy_A(liris)

    # Train+Val 만 비교 (test 는 frozen 평가, 사람이 안 본 상태 보존)
    if "split" in liris.columns:
        liris_use = liris[liris["split"].isin(["train", "val"])].reset_index(drop=True)
    else:
        liris_use = liris

    liris_summary = summarize(liris_use, "LIRIS (train+val)")
    cogn_summary = summarize(cogn, "COGNIMUSE (Hollywood 12)")

    liris_mood = mood_distribution(liris_use, "LIRIS")
    cogn_mood = mood_distribution(cogn, "COGNIMUSE")

    liris_quad = quadrant_distribution(liris_use)
    cogn_quad = quadrant_distribution(cogn)

    # Per-film breakdown for COGNIMUSE
    per_film = []
    for fid in sorted(cogn["film_id"].unique()):
        sub = cogn[cogn["film_id"] == fid]
        per_film.append(
            {
                "film_id": int(fid),
                "movie_code": sub["movie_code"].iloc[0],
                "film_name": sub["film_name"].iloc[0] if "film_name" in sub.columns else None,
                "n_windows": len(sub),
                "v_mean": float(sub["v_norm"].mean()),
                "a_mean": float(sub["a_norm"].mean()),
                "v_std": float(sub["v_norm"].std()),
                "a_std": float(sub["a_norm"].std()),
                "mood_k7_mode": int(sub["mood_k7"].mode().iloc[0]),
            }
        )

    # Gate check
    gate_checks: list[dict] = []
    gate_checks.append(
        {
            "name": "cognimuse_va_range_valid",
            "passed": bool(
                cogn["v_norm"].min() >= -1.0
                and cogn["v_norm"].max() <= 1.0
                and cogn["a_norm"].min() >= -1.0
                and cogn["a_norm"].max() <= 1.0
            ),
            "detail": (
                f"v ∈ [{cogn['v_norm'].min():.3f}, {cogn['v_norm'].max():.3f}] "
                f"/ a ∈ [{cogn['a_norm'].min():.3f}, {cogn['a_norm'].max():.3f}]"
            ),
        }
    )
    gate_checks.append(
        {
            "name": "cognimuse_window_count_ok",
            "passed": bool(len(cogn) >= 1500),
            "detail": f"n_windows={len(cogn)} (>=1500 required)",
        }
    )
    mood_above_1pct = sum(1 for v in cogn_mood["by_class"].values() if v["fraction"] >= 0.01)
    gate_checks.append(
        {
            "name": "cognimuse_mood_k7_diversity",
            "passed": bool(mood_above_1pct >= 4),
            "detail": f"{mood_above_1pct}/7 classes have ≥1% (>=4 required)",
        }
    )

    gate_pass = all(ch["passed"] for ch in gate_checks)

    report = {
        "phase": "4-A phase 0 — distribution gate",
        "gate_pass": gate_pass,
        "gate_checks": gate_checks,
        "liris": liris_summary,
        "cognimuse": cogn_summary,
        "delta_cognimuse_vs_liris": {
            "v_mean": cogn_summary["v_norm"]["mean"] - liris_summary["v_norm"]["mean"],
            "a_mean": cogn_summary["a_norm"]["mean"] - liris_summary["a_norm"]["mean"],
            "v_std": cogn_summary["v_norm"]["std"] - liris_summary["v_norm"]["std"],
            "a_std": cogn_summary["a_norm"]["std"] - liris_summary["a_norm"]["std"],
        },
        "mood_k7": {"liris": liris_mood, "cognimuse": cogn_mood},
        "quadrant_k4": {"liris": liris_quad, "cognimuse": cogn_quad},
        "per_film_cognimuse": per_film,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "distribution_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "mood_k7_distribution.json").write_text(
        json.dumps({"liris": liris_mood, "cognimuse": cogn_mood}, indent=2)
    )

    # Plot
    try:
        plot_va_overlay(liris_use, cogn, output_dir / "va_histogram.png")
    except Exception as e:
        print(f"[warn] plot skipped: {e}")

    # Summary markdown
    lines: list[str] = []
    lines.append("# COGNIMUSE vs LIRIS — Distribution Shift Summary\n")
    lines.append("## Gate\n")
    for ch in gate_checks:
        mark = "✅" if ch["passed"] else "❌"
        lines.append(f"- {mark} **{ch['name']}**: {ch['detail']}")
    lines.append(f"\n**Overall gate**: {'PASS' if gate_pass else 'FAIL'}\n")

    lines.append("## V/A Distribution\n")
    lines.append(
        f"| | LIRIS (train+val) | COGNIMUSE | Δ (COGN − LIRIS) |\n"
        f"|---|---|---|---|\n"
        f"| n_clips | {liris_summary['n_clips']} | {cogn_summary['n_clips']} | — |\n"
        f"| V mean | {liris_summary['v_norm']['mean']:+.3f} | "
        f"{cogn_summary['v_norm']['mean']:+.3f} | "
        f"{report['delta_cognimuse_vs_liris']['v_mean']:+.3f} |\n"
        f"| V std | {liris_summary['v_norm']['std']:.3f} | "
        f"{cogn_summary['v_norm']['std']:.3f} | "
        f"{report['delta_cognimuse_vs_liris']['v_std']:+.3f} |\n"
        f"| A mean | {liris_summary['a_norm']['mean']:+.3f} | "
        f"{cogn_summary['a_norm']['mean']:+.3f} | "
        f"{report['delta_cognimuse_vs_liris']['a_mean']:+.3f} |\n"
        f"| A std | {liris_summary['a_norm']['std']:.3f} | "
        f"{cogn_summary['a_norm']['std']:.3f} | "
        f"{report['delta_cognimuse_vs_liris']['a_std']:+.3f} |\n"
        f"| V-A corr | {liris_summary['va_correlation']:+.3f} | "
        f"{cogn_summary['va_correlation']:+.3f} | — |\n"
    )

    lines.append("\n## Mood K=7 Distribution\n")
    lines.append("| Class | Name | LIRIS % | COGNIMUSE % |\n|---|---|---|---|")
    for k in range(7):
        lp = liris_mood["by_class"][str(k)]["fraction"] * 100
        cp = cogn_mood["by_class"][str(k)]["fraction"] * 100
        lines.append(f"| {k} | {MOOD_NAMES[k]} | {lp:.1f}% | {cp:.1f}% |")

    lines.append("\n## Per-Film Breakdown (COGNIMUSE)\n")
    lines.append("| film_id | code | n_windows | V mean | A mean | mood_mode |\n|---|---|---|---|---|---|")
    for p in per_film:
        lines.append(
            f"| {p['film_id']} | {p['movie_code']} | {p['n_windows']} | "
            f"{p['v_mean']:+.3f} | {p['a_mean']:+.3f} | {MOOD_NAMES[p['mood_k7_mode']]} |"
        )

    (output_dir / "scale_shift_summary.md").write_text("\n".join(lines))

    print(f"\n[phase 0] wrote {output_dir}/*")
    print(f"[phase 0] gate: {'PASS' if gate_pass else 'FAIL'}")
    for ch in gate_checks:
        mark = "✅" if ch["passed"] else "❌"
        print(f"  {mark} {ch['name']}: {ch['detail']}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--liris-csv",
        type=Path,
        default=Path("dataset/autoEQ/liris/liris_metadata.csv"),
    )
    p.add_argument(
        "--cognimuse-csv",
        type=Path,
        default=Path("dataset/autoEQ/cognimuse/cognimuse_metadata.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/cognimuse/phase4a/phase0"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.liris_csv, args.cognimuse_csv, args.output_dir)
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
