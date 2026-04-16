"""mushra_analyzer.py — webMUSHRA 결과 CSV 분석.

작업 15 (Day 13~15):
- webMUSHRA가 생성한 CSV 결과 로드
- 조건별(원본/V3.1/V3.2/Anchor) 평균 점수 + 95% CI
- V3.1 vs V3.2 paired t-test
- mood/density별 차이 분석
- 시각화 (박스플롯, 막대그래프)
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .paths import EVALUATION_RESULTS_DIR, WEBMUSHRA_DIR


CONDITION_COLORS = {
    "original_hidden": "#888",
    "v3_1": "#3b82f6",
    "v3_2": "#ef4444",
    "anchor": "#9ca3af",
}

CONDITION_LABELS = {
    "original_hidden": "Original",
    "v3_1": "V3.1 (Baseline)",
    "v3_2": "V3.2 (Dramatic)",
    "anchor": "Anchor",
}


def load_mushra_csv(csv_path) -> pd.DataFrame:
    """webMUSHRA 결과 CSV 로드.

    예상 컬럼: session_test_id, trial_id, rating_stimulus, rating_score
    실제 컬럼명은 webMUSHRA 버전에 따라 다를 수 있어 자동 매핑 시도.
    """
    df = pd.read_csv(csv_path)
    print(f"  로드: {csv_path} ({len(df)}행, 컬럼: {list(df.columns)})")

    # 컬럼 자동 매핑
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "trial" in low or "page" in low:
            col_map[col] = "trial_id"
        elif "stimulus" in low or "condition" in low:
            col_map[col] = "stimulus"
        elif "score" in low or "rating" in low:
            col_map[col] = "score"
        elif "session" in low or "subject" in low or "user" in low:
            col_map[col] = "session"

    df = df.rename(columns=col_map)
    required = {"trial_id", "stimulus", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"필수 컬럼 누락: {missing}\n실제 컬럼: {list(df.columns)}\n"
            "webMUSHRA CSV 형식을 확인하세요."
        )
    return df


def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """조건별 평균 점수 + 95% CI."""
    summary = []
    for cond, group in df.groupby("stimulus"):
        scores = group["score"].dropna().values
        if len(scores) == 0:
            continue
        mean = np.mean(scores)
        std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        n = len(scores)
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        summary.append({
            "condition": cond,
            "n": n,
            "mean": round(mean, 2),
            "std": round(std, 2),
            "ci95_low": round(mean - ci95, 2),
            "ci95_high": round(mean + ci95, 2),
        })
    return pd.DataFrame(summary).sort_values("mean", ascending=False)


def paired_test_v31_vs_v32(df: pd.DataFrame) -> dict:
    """같은 trial 내 V3.1 vs V3.2 paired t-test + Wilcoxon."""
    pivot = df.pivot_table(
        index=["trial_id", "session"] if "session" in df.columns else ["trial_id"],
        columns="stimulus",
        values="score",
    )

    if "v3_1" not in pivot.columns or "v3_2" not in pivot.columns:
        return {"error": "v3_1 또는 v3_2 조건 없음"}

    paired = pivot[["v3_1", "v3_2"]].dropna()
    if len(paired) < 3:
        return {"error": f"paired 샘플 {len(paired)}개 — 너무 적음"}

    diff = paired["v3_2"] - paired["v3_1"]
    t_stat, t_p = stats.ttest_rel(paired["v3_2"], paired["v3_1"])
    w_stat, w_p = stats.wilcoxon(paired["v3_2"], paired["v3_1"])

    return {
        "n_pairs": len(paired),
        "v3_1_mean": round(paired["v3_1"].mean(), 2),
        "v3_2_mean": round(paired["v3_2"].mean(), 2),
        "diff_mean": round(diff.mean(), 2),
        "diff_std": round(diff.std(), 2),
        "ttest_t": round(t_stat, 3),
        "ttest_p": round(t_p, 4),
        "wilcoxon_W": round(w_stat, 3),
        "wilcoxon_p": round(w_p, 4),
        "significant_at_0.05": bool(t_p < 0.05),
    }


def plot_condition_bars(summary: pd.DataFrame, save=None) -> None:
    """조건별 평균 + 95% CI 막대그래프."""
    fig, ax = plt.subplots(figsize=(8, 5))
    conds = summary["condition"].tolist()
    means = summary["mean"].tolist()
    errs = [(m - lo) for m, lo in zip(means, summary["ci95_low"])]
    colors = [CONDITION_COLORS.get(c, "#888") for c in conds]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]

    bars = ax.bar(labels, means, yerr=errs, color=colors,
                   capsize=8, edgecolor="black", linewidth=0.8)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2, f"{m:.1f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("평균 점수 (0~100)")
    ax.set_title("조건별 평균 점수 (95% CI)", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  💾 {save} 저장")
    plt.close(fig)


def plot_condition_boxplot(df: pd.DataFrame, save=None) -> None:
    """조건별 점수 분포 박스플롯."""
    fig, ax = plt.subplots(figsize=(8, 5))
    conds = ["original_hidden", "v3_1", "v3_2", "anchor"]
    data = [df[df["stimulus"] == c]["score"].dropna().values for c in conds]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor(CONDITION_COLORS.get(c, "#888"))
        patch.set_alpha(0.7)

    ax.set_ylabel("점수 (0~100)")
    ax.set_title("조건별 점수 분포", fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  💾 {save} 저장")
    plt.close(fig)


def analyze_mushra(csv_path) -> dict:
    """전체 분석 파이프라인."""
    print(f"\n=== webMUSHRA 결과 분석: {csv_path} ===")
    df = load_mushra_csv(csv_path)

    print(f"\n[조건별 요약]")
    summary = summarize_by_condition(df)
    print(summary.to_string(index=False))

    print(f"\n[V3.1 vs V3.2 통계 검정]")
    test = paired_test_v31_vs_v32(df)
    if "error" in test:
        print(f"  ⚠️ {test['error']}")
    else:
        print(f"  N = {test['n_pairs']} pairs")
        print(f"  V3.1 mean = {test['v3_1_mean']}, V3.2 mean = {test['v3_2_mean']}")
        print(f"  diff = {test['diff_mean']:+.2f} ± {test['diff_std']:.2f}")
        print(f"  Paired t: t={test['ttest_t']}, p={test['ttest_p']}")
        print(f"  Wilcoxon: W={test['wilcoxon_W']}, p={test['wilcoxon_p']}")
        sig = "✓ 유의함" if test["significant_at_0.05"] else "✗ 유의차 없음"
        print(f"  α=0.05 기준: {sig}")

    # 시각화 + 요약은 evaluation/results/ 에 저장 (백엔드 outputs와 분리)
    out_dir = EVALUATION_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_condition_bars(summary, out_dir / "mushra_bars.png")
    plot_condition_boxplot(df, out_dir / "mushra_boxplot.png")

    # 결과 CSV
    summary.to_csv(out_dir / "mushra_summary.csv", index=False)
    print(f"\n  💾 요약 CSV: {out_dir / 'mushra_summary.csv'}")

    # Anchor 검증 (sanity check)
    if "anchor" in summary["condition"].values:
        anchor_mean = summary[summary["condition"] == "anchor"]["mean"].iloc[0]
        if anchor_mean > 50:
            print(
                f"\n  ⚠️ Anchor 평균 {anchor_mean:.1f}점 > 50 — 평가자가 신뢰성 없거나 "
                f"Anchor 설계가 충분히 저품질이 아닐 수 있음"
            )
        else:
            print(f"\n  ✓ Anchor 평균 {anchor_mean:.1f}점 — 평가 신뢰성 OK")

    return {
        "summary": summary.to_dict("records"),
        "v31_vs_v32": test,
        "n_total_ratings": len(df),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 기본 위치: webMUSHRA가 자체 results/ 폴더에 저장한 CSV
        # webMUSHRA는 보통 results/<testId>/<session>/...csv 형식으로 저장
        results_root = WEBMUSHRA_DIR / "results"
        candidates = list(results_root.rglob("*.csv")) if results_root.exists() else []
        if candidates:
            # 가장 최근 CSV 사용
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"가장 최근 CSV 자동 선택: {latest}")
            analyze_mushra(latest)
        else:
            print("사용법: python -m model.autoEQ.inference.mushra_analyzer <results.csv>")
            print(f"또는 webMUSHRA가 {results_root} 에 결과를 저장했는지 확인하세요.")
            sys.exit(1)
    else:
        analyze_mushra(sys.argv[1])
