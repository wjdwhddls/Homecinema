"""analyze_mushra_results.py — MUSHRA 결과 집계 + Wilcoxon 분석.

evaluation/webmushra/results/ 하 JSON 파일들을 자동으로 로드하여:
  1) 참가자별 / 조건별 / 씬별(segment MUSHRA의 경우) / 축별(full trailer의 경우) 집계
  2) Wilcoxon signed-rank test (V3.3 vs V3.5.5, V3.5.5 vs V3.5.6 등)
  3) 표/Markdown 보고서 출력

자동으로 두 page_type을 구분:
  - "segment_mushra": simple_player_v3.html 산출 (4 segment × 4 condition, 단일 점수)
  - "full_trailer":  full_trailer_comparison.html 산출 (1 트레일러 × 5 condition × 3 axis)

post_screening 플래그된 세션은 기본적으로 제외 (--include-flagged로 포함 가능).

실행:
  PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tools/analyze_mushra_results.py

  옵션:
    --results-dir evaluation/webmushra/results
    --output results/report.md        # Markdown 보고서 저장 (생략 시 stdout)
    --page-type [segment_mushra|full_trailer|all]  (기본 all)
    --include-flagged                  # post_screening.flagged 세션도 포함

의존성: scipy, numpy (이미 설치됨).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO / "evaluation" / "webmushra" / "results"


# ────────────────────────────────────────────────────────
# Loading + page_type 구분
# ────────────────────────────────────────────────────────
def load_all_results(results_dir: Path) -> list[dict]:
    """results/ 하 *.json 파일 모두 로드. page_type 미기재 시 metadata.mode로 유추."""
    files = sorted(results_dir.glob("*.json"))
    out = []
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)
            continue
        # 정규화: page_type 유추
        pt = obj.get("page_type")
        if not pt:
            mode = (obj.get("metadata") or {}).get("mode", "")
            if "full_trailer" in mode:
                pt = "full_trailer"
            elif "plan_b_html_player" in mode or "mushra" in mode:
                pt = "segment_mushra"
            else:
                pt = "unknown"
        obj["_page_type"] = pt
        obj["_filename"] = p.name
        out.append(obj)
    return out


def filter_by_page(results: list[dict], page_type: str) -> list[dict]:
    return [r for r in results if r["_page_type"] == page_type]


def is_flagged(r: dict) -> bool:
    ps = r.get("post_screening")
    if not isinstance(ps, dict):
        return False
    if ps.get("flagged") is True:
        return True
    if ps.get("validity") == "suspicious":
        return True
    return False


# ────────────────────────────────────────────────────────
# Segment MUSHRA (4 segments × 4 conditions, 단일 점수)
# ────────────────────────────────────────────────────────
def aggregate_segment(results: list[dict]) -> dict:
    """결과: {segment_id: {cond_key: [score1, score2, ...]}}"""
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        scores = r.get("scores") or {}
        for seg_id, seg_scores in scores.items():
            for cond, entry in seg_scores.items():
                if isinstance(entry, dict) and "score" in entry:
                    agg[seg_id][cond].append(float(entry["score"]))
                elif isinstance(entry, (int, float)):
                    agg[seg_id][cond].append(float(entry))
    return agg


def format_segment_report(agg: dict, results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Segment MUSHRA 결과 집계\n")
    lines.append(f"- 참가자 세션 수: **{len(results)}**")
    flagged = sum(1 for r in results if is_flagged(r))
    lines.append(f"- Post-screening 플래그(제외됨): **{flagged}** 세션")
    lines.append("")
    if not agg:
        lines.append("_결과 없음_"); return "\n".join(lines)

    for seg_id in sorted(agg.keys()):
        lines.append(f"## {seg_id}\n")
        lines.append(f"| Condition | N | Mean | Std | Min | Max |")
        lines.append(f"|---|---|---|---|---|---|")
        for cond in sorted(agg[seg_id].keys()):
            vals = agg[seg_id][cond]
            if not vals: continue
            n = len(vals)
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if n > 1 else 0.0
            lines.append(f"| {cond} | {n} | {mean:.1f} | {std:.1f} | {min(vals):.1f} | {max(vals):.1f} |")
        lines.append("")

        # Wilcoxon: v3_3 vs v3_5_5 (sample-paired, per participant)
        if HAS_SCIPY:
            for (a, b) in [("v3_3", "v3_5_5"), ("v3_5_5", "v3_5_6"), ("v3_3", "v3_5_6")]:
                va = agg[seg_id].get(a, [])
                vb = agg[seg_id].get(b, [])
                m = min(len(va), len(vb))
                if m >= 5:
                    try:
                        w = scipy_stats.wilcoxon(va[:m], vb[:m], zero_method="wilcox", nan_policy="omit")
                        lines.append(
                            f"- **Wilcoxon {a} vs {b}** (n={m}): "
                            f"statistic={w.statistic:.2f}, p={w.pvalue:.4f}  "
                            f"{'**유의 (p<0.05)**' if w.pvalue < 0.05 else '비유의'}"
                        )
                    except ValueError as e:
                        lines.append(f"- Wilcoxon {a} vs {b}: {e}")
                elif va and vb:
                    lines.append(f"- Wilcoxon {a} vs {b}: n={m} < 5 (스킵)")
            lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────
# Full Trailer (5 conditions × 3 axes)
# ────────────────────────────────────────────────────────
def aggregate_full_trailer(results: list[dict]) -> dict:
    """결과: {cond_key: {axis_key: [score1, score2, ...]}}"""
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        scores = r.get("scores") or {}
        for cond, axes in scores.items():
            if not isinstance(axes, dict): continue
            for axis_key, v in axes.items():
                if isinstance(v, (int, float)):
                    agg[cond][axis_key].append(float(v))
    return agg


def format_full_trailer_report(agg: dict, results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Full Trailer 비교 결과 집계\n")
    lines.append(f"- 참가자 세션 수: **{len(results)}**")
    flagged = sum(1 for r in results if is_flagged(r))
    lines.append(f"- Post-screening 플래그(제외됨): **{flagged}** 세션")
    lines.append("")
    if not agg:
        lines.append("_결과 없음_"); return "\n".join(lines)

    # 조건별 × 축별 집계표
    all_axes = sorted({a for d in agg.values() for a in d.keys()})
    all_conds = sorted(agg.keys())
    lines.append(f"## 조건별 × 축별 평균 (N = 세션 수)\n")
    lines.append("| Condition \\\\ Axis | " + " | ".join(all_axes) + " |")
    lines.append("|---" * (len(all_axes) + 1) + "|")
    for cond in all_conds:
        row = [cond]
        for axis in all_axes:
            vals = agg[cond].get(axis, [])
            if not vals:
                row.append("—"); continue
            n = len(vals)
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if n > 1 else 0.0
            row.append(f"{mean:.1f} ± {std:.1f} (n={n})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 축별 세부 + Wilcoxon
    for axis in all_axes:
        lines.append(f"## 축: {axis}\n")
        lines.append("| Condition | N | Mean | Std | Min | Max |")
        lines.append("|---|---|---|---|---|---|")
        for cond in all_conds:
            vals = agg[cond].get(axis, [])
            if not vals: continue
            n = len(vals)
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if n > 1 else 0.0
            lines.append(f"| {cond} | {n} | {mean:.1f} | {std:.1f} | {min(vals):.1f} | {max(vals):.1f} |")
        lines.append("")

        if HAS_SCIPY:
            for (a, b) in [("v3_3", "v3_5_5"), ("v3_5_5", "v3_5_6"), ("v3_3", "v3_5_6"),
                           ("reference", "v3_5_6"), ("v3_5_6", "anchor")]:
                va = agg[a].get(axis, []) if a in agg else []
                vb = agg[b].get(axis, []) if b in agg else []
                m = min(len(va), len(vb))
                if m >= 5:
                    try:
                        w = scipy_stats.wilcoxon(va[:m], vb[:m], zero_method="wilcox", nan_policy="omit")
                        lines.append(
                            f"- **Wilcoxon {a} vs {b}** (n={m}, {axis}): "
                            f"statistic={w.statistic:.2f}, p={w.pvalue:.4f}  "
                            f"{'**유의 (p<0.05)**' if w.pvalue < 0.05 else '비유의'}"
                        )
                    except ValueError as e:
                        lines.append(f"- Wilcoxon {a} vs {b}: {e}")
                elif va and vb:
                    lines.append(f"- Wilcoxon {a} vs {b} ({axis}): n={m} < 5 (스킵)")
            lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--output", default=None, help="Markdown 보고서 저장 경로 (생략 시 stdout)")
    ap.add_argument("--page-type", default="all",
                    choices=["segment_mushra", "full_trailer", "all"])
    ap.add_argument("--include-flagged", action="store_true",
                    help="post_screening 플래그 세션도 포함")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[error] 결과 디렉토리 없음: {results_dir}", file=sys.stderr); sys.exit(2)

    all_results = load_all_results(results_dir)
    print(f"로드된 세션: {len(all_results)}", file=sys.stderr)

    if not args.include_flagged:
        before = len(all_results)
        all_results = [r for r in all_results if not is_flagged(r)]
        dropped = before - len(all_results)
        if dropped:
            print(f"post_screening 플래그 제외: {dropped} 세션 (--include-flagged로 포함 가능)",
                  file=sys.stderr)

    sections: list[str] = []
    sections.append(f"# MUSHRA 결과 분석 보고서\n")
    sections.append(f"- 결과 디렉토리: `{results_dir}`")
    sections.append(f"- 로드된 유효 세션: **{len(all_results)}**")
    sections.append(f"- 의존성 scipy: {'✓ 사용' if HAS_SCIPY else '✗ 미설치 (Wilcoxon 생략됨)'}\n")

    if args.page_type in ("segment_mushra", "all"):
        seg = filter_by_page(all_results, "segment_mushra")
        if seg:
            agg = aggregate_segment(seg)
            sections.append(format_segment_report(agg, seg))
        else:
            sections.append("# Segment MUSHRA\n\n_결과 파일 없음._\n")

    if args.page_type in ("full_trailer", "all"):
        ft = filter_by_page(all_results, "full_trailer")
        if ft:
            agg = aggregate_full_trailer(ft)
            sections.append(format_full_trailer_report(agg, ft))
        else:
            sections.append("# Full Trailer 비교\n\n_결과 파일 없음._\n")

    report = "\n".join(sections)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"보고서 저장: {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
