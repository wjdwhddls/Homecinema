"""Aggregate V5-FINAL §11 Phase 2a-0 spec baseline vs previous runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "phase2a"
SEEDS = [42, 123, 2024]


def load_best(dir_name: str) -> dict | None:
    p = RUNS / dir_name / "summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())["best_val"]


def history_stats(dir_name: str) -> dict:
    h = json.loads((RUNS / dir_name / "history.json").read_text())
    best_i = max(range(len(h)), key=lambda i: h[i]["val"]["mean_ccc"])
    return {"best_ep": best_i, "total_ep": len(h), "gap_at_best": h[best_i]["overfit_gap"]}


def paired_t(x: list[float], y: list[float]) -> dict:
    d = [a - b for a, b in zip(x, y)]
    if all(abs(v - d[0]) < 1e-9 for v in d):
        return {"mean_diff": mean(d), "t": None, "note": "no variance"}
    t = mean(d) / (stdev(d) / math.sqrt(len(d)))
    return {"mean_diff": round(mean(d), 4), "std_diff": round(stdev(d), 4),
            "t": round(t, 3), "diffs": [round(v, 4) for v in d]}


def collect(prefix: str):
    rows = []
    for s in SEEDS:
        m = load_best(f"{prefix}{s}")
        if m is None:
            rows.append(None)
            continue
        rows.append({"seed": s, **m, **history_stats(f"{prefix}{s}")})
    return rows


def summary_row(tag: str, rows):
    present = [r for r in rows if r is not None]
    if not present:
        return None
    mc = [r["mean_ccc"] for r in present]
    cv = [r["ccc_v"] for r in present]
    ca = [r["ccc_a"] for r in present]
    beps = [r["best_ep"] for r in present]
    gaps = [r["gap_at_best"] for r in present]
    return {
        "tag": tag,
        "n": len(present),
        "mean_ccc": {"mean": mean(mc), "std": stdev(mc) if len(mc) > 1 else 0.0},
        "ccc_v": {"mean": mean(cv), "std": stdev(cv) if len(cv) > 1 else 0.0},
        "ccc_a": {"mean": mean(ca), "std": stdev(ca) if len(ca) > 1 else 0.0},
        "best_ep_mean": mean(beps),
        "gap_at_best_mean": mean(gaps),
    }


def main():
    v1 = collect("baseline_2a0_s") or collect("baseline_2a0")  # V1 single-seed
    # V1 was single-seed 42; check dir existence
    v1_dir = RUNS / "baseline_2a0"
    v1_single = load_best("baseline_2a0") if v1_dir.is_dir() else None

    groups = [
        ("V2 (quick wins)", collect("baseline_v2_s")),
        ("A a1 (head=0.5)", collect("A_a1_s")),
        ("A a2 (wd=1e-3)", collect("A_a2_s")),
        ("A a3 (head=0.5 wd=1e-3)", collect("A_a3_s")),
        ("EMA d=0.99", collect("EMA_d99_s")),
        ("SPEC baseline (§11)", collect("spec_baseline_s")),
    ]

    rows = []
    for tag, data in groups:
        s = summary_row(tag, data)
        if s is not None:
            rows.append(s)

    print("\n=== All Phase 2a runs (3-seed summaries) ===")
    print(f"{'tag':<28} {'n':>2} {'mean_CCC':>17} {'ccc_v':>17} {'ccc_a':>17} {'best_ep':>8} {'gap@best':>9}")
    print("-" * 113)
    for r in rows:
        print(f"{r['tag']:<28} {r['n']:>2} "
              f"{r['mean_ccc']['mean']:.4f}±{r['mean_ccc']['std']:.4f}  "
              f"{r['ccc_v']['mean']:.4f}±{r['ccc_v']['std']:.4f}  "
              f"{r['ccc_a']['mean']:.4f}±{r['ccc_a']['std']:.4f}  "
              f"{r['best_ep_mean']:>8.1f}  {r['gap_at_best_mean']:>9.3f}")

    spec = collect("spec_baseline_s")
    v2 = collect("baseline_v2_s")
    if all(s is not None for s in spec) and all(s is not None for s in v2):
        spec_cccs = [s["mean_ccc"] for s in spec]
        v2_cccs = [s["mean_ccc"] for s in v2]
        t_spec_v2 = paired_t(spec_cccs, v2_cccs)
        print(f"\npaired t-test SPEC vs V2 (n=3): {t_spec_v2}")

    out = RUNS / "spec_baseline_summary.json"
    out.write_text(json.dumps({"summary_rows": rows}, indent=2, default=str))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
