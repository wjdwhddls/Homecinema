"""Detailed overfit + training quality analysis for spec baseline 3-seed.

Report covers:
  1. Per-seed best epoch + gap dynamics
  2. Overfit-flag onset epoch (gap > 0.10)
  3. Train/val CCC trajectories (first 10 epochs, then best, then final)
  4. Loss term decomposition at best epoch
  5. ccc_v / ccc_a balance
  6. 3-seed consistency (std across seeds at same epoch)
  7. Comparison vs V2 on-same-epoch basis
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "phase2a"
SEEDS = [42, 123, 2024]


def load_h(tag: str, seed: int):
    p = RUNS / f"{tag}_s{seed}" / "history.json"
    return json.loads(p.read_text()) if p.is_file() else None


def summarize_group(tag: str, label: str):
    hists = [load_h(tag, s) for s in SEEDS]
    hists = [h for h in hists if h is not None]
    if not hists:
        return None

    # Best epoch per seed
    best_eps = [max(range(len(h)), key=lambda i: h[i]["val"]["mean_ccc"]) for h in hists]
    best_vals = [h[be]["val"]["mean_ccc"] for h, be in zip(hists, best_eps)]
    train_at_best = [h[be]["train"]["mean_ccc"] for h, be in zip(hists, best_eps)]
    gap_at_best = [h[be]["overfit_gap"] for h, be in zip(hists, best_eps)]

    # Overfit onset (first epoch with gap > 0.10)
    def onset(h):
        for i, row in enumerate(h):
            if row["overfit_gap"] > 0.10:
                return i
        return None
    onsets = [onset(h) for h in hists]

    # Final gap
    final_gap = [h[-1]["overfit_gap"] for h in hists]
    final_train = [h[-1]["train"]["mean_ccc"] for h in hists]
    final_val = [h[-1]["val"]["mean_ccc"] for h in hists]

    # Loss at best epoch
    def loss_at_best(h, be):
        return {k: h[be]["train"][k] for k in
                ["loss_total", "loss_va", "loss_va_mse", "loss_va_ccc",
                 "loss_mood", "loss_gate_entropy"]}
    losses = [loss_at_best(h, be) for h, be in zip(hists, best_eps)]

    return {
        "label": label,
        "n_seeds": len(hists),
        "best_ep_mean": mean(best_eps),
        "best_ep_range": (min(best_eps), max(best_eps)),
        "val_ccc_mean_std": (mean(best_vals), stdev(best_vals) if len(best_vals) > 1 else 0.0),
        "train_at_best_mean": mean(train_at_best),
        "gap_at_best_mean": mean(gap_at_best),
        "overfit_onset_ep": onsets,
        "final_gap_mean": mean(final_gap),
        "final_train_mean": mean(final_train),
        "final_val_mean": mean(final_val),
        "total_epochs_mean": mean(len(h) for h in hists),
        "loss_at_best_avg": {
            k: mean([L[k] for L in losses]) for k in losses[0]
        },
    }


def print_group(g: dict | None):
    if g is None:
        print("  (no data)")
        return
    mean_, std_ = g["val_ccc_mean_std"]
    print(f"  best_ep       : {g['best_ep_mean']:.1f}  (range {g['best_ep_range'][0]}-{g['best_ep_range'][1]})")
    print(f"  val CCC       : {mean_:.4f} ± {std_:.4f}")
    print(f"  train@best    : {g['train_at_best_mean']:.4f}")
    print(f"  gap@best      : {g['gap_at_best_mean']:.3f}")
    print(f"  overfit onset : {g['overfit_onset_ep']}")
    print(f"  final gap     : {g['final_gap_mean']:.3f}")
    print(f"  final train   : {g['final_train_mean']:.4f}")
    print(f"  final val     : {g['final_val_mean']:.4f}")
    print(f"  total epochs  : {g['total_epochs_mean']:.1f}")
    L = g["loss_at_best_avg"]
    print(f"  loss @ best   : total={L['loss_total']:.3f}  va={L['loss_va']:.3f}"
          f"  (mse={L['loss_va_mse']:.3f} ccc={L['loss_va_ccc']:.3f})"
          f"  mood={L['loss_mood']:.3f}  gate={L['loss_gate_entropy']:+.3f}")


def per_epoch_curves(tag: str, label: str, max_ep: int = 20):
    hists = [load_h(tag, s) for s in SEEDS]
    hists = [h for h in hists if h is not None]
    if not hists:
        return
    # Align to shortest history
    n = min(len(h) for h in hists)
    n = min(n, max_ep + 1)
    print(f"\n  Per-epoch val CCC ({label}, {len(hists)} seeds):")
    print(f"  {'ep':>3}  {'train':>8} {'val':>8}  {'gap':>6}  {'flag':>4}")
    for i in range(n):
        t_vals = [h[i]["train"]["mean_ccc"] for h in hists]
        v_vals = [h[i]["val"]["mean_ccc"] for h in hists]
        g_vals = [h[i]["overfit_gap"] for h in hists]
        flag = "!" if mean(g_vals) > 0.10 else " "
        print(f"  {i:>3}  {mean(t_vals):.4f}  {mean(v_vals):.4f}  {mean(g_vals):+.3f}  {flag:>4}")


def main():
    groups = [
        ("spec_baseline", "SPEC §11 baseline"),
        ("baseline_v2", "V2 quick-wins (old features, OAT violation)"),
        ("EMA_d99", "EMA d=0.99 (old features)"),
    ]

    for tag, label in groups:
        print(f"\n{'='*70}\n{label}   (tag={tag})\n{'='*70}")
        g = summarize_group(tag, label)
        print_group(g)

    # Detailed per-epoch for spec baseline
    print(f"\n{'='*70}\nSPEC baseline — per-epoch 3-seed mean\n{'='*70}")
    per_epoch_curves("spec_baseline", "SPEC", max_ep=20)

    # V2 for comparison
    print(f"\n{'='*70}\nV2 — per-epoch 3-seed mean (for comparison)\n{'='*70}")
    per_epoch_curves("baseline_v2", "V2", max_ep=20)

    # Save full report
    report = {}
    for tag, label in groups:
        report[tag] = summarize_group(tag, label)
    (RUNS / "spec_baseline_analysis.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(f"\n[saved] {RUNS / 'spec_baseline_analysis.json'}")


if __name__ == "__main__":
    main()
