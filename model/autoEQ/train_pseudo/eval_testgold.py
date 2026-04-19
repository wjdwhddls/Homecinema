"""Evaluate a trained checkpoint on test_gold windows against human ground truth.

Input:
  - checkpoint .pt (from run_train.py)
  - feature_dir holding <split_name>_{visual,audio,metadata}.pt
  - final_labels.csv with `has_human_label` / `human_v` / `human_a` / `split`

Output (JSON + markdown):
  - CCC_V, CCC_A, mean_CCC
  - MAE_V, MAE_A, mean_MAE
  - 4-quadrant confusion (human vs predicted)
  - 7-mood confusion (human_v/a → va_to_mood  vs  predicted v/a → va_to_mood)
  - per-film breakdown
  - gate weight stats

Usage:
    python -m model.autoEQ.train_pseudo.eval_testgold \\
        --ckpt runs/phase3_single/best_model.pt \\
        --feature_dir data/features/ccmovies \\
        --split_name ccmovies \\
        --labels_csv dataset/autoEQ/CCMovies/labels/final_labels.csv \\
        --output runs/phase3_single/testgold_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ..train.dataset import MOOD_CENTERS, va_to_mood
from ..train.utils import compute_ccc
from .config import TrainCogConfig
from .dataset import va_to_quadrant
from .model_base.model import AutoEQModelCog


GEMS_LABELS = ["Tension", "Sadness", "Peacefulness",
               "JoyfulActivation", "Tenderness", "Power", "Wonder"]
QUAD_LABELS = ["HVHA", "HVLA", "LVHA", "LVLA"]


def _quadrant_name(v: float, a: float) -> str:
    return QUAD_LABELS[0] if (v >= 0 and a >= 0) else \
           QUAD_LABELS[1] if (v >= 0 and a < 0) else \
           QUAD_LABELS[2] if (v < 0 and a >= 0) else QUAD_LABELS[3]


def _confusion_matrix(truth: list, pred: list, labels: list[str]) -> dict:
    idx = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    mat = [[0] * K for _ in range(K)]
    for t, p in zip(truth, pred):
        mat[idx[t]][idx[p]] += 1
    return {"labels": labels, "matrix": mat}


def _agreement(truth: list, pred: list) -> float:
    if not truth:
        return 0.0
    return sum(1 for t, p in zip(truth, pred) if t == p) / len(truth)


def evaluate(
    ckpt_path: Path,
    feature_dir: Path,
    split_name: str,
    labels_csv: Path,
    device: torch.device,
    config_overrides: dict | None = None,
) -> dict:
    cfg = TrainCogConfig(**(config_overrides or {}))

    # Load features/metadata
    visual = torch.load(feature_dir / f"{split_name}_visual.pt", weights_only=False)
    audio = torch.load(feature_dir / f"{split_name}_audio.pt", weights_only=False)
    metadata = torch.load(feature_dir / f"{split_name}_metadata.pt", weights_only=False)

    # Restrict to test_gold + has_human_label from CSV
    df = pd.read_csv(labels_csv)
    gold = df[(df["split"] == "test_gold") & (df["has_human_label"] == True)].copy()
    gold_wids = [str(w) for w in gold["window_id"] if str(w) in metadata]
    missing_in_features = set(gold["window_id"].astype(str)) - set(gold_wids)

    if not gold_wids:
        raise RuntimeError("No test_gold+human rows found in feature_dir metadata")

    # Build aligned tensors
    v_feat = torch.stack([visual[w] for w in gold_wids], dim=0).to(device)
    a_feat = torch.stack([audio[w] for w in gold_wids], dim=0).to(device)
    human_v = torch.tensor([float(gold[gold["window_id"] == w]["human_v"].iloc[0])
                            for w in gold_wids], dtype=torch.float32, device=device)
    human_a = torch.tensor([float(gold[gold["window_id"] == w]["human_a"].iloc[0])
                            for w in gold_wids], dtype=torch.float32, device=device)
    pseudo_v = torch.tensor([float(gold[gold["window_id"] == w]["final_v"].iloc[0])
                             for w in gold_wids], dtype=torch.float32, device=device)
    pseudo_a = torch.tensor([float(gold[gold["window_id"] == w]["final_a"].iloc[0])
                             for w in gold_wids], dtype=torch.float32, device=device)
    film_ids = [str(gold[gold["window_id"] == w]["film_id"].iloc[0]) for w in gold_wids]

    # Model
    model = AutoEQModelCog(cfg).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)

    with torch.no_grad():
        out = model(v_feat, a_feat)
    va_pred = out["va_pred"]  # (N, 2)
    gate_w = out["gate_weights"]  # (N, 2)
    pred_v = va_pred[:, 0]
    pred_a = va_pred[:, 1]

    # Regression metrics — human as ground truth
    def _metrics(pred_v, pred_a, true_v, true_a):
        ccc_v = compute_ccc(pred_v, true_v).item()
        ccc_a = compute_ccc(pred_a, true_a).item()
        mae_v = (pred_v - true_v).abs().mean().item()
        mae_a = (pred_a - true_a).abs().mean().item()
        return {
            "ccc_valence": ccc_v, "ccc_arousal": ccc_a,
            "mean_ccc": 0.5 * (ccc_v + ccc_a),
            "mae_valence": mae_v, "mae_arousal": mae_a,
            "mean_mae": 0.5 * (mae_v + mae_a),
        }

    pred_vs_human = _metrics(pred_v, pred_a, human_v, human_a)
    pred_vs_pseudo = _metrics(pred_v, pred_a, pseudo_v, pseudo_a)
    pseudo_vs_human = _metrics(pseudo_v, pseudo_a, human_v, human_a)

    # Quadrant + 7-mood confusion — human vs predicted
    pred_quad = [_quadrant_name(v.item(), a.item()) for v, a in zip(pred_v, pred_a)]
    human_quad = [_quadrant_name(v.item(), a.item()) for v, a in zip(human_v, human_a)]
    pred_mood = [GEMS_LABELS[va_to_mood(v.item(), a.item())] for v, a in zip(pred_v, pred_a)]
    human_mood = [GEMS_LABELS[va_to_mood(v.item(), a.item())] for v, a in zip(human_v, human_a)]

    # Per-film breakdown
    per_film: dict = {}
    for film in sorted(set(film_ids)):
        mask = [i for i, f in enumerate(film_ids) if f == film]
        if not mask:
            continue
        mask_t = torch.tensor(mask)
        per_film[film] = _metrics(
            pred_v[mask_t], pred_a[mask_t], human_v[mask_t], human_a[mask_t]
        )
        per_film[film]["n"] = len(mask)

    # Gate stats
    gate_stats = {
        "mean_w_v": gate_w[:, 0].mean().item(),
        "mean_w_a": gate_w[:, 1].mean().item(),
        "std_w_v": gate_w[:, 0].std().item(),
        "std_w_a": gate_w[:, 1].std().item(),
    }

    report = {
        "ckpt": str(ckpt_path),
        "feature_dir": str(feature_dir),
        "n_test_gold": len(gold_wids),
        "missing_in_features": sorted(missing_in_features),
        "pred_vs_human": pred_vs_human,
        "pred_vs_pseudo": pred_vs_pseudo,
        "pseudo_vs_human": pseudo_vs_human,
        "quadrant_agreement_pred_vs_human": _agreement(human_quad, pred_quad),
        "quadrant_confusion": _confusion_matrix(human_quad, pred_quad, QUAD_LABELS),
        "mood7_agreement_pred_vs_human": _agreement(human_mood, pred_mood),
        "mood7_confusion": _confusion_matrix(human_mood, pred_mood, GEMS_LABELS),
        "per_film": per_film,
        "gate_stats": gate_stats,
    }
    return report


def _md_summary(report: dict) -> str:
    p = report["pred_vs_human"]
    lines = [
        f"# test_gold Human Evaluation\n",
        f"- Checkpoint: {report['ckpt']}",
        f"- N windows: {report['n_test_gold']}\n",
        "## Primary (Model vs Human)",
        f"- **mean CCC: {p['mean_ccc']:.4f}** (V={p['ccc_valence']:.3f}, A={p['ccc_arousal']:.3f})",
        f"- **mean MAE: {p['mean_mae']:.4f}** (V={p['mae_valence']:.3f}, A={p['mae_arousal']:.3f})",
        f"- Quadrant agreement: **{report['quadrant_agreement_pred_vs_human']:.1%}**",
        f"- 7-mood agreement: **{report['mood7_agreement_pred_vs_human']:.1%}**\n",
        "## Baseline comparisons",
        f"- Pseudo-label (Gemini+ensemble) vs Human — mean CCC: "
        f"{report['pseudo_vs_human']['mean_ccc']:.4f}",
        f"- Model vs Pseudo-label — mean CCC: "
        f"{report['pred_vs_pseudo']['mean_ccc']:.4f}\n",
        "## Per-film",
        "| film | n | CCC | MAE |",
        "|---|---|---|---|",
    ]
    for film, m in report["per_film"].items():
        lines.append(f"| {film} | {m['n']} | {m['mean_ccc']:.3f} | {m['mean_mae']:.3f} |")
    lines.append("\n## Gate")
    g = report["gate_stats"]
    lines.append(f"- w_v: {g['mean_w_v']:.3f} ± {g['std_w_v']:.3f}")
    lines.append(f"- w_a: {g['mean_w_a']:.3f} ± {g['std_w_a']:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="test_gold human evaluation")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--feature_dir", type=Path, required=True)
    p.add_argument("--split_name", type=str, default="ccmovies")
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="output path stem (writes .json + .md)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--num_mood_classes", type=int, default=4, choices=[4, 7])
    args = p.parse_args(argv)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    report = evaluate(
        ckpt_path=args.ckpt,
        feature_dir=args.feature_dir,
        split_name=args.split_name,
        labels_csv=args.labels_csv,
        device=device,
        config_overrides={"num_mood_classes": args.num_mood_classes},
    )
    out_json = args.output.with_suffix(".json")
    out_md = args.output.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(_md_summary(report))

    p = report["pred_vs_human"]
    print(f"[done] N={report['n_test_gold']}  "
          f"mean_CCC={p['mean_ccc']:.4f}  mean_MAE={p['mean_mae']:.4f}  "
          f"quad_agree={report['quadrant_agreement_pred_vs_human']:.1%}  "
          f"mood7_agree={report['mood7_agreement_pred_vs_human']:.1%}")
    print(f"[wrote] {out_json}\n[wrote] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
