"""Localize overfit source in AutoEQModelLiris.

Architecture (total trainable ≈ 1.84 M):
  AudioProjection   Linear(2048→512)           ≈ 1.05M  (57%)
  GateNetwork       Linear(1024→256)+(256→2)   ≈ 263K   (14%)
  VA Head           Linear(1024→256)+(256→2)   ≈ 263K   (14%)
  Mood Head         Linear(1024→256)+(256→K=4) ≈ 263K   (14%)

Diagnostics:
  1. Per-block parameter count (definitive).
  2. Per-block weight L2 growth from init → trained (memorization pressure).
  3. Per-block train vs val activation stats (memorization signature).
  4. Gate weight distribution: does the model collapse onto one modality?
  5. VA head linear layer singular values: high kurtosis = overfit memorization.
  6. Train vs val gradient-to-weight norm ratio at best epoch (proxy).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainLirisConfig
from .dataset import MixupTargetShrinkageCollator, PrecomputedLirisDataset, official_split
from .model import AutoEQModelLiris

REPO = Path(__file__).resolve().parents[3]
META = REPO / "dataset" / "autoEQ" / "liris" / "liris_metadata.csv"
FEATS = REPO / "data" / "features" / "liris_panns_v5spec" / "features.pt"
CKPT = REPO / "runs" / "phase2a" / "spec_baseline_s42" / "best.pt"


def count_params(model: torch.nn.Module) -> dict:
    blocks = {
        "audio_projection": model.audio_projection,
        "gate_network": model.gate_network,
        "va_head": model.va_head,
        "mood_head": model.mood_head,
    }
    out = {}
    total = 0
    for name, blk in blocks.items():
        n = sum(p.numel() for p in blk.parameters() if p.requires_grad)
        out[name] = n
        total += n
    out["_total"] = total
    out["_pct"] = {k: 100.0 * v / total for k, v in out.items() if k != "_total"}
    return out


def weight_l2_stats(model: torch.nn.Module) -> dict:
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2:  # weight matrix
            l2 = float(p.detach().norm().item())
            mean_abs = float(p.detach().abs().mean().item())
            n = p.numel()
            # normalized by sqrt(n) — init ~1/sqrt(in_dim) Xavier/Kaiming
            l2_norm = l2 / (n ** 0.5)
            out[name] = {"l2": l2, "l2_per_sqrt_n": l2_norm, "mean_abs": mean_abs,
                         "shape": list(p.shape), "numel": n}
    return out


def compute_activation_stats(model, loader, device: str, name: str) -> dict:
    """Capture activation norms at each block output. Mean over batches."""
    model.eval()
    stats: dict = {"audio_proj_norm": [], "visual_norm": [],
                   "fused_norm": [], "gate_weights": []}
    with torch.no_grad():
        for batch in loader:
            v = batch["visual"].to(device)
            a_raw = batch["audio"].to(device)
            a_proj = model.audio_projection(a_raw)
            gate = model.gate_network(v, a_proj)
            w_v = gate[:, 0:1]
            w_a = gate[:, 1:2]
            fused = torch.cat([w_v * v, w_a * a_proj], dim=-1)
            stats["audio_proj_norm"].append(a_proj.norm(dim=-1).cpu().numpy())
            stats["visual_norm"].append(v.norm(dim=-1).cpu().numpy())
            stats["fused_norm"].append(fused.norm(dim=-1).cpu().numpy())
            stats["gate_weights"].append(gate.cpu().numpy())
    out = {}
    for k, arrs in stats.items():
        arr = np.concatenate(arrs, axis=0)
        if arr.ndim == 1:
            out[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "min": float(arr.min()), "max": float(arr.max())}
        else:
            out[k] = {"mean_w_v": float(arr[:, 0].mean()),
                      "mean_w_a": float(arr[:, 1].mean()),
                      "std_w_v": float(arr[:, 0].std()),
                      "std_w_a": float(arr[:, 1].std())}
    return out


def compute_head_prediction_stats(model, loader, device: str) -> dict:
    """VA prediction std and MSE-vs-mean — signature of overfit."""
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch in loader:
            v = batch["visual"].to(device)
            a = batch["audio"].to(device)
            out = model(v, a)
            preds.append(out["va_pred"].cpu().numpy())
            tgts.append(batch["va_target"].numpy())
    preds = np.concatenate(preds, axis=0)
    tgts = np.concatenate(tgts, axis=0)
    resid = preds - tgts
    return {
        "pred_mean_v": float(preds[:, 0].mean()),
        "pred_mean_a": float(preds[:, 1].mean()),
        "pred_std_v": float(preds[:, 0].std()),
        "pred_std_a": float(preds[:, 1].std()),
        "target_std_v": float(tgts[:, 0].std()),
        "target_std_a": float(tgts[:, 1].std()),
        "resid_std_v": float(resid[:, 0].std()),
        "resid_std_a": float(resid[:, 1].std()),
        "mse_v": float((resid[:, 0] ** 2).mean()),
        "mse_a": float((resid[:, 1] ** 2).mean()),
    }


def main():
    cfg = TrainLirisConfig()
    cfg.feature_file = str(FEATS)
    print(f"[cfg] features={cfg.feature_file}")
    device = "cpu"

    # Load model (re-init then overwrite with trained weights)
    model_init = AutoEQModelLiris(cfg)
    model_trained = AutoEQModelLiris(cfg)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model_trained.load_state_dict(ckpt["model"])
    print(f"[ckpt] best epoch={ckpt['epoch']}  "
          f"val_ccc={ckpt['val_metrics']['mean_ccc']:.4f}")

    # 1. Parameter count
    print("\n=== 1. Parameter count per block ===")
    p = count_params(model_trained)
    for name in ("audio_projection", "gate_network", "va_head", "mood_head"):
        print(f"  {name:<18} {p[name]:>9,}  ({p['_pct'][name]:>5.2f}%)")
    print(f"  {'TOTAL':<18} {p['_total']:>9,}")

    # 2. Weight L2 growth init → trained
    print("\n=== 2. Weight L2 growth (init → trained) ===")
    init_w = weight_l2_stats(model_init)
    trained_w = weight_l2_stats(model_trained)
    for k in init_w:
        ini = init_w[k]["l2_per_sqrt_n"]
        tra = trained_w[k]["l2_per_sqrt_n"]
        ratio = tra / max(ini, 1e-9)
        print(f"  {k:<35} init_L2/√n={ini:.4f}  trained={tra:.4f}  ratio={ratio:.2f}×  shape={init_w[k]['shape']}")

    # 3. Activation stats on train vs val
    print("\n=== 3. Activation stats (train vs val) ===")
    features = torch.load(FEATS, map_location=device, weights_only=False)
    splits = official_split(META)
    train_loader = DataLoader(
        PrecomputedLirisDataset(splits["train"], features),
        batch_size=256, shuffle=False, num_workers=0,
        collate_fn=MixupTargetShrinkageCollator(cfg, active=False),
    )
    val_loader = DataLoader(
        PrecomputedLirisDataset(splits["val"], features),
        batch_size=256, shuffle=False, num_workers=0,
        collate_fn=MixupTargetShrinkageCollator(cfg, active=False),
    )
    tr_act = compute_activation_stats(model_trained, train_loader, device, "train")
    va_act = compute_activation_stats(model_trained, val_loader, device, "val")
    for k in tr_act:
        print(f"  {k:<18} train: {tr_act[k]}")
        print(f"  {' ':<18} val  : {va_act[k]}")

    # 4. Prediction stats
    print("\n=== 4. Prediction stats (train vs val) ===")
    tr_pred = compute_head_prediction_stats(model_trained, train_loader, device)
    va_pred = compute_head_prediction_stats(model_trained, val_loader, device)
    for k in tr_pred:
        print(f"  {k:<15} train={tr_pred[k]:+.4f}  val={va_pred[k]:+.4f}  Δ={va_pred[k]-tr_pred[k]:+.4f}")

    # Save
    out = {
        "checkpoint": str(CKPT),
        "best_epoch": ckpt["epoch"],
        "val_ccc_at_best": ckpt["val_metrics"]["mean_ccc"],
        "param_count": p,
        "weight_l2_growth": {k: {"init": init_w[k]["l2_per_sqrt_n"],
                                  "trained": trained_w[k]["l2_per_sqrt_n"],
                                  "ratio": trained_w[k]["l2_per_sqrt_n"] / max(init_w[k]["l2_per_sqrt_n"], 1e-9)}
                              for k in init_w},
        "activation_train": tr_act,
        "activation_val": va_act,
        "prediction_train": tr_pred,
        "prediction_val": va_pred,
    }
    (REPO / "runs" / "phase2a" / "overfit_location_diagnostic.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"\n[saved] {REPO / 'runs/phase2a/overfit_location_diagnostic.json'}")


if __name__ == "__main__":
    main()
