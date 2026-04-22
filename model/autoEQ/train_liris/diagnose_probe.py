"""Ridge linear probe + per-film val CCC + feature ceiling estimate.

Two questions:
  A. Does a simple Ridge on concat(X-CLIP, PANNs) already reach V2 ceiling (0.356)?
     If YES → feature saturation. Model complexity is wasted.
     If NO → model adds value; ceiling might be pushable via better training.
  B. How does V2 val CCC distribute across 40 val films? If a few films dominate
     error, we have heteroskedastic difficulty; else ceiling is uniform.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
META = ROOT / "dataset" / "autoEQ" / "liris" / "liris_metadata.csv"
FEATS = ROOT / "data" / "features" / "liris_panns" / "features.pt"
V2_BEST = ROOT / "runs" / "phase2a" / "baseline_v2_s42" / "best.pt"
OUT = ROOT / "runs" / "phase2a" / "probe_diagnostics.json"


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0


def mean_ccc(pred: np.ndarray, true: np.ndarray) -> dict:
    cv = ccc(pred[:, 0], true[:, 0])
    ca = ccc(pred[:, 1], true[:, 1])
    return {"mean_ccc": (cv + ca) / 2, "ccc_v": cv, "ccc_a": ca}


def build_feature_matrix(df: pd.DataFrame, features: dict) -> np.ndarray:
    X = []
    missing = 0
    for name in df.name.values:
        if name in features:
            f = features[name]
            v = f["xclip"].numpy() if isinstance(f["xclip"], torch.Tensor) else f["xclip"]
            a = f["panns"].numpy() if isinstance(f["panns"], torch.Tensor) else f["panns"]
            X.append(np.concatenate([v, a]))
        else:
            missing += 1
            X.append(np.zeros(512 + 2048))
    if missing:
        print(f"[warn] {missing} clips missing features — used zeros")
    return np.stack(X)


def main():
    df = pd.read_csv(META)
    print(f"[load] meta {len(df)}")
    features = torch.load(FEATS, map_location="cpu", weights_only=False)
    print(f"[load] features {len(features)}")

    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    X_tr = build_feature_matrix(tr, features)
    X_va = build_feature_matrix(va, features)
    y_tr = tr[["v_norm", "a_norm"]].values
    y_va = va[["v_norm", "a_norm"]].values
    print(f"X_tr={X_tr.shape} X_va={X_va.shape}")

    # ── Standardize ──
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)

    # ── A1. Ridge probe on full concat ──
    print("\n[A1] Ridge on concat(X-CLIP 512 + PANNs 2048):")
    alpha_results = {}
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        pred_va = model.predict(X_va_s)
        pred_tr = model.predict(X_tr_s)
        m_va = mean_ccc(pred_va, y_va)
        m_tr = mean_ccc(pred_tr, y_tr)
        alpha_results[alpha] = {"val": m_va, "train": m_tr, "gap": m_tr["mean_ccc"] - m_va["mean_ccc"]}
        print(f"  alpha={alpha:>7.1f}  val CCC={m_va['mean_ccc']:.4f} (v={m_va['ccc_v']:.4f}, a={m_va['ccc_a']:.4f})  "
              f"train CCC={m_tr['mean_ccc']:.4f}  gap={m_tr['mean_ccc']-m_va['mean_ccc']:.4f}")

    # ── A2. Ridge on visual-only ──
    print("\n[A2] Ridge on X-CLIP 512 only:")
    X_tr_v = X_tr_s[:, :512]
    X_va_v = X_va_s[:, :512]
    best_v = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_v, y_tr)
        pred = model.predict(X_va_v)
        m = mean_ccc(pred, y_va)
        if best_v is None or m["mean_ccc"] > best_v["mean_ccc"]:
            best_v = {"alpha": alpha, **m}
        print(f"  alpha={alpha:>6.1f}  val CCC={m['mean_ccc']:.4f} (v={m['ccc_v']:.4f}, a={m['ccc_a']:.4f})")

    # ── A3. Ridge on audio-only ──
    print("\n[A3] Ridge on PANNs 2048 only:")
    X_tr_a = X_tr_s[:, 512:]
    X_va_a = X_va_s[:, 512:]
    best_a = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_a, y_tr)
        pred = model.predict(X_va_a)
        m = mean_ccc(pred, y_va)
        if best_a is None or m["mean_ccc"] > best_a["mean_ccc"]:
            best_a = {"alpha": alpha, **m}
        print(f"  alpha={alpha:>6.1f}  val CCC={m['mean_ccc']:.4f} (v={m['ccc_v']:.4f}, a={m['ccc_a']:.4f})")

    # ── Best-probe summary vs V2 ──
    print("\n[summary] best probe vs V2 model:")
    best_full = max(alpha_results.items(), key=lambda kv: kv[1]["val"]["mean_ccc"])
    b = best_full[1]["val"]
    print(f"  Ridge full concat (α={best_full[0]}):  val CCC={b['mean_ccc']:.4f} (v={b['ccc_v']:.4f}, a={b['ccc_a']:.4f})")
    print(f"  Ridge visual-only (α={best_v['alpha']}): val CCC={best_v['mean_ccc']:.4f} (v={best_v['ccc_v']:.4f}, a={best_v['ccc_a']:.4f})")
    print(f"  Ridge audio-only (α={best_a['alpha']}):  val CCC={best_a['mean_ccc']:.4f} (v={best_a['ccc_v']:.4f}, a={best_a['ccc_a']:.4f})")
    print(f"  V2 model (3-seed avg):              val CCC=0.3564 (v=0.3220, a=0.3909)")

    # ── B. Per-film val CCC with best Ridge model ──
    print("\n[B] Per-film val CCC with Ridge (α=best):")
    best_alpha = best_full[0]
    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)
    pred_va = model.predict(X_va_s)
    va_df = va.copy()
    va_df["pred_v"] = pred_va[:, 0]
    va_df["pred_a"] = pred_va[:, 1]
    film_cccs = []
    for fid, grp in va_df.groupby("film_id"):
        if len(grp) < 5:
            continue
        film_cccs.append({
            "film_id": str(fid),
            "n_clips": int(len(grp)),
            "ccc_v": round(ccc(grp.pred_v.values, grp.v_norm.values), 3),
            "ccc_a": round(ccc(grp.pred_a.values, grp.a_norm.values), 3),
            "v_norm_std": round(float(grp.v_norm.std()), 3),
            "a_norm_std": round(float(grp.a_norm.std()), 3),
        })
    film_cccs.sort(key=lambda x: (x["ccc_v"] + x["ccc_a"]) / 2)
    print(f"  n_val_films = {len(film_cccs)}")
    print("  worst 5 films:")
    for f in film_cccs[:5]:
        mm = (f["ccc_v"] + f["ccc_a"]) / 2
        print(f"    film {f['film_id']:<30}  n={f['n_clips']:>3}  CCC={mm:.3f} (v={f['ccc_v']:+.3f}, a={f['ccc_a']:+.3f})  label std v={f['v_norm_std']:.2f} a={f['a_norm_std']:.2f}")
    print("  best 5 films:")
    for f in film_cccs[-5:]:
        mm = (f["ccc_v"] + f["ccc_a"]) / 2
        print(f"    film {f['film_id']:<30}  n={f['n_clips']:>3}  CCC={mm:.3f} (v={f['ccc_v']:+.3f}, a={f['ccc_a']:+.3f})  label std v={f['v_norm_std']:.2f} a={f['a_norm_std']:.2f}")

    # Distribution of per-film CCC
    mms = np.array([(f["ccc_v"] + f["ccc_a"]) / 2 for f in film_cccs])
    print(f"\n  per-film mean_CCC distribution:")
    print(f"    mean={mms.mean():.3f}  std={mms.std():.3f}  median={np.median(mms):.3f}")
    print(f"    <0.0: {int((mms<0).sum())}/{len(mms)}  <0.2: {int((mms<0.2).sum())}/{len(mms)}  >0.5: {int((mms>0.5).sum())}/{len(mms)}")

    # ── Save ──
    report = {
        "ridge_full_concat": {
            "per_alpha": {float(a): {"val": r["val"], "train": r["train"], "gap": r["gap"]}
                          for a, r in alpha_results.items()},
            "best_alpha": float(best_full[0]),
            "best_val_ccc": best_full[1]["val"]["mean_ccc"],
        },
        "ridge_visual_only_best": best_v,
        "ridge_audio_only_best": best_a,
        "v2_model_val_ccc_ref": 0.3564,
        "per_film_ridge": {
            "n_films": len(film_cccs),
            "mean_ccc_mean": float(mms.mean()),
            "mean_ccc_std": float(mms.std()),
            "mean_ccc_median": float(np.median(mms)),
            "n_below_zero": int((mms < 0).sum()),
            "n_below_0.2": int((mms < 0.2).sum()),
            "n_above_0.5": int((mms > 0.5).sum()),
            "films": film_cccs,
        },
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
