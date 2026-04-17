"""clip_probs_rebuild.py — recover full CLIP probability distribution.

`scene_labeler.py` computes but discards the 9/10-dim CLIP softmax
vector (see `classify_scene_clip` return value unpacked to `_` on
line 290). This tool re-runs CLIP on the **cached mid-scene frames**
produced by `scene_labeler_review.extract_frames` and writes the
full `top_k` / `full_probs` distribution per scene to
`outputs/scene_labels/clip_probs_{name}.json`.

Read-only with respect to scene_labeler.py / paths.py / analyzer.py.
Only imports `CANDIDATE_LABELS`, `LABEL_TO_SCENE_NAME`, `load_clip`.

Usage:
    python -m model.autoEQ.inference.clip_probs_rebuild --name topgun
    python -m model.autoEQ.inference.clip_probs_rebuild --all
    python -m model.autoEQ.inference.clip_probs_rebuild --all --topk 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from .paths import SCENE_LABELS_DIR, scene_labels_json
from .scene_labeler import CANDIDATE_LABELS, LABEL_TO_SCENE_NAME, load_clip


def _trailer_key(name: str) -> str:
    return f"trailer_{name}"


def _frames_dir(name: str) -> Path:
    return SCENE_LABELS_DIR / f"review_frames_{name}"


def run_for(name: str, *, topk: int = 5) -> dict | None:
    """Re-run CLIP on cached frames of a single trailer. Returns doc or None on fatal setup issue."""
    key = _trailer_key(name)
    if key not in CANDIDATE_LABELS:
        print(f"  ✗ {name}: '{key}' not in CANDIDATE_LABELS (known: {list(CANDIDATE_LABELS)})")
        return None

    candidates: list[str] = CANDIDATE_LABELS[key]
    short_labels: list[str] = [LABEL_TO_SCENE_NAME[c] for c in candidates]

    auto_path = scene_labels_json(name, "auto")
    if not auto_path.exists():
        print(f"  ✗ {name}: {auto_path} not found — run scene_labeler first")
        return None
    auto = json.loads(auto_path.read_text(encoding="utf-8"))

    frames_dir = _frames_dir(name)
    frames_dir_exists = frames_dir.exists()
    if not frames_dir_exists:
        print(f"  ⚠ {frames_dir} not found — every scene will be marked frame_missing")

    model, processor = load_clip()

    print(f"\n=== clip_probs for '{name}' "
          f"({len(auto)} scenes · {len(candidates)} candidate prompts · topk={topk}) ===")

    scene_results: list[dict] = []
    skipped: list[int] = []
    errors: list[int] = []

    for idx, s in enumerate(auto):
        fp = frames_dir / f"scene_{idx:03d}.jpg"
        entry: dict = {
            "idx": idx,
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
        }
        if not fp.exists():
            print(f"  ⚠ scene {idx:>2}: frame missing ({fp.name}) — skip")
            entry["error"] = "frame_missing"
            entry["frame"] = str(fp)
            scene_results.append(entry)
            skipped.append(idx)
            continue

        try:
            image = Image.open(fp).convert("RGB")
            inputs = processor(
                text=candidates, images=image,
                return_tensors="pt", padding=True,
            )
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).squeeze().tolist()

            order = sorted(range(len(candidates)), key=lambda i: -probs[i])
            topk_list = [
                {
                    "label": short_labels[i],
                    "prompt": candidates[i],
                    "prob": round(float(probs[i]), 4),
                }
                for i in order[:topk]
            ]
            full_probs = [
                {"label": short_labels[i], "prob": round(float(probs[i]), 4)}
                for i in range(len(candidates))
            ]

            entry["frame"] = str(fp.relative_to(SCENE_LABELS_DIR)).replace("\\", "/")
            entry["top1"] = topk_list[0]
            entry["top_k"] = topk_list
            entry["full_probs"] = full_probs

            auto_name = s["scene_name_auto"].rstrip("?")
            marker = "✓" if topk_list[0]["label"] == auto_name else "~"
            p1 = topk_list[0]["prob"]
            p2 = topk_list[1]["prob"] if len(topk_list) >= 2 else 0.0
            print(f"  {marker} scene {idx:>2}: top1={topk_list[0]['label']:<14} "
                  f"p1={p1:.3f}  top2={topk_list[1]['label'] if len(topk_list)>=2 else '-':<14} "
                  f"p2={p2:.3f}  (auto={auto_name})")

        except Exception as e:
            print(f"  ✗ scene {idx:>2}: {type(e).__name__}: {e}")
            entry["error"] = f"{type(e).__name__}: {e}"
            scene_results.append(entry)
            errors.append(idx)
            continue

        scene_results.append(entry)

    processed = len(auto) - len(skipped) - len(errors)
    doc = {
        "metadata": {
            "trailer": name,
            "video_key": key,
            "candidate_labels": candidates,
            "candidate_short_names": short_labels,
            "topk": topk,
            "total_scenes": len(auto),
            "processed_scenes": processed,
            "skipped_missing_frame": skipped,
            "processing_errors": errors,
            "auto_source": auto_path.name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": (
                "CLIP re-run on review_frames_*/scene_NNN.jpg (q:v 3). "
                "Minor numeric drift vs scene_labeler.py original (q:v 2) possible; "
                "top-1 label is expected to match in ≥~95% of high-confidence scenes."
            ),
        },
        "scenes": scene_results,
    }

    out_path = SCENE_LABELS_DIR / f"clip_probs_{name}.json"
    out_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✓ saved: {out_path}  ({out_path.stat().st_size:,}B · processed={processed} · skipped={len(skipped)} · errors={len(errors)})")
    return doc


def validate(name: str, doc: dict) -> dict:
    auto = json.loads(scene_labels_json(name, "auto").read_text(encoding="utf-8"))
    n_auto = len(auto)
    n_doc = len(doc["scenes"])
    valid = [s for s in doc["scenes"] if "top1" in s]
    matches = []
    mismatches = []
    for s in valid:
        auto_name = auto[s["idx"]]["scene_name_auto"].rstrip("?")
        if s["top1"]["label"] == auto_name:
            matches.append(s["idx"])
        else:
            mismatches.append({
                "idx": s["idx"],
                "new_top1": s["top1"]["label"],
                "new_prob": s["top1"]["prob"],
                "auto_name": auto_name,
                "auto_prob": auto[s["idx"]]["clip_confidence"],
                "new_top2": s["top_k"][1]["label"] if len(s["top_k"]) >= 2 else None,
                "new_top2_prob": s["top_k"][1]["prob"] if len(s["top_k"]) >= 2 else None,
            })
    return {
        "count_match": n_doc == n_auto,
        "n_auto": n_auto,
        "n_doc": n_doc,
        "n_valid": len(valid),
        "top1_match": len(matches),
        "top1_match_rate": (len(matches) / max(1, len(valid))),
        "mismatches": mismatches,
    }


def title_near_top2(name: str, doc: dict, *, delta: float = 0.1) -> dict:
    """For scenes whose CLIP top1 is 'Title', count how many have top1-top2 gap < delta
    and summarise which labels dominate top-2 in that population."""
    valid_titles = [
        s for s in doc["scenes"]
        if "top1" in s and s["top1"]["label"] == "Title"
    ]
    near = []
    for s in valid_titles:
        if len(s["top_k"]) < 2:
            continue
        gap = s["top_k"][0]["prob"] - s["top_k"][1]["prob"]
        if gap < delta:
            near.append({
                "idx": s["idx"],
                "gap": round(gap, 4),
                "top1_prob": s["top_k"][0]["prob"],
                "top2_label": s["top_k"][1]["label"],
                "top2_prob": s["top_k"][1]["prob"],
                "top3_label": s["top_k"][2]["label"] if len(s["top_k"]) >= 3 else None,
                "top3_prob": s["top_k"][2]["prob"] if len(s["top_k"]) >= 3 else None,
            })
    return {
        "title_total": len(valid_titles),
        "delta": delta,
        "near_count": len(near),
        "top2_distribution": dict(Counter(n["top2_label"] for n in near)),
        "near_details": near,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rebuild CLIP top-k probs from cached frames")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="Short trailer name, e.g. topgun or lalaland")
    g.add_argument("--all", action="store_true")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--near-delta", type=float, default=0.1,
                   help="Threshold for 'Title top1-top2 near' detection")
    args = p.parse_args(argv)

    SCENE_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    names = ["topgun", "lalaland"] if args.all else [args.name]

    summaries: dict[str, dict] = {}
    for name in names:
        doc = run_for(name, topk=args.topk)
        if doc is None:
            continue
        v = validate(name, doc)
        t = title_near_top2(name, doc, delta=args.near_delta)
        summaries[name] = {"doc": doc, "validate": v, "title_near": t}

    # ============================================================
    print("\n" + "=" * 70)
    print("CLIP PROBS REBUILD — FINAL SUMMARY")
    print("=" * 70)
    for name, summ in summaries.items():
        out_path = SCENE_LABELS_DIR / f"clip_probs_{name}.json"
        size = out_path.stat().st_size if out_path.exists() else 0
        v = summ["validate"]
        t = summ["title_near"]

        print(f"\n--- {name} ---")
        print(f"  file: {out_path.name}  size: {size:,}B")
        print(f"  scenes: {v['n_doc']}/{v['n_auto']}  valid: {v['n_valid']}  "
              f"count_match={v['count_match']}")
        mr = v["top1_match_rate"] * 100
        print(f"  top-1 vs auto scene_name_auto: {v['top1_match']}/{v['n_valid']} = {mr:.1f}%")
        if v["mismatches"]:
            print(f"  mismatches: {len(v['mismatches'])} (showing up to 3):")
            for m in v["mismatches"][:3]:
                print(f"    idx={m['idx']:>2}  new top1={m['new_top1']}({m['new_prob']:.3f})  "
                      f"vs auto={m['auto_name']}({m['auto_prob']:.3f})  "
                      f"new top2={m['new_top2']}({m['new_top2_prob']})")
        print(f"  Title scenes: total={t['title_total']}  "
              f"top1-top2 gap<{t['delta']}: {t['near_count']}")
        if t["top2_distribution"]:
            print(f"    top-2 label distribution among near: {t['top2_distribution']}")
        if t["near_details"]:
            print(f"    sample near details (first 5):")
            for n in t["near_details"][:5]:
                print(f"      idx={n['idx']:>2}  gap={n['gap']:.3f}  "
                      f"top1=Title({n['top1_prob']:.3f})  "
                      f"top2={n['top2_label']}({n['top2_prob']:.3f})")

    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
