"""run_demo_visualizer.py — 시연용 EQ 시각화 생성 드라이버.

timeline.json(들)에서 씬별 mood/density를 꺼내 visualize_all 형식으로 변환 후
eq_visualizer.visualize_all 호출.

실행: venv/Scripts/python.exe tools/run_demo_visualizer.py
"""

from __future__ import annotations

import json
from pathlib import Path

from model.autoEQ.inference.eq_visualizer import visualize_all


TIMELINE_SOURCES = {
    "topgun":   "backend/data/jobs/fe2ecad8-dc25-4131-adfe-ffeea6d977a1/timeline.json",
    "lalaland": "backend/data/jobs/lalaland-demo/timeline.json",
}


def timeline_to_manual_labels(timeline_path: Path) -> list[dict]:
    tl = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    labels = []
    for s in tl["scenes"]:
        mood = s["aggregated"]["category"]
        prob = s["aggregated"]["mood_probs_mean"].get(mood, 0.65)
        density = s["dialogue"]["density"]
        labels.append({
            "start": s["start_sec"],
            "end": s["end_sec"],
            "scene_name": f"scene_{s['scene_id']:03d}",
            "mood": mood,
            "prob": float(prob),
            "density": float(density),
        })
    return labels


def main() -> None:
    manual_mood_labels = {}
    for name, path in TIMELINE_SOURCES.items():
        p = Path(path)
        if not p.exists():
            print(f"건너뜀: {name} (timeline 없음)")
            continue
        labels = timeline_to_manual_labels(p)
        manual_mood_labels[name] = labels
        print(f"  {name}: {len(labels)}개 씬 로드")

    if not manual_mood_labels:
        print("timeline 없음 — 중단")
        return

    visualize_all(manual_mood_labels)


if __name__ == "__main__":
    main()
