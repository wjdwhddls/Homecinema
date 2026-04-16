"""scene_splitter.py — PySceneDetect 기반 씬 분할 + 전환 유형 분류.

작업 2~3 (Day 2):
- detect_scenes_with_transitions: 씬 경계 + cut/dissolve 분류
- merge_short_scenes: 2초 미만 씬을 인접 씬에 병합 (dissolve 보존)
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
from scenedetect import detect, ContentDetector

from .paths import scenes_json, trailer_path, ensure_dirs
from .utils import get_duration


# ────────────────────────────────────────────────────────
# 씬 분할 + 전환 유형 분류
# ────────────────────────────────────────────────────────
def detect_scenes_with_transitions(video_path, threshold: float = 27.0) -> list[dict]:
    """씬 경계 + 전환 유형(cut/dissolve)을 함께 감지.

    부동소수점 반올림 이슈 주의:
    start와 end를 먼저 round한 뒤, duration은 round된 값들의 차이로 계산해야
    `end_sec - start_sec == duration_sec`이 항상 성립합니다.
    """
    scene_list = detect(
        str(video_path),
        detector=ContentDetector(threshold=threshold),
        show_progress=False,
    )

    scenes = []
    for i, scene in enumerate(scene_list):
        start = round(scene[0].get_seconds(), 2)
        end = round(scene[1].get_seconds(), 2)

        if i < len(scene_list) - 1:
            transition_out = classify_transition(str(video_path), end)
        else:
            transition_out = "cut"

        scenes.append(
            {
                "scene_id": i,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 2),
                "transition_out": transition_out,
            }
        )
    return scenes


def classify_transition(video_path: str, boundary_sec: float, window_sec: float = 0.5) -> str:
    """경계 전후 프레임 변화량으로 cut/dissolve 판정.

    한 프레임에 변화가 집중되면 cut, 여러 프레임에 걸쳐 퍼져있으면 dissolve.
    try/finally로 cap 자원 누수 방지.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return "cut"

        start_frame = max(0, int((boundary_sec - window_sec) * fps))
        end_frame = int((boundary_sec + window_sec) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        diffs = []
        prev_gray = None
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                diffs.append(diff)
            prev_gray = gray

        if not diffs:
            return "cut"

        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        return "cut" if max_diff > mean_diff * 3 else "dissolve"
    finally:
        cap.release()


# ────────────────────────────────────────────────────────
# 짧은 씬 병합
# ────────────────────────────────────────────────────────
def merge_short_scenes(
    scenes: list[dict], min_duration: float = 2.0, preserve_dissolve: bool = True
) -> list[dict]:
    """2초 미만인 씬을 인접 씬에 병합.

    Args:
        scenes: 씬 리스트
        min_duration: 최소 씬 길이 (초)
        preserve_dissolve: True면 병합되는 두 씬 중 하나라도 dissolve면 유지
    """
    MAX_ITERATIONS = 1000
    merged = [dict(s) for s in scenes]

    for _ in range(MAX_ITERATIONS):
        if len(merged) <= 1:
            break
        if all(s["duration_sec"] >= min_duration for s in merged):
            break

        durations = [s["duration_sec"] for s in merged]
        shortest_idx = durations.index(min(durations))

        if durations[shortest_idx] >= min_duration:
            break

        if shortest_idx == 0:
            target = 1
        elif shortest_idx == len(merged) - 1:
            target = len(merged) - 2
        else:
            target = (
                shortest_idx - 1
                if durations[shortest_idx - 1] <= durations[shortest_idx + 1]
                else shortest_idx + 1
            )

        low, high = min(shortest_idx, target), max(shortest_idx, target)

        if preserve_dissolve and (
            merged[low]["transition_out"] == "dissolve"
            or merged[high]["transition_out"] == "dissolve"
        ):
            new_transition = "dissolve"
        else:
            new_transition = merged[high]["transition_out"]

        merged[low] = {
            "scene_id": merged[low]["scene_id"],
            "start_sec": merged[low]["start_sec"],
            "end_sec": merged[high]["end_sec"],
            "duration_sec": round(merged[high]["end_sec"] - merged[low]["start_sec"], 2),
            "transition_out": new_transition,
        }
        merged.pop(high)

    for i, s in enumerate(merged):
        s["scene_id"] = i
    return merged


# ────────────────────────────────────────────────────────
# 검증 함수들
# ────────────────────────────────────────────────────────
def verify_scene_detection(scenes: list[dict], video_path) -> None:
    """씬 분할 결과 검증. 자동 감지 + 범위 기반."""
    video_duration = get_duration(video_path)

    assert len(scenes) >= 1, "씬이 0개"

    required = {"scene_id", "start_sec", "end_sec", "duration_sec", "transition_out"}
    for s in scenes:
        assert required.issubset(s.keys()), f"필드 누락: {s}"

    for s in scenes:
        assert s["transition_out"] in ("cut", "dissolve"), \
            f"잘못된 transition: {s['transition_out']}"

    for i in range(len(scenes) - 1):
        gap = scenes[i + 1]["start_sec"] - scenes[i]["end_sec"]
        assert abs(gap) < 0.1, f"씬 {i}와 {i+1} 사이 {gap}초 공백"

    total = scenes[-1]["end_sec"] - scenes[0]["start_sec"]
    assert abs(total - video_duration) < 1.0, \
        f"총 길이 불일치: {total:.1f}s vs 영상 {video_duration:.1f}s"

    for s in scenes:
        calc = round(s["end_sec"] - s["start_sec"], 2)
        assert abs(s["duration_sec"] - calc) <= 0.02, \
            f"duration 불일치: {s} (재계산: {calc})"

    expected_min = video_duration * 0.4
    expected_max = video_duration * 1.2
    if not (expected_min <= len(scenes) <= expected_max):
        print(
            f"  ⚠️  씬 수 {len(scenes)}개가 예상 범위({expected_min:.0f}~"
            f"{expected_max:.0f})를 벗어남. threshold 조정 검토."
        )

    cuts = sum(1 for s in scenes if s["transition_out"] == "cut")
    dissolves = sum(1 for s in scenes if s["transition_out"] == "dissolve")
    short = sum(1 for s in scenes if s["duration_sec"] < 2.0)
    avg_dur = total / len(scenes)
    print(f"  ✓ 씬 {len(scenes)}개 (cut {cuts} / dissolve {dissolves})")
    print(f"  ✓ 평균 길이 {avg_dur:.2f}초, 2초 미만 {short}개")
    print(f"  ✓ 모든 검증 통과 (영상 길이 {video_duration:.1f}초)")


def verify_merge(
    scenes_before: list[dict], scenes_after: list[dict], min_duration: float = 2.0
) -> None:
    """병합 결과 자동 검증."""
    violations = [s for s in scenes_after if s["duration_sec"] < min_duration]
    assert not violations, f"{min_duration}초 미만 씬 {len(violations)}개 남음"

    total_before = scenes_before[-1]["end_sec"] - scenes_before[0]["start_sec"]
    total_after = scenes_after[-1]["end_sec"] - scenes_after[0]["start_sec"]
    assert abs(total_before - total_after) < 0.01, \
        f"길이 손실: {total_before:.2f} → {total_after:.2f}"

    for i in range(len(scenes_after) - 1):
        gap = scenes_after[i + 1]["start_sec"] - scenes_after[i]["end_sec"]
        assert abs(gap) < 0.01, f"씬 {i}~{i+1} 공백 {gap}초"

    ids = [s["scene_id"] for s in scenes_after]
    assert ids == list(range(len(scenes_after))), "scene_id 불일치"

    assert abs(scenes_after[0]["start_sec"] - scenes_before[0]["start_sec"]) < 0.01
    assert abs(scenes_after[-1]["end_sec"] - scenes_before[-1]["end_sec"]) < 0.01

    d_before = sum(1 for s in scenes_before if s["transition_out"] == "dissolve")
    d_after = sum(1 for s in scenes_after if s["transition_out"] == "dissolve")
    preservation = d_after / d_before * 100 if d_before > 0 else 100
    print(f"  ✓ 씬 {len(scenes_before)} → {len(scenes_after)}개")
    print(f"  ✓ dissolve {d_before} → {d_after}개 ({preservation:.0f}% 보존)")
    print(f"  ✓ 모든 검증 통과")


# ────────────────────────────────────────────────────────
# 단일 영상 처리 헬퍼 (Day 2의 두 영상 처리 루프에서 재사용)
# ────────────────────────────────────────────────────────
def process_video(video_path, name: str, threshold: float = 27.0) -> list[dict]:
    """detect → merge → verify → JSON 저장 까지 한 번에.

    Args:
        video_path: 영상 경로
        name: 저장 키 (예: 'topgun', 'lalaland')

    Returns:
        병합된 씬 리스트
    """
    ensure_dirs()
    print(f"\n=== {name} ({video_path}) ===")
    scenes_raw = detect_scenes_with_transitions(video_path, threshold)
    verify_scene_detection(scenes_raw, video_path)

    scenes_merged = merge_short_scenes(scenes_raw, 2.0, preserve_dissolve=True)
    verify_merge(scenes_raw, scenes_merged)

    out_path = scenes_json(name)
    out_path.write_text(
        json.dumps(scenes_merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  💾 저장: {out_path}")
    return scenes_merged


# ────────────────────────────────────────────────────────
# 직접 실행
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 기본: 두 트레일러 모두 처리
        for name in ["topgun", "lalaland"]:
            video = trailer_path(name)
            if not video.exists():
                print(f"⏭ {video} 없음, 스킵")
                continue
            process_video(video, name)
    elif len(sys.argv) == 3:
        # 단일 영상: python -m ... <video_path> <name>
        process_video(sys.argv[1], sys.argv[2])
    else:
        print("사용법:")
        print("  python -m model.autoEQ.inference.scene_splitter")
        print("    → topgun + lalaland 자동 처리")
        print("  python -m model.autoEQ.inference.scene_splitter <video_path> <name>")
        sys.exit(1)
