"""smoothing.py — EMA 시간축 스무딩 + 전환 유형별 크로스페이드.

작업 12~13 (Day 8~9):
- ema_smooth_with_scene_reset: 씬 경계에서 리셋되는 EMA (V/A + Mood만)
- aggregate_windows_to_scene: EMA 후 윈도우들을 씬 단위로 평균
- get_eq_at_time: cut(0.3초) vs dissolve(2.0초) 차등 크로스페이드
"""

from __future__ import annotations

import numpy as np

# ────────────────────────────────────────────────────────
# EMA 시간축 스무딩
# ────────────────────────────────────────────────────────
def ema_smooth_with_scene_reset(
    windows: list[dict],
    scenes: list[dict],
    alpha: float = 0.3,
    cold_start: int = 3,
) -> list[dict]:
    """씬 경계에서 리셋되는 EMA 스무딩.

    적용 대상:
        ✅ V/A, Mood 확률 (씬 경계 리셋)
        ❌ Gate 가중치, Congruence (윈도우별 분석용으로 보존)

    Args:
        windows: [{start_sec, end_sec, va: {valence, arousal}, mood_probs}, ...]
        scenes: 병합된 씬 리스트
        alpha: 0에 가까울수록 부드러움 (기본 0.3)
        cold_start: 씬 시작 후 이만큼은 EMA 없이 원본 사용
    """
    smoothed = []

    for scene in scenes:
        scene_windows = [
            w for w in windows
            if scene["start_sec"] <= w["start_sec"] < scene["end_sec"]
        ]
        prev_va = None
        prev_mood = None

        for i, w in enumerate(scene_windows):
            if i < cold_start or prev_va is None:
                smooth_va = dict(w["va"])
                smooth_mood = list(w["mood_probs"])
            else:
                smooth_va = {
                    "valence": alpha * w["va"]["valence"] + (1 - alpha) * prev_va["valence"],
                    "arousal": alpha * w["va"]["arousal"] + (1 - alpha) * prev_va["arousal"],
                }
                smooth_mood = [
                    alpha * w["mood_probs"][k] + (1 - alpha) * prev_mood[k]
                    for k in range(len(w["mood_probs"]))
                ]
                s = sum(smooth_mood)
                if s > 0:
                    smooth_mood = [v / s for v in smooth_mood]

            smoothed.append({**w, "ema": {"va": smooth_va, "mood_probs": smooth_mood}})
            prev_va = smooth_va
            prev_mood = smooth_mood

    return smoothed


def aggregate_windows_to_scene(
    scene: dict, smoothed_windows: list[dict], mood_categories: list[str]
) -> dict | None:
    """씬에 속한 EMA 후 윈도우들을 평균내 씬 단위 aggregated 생성.

    Returns:
        dict (씬의 aggregated 필드에 그대로 사용 가능) or None (윈도우 없음)
    """
    in_scene = [
        w for w in smoothed_windows
        if scene["start_sec"] <= w["start_sec"] < scene["end_sec"]
    ]
    if not in_scene:
        return None

    mood_arr = np.array([w["ema"]["mood_probs"] for w in in_scene])
    mood_mean = mood_arr.mean(axis=0)
    if mood_mean.sum() > 0:
        mood_mean = mood_mean / mood_mean.sum()

    valences = [w["ema"]["va"]["valence"] for w in in_scene]
    arousals = [w["ema"]["va"]["arousal"] for w in in_scene]

    max_idx = int(np.argmax(mood_mean))
    category = mood_categories[max_idx]

    return {
        "valence": float(np.mean(valences)),
        "arousal": float(np.mean(arousals)),
        "category": category,
        "mood_probs_mean": {cat: float(p) for cat, p in zip(mood_categories, mood_mean)},
    }


# ────────────────────────────────────────────────────────
# 전환 유형별 차등 크로스페이드
# ────────────────────────────────────────────────────────
CROSSFADE_DURATIONS = {
    "cut": 0.3,
    "dissolve": 2.0,
}


def sigmoid_crossfade(prev_eq: np.ndarray, next_eq: np.ndarray, progress: float) -> np.ndarray:
    """progress: 0.0(이전) → 1.0(다음), S자 곡선."""
    t = 1 / (1 + np.exp(-12 * (progress - 0.5)))
    return prev_eq * (1 - t) + next_eq * t


def get_crossfade_duration(transition_out: str) -> float:
    return CROSSFADE_DURATIONS.get(transition_out, 0.3)


def get_eq_at_time(current_sec: float, scenes_eq: list[dict]) -> np.ndarray:
    """현재 시각의 EQ 게인 (크로스페이드 적용).

    scenes_eq: [{'start_sec', 'end_sec', 'transition_out', 'effective_gains'}, ...]
    """
    for i, scene in enumerate(scenes_eq):
        if scene["start_sec"] <= current_sec < scene["end_sec"]:
            next_scene = scenes_eq[i + 1] if i + 1 < len(scenes_eq) else None
            if next_scene:
                fade_dur = get_crossfade_duration(scene["transition_out"])
                fade_start = scene["end_sec"] - fade_dur
                if current_sec >= fade_start:
                    progress = (current_sec - fade_start) / fade_dur
                    return sigmoid_crossfade(
                        scene["effective_gains"],
                        next_scene["effective_gains"],
                        progress,
                    )
            return scene["effective_gains"]
    return np.zeros(10)


def get_eq_at_time_simple(current_sec: float, scenes_eq: list[dict]) -> np.ndarray:
    """크로스페이드 없는 단순 버전 (Day 4~5용 폴백)."""
    for scene in scenes_eq:
        if scene["start_sec"] <= current_sec < scene["end_sec"]:
            return scene["effective_gains"]
    return np.zeros(10)


# ────────────────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────────────────
def verify_ema() -> None:
    from .eq_engine import MOOD_CATEGORIES

    scenes = [
        {"start_sec": 0, "end_sec": 5},
        {"start_sec": 5, "end_sec": 10},
    ]

    windows = []
    for i in range(5):
        windows.append({
            "start_sec": i, "end_sec": i + 1,
            "va": {"valence": 0.1 + (0.8 if i % 2 else 0), "arousal": 0.5},
            "mood_probs": [1 / 7] * 7,
        })
    for i in range(5):
        windows.append({
            "start_sec": 5 + i, "end_sec": 6 + i,
            "va": {"valence": -0.5, "arousal": 0.3},
            "mood_probs": [1 / 7] * 7,
        })

    result = ema_smooth_with_scene_reset(windows, scenes, alpha=0.3, cold_start=3)
    assert len(result) == len(windows)

    # cold start: 처음 3개는 원본 유지
    for i in range(3):
        assert result[i]["ema"]["va"] == result[i]["va"]
    # 4번째부터는 스무딩 적용
    assert result[3]["ema"]["va"]["valence"] != windows[3]["va"]["valence"]
    # 씬 경계에서 리셋
    assert result[5]["ema"]["va"] == result[5]["va"]
    # mood 확률 합 = 1
    for r in result:
        s = sum(r["ema"]["mood_probs"])
        assert abs(s - 1.0) < 0.001

    # 집계 함수
    agg = aggregate_windows_to_scene(scenes[0], result, MOOD_CATEGORIES)
    assert agg is not None
    assert "mood_probs_mean" in agg
    assert "valence" in agg
    assert "category" in agg
    assert abs(sum(agg["mood_probs_mean"].values()) - 1.0) < 0.01

    print(f"  ✓ Cold start (3개) 원본 유지")
    print(f"  ✓ 씬 경계 리셋 동작")
    print(f"  ✓ Mood 확률 합 1.0 보장")
    print(f"  ✓ 윈도우 EMA → 씬 mood_probs_mean 집계 정상")


def verify_crossfade() -> None:
    from .eq_engine import EQ_PRESETS

    prev = np.array([0.0] * 10)
    next_ = np.array([10.0] * 10)

    # sigmoid 계수 -12 기준 tail 값: progress=0에서 ≈0.025, progress=1에서 ≈9.975
    # 청감상 자연스러운 전환을 위해 의도된 값이며 0/10에 정확히 맞추지 않음
    r0 = sigmoid_crossfade(prev, next_, 0.0)
    assert r0[0] < 0.05, f"sigmoid 시작값이 너무 큼: {r0[0]:.4f}"
    r5 = sigmoid_crossfade(prev, next_, 0.5)
    assert abs(r5[0] - 5.0) < 0.01
    r1 = sigmoid_crossfade(prev, next_, 1.0)
    assert r1[0] > 9.95, f"sigmoid 끝값이 너무 작음: {r1[0]:.4f}"

    # monotone
    vals = [sigmoid_crossfade(prev, next_, p / 10)[0] for p in range(11)]
    for i in range(len(vals) - 1):
        assert vals[i + 1] >= vals[i]

    assert get_crossfade_duration("cut") == 0.3
    assert get_crossfade_duration("dissolve") == 2.0

    scenes_eq = [
        {"start_sec": 0, "end_sec": 5, "transition_out": "cut",
         "effective_gains": EQ_PRESETS["Peacefulness"]},
        {"start_sec": 5, "end_sec": 10, "transition_out": "cut",
         "effective_gains": EQ_PRESETS["Tension"]},
    ]

    g1 = get_eq_at_time(2.0, scenes_eq)
    assert np.allclose(g1, EQ_PRESETS["Peacefulness"])

    g2 = get_eq_at_time(4.85, scenes_eq)
    assert not np.allclose(g2, EQ_PRESETS["Peacefulness"])
    assert not np.allclose(g2, EQ_PRESETS["Tension"])

    g3 = get_eq_at_time(7.0, scenes_eq)
    assert np.allclose(g3, EQ_PRESETS["Tension"])

    # dissolve가 cut보다 크로스페이드가 길게 적용됨
    scenes_eq_diss = [
        {"start_sec": 0, "end_sec": 5, "transition_out": "dissolve",
         "effective_gains": EQ_PRESETS["Peacefulness"]},
        {"start_sec": 5, "end_sec": 10, "transition_out": "cut",
         "effective_gains": EQ_PRESETS["Tension"]},
    ]
    g_cut_mid = get_eq_at_time(4.85, scenes_eq)
    g_diss_mid = get_eq_at_time(4.85, scenes_eq_diss)
    assert not np.allclose(g_cut_mid, g_diss_mid)

    print(f"  ✓ sigmoid 경계값 정상")
    print(f"  ✓ cut 0.3초 / dissolve 2.0초 차등 적용")


if __name__ == "__main__":
    print("=" * 60)
    print("Smoothing 검증")
    print("=" * 60)
    print("\n[EMA + 집계]")
    verify_ema()
    print("\n[크로스페이드]")
    verify_crossfade()
    print("\n🎉 Smoothing 모든 검증 통과")
