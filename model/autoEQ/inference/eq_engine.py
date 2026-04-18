"""eq_engine.py — V3.1/V3.2 EQ 프리셋 + 확률 블렌딩 + 대사 보호.

작업 7~9 (Day 4~5):
- 10밴드 EQ 프리셋 정의 (V3.1 baseline ±3dB / V3.2 dramatic ±4dB)
- Probabilistic Preset Blending (확률 가중 평균)
- Density-aware Gain Modulation (대사 보호: g × (1 - (1-α_d)×density))
- Confidence-based scaling (지배 감정 확률에 따라 강도 조절)
"""

from __future__ import annotations

import numpy as np

# ────────────────────────────────────────────────────────
# 10밴드 정의 + 대사 보호 대상
# ────────────────────────────────────────────────────────
BAND_FREQS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
BAND_Q     = [0.7,  0.7, 1.0, 1.2, 1.4, 1.4,  1.4,  1.2,  0.7,  0.7]
VOICE_BANDS_IDX = [5, 6, 7]  # 대사 대역: B6(1k), B7(2k), B8(4k)

# ────────────────────────────────────────────────────────
# V3.1 Baseline: ±3dB, boost 위주
# ────────────────────────────────────────────────────────
EQ_PRESETS_V3_1 = {
    "Tension":           np.array([+2.0, +2.0, +1.0,  0.0,  0.0, +1.0, +2.5, +2.0,  0.0, -1.0]),
    "Sadness":           np.array([ 0.0, +1.0, +1.0, +1.0,  0.0,  0.0, -2.0, -2.0, -1.5, -1.5]),
    "Peacefulness":      np.array([ 0.0,  0.0, +0.5, +0.5,  0.0,  0.0, -0.5, -1.0, -0.5,  0.0]),
    "Joyful Activation": np.array([-1.0, -1.0,  0.0,  0.0, +1.0, +1.5, +2.0, +2.0, +1.5, +1.0]),
    "Tenderness":        np.array([ 0.0, +1.0, +2.0, +1.5, +0.5,  0.0, -1.0, -1.5, -1.0, -0.5]),
    "Power":             np.array([+2.5, +2.0, +2.0, +1.0, +0.5, +1.0, +1.5, +1.0,  0.0,  0.0]),
    "Wonder":            np.array([ 0.0,  0.0,  0.0, -0.5,  0.0, +0.5, +1.0, +1.5, +1.5, +2.0]),
}

# ────────────────────────────────────────────────────────
# V3.2 Dramatic: ±4dB, boost+cut 대비 구조
# ────────────────────────────────────────────────────────
EQ_PRESETS_V3_2 = {
    "Tension":           np.array([+4.0, +3.5, +1.0, -2.0, -2.5, -1.0, +3.5, +4.0, +1.5, -1.0]),
    "Sadness":           np.array([+1.0, +2.5, +3.0, +2.5, +0.5, -1.0, -3.0, -3.5, -3.0, -2.5]),
    "Peacefulness":      np.array([+0.5, +1.0, +1.5, +1.5,  0.0, -0.5, -1.5, -2.0, -1.0,  0.0]),
    "Joyful Activation": np.array([+2.0, +1.0, -1.0, -0.5, +1.5, +2.5, +3.5, +3.5, +3.0, +2.0]),
    "Tenderness":        np.array([-0.5, +1.0, +2.5, +3.0, +2.0, +0.5, -1.5, -2.5, -2.0, -1.0]),
    "Power":             np.array([+4.0, +3.5, +2.5,  0.0, -1.5, -0.5, +2.0, +3.0, +2.5, +1.0]),
    "Wonder":            np.array([-0.5, -0.5,  0.0, -1.0, -0.5, +1.0, +2.0, +3.0, +3.5, +4.0]),
}

PRESET_VERSIONS = {
    "v3_1": EQ_PRESETS_V3_1,
    "v3_2": EQ_PRESETS_V3_2,
    "v3_3": None,  # V3_3 아래에서 정의 후 채움 (forward declaration 회피)
}

# ────────────────────────────────────────────────────────
# V3.3 Extended: ±6dB, 강한 mood 확대 (Compressor 후처리와 함께)
# 방식 C(하이브리드): PRESET_VERSIONS에는 미등록 — 별도 wrapper에서만 사용.
# 청취 평가 결과에 따라 PRESET_VERSIONS 등록 여부 결정.
# ────────────────────────────────────────────────────────
EQ_PRESETS_V3_3 = {
    # 강한 mood — 시그니처 밴드를 ±6dB까지 확대 (V3.2 V-shape/tilt 유지)
    "Tension":           np.array([+5.0, +5.0, +2.0, -3.0, -3.5, -1.5, +5.5, +6.0, +2.5, -1.0]),
    "Sadness":           np.array([+3.0, +6.0, +6.0, +4.5, +1.0, -1.5, -5.0, -5.0, -4.5, -3.5]),
    "Joyful Activation": np.array([+2.5, +1.5, -1.5, -1.0, +2.0, +4.0, +6.0, +6.0, +4.5, +3.0]),
    "Power":             np.array([+6.0, +6.0, +3.5, +0.5, -2.0, -0.5, +4.0, +5.5, +3.5, +1.5]),
    "Wonder":            np.array([-0.5, -0.5,  0.0, -1.5, -0.5, +1.5, +3.0, +4.5, +6.0, +6.0]),
    # 약한 mood — V3.2 그대로 (균질화 방지, 영화 음향 디자인 원칙)
    "Tenderness":        np.array([-0.5, +1.0, +2.5, +3.0, +2.0, +0.5, -1.5, -2.5, -2.0, -1.0]),
    "Peacefulness":      np.array([+0.5, +1.0, +1.5, +1.5,  0.0, -0.5, -1.5, -2.0, -1.0,  0.0]),
}

# PRESET_VERSIONS에 V3.3 정식 등록 (청취 평가 후 채택)
# analyzer.py의 for-loop가 자동으로 processed_v3_3.mp4도 생성하게 됨
PRESET_VERSIONS["v3_3"] = EQ_PRESETS_V3_3

# 기본 프리셋 — 청취 평가 후 변경 가능
DEFAULT_PRESET_VERSION = "v3_1"
EQ_PRESETS = PRESET_VERSIONS[DEFAULT_PRESET_VERSION]

MOOD_CATEGORIES = list(EQ_PRESETS.keys())


# ────────────────────────────────────────────────────────
# 확률 블렌딩
# ────────────────────────────────────────────────────────
def blend_eq(mood_probabilities: dict, presets: dict | None = None) -> np.ndarray:
    """7개 카테고리 확률 분포로 10밴드 EQ를 가중 평균.

    수학: EQ_final = Σ p_k · EQ_preset_k
    이후 compute_effective_eq()에서 confidence scaling, density modulation,
    temporal smoothing 등의 비선형 보정이 추가됩니다.
    """
    if presets is None:
        presets = EQ_PRESETS

    eq = np.zeros(10)
    total_prob = 0.0
    for mood, prob in mood_probabilities.items():
        if mood in presets and prob > 0:
            eq += prob * presets[mood]
            total_prob += prob

    if total_prob > 0:
        eq = eq / total_prob
    return eq


# ────────────────────────────────────────────────────────
# 대사 보호 (Density-aware Modulation)
# ────────────────────────────────────────────────────────
def apply_dialogue_protection(
    blended_gains: np.ndarray, dialogue_density: float, alpha_d: float = 0.5
) -> np.ndarray:
    """B6/B7/B8 게인을 dialogue density에 비례해 약화.

    공식: g_effective = g_original × (1 - (1 - α_d) × density)
    """
    effective = blended_gains.copy()

    if dialogue_density > 0:
        factor = 1.0 - (1.0 - alpha_d) * dialogue_density
        for i in VOICE_BANDS_IDX:
            effective[i] = blended_gains[i] * factor

    return effective


# ────────────────────────────────────────────────────────
# 통합 파이프라인
# ────────────────────────────────────────────────────────
def compute_effective_eq(
    mood_probabilities: dict,
    dialogue_density: float,
    alpha_d: float = 0.5,
    intensity: float = 1.0,
    confidence_scaling: bool = True,
    confidence_strength: float = 0.4,
    presets: dict | None = None,
) -> np.ndarray:
    """블렌딩 → confidence scaling → 대사 보호 → intensity.

    Confidence-based Scaling:
        지배 감정 확률이 낮으면(애매한 씬) EQ 효과 약화,
        확률이 높으면(확실한 씬) 효과 강화.
    """
    blended = blend_eq(mood_probabilities, presets=presets)

    if confidence_scaling and mood_probabilities:
        max_prob = max(mood_probabilities.values())
        confidence_factor = 1.0 + (max_prob - 0.5) * confidence_strength
        blended = blended * confidence_factor

    protected = apply_dialogue_protection(blended, dialogue_density, alpha_d)
    return protected * intensity


def compute_effective_eq_both_versions(
    mood_probabilities: dict,
    dialogue_density: float,
    alpha_d: float = 0.5,
    intensity: float = 1.0,
) -> dict[str, np.ndarray]:
    """PRESET_VERSIONS에 등록된 모든 버전에 대해 effective EQ를 동시 계산.

    함수명은 '_both_versions' 레거시 (V3.1/V3.2 2개 시절). 지금은 PRESET_VERSIONS
    등록 키 전부 동적 반환. analyzer.py가 이 함수의 반환 dict을 그대로 alt_versions로 쓰므로
    새 프리셋 등록 시 추가 수정 없이 자동 반영됨.
    """
    return {
        version: compute_effective_eq(
            mood_probabilities, dialogue_density,
            alpha_d, intensity, presets=presets,
        )
        for version, presets in PRESET_VERSIONS.items()
    }


def manual_label_to_probs(dominant_mood: str, dominant_prob: float) -> dict:
    """지배 감정과 확률 → 7개 카테고리 확률 분포.

    Day 6~7의 수동 감정 라벨링에서 사용.
    """
    remaining = 1.0 - dominant_prob
    probs = {}
    for cat in MOOD_CATEGORIES:
        probs[cat] = dominant_prob if cat == dominant_mood else remaining / 6.0
    return probs


# ────────────────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────────────────
def verify_eq_presets() -> None:
    """EQ 프리셋 무결성 검증 — V3.1과 V3.2 모두."""
    expected_categories = set(MOOD_CATEGORIES)

    for version_name, presets in PRESET_VERSIONS.items():
        assert len(presets) == 7, f"{version_name}: {len(presets)}개 ≠ 7"
        assert set(presets.keys()) == expected_categories

        for name, gains in presets.items():
            assert len(gains) == 10, f"{version_name}/{name}: {len(gains)}밴드 ≠ 10"

        max_allowed = {"v3_1": 3.0, "v3_2": 5.0, "v3_3": 6.0}.get(version_name, 6.0)
        for name, gains in presets.items():
            max_gain = np.abs(gains).max()
            assert max_gain <= max_allowed, \
                f"{version_name}/{name}: {max_gain:.1f}dB > ±{max_allowed}dB"

        all_gains = np.concatenate([g for g in presets.values()])
        print(
            f"  ✓ {version_name}: 7개 × 10밴드, max={np.abs(all_gains).max():.1f}dB, "
            f"mean_abs={np.abs(all_gains).mean():.2f}dB"
        )

    assert len(BAND_FREQS) == 10
    assert len(BAND_Q) == 10

    for cat in expected_categories:
        diff = np.abs(EQ_PRESETS_V3_1[cat] - EQ_PRESETS_V3_2[cat]).max()
        assert diff > 0.1, f"{cat}: V3.1과 V3.2가 거의 동일"

    print(f"  ✓ V3.1 ↔ V3.2 충분한 차이 확인")
    print(f"  ✓ 현재 기본 버전: {DEFAULT_PRESET_VERSION}")


def verify_blend_eq() -> None:
    """확률 블렌딩 검증."""
    # 단일 카테고리
    result = blend_eq({"Tension": 1.0})
    assert np.allclose(result, EQ_PRESETS["Tension"])

    # 블렌딩 수동 계산
    probs = {"Tension": 0.6, "Power": 0.4}
    result = blend_eq(probs)
    expected = 0.6 * EQ_PRESETS["Tension"] + 0.4 * EQ_PRESETS["Power"]
    assert np.allclose(result, expected)

    # 정규화
    result_06 = blend_eq({"Tension": 0.6})
    result_10 = blend_eq({"Tension": 1.0})
    assert np.allclose(result_06, result_10)

    # 알 수 없는 카테고리 무시
    result = blend_eq({"Tension": 0.5, "Unknown": 0.5})
    assert np.allclose(result, EQ_PRESETS["Tension"])

    # 모든 프리셋 섞기
    uniform = {k: 1 / 7 for k in EQ_PRESETS.keys()}
    result = blend_eq(uniform)
    assert result.shape == (10,) and not np.isnan(result).any()

    # V3.1 vs V3.2 동시 블렌딩
    example = {"Tension": 0.72, "Power": 0.15, "Sadness": 0.08, "Wonder": 0.05}
    blended_v31 = blend_eq(example, presets=EQ_PRESETS_V3_1)
    blended_v32 = blend_eq(example, presets=EQ_PRESETS_V3_2)
    range_v31 = np.abs(blended_v31).max()
    range_v32 = np.abs(blended_v32).max()
    assert range_v32 > range_v31, "V3.2가 V3.1보다 약함 — 설계 오류"

    print(f"  ✓ 6가지 블렌딩 시나리오 통과")
    print(f"  ✓ V3.1 최대 게인 {range_v31:.2f}dB, V3.2 최대 게인 {range_v32:.2f}dB")


def verify_dialogue_protection() -> None:
    """대사 보호 + 파이프라인 통합 검증."""
    tension = EQ_PRESETS["Tension"]

    # density=0 → 변화 없음
    result = apply_dialogue_protection(tension, dialogue_density=0.0)
    assert np.allclose(result, tension)

    # 설계서 예시: density=0.22, α_d=0.5
    result = apply_dialogue_protection(tension.copy(), 0.22, alpha_d=0.5)
    assert abs(result[5] - 0.89) < 0.01
    assert abs(result[6] - 2.225) < 0.01
    assert abs(result[7] - 1.78) < 0.01
    for i in [0, 1, 2, 3, 4, 8, 9]:
        assert result[i] == tension[i]

    # density=1, α_d=0.5 → 절반
    result = apply_dialogue_protection(tension.copy(), 1.0, alpha_d=0.5)
    for i in VOICE_BANDS_IDX:
        assert abs(result[i] - tension[i] * 0.5) < 0.01

    # α_d=1.0 → 변화 없음
    result = apply_dialogue_protection(tension.copy(), 0.5, alpha_d=1.0)
    assert np.allclose(result, tension)

    # 통합 파이프라인 — V3.1 단순 동작
    final = compute_effective_eq(
        {"Tension": 1.0}, dialogue_density=0.22,
        alpha_d=0.5, intensity=1.0, confidence_scaling=False,
    )
    assert abs(final[6] - 2.225) < 0.01

    # intensity
    final_half = compute_effective_eq(
        {"Tension": 1.0}, 0.0, 1.0, intensity=0.5, confidence_scaling=False
    )
    assert np.allclose(final_half, tension * 0.5)

    # Confidence scaling (기본 strength 0.4)
    final_confident = compute_effective_eq(
        {"Tension": 1.0}, 0.0, 1.0, intensity=1.0, confidence_scaling=True
    )
    expected_factor = 1.0 + (1.0 - 0.5) * 0.4  # 1.2
    assert np.allclose(final_confident, tension * expected_factor)

    uniform_probs = {k: 1 / 7 for k in EQ_PRESETS.keys()}
    final_uncertain = compute_effective_eq(
        uniform_probs, 0.0, 1.0, intensity=1.0, confidence_scaling=True
    )
    expected_factor_low = 1.0 + (1 / 7 - 0.5) * 0.4
    uniform_blended = blend_eq(uniform_probs)
    assert np.allclose(final_uncertain, uniform_blended * expected_factor_low)

    # confidence_strength 0.8
    final_strong = compute_effective_eq(
        {"Tension": 1.0}, 0.0, 1.0,
        intensity=1.0, confidence_scaling=True, confidence_strength=0.8,
    )
    assert np.allclose(final_strong, tension * 1.4)

    print(f"  ✓ 9가지 대사 보호 시나리오 통과")
    print(f"  ✓ Tension + density 0.22: B6 1.0→0.89, B7 2.5→2.23, B8 2.0→1.78")
    print(f"  ✓ Confidence scaling: 확실(1.2x) / 불확실(0.86x)")


if __name__ == "__main__":
    print("=" * 60)
    print("EQ 엔진 검증")
    print("=" * 60)
    print("\n[프리셋 무결성]")
    verify_eq_presets()
    print("\n[확률 블렌딩]")
    verify_blend_eq()
    print("\n[대사 보호 + 통합 파이프라인]")
    verify_dialogue_protection()
    print("\n🎉 EQ 엔진 모든 검증 통과")
