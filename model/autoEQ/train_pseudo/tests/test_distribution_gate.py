from model.autoEQ.train_pseudo.analyze_cognimuse_distribution import (
    GATE_THRESHOLD,
    analyze_distribution,
)


def _meta_with_va(va_pairs: list[tuple[float, float]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, (v, a) in enumerate(va_pairs):
        out[f"W_{i:05d}"] = {
            "movie_id": 0,
            "movie_code": "BMI",
            "valence": v,
            "arousal": a,
            "valence_std": 0.0,
            "arousal_std": 0.0,
            "t0": i * 2.0,
            "t1": i * 2.0 + 4.0,
            "annotation_source": "experienced",
        }
    return out


def test_gate_fails_on_missing_class():
    # All samples in upper-right quadrant → most mood classes will be 0%
    va = [(0.9, 0.9)] * 200
    meta = _meta_with_va(va)
    report = analyze_distribution(meta, num_mood_classes=7)
    assert report["gate_passed"] is False
    assert report["min_mood_class_pct"] < GATE_THRESHOLD


def test_gate_passes_on_balanced():
    # Spread samples across 4 quadrants uniformly
    va = []
    for v_sign in (-0.7, 0.7):
        for a_sign in (-0.7, 0.7):
            va.extend([(v_sign, a_sign)] * 50)
    meta = _meta_with_va(va)
    report = analyze_distribution(meta, num_mood_classes=4)
    # With K=4 and 4 equal quadrants, each class should be ~25%
    assert report["gate_passed"] is True
    for pct in report["mood_class_pct"].values():
        assert pct >= GATE_THRESHOLD


def test_va_quadrant_pct_sums_to_one():
    va = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    meta = _meta_with_va(va)
    report = analyze_distribution(meta, num_mood_classes=7)
    assert abs(sum(report["va_quadrant_pct"].values()) - 1.0) < 1e-6
