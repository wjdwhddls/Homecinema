"""분포 분석 스크립트 테스트 (spec [1-5], [3-1], [3-3])."""

import json
from pathlib import Path

import pytest
import torch

from ..analyze_va import (
    MOOD_LABELS,
    _build_parser,
    _report_for_json,
    analyze_distribution,
    compute_mood_distribution,
    compute_quadrant_film_counts,
    compute_va_pairwise_distances,
    main,
    plot_mood_counts,
    plot_quadrant_films,
    plot_va_distance_histogram,
    plot_va_scatter,
    recommend_class_weights,
    recommend_negative_sampler_thresholds,
)
from ..precompute import save_features


# ---------- core analysis ----------


def test_mood_distribution_counts_sum():
    valences = [0.5, -0.5, 0.0, 0.7]
    arousals = [0.5, -0.5, 0.8, 0.6]
    dist = compute_mood_distribution(valences, arousals)
    assert sum(dist["counts"]) == dist["total"] == 4
    assert len(dist["labels"]) == 7
    assert abs(sum(dist["per_label_ratio"]) - 1.0) < 1e-6


def test_mood_distribution_imbalance_detection():
    # 10 samples all landing near the same mood center (Tension: -0.6, +0.7)
    valences = [-0.6] * 10 + [0.5]
    arousals = [0.7] * 10 + [-0.5]
    dist = compute_mood_distribution(valences, arousals)
    assert dist["imbalance_ratio"] >= 10.0


def test_va_pairwise_distances_known_values():
    # 3 points forming right triangle: (0,0),(1,0),(0,1) → d=1,1,sqrt(2)
    out = compute_va_pairwise_distances([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    dists = sorted(out["distances"].tolist())
    assert dists[0] == pytest.approx(1.0)
    assert dists[1] == pytest.approx(1.0)
    assert dists[2] == pytest.approx(2.0 ** 0.5)
    assert out["num_pairs"] == 3


def test_va_pairwise_percentiles_monotonic():
    torch.manual_seed(0)
    v = torch.randn(200).tolist()
    a = torch.randn(200).tolist()
    out = compute_va_pairwise_distances(v, a)
    p = out["percentiles"]
    assert p["p25"] <= p["p50"] <= p["p75"] <= p["p90"]


def test_va_pairwise_subsampling_respects_max_pairs():
    # 100 points → 4950 pairs; cap at 500 → must sample
    v = [0.1 * i for i in range(100)]
    a = [0.1 * (i % 5) for i in range(100)]
    out = compute_va_pairwise_distances(v, a, max_pairs=500)
    assert out["num_pairs"] <= 500
    assert out["num_pairs"] > 0


def test_quadrant_film_counts_sum():
    movie_va = {0: (0.5, 0.5), 1: (0.5, -0.5), 2: (-0.5, 0.5), 3: (-0.5, -0.5), 4: (0.1, 0.1)}
    q = compute_quadrant_film_counts(movie_va)
    assert sum(q.values()) == 5
    assert q["HVHA"] == 2  # (0.5,0.5) + (0.1,0.1)
    assert q["HVLA"] == 1
    assert q["LVHA"] == 1
    assert q["LVLA"] == 1


def test_recommend_class_weights_balanced():
    assert recommend_class_weights([100, 110, 95, 105, 100, 102, 98]) is None


def test_recommend_class_weights_imbalanced():
    counts = [500, 50, 50, 50, 50, 50, 50]  # 500/50 = 10x
    w = recommend_class_weights(counts)
    assert w is not None
    assert len(w) == 7
    # rare classes get larger weights
    assert w[1] > w[0]
    # mean approximately 1
    assert abs(sum(w) / len(w) - 1.0) < 1e-5


def test_recommend_neg_sampler_thresholds():
    out = recommend_negative_sampler_thresholds(
        {"p25": 0.3, "p50": 0.7, "p75": 1.1, "p90": 1.4}
    )
    assert out == {"slight_lower": 0.3, "slight_upper": 0.7, "strong_lower": 1.1}


def test_analyze_distribution_full_dict():
    valences = [0.2 * i - 1.0 for i in range(20)]
    arousals = [0.1 * i - 1.0 for i in range(20)]
    movie_ids = [i // 4 for i in range(20)]
    rep = analyze_distribution(movie_ids, valences, arousals)
    assert set(rep.keys()) >= {
        "mood_distribution", "va_distances", "quadrant_films",
        "num_films", "num_samples", "recommendations",
    }
    assert rep["num_samples"] == 20
    assert rep["num_films"] == 5
    assert set(rep["recommendations"]) == {"class_weights", "negative_sampler_thresholds"}


# ---------- plotting ----------


def test_plot_va_scatter_saves_png(tmp_path):
    out = tmp_path / "scatter.png"
    plot_va_scatter([0.1, -0.3, 0.5], [0.2, 0.6, -0.4], out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_mood_counts_saves_png(tmp_path):
    out = tmp_path / "counts.png"
    plot_mood_counts([10, 20, 15, 8, 5, 12, 7], out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_va_distance_histogram_saves_png(tmp_path):
    out = tmp_path / "hist.png"
    distances = torch.rand(300)
    percentiles = {"p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9}
    plot_va_distance_histogram(distances, percentiles, out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_quadrant_films_saves_png(tmp_path):
    out = tmp_path / "quad.png"
    plot_quadrant_films({"HVHA": 3, "HVLA": 2, "LVHA": 1, "LVLA": 4}, out)
    assert out.is_file() and out.stat().st_size > 0


# ---------- JSON sanitization ----------


def test_report_for_json_strips_raw_distances():
    rep = {
        "num_samples": 5, "num_films": 2,
        "mood_distribution": {"counts": [1, 0, 0, 0, 0, 0, 0]},
        "va_distances": {
            "distances": torch.randn(1000),
            "percentiles": {"p25": 0.1, "p50": 0.2, "p75": 0.3, "p90": 0.4},
            "num_pairs": 1000,
        },
        "quadrant_films": {"HVHA": 1, "HVLA": 1, "LVHA": 0, "LVLA": 0},
        "recommendations": {"class_weights": None, "negative_sampler_thresholds": {}},
    }
    sanitized = _report_for_json(rep)
    # Round-trip through json must not error
    encoded = json.dumps(sanitized)
    # raw tensor was stripped; only percentiles+num_pairs remain under va_distances
    assert set(sanitized["va_distances"].keys()) == {"percentiles", "num_pairs"}
    assert "percentiles" in encoded


# ---------- CLI smoke ----------


def test_cli_synthetic_smoke(tmp_path):
    report = main([
        "--source", "synthetic",
        "--synthetic_num_clips", "60",
        "--synthetic_num_films", "6",
        "--seed", "0",
        "--output_dir", str(tmp_path),
    ])
    assert (tmp_path / "distribution_report.json").is_file()
    for png in ("va_scatter.png", "mood_counts.png", "va_distance_hist.png", "quadrant_films.png"):
        assert (tmp_path / png).is_file()
    # JSON must parse and contain expected top-level keys
    with (tmp_path / "distribution_report.json").open() as f:
        data = json.load(f)
    assert set(data.keys()) >= {
        "num_samples", "num_films", "mood_distribution",
        "va_distances", "quadrant_films", "recommendations",
    }


def test_cli_feature_dir_smoke(tmp_path):
    # Build a dummy precompute output first
    visual, audio, metadata = {}, {}, {}
    for i in range(20):
        wid = f"clip{i}_w0"
        visual[wid] = torch.randn(512)
        audio[wid] = torch.randn(2048)
        metadata[wid] = {
            "clip_id": f"clip{i}",
            "movie_id": i % 4,
            "valence": 0.5 if i % 2 == 0 else -0.5,
            "arousal": 0.3 if i < 10 else -0.3,
            "start": 0.0, "end": 4.0,
        }
    save_features(visual, audio, metadata, tmp_path, split_name="toy")

    out_dir = tmp_path / "analysis"
    report = main([
        "--feature_dir", str(tmp_path),
        "--split_name", "toy",
        "--output_dir", str(out_dir),
    ])
    assert (out_dir / "distribution_report.json").is_file()
    assert report["num_samples"] == 20
    assert report["num_films"] == 4


def test_parser_requires_source():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
