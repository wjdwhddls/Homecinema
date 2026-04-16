"""
Phase 2~3 end-to-end 테스트.

검증 항목:
- DB 로드 및 기본 조회
- RoomPlan JSON 변환
- 단일 시뮬레이션 실행
- 최적화 파이프라인 (3분 이내 완료)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

FIXTURE = Path(__file__).parent / "fixtures" / "test_scan_office.json"


@pytest.fixture(scope="module")
def scan_json():
    with open(FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def db():
    from core.absorption_db import HybridAbsorptionDatabase
    return HybridAbsorptionDatabase()


@pytest.fixture(scope="module")
def converted_room(scan_json, db):
    from core.room_converter import convert_roomplan_to_pra_input
    return convert_roomplan_to_pra_input(scan_json, db)


def test_absorption_db_load(db):
    wall = db.get_entry("wall")
    assert wall.pra_available is True
    assert wall.pra_keyword == "hard_surface"
    assert len(wall.absorption_7band) == 7


def test_absorption_db_material_factory(db):
    import pyroomacoustics as pra
    m = db.make_pra_material("sofa")
    assert isinstance(m, pra.Material)


def test_absorption_db_unknown_fallback(db):
    entry = db.get_entry("nonexistent_category")
    assert entry.category in ("unknown", "wall")


def test_roomplan_conversion(converted_room, scan_json):
    assert converted_room.floor_corners_2d.shape[0] >= 3
    assert 2.0 < converted_room.height < 4.0
    assert len(converted_room.objects) == len(scan_json["objects"])
    assert converted_room.floor_area_m2 > 5.0
    assert all(0.01 <= a <= 0.95 for a in converted_room.wall_avg_absorption)


def test_listener_inside_polygon(converted_room):
    poly = Polygon(converted_room.floor_corners_2d.tolist())
    pt = Point(
        converted_room.listener_position[0],
        converted_room.listener_position[1],
    )
    assert poly.contains(pt)


def test_simulation_single_speaker(converted_room):
    from core.acoustic_simulator import AcousticSimulator
    sim = AcousticSimulator(converted_room)

    poly = Polygon(converted_room.floor_corners_2d.tolist())
    centroid = poly.centroid
    speaker = np.array([centroid.x + 0.5, centroid.y + 0.3, 1.2])
    assert poly.contains(Point(speaker[0], speaker[1]))

    rir = sim.simulate(speaker, max_order=3, ray_tracing=True)
    assert rir is not None
    assert len(rir) > 100
    metrics = sim.compute_metrics(rir)
    assert 0.05 < metrics["rt60_seconds"] < 5.0


def test_optimization_end_to_end(converted_room):
    from core.acoustic_simulator import AcousticSimulator
    from core.speaker_optimizer import SpeakerOptimizer

    sim = AcousticSimulator(converted_room)
    opt = SpeakerOptimizer(
        sim,
        config_type="stereo",
        time_budget_seconds=180.0,
        speaker_dimensions={"width_m": 0.20, "height_m": 0.35, "depth_m": 0.28},
    )

    t0 = time.time()
    result = opt.find_optimal()
    elapsed = time.time() - t0

    assert elapsed < 180.0, f"최적화가 3분을 초과함: {elapsed:.1f}s"
    assert result["best"].score < float("inf")
    assert result["best"].metrics is not None

    rt60 = result["best"].metrics["rt60_seconds"]
    assert 0.3 < rt60 < 2.5, f"RT60이 합리 범위 밖: {rt60:.2f}s"

    poly = Polygon(converted_room.floor_corners_2d.tolist())
    left = result["best"].candidate.left
    right = result["best"].candidate.right
    assert poly.contains(Point(left[0], left[1]))
    assert poly.contains(Point(right[0], right[1]))

    listener = converted_room.listener_position
    dist_l = float(np.linalg.norm(left - listener))
    dist_r = float(np.linalg.norm(right - listener))
    assert abs(dist_l - dist_r) < 0.20, (
        f"스테레오 대칭성 깨짐: |dL - dR| = {abs(dist_l - dist_r):.3f}"
    )


def test_optimize_request_schema():
    from models.optimization_schemas import OptimizeRequest
    req = OptimizeRequest(
        roomplan_scan={"walls": [], "objects": [], "doors": [],
                       "windows": [], "openings": [], "scannedAt": ""},
        speaker_dimensions={"width_m": 0.20, "height_m": 0.35, "depth_m": 0.28},
        listener_height_m=1.2,
        config_type="stereo",
    )
    assert req.config_type == "stereo"
    assert req.speaker_dimensions.width_m == 0.20


def test_optimize_request_requires_speaker_dimensions():
    """speaker_dimensions는 필수 필드여야 한다."""
    from pydantic import ValidationError
    from models.optimization_schemas import OptimizeRequest
    with pytest.raises(ValidationError):
        OptimizeRequest(  # type: ignore[call-arg]
            roomplan_scan={"walls": [], "objects": [], "doors": [],
                           "windows": [], "openings": [], "scannedAt": ""},
        )


def test_valid_position_scales_with_speaker_size(converted_room):
    """큰 스피커는 작은 스피커 대비 유효 위치 수가 감소해야 한다."""
    from core.acoustic_simulator import AcousticSimulator
    from core.speaker_optimizer import SpeakerOptimizer

    sim = AcousticSimulator(converted_room)

    small = SpeakerOptimizer(
        sim,
        config_type="stereo",
        speaker_dimensions={"width_m": 0.15, "height_m": 0.25, "depth_m": 0.20},
    )
    large = SpeakerOptimizer(
        sim,
        config_type="stereo",
        speaker_dimensions={"width_m": 0.60, "height_m": 1.20, "depth_m": 0.50},
    )

    small_candidates = small.generate_candidates(n_candidates=60)
    large_candidates = large.generate_candidates(n_candidates=60)

    # 큰 스피커는 더 엄격한 클리어런스로 후보가 같거나 적어야 한다.
    assert len(large_candidates) <= len(small_candidates)
    # 벽 마진은 half_max + 5cm가 기본 0.30을 넘는 순간 커져야 한다.
    assert large._wall_margin > small._wall_margin
    # 코너 마진은 기본 0.80 하한 유지. 최소 하한만 깨지지 않으면 OK.
    assert large._corner_margin >= 0.80
    # 청취 최소 거리도 대형 스피커에서 증가해야 한다.
    assert large._min_listen_distance >= small._min_listen_distance


def test_small_speaker_uses_baseline_margins(converted_room):
    """스피커가 작으면 기본 마진(0.30 / 0.80)이 하한으로 유지돼야 한다."""
    from core.acoustic_simulator import AcousticSimulator
    from core.speaker_optimizer import SpeakerOptimizer

    sim = AcousticSimulator(converted_room)
    opt = SpeakerOptimizer(
        sim,
        config_type="stereo",
        speaker_dimensions={"width_m": 0.15, "height_m": 0.25, "depth_m": 0.20},
    )
    assert opt._wall_margin == pytest.approx(0.30)
    assert opt._corner_margin == pytest.approx(0.80)
