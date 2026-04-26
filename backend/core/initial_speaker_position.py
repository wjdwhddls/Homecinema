"""
임시 스피커 위치 계산

청취자 위치(원점)에서 정면 방향 벽까지의 중간 지점을 임시 스피커 위치로 계산.

정면 방향 결정 규칙:
  1. 청취자 위치에서 360° 균등 ray cast (5° 간격)으로 가장 멀리 닿는 boundary 방향 = candidate
  2. 사용자 hint 방향(RoomKit 시작 방향 = +y)도 별도로 거리 측정
  3. hint 방향의 거리가 candidate 거리의 70% 이상 이거나
     hint와 candidate의 각도 차가 30° 이내면 hint를 그대로 사용
  4. 그 외엔 candidate 사용 + 경고 로그 (사용자 시작 방향이 거실 정면과 다른 케이스)
"""

import numpy as np
from shapely.geometry import Point, Polygon, LineString


def _ray_distance_to_boundary(listener_xy, direction, boundary):
    """청취자에서 direction 방향으로 ray를 쏜 뒤 boundary와의 가장 가까운 교점까지 거리.
    교점 없으면 None."""
    far_point = listener_xy + direction * 100.0
    ray = LineString([listener_xy.tolist(), far_point.tolist()])
    intersection = ray.intersection(boundary)

    if intersection.is_empty:
        return None

    pts = []
    if intersection.geom_type == 'Point':
        pts = [(intersection.x, intersection.y)]
    elif intersection.geom_type == 'MultiPoint':
        pts = [(g.x, g.y) for g in intersection.geoms]
    elif intersection.geom_type == 'GeometryCollection':
        for g in intersection.geoms:
            if g.geom_type == 'Point':
                pts.append((g.x, g.y))

    if not pts:
        return None

    dists = [np.linalg.norm(np.array(p) - listener_xy) for p in pts]
    return float(min(dists))


def estimate_forward_direction(
    listener_xy: np.ndarray,
    floor_polygon: np.ndarray,
    user_hint_direction: np.ndarray = None,
    n_samples: int = 72,
    hint_tolerance_deg: float = 30.0,
    hint_distance_ratio: float = 0.7,
):
    """
    청취자 위치에서 정면 방향(2D unit vector)과 그 방향의 boundary 거리 반환.

    Args:
        listener_xy: (2,) 청취자 xy 좌표
        floor_polygon: (N, 2) 방 바닥 폴리곤
        user_hint_direction: 사용자가 의도한 방향 (예: RoomKit 시작 방향 +y)
        n_samples: 360° 분할 수
        hint_tolerance_deg: hint와 best 방향 각도 차가 이 값 이내면 hint 사용
        hint_distance_ratio: hint 방향 거리가 best 거리의 이 비율 이상이면 hint 사용

    Returns:
        (forward_2d, distance_m, used_hint)
    """
    poly = Polygon(floor_polygon.tolist())
    boundary = poly.boundary

    # 1) 360° 자동 추정
    angles = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
    best_angle = 0.0
    best_dist  = 0.0
    for theta in angles:
        direction = np.array([np.cos(theta), np.sin(theta)])
        d = _ray_distance_to_boundary(listener_xy, direction, boundary)
        if d is not None and d > best_dist:
            best_dist = d
            best_angle = theta
    best_dir = np.array([np.cos(best_angle), np.sin(best_angle)])

    # 2) hint 방향이 없으면 자동 추정 사용
    if user_hint_direction is None:
        return best_dir, best_dist, False

    hint = np.asarray(user_hint_direction, dtype=float)
    hint_norm = np.linalg.norm(hint)
    if hint_norm < 1e-6:
        return best_dir, best_dist, False
    hint = hint / hint_norm

    hint_dist = _ray_distance_to_boundary(listener_xy, hint, boundary)
    if hint_dist is None:
        # hint 방향에 벽 없음 → 자동 추정 사용
        return best_dir, best_dist, False

    # 3) hint와 best 방향 비교
    cos_angle = float(np.clip(np.dot(hint, best_dir), -1.0, 1.0))
    angle_diff_deg = np.degrees(np.arccos(cos_angle))

    use_hint = (
        angle_diff_deg <= hint_tolerance_deg
        or (best_dist > 0 and hint_dist >= best_dist * hint_distance_ratio)
    )

    if use_hint:
        return hint, hint_dist, True
    return best_dir, best_dist, False


def compute_initial_speaker_position(
    walls,
    listener_pos,
    speaker_height=1.2,
):
    from core.roomplan_to_numpy import extract_floor_polygon

    floor_corners = extract_floor_polygon(walls)
    poly = Polygon(floor_corners.tolist())

    listener_xy = np.array([listener_pos[0], listener_pos[1]])

    # RoomKit 시작 시 카메라가 본 방향(-z) → xRIR 좌표계 +y. 사용자 hint로 사용.
    user_hint = np.array([0.0, 1.0])

    direction, max_dist, used_hint = estimate_forward_direction(
        listener_xy=listener_xy,
        floor_polygon=floor_corners,
        user_hint_direction=user_hint,
    )

    # ray casting으로 정면 벽 찾기 (위에서 거리 계산은 했지만 교점 좌표가 필요)
    far_point = listener_xy + direction * 50.0
    ray = LineString([listener_xy.tolist(), far_point.tolist()])
    boundary = poly.boundary
    intersection = ray.intersection(boundary)

    if intersection.is_empty:
        # 정면에 벽 없음 → 방 중심 fallback
        centroid = np.array([poly.centroid.x, poly.centroid.y])
        wall_point = centroid
    elif intersection.geom_type == 'Point':
        wall_point = np.array([intersection.x, intersection.y])
    elif intersection.geom_type == 'MultiPoint':
        pts = [(geom.x, geom.y) for geom in intersection.geoms]
        dists = [np.linalg.norm(np.array(p) - listener_xy) for p in pts]
        wall_point = np.array(pts[np.argmin(dists)])  # 가장 가까운 벽
    else:
        wall_point = np.array([poly.centroid.x, poly.centroid.y])

    # 청취자와 벽 사이 중간 지점
    mid_point = (listener_xy + wall_point) / 2.0

    initial_pos = np.array([
        float(mid_point[0]),
        float(mid_point[1]),
        float(speaker_height),
    ], dtype=np.float32)

    print(f"청취자 위치: {listener_xy}")
    if used_hint:
        print(f"정면 방향: {direction} (사용자 시작 방향 +y 채택, 거리={max_dist:.2f}m)")
    else:
        print(f"정면 방향: {direction} (자동 추정 — 사용자 +y는 거실 정면과 어긋남, 거리={max_dist:.2f}m)")
    print(f"정면 벽: {wall_point}")
    print(f"임시 스피커 위치: {initial_pos}")

    return initial_pos


# ── 테스트 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append("/home/piai/Homecinema/backend")

    sample_roomplan = {
        "walls": [
            {"id": "w1", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2,0,0,1], "dimensions": [4.0, 2.7, 0.2]},
            {"id": "w2", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, -2,0,0,1], "dimensions": [4.0, 2.7, 0.2]},
            {"id": "w3", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,3,1], "dimensions": [6.0, 2.7, 0.2]},
            {"id": "w4", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,-3,1], "dimensions": [6.0, 2.7, 0.2]},
        ],
        "objects": []
    }

    listener_pos = np.array([0.0, 0.0, 1.2])

    initial_pos = compute_initial_speaker_position(
        walls=sample_roomplan["walls"],
        listener_pos=listener_pos,
        speaker_height=1.2,
    )

    print(f"\n임시 스피커 위치: x={initial_pos[0]:.2f}, y={initial_pos[1]:.2f}, z={initial_pos[2]:.2f}")
