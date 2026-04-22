"""
임시 스피커 위치 계산

청취자 위치(원점)에서 정면 방향 벽까지의 중간 지점을 임시 스피커 위치로 계산
"""

import numpy as np
from shapely.geometry import Point, Polygon


def compute_initial_speaker_position(
    walls,
    listener_pos,
    speaker_height=1.2,
):
    """
    임시 스피커 위치 계산
    
    방법:
    - 청취자 위치에서 방 중심 방향으로 ray를 쏴서 반대편 벽을 찾음
    - 청취자와 반대편 벽 사이의 중간 지점을 임시 스피커 위치로 사용

    Args:
        walls        : roomplan JSON의 walls 리스트
        listener_pos : 청취자 위치 [x, y, z]
        speaker_height: 스피커 높이 (m)

    Returns:
        initial_pos: 임시 스피커 위치 [x, y, z] numpy array
    """
    from core.roomplan_to_numpy import extract_floor_polygon

    floor_corners = extract_floor_polygon(walls)
    poly = Polygon(floor_corners.tolist())

    listener_xy = np.array([listener_pos[0], listener_pos[1]])
    centroid = np.array([poly.centroid.x, poly.centroid.y])

    # 청취자 → 방 중심 방향
    direction = centroid - listener_xy
    direction_norm = np.linalg.norm(direction)

    if direction_norm < 1e-6:
        # 청취자가 이미 방 중심이면 임의 방향(+x)으로
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / direction_norm

    # ray casting으로 반대편 벽 찾기
    far_point = listener_xy + direction * 50.0  # 충분히 먼 지점

    from shapely.geometry import LineString
    ray = LineString([listener_xy.tolist(), far_point.tolist()])
    boundary = poly.boundary
    intersection = ray.intersection(boundary)

    if intersection.is_empty:
        # 교점 없으면 방 중심 사용
        wall_point = centroid
    elif intersection.geom_type == 'Point':
        wall_point = np.array([intersection.x, intersection.y])
    elif intersection.geom_type == 'MultiPoint':
        # 여러 교점이면 가장 먼 것 선택
        pts = [(geom.x, geom.y) for geom in intersection.geoms]
        dists = [np.linalg.norm(np.array(p) - listener_xy) for p in pts]
        wall_point = np.array(pts[np.argmax(dists)])
    else:
        wall_point = centroid

    # 청취자와 벽 사이 중간 지점
    mid_point = (listener_xy + wall_point) / 2.0

    initial_pos = np.array([
        float(mid_point[0]),
        float(mid_point[1]),
        float(speaker_height),
    ], dtype=np.float32)

    print(f"청취자 위치: {listener_xy}")
    print(f"반대편 벽: {wall_point}")
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
