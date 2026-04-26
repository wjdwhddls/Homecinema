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
    from core.roomplan_to_numpy import extract_floor_polygon
    from shapely.geometry import Point, Polygon, LineString

    floor_corners = extract_floor_polygon(walls)
    poly = Polygon(floor_corners.tolist())

    listener_xy = np.array([listener_pos[0], listener_pos[1]])
    
    # 사용자 정면 방향 = xRIR 좌표계 +y
    # (RoomPlan 시작 시 카메라가 본 방향이 -z, xRIR로 변환하면 +y)
    direction = np.array([0.0, 1.0])  # +y 방향이 정면
    
    # ray casting으로 정면 벽 찾기
    far_point = listener_xy + direction * 50.0
    
    ray = LineString([listener_xy.tolist(), far_point.tolist()])
    boundary = poly.boundary
    intersection = ray.intersection(boundary)
    
    if intersection.is_empty:
        # 정면에 벽 없음 → 청취자가 방 밖 또는 폴리곤 이상
        # 방 중심 사용 fallback
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
    print(f"정면 방향: {direction}")
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
