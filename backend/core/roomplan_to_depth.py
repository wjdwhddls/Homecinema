"""
roomplan JSON → depth.npy 변환

청취자 위치에서 360도 방향으로 ray casting해서
각 방향의 벽까지 거리를 계산 → (256, 512) 파노라마 depth 이미지 생성
"""

import numpy as np
from pathlib import Path


# ── 좌표 변환 ────────────────────────────────────────────────────

def roomplan_to_xrir_coords(x, y, z):
    """RoomPlan(x, y, z) → xRIR(x, -z, y)"""
    return np.array([x, -z, y], dtype=np.float32)


# ── 벽 삼각형 추출 ───────────────────────────────────────────────

def extract_wall_triangles(walls):
    """
    각 벽의 transform + dimensions으로 사각형 면 추출
    → 삼각형 2개로 분할해서 반환
    반환: list of (v0, v1, v2) 각각 (3,) numpy array (xRIR 좌표계)
    """
    triangles = []

    for wall in walls:
        m = np.array(wall["transform"], dtype=float).reshape(4, 4, order="F")
        dims = wall["dimensions"]
        half_w = float(dims[0]) * 0.5
        half_h = float(dims[1]) * 0.5

        # 벽 4개 꼭짓점 (로컬 좌표)
        local_corners = np.array([
            [-half_w, -half_h, 0.0, 1.0],
            [+half_w, -half_h, 0.0, 1.0],
            [+half_w, +half_h, 0.0, 1.0],
            [-half_w, +half_h, 0.0, 1.0],
        ])

        # 월드 좌표로 변환
        world_corners = (m @ local_corners.T).T[:, :3]

        # xRIR 좌표계로 변환
        xrir_corners = np.array([
            roomplan_to_xrir_coords(*wc) for wc in world_corners
        ])

        # 사각형 → 삼각형 2개
        triangles.append((xrir_corners[0], xrir_corners[1], xrir_corners[2]))
        triangles.append((xrir_corners[0], xrir_corners[2], xrir_corners[3]))

    return triangles


def extract_floor_ceiling_triangles(walls, floor_corners, height):
    """바닥/천장 삼각형 추출 (fan triangulation)"""
    triangles = []

    n = len(floor_corners)
    if n < 3:
        return triangles

    # 바닥 (z=0)
    floor_pts = np.array([[c[0], c[1], 0.0] for c in floor_corners])
    # 천장 (z=height)
    ceil_pts  = np.array([[c[0], c[1], height] for c in floor_corners])

    for i in range(1, n - 1):
        triangles.append((floor_pts[0], floor_pts[i], floor_pts[i+1]))
        triangles.append((ceil_pts[0],  ceil_pts[i],  ceil_pts[i+1]))

    return triangles


# ── Möller–Trumbore ray-triangle intersection ────────────────────

def ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2, eps=1e-7):
    """
    ray와 삼각형의 교점 거리 반환
    교점 없으면 None 반환
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if abs(a) < eps:
        return None  # 평행

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    if v < 0.0 or (u + v) > 1.0:
        return None

    t = f * np.dot(edge2, q)
    if t > eps:
        return t
    return None


# ── Ray casting → depth map ──────────────────────────────────────

def render_depth_map(triangles, listener_pos, img_h=256, img_w=512, max_dist=20.0):
    """
    청취자 위치에서 equirectangular 방향으로 ray casting
    반환: (256, 512) depth map (float32, 미터 단위)
    """
    depth_map = np.full((img_h, img_w), max_dist, dtype=np.float32)

    # 각도 격자 생성
    phi_vals   = (np.arange(img_h) + 0.5) * np.pi / img_h - np.pi / 2       # 위도 (-π/2 ~ π/2)
    theta_vals = (np.arange(img_w) + 0.5) * 2.0 * np.pi / img_w - np.pi    # 경도 (-π ~ π)

    origin = np.array(listener_pos, dtype=np.float64)

    total = img_h * img_w
    print(f"Ray casting 시작... ({img_h}x{img_w} = {total:,}개 픽셀)")

    for pi, phi in enumerate(phi_vals):
        if (pi + 1) % 32 == 0:
            print(f"  {pi+1}/{img_h} 행 완료...")

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        for ti, theta in enumerate(theta_vals):
            # 구면 → 직교 방향 벡터
            ray_dir = np.array([
                cos_phi * np.cos(theta),
                cos_phi * np.sin(theta),
                -sin_phi
            ], dtype=np.float64)

            # 모든 삼각형과 교점 검사
            min_dist = max_dist
            for (v0, v1, v2) in triangles:
                t = ray_triangle_intersect(origin, ray_dir,
                                           v0.astype(np.float64),
                                           v1.astype(np.float64),
                                           v2.astype(np.float64))
                if t is not None and t < min_dist:
                    min_dist = t

            depth_map[pi, ti] = min_dist

    return depth_map


# ── 메인 변환 함수 ───────────────────────────────────────────────

def convert_roomplan_to_depth(
    roomplan_json,
    listener_pos,
    output_dir,
    img_h=256,
    img_w=512,
    max_dist=20.0,
):
    """
    roomplan JSON + 청취자 위치 → depth.npy 저장

    Args:
        roomplan_json : dict (POST /api/optimize/speakers의 roomplan_scan)
        listener_pos  : [x, y, z] numpy array (listener.npy)
        output_dir    : 저장할 폴더 경로
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    walls = roomplan_json.get("walls", [])

    # 벽 삼각형 추출
    triangles = extract_wall_triangles(walls)
    print(f"벽 삼각형: {len(triangles)}개")

    # 바닥/천장 추가
    from core.roomplan_to_numpy import extract_floor_polygon, compute_room_height
    floor_corners = extract_floor_polygon(walls)
    height = compute_room_height(walls)
    triangles += extract_floor_ceiling_triangles(walls, floor_corners, height)
    print(f"전체 삼각형 (바닥/천장 포함): {len(triangles)}개")

    # Ray casting
    depth_map = render_depth_map(triangles, listener_pos, img_h, img_w, max_dist)

    # 저장
    np.save(output_dir / "depth.npy", depth_map)
    print(f"depth.npy 저장 완료: shape={depth_map.shape}, min={depth_map.min():.2f}m, max={depth_map.max():.2f}m")

    return depth_map


# ── 테스트 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append("/home/piai/Homecinema/backend/core")

    sample_roomplan = {
        "walls": [
            {"id": "w1", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2,0,0,1], "dimensions": [4.0, 2.7, 0.2]},
            {"id": "w2", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, -2,0,0,1], "dimensions": [4.0, 2.7, 0.2]},
            {"id": "w3", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,3,1], "dimensions": [6.0, 2.7, 0.2]},
            {"id": "w4", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,-3,1], "dimensions": [6.0, 2.7, 0.2]},
        ],
        "objects": []
    }

    listener_pos = np.array([0.0, 0.0, 1.2], dtype=np.float32)

    depth_map = convert_roomplan_to_depth(
        roomplan_json=sample_roomplan,
        listener_pos=listener_pos,
        output_dir="./test_output",
    )

    print(f"\n완료! depth.npy shape: {depth_map.shape}")
