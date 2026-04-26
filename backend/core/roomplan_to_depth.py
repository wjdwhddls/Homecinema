"""
roomplan JSON → depth.npy 변환

mesh.bin이 있으면 ARMeshAnchor 실제 LiDAR 메쉬로 ray casting (정확)
mesh.bin이 없으면 roomplan JSON 벽 정보로 ray casting (fallback)

청취자 위치에서 360도 방향으로 ray casting해서
각 방향의 벽까지 거리를 계산 → (256, 512) 파노라마 depth 이미지 생성
"""

import numpy as np
import struct
from pathlib import Path


# ── 좌표 변환 ────────────────────────────────────────────────────

def roomplan_to_xrir_coords(x, y, z):
    """RoomPlan(x, y, z) → xRIR(x, -z, y)"""
    return np.array([x, -z, y], dtype=np.float32)


# ── mesh.bin 파싱 ────────────────────────────────────────────────

def load_mesh_bin(mesh_bin_path):
    """
    mesh.bin 파싱 → 삼각형 리스트 반환

    mesh.bin 구조:
      [vertex_count: int32]
      [face_count: int32]
      [vertices: float32 x,y,z * vertex_count]  ← world space, RoomPlan 좌표계
      [faces: int32 i0,i1,i2 * face_count]

    반환: list of (v0, v1, v2) 각각 (3,) numpy array (xRIR 좌표계)
    """
    data = Path(mesh_bin_path).read_bytes()
    offset = 0

    vertex_count = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    face_count = struct.unpack_from("<i", data, offset)[0]
    offset += 4

    # 버텍스 파싱
    vertices_flat = struct.unpack_from(f"<{vertex_count * 3}f", data, offset)
    offset += vertex_count * 3 * 4
    vertices = np.array(vertices_flat, dtype=np.float32).reshape(-1, 3)

    # 페이스 파싱
    faces_flat = struct.unpack_from(f"<{face_count * 3}i", data, offset)
    faces = np.array(faces_flat, dtype=np.int32).reshape(-1, 3)

    # RoomPlan → xRIR 좌표 변환 후 삼각형 리스트 생성
    triangles = []
    for face in faces:
        v0 = roomplan_to_xrir_coords(*vertices[face[0]])
        v1 = roomplan_to_xrir_coords(*vertices[face[1]])
        v2 = roomplan_to_xrir_coords(*vertices[face[2]])
        triangles.append((v0, v1, v2))

    print(f"mesh.bin 로드 완료: 버텍스 {vertex_count}개, 페이스 {face_count}개 → 삼각형 {len(triangles)}개")
    return triangles


# ── roomplan JSON 기반 삼각형 추출 (fallback) ────────────────────

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

        local_corners = np.array([
            [-half_w, -half_h, 0.0, 1.0],
            [+half_w, -half_h, 0.0, 1.0],
            [+half_w, +half_h, 0.0, 1.0],
            [-half_w, +half_h, 0.0, 1.0],
        ])

        world_corners = (m @ local_corners.T).T[:, :3]
        xrir_corners = np.array([
            roomplan_to_xrir_coords(*wc) for wc in world_corners
        ])

        triangles.append((xrir_corners[0], xrir_corners[1], xrir_corners[2]))
        triangles.append((xrir_corners[0], xrir_corners[2], xrir_corners[3]))

    return triangles


def extract_floor_ceiling_triangles(floor_corners, height):
    """바닥/천장 삼각형 추출 (fan triangulation)"""
    triangles = []
    n = len(floor_corners)
    if n < 3:
        return triangles

    floor_pts = np.array([[c[0], c[1], 0.0] for c in floor_corners])
    ceil_pts  = np.array([[c[0], c[1], height] for c in floor_corners])

    for i in range(1, n - 1):
        triangles.append((floor_pts[0], floor_pts[i], floor_pts[i+1]))
        triangles.append((ceil_pts[0],  ceil_pts[i],  ceil_pts[i+1]))

    return triangles


# ── Möller–Trumbore ray-triangle intersection ────────────────────

def ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2, eps=1e-7):
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if abs(a) < eps:
        return None

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

def render_depth_map_fast(triangles, listener_pos, img_h=256, img_w=512, max_dist=20.0):
    """벡터화된 ray casting"""
    # 모든 ray 방향을 한 번에 생성
    phi_vals = (np.arange(img_h) + 0.5) * np.pi / img_h - np.pi / 2
    theta_vals = (np.arange(img_w) + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals, indexing='ij')
    
    cos_phi = np.cos(phi_grid)
    ray_dirs = np.stack([
        cos_phi * np.cos(theta_grid),
        cos_phi * np.sin(theta_grid),
        -np.sin(phi_grid)
    ], axis=-1).reshape(-1, 3)  # (H*W, 3)
    
    origin = np.array(listener_pos, dtype=np.float64)
    
    # 모든 삼각형을 numpy 배열로
    v0s = np.array([t[0] for t in triangles], dtype=np.float64)  # (N, 3)
    v1s = np.array([t[1] for t in triangles], dtype=np.float64)
    v2s = np.array([t[2] for t in triangles], dtype=np.float64)
    
    edge1 = v1s - v0s  # (N, 3)
    edge2 = v2s - v0s
    
    depth_map = np.full(img_h * img_w, max_dist, dtype=np.float32)
    
    print(f"벡터화 ray casting 시작... ({len(ray_dirs):,}개 ray × {len(triangles)}개 삼각형)")
    
    # 각 삼각형에 대해 모든 ray와 한 번에 계산
    for i in range(len(triangles)):
        e1 = edge1[i]  # (3,)
        e2 = edge2[i]
        v0 = v0s[i]
        
        h = np.cross(ray_dirs, e2)  # (H*W, 3)
        a = np.einsum('ij,j->i', h, e1)  # (H*W,)
        
        valid = np.abs(a) > 1e-7
        f = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)
        
        s = origin - v0  # (3,)
        u = f * np.einsum('ij,j->i', h, s)
        
        valid &= (u >= 0.0) & (u <= 1.0)
        
        q = np.cross(s, e1)  # (3,)
        v = f * np.einsum('ij,j->i', ray_dirs, q)
        
        valid &= (v >= 0.0) & ((u + v) <= 1.0)
        
        t = f * np.dot(e2, q)
        valid &= (t > 1e-7)
        
        # 더 가까운 거리로 업데이트
        mask = valid & (t < depth_map)
        depth_map[mask] = t[mask]
    
    return depth_map.reshape(img_h, img_w)


# ── 메인 변환 함수 ───────────────────────────────────────────────

def convert_roomplan_to_depth(
    roomplan_json,
    listener_pos,
    output_dir,
    mesh_bin_path=None,   # ← 추가: mesh.bin 경로 (있으면 정확한 버전 사용)
    img_h=256,
    img_w=512,
    max_dist=20.0,
):
    """
    roomplan JSON + 청취자 위치 → depth.npy 저장

    Args:
        roomplan_json  : dict
        listener_pos   : [x, y, z] numpy array
        output_dir     : 저장할 폴더 경로
        mesh_bin_path  : mesh.bin 경로 (None이면 roomplan JSON fallback)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 삼각형 소스 선택 ──────────────────────────────────────────
    if mesh_bin_path and Path(mesh_bin_path).exists():
        print("mesh.bin 감지 → LiDAR 메쉬 기반 ray casting (정확)")
        triangles = load_mesh_bin(mesh_bin_path)
    else:
        print("mesh.bin 없음 → roomplan JSON 기반 ray casting (fallback)")
        walls = roomplan_json.get("walls", [])
        triangles = extract_wall_triangles(walls)
        print(f"벽 삼각형: {len(triangles)}개")

        from core.roomplan_to_numpy import extract_floor_polygon, compute_room_height
        floor_corners = extract_floor_polygon(walls)
        height = compute_room_height(walls)
        triangles += extract_floor_ceiling_triangles(floor_corners, height)
        print(f"전체 삼각형 (바닥/천장 포함): {len(triangles)}개")

    # ── Ray casting ───────────────────────────────────────────────
    depth_map = render_depth_map_fast(triangles, listener_pos, img_h, img_w, max_dist)

    # ── 저장 ──────────────────────────────────────────────────────
    np.save(output_dir / "depth.npy", depth_map)
    print(f"depth.npy 저장 완료: shape={depth_map.shape}, min={depth_map.min():.2f}m, max={depth_map.max():.2f}m")

    return depth_map
