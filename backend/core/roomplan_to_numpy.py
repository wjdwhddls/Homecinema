"""
roomplan JSON → listener.npy, xyzs.npy 변환

listener.npy : 청취자 위치 [x, y, z]
xyzs.npy     : 후보 스피커 위치들 [N, 3]
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon


# ── 좌표 변환 (RoomPlan y-up → xRIR z-up) ──────────────────────

def roomplan_to_xrir_coords(x, y, z):
    """RoomPlan(x, y, z) → xRIR(x, -z, y)"""
    return np.array([x, -z, y], dtype=np.float32)


# ── 벽에서 바닥 폴리곤 추출 ──────────────────────────────────────

def extract_floor_polygon(walls):
    """
    벽 transform 행렬에서 바닥 모서리 추출 → convex hull로 방 형태 계산
    반환: (N, 2) 바닥 꼭짓점 배열 (xRIR 좌표계 x-y 평면)
    """
    pts = []
    for wall in walls:
        m = np.array(wall["transform"], dtype=float).reshape(4, 4, order="F")
        center = m[:3, 3]
        dims = wall["dimensions"]
        half_w = float(dims[0]) * 0.5
        half_h = float(dims[1]) * 0.5

        # 벽 하단 좌우 모서리 (로컬 → 월드)
        local_pts = np.array([
            [-half_w, -half_h, 0.0, 1.0],
            [+half_w, -half_h, 0.0, 1.0],
        ])
        world_pts = (m @ local_pts.T).T[:, :3]

        for wp in world_pts:
            xrir = roomplan_to_xrir_coords(*wp)
            pts.append(xrir[:2])  # x-y 평면만

    pts = np.array(pts)
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def compute_room_height(walls):
    heights = [float(w["dimensions"][1]) for w in walls if len(w.get("dimensions", [])) >= 2]
    return float(np.mean(heights)) if heights else 2.7


# ── listener.npy 생성 ────────────────────────────────────────────

def compute_listener_position(walls, listener_height=1.2):
    """
    스캔 원점(0,0)을 청취자 위치로 사용
    원점이 방 밖에 있으면 centroid로 대체
    반환: [x, y, z] numpy array
    """
    
    floor_corners = extract_floor_polygon(walls)
    poly = Polygon(floor_corners.tolist())
    origin = Point(0.0, 0.0)
    
    if poly.contains(origin):
        listener_xy = np.array([0.0, 0.0])
    else:
        centroid = poly.centroid
        listener_xy = np.array([centroid.x, centroid.y])
    
    return np.array([listener_xy[0], listener_xy[1], listener_height], dtype=np.float32)

# ── 후보 위치 생성 ────────────────────────────────────────────────
def generate_candidate_positions(walls, speaker_height=1.2,
                                  grid_step=0.3, wall_margin=0.5):
    """
    방 바닥 폴리곤 안에 격자 형태로 후보 스피커 위치 생성
    (현재는 안 쓰지만 convert_roomplan_to_xrir_inputs가 호출하므로 유지)
    """
    floor_corners = extract_floor_polygon(walls)
    poly = Polygon(floor_corners.tolist()).buffer(-wall_margin)

    x_min, y_min, x_max, y_max = poly.bounds
    xs = np.arange(x_min, x_max, grid_step)
    ys = np.arange(y_min, y_max, grid_step)

    candidates = []
    for x in xs:
        for y in ys:
            if poly.contains(Point(x, y)):
                candidates.append([x, y, speaker_height])

    return np.array(candidates, dtype=np.float32)

# ── 가구 폴리곤 추출 ────────────────────────────────────────────────
def extract_object_polygons(objects, margin=0.0):
    """
    가구 transform + dimensions에서 바닥 폴리곤 추출
    각 가구를 사각형 폴리곤으로 표현
    
    Args:
        objects: roomplan JSON의 "objects" 리스트
        margin: 가구 주변 추가 마진 (m). 스피커가 가구에서 떨어지길 원하면 양수
    
    Returns:
        list of shapely Polygon (xRIR 좌표계 x-y 평면)
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    
    polygons = []
    for obj in objects:
        if "transform" not in obj or "dimensions" not in obj:
            continue
        
        m = np.array(obj["transform"], dtype=float).reshape(4, 4, order="F")
        dims = obj["dimensions"]
        half_w = float(dims[0]) * 0.5
        half_d = float(dims[2]) * 0.5  # depth = z축
        
        # 가구 바닥 4개 모서리 (로컬 좌표)
        local_corners = np.array([
            [-half_w, 0.0, -half_d, 1.0],
            [+half_w, 0.0, -half_d, 1.0],
            [+half_w, 0.0, +half_d, 1.0],
            [-half_w, 0.0, +half_d, 1.0],
        ])
        
        # 월드 좌표로 변환
        world_corners = (m @ local_corners.T).T[:, :3]
        
        # xRIR 좌표계로 변환 + x-y 평면만
        xy_corners = []
        for wc in world_corners:
            xrir = roomplan_to_xrir_coords(*wc)
            xy_corners.append([xrir[0], xrir[1]])
        
        try:
            poly = ShapelyPolygon(xy_corners)
            if margin > 0:
                poly = poly.buffer(margin)
            if poly.is_valid and poly.area > 1e-6:
                polygons.append(poly)
        except:
            continue
    
    return polygons

# ── 스피커 적합 가구 추출 ────────────────────────────────────────

# 스피커 올릴 수 있는 카테고리
SPEAKER_FRIENDLY_CATEGORIES = {"storage", "table"}

# 가구 높이 범위
MIN_FURNITURE_HEIGHT = 0.3   # 30cm
MAX_FURNITURE_HEIGHT = 1.0   # 100cm

# 스피커-가구 마진
FURNITURE_SIZE_MARGIN = 0.03  # 3cm

def extract_speaker_friendly_furniture(objects, spk_width_m, spk_depth_m):
    """
    스피커 올릴 수 있는 가구만 추출
    
    Args:
        objects: roomplan JSON의 "objects" 리스트
        spk_width_m: 스피커 가로 (m)
        spk_depth_m: 스피커 깊이 (m)
    
    Returns:
        list of dict: [{"polygon": Polygon, "height": float, "centroid": (x,y), "category": str}]
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    
    furniture_list = []
    
    for obj in objects:
        category = obj.get("category", "")
        
        # 카테고리 필터
        if category not in SPEAKER_FRIENDLY_CATEGORIES:
            continue
        
        if "transform" not in obj or "dimensions" not in obj:
            continue
        
        dims = obj["dimensions"]
        furn_w = float(dims[0])  # 가로
        furn_h = float(dims[1])  # 높이
        furn_d = float(dims[2])  # 깊이
        
        # 높이 범위 체크
        if not (MIN_FURNITURE_HEIGHT <= furn_h <= MAX_FURNITURE_HEIGHT):
            continue
        
        # 윗면 크기 체크 (가로/세로 각각 스피커보다 큼 + 마진)
        required_w = spk_width_m + FURNITURE_SIZE_MARGIN * 2
        required_d = spk_depth_m + FURNITURE_SIZE_MARGIN * 2
        if furn_w < required_w or furn_d < required_d:
            continue
        
        # 폴리곤 생성 (xRIR 좌표계 x-y 평면)
        m = np.array(obj["transform"], dtype=float).reshape(4, 4, order="F")
        half_w = furn_w * 0.5
        half_d = furn_d * 0.5
        
        local_corners = np.array([
            [-half_w, 0.0, -half_d, 1.0],
            [+half_w, 0.0, -half_d, 1.0],
            [+half_w, 0.0, +half_d, 1.0],
            [-half_w, 0.0, +half_d, 1.0],
        ])
        world_corners = (m @ local_corners.T).T[:, :3]
        
        xy_corners = []
        for wc in world_corners:
            xrir = roomplan_to_xrir_coords(*wc)
            xy_corners.append([xrir[0], xrir[1]])
        
        try:
            poly = ShapelyPolygon(xy_corners)
            if not poly.is_valid or poly.area < 1e-6:
                continue
            
            centroid = poly.centroid
            furniture_list.append({
                "polygon": poly,
                "height": furn_h,
                "centroid": (centroid.x, centroid.y),
                "category": category,
            })
        except:
            continue
    
    return furniture_list

# ── 메인 변환 함수 ───────────────────────────────────────────────

def convert_roomplan_to_xrir_inputs(
    roomplan_json,
    output_dir,
    listener_height=1.2,
    speaker_height=1.2,
    grid_step=0.3,
    wall_margin=0.5,
):
    """
    roomplan JSON → listener.npy, xyzs.npy 저장

    Args:
        roomplan_json : dict (POST /api/optimize/speakers의 roomplan_scan)
        output_dir    : 저장할 폴더 경로
        listener_height : 청취자 귀 높이 (m)
        speaker_height  : 후보 스피커 높이 (m)
        grid_step       : 후보 격자 간격 (m)
        wall_margin     : 벽 마진 (m)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    walls = roomplan_json.get("walls", [])
    if len(walls) < 3:
        raise ValueError(f"벽이 최소 3개 필요합니다 (현재: {len(walls)}개)")

    # 청취자 위치
    listener = compute_listener_position(walls, listener_height)
    np.save(output_dir / "listener.npy", listener)
    print(f"listener.npy 저장: {listener}")

    # 후보 스피커 위치
    xyzs = generate_candidate_positions(walls, speaker_height, grid_step, wall_margin)
    np.save(output_dir / "xyzs.npy", xyzs)
    print(f"xyzs.npy 저장: {len(xyzs)}개 후보 위치")

    return listener, xyzs


# ── 테스트 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # 아까 테스트했던 가짜 roomplan JSON
    sample_roomplan = {
        "walls": [
            {"id": "w1", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2,0,0,1], "dimensions": [4.0, 2.7, 0.2]},
            {"id": "w2", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, -2,0,0,1], "dimensions": [4.0, 2.7, 0.2]},
            {"id": "w3", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,3,1], "dimensions": [6.0, 2.7, 0.2]},
            {"id": "w4", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, 0,0,-3,1], "dimensions": [6.0, 2.7, 0.2]},
        ],
        "objects": []
    }

    listener, xyzs = convert_roomplan_to_xrir_inputs(
        roomplan_json=sample_roomplan,
        output_dir="./test_output",
    )

    print(f"\n청취자 위치: {listener}")
    print(f"후보 위치 수: {len(xyzs)}")
    print(f"후보 위치 샘플:\n{xyzs[:5]}")
