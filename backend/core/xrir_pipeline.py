"""
xRIR 파이프라인 통합 모듈

roomplan JSON + ref_rir.wav (+ 선택: mesh.bin)
→ listener.npy, depth.npy 생성
→ 스테레오 후보 위치 생성 (정면 벽 기준 d × 협각)
→ xRIR 추론
→ 최적 스테레오 스피커 위치 반환
"""

import os
import sys
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from shapely.geometry import Point, Polygon, LineString

from core.roomplan_to_numpy import (
    convert_roomplan_to_xrir_inputs,
    extract_floor_polygon,
    extract_object_polygons,
)
from core.roomplan_to_depth import convert_roomplan_to_depth, ray_triangle_intersect, extract_wall_triangles

XRIR_REPO_PATH  = os.environ.get("XRIR_REPO_PATH",  "/home/piai/AcousticRooms/xRIR_code-main")
CHECKPOINT_PATH = os.environ.get("XRIR_CHECKPOINT_PATH", f"{XRIR_REPO_PATH}/checkpoints/xRIR_unseen.pth")

_model = None
_xrir_imports = None


def _load_xrir_imports():
    global _xrir_imports
    if _xrir_imports is not None:
        return _xrir_imports

    if XRIR_REPO_PATH not in sys.path:
        sys.path.append(XRIR_REPO_PATH)

    import torch
    from model.xRIR import xRIR as xRIRModel
    from inference import (
        convert_equirect_to_camera_coord,
        predict_rir,
        score_position,
    )

    _xrir_imports = {
        "torch": torch,
        "xRIR": xRIRModel,
        "convert_equirect_to_camera_coord": convert_equirect_to_camera_coord,
        "predict_rir": predict_rir,
        "score_position": score_position,
    }
    return _xrir_imports


def _get_model(device="cuda"):
    global _model
    if _model is None:
        imps = _load_xrir_imports()
        torch = imps["torch"]
        xRIRModel = imps["xRIR"]
        m = xRIRModel(num_channels=1)
        m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        m.to(device)
        m.eval()
        _model = m
        print("xRIR 모델 로드 완료")
    return _model


# ── 정면 벽 교점 계산 ────────────────────────────────────────────

def find_front_wall_point(listener_xy, forward, floor_polygon):
    """청취자에서 정면 방향으로 ray를 쏴서 방 폴리곤 경계와의 교점 계산"""
    poly = Polygon(floor_polygon.tolist())
    ray_end = listener_xy + forward * 100.0
    ray = LineString([listener_xy, ray_end])
    intersection = ray.intersection(poly.boundary)

    if intersection.is_empty:
        return None

    if intersection.geom_type == 'MultiPoint':
        points = list(intersection.geoms)
    elif intersection.geom_type == 'Point':
        points = [intersection]
    else:
        points = [intersection.centroid]

    closest = min(points, key=lambda p: np.linalg.norm(
        np.array([p.x, p.y]) - listener_xy
    ))
    return np.array([closest.x, closest.y])


# ── 스테레오 후보 위치 생성 ──────────────────────────────────────

def generate_stereo_candidates(
    listener_pos,
    initial_speaker_pos,
    floor_polygon,
    speaker_height=1.2,
    wall_margin=0.1,
):
    """정면 벽 기준 스테레오 L/R 후보 쌍 생성"""
    listener_xy = listener_pos[:2]

    forward = initial_speaker_pos[:2] - listener_xy
    dist = np.linalg.norm(forward)
    if dist < 1e-6:
        forward = np.array([1.0, 0.0])
    else:
        forward = forward / dist

    wall_point = find_front_wall_point(listener_xy, forward, floor_polygon)
    if wall_point is None:
        print("정면 벽 교점을 찾지 못했습니다. initial_speaker_pos 방향으로 fallback.")
        wall_point = initial_speaker_pos[:2]

    distances  = [0.1, 0.2, 0.3, 0.5]
    angles_deg = [40, 50, 60, 70, 80]

    poly = Polygon(floor_polygon.tolist()).buffer(-wall_margin)
    candidates = []

    for d in distances:
        base_point = wall_point - forward * d

        for angle_deg in angles_deg:
            half_rad = np.radians(angle_deg / 2)

            perp = np.array([-forward[1], forward[0]])
            spread = d * np.tan(half_rad)
            pos_L = base_point + perp * spread
            pos_R = base_point - perp * spread

            left  = np.array([pos_L[0], pos_L[1], speaker_height], dtype=np.float32)
            right = np.array([pos_R[0], pos_R[1], speaker_height], dtype=np.float32)

            if (poly.contains(Point(pos_L)) and poly.contains(Point(pos_R))):
                candidates.append((left, right, d, angle_deg))

    return candidates


def generate_candidates_with_fallback(
    listener_pos,
    initial_speaker_pos,
    floor_polygon,
    object_polygons,
    depth_np,
    speaker_height=1.2,
    min_candidates=5,
):
    """
    Fallback 전략으로 후보 생성
    1단계: 기본 (wall_margin=0.1)
    2단계: 마진 완화 (wall_margin=0.05)
    3단계: 격자 촘촘 + 마진 완화
    """
    fallback_configs = [
        # (단계 이름, wall_margin, distances, angles)
        ("1단계: 기본", 0.1, [0.1, 0.2, 0.3, 0.5], [40, 50, 60, 70, 80]),
        ("2단계: 마진 완화", 0.05, [0.1, 0.2, 0.3, 0.5], [40, 50, 60, 70, 80]),
        ("3단계: 격자 촘촘", 0.05, [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], 
         [40, 45, 50, 55, 60, 65, 70, 75, 80]),
    ]
    
    for stage_name, margin, distances, angles in fallback_configs:
        print(f"\n{stage_name} (margin={margin}, d={len(distances)}개, angle={len(angles)}개)")
        
        candidates = _generate_stereo_candidates_custom(
            listener_pos, initial_speaker_pos, floor_polygon,
            speaker_height=speaker_height,
            wall_margin=margin,
            distances=distances,
            angles_deg=angles,
        )
        
        # 가구 + depth 필터링
        valid = []
        filtered_furniture = 0
        filtered_obstacle = 0
        
        for left, right, d, angle in candidates:
            # 가구 체크
            left_pt = Point(left[0], left[1])
            right_pt = Point(right[0], right[1])
            in_furniture = False
            for furn_poly in object_polygons:
                if furn_poly.contains(left_pt) or furn_poly.contains(right_pt):
                    in_furniture = True
                    break
            if in_furniture:
                filtered_furniture += 1
                continue
            
            # depth 장애물 체크
            if not (check_obstacle_depth(listener_pos, left, depth_np) and
                    check_obstacle_depth(listener_pos, right, depth_np)):
                filtered_obstacle += 1
                continue
            
            valid.append((left, right, d, angle))
        
        print(f"  전체: {len(candidates)}쌍 / 가구 제외: {filtered_furniture} / 장애물 제외: {filtered_obstacle} / 유효: {len(valid)}쌍")
        
        if len(valid) >= min_candidates:
            print(f"  → {stage_name}에서 충분한 후보 확보. 진행.")
            return valid, stage_name
    
    # 3단계도 부족하면 그대로 반환 (0일 수도 있음)
    print(f"\n경고: 3단계 fallback 후에도 후보 {len(valid)}쌍만 있음")
    return valid, "3단계 (부족)"

# ── depth.npy 기반 장애물 체크 ───────────────────────────────────

def _generate_stereo_candidates_custom(
    listener_pos, initial_speaker_pos, floor_polygon,
    speaker_height=1.2, wall_margin=0.1,
    distances=None, angles_deg=None,
):
    """generate_stereo_candidates의 distances/angles 커스텀 버전"""
    if distances is None:
        distances = [0.1, 0.2, 0.3, 0.5]
    if angles_deg is None:
        angles_deg = [40, 50, 60, 70, 80]
    
    listener_xy = listener_pos[:2]
    
    forward = initial_speaker_pos[:2] - listener_xy
    dist = np.linalg.norm(forward)
    if dist < 1e-6:
        forward = np.array([1.0, 0.0])
    else:
        forward = forward / dist
    
    wall_point = find_front_wall_point(listener_xy, forward, floor_polygon)
    if wall_point is None:
        wall_point = initial_speaker_pos[:2]
    
    poly = Polygon(floor_polygon.tolist()).buffer(-wall_margin)
    candidates = []
    
    for d in distances:
        base_point = wall_point - forward * d
        for angle_deg in angles_deg:
            half_rad = np.radians(angle_deg / 2)
            perp = np.array([-forward[1], forward[0]])
            spread = d * np.tan(half_rad)
            pos_L = base_point + perp * spread
            pos_R = base_point - perp * spread
            
            left  = np.array([pos_L[0], pos_L[1], speaker_height], dtype=np.float32)
            right = np.array([pos_R[0], pos_R[1], speaker_height], dtype=np.float32)
            
            if (poly.contains(Point(pos_L)) and poly.contains(Point(pos_R))):
                candidates.append((left, right, d, angle_deg))
    
    return candidates

# ── depth.npy 기반 장애물 체크 ───────────────────────────────────

def check_obstacle_depth(
    listener_pos, candidate_pos, depth_np, img_h=256, img_w=512, threshold=0.5
):
    """depth map에서 청취자 → 후보 위치 직선 경로 장애물 체크"""
    direction = candidate_pos[:2] - listener_pos[:2]
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return True

    direction_3d = np.array([direction[0], direction[1], 0.0]) / dist

    theta = np.arctan2(direction_3d[1], direction_3d[0])
    phi   = np.arctan2(direction_3d[2], np.sqrt(direction_3d[0]**2 + direction_3d[1]**2))

    ti = int((theta + np.pi) / (2 * np.pi) * img_w) % img_w
    pi = int((phi + np.pi/2) / np.pi * img_h) % img_h

    depth_val = depth_np[pi, ti]

    if depth_val < dist - threshold:
        return False
    return True


# ── 메인 파이프라인 ──────────────────────────────────────────────

def run_xrir_pipeline(
    roomplan_json: dict,
    ref_rir_bytes: bytes,
    ref_src_pos: np.ndarray,
    initial_speaker_pos: np.ndarray,
    top_k: int = 2,
    listener_height: float = 1.2,
    speaker_height: float = 1.2,
    wall_margin: float = 0.1,
    furniture_margin: float = 0.1,
    mesh_bin_path: str = None,
) -> list:
    imps = _load_xrir_imports()
    torch = imps["torch"]
    convert_equirect_to_camera_coord = imps["convert_equirect_to_camera_coord"]
    predict_rir = imps["predict_rir"]
    score_position = imps["score_position"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── 1. ref_rir.wav 저장 ──────────────────────────────────
        ref_rir_path = tmpdir / "ref_rir.wav"
        ref_rir_path.write_bytes(ref_rir_bytes)
        ref_rir_np, sr = sf.read(str(ref_rir_path))
        ref_rir_np = ref_rir_np.astype(np.float32)

        # ── 2. roomplan JSON → listener.npy ──────────────────────
        walls = roomplan_json.get("walls", [])
        objects = roomplan_json.get("objects", [])
        listener, _ = convert_roomplan_to_xrir_inputs(
            roomplan_json=roomplan_json,
            output_dir=str(tmpdir),
            listener_height=listener_height,
            speaker_height=speaker_height,
        )
        listener_pos = listener.astype(np.float32)

        # ── 3. depth.npy 생성 ────────────────────────────────────
        object_polygons = extract_object_polygons(objects, margin=furniture_margin)
        print(f"가구 {len(object_polygons)}개 감지됨")

        # ── 4. depth.npy 생성 ────────────────────────────────────
        depth_np = convert_roomplan_to_depth(
            roomplan_json=roomplan_json,
            listener_pos=listener,
            output_dir=str(tmpdir),
            mesh_bin_path=mesh_bin_path,
        )

        # ── 5. 스테레오 후보 위치 생성 ───────────────────────────
        floor_polygon = extract_floor_polygon(walls)
        valid_candidates, stage_used = generate_candidates_with_fallback(
            listener_pos=listener_pos,
            initial_speaker_pos=initial_speaker_pos,
            floor_polygon=floor_polygon,
            object_polygons=object_polygons,
            depth_np=depth_np,
            speaker_height=speaker_height,
            min_candidates=5,
        )

        if not valid_candidates:
            print("유효한 후보 없음")
            return []

        print(f"\n최종 사용: {stage_used}, 후보 {len(valid_candidates)}쌍")

        # 필터링
        valid_candidates = []
        filtered_furniture = 0
        filtered_obstacle = 0
        
        for left, right, d, angle in candidates:
            # 가구 폴리곤 체크
            left_pt = Point(left[0], left[1])
            right_pt = Point(right[0], right[1])
            
            in_furniture = False
            for furn_poly in object_polygons:
                if furn_poly.contains(left_pt) or furn_poly.contains(right_pt):
                    in_furniture = True
                    break
            if in_furniture:
                filtered_furniture += 1
                continue
            
            # depth.npy 장애물 체크
            if not (check_obstacle_depth(listener_pos, left, depth_np) and
                    check_obstacle_depth(listener_pos, right, depth_np)):
                filtered_obstacle += 1
                continue
            
            valid_candidates.append((left, right, d, angle))

        print(f"전체 후보: {len(candidates)}쌍")
        print(f"  가구 충돌 제외: {filtered_furniture}쌍")
        print(f"  장애물 제외: {filtered_obstacle}쌍")
        print(f"  유효 후보: {len(valid_candidates)}쌍")

        if not valid_candidates:
            print("유효한 후보 없음")
            return []

        # ── 5. xRIR 추론 ─────────────────────────────────────────
        depth_tensor = torch.from_numpy(depth_np.astype(np.float32))
        depth_coord  = convert_equirect_to_camera_coord(depth_tensor)
        model = _get_model(str(device))

        raw_results = []
        print(f"{len(valid_candidates)}쌍 음향 점수 예측 중...")

        for i, (left, right, d, angle) in enumerate(valid_candidates):
            rir_L = predict_rir(
                model, depth_coord, ref_rir_np,
                ref_src_pos, left, listener_pos, device=str(device)
            )
            rir_R = predict_rir(
                model, depth_coord, ref_rir_np,
                ref_src_pos, right, listener_pos, device=str(device)
            )

            score_L = score_position(rir_L, sr=sr)
            score_R = score_position(rir_R, sr=sr)

            pair_score = (score_L["total"] + score_R["total"]) / 2

            raw_results.append({
                "left":       left,
                "right":      right,
                "d":          d,
                "angle":      angle,
                "score_L":    score_L,
                "score_R":    score_R,
                "pair_score": pair_score,
            })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(valid_candidates)} 완료...")

        # ── 6. 정렬 후 상위 K개 반환 ─────────────────────────────
        raw_results.sort(key=lambda x: x["pair_score"], reverse=True)
        top_raw = raw_results[:top_k]

        results = []
        for rank, r in enumerate(top_raw):
            left  = r["left"]
            right = r["right"]
            sL    = r["score_L"]
            sR    = r["score_R"]

            result = {
                "placement": {
                    "left": {
                        "x": round(float(left[0]), 3),
                        "y": round(float(left[1]), 3),
                        "z": round(float(left[2]), 3),
                    },
                    "right": {
                        "x": round(float(right[0]), 3),
                        "y": round(float(right[1]), 3),
                        "z": round(float(right[2]), 3),
                    },
                    "listener": {
                        "x": round(float(listener_pos[0]), 3),
                        "y": round(float(listener_pos[1]), 3),
                        "z": round(float(listener_pos[2]), 3),
                    },
                },
                "score": round(float(r["pair_score"]), 4),
                "metrics": {
                    "edt_seconds": round((sL["edt"] + sR["edt"]) / 2, 3),
                    "c50_db":      round((sL["c50"] + sR["c50"]) / 2, 2),
                    "t60_seconds": round((sL["t60"] + sR["t60"]) / 2, 3),
                    "edt_score":   round((sL["edt_score"] + sR["edt_score"]) / 2, 3),
                    "c50_score":   round((sL["c50_score"] + sR["c50_score"]) / 2, 3),
                    "t60_score":   round((sL["t60_score"] + sR["t60_score"]) / 2, 3),
                },
                "angle_deg":  r["angle"],
                "distance_m": r["d"],
                "rank": rank,
            }
            results.append(result)

            print(f"{rank+1}위: L({left[0]:.2f}, {left[1]:.2f}) "
                  f"R({right[0]:.2f}, {right[1]:.2f}) "
                  f"협각={r['angle']}° d={r['d']}m "
                  f"| score={r['pair_score']:.3f}")

        return results