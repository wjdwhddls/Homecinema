"""
xRIR 파이프라인 통합 모듈

roomplan JSON + ref_rir.wav (+ 선택: mesh.bin)
→ listener.npy, depth.npy 생성
→ 스테레오 후보 위치 생성 (정면 벽 기준 d × 협각)
→ xRIR 추론
→ 최적 스테레오 스피커 위치 반환
"""

import logging
import os
import sys
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from scipy.stats import linregress
from shapely.geometry import Point, Polygon, LineString

from core.roomplan_to_numpy import (
    convert_roomplan_to_xrir_inputs,
    extract_floor_polygon,
    extract_object_polygons,
)
from core.roomplan_to_depth import convert_roomplan_to_depth, ray_triangle_intersect, extract_wall_triangles

logger = logging.getLogger(__name__)

XRIR_REPO_PATH  = os.environ.get("XRIR_REPO_PATH")
CHECKPOINT_PATH = os.environ.get("XRIR_CHECKPOINT_PATH")
if not XRIR_REPO_PATH or not CHECKPOINT_PATH:
    raise RuntimeError(
        "XRIR_REPO_PATH 와 XRIR_CHECKPOINT_PATH 환경변수가 설정되어야 합니다 "
        "(backend/.env 확인)"
    )


def _select_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

_model = None
_xrir_imports = None


# ── 음향 지표 계산 (T60 / C50 / EDT) ───────────────────────────
# RIR 한 개에서 직접 계산. 외부 inference.score_position 의존 제거.

def _schroeder_edc_db(rir: np.ndarray) -> np.ndarray:
    """Schroeder 역적분 EDC를 dB로 (peak를 0 dB 기준)."""
    energy = np.asarray(rir, dtype=np.float64).squeeze() ** 2
    schroeder = np.cumsum(energy[::-1])[::-1]
    peak = float(np.max(schroeder))
    if peak <= 0:
        return np.zeros_like(schroeder)
    return 10.0 * np.log10(schroeder / peak + 1e-20)


def _decay_time_from_edc(edc_db: np.ndarray, sr: int,
                          start_db: float, end_db: float) -> float:
    """EDC에서 [start_db, end_db] 구간 선형회귀 → 60 dB 외삽."""
    t = np.arange(len(edc_db)) / sr
    mask = (edc_db <= start_db) & (edc_db >= end_db)
    if mask.sum() < 5:
        return 0.5  # fallback
    slope = linregress(t[mask], edc_db[mask]).slope
    if slope >= 0:
        return 0.5
    return float(np.clip(-60.0 / slope, 0.05, 5.0))


def _compute_t60(rir: np.ndarray, sr: int) -> float:
    """T60 — Schroeder EDC, -5 ~ -35 dB 구간 회귀(=T30식)."""
    return _decay_time_from_edc(_schroeder_edc_db(rir), sr, -5.0, -35.0)


def _compute_edt(rir: np.ndarray, sr: int) -> float:
    """EDT — Schroeder EDC, 0 ~ -10 dB 구간 회귀를 -60 dB로 외삽.
    초기 감쇠만 보므로 RIR 길이/SNR 영향 큼 → linregress 사용해 안정성 확보."""
    return _decay_time_from_edc(_schroeder_edc_db(rir), sr, 0.0, -10.0)


def _compute_c50(rir: np.ndarray, sr: int) -> float:
    """C50 — early(0~50 ms) / late(50 ms~) 에너지 비 (dB)."""
    rir = np.asarray(rir, dtype=np.float64).squeeze()
    split = int(0.050 * sr)
    early = float(np.sum(rir[:split] ** 2))
    late  = float(np.sum(rir[split:] ** 2))
    if late < 1e-10:
        return 20.0
    return float(10.0 * np.log10(early / late))


def _compute_metrics(rir: np.ndarray, sr: int) -> dict:
    """RIR 한 개의 (t60, c50, edt) 묶음."""
    return {
        "t60": _compute_t60(rir, sr),
        "c50": _compute_c50(rir, sr),
        "edt": _compute_edt(rir, sr),
    }


def _compute_subscore(t60: float, c50: float, edt: float):
    """T60/C50/EDT raw → 0~1 sub-score + 가중 합.

    Targets / 분모 / 가중치:
    - T60: target 0.40 s, denom 0.40, weight 0.4
    - C50: target  +2 dB, denom 20 dB, weight 0.3
    - EDT: target 0.35 s, denom 0.35, weight 0.3

    EDT target은 거실/홈시어터 BR ≈ T60 × 0.85~0.95 가이드라인 기반.
    Returns: (t60_score, c50_score, edt_score, total) — 모두 0~1.
    """
    t60_score = max(0.0, 1.0 - abs(t60 - 0.40) / 0.40)
    c50_score = max(0.0, 1.0 - abs(c50 -  2.0) / 20.0)
    edt_score = max(0.0, 1.0 - abs(edt - 0.35) / 0.35)
    total     = 0.4 * t60_score + 0.3 * c50_score + 0.3 * edt_score
    return t60_score, c50_score, edt_score, total


def _load_xrir_imports():
    global _xrir_imports
    if _xrir_imports is not None:
        return _xrir_imports

    if XRIR_REPO_PATH not in sys.path:
        sys.path.append(XRIR_REPO_PATH)

    import torch
    from model.xRIR_ConvNeXT import xRIR as xRIRModel
    from inference import (
        convert_equirect_to_camera_coord,
        predict_rir,
    )

    _xrir_imports = {
        "torch": torch,
        "xRIR": xRIRModel,
        "convert_equirect_to_camera_coord": convert_equirect_to_camera_coord,
        "predict_rir": predict_rir,
    }
    return _xrir_imports


def _get_model(device: str | None = None):
    global _model
    if _model is None:
        imps = _load_xrir_imports()
        torch = imps["torch"]
        xRIRModel = imps["xRIR"]
        if device is None:
            device = _select_device()
        m = xRIRModel(num_channels=1)
        m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        m.to(device)
        m.eval()
        _model = m
        logger.info(f"xRIR 모델 로드 완료 (device={device})")
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
        logger.info("정면 벽 교점을 찾지 못했습니다. initial_speaker_pos 방향으로 fallback.")
        wall_point = initial_speaker_pos[:2]

    distances  = [0.1, 0.2, 0.3, 0.5]
    angles_deg = [40, 50, 60, 70, 80]

    poly = Polygon(floor_polygon.tolist()).buffer(-wall_margin)
    candidates = []

    for d in distances:
        base_point = wall_point - forward * d
        listener_to_base = np.linalg.norm(base_point - listener_xy)

        for angle_deg in angles_deg:
            half_rad = np.radians(angle_deg / 2)

            perp = np.array([-forward[1], forward[0]])
            spread = listener_to_base * np.tan(half_rad)
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
        # 4단계: 작은 방 / 짧은 정면 대응 — 청취자에 더 가까운 base + 좁은 stereo 허용
        ("4단계: 작은 방 대응", 0.05, [0.7, 1.0, 1.3, 1.6, 2.0],
         [25, 30, 35, 40, 50]),
    ]
    
    for stage_name, margin, distances, angles in fallback_configs:
        logger.info(f"\n{stage_name} (margin={margin}, d={len(distances)}개, angle={len(angles)}개)")
        
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
        
        logger.info(f"  전체: {len(candidates)}쌍 / 가구 제외: {filtered_furniture} / 장애물 제외: {filtered_obstacle} / 유효: {len(valid)}쌍")
        
        if len(valid) >= min_candidates:
            logger.info(f"  → {stage_name}에서 충분한 후보 확보. 진행.")
            return valid, stage_name
    
    # 모든 fallback 단계도 부족하면 그대로 반환 (0일 수도 있음)
    logger.warning("4단계 fallback 후에도 후보 %d쌍만 확보됨", len(valid))
    return valid, f"4단계 fallback 후 {len(valid)}쌍 (부족)"

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
        listener_to_base = np.linalg.norm(base_point - listener_xy)
        
        for angle_deg in angles_deg:
            half_rad = np.radians(angle_deg / 2)
            perp = np.array([-forward[1], forward[0]])
            spread = listener_to_base * np.tan(half_rad)
            pos_L = base_point + perp * spread
            pos_R = base_point - perp * spread
            
            left  = np.array([pos_L[0], pos_L[1], speaker_height], dtype=np.float32)
            right = np.array([pos_R[0], pos_R[1], speaker_height], dtype=np.float32)
            
            if (poly.contains(Point(pos_L)) and poly.contains(Point(pos_R))):
                candidates.append((left, right, d, angle_deg))
    
    return candidates

# ── 가구 위 후보 생성 ────────────────────────────────────────────

HEIGHT_TOLERANCE_M = 0.1            # L/R 가구 높이 차이 허용
MIN_LR_DISTANCE_M = 1.5             # L↔R 사이 최소 거리 (좁은 stereo 하한)
MAX_LR_DISTANCE_M = 3.5             # L↔R 사이 최대 거리 (넓은 stereo 상한)
MAX_LISTENER_DIST_DIFF_M = 1.0      # 청취자→L vs 청취자→R 거리 차이 허용
MAX_DEPTH_DIFF_M = 0.5              # 정면 방향 깊이 차이 허용 (좌우 정렬)

def generate_furniture_top_candidates(
    listener_pos,
    initial_speaker_pos,
    speaker_friendly_furniture,
    spk_height_m,
):
    """
    좌우 가구 페어링해서 가구 위 L/R 후보 쌍 생성
    
    Args:
        listener_pos: 청취자 위치 [x, y, z]
        initial_speaker_pos: 임시 스피커 위치 (정면 방향 판단용)
        speaker_friendly_furniture: extract_speaker_friendly_furniture 결과
        spk_height_m: 스피커 자체 높이 (m)
    
    Returns:
        list of (left_pos, right_pos, "furniture", furn_height)
    """
    listener_xy = np.array(listener_pos[:2])
    
    # 정면 방향 벡터
    forward = np.array(initial_speaker_pos[:2]) - listener_xy
    dist = np.linalg.norm(forward)
    if dist < 1e-6:
        forward = np.array([1.0, 0.0])
    else:
        forward = forward / dist
    
    # 좌우 판단: 정면 기준 외적
    perp = np.array([-forward[1], forward[0]])  # 왼쪽 방향 vector
    
    # 좌/우 가구 분류
    left_furniture = []
    right_furniture = []
    
    for furn in speaker_friendly_furniture:
        cx, cy = furn["centroid"]
        offset = np.array([cx, cy]) - listener_xy
        
        # 정면 거리 (앞쪽이어야 함)
        forward_dist = np.dot(offset, forward)
        if forward_dist < 0.3:  # 청취자 뒤쪽 또는 너무 가까운 가구 제외
            continue
        
        # 좌우 판단
        side = np.dot(offset, perp)
        if side > 0:
            left_furniture.append(furn)
        else:
            right_furniture.append(furn)
    
    logger.info(f"  좌측 가구: {len(left_furniture)}개, 우측 가구: {len(right_furniture)}개")

    # L/R 페어링 — 높이/거리/대칭성 검증
    candidates = []
    rejected = {"height": 0, "lr_distance": 0, "asymmetry": 0, "depth": 0}

    for left_furn in left_furniture:
        for right_furn in right_furniture:
            # (1) 높이 차이
            height_diff = abs(left_furn["height"] - right_furn["height"])
            if height_diff > HEIGHT_TOLERANCE_M:
                rejected["height"] += 1
                continue

            l_xy = np.array(left_furn["centroid"], dtype=np.float64)
            r_xy = np.array(right_furn["centroid"], dtype=np.float64)

            # (2) L↔R 거리 (일반 stereo 권장 1.5~3.5m)
            lr_distance = np.linalg.norm(l_xy - r_xy)
            if lr_distance < MIN_LR_DISTANCE_M or lr_distance > MAX_LR_DISTANCE_M:
                rejected["lr_distance"] += 1
                continue

            # (3) 청취자 거리 비대칭 (좌/우 한쪽이 더 멀면 음상이 한쪽으로 끌림)
            l_dist = np.linalg.norm(l_xy - listener_xy)
            r_dist = np.linalg.norm(r_xy - listener_xy)
            if abs(l_dist - r_dist) > MAX_LISTENER_DIST_DIFF_M:
                rejected["asymmetry"] += 1
                continue

            # (4) 정면 방향 깊이 차이 (한쪽이 앞 다른쪽이 뒤)
            l_depth = float(np.dot(l_xy - listener_xy, forward))
            r_depth = float(np.dot(r_xy - listener_xy, forward))
            if abs(l_depth - r_depth) > MAX_DEPTH_DIFF_M:
                rejected["depth"] += 1
                continue

            # 스피커는 가구 윗면 중앙에 음향 중심
            avg_furn_height = (left_furn["height"] + right_furn["height"]) / 2
            spk_z = avg_furn_height + spk_height_m / 2

            left_pos = np.array([l_xy[0], l_xy[1], spk_z], dtype=np.float32)
            right_pos = np.array([r_xy[0], r_xy[1], spk_z], dtype=np.float32)

            candidates.append((left_pos, right_pos, "furniture", avg_furn_height))

    if any(rejected.values()):
        logger.info(f"  가구 페어 reject: 높이={rejected['height']}, LR거리={rejected['lr_distance']}, "
              f"비대칭={rejected['asymmetry']}, 깊이차={rejected['depth']}")

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
    speaker_dimensions: dict,
    top_k: int = 2,
    listener_height: float = 1.2,
    wall_margin: float = 0.1,
    furniture_margin: float = 0.1,
    mesh_bin_path: str = None,
) -> tuple[list, str]:
    """
    Returns:
        (results, no_results_reason)
        - results: top-K placements list (empty if 0 valid candidates)
        - no_results_reason: results 비어있을 때 사유 문자열, 정상이면 ""
    """
    from core.roomplan_to_numpy import extract_speaker_friendly_furniture
    
    imps = _load_xrir_imports()
    torch = imps["torch"]
    convert_equirect_to_camera_coord = imps["convert_equirect_to_camera_coord"]
    predict_rir = imps["predict_rir"]

    device = torch.device(_select_device())

    # 스피커 크기 (cm → m)
    spk_w = speaker_dimensions["width_cm"] / 100
    spk_h = speaker_dimensions["height_cm"] / 100
    spk_d = speaker_dimensions["depth_cm"] / 100
    
    # 바닥 배치 시 스피커 음향 중심 높이
    speaker_height_floor = spk_h / 2

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
            speaker_height=speaker_height_floor,
        )
        listener_pos = listener.astype(np.float32)

        # ── 3. 가구 폴리곤 추출 ──────────────────────────────────
        # 충돌 검사용 (모든 가구)
        object_polygons = extract_object_polygons(objects, margin=furniture_margin)
        logger.info(f"가구 {len(object_polygons)}개 감지됨 (충돌 검사용)")
        
        # 스피커 올릴 수 있는 가구
        speaker_friendly = extract_speaker_friendly_furniture(objects, spk_w, spk_d)
        logger.info(f"스피커 올릴 수 있는 가구 {len(speaker_friendly)}개")

        # ── 4. depth.npy 생성 ────────────────────────────────────
        depth_np = convert_roomplan_to_depth(
            roomplan_json=roomplan_json,
            listener_pos=listener,
            output_dir=str(tmpdir),
            mesh_bin_path=mesh_bin_path,
        )

        # ── 5-A. 바닥 후보 생성 ──────────────────────────────────
        floor_polygon = extract_floor_polygon(walls)
        floor_candidates, stage_used = generate_candidates_with_fallback(
            listener_pos=listener_pos,
            initial_speaker_pos=initial_speaker_pos,
            floor_polygon=floor_polygon,
            object_polygons=object_polygons,
            depth_np=depth_np,
            speaker_height=speaker_height_floor,
            min_candidates=5,
        )
        logger.info(f"\n바닥 후보 ({stage_used}): {len(floor_candidates)}쌍")
        
        # ── 5-B. 가구 위 후보 생성 ──────────────────────────────
        furniture_candidates = generate_furniture_top_candidates(
            listener_pos=listener_pos,
            initial_speaker_pos=initial_speaker_pos,
            speaker_friendly_furniture=speaker_friendly,
            spk_height_m=spk_h,
        )
        logger.info(f"가구 위 후보: {len(furniture_candidates)}쌍")

        # ── 5-C. 후보 통합 ───────────────────────────────────────
        all_candidates = []
        for l, r, d, ang in floor_candidates:
            all_candidates.append({
                "left": l, "right": r,
                "placement": "floor",
                "d": d, "angle": ang, "furn_height": 0.0,
            })
        for l, r, _, furn_h in furniture_candidates:
            all_candidates.append({
                "left": l, "right": r,
                "placement": "furniture",
                "d": None, "angle": None, "furn_height": furn_h,
            })
        
        if not all_candidates:
            # fallback 단계와 가구 정보를 사유로 묶어 반환
            reason = (
                f"바닥 후보 0쌍 (마지막 fallback={stage_used}), "
                f"가구 위 후보 0쌍 (스피커 친화 가구 {len(speaker_friendly)}개 중 페어 가능한 쌍 없음)"
            )
            logger.warning("유효한 후보 없음 — %s", reason)
            return [], reason

        logger.info(f"\n총 후보: {len(all_candidates)}쌍 (바닥 {len(floor_candidates)} + 가구 위 {len(furniture_candidates)})")

        # ── 6. xRIR 추론 ─────────────────────────────────────────
        depth_tensor = torch.from_numpy(depth_np.astype(np.float32))
        depth_coord  = convert_equirect_to_camera_coord(depth_tensor)
        model = _get_model(str(device))

        raw_results = []
        logger.info(f"{len(all_candidates)}쌍 음향 점수 예측 중...")

        for i, cand in enumerate(all_candidates):
            rir_L = predict_rir(
                model, depth_coord, ref_rir_np,
                ref_src_pos, cand["left"], listener_pos, device=str(device)
            )
            rir_R = predict_rir(
                model, depth_coord, ref_rir_np,
                ref_src_pos, cand["right"], listener_pos, device=str(device)
            )

            metrics_L = _compute_metrics(rir_L, sr=sr)
            metrics_R = _compute_metrics(rir_R, sr=sr)

            sub_L = _compute_subscore(metrics_L["t60"], metrics_L["c50"], metrics_L["edt"])
            sub_R = _compute_subscore(metrics_R["t60"], metrics_R["c50"], metrics_R["edt"])
            pair_score = (sub_L[3] + sub_R[3]) / 2

            raw_results.append({
                **cand,
                "metrics_L": metrics_L,
                "metrics_R": metrics_R,
                "sub_L":     sub_L,
                "sub_R":     sub_R,
                "pair_score": pair_score,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(all_candidates)} 완료...")

        # ── 7. 정렬 후 상위 K개 반환 ─────────────────────────────
        raw_results.sort(key=lambda x: x["pair_score"], reverse=True)
        top_raw = raw_results[:top_k]

        results = []
        for rank, r in enumerate(top_raw):
            left  = r["left"]
            right = r["right"]
            mL    = r["metrics_L"]
            mR    = r["metrics_R"]
            sub_L = r["sub_L"]
            sub_R = r["sub_R"]

            result = {
                "placement_type": r["placement"],
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
                    "t60_seconds": round((mL["t60"] + mR["t60"]) / 2, 3),
                    "c50_db":      round((mL["c50"] + mR["c50"]) / 2, 2),
                    "edt_seconds": round((mL["edt"] + mR["edt"]) / 2, 3),
                    "t60_score":   round((sub_L[0] + sub_R[0]) / 2, 4),
                    "c50_score":   round((sub_L[1] + sub_R[1]) / 2, 4),
                    "edt_score":   round((sub_L[2] + sub_R[2]) / 2, 4),
                },
                "angle_deg":          r.get("angle"),
                "distance_m":         r.get("d"),
                "furniture_height_m": round(r.get("furn_height", 0.0), 2),
                "rank": rank,
            }
            results.append(result)

            placement_str = "가구 위" if r["placement"] == "furniture" else "바닥"
            logger.info(f"{rank+1}위 [{placement_str}]: "
                  f"L({left[0]:.2f}, {left[1]:.2f}, {left[2]:.2f}) "
                  f"R({right[0]:.2f}, {right[1]:.2f}, {right[2]:.2f}) "
                  f"| score={r['pair_score']:.3f}")

        return results, ""