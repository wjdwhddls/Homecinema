"""
앱으로 스캔한 roomplan JSON에서 후보 쌍 생성
backend/data/roomplan_scans/scan_{job_id}.json 사용
"""

import sys
sys.path.append("/home/piai/Homecinema/backend")

import numpy as np
import json
from pathlib import Path

from core.xrir_pipeline import generate_candidates_with_fallback
from core.roomplan_to_numpy import (
    compute_listener_position,
    extract_floor_polygon,
    extract_object_polygons,
)
from core.roomplan_to_depth import convert_roomplan_to_depth
from core.initial_speaker_position import compute_initial_speaker_position


SCAN_DIR = Path("/home/piai/Homecinema/backend/data/roomplan_scans")
EXPERIMENT_DIR = Path("/home/piai/Homecinema/backend/data/experiment")


def check_candidates(job_id, room_name,
                     listener_height=1.2, speaker_height=1.2,
                     furniture_margin=0.1):
    """
    Args:
        job_id: scan_{job_id}.json의 job_id (UUID)
        room_name: experiment 폴더 안에 만들 방 이름
        furniture_margin: 가구 주변 추가 마진 (m)
    """
    # roomplan JSON 로드
    scan_path = SCAN_DIR / f"scan_{job_id}.json"
    if not scan_path.exists():
        raise FileNotFoundError(f"스캔 파일 없음: {scan_path}")
    
    with open(scan_path) as f:
        roomplan_json = json.load(f)
    
    walls = roomplan_json["walls"]
    objects = roomplan_json.get("objects", [])
    listener_pos = compute_listener_position(walls, listener_height)
    initial_speaker_pos = compute_initial_speaker_position(
        walls=walls, listener_pos=listener_pos, speaker_height=speaker_height
    )
    floor_polygon = extract_floor_polygon(walls)
    
    # 가구 폴리곤 추출
    object_polygons = extract_object_polygons(objects, margin=furniture_margin)
    print(f"가구 {len(object_polygons)}개 감지됨")
    
    output_dir = EXPERIMENT_DIR / room_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # depth.npy 생성
    print("depth.npy 생성 중...")
    depth_np = convert_roomplan_to_depth(
        roomplan_json=roomplan_json,
        listener_pos=listener_pos,
        output_dir=str(output_dir),
    )
    
    # 후보 생성 + 필터링 (fallback 전략 적용)
    valid_candidates, stage_used = generate_candidates_with_fallback(
        listener_pos=listener_pos,
        initial_speaker_pos=initial_speaker_pos,
        floor_polygon=floor_polygon,
        object_polygons=object_polygons,
        depth_np=depth_np,
        speaker_height=speaker_height,
        min_candidates=5,
    )
    
    print(f"\n최종 사용: {stage_used}, 후보 {len(valid_candidates)}쌍")
    
    # JSON 형식으로 변환
    valid = []
    for left, right, d, angle in valid_candidates:
        valid.append({
            "id": len(valid),
            "left":  {"x": float(left[0]),  "y": float(left[1]),  "z": float(left[2])},
            "right": {"x": float(right[0]), "y": float(right[1]), "z": float(right[2])},
            "d_m": d,
            "angle_deg": angle,
        })
    
    # candidates.json 저장
    with open(output_dir / "candidates.json", "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job_id,
            "room_name": room_name,
            "stage_used": stage_used,
            "listener": {
                "x": float(listener_pos[0]),
                "y": float(listener_pos[1]),
                "z": float(listener_pos[2]),
            },
            "initial_speaker": {
                "x": float(initial_speaker_pos[0]),
                "y": float(initial_speaker_pos[1]),
                "z": float(initial_speaker_pos[2]),
            },
            "candidates": valid,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"저장: {output_dir / 'candidates.json'}")
    
    print("\n=== 측정 위치 ===")
    for c in valid:
        L, R = c["left"], c["right"]
        print(f"쌍 {c['id']}: L({L['x']:.2f}, {L['y']:.2f}) R({R['x']:.2f}, {R['y']:.2f}) "
              f"| d={c['d_m']}m, 협각={c['angle_deg']}°")
    
    return valid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="scan_{job_id}.json의 job_id")
    parser.add_argument("--room-name", required=True, help="experiment 폴더 안 방 이름")
    args = parser.parse_args()
    
    check_candidates(args.job_id, args.room_name)