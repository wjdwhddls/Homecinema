"""
실험용: 각 후보 쌍 위치에서 측정한 sweep 녹음을 RIR로 변환
- 사용자가 직접 측정한 wav를 입력으로 받음
- recorded_pair_X_L.wav 형태로 저장된 녹음을 deconvolution
- gt_rirs/pair_X_L.wav 형태로 RIR 저장

폴더 구조:
experiment/방이름/
├── candidates.json
├── recordings/              ← 사용자가 측정한 sweep 녹음 저장
│   ├── pair_0_L.wav         ← L 위치에서 sweep 재생/녹음 결과
│   ├── pair_0_R.wav
│   └── ...
└── gt_rirs/                 ← 이 스크립트가 생성
    ├── pair_0_L.wav
    └── ...

* sweep.wav는 SCAN_DIR/{job_id}/sweep.wav 사용 (같은 sweep 신호 가정)
"""

import sys
sys.path.append("/home/piai/Homecinema/backend")

import json
from pathlib import Path

from core.sweep_deconvolution import deconvolve_sweep


SCAN_DIR = Path("/home/piai/Homecinema/backend/data/roomplan_scans")
EXPERIMENT_DIR = Path("/home/piai/Homecinema/backend/data/experiment")


def generate_gt_rirs(job_id, room_name):
    """
    Args:
        job_id: scan_{job_id}.json의 job_id (sweep.wav 가져올 위치)
        room_name: experiment 폴더의 방 이름
    """
    room_dir = EXPERIMENT_DIR / room_name
    recordings_dir = room_dir / "recordings"
    gt_dir = room_dir / "gt_rirs"
    gt_dir.mkdir(exist_ok=True)
    
    # sweep 신호 (앱에서 사용한 것과 동일)
    sweep_path = SCAN_DIR / job_id / "sweep.wav"
    if not sweep_path.exists():
        raise FileNotFoundError(f"sweep.wav 없음: {sweep_path}")
    
    # candidates.json에서 후보 수 확인
    with open(room_dir / "candidates.json") as f:
        candidates = json.load(f)["candidates"]
    
    print(f"{len(candidates)}쌍 deconvolution 시작...")
    
    success_count = 0
    missing = []
    
    for c in candidates:
        i = c["id"]
        for side in ["L", "R"]:
            recorded_path = recordings_dir / f"pair_{i}_{side}.wav"
            if not recorded_path.exists():
                missing.append(f"pair_{i}_{side}.wav")
                continue
            
            output_path = gt_dir / f"pair_{i}_{side}.wav"
            deconvolve_sweep(
                recorded_path=str(recorded_path),
                sweep_path=str(sweep_path),
                output_path=str(output_path),
            )
            success_count += 1
    
    print(f"\n완료: {success_count}개 RIR 생성")
    if missing:
        print(f"누락된 측정 파일 ({len(missing)}개):")
        for m in missing:
            print(f"  - {m}")
    print(f"저장 경로: {gt_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--room-name", required=True)
    args = parser.parse_args()
    
    generate_gt_rirs(args.job_id, args.room_name)