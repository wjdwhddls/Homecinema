"""
실험용: candidates.json 기반으로 xRIR 예측 RIR 생성 후 wav 저장
- check_candidates.py로 candidates.json 만든 후 실행
- ref_rir.wav (sweep 측정 결과)는 미리 준비되어 있어야 함
"""

import sys
sys.path.append("/home/piai/AcousticRooms/xRIR_code-main")
sys.path.append("/home/piai/Homecinema/backend")

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import numpy as np
import soundfile as sf
import json
from pathlib import Path

from inference import (
    convert_equirect_to_camera_coord,
    load_model,
    predict_rir,
)
from core.sweep_deconvolution import deconvolve_sweep


SCAN_DIR = Path("/home/piai/Homecinema/backend/data/roomplan_scans")
EXPERIMENT_DIR = Path("/home/piai/Homecinema/backend/data/experiment")
CHECKPOINT_PATH = "/home/piai/AcousticRooms/xRIR_code-main/checkpoints/xRIR_unseen.pth"


def generate_pred_rirs(job_id, room_name):
    """
    Args:
        job_id: scan_{job_id}.json의 job_id
        room_name: experiment 폴더의 방 이름
    """
    room_dir = EXPERIMENT_DIR / room_name
    
    # candidates.json 로드
    with open(room_dir / "candidates.json") as f:
        data = json.load(f)
    candidates = data["candidates"]
    listener_pos = np.array([
        data["listener"]["x"],
        data["listener"]["y"],
        data["listener"]["z"],
    ], dtype=np.float32)
    
    ref_src_pos = np.array([
        data["initial_speaker"]["x"],
        data["initial_speaker"]["y"],
        data["initial_speaker"]["z"],
    ], dtype=np.float32)


    # depth.npy 로드
    depth_np = np.load(room_dir / "depth.npy").astype(np.float32)
    depth_tensor = torch.from_numpy(depth_np)
    depth_coord = convert_equirect_to_camera_coord(depth_tensor)
    
    # ref_rir.wav 준비 (sweep deconvolution으로 생성)
    ref_rir_path = room_dir / "ref_rir.wav"
    if not ref_rir_path.exists():
        print("ref_rir.wav 생성 중 (sweep deconvolution)...")
        recorded_path = SCAN_DIR / job_id / "recorded.wav"
        sweep_path = SCAN_DIR / job_id / "sweep.wav"
        deconvolve_sweep(
            recorded_path=str(recorded_path),
            sweep_path=str(sweep_path),
            output_path=str(ref_rir_path),
        )
    
    ref_rir_np, sr = sf.read(str(ref_rir_path))
    ref_rir_np = ref_rir_np.astype(np.float32)
    
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, num_channels=1, device=str(device))
    
    # pred_rirs 폴더 생성
    pred_dir = room_dir / "pred_rirs"
    pred_dir.mkdir(exist_ok=True)
    
    # 각 후보 쌍의 L/R 예측
    print(f"{len(candidates)}쌍 예측 중...")
    for c in candidates:
        i = c["id"]
        left  = np.array([c["left"]["x"],  c["left"]["y"],  c["left"]["z"]],  dtype=np.float32)
        right = np.array([c["right"]["x"], c["right"]["y"], c["right"]["z"]], dtype=np.float32)
        
        rir_L = predict_rir(model, depth_coord, ref_rir_np, ref_src_pos, left, listener_pos, device=str(device))
        rir_R = predict_rir(model, depth_coord, ref_rir_np, ref_src_pos, right, listener_pos, device=str(device))
        
        sf.write(str(pred_dir / f"pair_{i}_L.wav"), rir_L, sr)
        sf.write(str(pred_dir / f"pair_{i}_R.wav"), rir_R, sr)
        
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(candidates)} 완료")
    
    print(f"\n예측 RIR 저장 완료: {pred_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--room-name", required=True)
    args = parser.parse_args()
    
    generate_pred_rirs(args.job_id, args.room_name)