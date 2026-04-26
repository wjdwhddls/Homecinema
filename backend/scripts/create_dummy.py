"""
실험용 더미 데이터 생성
- candidates.json은 check_candidates.py로 이미 만들어졌다고 가정
- gt_rirs, pred_rirs 폴더에 더미 wav 파일 생성
"""

import sys
sys.path.append("/home/piai/AcousticRooms/xRIR_code-main")

import numpy as np
import soundfile as sf
import json
from pathlib import Path


def generate_dummy_rir(sr=22050, length_sec=0.5, decay=8.0, seed=None):
    """랜덤 노이즈 + 지수 감쇠로 더미 RIR 생성"""
    if seed is not None:
        np.random.seed(seed)
    
    n = int(sr * length_sec)
    noise = np.random.randn(n).astype(np.float32)
    envelope = np.exp(-decay * np.arange(n) / sr)
    rir = noise * envelope
    
    # 직접음 임펄스 추가
    rir[0] = 1.0
    
    # 정규화
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir


def create_dummy_data(room_dir, n_pairs=10):
    """
    가짜 candidates.json + gt_rirs + pred_rirs 생성
    """
    room_dir = Path(room_dir)
    (room_dir / "gt_rirs").mkdir(parents=True, exist_ok=True)
    (room_dir / "pred_rirs").mkdir(parents=True, exist_ok=True)
    
    # 가짜 candidates.json
    candidates = []
    distances = [0.1, 0.2, 0.3, 0.5]
    angles = [40, 50, 60, 70, 80]
    idx = 0
    for d in distances:
        for a in angles:
            if idx >= n_pairs:
                break
            candidates.append({
                "id": idx,
                "left":  {"x": -1.0 + idx*0.1, "y": 2.0, "z": 1.2},
                "right": {"x":  1.0 - idx*0.1, "y": 2.0, "z": 1.2},
                "d_m": d,
                "angle_deg": a,
            })
            idx += 1
        if idx >= n_pairs:
            break
    
    with open(room_dir / "candidates.json", "w") as f:
        json.dump({
            "listener": {"x": 0.0, "y": 0.0, "z": 1.2},
            "initial_speaker": {"x": 0.0, "y": 2.0, "z": 1.2},
            "candidates": candidates,
        }, f, indent=2, ensure_ascii=False)
    
    # 각 쌍에 대해 GT와 예측 RIR 생성
    for c in candidates:
        i = c["id"]
        
        # GT: 위치별로 다른 특성
        gt_L = generate_dummy_rir(decay=8.0 + i * 0.3, seed=i*100)
        gt_R = generate_dummy_rir(decay=8.0 + i * 0.3, seed=i*100 + 1)
        sf.write(str(room_dir / "gt_rirs" / f"pair_{i}_L.wav"), gt_L, 22050)
        sf.write(str(room_dir / "gt_rirs" / f"pair_{i}_R.wav"), gt_R, 22050)
        
        # 예측: GT랑 살짝 다르게 (모델 오차 시뮬레이션)
        pred_L = generate_dummy_rir(decay=8.0 + i * 0.3 + np.random.uniform(-0.5, 0.5), seed=i*200)
        pred_R = generate_dummy_rir(decay=8.0 + i * 0.3 + np.random.uniform(-0.5, 0.5), seed=i*200 + 1)
        sf.write(str(room_dir / "pred_rirs" / f"pair_{i}_L.wav"), pred_L, 22050)
        sf.write(str(room_dir / "pred_rirs" / f"pair_{i}_R.wav"), pred_R, 22050)
    
    print(f"{room_dir.name}: {len(candidates)}쌍 더미 데이터 생성 완료")


if __name__ == "__main__":
    base = Path("/home/piai/Homecinema/backend/data/experiment")
    
    create_dummy_data(base / "international_hall", n_pairs=12)
    create_dummy_data(base / "4f_lounge_1", n_pairs=10)
    create_dummy_data(base / "4f_lounge_2", n_pairs=15)
    
    print("\n전체 더미 데이터 생성 완료!")
    print("이제 grid_search.py 실행 가능")