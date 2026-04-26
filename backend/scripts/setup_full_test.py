"""
전체 파이프라인 테스트용 더미 데이터 생성 + 실행
"""

import sys
sys.path.append("/home/piai/Homecinema/backend")

import os
import numpy as np
import soundfile as sf
import json
import subprocess
from pathlib import Path
from scipy.signal import chirp


SCAN_DIR = Path("/home/piai/Homecinema/backend/data/roomplan_scans")
EXPERIMENT_DIR = Path("/home/piai/Homecinema/backend/data/experiment")
FAKE_RECORDINGS_DIR = Path("/home/piai/Homecinema/backend/data/fake_recordings")


def create_fake_sweep(duration=3.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    sweep = chirp(t, f0=20, f1=20000, t1=duration, method='logarithmic').astype(np.float32)
    return sweep, sr


def create_fake_rir(sr=22050, length_sec=0.5, decay=8.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = int(sr * length_sec)
    noise = np.random.randn(n).astype(np.float32)
    envelope = np.exp(-decay * np.arange(n) / sr)
    rir = noise * envelope
    rir[0] = 1.0
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir


def convolve_sweep_with_rir(sweep, rir):
    convolved = np.convolve(sweep, rir, mode='full')
    convolved = convolved / (np.max(np.abs(convolved)) + 1e-8) * 0.9
    return convolved.astype(np.float32)


def create_fake_roomplan(room_name, room_size=(6.0, 5.0, 2.7)):
    w, d, h = room_size
    walls = [
        {"id": "w1", "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0, 0, d/2, 1], "dimensions": [w, h, 0.2]},
        {"id": "w2", "transform": [-1,0,0,0, 0,1,0,0, 0,0,-1,0, 0, 0, -d/2, 1], "dimensions": [w, h, 0.2]},
        {"id": "w3", "transform": [0,0,-1,0, 0,1,0,0, 1,0,0,0, -w/2, 0, 0, 1], "dimensions": [d, h, 0.2]},
        {"id": "w4", "transform": [0,0,1,0, 0,1,0,0, -1,0,0,0, w/2, 0, 0, 1], "dimensions": [d, h, 0.2]},
    ]
    return {
        "walls": walls,
        "objects": [],
        "doors": [],
        "windows": [],
        "openings": [],
        "scannedAt": "2026-04-25T12:00:00Z",
    }


def setup_test_room(job_id, room_name, room_size=(6.0, 5.0, 2.7), n_pairs=20):
    print(f"\n=== {room_name} 셋업 시작 ===")
    
    # 1. roomplan JSON
    roomplan = create_fake_roomplan(room_name, room_size)
    scan_path = SCAN_DIR / f"scan_{job_id}.json"
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    with open(scan_path, "w") as f:
        json.dump(roomplan, f, indent=2)
    print(f"  roomplan JSON: {scan_path}")
    
    # 2. sweep.wav, recorded.wav
    sweep, sr = create_fake_sweep()
    rir_initial = create_fake_rir(sr, decay=10.0, seed=42)
    recorded = convolve_sweep_with_rir(sweep, rir_initial)
    
    job_dir = SCAN_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    sf.write(str(job_dir / "sweep.wav"), sweep, sr)
    sf.write(str(job_dir / "recorded.wav"), recorded, sr)
    print(f"  sweep + recorded: {job_dir}")
    
    # 3. 가짜 측정 녹음 (n_pairs × 2 = 2*n_pairs개)
    fake_rec_dir = FAKE_RECORDINGS_DIR / room_name
    fake_rec_dir.mkdir(parents=True, exist_ok=True)
    
    for pair_id in range(n_pairs):
        for j, side in enumerate(["L", "R"]):
            recording_idx = pair_id * 2 + j + 1
            
            rir = create_fake_rir(sr, decay=8.0 + pair_id * 0.3, seed=pair_id * 100 + j)
            recorded = convolve_sweep_with_rir(sweep, rir)
            
            wav_path = fake_rec_dir / f"recording_{recording_idx:03d}.wav"
            sf.write(str(wav_path), recorded, sr)
            
            m4a_path = fake_rec_dir / f"recording_{recording_idx:03d}.m4a"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(wav_path),
                "-c:a", "aac", "-b:a", "128k",
                str(m4a_path)
            ], check=True, capture_output=True)
            
            wav_path.unlink()
    
    print(f"  가짜 측정 녹음: {fake_rec_dir} ({n_pairs*2}개 m4a)")


def run_pipeline(job_id, room_name):
    """파이프라인 4단계 자동 실행"""
    scripts_dir = Path("/home/piai/Homecinema/backend/scripts")
    
    print(f"\n=== {room_name} 파이프라인 실행 ===")
    
    env = os.environ.copy()
    env['MKL_SERVICE_FORCE_INTEL'] = '1'

    commands = [
        ["python", str(scripts_dir / "check_candidates.py"),
         "--job-id", job_id, "--room-name", room_name],
        
        ["python", str(scripts_dir / "convert_recordings.py"),
         "--input", str(FAKE_RECORDINGS_DIR / room_name),
         "--room-name", room_name],
        
        ["python", str(scripts_dir / "generate_gt_rirs.py"),
         "--job-id", job_id, "--room-name", room_name],
        
        ["python", str(scripts_dir / "generate_pred_rirs.py"),
         "--job-id", job_id, "--room-name", room_name],
    ]
    
    for cmd in commands:
        print(f"\n>>> {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd="/home/piai/Homecinema/backend", env=env)
        if result.returncode != 0:
            print(f"!! 실패: {cmd}")
            return False
    
    return True


def main():
    test_rooms = [
        ("test-room-1", "test_room_1", (6.0, 5.0, 2.7), 20),
        ("test-room-2", "test_room_2", (5.0, 4.0, 2.5), 20),
        ("test-room-3", "test_room_3", (7.0, 6.0, 3.0), 20),
    ]
    
    # 1. 가짜 데이터 생성
    print("=" * 60)
    print("STEP 1: 가짜 데이터 생성")
    print("=" * 60)
    for job_id, room_name, room_size, n_pairs in test_rooms:
        setup_test_room(job_id, room_name, room_size, n_pairs)
    
    # 2. 각 방 파이프라인 실행
    print("\n" + "=" * 60)
    print("STEP 2: 파이프라인 실행")
    print("=" * 60)
    for job_id, room_name, _, _ in test_rooms:
        success = run_pipeline(job_id, room_name)
        if not success:
            print(f"{room_name} 실패. 중단합니다.")
            return
    
    # 3. grid search
    print("\n" + "=" * 60)
    print("STEP 3: Grid Search")
    print("=" * 60)
    subprocess.run([
        "python", "/home/piai/Homecinema/backend/scripts/grid_search.py",
        "--rooms", "test_room_1", "test_room_2", "test_room_3"
    ], cwd="/home/piai/Homecinema/backend")
    
    print("\n" + "=" * 60)
    print("전체 파이프라인 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()