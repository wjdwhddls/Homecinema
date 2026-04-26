# convert_recordings.py
"""
m4a 녹음 파일들을 wav로 변환 + pair_X_L/R 형식으로 이름 변경
"""

import sys
from pathlib import Path
import subprocess


def convert_recordings(input_dir, output_dir, candidates_json):
    """
    Args:
        input_dir: 폰에서 옮긴 m4a 파일들이 있는 폴더 (recording_001.m4a 순서대로)
        output_dir: 변환된 wav 저장 폴더 (recordings/)
        candidates_json: candidates.json 경로 (쌍 수 확인용)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 입력 파일 정렬
    m4a_files = sorted(input_dir.glob("*.m4a"))
    
    import json
    with open(candidates_json) as f:
        n_pairs = len(json.load(f)["candidates"])
    
    expected = n_pairs * 2  # L + R
    if len(m4a_files) != expected:
        print(f"경고: 예상 {expected}개, 실제 {len(m4a_files)}개")
    
    # 순서대로 매기기 (pair_0_L, pair_0_R, pair_1_L, pair_1_R, ...)
    for i, m4a in enumerate(m4a_files):
        pair_id = i // 2
        side = "L" if i % 2 == 0 else "R"
        output_path = output_dir / f"pair_{pair_id}_{side}.wav"
        
        # ffmpeg로 변환
        subprocess.run([
            "ffmpeg", "-y", "-i", str(m4a),
            "-ar", "22050",  # xRIR sample rate
            "-ac", "1",      # mono
            str(output_path)
        ], check=True, capture_output=True)
        
        print(f"{m4a.name} → {output_path.name}")
    
    print(f"\n완료: {len(m4a_files)}개 변환")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="m4a 파일들이 있는 폴더")
    parser.add_argument("--room-name", required=True)
    args = parser.parse_args()
    
    EXPERIMENT_DIR = Path("/home/piai/Homecinema/backend/data/experiment")
    room_dir = EXPERIMENT_DIR / args.room_name
    
    convert_recordings(
        input_dir=args.input,
        output_dir=room_dir / "recordings",
        candidates_json=room_dir / "candidates.json",
    )