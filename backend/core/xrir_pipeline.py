"""
xRIR 파이프라인 통합 모듈

roomplan JSON + ref_rir.wav (+ 선택: mesh.bin)
→ listener.npy, xyzs.npy, depth.npy 생성
→ xRIR 추론
→ 최적 스피커 위치 반환
"""

import sys
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

sys.path.append("/home/piai/AcousticRooms/xRIR_code-main")

import torch
from model.xRIR import xRIR
from inference import (
    convert_equirect_to_camera_coord,
    predict_rir,
    score_position,
)
from core.roomplan_to_numpy import convert_roomplan_to_xrir_inputs
from core.roomplan_to_depth import convert_roomplan_to_depth


CHECKPOINT_PATH = "/home/piai/AcousticRooms/xRIR_code-main/checkpoints/xRIR_unseen.pth"

_model = None


def _get_model(device="cuda"):
    """모델 싱글톤 (한 번만 로드)"""
    global _model
    if _model is None:
        m = xRIR(num_channels=1)
        m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        m.to(device)
        m.eval()
        _model = m
        print("xRIR 모델 로드 완료")
    return _model


def run_xrir_pipeline(
    roomplan_json: dict,
    ref_rir_bytes: bytes,
    top_k: int = 5,
    listener_height: float = 1.2,
    speaker_height: float = 1.2,
    grid_step: float = 0.3,
    wall_margin: float = 0.5,
    mesh_bin_path: str = None,   # ← 추가: mesh.bin 경로 (없으면 roomplan JSON fallback)
) -> list:
    """
    메인 파이프라인 함수

    Args:
        roomplan_json  : POST /api/xrir/speakers의 roomplan_scan dict
        ref_rir_bytes  : ref_rir.wav 파일의 bytes (deconvolution 결과)
        top_k          : 상위 몇 개 위치 반환
        listener_height: 청취자 귀 높이 (m)
        speaker_height : 후보 스피커 높이 (m)
        grid_step      : 후보 격자 간격 (m)
        wall_margin    : 벽 마진 (m)
        mesh_bin_path  : LiDAR mesh.bin 경로 (None이면 roomplan JSON fallback)

    Returns:
        상위 K개 위치와 점수 리스트
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── 1. ref_rir.wav 저장 ──────────────────────────────────
        ref_rir_path = tmpdir / "ref_rir.wav"
        ref_rir_path.write_bytes(ref_rir_bytes)
        ref_rir_np, sr = sf.read(str(ref_rir_path))
        ref_rir_np = ref_rir_np.astype(np.float32)

        # ── 2. roomplan JSON → listener.npy, xyzs.npy ───────────
        listener, xyzs = convert_roomplan_to_xrir_inputs(
            roomplan_json=roomplan_json,
            output_dir=str(tmpdir),
            listener_height=listener_height,
            speaker_height=speaker_height,
            grid_step=grid_step,
            wall_margin=wall_margin,
        )

        # ── 3. depth.npy 생성 ────────────────────────────────────
        # mesh.bin 있으면 LiDAR 메쉬 사용 (정확)
        # 없으면 roomplan JSON ray casting (fallback)
        depth_np = convert_roomplan_to_depth(
            roomplan_json=roomplan_json,
            listener_pos=listener,
            output_dir=str(tmpdir),
            mesh_bin_path=mesh_bin_path,
        )

        # ── 4. xRIR 추론 ─────────────────────────────────────────
        depth_tensor = torch.from_numpy(depth_np.astype(np.float32))
        depth_coord  = convert_equirect_to_camera_coord(depth_tensor)

        listener_pos = listener.astype(np.float32)
        ref_src_pos  = listener_pos.copy()

        model = _get_model(str(device))

        results = []
        print(f"후보 위치: {len(xyzs)}개 음향 점수 예측 중...")

        for i, candidate_pos in enumerate(xyzs):
            pred_rir = predict_rir(
                model, depth_coord, ref_rir_np,
                ref_src_pos, candidate_pos, listener_pos,
                device=str(device),
            )
            scores = score_position(pred_rir, sr=sr)
            scores["position"] = {
                "x": round(float(candidate_pos[0]), 3),
                "y": round(float(candidate_pos[1]), 3),
                "z": round(float(candidate_pos[2]), 3),
            }
            results.append(scores)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(xyzs)} 완료...")

        # ── 5. 상위 K개 정렬 ─────────────────────────────────────
        results.sort(key=lambda x: x["total"], reverse=True)
        top_results = results[:top_k]

        for rank, r in enumerate(top_results, 1):
            pos = r["position"]
            print(f"{rank}위: ({pos['x']}, {pos['y']}, {pos['z']}m) | 총점: {r['total']}")

        return top_results
