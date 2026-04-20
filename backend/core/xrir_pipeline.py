"""
xRIR 파이프라인 통합 모듈

roomplan JSON + ref_rir.wav (+ 선택: mesh.bin)
→ listener.npy, xyzs.npy, depth.npy 생성
→ xRIR 추론
→ 최적 스피커 위치 반환 (스테레오 left/right 포함)
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


# ── 스테레오 배치 계산 ────────────────────────────────────────────

def compute_stereo_placement(
    center_pos: np.ndarray,
    listener_pos: np.ndarray,
    stereo_offset_m: float = 1.0,
) -> dict:
    """
    최적 단일 위치(center_pos)를 기준으로 left/right 스테레오 배치 계산.

    청취자-스피커 방향 벡터에 수직인 방향으로 ±offset만큼 이동.

    Args:
        center_pos     : xRIR이 찾은 최적 단일 스피커 위치 [x, y, z]
        listener_pos   : 청취자 위치 [x, y, z]
        stereo_offset_m: 센터에서 left/right까지 거리 (기본 1.0m)

    Returns:
        { "left": {x,y,z}, "center": {x,y,z}, "right": {x,y,z} }
    """
    # 수평면(x-y)에서만 계산
    spk_xy = center_pos[:2]
    lis_xy = listener_pos[:2]

    # 청취자 → 스피커 방향 벡터
    forward = spk_xy - lis_xy
    dist = np.linalg.norm(forward)

    if dist < 1e-6:
        # 청취자와 스피커가 같은 위치이면 임의 방향 사용
        forward = np.array([1.0, 0.0])
    else:
        forward = forward / dist

    # 수직 벡터 (왼쪽: 반시계방향 90도)
    perp = np.array([-forward[1], forward[0]])

    z = float(center_pos[2])
    left_xy  = spk_xy + perp * stereo_offset_m
    right_xy = spk_xy - perp * stereo_offset_m

    return {
        "left": {
            "x": round(float(left_xy[0]), 3),
            "y": round(float(left_xy[1]), 3),
            "z": z,
        },
        "center": {
            "x": round(float(spk_xy[0]), 3),
            "y": round(float(spk_xy[1]), 3),
            "z": z,
        },
        "right": {
            "x": round(float(right_xy[0]), 3),
            "y": round(float(right_xy[1]), 3),
            "z": z,
        },
    }


def run_xrir_pipeline(
    roomplan_json: dict,
    ref_rir_bytes: bytes,
    top_k: int = 5,
    listener_height: float = 1.2,
    speaker_height: float = 1.2,
    grid_step: float = 0.3,
    wall_margin: float = 0.5,
    stereo_offset_m: float = 1.0,
    mesh_bin_path: str = None,
) -> list:
    """
    메인 파이프라인 함수

    Returns:
        상위 K개 결과 리스트. 각 항목:
        {
            "placement": {
                "left":     {x, y, z},
                "center":   {x, y, z},   ← xRIR 최적 위치
                "right":    {x, y, z},
                "listener": {x, y, z},
            },
            "score": float,               ← xRIR total score
            "metrics": {
                "rt60_seconds": float,
                "c80_db": float,
                "drr_db": float,
                "rt60_score": float,
                "c80_score": float,
                "drr_score": float,
            },
            "rank": int,
        }
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

        raw_results = []
        print(f"후보 위치: {len(xyzs)}개 음향 점수 예측 중...")

        for i, candidate_pos in enumerate(xyzs):
            pred_rir = predict_rir(
                model, depth_coord, ref_rir_np,
                ref_src_pos, candidate_pos, listener_pos,
                device=str(device),
            )
            scores = score_position(pred_rir, sr=sr)
            scores["_candidate_pos"] = candidate_pos
            raw_results.append(scores)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(xyzs)} 완료...")

        # ── 5. 정렬 후 상위 K개를 앱 형식으로 변환 ───────────────
        raw_results.sort(key=lambda x: x["total"], reverse=True)
        top_raw = raw_results[:top_k]

        results = []
        for rank, r in enumerate(top_raw):
            candidate_pos = r["_candidate_pos"]
            stereo = compute_stereo_placement(
                center_pos=candidate_pos,
                listener_pos=listener_pos,
                stereo_offset_m=stereo_offset_m,
            )

            result = {
                "placement": {
                    "left":     stereo["left"],
                    "right":    stereo["right"],
                    "listener": {
                        "x": round(float(listener_pos[0]), 3),
                        "y": round(float(listener_pos[1]), 3),
                        "z": round(float(listener_pos[2]), 3),
                    },
                },
                "score": round(float(r["total"]), 4),
                "metrics": {
                    "rt60_seconds":          round(float(r.get("rt60", 0.0)), 3),
                    "c80_db":                round(float(r.get("c80", 0.0)), 2),
                    "drr_db":                round(float(r.get("drr", 0.0)), 2),
                    "rt60_score":            round(float(r.get("rt60_score", 0.0)), 3),
                    "c80_score":             round(float(r.get("c80_score", 0.0)), 3),
                    "drr_score":             round(float(r.get("drr_score", 0.0)), 3),
                },
                "rank": rank,
            }
            results.append(result)

            pos = stereo["center"]
            print(f"{rank+1}위: center({pos['x']}, {pos['y']}, {pos['z']}) "
                  f"| score={r['total']:.3f}")

        return results
