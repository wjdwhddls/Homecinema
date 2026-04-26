"""
실험 3: 후처리 가중치 최적화
- 수집한 GT RIR로 정답 Top-2 결정
- xRIR 예측 RIR로 grid search 가중치별 Top-2 정확도 평가
"""

import sys
sys.path.append("/home/piai/AcousticRooms/xRIR_code-main")
sys.path.append("/home/piai/Homecinema/backend")

import numpy as np
import soundfile as sf
import json
import time
from pathlib import Path

from inference import compute_edt, compute_c50, compute_t60


# ── 점수 계산 ──────────────────────────────────────────────────

def calc_score(rir, sr, w1, w2, w3, edt_target=0.25, t60_target=0.3):
    edt = compute_edt(rir, sr)
    c50 = compute_c50(rir, sr)
    t60 = compute_t60(rir, sr)

    edt_score = 1.0 - min(abs(edt - edt_target) / edt_target, 1.0)
    c50_score = 1.0 - min(abs(c50 - 4.0) / 8.0, 1.0)
    t60_score = 1.0 - min(abs(t60 - t60_target) / t60_target, 1.0)

    return edt_score * w1 + c50_score * w2 + t60_score * w3


def calc_pair_score(rir_L, rir_R, sr, w1, w2, w3):
    score_L = calc_score(rir_L, sr, w1, w2, w3)
    score_R = calc_score(rir_R, sr, w1, w2, w3)
    return (score_L + score_R) / 2


# ── 데이터 로드 ────────────────────────────────────────────────

def load_room_data(room_dir):
    """
    room_dir/
    ├── candidates.json
    ├── gt_rirs/   pair_0_L.wav, pair_0_R.wav, ...
    └── pred_rirs/ pair_0_L.wav, pair_0_R.wav, ...
    """
    room_dir = Path(room_dir)
    
    with open(room_dir / "candidates.json", "r") as f:
        data = json.load(f)
    candidates = data["candidates"]
    
    gt_rirs = {}
    pred_rirs = {}
    
    for c in candidates:
        i = c["id"]
        gt_L, sr = sf.read(str(room_dir / "gt_rirs" / f"pair_{i}_L.wav"))
        gt_R, _  = sf.read(str(room_dir / "gt_rirs" / f"pair_{i}_R.wav"))
        gt_rirs[i] = (gt_L.astype(np.float32), gt_R.astype(np.float32))
        
        pred_L, _ = sf.read(str(room_dir / "pred_rirs" / f"pair_{i}_L.wav"))
        pred_R, _ = sf.read(str(room_dir / "pred_rirs" / f"pair_{i}_R.wav"))
        pred_rirs[i] = (pred_L.astype(np.float32), pred_R.astype(np.float32))
    
    return candidates, gt_rirs, pred_rirs, sr


# ── Top-2 정확도 ────────────────────────────────────────────────

def evaluate_accuracy(rooms_data, w1, w2, w3):
    """정답 1위가 예측 Top-2에 있으면 정확"""
    correct = 0
    total = 0
    
    for room_name, (candidates, gt_rirs, pred_rirs, sr) in rooms_data.items():
        # 정답 1위 (GT 기준)
        gt_scores = []
        for c in candidates:
            i = c["id"]
            gt_L, gt_R = gt_rirs[i]
            score = calc_pair_score(gt_L, gt_R, sr, w1, w2, w3)
            gt_scores.append((i, score))
        gt_scores.sort(key=lambda x: x[1], reverse=True)
        gt_top1 = gt_scores[0][0]
        
        # 예측 Top-2 (xRIR 기준)
        pred_scores = []
        for c in candidates:
            i = c["id"]
            pred_L, pred_R = pred_rirs[i]
            score = calc_pair_score(pred_L, pred_R, sr, w1, w2, w3)
            pred_scores.append((i, score))
        pred_scores.sort(key=lambda x: x[1], reverse=True)
        pred_top2 = [pred_scores[0][0], pred_scores[1][0]]
        
        # 정답 1위가 예측 Top-2에 있으면 맞음
        if gt_top1 in pred_top2:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


# ── Grid Search ────────────────────────────────────────────────

def grid_search(rooms_data, step=0.1):
    """w1 + w2 + w3 = 1, step 단위 탐색"""
    weights = np.round(np.arange(0.1, 1.0, step), 2)
    
    results = []
    best_acc = 0.0
    best_weights = None
    
    print(f"Grid search 시작 (step={step})...")
    start = time.time()
    
    for w1 in weights:
        for w2 in weights:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.1 or w3 > 0.9:
                continue
            
            acc = evaluate_accuracy(rooms_data, w1, w2, w3)
            results.append({
                "w1_edt": float(w1),
                "w2_c50": float(w2),
                "w3_t60": float(w3),
                "accuracy": float(acc),
            })
            
            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2, w3)
                print(f"  새 최고: EDT={w1}, C50={w2}, T60={w3} → acc={acc:.3f}")
    
    elapsed = time.time() - start
    print(f"\n완료 ({elapsed:.1f}초)")
    print(f"최적: EDT={best_weights[0]}, C50={best_weights[1]}, T60={best_weights[2]}")
    print(f"Top-2 정확도: {best_acc:.3f}")
    
    return best_weights, best_acc, results


# ── 메인 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rooms", 
        nargs="+", 
        required=True,
        help="실험 방 폴더 이름들 (공백으로 구분)"
    )
    args = parser.parse_args()
    
    EXPERIMENT_DIR = Path("/home/piai/Homecinema/backend/data/experiment")
    room_dirs = [str(EXPERIMENT_DIR / r) for r in args.rooms]
    
    rooms_data = {}
    for room_dir in room_dirs:
        room_name = Path(room_dir).name
        print(f"로드: {room_name}")
        rooms_data[room_name] = load_room_data(room_dir)
    
    print(f"\n{len(rooms_data)}개 방 로드 완료")
    print(f"방별 후보 수: {[len(rooms_data[r][0]) for r in rooms_data]}\n")
    
    best_weights, best_acc, all_results = grid_search(rooms_data, step=0.1)
    
    output_path = EXPERIMENT_DIR / "grid_search_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_weights": {
                "edt": float(best_weights[0]),
                "c50": float(best_weights[1]),
                "t60": float(best_weights[2]),
            },
            "best_accuracy": float(best_acc),
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {output_path}")