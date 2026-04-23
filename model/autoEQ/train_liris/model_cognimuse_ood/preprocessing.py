"""COGNIMUSE Emotion Annotation → LIRIS-compatible metadata CSV.

입력:
    dataset/autoEQ/cognimuse/Emotion Annotation/experienced/<subj{S}_{I}_{MOVIE}.dat>
        - 각 파일: `<time_sec> <valence> <arousal>` 한 행, 40ms 간격 (25Hz)
        - 값 범위: V, A ∈ [-1, +1]
        - 영화당 12 traces (7 subjects × 1~2 iterations)

처리 파이프라인:
    1. 영화별 12 traces 를 공통 시간축에 정렬 (최소 길이 기준 truncate)
    2. Per-frame median across 12 annotators → (T, 2) robust consensus
    3. 10s non-overlap window 로 window mean aggregation → (N_window, 2)
       (LIRIS audio_pad_to_sec=10.0 과 정합, 영상 뒷꼬리 <10s 는 drop)
    4. v_var, a_var: window 내 time-axis std + annotator-axis std 결합
    5. mood_k7, quadrant_k4: BASE va_to_mood / va_to_quadrant_k4 재사용

출력:
    dataset/autoEQ/cognimuse/cognimuse_metadata.csv
    dataset/autoEQ/cognimuse/window_name_map.json
       (LIRIS PrecomputedLirisDataset 호환: name, film_id, split, v_raw, a_raw,
        v_var, a_var, v_norm, a_norm, mood_k7, quadrant_k4, start_sec, end_sec, ...)

중요 설계 결정:
    - Annotation 채널: experienced primary (README: 관객 실제 감정, 12 traces)
    - VA norm: IDENTITY (COGNIMUSE 값이 이미 [-1,+1], Strategy A 미적용)
    - split: "test" 전체 (Phase 4-A 는 OOD eval only)
    - film_id: 1~12 (영화 이름 정렬 순)
    - movie_code: 3자 약어 (ABE, BMI, CHI, CRA, FNE, GLA, LOR, MDB, NCO, RAT, SIL, DEP)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.autoEQ.train_liris.dataset import va_to_quadrant_k4
from model.autoEQ.train.dataset import va_to_mood

WINDOW_SEC = 10.0          # LIRIS audio_pad_to_sec 정합
FRAME_DT = 0.04            # 40ms = 25Hz
EXPECTED_ANNOTATORS = 12   # 영화당 experienced traces
ANNOTATION_SOURCE = "experienced_median"

# Annotation 파일의 영화 이름 → COGNIMUSE Data 디렉토리 폴더명
# (Emotion Annotation 폴더명 ↔ Data/Data-NotOfficial 폴더명)
FILM_MAP: dict[str, dict[str, str]] = {
    "AmericanBeauty":    {"movie_code": "ABE", "video_folder": "2000ABE", "video_basename": "ABE"},
    "BeautifulMind":     {"movie_code": "BMI", "video_folder": "2002BMI", "video_basename": "BMI"},
    "Chicago":           {"movie_code": "CHI", "video_folder": "2003CHI", "video_basename": "CHI"},
    "Crash":             {"movie_code": "CRA", "video_folder": "2006CRA", "video_basename": "CRA"},
    "FindingNemo":       {"movie_code": "FNE", "video_folder": "2004FNE", "video_basename": "FNE"},
    "Gladiator":         {"movie_code": "GLA", "video_folder": "2001GLA", "video_basename": "GLA"},
    "LOTRReturn":        {"movie_code": "LOR", "video_folder": "2004LOR", "video_basename": "LOR"},
    "MDB":               {"movie_code": "MDB", "video_folder": "2005MDB", "video_basename": "MDB"},
    "NoCountry":         {"movie_code": "NCO", "video_folder": "2008NCO", "video_basename": "NCO"},
    "Ratatouille":       {"movie_code": "RAT", "video_folder": "2008RAT", "video_basename": "RAT"},
    "ShakespeareInLove": {"movie_code": "SIL", "video_folder": "1999SIL", "video_basename": "SIL"},
    "TheDeparted":       {"movie_code": "DEP", "video_folder": "2007DEP", "video_basename": "DEP"},
}


def load_trace(dat_path: Path) -> np.ndarray:
    """Parse one .dat file. Returns (T, 2) array with columns [v, a].

    Rows with malformed lines are skipped. NaN/inf rejected.
    """
    rows: list[list[float]] = []
    with dat_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                v = float(parts[1])
                a = float(parts[2])
            except ValueError:
                continue
            if not (np.isfinite(v) and np.isfinite(a)):
                continue
            rows.append([t, v, a])
    if not rows:
        raise ValueError(f"empty or invalid .dat: {dat_path}")
    arr = np.array(rows, dtype=np.float64)
    # Enforce monotone time & 40ms assumed — sort just in case
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr


def load_film_traces(
    emotion_dir: Path, film_name: str, expected_n: int = EXPECTED_ANNOTATORS
) -> tuple[np.ndarray, list[str]]:
    """Load all experienced traces for one film.

    Returns:
        traces: (T_min, 2, N) stacked after time-align truncation to common length.
        filenames: list of trace filenames (for provenance).

    Each trace row is [v, a]. Time column is dropped after verifying uniform dt.
    """
    files = sorted(emotion_dir.glob(f"subj*_{film_name}.dat"))
    if len(files) != expected_n:
        raise ValueError(
            f"{film_name}: expected {expected_n} traces, found {len(files)} "
            f"({[f.name for f in files]})"
        )
    raw_traces: list[np.ndarray] = []
    for f in files:
        arr = load_trace(f)  # (T_i, 3) = [t, v, a]
        # Verify uniform dt ≈ 0.04
        dt = float(np.median(np.diff(arr[:, 0])))
        if abs(dt - FRAME_DT) > 1e-3:
            raise ValueError(
                f"{f.name}: median dt={dt:.4f}s, expected {FRAME_DT}s (25Hz)"
            )
        raw_traces.append(arr)
    min_len = min(tr.shape[0] for tr in raw_traces)
    # Truncate to shortest
    stacked = np.stack([tr[:min_len, 1:3] for tr in raw_traces], axis=-1)  # (T, 2, N)
    return stacked, [f.name for f in files]


def median_consensus(traces: np.ndarray) -> np.ndarray:
    """Per-frame median across N annotators → (T, 2).

    Median is robust to outlier annotators (Kollias Aff-Wild2 practice).
    """
    # traces: (T, 2, N)
    return np.median(traces, axis=-1)  # (T, 2)


def window_aggregate(
    consensus: np.ndarray,
    traces: np.ndarray,
    window_sec: float = WINDOW_SEC,
    frame_dt: float = FRAME_DT,
) -> pd.DataFrame:
    """Split consensus time-series into non-overlapping windows of `window_sec`
    and compute per-window (v_mean, a_mean, v_var, a_var).

    v_var = std across (time × annotators) inside the window — 2D uncertainty.

    Returns DataFrame with columns: start_sec, end_sec, v_raw, a_raw, v_var, a_var.
    Tail windows shorter than `window_sec` are dropped (LIRIS 정합).
    """
    frames_per_window = int(round(window_sec / frame_dt))
    T = consensus.shape[0]
    n_windows = T // frames_per_window
    rows = []
    for w in range(n_windows):
        start_idx = w * frames_per_window
        end_idx = start_idx + frames_per_window
        c_win = consensus[start_idx:end_idx]  # (W, 2)
        t_win = traces[start_idx:end_idx]     # (W, 2, N) — time × VA × annotator
        v_mean = float(np.mean(c_win[:, 0]))
        a_mean = float(np.mean(c_win[:, 1]))
        # Combined uncertainty: std across time-and-annotator flatten
        v_var = float(np.std(t_win[:, 0, :].reshape(-1)))
        a_var = float(np.std(t_win[:, 1, :].reshape(-1)))
        rows.append(
            {
                "start_sec": start_idx * frame_dt,
                "end_sec": end_idx * frame_dt,
                "v_raw": v_mean,
                "a_raw": a_mean,
                "v_var": v_var,
                "a_var": a_var,
            }
        )
    return pd.DataFrame(rows)


def build_film_metadata(
    film_name: str,
    film_id: int,
    emotion_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Process one film → window-level metadata DataFrame + provenance.

    CSV columns follow LIRIS liris_metadata.csv format:
        name, film_id, movie_code, split, v_raw, a_raw, v_var, a_var,
        v_norm, a_norm, mood_k7, quadrant_k4,
        start_sec, end_sec, annotation_source, n_annotators
    """
    info = FILM_MAP[film_name]
    traces, trace_files = load_film_traces(emotion_dir, film_name)
    consensus = median_consensus(traces)
    df = window_aggregate(consensus, traces)

    df["film_name"] = film_name
    df["film_id"] = film_id
    df["movie_code"] = info["movie_code"]
    df["video_folder"] = info["video_folder"]
    df["video_basename"] = info["video_basename"]
    df["split"] = "test"
    df["annotation_source"] = ANNOTATION_SOURCE
    df["n_annotators"] = EXPECTED_ANNOTATORS

    # VA norm strategy = IDENTITY (COGNIMUSE 원본이 이미 [-1,+1])
    df["v_norm"] = df["v_raw"].clip(-1.0, 1.0)
    df["a_norm"] = df["a_raw"].clip(-1.0, 1.0)

    df["mood_k7"] = df.apply(
        lambda r: va_to_mood(float(r["v_norm"]), float(r["a_norm"])), axis=1
    )
    df["quadrant_k4"] = df.apply(
        lambda r: va_to_quadrant_k4(float(r["v_norm"]), float(r["a_norm"])), axis=1
    )

    # window 이름: {movie_code}_{window_idx:05d}
    df = df.reset_index(drop=True)
    df["name"] = df.index.map(lambda i: f"{info['movie_code']}_{i:05d}")

    provenance = {
        "film_name": film_name,
        "film_id": film_id,
        "movie_code": info["movie_code"],
        "n_traces": len(trace_files),
        "trace_files": trace_files,
        "n_windows": len(df),
        "total_duration_sec": float(df["end_sec"].max()) if len(df) else 0.0,
        "v_mean": float(df["v_raw"].mean()) if len(df) else 0.0,
        "a_mean": float(df["a_raw"].mean()) if len(df) else 0.0,
    }
    return df, provenance


def build_all(
    cognimuse_root: Path,
    output_dir: Path,
) -> dict:
    """12 Hollywood 영화 전체 처리 → 단일 CSV + provenance JSON.

    Args:
        cognimuse_root: `dataset/autoEQ/cognimuse/` 디렉토리 (Emotion Annotation/ 포함)
        output_dir: CSV 저장 경로 (기본: cognimuse_root 동일)
    """
    emotion_dir = cognimuse_root / "Emotion Annotation" / "experienced"
    if not emotion_dir.is_dir():
        raise FileNotFoundError(f"experienced directory not found: {emotion_dir}")

    film_names = sorted(FILM_MAP.keys())  # 알파벳 순 film_id 할당
    all_dfs: list[pd.DataFrame] = []
    provenance: dict = {
        "window_sec": WINDOW_SEC,
        "frame_dt_sec": FRAME_DT,
        "n_annotators_expected": EXPECTED_ANNOTATORS,
        "annotation_source": ANNOTATION_SOURCE,
        "va_norm_strategy": "IDENTITY",
        "films": [],
    }
    for film_id, film_name in enumerate(film_names, start=1):
        df, prov = build_film_metadata(film_name, film_id, emotion_dir)
        all_dfs.append(df)
        provenance["films"].append(prov)
        print(
            f"[{film_id:>2}/12] {film_name:<20} → {len(df):>4} windows, "
            f"{prov['total_duration_sec']:>6.1f}s, "
            f"V={prov['v_mean']:+.3f} A={prov['a_mean']:+.3f}"
        )

    full = pd.concat(all_dfs, ignore_index=True)
    provenance["total_windows"] = len(full)
    provenance["total_films"] = len(film_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cognimuse_metadata.csv"
    cols = [
        "name", "film_id", "film_name", "movie_code", "video_folder",
        "video_basename", "split", "v_raw", "a_raw", "v_var", "a_var",
        "v_norm", "a_norm", "mood_k7", "quadrant_k4",
        "start_sec", "end_sec", "annotation_source", "n_annotators",
    ]
    full[cols].to_csv(csv_path, index=False)
    prov_path = output_dir / "window_name_map.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"\n[done] wrote {csv_path} ({len(full)} rows)")
    print(f"       wrote {prov_path}")
    return provenance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cognimuse-root",
        type=Path,
        default=Path("dataset/autoEQ/cognimuse"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/autoEQ/cognimuse"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    build_all(args.cognimuse_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
