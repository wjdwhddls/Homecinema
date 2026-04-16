"""audio_analyzer.py — PANNs 임베딩 + Silero VAD + dialogue density.

작업 4~6 (Day 3):
- PANNs Cnn14로 4초 윈도우 2048-d 임베딩 추출
- Silero VAD로 대사 구간 감지 (librosa로 직접 로드해 torchcodec 회피)
- 씬별 dialogue density 계산
"""

from __future__ import annotations

import json
import sys

import librosa
import numpy as np
import torch

from .paths import (
    PANNS_PYTORCH_DIR,
    PANNS_WEIGHTS,
    audio_features_json,
    audio_path,
    scenes_json,
    trailer_path,
    ensure_dirs,
)
from .utils import get_duration

# PANNs 코드 경로 추가 (디렉토리 없어도 sys.path.append 자체는 안전)
sys.path.append(str(PANNS_PYTORCH_DIR))


# ────────────────────────────────────────────────────────
# PANNs 임베딩
# ────────────────────────────────────────────────────────
_panns = None


def load_panns():
    """PANNs Cnn14 모델 싱글톤 로더."""
    global _panns
    if _panns is None:
        # PANNs 코드 디렉토리 사전 체크 (친화적 에러)
        if not PANNS_PYTORCH_DIR.exists():
            raise FileNotFoundError(
                f"PANNs 코드 디렉토리 없음: {PANNS_PYTORCH_DIR}\n"
                "다음 명령으로 클론하세요:\n"
                "  cd model/autoEQ/assets\n"
                "  git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git"
            )
        if not PANNS_WEIGHTS.exists():
            raise FileNotFoundError(
                f"PANNs 가중치 없음: {PANNS_WEIGHTS}\n"
                "Zenodo에서 다운로드 후 model/autoEQ/assets/에 두세요:\n"
                "  curl -L -o model/autoEQ/assets/Cnn14_mAP=0.431.pth \\\n"
                "    'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth'"
            )

        from models import Cnn14  # noqa: E402 (sys.path 추가 후 import)

        _panns = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        )
        checkpoint = torch.load(str(PANNS_WEIGHTS), map_location="cpu")
        _panns.load_state_dict(checkpoint["model"])
        _panns.eval()
    return _panns


def get_audio_embedding(audio_file, start_sec: float, end_sec: float) -> np.ndarray:
    """오디오 구간을 2048차원 벡터로 요약."""
    panns = load_panns()
    y, _ = librosa.load(
        str(audio_file), sr=32000, offset=start_sec, duration=end_sec - start_sec
    )
    waveform = torch.FloatTensor(y).unsqueeze(0)
    with torch.no_grad():
        output = panns(waveform)
    return output["embedding"].cpu().numpy().squeeze()


def extract_window_embeddings(
    audio_file, scene_start: float, scene_end: float,
    window_sec: float = 4.0, stride_sec: float = 1.0,
) -> list[dict]:
    """씬 안에서 4초 윈도우를 1초씩 밀면서 임베딩 추출.

    끝 보정 시 직전 윈도우와 0.5초 이상 차이날 때만 추가 (중복 방지).
    """
    embeddings = []
    t = scene_start
    while t + window_sec <= scene_end:
        emb = get_audio_embedding(audio_file, t, t + window_sec)
        embeddings.append(
            {"start_sec": round(t, 2), "end_sec": round(t + window_sec, 2), "embedding": emb}
        )
        t += stride_sec

    # 끝 보정 (중복 방지)
    if embeddings and embeddings[-1]["end_sec"] < scene_end - 0.1:
        new_start = scene_end - window_sec
        if (
            new_start >= scene_start
            and new_start - embeddings[-1]["start_sec"] >= 0.5
        ):
            emb = get_audio_embedding(audio_file, new_start, scene_end)
            embeddings.append(
                {"start_sec": round(new_start, 2), "end_sec": round(scene_end, 2), "embedding": emb}
            )

    # 씬이 4초보다 짧으면 씬 전체를 한 번에
    if not embeddings:
        emb = get_audio_embedding(audio_file, scene_start, scene_end)
        embeddings.append(
            {"start_sec": round(scene_start, 2), "end_sec": round(scene_end, 2), "embedding": emb}
        )
    return embeddings


def verify_panns_embedding(audio_file, scenes: list[dict]) -> None:
    """PANNs 임베딩 품질 검증."""
    emb = get_audio_embedding(audio_file, 0.0, 4.0)
    assert emb.shape == (2048,), f"shape {emb.shape} ≠ (2048,)"
    assert not np.isnan(emb).any(), "NaN 포함"
    assert not np.isinf(emb).any(), "Inf 포함"
    assert np.abs(emb).sum() > 0.1, "임베딩이 거의 0 — 입력 오디오가 무음?"

    duration = get_duration(audio_file)
    if duration > 60:
        emb2 = get_audio_embedding(audio_file, duration * 0.4, duration * 0.4 + 4.0)
        cosine = np.dot(emb, emb2) / (np.linalg.norm(emb) * np.linalg.norm(emb2))
        assert cosine < 0.99, f"다른 구간인데 거의 동일 (cos={cosine:.3f})"
        print(
            f"  ✓ 0~4초 vs {duration*0.4:.0f}~{duration*0.4+4:.0f}초 cosine = {cosine:.3f}"
        )

    print(f"  ✓ 임베딩 shape (2048,)")
    print(f"  ✓ NaN/Inf 없음, L2 norm = {np.linalg.norm(emb):.2f}")

    scene = scenes[0]
    windows = extract_window_embeddings(audio_file, scene["start_sec"], scene["end_sec"])
    assert len(windows) >= 1, "윈도우 0개"
    assert all(w["embedding"].shape == (2048,) for w in windows)
    print(f"  ✓ 씬 0({scene['duration_sec']}초)에서 윈도우 {len(windows)}개 추출")


# ────────────────────────────────────────────────────────
# Silero VAD
# ────────────────────────────────────────────────────────
_vad = None


def load_vad():
    """Silero VAD 모델 싱글톤 로더."""
    global _vad
    if _vad is None:
        from silero_vad import load_silero_vad
        _vad = load_silero_vad()
    return _vad


def detect_speech(
    audio_file,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
) -> list[dict]:
    """대사 구간 감지. librosa로 직접 로드해 torchcodec 의존 회피."""
    from silero_vad import get_speech_timestamps

    vad = load_vad()
    y, _sr = librosa.load(str(audio_file), sr=16000, mono=True)
    wav = torch.from_numpy(y).float()

    timestamps = get_speech_timestamps(
        wav, vad,
        sampling_rate=16000,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    return [
        {"start": round(t["start"], 2), "end": round(t["end"], 2)}
        for t in timestamps
    ]


def verify_vad(audio_file) -> list[dict]:
    """VAD 결과 검증 + 검증된 segments 반환."""
    total_duration = get_duration(audio_file)
    segs = detect_speech(audio_file)

    if len(segs) == 0:
        print("  ⚠️ 대사 0개 감지 — 효과음만 있는 영상인지 확인")
        return segs

    for s in segs:
        assert s["start"] < s["end"], f"잘못된 구간: {s}"
    for i in range(len(segs) - 1):
        assert segs[i]["end"] <= segs[i + 1]["start"] + 0.01, f"구간 겹침"
    for s in segs:
        assert 0 <= s["start"] < total_duration
        assert 0 < s["end"] <= total_duration + 0.5
    for s in segs:
        dur = s["end"] - s["start"]
        assert dur >= 0.25, f"{dur}초 구간 너무 짧음"

    total_speech = sum(s["end"] - s["start"] for s in segs)
    avg_len = total_speech / len(segs)
    speech_ratio = total_speech / total_duration

    print(f"  ✓ 대사 {len(segs)}개 구간 감지")
    print(f"  ✓ 평균 길이 {avg_len:.2f}초, 총 {total_speech:.1f}초 ({speech_ratio*100:.1f}%)")

    if speech_ratio > 0.8:
        print(f"  ⚠️ 대사 비율 {speech_ratio*100:.0f}% — 음악을 대사로 오인 가능성")
    if avg_len > 10:
        print(f"  ⚠️ 평균 구간 {avg_len:.0f}초 — min_silence_duration_ms 조정 검토")

    durations = [s["end"] - s["start"] for s in segs]
    short_n = sum(1 for d in durations if d < 1.0)
    mid_n = sum(1 for d in durations if 1.0 <= d < 3.0)
    long_n = sum(1 for d in durations if d >= 3.0)
    print(f"  📊 분포: <1초 {short_n}개 / 1~3초 {mid_n}개 / 3초+ {long_n}개")

    return segs


# ────────────────────────────────────────────────────────
# Dialogue density
# ────────────────────────────────────────────────────────
def compute_dialogue_density(
    scene_start: float, scene_end: float, speech_timestamps: list[dict]
) -> tuple[float, list[list[float]]]:
    """씬 안에서 대사가 차지하는 비율 (0~1) + 씬 내 상대 좌표 segments."""
    scene_duration = scene_end - scene_start
    if scene_duration <= 0:
        return 0.0, []

    total_speech = 0.0
    segments_in_scene = []

    for ts in speech_timestamps:
        overlap_start = max(scene_start, ts["start"])
        overlap_end = min(scene_end, ts["end"])
        if overlap_end > overlap_start:
            total_speech += overlap_end - overlap_start
            segments_in_scene.append(
                [round(overlap_start - scene_start, 2), round(overlap_end - scene_start, 2)]
            )

    density = total_speech / scene_duration
    return round(density, 3), segments_in_scene


def verify_dialogue_density_logic() -> None:
    """edge case 검증."""
    # 1. 대사 없음
    d, segs = compute_dialogue_density(0, 10, [])
    assert d == 0.0 and segs == []
    # 2. 완전 덮음
    d, segs = compute_dialogue_density(5, 10, [{"start": 4, "end": 11}])
    assert d == 1.0 and segs == [[0.0, 5.0]]
    # 3. 부분 겹침
    d, segs = compute_dialogue_density(5, 10, [{"start": 3, "end": 7}])
    assert abs(d - 0.4) < 0.01 and segs == [[0.0, 2.0]]
    # 4. 여러 구간
    d, segs = compute_dialogue_density(0, 10, [{"start": 1, "end": 3}, {"start": 5, "end": 8}])
    assert abs(d - 0.5) < 0.01 and len(segs) == 2
    # 5. 씬 밖 무시
    d, segs = compute_dialogue_density(0, 10, [{"start": 15, "end": 20}])
    assert d == 0.0
    print(f"  ✓ Edge case 5개 모두 통과")


# ────────────────────────────────────────────────────────
# 단일 영상 처리 헬퍼
# ────────────────────────────────────────────────────────
def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"직렬화 불가: {type(obj)}")


def process_audio(name: str, save_embeddings: bool = False) -> dict:
    """VAD + density 계산 + audio_features JSON 저장.

    save_embeddings=True면 PANNs 윈도우 임베딩까지 저장 (파일이 매우 커짐).
    기본은 False — 임베딩은 EQ 계산 시 즉석 추출.
    """
    ensure_dirs()
    audio_32k = audio_path(name, "32k")
    scenes_path = scenes_json(name)

    if not audio_32k.exists():
        raise FileNotFoundError(f"{audio_32k} 없음. audio_extractor를 먼저 실행하세요.")
    if not scenes_path.exists():
        raise FileNotFoundError(f"{scenes_path} 없음. scene_splitter를 먼저 실행하세요.")

    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    print(f"  씬 {len(scenes)}개 로드")

    # PANNs 임베딩 검증 (첫 씬만)
    print("\n  [PANNs 임베딩 검증]")
    verify_panns_embedding(audio_32k, scenes)

    # VAD
    print("\n  [Silero VAD]")
    speech_timestamps = verify_vad(audio_32k)

    # 씬별 dialogue density
    print("\n  [Dialogue density]")
    for s in scenes:
        density, segs = compute_dialogue_density(
            s["start_sec"], s["end_sec"], speech_timestamps
        )
        s["density"] = density
        s["segments_in_scene"] = segs
        assert 0 <= density <= 1, f"씬 {s['scene_id']}: density {density}"

    avg_density = sum(s["density"] for s in scenes) / len(scenes)
    dialogue_scenes = sum(1 for s in scenes if s["density"] > 0)
    print(f"  ✓ 대사 있는 씬: {dialogue_scenes}/{len(scenes)}개, 평균 density {avg_density:.2f}")

    # 옵션: 윈도우 임베딩 저장
    if save_embeddings:
        print("\n  [PANNs 윈도우 임베딩 추출 — 시간 걸림]")
        for s in scenes:
            s["windows"] = extract_window_embeddings(
                audio_32k, s["start_sec"], s["end_sec"]
            )

    out = {"scenes": scenes, "speech_timestamps": speech_timestamps}
    out_path = audio_features_json(name)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=_to_serializable),
        encoding="utf-8",
    )
    print(f"\n  💾 저장: {out_path}")
    return out


if __name__ == "__main__":
    print("[로직 검증]")
    verify_dialogue_density_logic()

    if len(sys.argv) < 2:
        # 기본: 두 트레일러
        for name in ["topgun", "lalaland"]:
            if not trailer_path(name).exists():
                print(f"\n⏭ trailer_{name}.mp4 없음, 스킵")
                continue
            print(f"\n=== {name} ===")
            process_audio(name)
    else:
        # 단일: python -m ... <name>
        process_audio(sys.argv[1])
