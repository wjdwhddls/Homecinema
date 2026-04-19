"""Silero VAD wrapper.

Spec V3.3 §5-6 (= V3.2 §5-8):
    - Input: 16 kHz mono PCM
    - Output: list of SpeechSegment(start_sec, end_sec)
    - Parameters: threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100

Designed to be runnable in parallel with scene detection / model inference —
only depends on the video's audio track.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .types import SpeechSegment

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 100
SAMPLE_RATE = 16000


def extract_audio_16k_mono(
    video_path: str | Path,
    out_wav: str | Path | None = None,
) -> Path:
    """Use ffmpeg to extract 16 kHz mono PCM from a video; returns path.

    If `out_wav` is None, a temp file is created (caller must clean up).
    """
    if out_wav is None:
        out_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
    else:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "wav",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def run_vad(
    audio_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
) -> list[SpeechSegment]:
    """Run Silero VAD on a 16 kHz mono WAV and return speech segments.

    Raises ImportError if silero_vad not installed.
    """
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"VAD requires {SAMPLE_RATE} Hz audio, got {sr}")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    waveform = torch.from_numpy(data).float()

    model = load_silero_vad()
    stamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    # stamps: [{'start': int sample, 'end': int sample}, ...]
    segments: list[SpeechSegment] = []
    for ts in stamps:
        segments.append(SpeechSegment(
            start_sec=ts["start"] / SAMPLE_RATE,
            end_sec=ts["end"] / SAMPLE_RATE,
        ))
    return segments


def extract_and_detect(
    video_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    keep_audio_path: Path | None = None,
) -> list[SpeechSegment]:
    """End-to-end: video → 16k mono wav → Silero VAD → segments.

    If `keep_audio_path` is given, the extracted WAV is preserved at that path
    (e.g. to feed model_inference.py); otherwise a temp file is deleted.
    """
    if keep_audio_path is not None:
        wav_path = extract_audio_16k_mono(video_path, keep_audio_path)
    else:
        wav_path = extract_audio_16k_mono(video_path)
    try:
        return run_vad(wav_path, threshold, min_speech_ms, min_silence_ms)
    finally:
        if keep_audio_path is None:
            try:
                Path(wav_path).unlink()
            except OSError:
                pass
