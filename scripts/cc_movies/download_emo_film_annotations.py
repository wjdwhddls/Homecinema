"""OpenNeuro ds004872 (Emo-FilM Annotations) 다운 + V/A 파생.

Emo-FilM 14편 각각에 대해 consensus annotation (1Hz, 50 columns) 다운로드 후
V/A (valence/arousal) 2차원 스칼라로 축약한다.

V/A 파생 공식 (Russell's circumplex 원칙 + feeling component 결합):
  valence_raw = (Good - Bad) + mean(positive discrete emotions) - mean(negative discrete emotions)
    pos = {Happiness, WarmHeartedness, Satisfaction, Love, Pride}
    neg = {Sad, Anger, Disgust, Fear, Anxiety, Guilt}
  arousal_raw = (IntenseEmotion + Alert - AtEase - Calm) + mean(high-arousal emotions) - mean(low-arousal emotions)
    high = {Anger, Fear, Anxiety, Surprise}
    low  = {Sad, Calm, AtEase}

그리고 per-film min-max로 대략 [-1, 1] 스케일링. (글로벌 캘리브레이션은 Layer 1 pipeline에서)

출력:
  <out_dir>/Annot_<Film>_stim.tsv.gz — 원본 50 col
  <out_dir>/Annot_<Film>_stim.json — 50 col 이름
  <out_dir>/va_<film>.csv — 시간(1Hz) × (valence, arousal) 파생
  <out_dir>/manifest.json — 각 파일 SHA + 파생 공식 기록
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds004872"
GH_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds004872/main"

# Emo-FilM에서 사용된 14편 파일명 (ds004872 derivatives 기준).
EMO_FILM_NAMES = [
    "AfterTheRain",
    "BetweenViewings",
    "BigBuckBunny",
    "Chatter",
    "FirstBite",
    "LessonLearned",
    "Payload",
    "Sintel",
    "Spaceman",
    "Superhero",
    "TearsOfSteel",
    "TheSecretNumber",
    "ToClaireFromSonny",
    "YouAgain",
]

# V/A 파생에 쓸 컬럼 그룹.
POSITIVE_EMOTIONS = ["Happiness", "WarmHeartedness", "Satisfaction", "Love", "Pride"]
NEGATIVE_EMOTIONS = ["Sad", "Anger", "Disgust", "Fear", "Anxiety", "Guilt"]
HIGH_AROUSAL = ["Anger", "Fear", "Anxiety", "Surprise"]
LOW_AROUSAL = ["Sad", "Calm", "AtEase"]


def _curl_download(url: str, dest: Path) -> None:
    subprocess.run(
        ["curl", "-s", "-L", "--fail", "-o", str(dest), url],
        check=True,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_va(arr: np.ndarray, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(T, 50) 배열에서 (T,) valence, (T,) arousal 파생. min-max → [-1, 1]."""
    col_idx = {c: i for i, c in enumerate(columns)}

    # feeling 기반 기본항
    good_minus_bad = arr[:, col_idx["Good"]] - arr[:, col_idx["Bad"]]
    alert_intensity = (
        arr[:, col_idx["IntenseEmotion"]] + arr[:, col_idx["Alert"]]
        - arr[:, col_idx["AtEase"]] - arr[:, col_idx["Calm"]]
    )

    # 이산 감정 기반 valence
    pos = np.stack([arr[:, col_idx[c]] for c in POSITIVE_EMOTIONS], axis=1).mean(axis=1)
    neg = np.stack([arr[:, col_idx[c]] for c in NEGATIVE_EMOTIONS], axis=1).mean(axis=1)
    valence_raw = good_minus_bad + (pos - neg)

    # 이산 감정 기반 arousal
    high = np.stack([arr[:, col_idx[c]] for c in HIGH_AROUSAL], axis=1).mean(axis=1)
    low = np.stack([arr[:, col_idx[c]] for c in LOW_AROUSAL], axis=1).mean(axis=1)
    arousal_raw = alert_intensity + (high - low)

    # min-max → [-1, 1] (per-film)
    def _scale(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, [2, 98])  # robust
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        clipped = np.clip(x, lo, hi)
        return 2.0 * (clipped - lo) / (hi - lo) - 1.0

    return _scale(valence_raw), _scale(arousal_raw)


def process_film(film_name: str, out_dir: Path) -> dict:
    """한 영화의 annotation 다운 + V/A 파생. 반환: 요약 dict."""
    tsv_url = f"{S3_BASE}/derivatives/Annot_{film_name}_stim.tsv.gz"
    json_url = f"{GH_BASE}/derivatives/Annot_{film_name}_stim.json"

    tsv_path = out_dir / f"Annot_{film_name}_stim.tsv.gz"
    json_path = out_dir / f"Annot_{film_name}_stim.json"

    if not tsv_path.is_file():
        print(f"  [download] {film_name} tsv.gz")
        _curl_download(tsv_url, tsv_path)
    if not json_path.is_file():
        print(f"  [download] {film_name} json")
        _curl_download(json_url, json_path)

    meta = json.loads(json_path.read_text())
    columns = meta["Columns"]
    assert len(columns) == 50, f"{film_name}: expected 50 columns, got {len(columns)}"

    with gzip.open(tsv_path, "rt") as f:
        arr = np.loadtxt(f, delimiter="\t")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 50)

    valence, arousal = derive_va(arr, columns)

    # 1Hz sample별 CSV
    va_csv = out_dir / f"va_{film_name}.csv"
    with open(va_csv, "w") as f:
        f.write("t_sec,valence,arousal\n")
        for i, (v, a) in enumerate(zip(valence, arousal)):
            f.write(f"{i},{v:.6f},{a:.6f}\n")

    return {
        "film": film_name,
        "n_timepoints_1hz": int(arr.shape[0]),
        "valence_mean": float(np.mean(valence)),
        "valence_std": float(np.std(valence)),
        "arousal_mean": float(np.mean(arousal)),
        "arousal_std": float(np.std(arousal)),
        "tsv_sha256": _sha256(tsv_path),
        "va_csv": str(va_csv),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emo-FilM annotations download + V/A derive")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--films", type=str, default="",
                        help="쉼표 구분 영화명 (대소문자 주의). 빈 값이면 14편 전체.")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = EMO_FILM_NAMES if not args.films else [x.strip() for x in args.films.split(",")]
    print(f"[info] {len(targets)}편 처리 → {args.out_dir}")

    results: list[dict] = []
    for name in targets:
        try:
            r = process_film(name, args.out_dir)
            print(f"  OK {name}: T={r['n_timepoints_1hz']}  V={r['valence_mean']:+.2f}±{r['valence_std']:.2f}  A={r['arousal_mean']:+.2f}±{r['arousal_std']:.2f}")
            results.append(r)
        except Exception as e:
            print(f"  ERROR {name}: {e}", file=sys.stderr)
            results.append({"film": name, "error": str(e)})

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "OpenNeuro ds004872 (Emo-FilM Annotations, CC0)",
        "columns_count": 50,
        "va_derivation": {
            "valence": "(Good - Bad) + mean(positive_emotions) - mean(negative_emotions)",
            "arousal": "(IntenseEmotion + Alert - AtEase - Calm) + mean(high_arousal) - mean(low_arousal)",
            "positive_emotions": POSITIVE_EMOTIONS,
            "negative_emotions": NEGATIVE_EMOTIONS,
            "high_arousal": HIGH_AROUSAL,
            "low_arousal": LOW_AROUSAL,
            "normalize": "per-film robust min-max using 2/98 percentile → [-1, 1]",
        },
        "films": results,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    errors = sum(1 for r in results if "error" in r)
    print(f"\n[info] OK={len(results)-errors} err={errors}")
    print(f"[info] manifest: {args.out_dir / 'manifest.json'}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
