"""Layer 2 — Gemini 2.5 Pro V/A 추론.

각 4s window에 대해:
  1. 영화 원본을 File API에 1회 업로드 (영화별 캐시; 중복 업로드 방지)
  2. `Part.from_uri` + `VideoMetadata(start_offset, end_offset)` 로 해당 window 구간만 추론 대상으로 지정
  3. JSON schema 강제: {valence, arousal, confidence, reasoning}

비용 설계:
  - video_metadata 슬라이싱은 Gemini가 해당 구간만 토큰화 → 4s × ~258 tok/s ≈ 1K 비디오 토큰/call
  - 1,109 windows × 1 call ≈ 1.1M video tokens × $1.25/1M ≈ ~$1.5 (추정, Gemini 2.5 Pro)

Env:
  - GEMINI_API_KEY (.env 또는 export) — 필수

Usage:
  # pilot 10 clips (검증)
  python -m model.autoEQ.pseudo_label.layer2_gemini \\
    --windows_dir dataset/autoEQ/CCMovies/windows \\
    --films_dir   dataset/autoEQ/CCMovies/films \\
    --metadata_csv dataset/autoEQ/CCMovies/windows/metadata.csv \\
    --output_csv  dataset/autoEQ/CCMovies/labels/layer2_gemini_pilot.csv \\
    --pilot 10

  # 전체
  python -m model.autoEQ.pseudo_label.layer2_gemini \\
    --windows_dir dataset/autoEQ/CCMovies/windows \\
    --films_dir   dataset/autoEQ/CCMovies/films \\
    --metadata_csv dataset/autoEQ/CCMovies/windows/metadata.csv \\
    --output_csv  dataset/autoEQ/CCMovies/labels/layer2_gemini.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


MODEL_NAME = "gemini-2.5-pro"
SYSTEM_INSTRUCTION = """You are an expert film affect annotator.
You will see a short 4-second clip from a film.
Rate the emotional dimensions using Russell's circumplex model:
- valence: negative/unpleasant (-1) to positive/pleasant (+1).
  -1 = sad, distressed, gloomy. 0 = neutral. +1 = joyful, heartwarming, pleasant.
- arousal: calm/deactivated (-1) to excited/activated (+1).
  -1 = serene, sleepy, peaceful. 0 = neutral. +1 = intense, thrilling, tense.
- confidence: your certainty of the rating in [0, 1].
- reasoning: 1-2 short sentences citing visual, audio, or narrative cues.

Use ALL modalities available (motion, facial expression, music, sfx, dialog tone).
Return ONLY the JSON object.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "valence": {"type": "NUMBER", "description": "Valence in [-1, 1]"},
        "arousal": {"type": "NUMBER", "description": "Arousal in [-1, 1]"},
        "confidence": {"type": "NUMBER", "description": "Confidence in [0, 1]"},
        "reasoning": {"type": "STRING", "description": "1-2 sentences of rationale"},
    },
    "required": ["valence", "arousal", "confidence", "reasoning"],
}


def load_env() -> str:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("[fatal] GEMINI_API_KEY 가 비어 있습니다.")
        print(f"        {env_path} 파일에 GEMINI_API_KEY=... 를 추가하세요.")
        print("        API 키 발급: https://aistudio.google.com/app/apikey")
        sys.exit(2)
    return key


def upload_film(client, film_path: Path, cache: dict) -> object:
    """업로드 1회 + ACTIVE 대기. cache: {film_id: File object}."""
    film_id = film_path.stem
    if film_id in cache:
        return cache[film_id]
    print(f"[info] uploading {film_path.name} ({film_path.stat().st_size / 1e6:.1f} MB)")
    file_obj = client.files.upload(file=str(film_path))
    while getattr(file_obj.state, "name", str(file_obj.state)) == "PROCESSING":
        time.sleep(3)
        file_obj = client.files.get(name=file_obj.name)
    state_name = getattr(file_obj.state, "name", str(file_obj.state))
    if state_name != "ACTIVE":
        raise RuntimeError(f"upload failed for {film_id}: state={state_name}")
    print(f"[info] {film_id} ACTIVE — uri={file_obj.uri}")
    cache[film_id] = file_obj
    return file_obj


def predict_window(client, file_obj, t0: float, t1: float, retries: int = 2) -> dict:
    """Gemini 호출 + JSON parse. 실패 시 retry."""
    from google.genai import types  # type: ignore

    video_part = types.Part(
        file_data=types.FileData(file_uri=file_obj.uri, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(
            start_offset=f"{t0}s", end_offset=f"{t1}s"
        ),
    )
    prompt = (
        f"This clip spans {t0:.1f}s to {t1:.1f}s of the film. "
        "Rate its valence and arousal."
    )

    last_error = None
    for attempt in range(retries + 1):
        temp = 0.0 if attempt == 0 else 0.2
        try:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=temp,
                seed=42,
                max_output_tokens=2048,
            )
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=[video_part, prompt],
                config=config,
            )
            text = resp.text
            if not text:
                raise ValueError("empty response text")
            parsed = json.loads(text)
            # clamp + type coerce
            return {
                "valence": max(-1.0, min(1.0, float(parsed["valence"]))),
                "arousal": max(-1.0, min(1.0, float(parsed["arousal"]))),
                "confidence": max(0.0, min(1.0, float(parsed["confidence"]))),
                "reasoning": str(parsed.get("reasoning", ""))[:500],
                "parse_ok": 1,
                "n_calls": attempt + 1,
                "usage": getattr(resp, "usage_metadata", None),
            }
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if "429" in last_error or "quota" in last_error.lower():
                wait = min(60, 2 ** attempt)
                print(f"[warn] rate limit; sleep {wait}s")
                time.sleep(wait)
            else:
                time.sleep(1)

    return {
        "valence": float("nan"), "arousal": float("nan"),
        "confidence": 0.0, "reasoning": f"ERROR: {last_error}",
        "parse_ok": 0, "n_calls": retries + 1, "usage": None,
    }


def run(
    windows_dir: Path,
    films_dir: Path,
    metadata_csv: Path,
    output_csv: Path,
    pilot: int = 0,
    film_ids: set[str] | None = None,
    resume: bool = True,
) -> dict:
    from google import genai  # type: ignore

    api_key = load_env()
    client = genai.Client(api_key=api_key)
    print(f"[info] google-genai client ready, model={MODEL_NAME}")

    metadata = pd.read_csv(metadata_csv)
    if film_ids:
        metadata = metadata[metadata["film_id"].isin(film_ids)]
    if pilot > 0:
        # pilot은 film 고르게 섞어 샘플링 (재현성 위해 매 N번째)
        metadata = metadata.iloc[:: max(1, len(metadata) // pilot)].head(pilot)

    print(f"[info] target windows: {len(metadata)}")

    # resume: 기존 CSV 존재 시 이미 처리한 window_id 제외
    done = set()
    if resume and output_csv.is_file():
        try:
            prev = pd.read_csv(output_csv)
            done = set(prev["window_id"].astype(str).tolist())
            print(f"[info] resume — already done: {len(done)}")
        except Exception:
            done = set()
    metadata = metadata[~metadata["window_id"].astype(str).isin(done)]
    print(f"[info] remaining: {len(metadata)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_cache: dict = {}
    n_ok = n_err = 0
    total_input_tokens = 0
    total_output_tokens = 0
    t0_all = time.time()

    # 처음 write인지 header 판단
    is_new = not output_csv.is_file() or output_csv.stat().st_size == 0
    mode = "w" if is_new else "a"

    with output_csv.open(mode, newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "film_id", "window_id",
                "gemini_v", "gemini_a", "gemini_confidence",
                "gemini_reasoning", "n_calls", "parse_ok",
            ])

        for i, row in enumerate(metadata.itertuples(index=False)):
            film_id = row.film_id
            window_id = row.window_id
            t0 = float(row.t0)
            t1 = float(row.t1)
            film_path = films_dir / f"{film_id}.mp4"
            if not film_path.is_file():
                print(f"[err] film mp4 not found: {film_path}")
                n_err += 1
                continue

            try:
                file_obj = upload_film(client, film_path, file_cache)
            except Exception as e:  # noqa: BLE001
                print(f"[err] upload failed {film_id}: {e}")
                n_err += 1
                continue

            result = predict_window(client, file_obj, t0, t1)
            if result["parse_ok"]:
                n_ok += 1
            else:
                n_err += 1

            usage = result.get("usage")
            if usage is not None:
                total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0

            writer.writerow([
                film_id, window_id,
                f"{result['valence']:.4f}" if result["parse_ok"] else "",
                f"{result['arousal']:.4f}" if result["parse_ok"] else "",
                f"{result['confidence']:.4f}",
                result["reasoning"].replace("\n", " "),
                result["n_calls"],
                result["parse_ok"],
            ])
            f.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(metadata):
                dt = time.time() - t0_all
                rate = (i + 1) / dt
                est_cost = (total_input_tokens * 1.25 + total_output_tokens * 10) / 1e6
                print(
                    f"[info] {i+1}/{len(metadata)} "
                    f"({rate:.2f}/s, ok={n_ok}, err={n_err}, "
                    f"in_tok={total_input_tokens}, out_tok={total_output_tokens}, "
                    f"~${est_cost:.3f})"
                )

    summary = {
        "total_processed": n_ok + n_err,
        "parse_ok": n_ok,
        "errors": n_err,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": (total_input_tokens * 1.25 + total_output_tokens * 10) / 1e6,
        "elapsed_sec": time.time() - t0_all,
    }
    print(f"[done] {summary}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", type=Path, required=True)
    p.add_argument("--films_dir", type=Path, required=True)
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--pilot", type=int, default=0,
                   help="N>0이면 균등 샘플 N개만 처리 (pilot). 0이면 전체.")
    p.add_argument("--film_ids", type=str, default=None,
                   help="comma-separated film_id filter")
    p.add_argument("--no_resume", action="store_true",
                   help="기존 CSV 무시하고 처음부터 (기본: resume)")
    args = p.parse_args()

    film_ids = set(args.film_ids.split(",")) if args.film_ids else None
    run(
        windows_dir=args.windows_dir,
        films_dir=args.films_dir,
        metadata_csv=args.metadata_csv,
        output_csv=args.output_csv,
        pilot=args.pilot,
        film_ids=film_ids,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
