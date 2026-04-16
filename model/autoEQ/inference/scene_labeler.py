"""scene_labeler.py — CLIP + PANNs 멀티모달 자동 씬 라벨링.

작업 10-2.5 (Day 4~5):
- CLIP ViT-B/32로 키프레임 비주얼 분류
- PANNs Cnn14로 오디오 태그 추출
- 양립 가능성(COMPATIBLE_AUDIO_HINTS)으로 confidence 등급화
- auto_labels → MANUAL_MOOD_LABELS 자동 변환

학술 포지셔닝: "Audio-Visual Zero-Shot Scene Classification with HITL Verification"
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .audio_analyzer import load_panns
from .paths import (
    PANNS_LABELS_CSV,
    audio_path,
    scene_labels_json,
    scenes_json,
    trailer_path,
    ensure_dirs,
)

# ────────────────────────────────────────────────────────
# CLIP 로더 (싱글톤)
# ────────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None


def load_clip():
    """CLIP ViT-B/32 (~600MB) 싱글톤 로더."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor


# ────────────────────────────────────────────────────────
# 후보 라벨 (영상별로 세분화 — Outro 쏠림 방지)
# ────────────────────────────────────────────────────────
CANDIDATE_LABELS = {
    "trailer_topgun": [
        "a photo of a text logo on black background with no people",
        "a photo of a pilot in a cockpit wearing a helmet and oxygen mask",
        "a photo of a fighter jet flying through clouds or sky",
        "a photo of fighter jets in close combat maneuvers",
        "a photo of a fighter jet on an aircraft carrier deck",
        "a photo of two people talking face to face in close-up",
        "a photo of a person with a serious emotional expression",
        "a photo of an explosion or fireball",
        "a photo of military personnel in uniform walking or running",
    ],
    "trailer_lalaland": [
        "a photo of cars on a highway or Los Angeles street",
        "a photo of a city skyline or urban landscape at sunset",
        "a photo of people dancing with colorful clothes",
        "a photo of a couple dancing together romantically",
        "a photo of a person playing piano",
        "a photo of a jazz musician playing a brass instrument",
        "a photo of two people having a close intimate conversation",
        "a photo of a man and woman looking at each other romantically",
        "a photo of a theater stage with spotlights or performance",
        "a photo of a text logo or movie credits on plain background",
    ],
}

# CLIP 출력 → 짧은 씬 이름 매핑
LABEL_TO_SCENE_NAME = {
    # Top Gun
    "a photo of a text logo on black background with no people": "Title",
    "a photo of a pilot in a cockpit wearing a helmet and oxygen mask": "Cockpit",
    "a photo of a fighter jet flying through clouds or sky": "Flight",
    "a photo of fighter jets in close combat maneuvers": "Dogfight",
    "a photo of a fighter jet on an aircraft carrier deck": "Carrier",
    "a photo of two people talking face to face in close-up": "Dialogue",
    "a photo of a person with a serious emotional expression": "Emotional",
    "a photo of an explosion or fireball": "Explosion",
    "a photo of military personnel in uniform walking or running": "Ground",
    # La La Land
    "a photo of cars on a highway or Los Angeles street": "Highway",
    "a photo of a city skyline or urban landscape at sunset": "Cityscape",
    "a photo of people dancing with colorful clothes": "Dance",
    "a photo of a couple dancing together romantically": "Couple dance",
    "a photo of a person playing piano": "Piano",
    "a photo of a jazz musician playing a brass instrument": "Jazz",
    "a photo of two people having a close intimate conversation": "Dialogue",
    "a photo of a man and woman looking at each other romantically": "Romance",
    "a photo of a theater stage with spotlights or performance": "Stage",
    "a photo of a text logo or movie credits on plain background": "Title",
}


# ────────────────────────────────────────────────────────
# AudioSet 태그 → 씬 성격 매핑
# ────────────────────────────────────────────────────────
AUDIO_TO_SCENE_HINT = {
    # Action
    "Explosion": "Action",
    "Gunshot, gunfire": "Action",
    "Machine gun": "Action",
    "Aircraft": "Flight",
    "Aircraft engine": "Flight",
    "Jet engine": "Flight",
    "Helicopter": "Flight",
    "Engine": "Vehicle",
    "Vehicle": "Vehicle",
    # Dialogue
    "Speech": "Dialogue",
    "Male speech, man speaking": "Dialogue",
    "Female speech, woman speaking": "Dialogue",
    "Conversation": "Dialogue",
    "Narration, monologue": "Dialogue",
    # Music
    "Music": "Music",
    "Musical instrument": "Music",
    "Piano": "Music (Piano)",
    "Singing": "Music (Vocal)",
    "Choir": "Music (Vocal)",
    "Jazz": "Jazz",
    "Saxophone": "Jazz",
    "Trumpet": "Jazz",
    "Dance music": "Dance",
    "Pop music": "Music",
    "Orchestra": "Music",
    "Soundtrack music": "Music",
    "Ambient music": "Ambient",
    "Background music": "Ambient",
    "Silence": "Silence",
    # Nature/Ambient
    "Wind": "Nature",
    "Rain": "Nature",
    "Traffic noise, roadway noise": "Cityscape",
    "Vehicle horn, car horn, honking": "Cityscape",
}


# ────────────────────────────────────────────────────────
# 양립 가능성 매핑 (CLIP scene_name과 PANNs audio_hint이 의미적으로 양립)
# ────────────────────────────────────────────────────────
COMPATIBLE_AUDIO_HINTS = {
    # Top Gun
    "Title":      ["Music", "Ambient", "Silence", "Unknown"],
    "Cockpit":    ["Flight", "Vehicle", "Dialogue", "Music", "Engine", "Unknown"],
    "Flight":     ["Flight", "Vehicle", "Music", "Engine", "Unknown"],
    "Dogfight":   ["Flight", "Action", "Music", "Vehicle", "Engine", "Unknown"],
    "Carrier":    ["Flight", "Vehicle", "Music", "Engine", "Unknown"],
    "Dialogue":   ["Dialogue", "Music", "Ambient", "Unknown"],
    "Emotional":  ["Dialogue", "Music", "Ambient", "Unknown"],
    "Explosion":  ["Action", "Music", "Unknown"],
    "Ground":     ["Dialogue", "Music", "Ambient", "Unknown"],
    # La La Land
    "Highway":    ["Cityscape", "Music", "Ambient", "Unknown"],
    "Cityscape":  ["Cityscape", "Music", "Ambient", "Unknown"],
    "Dance":      ["Music", "Music (Vocal)", "Music (Piano)", "Dance", "Jazz", "Unknown"],
    "Couple dance": ["Music", "Music (Vocal)", "Music (Piano)", "Dance", "Unknown"],
    "Piano":      ["Music (Piano)", "Music", "Unknown"],
    "Jazz":       ["Jazz", "Music", "Music (Vocal)", "Unknown"],
    "Romance":    ["Music", "Music (Piano)", "Dialogue", "Ambient", "Unknown"],
    "Stage":      ["Music", "Music (Vocal)", "Dance", "Unknown"],
}


# ────────────────────────────────────────────────────────
# 키프레임 추출 + CLIP 분류
# ────────────────────────────────────────────────────────
def extract_keyframe(video_path, time_sec: float, output_path) -> None:
    """특정 시각의 프레임을 JPG로 저장."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(time_sec),
            "-i", str(video_path), "-vframes", "1", "-q:v", "2",
            str(output_path),
        ],
        check=True, capture_output=True,
    )


def classify_scene_clip(
    video_path, start_sec: float, end_sec: float, candidate_labels: list[str]
) -> tuple[str, float, list[float]]:
    """씬의 중앙 프레임을 CLIP으로 분류.

    Returns: (best_label, confidence, all_probs)
    """
    center = (start_sec + end_sec) / 2
    tmp_frame = Path("/tmp/_clip_keyframe.jpg")
    extract_keyframe(video_path, center, tmp_frame)

    try:
        model, processor = load_clip()
        image = Image.open(tmp_frame)
        inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=-1).squeeze().numpy()
        best_idx = int(probs.argmax())
        return candidate_labels[best_idx], float(probs[best_idx]), probs.tolist()
    finally:
        tmp_frame.unlink(missing_ok=True)


# ────────────────────────────────────────────────────────
# PANNs AudioSet 태그
# ────────────────────────────────────────────────────────
def get_audio_tags(
    audio_file, start_sec: float, end_sec: float, top_k: int = 5
) -> list[tuple[str, float]]:
    """씬 구간의 상위 K개 AudioSet 태그 반환."""
    import librosa

    y, _ = librosa.load(
        str(audio_file), sr=32000,
        offset=start_sec, duration=end_sec - start_sec, mono=True,
    )

    panns = load_panns()
    waveform = torch.FloatTensor(y).unsqueeze(0)
    with torch.no_grad():
        output = panns(waveform)

    probs = output["clipwise_output"].cpu().numpy().squeeze()
    top_indices = probs.argsort()[-top_k:][::-1]

    if not PANNS_LABELS_CSV.exists():
        raise FileNotFoundError(
            f"AudioSet 라벨 메타 없음: {PANNS_LABELS_CSV}\n"
            "audioset_tagging_cnn 레포 메타 폴더 확인."
        )
    with open(PANNS_LABELS_CSV) as f:
        reader = csv.DictReader(f)
        labels = [row["display_name"] for row in reader]

    return [(labels[i], float(probs[i])) for i in top_indices]


def infer_scene_from_audio(audio_tags: list[tuple[str, float]]) -> tuple[str, float]:
    """Top-K 오디오 태그 → 씬 성격 추정."""
    hints = []
    for tag, prob in audio_tags:
        if tag in AUDIO_TO_SCENE_HINT:
            hints.append((AUDIO_TO_SCENE_HINT[tag], prob))
    if not hints:
        return "Unknown", 0.0
    hints.sort(key=lambda x: -x[1])
    return hints[0]


# ────────────────────────────────────────────────────────
# 두 신호 결합 + confidence 등급
# ────────────────────────────────────────────────────────
def is_audio_consistent(scene_name: str, audio_hint: str) -> bool:
    """CLIP scene_name과 PANNs audio_hint이 의미적으로 양립 가능한지."""
    compatible = COMPATIBLE_AUDIO_HINTS.get(scene_name, [])
    return audio_hint in compatible or audio_hint == "Unknown"


def auto_label_scenes(
    video_key: str, video_path, audio_file, scene_boundaries: list[tuple[float, float]]
) -> list[dict]:
    """CLIP + PANNs 조합으로 씬 자동 라벨링.

    Returns: [{scene_name_clip, clip_confidence, audio_scene_hint,
               audio_top_tags, audio_consistent, scene_name_auto,
               auto_confidence, needs_review}, ...]
    """
    candidate_labels = CANDIDATE_LABELS.get(video_key)
    if candidate_labels is None:
        raise ValueError(
            f"CANDIDATE_LABELS에 '{video_key}' 없음. "
            f"가능한 키: {list(CANDIDATE_LABELS.keys())}"
        )

    results = []
    for start, end in scene_boundaries:
        # CLIP
        clip_label, clip_conf, _ = classify_scene_clip(
            video_path, start, end, candidate_labels
        )
        scene_name_clip = LABEL_TO_SCENE_NAME.get(clip_label, "Unknown")

        # PANNs
        top_tags = get_audio_tags(audio_file, start, end, top_k=5)
        audio_hint, _audio_conf = infer_scene_from_audio(top_tags)

        # 양립 가능성
        audio_consistent = is_audio_consistent(scene_name_clip, audio_hint)

        # confidence 등급
        if clip_conf > 0.5 and audio_consistent:
            final_name = scene_name_clip
            confidence = "high"
            needs_review = False
        elif clip_conf > 0.7:
            final_name = scene_name_clip
            confidence = "high"
            needs_review = False
        elif clip_conf > 0.4:
            final_name = scene_name_clip
            confidence = "medium"
            needs_review = True
        else:
            final_name = scene_name_clip + "?"
            confidence = "low"
            needs_review = True

        results.append({
            "start_sec": start,
            "end_sec": end,
            "scene_name_clip": scene_name_clip,
            "clip_confidence": round(clip_conf, 3),
            "audio_scene_hint": audio_hint,
            "audio_top_tags": [(t, round(p, 3)) for t, p in top_tags[:3]],
            "audio_consistent": audio_consistent,
            "scene_name_auto": final_name,
            "auto_confidence": confidence,
            "needs_review": needs_review,
        })

        marker = "✓" if confidence == "high" else ("?" if confidence == "medium" else "✗")
        print(
            f"  {marker} 씬 {start:5.1f}~{end:5.1f}s: {final_name:15s} "
            f"(CLIP {clip_conf:.2f}, audio={audio_hint:12s}, conf={confidence})"
        )

    return results


# ────────────────────────────────────────────────────────
# auto_labels → MANUAL_MOOD_LABELS 자동 변환
# ────────────────────────────────────────────────────────
DEFAULT_MOOD_BY_SCENE = {
    # Top Gun
    "Title":        ("Power",             0.75, 0.0),
    "Cockpit":      ("Tension",           0.70, 0.2),
    "Flight":       ("Power",             0.75, 0.1),
    "Dogfight":     ("Tension",           0.80, 0.2),
    "Carrier":      ("Power",             0.70, 0.1),
    "Dialogue":     ("Tenderness",        0.65, 0.5),
    "Emotional":    ("Tenderness",        0.75, 0.4),
    "Explosion":    ("Power",             0.85, 0.0),
    "Ground":       ("Tension",           0.65, 0.3),
    # La La Land
    "Highway":      ("Wonder",            0.65, 0.2),
    "Cityscape":    ("Wonder",            0.70, 0.1),
    "Dance":        ("Joyful Activation", 0.80, 0.3),
    "Couple dance": ("Joyful Activation", 0.80, 0.2),
    "Piano":        ("Peacefulness",      0.70, 0.1),
    "Jazz":         ("Peacefulness",      0.70, 0.2),
    "Romance":      ("Tenderness",        0.80, 0.4),
    "Stage":        ("Joyful Activation", 0.75, 0.3),
}


def auto_labels_to_manual_template(auto_labels_dict: dict) -> dict:
    """auto_labels → MANUAL_MOOD_LABELS 템플릿 생성.

    씬 경계는 Day 2 결과 그대로. scene_name은 CLIP 자동 라벨.
    감정(mood/prob/density)은 scene_name 기반 기본값으로 채우고,
    필요 시 사람이 개별 씬만 조정 가능.
    """
    template = {}
    for video_key, scenes in auto_labels_dict.items():
        template[video_key] = []
        for s in scenes:
            name = s["scene_name_auto"].rstrip("?")
            mood, prob, density = DEFAULT_MOOD_BY_SCENE.get(
                name, ("Peacefulness", 0.60, 0.2)
            )
            template[video_key].append({
                "start": s["start_sec"],
                "end": s["end_sec"],
                "scene_name": name,
                "mood": mood,
                "prob": prob,
                "density": density,
                "auto_confidence": s.get("auto_confidence", "unknown"),
            })
    return template


# ────────────────────────────────────────────────────────
# 영상 처리 헬퍼
# ────────────────────────────────────────────────────────
def load_scene_boundaries(video_key: str) -> list[tuple[float, float]]:
    """Day 2 결과에서 씬 경계만 뽑아 [(start, end), ...] 반환."""
    sj = scenes_json(video_key.replace("trailer_", ""))
    if not sj.exists():
        raise FileNotFoundError(
            f"{sj} 없음. scene_splitter를 먼저 실행하세요."
        )
    scenes = json.loads(sj.read_text(encoding="utf-8"))
    return [(s["start_sec"], s["end_sec"]) for s in scenes]


def process_labeling(video_key: str) -> list[dict]:
    """단일 영상의 자동 라벨링 + JSON 저장."""
    ensure_dirs()
    short_name = video_key.replace("trailer_", "")
    video = trailer_path(short_name)
    audio = audio_path(short_name, "32k")
    if not audio.exists():
        raise FileNotFoundError(f"{audio} 없음. audio_extractor 먼저 실행")

    boundaries = load_scene_boundaries(video_key)
    print(f"\n=== {video_key} (총 {len(boundaries)}개 씬) ===")
    results = auto_label_scenes(video_key, video, audio, boundaries)

    out_path = scene_labels_json(short_name, "auto")
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    high = sum(1 for r in results if r["auto_confidence"] == "high")
    needs_review = sum(1 for r in results if r["needs_review"])
    print(f"\n  ✓ 총 {len(results)}개 씬 자동 라벨링")
    print(f"  ✓ High confidence: {high}개 ({high/len(results)*100:.0f}%)")
    print(f"  ✓ 검수 필요: {needs_review}개")
    print(f"  💾 저장: {out_path}")
    return results


def apply_manual_corrections(video_key: str, corrections: dict[int, str]) -> list[dict]:
    """auto 라벨에 수동 수정 적용 → final 저장.

    corrections: {scene_idx: 'Correct Name'}
    """
    short_name = video_key.replace("trailer_", "")
    auto_path = scene_labels_json(short_name, "auto")
    auto_labels = json.loads(auto_path.read_text(encoding="utf-8"))

    for idx, correct_name in corrections.items():
        auto_labels[idx]["scene_name_final"] = correct_name
        auto_labels[idx]["manually_corrected"] = True

    for s in auto_labels:
        if "scene_name_final" not in s:
            s["scene_name_final"] = s["scene_name_auto"]
            s["manually_corrected"] = False

    final_path = scene_labels_json(short_name, "final")
    final_path.write_text(
        json.dumps(auto_labels, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  💾 final 저장: {final_path}")
    return auto_labels


if __name__ == "__main__":
    # 기본: 두 트레일러 모두 자동 라벨링
    if len(sys.argv) < 2:
        for video_key in ["trailer_topgun", "trailer_lalaland"]:
            short = video_key.replace("trailer_", "")
            if not trailer_path(short).exists():
                print(f"\n⏭ {video_key} 영상 없음, 스킵")
                continue
            process_labeling(video_key)
    else:
        process_labeling(sys.argv[1])
