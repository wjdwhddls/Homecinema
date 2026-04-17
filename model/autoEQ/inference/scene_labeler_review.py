"""scene_labeler_review.py — Day 6 review tooling.

Consumes `scene_labels_{name}_auto.json` and emits three review aids:

1. Representative frame per scene (JPG) under `outputs/scene_labels/review_frames_{name}/`.
2. Priority review report (Markdown) listing low-confidence scenes and
   scenes whose auto mood falls into known-ambiguous category pairs.
3. Manual review template JSON (`scene_labels_{name}_manual.json`) with
   `manual_category` / `reviewed` / `notes` fields per scene.

Usage:
    python -m model.autoEQ.inference.scene_labeler_review --name topgun
    python -m model.autoEQ.inference.scene_labeler_review --all
    python -m model.autoEQ.inference.scene_labeler_review --name topgun --skip-frames
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .paths import SCENE_LABELS_DIR, scene_labels_json, trailer_path
from .scene_labeler import DEFAULT_MOOD_BY_SCENE


SEVEN_CATEGORIES: list[str] = [
    "Tension", "Sadness", "Peacefulness", "Joyful Activation",
    "Tenderness", "Power", "Wonder",
]


# Section B: 7-category decision reference (constant; same for every trailer).
CATEGORY_REFERENCE: list[dict[str, str]] = [
    {"category": "Tension",
     "visual": "격투·추격, 긴박한 근접컷, 어두운 톤, 흔들리는 카메라",
     "audio": "타격음/엔진/기관포, 드라마틱 현악, 심장박동",
     "confusion": "Power (고각성 공유 — Tension=위협/불안, Power=승리/웅장)"},
    {"category": "Sadness",
     "visual": "고개 숙임, 눈물, 어두운 실내, 정지된 구성",
     "audio": "단조 솔로 보컬/피아노, 침묵 후 느린 현악",
     "confusion": "Tenderness (저각성 공유 — Sadness=상실, Tenderness=애정)"},
    {"category": "Peacefulness",
     "visual": "자연 풍경, 잔잔한 물, 소프트 포커스, 정지/슬로우",
     "audio": "앰비언트, 부드러운 피아노, 새소리, 침묵",
     "confusion": "Wonder (경외면 Wonder, 평온이면 Peacefulness)"},
    {"category": "Joyful Activation",
     "visual": "춤·축하, 밝은 톤, 웃음, 포화색",
     "audio": "업템포 메이저 키, 보컬/합창, 박수",
     "confusion": "Power (고각성 양성 공유)"},
    {"category": "Tenderness",
     "visual": "근접 클로즈업, 로맨스, 따뜻한 조명, 눈맞춤",
     "audio": "대화 단독, 부드러운 음악, 보컬",
     "confusion": "Sadness (저각성) / Peacefulness (정서 강도)"},
    {"category": "Power",
     "visual": "승리 포즈, 웅장한 와이드샷, 로고·타이틀, 집결",
     "audio": "금관 팡파레, 오케스트라 클라이맥스, 드럼 히트",
     "confusion": "Tension / Joyful Activation (둘 다 고각성)"},
    {"category": "Wonder",
     "visual": "경외로운 풍경·도시, 시네마틱 리빌, 느린 팬, 황혼",
     "audio": "스위핑 현악, 빌드업, 몽환적 패드",
     "confusion": "Peacefulness (저각성 유사), Power (숭고함 경계)"},
]


# Step 4: per-trailer target distribution (min, max inclusive).
DISTRIBUTION_TARGETS: dict[str, dict[str, tuple[int, int]]] = {
    "trailer_topgun": {
        "Tension":           (15, 18),
        "Power":             (10, 12),
        "Tenderness":        (8, 10),
        "Wonder":            (2, 4),
        "Joyful Activation": (2, 3),
        "Sadness":           (0, 2),
        "Peacefulness":      (0, 0),
    },
    "trailer_lalaland": {
        "Wonder":            (8, 12),
        "Tenderness":        (8, 10),
        "Joyful Activation": (6, 8),
        "Peacefulness":      (4, 6),
        "Sadness":           (1, 3),
        "Power":             (0, 0),
        "Tension":           (0, 0),
    },
}

DISTRIBUTION_TARGET_RATIONALE: dict[str, str] = {
    "trailer_topgun":
        "Tension 60% 쏠림이면 액션 트레일러 편견 고착. 다양성 확보로 EQ 동적 범위 시연.",
    "trailer_lalaland":
        "Wonder 과반 유지 시 발표 시각화에서 'EQ 동적 다양성' 설득 약화. 원작 정서축(뮤지컬)도 반영.",
}


# PANNs AudioSet tag families for heuristic mood suggestions.
VOCAL_TAGS = {
    "Singing", "Choir", "Music (Vocal)",
}
SILENCE_AMBIENT_TAGS = {
    "Silence", "Ambient music", "Background music",
}
SPEECH_TAGS = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue",
}
MUSIC_TAGS = {
    "Music", "Musical instrument", "Piano", "Saxophone", "Trumpet",
    "Jazz", "Pop music", "Orchestra", "Dance music", "Soundtrack music",
    "Theme music", "Ambient music", "Background music",
}


# High-arousal positive + high-arousal negative / low-arousal nearby — see Day 6 spec.
AMBIGUOUS_PAIRS: list[tuple[str, str]] = [
    ("Tension", "Power"),
    ("Sadness", "Tenderness"),
    ("Peacefulness", "Tenderness"),
    ("Joyful Activation", "Power"),
]


# Per-trailer overrides applied on top of scene_labeler.DEFAULT_MOOD_BY_SCENE.
# Keep scene_labeler.py read-only (rule 3) — encode genre-specific corrections here.
# La La Land: CLIP's "Title" prompt over-absorbs credits / black cuts / low-contrast
# transitions. DEFAULT maps Title → Power which inflates Power from 1 to 20 on a
# romantic musical trailer. Wonder is the best fit for those neutral-cinematic frames.
OVERRIDE_MOOD: dict[str, dict[str, tuple[str, float, float]]] = {
    "trailer_lalaland": {
        "Title": ("Wonder", 0.50, 0.1),
    },
}


def resolve_mood(video_key: str, scene_name_auto: str) -> tuple[str, float, float]:
    """Resolve (mood, prob, density) with per-trailer OVERRIDE_MOOD > DEFAULT_MOOD_BY_SCENE."""
    clean = scene_name_auto.rstrip("?")
    override = OVERRIDE_MOOD.get(video_key, {})
    if clean in override:
        return override[clean]
    return DEFAULT_MOOD_BY_SCENE.get(clean, ("Peacefulness", 0.60, 0.2))


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def extract_frames(name: str, *, quality: int = 3) -> Path:
    """Extract one representative (midpoint) JPG per scene."""
    auto_path = scene_labels_json(name, "auto")
    if not auto_path.exists():
        raise FileNotFoundError(f"{auto_path} not found — run scene_labeler first.")
    video = trailer_path(name)
    if not video.exists():
        print(f"  ⚠  trailer not found ({video}), skipping frame extraction")
        return SCENE_LABELS_DIR / f"review_frames_{name}"

    frames_dir = SCENE_LABELS_DIR / f"review_frames_{name}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    print(f"  ◉ Extracting {len(auto)} frames → {frames_dir}")

    ok = 0
    for idx, s in enumerate(auto):
        center = (s["start_sec"] + s["end_sec"]) / 2
        out = frames_dir / f"scene_{idx:03d}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{center:.3f}",
                    "-i", str(video),
                    "-vframes", "1",
                    "-q:v", str(quality),
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  ⚠  scene {idx} frame extract failed: {e.returncode}")

    print(f"  ✓ {ok}/{len(auto)} frames written")
    return frames_dir


def build_report(name: str) -> Path:
    auto_path = scene_labels_json(name, "auto")
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    frames_dir_name = f"review_frames_{name}"
    video_key = f"trailer_{name}"

    enriched = []
    for idx, s in enumerate(auto):
        mood, prob, _density = resolve_mood(video_key, s["scene_name_auto"])
        enriched.append({
            "idx": idx,
            "start": s["start_sec"],
            "end": s["end_sec"],
            "duration": s["end_sec"] - s["start_sec"],
            "scene_name_auto": s["scene_name_auto"],
            "clip_confidence": s["clip_confidence"],
            "auto_confidence": s["auto_confidence"],
            "audio_hint": s["audio_scene_hint"],
            "audio_consistent": s["audio_consistent"],
            "mood": mood,
            "default_mood_prob": prob,
            "needs_review": s["needs_review"],
            "frame_rel": f"{frames_dir_name}/scene_{idx:03d}.jpg",
        })

    # Section 1: priority set = (bottom 20% by clip_confidence) ∪ (clip_conf < 0.4)
    sorted_by_conf = sorted(enriched, key=lambda x: x["clip_confidence"])
    bottom_n = max(1, len(enriched) // 5)
    priority_ids: set[int] = set(x["idx"] for x in sorted_by_conf[:bottom_n])
    priority_ids.update(x["idx"] for x in enriched if x["clip_confidence"] < 0.4)
    priority = sorted((x for x in enriched if x["idx"] in priority_ids),
                      key=lambda x: x["clip_confidence"])

    lines: list[str] = []
    lines.append(f"# Review Report — {name}")
    lines.append("")
    lines.append(f"- Source: `{auto_path.name}`")
    lines.append(f"- Total scenes: **{len(enriched)}**")
    high = sum(1 for x in enriched if x["auto_confidence"] == "high")
    medium = sum(1 for x in enriched if x["auto_confidence"] == "medium")
    low = sum(1 for x in enriched if x["auto_confidence"] == "low")
    lines.append(
        f"- Confidence: high={high} · medium={medium} · low={low}"
    )
    needs = sum(1 for x in enriched if x["needs_review"])
    lines.append(f"- `needs_review` flagged by auto pipeline: **{needs}**")
    lines.append("")

    # Section 1
    lines.append("## 1. 우선 검수 대상 (bottom-20% OR clip<0.4)")
    lines.append("")
    lines.append(
        "| idx | start | end | dur | scene_auto | CLIP | audio_hint | mood (default) | frame |"
    )
    lines.append("|----:|------:|----:|----:|------------|-----:|-----------|-----|-------|")
    for x in priority:
        lines.append(
            f"| {x['idx']} | {_fmt_time(x['start'])} | {_fmt_time(x['end'])} "
            f"| {x['duration']:.1f}s | {x['scene_name_auto']} "
            f"| {x['clip_confidence']:.2f} | {x['audio_hint']} "
            f"| {x['mood']} | `{x['frame_rel']}` |"
        )
    lines.append("")

    # Section 2: ambiguous pairs
    lines.append("## 2. 애매한 카테고리 쌍")
    lines.append("")
    lines.append(
        "각 쌍에 속하는 씬들을 함께 비교해 자동 라벨이 올바른 감정으로 "
        "매핑됐는지 확인하세요. 기본 mood는 `scene_name_auto` → `DEFAULT_MOOD_BY_SCENE` "
        "매핑 결과입니다 (Day 10 gate 모델로 교체 예정)."
    )
    lines.append("")
    for a, b in AMBIGUOUS_PAIRS:
        matched = [x for x in enriched if x["mood"] in (a, b)]
        lines.append(f"### ({a}, {b}) — {len(matched)}개 씬")
        lines.append("")
        if not matched:
            lines.append("_해당 mood로 매핑된 씬 없음._")
            lines.append("")
            continue
        lines.append(
            "| idx | start~end | scene_auto | CLIP | mood | frame |"
        )
        lines.append(
            "|----:|-----------|------------|-----:|------|-------|"
        )
        for x in sorted(matched, key=lambda v: v["start"]):
            lines.append(
                f"| {x['idx']} | {_fmt_time(x['start'])}~{_fmt_time(x['end'])} "
                f"| {x['scene_name_auto']} | {x['clip_confidence']:.2f} "
                f"| **{x['mood']}** | `{x['frame_rel']}` |"
            )
        lines.append("")

    # Section 3: full table
    lines.append("## 3. 전체 씬 표")
    lines.append("")
    lines.append(
        "| idx | start | end | dur | scene_auto | CLIP | audio | consistent | auto_conf | mood | frame |"
    )
    lines.append(
        "|----:|------:|----:|----:|------------|-----:|-------|:----------:|:---------:|------|-------|"
    )
    for x in enriched:
        consistent = "✓" if x["audio_consistent"] else "✗"
        lines.append(
            f"| {x['idx']} | {_fmt_time(x['start'])} | {_fmt_time(x['end'])} "
            f"| {x['duration']:.1f}s | {x['scene_name_auto']} "
            f"| {x['clip_confidence']:.2f} | {x['audio_hint']} | {consistent} "
            f"| {x['auto_confidence']} | {x['mood']} | `{x['frame_rel']}` |"
        )
    lines.append("")

    report_path = SCENE_LABELS_DIR / f"review_report_{name}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ report → {report_path}")
    return report_path


def build_manual_template(name: str) -> Path:
    """Create scene_labels_{name}_manual.json. Never overwrite an existing one."""
    auto_path = scene_labels_json(name, "auto")
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    video_key = f"trailer_{name}"

    scenes_out = []
    for s in auto:
        mood, _prob, _density = resolve_mood(video_key, s["scene_name_auto"])
        enriched = dict(s)
        enriched["manual_category"] = mood
        enriched["reviewed"] = False
        enriched["notes"] = ""
        scenes_out.append(enriched)

    doc = {
        "review_metadata": {
            "auto_source": auto_path.name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reviewed_count": 0,
            "total_scenes": len(scenes_out),
        },
        "scenes": scenes_out,
    }

    target = scene_labels_json(name, "manual")
    if target.exists():
        target_new = target.with_suffix(".json.new")
        target_new.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"  ⚠  {target.name} already exists — wrote {target_new.name} instead.\n"
            "     Run `diff` and decide manually whether to promote."
        )
        return target_new

    target.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✓ manual template → {target}")
    return target


def title_gap_candidates(name: str, *, threshold: float = 0.3) -> list[dict]:
    """Scenes whose CLIP top1 is 'Title' but top1-top2 gap < threshold.

    Reads `clip_probs_{name}.json` (produced by clip_probs_rebuild). If the file
    is absent, returns [] and prints a hint. Each returned record carries the
    alternative mood that would be assigned if the reviewer promoted top-2.
    """
    probs_path = SCENE_LABELS_DIR / f"clip_probs_{name}.json"
    if not probs_path.exists():
        print(f"  ◉ {probs_path.name} not found — run clip_probs_rebuild to enable "
              "Title gap analysis.")
        return []

    doc = json.loads(probs_path.read_text(encoding="utf-8"))
    video_key = f"trailer_{name}"
    out: list[dict] = []
    for s in doc["scenes"]:
        if "top1" not in s or s["top1"]["label"] != "Title":
            continue
        if len(s.get("top_k", [])) < 2:
            continue
        gap = s["top_k"][0]["prob"] - s["top_k"][1]["prob"]
        if gap >= threshold:
            continue
        alt_label = s["top_k"][1]["label"]
        alt_mood, _p, _d = resolve_mood(video_key, alt_label)
        out.append({
            "idx": s["idx"],
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
            "top1_prob": s["top_k"][0]["prob"],
            "top2_label": alt_label,
            "top2_prob": s["top_k"][1]["prob"],
            "gap": round(gap, 4),
            "alt_mood_if_promoted": alt_mood,
        })
    return out


def print_title_gap(name: str, *, threshold: float = 0.3) -> None:
    cands = title_gap_candidates(name, threshold=threshold)
    if not cands:
        print(f"  ◉ Title-gap(<{threshold}) 후보 없음.")
        return
    print(f"  ◉ Title top1-top2 gap<{threshold} 후보 {len(cands)}개 "
          "(수동 검수 집중 대상):")
    print(f"    {'idx':>3}  {'start':>6}~{'end':>6}  "
          f"{'top1(Title)':>11}  {'top2':<14}  {'gap':>5}  "
          f"{'alt_mood':<18}")
    for c in cands:
        print(
            f"    {c['idx']:>3}  {c['start_sec']:>6.2f}~{c['end_sec']:>6.2f}  "
            f"{c['top1_prob']:>11.3f}  {c['top2_label']:<14}  "
            f"{c['gap']:>5.3f}  {c['alt_mood_if_promoted']:<18}"
        )


# ──────────────────────────────────────────────────────────────
# Full review guide (Section A..E)
# ──────────────────────────────────────────────────────────────
def _load_clip_probs(name: str) -> dict | None:
    path = SCENE_LABELS_DIR / f"clip_probs_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _source_rule(video_key: str, scene_name_auto: str, current_mood: str) -> str:
    """Detect how current manual_category was set: 'override' / 'default' / 'manual'."""
    clean = scene_name_auto.rstrip("?")
    if clean in OVERRIDE_MOOD.get(video_key, {}):
        if OVERRIDE_MOOD[video_key][clean][0] == current_mood:
            return "override"
    default = DEFAULT_MOOD_BY_SCENE.get(clean, ("Peacefulness", 0.60, 0.2))
    if default[0] == current_mood:
        return "default"
    return "manual"


def _audio_top1(scene: dict) -> tuple[str, float] | None:
    tags = scene.get("audio_top_tags") or []
    if not tags:
        return None
    t = tags[0]
    if isinstance(t, (list, tuple)) and len(t) >= 2:
        return t[0], float(t[1])
    return None


def _clip_top1_conf(idx: int, clip_probs_by_idx: dict, auto_entry: dict) -> float:
    entry = clip_probs_by_idx.get(idx) if clip_probs_by_idx else None
    if entry and "top1" in entry:
        return float(entry["top1"]["prob"])
    return float(auto_entry.get("clip_confidence", 0.0))


def _suggest_candidate_moods(
    auto_entry: dict,
    clip_probs_entry: dict | None,
    video_key: str,
    current_mood: str,
) -> list[tuple[str, str]]:
    """Return [(mood, reason), ...] suggestion list, skipping current_mood."""
    candidates: list[tuple[str, str]] = []
    seen: set[str] = {current_mood}

    # 1) CLIP top-2 different scene_name → its mapped mood
    if clip_probs_entry and clip_probs_entry.get("top_k"):
        topk = clip_probs_entry["top_k"]
        if len(topk) >= 1:
            t1 = topk[0]
            t1_mood = resolve_mood(video_key, t1["label"])[0]
            if t1_mood not in seen:
                candidates.append((
                    t1_mood,
                    f"CLIP top-1={t1['label']} ({t1['prob']:.2f}) → {t1_mood} 매핑",
                ))
                seen.add(t1_mood)
        if len(topk) >= 2:
            t2 = topk[1]
            t2_mood = resolve_mood(video_key, t2["label"])[0]
            if t2_mood not in seen:
                candidates.append((
                    t2_mood,
                    f"CLIP top-2={t2['label']} ({t2['prob']:.2f}) → {t2_mood} 매핑",
                ))
                seen.add(t2_mood)

    # 2) Audio-driven hints
    atop = _audio_top1(auto_entry)
    if atop:
        tag, p = atop
        if tag in VOCAL_TAGS:
            for m in ("Joyful Activation", "Tenderness"):
                if m not in seen:
                    candidates.append((
                        m,
                        f"Audio top-1={tag} ({p:.2f}) → 보컬 기반 {m} 후보",
                    ))
                    seen.add(m)
        elif tag in SILENCE_AMBIENT_TAGS:
            for m in ("Peacefulness", "Wonder"):
                if m not in seen:
                    candidates.append((
                        m,
                        f"Audio top-1={tag} ({p:.2f}) → 정적/공간감 {m} 후보",
                    ))
                    seen.add(m)
        elif tag in SPEECH_TAGS:
            tags_all = auto_entry.get("audio_top_tags") or []
            music_present = any(
                (isinstance(t, (list, tuple)) and len(t) >= 1 and t[0] in MUSIC_TAGS)
                for t in tags_all
            )
            if not music_present and "Tenderness" not in seen:
                candidates.append((
                    "Tenderness",
                    f"Audio top-1=Speech ({p:.2f}) 단독 (음악 없음) → 대화 장면 Tenderness 후보",
                ))
                seen.add("Tenderness")

    return candidates[:3]


def _simulate_target_dispersion(
    current: Counter, targets: dict[str, tuple[int, int]], total: int
) -> dict[str, int]:
    """Very naive: for each category push toward mid(target range). Over = shed, under = absorb."""
    mid = {c: (mn + mx) / 2 for c, (mn, mx) in targets.items()}
    plan: dict[str, int] = {c: current.get(c, 0) for c in SEVEN_CATEGORIES}
    # nothing fancy — just clip each to midpoint
    for c in SEVEN_CATEGORIES:
        if c in mid:
            plan[c] = round(mid[c])
    # preserve total approximately by rebalancing residual into largest target-headroom
    diff = total - sum(plan.values())
    if diff != 0 and targets:
        biggest = max(targets.items(), key=lambda kv: kv[1][1])[0]
        plan[biggest] = max(0, plan[biggest] + diff)
    return plan


def build_full_review_guide(name: str) -> Path:
    """Write outputs/scene_labels/review_guide_{name}.md (Sections A..E)."""
    manual_path = scene_labels_json(name, "manual")
    if not manual_path.exists():
        raise FileNotFoundError(
            f"{manual_path} not found — run build_manual_template or promote .new first."
        )
    manual = json.loads(manual_path.read_text(encoding="utf-8"))

    auto_path = scene_labels_json(name, "auto")
    auto = json.loads(auto_path.read_text(encoding="utf-8"))

    clip_probs_doc = _load_clip_probs(name)
    clip_probs_by_idx: dict[int, dict] = {}
    if clip_probs_doc:
        for s in clip_probs_doc["scenes"]:
            clip_probs_by_idx[s["idx"]] = s

    video_key = f"trailer_{name}"
    frames_dir = f"review_frames_{name}"

    scenes = manual["scenes"]
    total = len(scenes)
    reviewed = sum(1 for s in scenes if s.get("reviewed"))
    dist = Counter(s["manual_category"] for s in scenes)
    zero_cats = [c for c in SEVEN_CATEGORIES if dist.get(c, 0) == 0]
    max_cat, max_count = dist.most_common(1)[0]
    max_ratio = max_count / total

    targets = DISTRIBUTION_TARGETS.get(video_key, {})
    rationale = DISTRIBUTION_TARGET_RATIONALE.get(video_key, "")

    L: list[str] = []
    L.append(f"# Review Guide — `{name}`")
    L.append("")
    L.append(
        f"- Source manual: `{manual_path.name}`  "
        f"·  auto: `{auto_path.name}`  "
        f"·  clip_probs: "
        f"{'`clip_probs_'+name+'.json`' if clip_probs_doc else '_missing_'}"
    )
    L.append(f"- Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append("")

    # ── Section A ──────────────────────────────────────────
    L.append("## A. 진행 대시보드")
    L.append("")
    L.append(
        f"- 전체 씬: **{total}**  ·  reviewed: **{reviewed}/{total}**  "
        f"({(reviewed/total*100 if total else 0):.1f}%)"
    )
    warn = "  ⚠ 쏠림 주의" if max_ratio >= 0.5 else ""
    L.append(
        f"- 최다 카테고리: **{max_cat} {max_count}/{total} "
        f"({max_ratio*100:.0f}%)**{warn}"
    )
    if zero_cats:
        L.append(f"- 0개 카테고리: {', '.join(zero_cats)}  ⚠")
    else:
        L.append("- 0개 카테고리: 없음")
    L.append("")

    L.append("### 현재 분포 vs 권장 타깃")
    L.append("")
    L.append("| Category | 현재 | 타깃 범위 | 상태 |")
    L.append("|---|---:|:---:|---|")
    for c in SEVEN_CATEGORIES:
        cur = dist.get(c, 0)
        t = targets.get(c)
        if t:
            mn, mx = t
            if cur < mn:
                status = f"🔽 {mn-cur}개 더 필요"
            elif cur > mx:
                status = f"🔼 {cur-mx}개 과다"
            else:
                status = "✅ 범위 내"
            rng = f"{mn}~{mx}"
        else:
            rng = "-"
            status = "(타깃 미정)"
        L.append(f"| {c} | {cur} | {rng} | {status} |")
    L.append("")
    if rationale:
        L.append(f"> **타깃 근거**: {rationale}")
        L.append("")

    # ── Section B ──────────────────────────────────────────
    L.append("## B. 7 카테고리 의사결정 레퍼런스 (상수, 모든 영상 공통)")
    L.append("")
    L.append("| Category | 시각 cue | 오디오 cue | 주요 혼동 |")
    L.append("|---|---|---|---|")
    for row in CATEGORY_REFERENCE:
        L.append(
            f"| **{row['category']}** | {row['visual']} | "
            f"{row['audio']} | {row['confusion']} |"
        )
    L.append("")

    # ── Section C ──────────────────────────────────────────
    L.append("## C. 빠른 확인 그룹 (CLIP top-1 > 0.8)")
    L.append("")
    L.append("대부분 '그대로 유지'. 이상 없으면 JSON에서 `reviewed: true`만 표시.")
    L.append("")
    L.append("| idx | start~end | mood | clip_conf | frame | 유지 |")
    L.append("|----:|-----------|------|----------:|-------|:---:|")
    quick = 0
    for idx, s in enumerate(scenes):
        conf = _clip_top1_conf(idx, clip_probs_by_idx, auto[idx])
        if conf > 0.8:
            L.append(
                f"| {idx} | {_fmt_time(s['start_sec'])}~{_fmt_time(s['end_sec'])} "
                f"| **{s['manual_category']}** | {conf:.2f} "
                f"| `{frames_dir}/scene_{idx:03d}.jpg` | [ ] |"
            )
            quick += 1
    eta_lo = max(1, quick * 7 // 60)
    eta_hi = max(1, quick * 12 // 60)
    L.append("")
    L.append(
        f"_총 **{quick}**개 씬 · 씬당 7~12초 → 약 **{eta_lo}~{eta_hi}분**_"
    )
    L.append("")

    # ── Section D ──────────────────────────────────────────
    L.append("## D. 집중 검수 그룹")
    L.append("")
    L.append(
        "아래 조건 중 하나라도 해당하는 씬. 각 블록 끝에서 `manual_category` 결정 → "
        "`reviewed: true`로 표시."
    )
    L.append("")

    # classify focus
    focused: list[tuple[int, list[str]]] = []
    for idx, s in enumerate(scenes):
        tags: list[str] = []
        conf = _clip_top1_conf(idx, clip_probs_by_idx, auto[idx])
        if conf < 0.5:
            tags.append("LOW_CONF")
        entry = clip_probs_by_idx.get(idx)
        if entry and entry.get("top_k") and len(entry["top_k"]) >= 2:
            if entry["top_k"][0]["label"] == "Title":
                gap = entry["top_k"][0]["prob"] - entry["top_k"][1]["prob"]
                if gap < 0.3:
                    tags.append("TITLE_GAP")
        if _source_rule(video_key, s["scene_name_auto"], s["manual_category"]) == "override":
            tags.append("OVERRIDE")
        if max_ratio >= 0.5 and s["manual_category"] == max_cat:
            tags.append("DOMINANT")
        atop = _audio_top1(s)
        if atop:
            tag, _ = atop
            if tag in VOCAL_TAGS or tag in SILENCE_AMBIENT_TAGS:
                tags.append("AUDIO_STORY")
            elif tag in SPEECH_TAGS:
                tags_all = s.get("audio_top_tags") or []
                music_present = any(
                    (isinstance(t, (list, tuple)) and len(t) >= 1 and t[0] in MUSIC_TAGS)
                    for t in tags_all
                )
                if not music_present:
                    tags.append("AUDIO_STORY")
        if tags:
            focused.append((idx, tags))

    L.append(
        f"_집중 검수 대상: **{len(focused)}**개 씬  "
        f"(씬당 2~3분 → 약 **{len(focused)*2}~{len(focused)*3}분**)_"
    )
    L.append("")
    L.append("**플래그 범례**: "
             "`LOW_CONF` CLIP top-1 < 0.5  ·  "
             "`TITLE_GAP` Title top-2와 0.3 미만 차  ·  "
             "`OVERRIDE` 자동 재매핑 대상  ·  "
             "`DOMINANT` 최다 카테고리 과반 참여  ·  "
             "`AUDIO_STORY` 보컬/정적/단독 대사")
    L.append("")

    for idx, tags in focused:
        s = scenes[idx]
        auto_entry = auto[idx]
        entry = clip_probs_by_idx.get(idx)
        conf = _clip_top1_conf(idx, clip_probs_by_idx, auto_entry)
        source = _source_rule(video_key, s["scene_name_auto"], s["manual_category"])
        dur = s["end_sec"] - s["start_sec"]

        L.append(
            f"### Scene {idx} — {_fmt_time(s['start_sec'])}~{_fmt_time(s['end_sec'])} "
            f"[{dur:.1f}s]  ·  flags: {', '.join('`'+t+'`' for t in tags)}"
        )
        L.append("")
        L.append(
            f"현재: **{s['manual_category']}**  (CLIP={conf:.2f}, "
            f"scene_name_auto=`{s['scene_name_auto']}`, source=`{source}`)"
        )
        L.append("")
        L.append(f"![scene {idx}]({frames_dir}/scene_{idx:03d}.jpg)")
        L.append("")
        if entry and entry.get("top_k"):
            tops = "  ·  ".join(
                f"{t['label']} ({t['prob']:.2f})" for t in entry["top_k"][:3]
            )
            L.append(f"- **CLIP top-3**: {tops}")
        else:
            L.append(
                f"- **CLIP top-1**: {auto_entry['scene_name_clip']} ({conf:.2f}) "
                "(top-3 은 clip_probs_rebuild 필요)"
            )
        at3 = (auto_entry.get("audio_top_tags") or [])[:3]
        if at3:
            astr = "  ·  ".join(
                f"{t[0]} ({t[1]:.2f})" if isinstance(t, (list, tuple)) else str(t)
                for t in at3
            )
            L.append(f"- **Audio top-3**: {astr}")
        L.append("")
        cands = _suggest_candidate_moods(auto_entry, entry, video_key, s["manual_category"])
        L.append("**이 씬의 가능한 mood (heuristic):**")
        if cands:
            for mood, reason in cands:
                L.append(f"- `{mood}` — {reason}")
        else:
            L.append("- _휴리스틱 후보 없음 — 현재값 유지 권장_")
        L.append("")
        L.append(
            f"→ JSON: `scenes[{idx}].manual_category` 수정 후 `reviewed: true`."
        )
        L.append("")

    # ── Section E ──────────────────────────────────────────
    L.append("## E. 최종 분포 시뮬레이션 (참고)")
    L.append("")
    L.append("현재 분포를 타깃 중앙값으로 단순 정렬한 시뮬레이션. 사용자가 다르게 배정해도 무방.")
    L.append("")
    sim = _simulate_target_dispersion(dist, targets, total)
    L.append("| Category | 현재 | 타깃 | 시뮬레이션 |")
    L.append("|---|---:|:---:|---:|")
    for c in SEVEN_CATEGORIES:
        t = targets.get(c)
        rng = f"{t[0]}~{t[1]}" if t else "-"
        L.append(f"| {c} | {dist.get(c, 0)} | {rng} | {sim.get(c, 0)} |")
    L.append("")
    L.append(f"_합계: 현재 {sum(dist.values())}  ·  시뮬레이션 {sum(sim.values())}_")
    L.append("")

    L.append("---")
    L.append(
        f"_Refresh_: `python -m model.autoEQ.inference.scene_labeler_review "
        f"--name {name} --refresh-guide`"
    )

    out_path = SCENE_LABELS_DIR / f"review_guide_{name}.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  ✓ guide → {out_path}  "
          f"(sections A-E, focused={len(focused)}, quick={quick})")
    return out_path


def run_for(name: str, *, skip_frames: bool = False,
            guide_only: bool = False) -> None:
    print(f"\n=== Review artefacts for trailer '{name}' ===")
    if guide_only:
        print("  ◉ --refresh-guide mode: skipping frames/report/manual regen")
        build_full_review_guide(name)
        return
    if not skip_frames:
        extract_frames(name)
    else:
        print("  ◉ Frame extraction skipped (--skip-frames)")
    build_report(name)
    build_manual_template(name)
    print_title_gap(name)
    build_full_review_guide(name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Day 6 scene-label review artefacts")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Short trailer name, e.g. topgun or lalaland")
    group.add_argument(
        "--all",
        action="store_true",
        help="Run for every *_auto.json under outputs/scene_labels/",
    )
    p.add_argument(
        "--skip-frames",
        action="store_true",
        help="Skip ffmpeg frame extraction (for quick report re-builds)",
    )
    p.add_argument(
        "--refresh-guide",
        action="store_true",
        help="Only regenerate review_guide_{name}.md from existing manual/auto/clip_probs "
             "(no frame extraction, no report rebuild, no manual template regen).",
    )
    args = p.parse_args(argv)

    SCENE_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        autos = sorted(SCENE_LABELS_DIR.glob("scene_labels_*_auto.json"))
        if not autos:
            print("No *_auto.json found.", file=sys.stderr)
            return 1
        for ap in autos:
            stem = ap.stem
            name = stem[len("scene_labels_"):-len("_auto")]
            run_for(
                name,
                skip_frames=args.skip_frames,
                guide_only=args.refresh_guide,
            )
    else:
        run_for(
            args.name,
            skip_frames=args.skip_frames,
            guide_only=args.refresh_guide,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
