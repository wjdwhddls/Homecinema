"""eq_visualizer.py — EQ 변화 시각화 4종.

작업 10-3 (Day 4~5):
1. plot_eq_heatmap — 씬 × 밴드 히트맵 (발표용 핵심)
2. plot_eq_timeline — 시간축 EQ 게인 변화
3. plot_spectrogram_comparison — 원본 vs EQ 적용본 + 차이맵
4. plot_dialogue_protection_effect — density vs 게인 산점도
"""

from __future__ import annotations

import platform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .eq_engine import (
    BAND_FREQS,
    EQ_PRESETS,
    PRESET_VERSIONS,
    VOICE_BANDS_IDX,
    compute_effective_eq,
    manual_label_to_probs,
)
from .paths import EQ_VIZ_DIR, ensure_dirs


# ────────────────────────────────────────────────────────
# 한글 폰트 설정
# ────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:  # Linux/Ubuntu
    # NanumGothic이 설치되어 있으면 사용
    try:
        from matplotlib import font_manager
        for fp in font_manager.findSystemFonts(fontpaths=None, fontext="ttf"):
            if "Nanum" in fp:
                plt.rcParams["font.family"] = "NanumGothic"
                break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


BAND_LABELS = ["31Hz", "63Hz", "125Hz", "250Hz", "500Hz",
               "1kHz", "2kHz", "4kHz", "8kHz", "16kHz"]

MOOD_COLORS = {
    "Tension":           "#ef4444",
    "Sadness":           "#6366f1",
    "Peacefulness":      "#22c55e",
    "Power":             "#f97316",
    "Wonder":            "#a855f7",
    "Tenderness":        "#ec4899",
    "Joyful Activation": "#eab308",
}


# ────────────────────────────────────────────────────────
# 1. EQ 히트맵 (씬 × 밴드)
# ────────────────────────────────────────────────────────
def plot_eq_heatmap(scenes_eq: list[dict], title: str = "EQ 게인 히트맵", save=None):
    """씬별 10밴드 EQ 게인을 히트맵으로 시각화."""
    n_scenes = len(scenes_eq)
    gains_matrix = np.array([s["effective_gains"] for s in scenes_eq])

    fig, (ax_heat, ax_mood) = plt.subplots(
        2, 1, figsize=(max(14, n_scenes * 0.5), 7),
        gridspec_kw={"height_ratios": [5, 1]}, sharex=True,
    )

    vmax = max(abs(gains_matrix.min()), abs(gains_matrix.max()), 1.0)
    im = ax_heat.imshow(
        gains_matrix.T, aspect="auto", cmap=plt.cm.RdBu_r,
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax_heat.set_yticks(range(10))
    ax_heat.set_yticklabels(BAND_LABELS, fontsize=9)
    ax_heat.set_ylabel("주파수 밴드")
    ax_heat.set_title(title, fontweight="bold", fontsize=13)

    # 셀에 dB 값 표시
    for i in range(n_scenes):
        for j in range(10):
            val = gains_matrix[i, j]
            color = "white" if abs(val) > vmax * 0.4 else "gray"
            if abs(val) > 0.05:
                ax_heat.text(i, j, f"{val:+.1f}", ha="center", va="center",
                             fontsize=7, color=color, fontweight="bold")

    # 대사 보호 밴드 강조
    for b in VOICE_BANDS_IDX:
        ax_heat.axhline(y=b - 0.5, color="lime", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_heat.axhline(y=b + 0.5, color="lime", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_heat.text(-0.8, 6, "대사\n보호", fontsize=7, color="lime", ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label("EQ 게인 (dB)", fontsize=9)

    # 씬 감정 스트립
    for i, s in enumerate(scenes_eq):
        mood = s.get("mood", "Unknown")
        scene_name = s.get("scene_name", "")
        color = MOOD_COLORS.get(mood, "#888")
        ax_mood.add_patch(Rectangle((i - 0.5, 0), 1, 1,
                                     facecolor=color, edgecolor="white",
                                     linewidth=0.5, alpha=0.8))
        display_label = scene_name if scene_name else mood[:4]
        ax_mood.text(i, 0.5, f'{display_label}\n{s["start_sec"]:.0f}s',
                     ha="center", va="center", fontsize=6,
                     color="white", fontweight="bold")

    ax_mood.set_xlim(-0.5, n_scenes - 0.5)
    ax_mood.set_ylim(0, 1)
    ax_mood.set_yticks([])
    ax_mood.set_xlabel("씬 번호")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  💾 {save} 저장")
    plt.close(fig)


# ────────────────────────────────────────────────────────
# 2. EQ 시계열
# ────────────────────────────────────────────────────────
def plot_eq_timeline(
    scenes_eq: list[dict],
    bands_to_show: list[int] | None = None,
    title: str = "EQ 게인 시계열",
    save=None,
):
    """시간축(초)에 따른 밴드별 EQ 게인 변화."""
    if bands_to_show is None:
        bands_to_show = list(range(10))

    fig, (ax_eq, ax_mood) = plt.subplots(
        2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [4, 1]}, sharex=True,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for band_idx in bands_to_show:
        times, gains = [], []
        for s in scenes_eq:
            times.extend([s["start_sec"], s["end_sec"]])
            gains.extend([s["effective_gains"][band_idx]] * 2)
        ax_eq.plot(times, gains, color=colors[band_idx],
                   linewidth=1.5, label=BAND_LABELS[band_idx], alpha=0.85)
        if band_idx in VOICE_BANDS_IDX:
            ax_eq.plot(times, gains, color=colors[band_idx],
                       linewidth=2.5, alpha=0.3)

    total_dur = scenes_eq[-1]["end_sec"]
    ax_eq.axhline(y=0, color="white", linewidth=1.5, alpha=0.4)
    ax_eq.text(total_dur * 0.01, 0.15, "Original (0dB)", fontsize=8, color="gray")
    ax_eq.set_ylabel("EQ 게인 (dB)")
    ax_eq.set_title(title, fontweight="bold", fontsize=13)
    ax_eq.legend(loc="upper right", fontsize=8, ncol=len(bands_to_show), framealpha=0.7)
    ax_eq.set_ylim(-5.5, 5.5)
    ax_eq.grid(True, alpha=0.2)
    ax_eq.set_facecolor("#1a1a2e")

    for s in scenes_eq:
        mood = s.get("mood", "Unknown")
        scene_name = s.get("scene_name", "")
        color = MOOD_COLORS.get(mood, "#888")
        ax_eq.axvline(x=s["start_sec"], color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
        ax_mood.barh(0, s["end_sec"] - s["start_sec"], left=s["start_sec"],
                     height=0.8, color=color, edgecolor="black",
                     linewidth=0.3, alpha=0.8)
        display_label = scene_name if scene_name else mood[:5]
        ax_mood.text(s["start_sec"] + (s["end_sec"] - s["start_sec"]) / 2, 0,
                     display_label, ha="center", va="center", fontsize=7,
                     color="white", fontweight="bold")

    ax_mood.set_xlim(0, total_dur)
    ax_mood.set_ylim(-0.5, 0.5)
    ax_mood.set_yticks([])
    ax_mood.set_xlabel("시간 (초)")
    ax_mood.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
        print(f"  💾 {save} 저장")
    plt.close(fig)


# ────────────────────────────────────────────────────────
# 3. 스펙트로그램 Before/After
# ────────────────────────────────────────────────────────
def plot_spectrogram_comparison(
    original_wav, processed_wav,
    title: str = "스펙트로그램 비교",
    time_range: tuple[float, float] | None = None,
    save=None,
):
    """원본 vs EQ 적용본 스펙트로그램 + 차이 맵."""
    import librosa
    import librosa.display

    y_orig, sr = librosa.load(str(original_wav), sr=None, mono=True)
    y_proc, _ = librosa.load(str(processed_wav), sr=sr, mono=True)

    if time_range:
        s, e = int(time_range[0] * sr), int(time_range[1] * sr)
        y_orig, y_proc = y_orig[s:e], y_proc[s:e]

    n_fft, hop = 4096, 512
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    S_orig = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop)), ref=np.max
    )
    img1 = librosa.display.specshow(
        S_orig, sr=sr, hop_length=hop, x_axis="time", y_axis="log",
        ax=axes[0], cmap="magma", vmin=-80, vmax=0,
    )
    axes[0].set_title("Original (EQ 미적용)", fontweight="bold")
    axes[0].set_ylabel("주파수 (Hz)")
    fig.colorbar(img1, ax=axes[0], format="%+.0f dB", shrink=0.8)

    S_proc = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_proc, n_fft=n_fft, hop_length=hop)), ref=np.max
    )
    img2 = librosa.display.specshow(
        S_proc, sr=sr, hop_length=hop, x_axis="time", y_axis="log",
        ax=axes[1], cmap="magma", vmin=-80, vmax=0,
    )
    axes[1].set_title("EQ 적용 후", fontweight="bold")
    axes[1].set_ylabel("주파수 (Hz)")
    fig.colorbar(img2, ax=axes[1], format="%+.0f dB", shrink=0.8)

    min_len = min(S_orig.shape[1], S_proc.shape[1])
    S_diff = S_proc[:, :min_len] - S_orig[:, :min_len]
    img3 = librosa.display.specshow(
        S_diff, sr=sr, hop_length=hop, x_axis="time", y_axis="log",
        ax=axes[2], cmap="RdBu_r", vmin=-10, vmax=10,
    )
    axes[2].set_title("차이 (EQ − Original): 빨강=boost, 파랑=cut", fontweight="bold")
    axes[2].set_ylabel("주파수 (Hz)")
    axes[2].set_xlabel("시간 (초)")
    fig.colorbar(img3, ax=axes[2], format="%+.0f dB", shrink=0.8)

    for ax in axes:
        for freq in BAND_FREQS:
            ax.axhline(y=freq, color="cyan", linewidth=0.3, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  💾 {save} 저장")
    plt.close(fig)


# ────────────────────────────────────────────────────────
# 4. 대사 보호 효과 산점도
# ────────────────────────────────────────────────────────
def plot_dialogue_protection_effect(
    scenes_eq: list[dict], title: str = "대사 보호 효과", save=None
):
    """B6/B7/B8 게인 vs dialogue density 산점도."""
    densities = [s.get("density", 0) for s in scenes_eq]
    moods = [s.get("mood", "") for s in scenes_eq]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    band_names = ["B6 (1kHz)", "B7 (2kHz)", "B8 (4kHz)"]

    for ax, band_idx, bname in zip(axes, VOICE_BANDS_IDX, band_names):
        gains = [s["effective_gains"][band_idx] for s in scenes_eq]
        for d, g, m in zip(densities, gains, moods):
            color = MOOD_COLORS.get(m, "#888")
            ax.scatter(d, g, c=color, s=60, alpha=0.7, edgecolors="white", linewidths=0.5)

        if len(densities) > 2:
            z = np.polyfit(densities, gains, 1)
            x_line = np.linspace(0, max(max(densities), 0.01), 50)
            ax.plot(x_line, np.poly1d(z)(x_line), "--", color="white",
                    alpha=0.4, linewidth=1.5)

        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_xlabel("Dialogue Density")
        ax.set_title(bname, fontweight="bold")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("EQ 게인 (dB)")
    plt.suptitle(title + "\n(대사 비율이 높을수록 대사 대역 EQ가 줄어듦)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  💾 {save} 저장")
    plt.close(fig)


# ────────────────────────────────────────────────────────
# 데이터 준비 헬퍼
# ────────────────────────────────────────────────────────
def prepare_scenes_eq_for_viz(
    manual_mood_labels: list,
    version: str = "v3_1",
) -> list[dict]:
    """MANUAL_MOOD_LABELS(dict 형식) → 시각화용 scenes_eq 데이터.

    dict 형식 (v5+) + 튜플 형식 (v4 호환) 모두 지원.
    """
    presets = PRESET_VERSIONS[version]
    scenes_eq = []
    for label in manual_mood_labels:
        if isinstance(label, dict):
            start = label["start"]
            end = label["end"]
            scene_name = label.get("scene_name", "Unknown")
            mood = label["mood"]
            prob = label["prob"]
            density = label["density"]
        else:
            start, end, mood, prob, density = label
            scene_name = None

        probs = manual_label_to_probs(mood, prob)
        gains = compute_effective_eq(
            probs, density, alpha_d=0.5, intensity=1.0, presets=presets
        )
        scenes_eq.append({
            "start_sec": start,
            "end_sec": end,
            "scene_name": scene_name,
            "mood": mood,
            "density": density,
            "effective_gains": gains,
        })
    return scenes_eq


def plot_v31_vs_v32_comparison(
    video_name: str, manual_mood_labels: list, save=None
):
    """V3.1 vs V3.2 히트맵을 나란히 비교 — 발표 핵심 슬라이드."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, version, vtitle in zip(
        axes, ["v3_1", "v3_2"],
        ["V3.1 Baseline (±3dB)", "V3.2 Dramatic (±4dB)"],
    ):
        scenes_eq = prepare_scenes_eq_for_viz(manual_mood_labels, version)
        gains_matrix = np.array([s["effective_gains"] for s in scenes_eq])

        im = ax.imshow(gains_matrix.T, aspect="auto", cmap=plt.cm.RdBu_r,
                       vmin=-4.5, vmax=4.5, interpolation="nearest")
        ax.set_yticks(range(10))
        ax.set_yticklabels(BAND_LABELS, fontsize=9)
        ax.set_xlabel("씬 번호")
        ax.set_title(vtitle, fontweight="bold", fontsize=12)

        for i in range(gains_matrix.shape[0]):
            for j in range(10):
                val = gains_matrix[i, j]
                if abs(val) > 0.1:
                    ax.text(i, j, f"{val:+.1f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(val) > 1.5 else "gray",
                            fontweight="bold")

    axes[0].set_ylabel("주파수 밴드")
    fig.colorbar(im, ax=axes, shrink=0.7, label="EQ 게인 (dB)")
    plt.suptitle(
        f"{video_name} — V3.1 vs V3.2 비교\n"
        "(V3.2가 빨강/파랑 대비가 더 강해야 정상)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  💾 {save} 저장")
    plt.close(fig)


# ────────────────────────────────────────────────────────
# 일괄 시각화 (두 영상 × 두 버전)
# ────────────────────────────────────────────────────────
def visualize_all(manual_mood_labels_dict: dict) -> None:
    """두 영상 × 두 버전 = 12개 + 비교 2개 = 14개 그래프 일괄 생성."""
    ensure_dirs()

    for video_name, labels in manual_mood_labels_dict.items():
        if not labels:
            continue
        for version in ["v3_1", "v3_2"]:
            scenes_eq = prepare_scenes_eq_for_viz(labels, version)

            print(f"\n--- {video_name} / {version} ---")
            plot_eq_heatmap(
                scenes_eq, title=f"{video_name} — {version.upper()} EQ 히트맵",
                save=EQ_VIZ_DIR / f"eq_heatmap_{video_name}_{version}.png",
            )
            plot_eq_timeline(
                scenes_eq, bands_to_show=[0, 1, 5, 6, 7, 9],
                title=f"{video_name} — {version.upper()} EQ 시계열",
                save=EQ_VIZ_DIR / f"eq_timeline_{video_name}_{version}.png",
            )
            plot_dialogue_protection_effect(
                scenes_eq, title=f"{video_name} — {version.upper()} 대사 보호",
                save=EQ_VIZ_DIR / f"eq_protection_{video_name}_{version}.png",
            )

        # V3.1 vs V3.2 비교
        plot_v31_vs_v32_comparison(
            video_name, labels,
            save=EQ_VIZ_DIR / f"eq_comparison_{video_name}.png",
        )

    print("\n🎉 시각화 완료")


if __name__ == "__main__":
    print("eq_visualizer는 import해서 사용하세요:")
    print("  from model.autoEQ.inference.eq_visualizer import visualize_all")
    print("  visualize_all(MANUAL_MOOD_LABELS)")
