"""A/B 주파수 비교 시각화 — 원본 vs MoodEQ 적용 결과.

원본 영상과 MoodEQ 가 적용된 영상의 오디오 주파수 스펙트럼을 나란히 비교해
한 장의 그림으로 저장하고 보여줍니다.

3개 subplot:
    (1) Full-range FFT spectrum (log freq, dB magnitude) — 두 신호 overlay
    (2) 10-band power (31.5Hz~16kHz) — 두 막대 나란히 + Δ dB 표시
    (3) Scene-wise EQ preset heatmap (timeline.json 있을 때)

기본값으로 실행하면 runs/demo_iron_sky 의 Iron Sky teaser 결과를 비교합니다.

Usage:
    venv/bin/python compare_spectra.py                          # 기본 demo 비교
    venv/bin/python compare_spectra.py --original X.mp4 --processed Y.mp4
    venv/bin/python compare_spectra.py --no-show                # PNG 만 저장
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 10-band octave-spaced EQ centers (matches infer_pseudo/eq_preset bands)
BAND_CENTERS_HZ = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]

# GEMS-7 mood → 고정 색상 (overlay 플롯에서 mood 구분용)
MOOD_COLORS = {
    "Tension":          "#d62728",  # 빨강
    "Sadness":          "#4b6cb7",  # 파랑
    "Tenderness":       "#ffa500",  # 주황
    "Wonder":           "#2ca02c",  # 초록
    "Peacefulness":     "#17becf",  # 청록
    "Power":            "#9467bd",  # 보라
    "JoyfulActivation": "#e377c2",  # 분홍
}


def extract_audio_to_wav(video: Path, dst: Path) -> tuple[np.ndarray, int]:
    """ffmpeg 로 영상에서 오디오를 WAV 로 추출 후 mono array 반환."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vn", "-acodec", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)
    audio, sr = sf.read(str(dst), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def fft_magnitude_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """FFT 매그니튜드 → dB (freqs, dB)."""
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.abs(X) / n * 2.0
    db = 20.0 * np.log10(mag + 1e-10)
    return freqs, db


def smooth_db(freqs: np.ndarray, db: np.ndarray, n_out: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """log-spaced frequency grid 로 부드럽게 resample (가독성)."""
    f_lo = max(20.0, freqs[1])
    f_hi = min(freqs[-1], 22050.0)
    grid = np.geomspace(f_lo, f_hi, n_out)
    # bin average in log space
    smoothed = np.interp(grid, freqs, db)
    # small rolling mean for visual clarity
    kernel = 5
    pad = np.r_[smoothed[:kernel // 2], smoothed, smoothed[-kernel // 2:]]
    smoothed = np.convolve(pad, np.ones(kernel) / kernel, mode="same")[kernel // 2: -kernel // 2]
    return grid, smoothed


def band_power_db(x: np.ndarray, sr: int) -> list[float]:
    """10 EQ 밴드 각각의 에너지 (dB). half-octave 좌우 경계."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    out: list[float] = []
    for fc in BAND_CENTERS_HZ:
        lo = fc / (2 ** 0.5)
        hi = fc * (2 ** 0.5)
        mask = (freqs >= lo) & (freqs < hi)
        p = float(np.mean(np.abs(X[mask]) ** 2)) if mask.any() else 1e-12
        out.append(10.0 * np.log10(p + 1e-12))
    return out


def plot_comparison(
    orig: np.ndarray,
    proc: np.ndarray,
    sr: int,
    out_png: Path,
    timeline: dict | None = None,
    show: bool = True,
    embed_heatmap: bool = False,
) -> None:
    """A/B 스펙트럼 비교 플롯.

    embed_heatmap=True 일 때만 하단에 scene heatmap 을 포함(기존 동작).
    기본은 False — heatmap 은 plot_scene_heatmap 로 별도 PNG 에 저장해 subplot 과밀을 해소.
    """
    include_heat = bool(timeline) and embed_heatmap
    n_rows = 4 if include_heat else 3
    height_ratios = [2.2, 1.0, 1.4] + ([1.4] if include_heat else [])
    fig = plt.figure(figsize=(13, 11 if include_heat else 9.0))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.45)

    # === (1a) Full-range FFT spectrum overlay ===
    f_o, db_o = fft_magnitude_db(orig, sr)
    f_p, db_p = fft_magnitude_db(proc, sr)
    g_o, s_o = smooth_db(f_o, db_o)
    g_p, s_p = smooth_db(f_p, db_p)

    ax1 = fig.add_subplot(gs[0])
    ax1.semilogx(g_o, s_o, label="Original",
                 color="#1f3a68", linestyle="-", linewidth=2.2, alpha=0.95)
    ax1.semilogx(g_p, s_p, label="MoodEQ Applied",
                 color="#d62728", linestyle="--", linewidth=2.0, alpha=0.95)
    ax1.set_xlim(20, 22050)
    aud_mask = (g_o >= 20) & (g_o <= 16000)
    all_vals = np.concatenate([s_o[aud_mask], s_p[aud_mask]])
    y_lo = float(np.percentile(all_vals, 2)) - 3.0
    y_hi = float(np.percentile(all_vals, 98)) + 3.0
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_ylabel("Magnitude (dB)", fontsize=12)
    ax1.set_title(
        f"(1) Full-range Spectrum  —  Original vs MoodEQ Applied"
        f"   (y-zoomed to audible range, {y_lo:.0f}~{y_hi:.0f} dB)",
        fontsize=13,
    )
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower left", fontsize=10)
    for fc in BAND_CENTERS_HZ:
        ax1.axvline(fc, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax1.tick_params(labelbottom=False, labelsize=10)

    # === (1b) Difference plot: MoodEQ − Original ===
    ax_diff = fig.add_subplot(gs[1], sharex=ax1)
    diff = s_p - s_o
    ax_diff.semilogx(g_o, diff, color="#2ca02c", linewidth=1.8)
    ax_diff.fill_between(g_o, 0, diff, where=(diff >= 0), color="#2ca02c", alpha=0.3, label="boost")
    ax_diff.fill_between(g_o, 0, diff, where=(diff < 0),  color="#d62728", alpha=0.3, label="cut")
    ax_diff.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    d_range = float(max(abs(diff.min()), abs(diff.max())))
    ax_diff.set_ylim(-d_range * 1.2 - 0.5, d_range * 1.2 + 0.5)
    ax_diff.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_diff.set_ylabel("Δ dB", fontsize=12)
    ax_diff.set_title(
        f"(1b) Difference  (MoodEQ − Original)   max boost = {diff.max():+.2f} dB,"
        f" max cut = {diff.min():+.2f} dB",
        fontsize=13,
    )
    ax_diff.grid(True, which="both", alpha=0.25)
    ax_diff.legend(loc="upper left", fontsize=10)
    ax_diff.tick_params(labelsize=10)
    for fc in BAND_CENTERS_HZ:
        ax_diff.axvline(fc, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

    # === (2) 10-band power bar chart with delta ===
    p_o_bands = band_power_db(orig, sr)
    p_p_bands = band_power_db(proc, sr)
    deltas = [p - o for o, p in zip(p_o_bands, p_p_bands)]

    ax2 = fig.add_subplot(gs[2])
    x = np.arange(len(BAND_CENTERS_HZ))
    w = 0.4
    ax2.bar(x - w / 2, p_o_bands, w, label="Original",
            color="#1f3a68", edgecolor="white", linewidth=0.5, alpha=0.95)
    ax2.bar(x + w / 2, p_p_bands, w, label="MoodEQ Applied",
            color="#d62728", edgecolor="white", linewidth=0.5, alpha=0.95,
            hatch="//")
    ymin = min(min(p_o_bands), min(p_p_bands)) - 3
    for xi, d in zip(x, deltas):
        sign = "+" if d >= 0 else ""
        color = "green" if d > 0.05 else ("crimson" if d < -0.05 else "gray")
        ax2.annotate(f"{sign}{d:.2f} dB", (xi, max(p_o_bands[xi], p_p_bands[xi]) + 0.3),
                     ha="center", fontsize=9, color=color, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(f) if f >= 1 else f}Hz" if f < 1000 else f"{int(f/1000)}kHz"
                          for f in BAND_CENTERS_HZ], fontsize=10)
    ax2.set_ylabel("Band Power (dB)", fontsize=12)
    ax2.set_title(
        f"(2) 10-band Power  —  mean Δ = "
        f"{np.mean(deltas):+.2f} dB, max Δ = {max(deltas, key=abs):+.2f} dB",
        fontsize=13,
    )
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.set_ylim(ymin, max(max(p_o_bands), max(p_p_bands)) + 5)

    # === (3) Scene-wise EQ heatmap (legacy embed only) ===
    if include_heat:
        ax3 = fig.add_subplot(gs[3])
        _draw_scene_heatmap(ax3, fig, timeline)

    fig.suptitle(
        f"MoodEQ A/B Spectrum Comparison  —  sr={sr}Hz, duration={len(orig)/sr:.1f}s",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def _draw_scene_heatmap(ax, fig, timeline: dict) -> None:
    """scene × band effective-gain heatmap 을 주어진 axis 에 그린다."""
    scenes = timeline.get("scenes", [])
    n_scenes = len(scenes)
    eff_matrix = np.zeros((len(BAND_CENTERS_HZ), n_scenes))
    mood_names: list[str] = []
    for j, sc in enumerate(scenes):
        bands = sc["eq_preset"]["effective_bands"]
        for i, b in enumerate(bands):
            eff_matrix[i, j] = b["gain_db"]
        mood_names.append(sc["mood"]["name"])

    vmax = float(np.abs(eff_matrix).max()) or 1.0
    im = ax.imshow(
        eff_matrix, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(-0.5, n_scenes - 0.5, -0.5, len(BAND_CENTERS_HZ) - 0.5),
    )
    ax.set_yticks(range(len(BAND_CENTERS_HZ)))
    ax.set_yticklabels(
        [f"{int(f)}Hz" if f < 1000 else f"{int(f/1000)}kHz" for f in BAND_CENTERS_HZ],
        fontsize=10,
    )
    ax.set_xticks(range(n_scenes))
    ax.set_xticklabels(
        [f"s{j}\n{t}\n{scenes[j]['start_sec']:.0f}s"
         for j, t in enumerate(mood_names)],
        rotation=0, fontsize=8,
    )
    ax.set_xlabel("Scene (idx / mood / start)", fontsize=12)
    ax.set_title(
        f"Effective EQ Gain per Scene  —  {n_scenes} scenes, "
        f"mood = {{{', '.join(sorted(set(mood_names)))}}}",
        fontsize=13,
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cbar.set_label("Gain (dB)", fontsize=11)


def plot_scene_heatmap(timeline: dict, out_png: Path, show: bool = True) -> None:
    """scene × band EQ-gain heatmap 단독 PNG."""
    n_scenes = len(timeline.get("scenes", []))
    width = max(10, min(24, 0.9 * n_scenes + 4))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    _draw_scene_heatmap(ax, fig, timeline)
    fig.suptitle("MoodEQ Scene-wise Effective EQ", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def _extract_thumbnail(video: Path, at_sec: float, dst: Path) -> None:
    """ffmpeg 으로 영상의 at_sec 지점에서 정지 프레임 1장을 JPEG 로 추출."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, at_sec):.3f}", "-i", str(video),
        "-vframes", "1", "-q:v", "2", str(dst),
    ]
    subprocess.run(cmd, check=True)


def _flatten_va_windows(
    timeline: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """timeline.scenes[*].windows[*] 를 시간순으로 flatten 해 (t, v, a) 반환.

    windows 가 비어있으면 scene-level va 로 fallback (is_window=False).
    """
    pts: list[tuple[float, float, float]] = []
    for sc in timeline.get("scenes", []):
        for w in sc.get("windows", []) or []:
            mid = 0.5 * (w["start_sec"] + w["end_sec"])
            pts.append((mid, float(w["valence"]), float(w["arousal"])))
    is_window = bool(pts)
    if not is_window:
        for sc in timeline.get("scenes", []):
            mid = 0.5 * (sc["start_sec"] + sc["end_sec"])
            va = sc.get("va", {})
            pts.append((mid, float(va.get("valence", 0.0)), float(va.get("arousal", 0.0))))
    pts.sort(key=lambda p: p[0])
    t = np.array([p[0] for p in pts], dtype=np.float32)
    v = np.array([p[1] for p in pts], dtype=np.float32)
    a = np.array([p[2] for p in pts], dtype=np.float32)
    return t, v, a, is_window


def plot_va_timeline(
    video_path: Path,
    timeline: dict,
    out_png: Path,
    show: bool = True,
    dark: bool = True,
) -> None:
    """LIRIS-ACCEDE 스타일: 좌 영상 프레임 | 우 시간축 Valence/Arousal 곡선.

    이미지 레퍼런스 재현: 검은 배경, 큰 초록색 라벨("Input Movie"/"Output"),
    파란 Valence / 초록 Arousal 두 곡선, y∈[-1,1].
    """
    duration = float(timeline.get("metadata", {}).get("duration_sec") or 0.0)
    if duration <= 0.0:
        last = timeline["scenes"][-1] if timeline.get("scenes") else {"end_sec": 0.0}
        duration = float(last.get("end_sec", 0.0))

    t, v, a, is_window = _flatten_va_windows(timeline)

    bg = "black" if dark else "white"
    fg = "white" if dark else "black"
    label_green = "#2ca02c"

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.12},
    )
    fig.patch.set_facecolor(bg)

    # === 좌: 영상 썸네일 ===
    ax_left.set_facecolor(bg)
    with tempfile.TemporaryDirectory() as td:
        thumb = Path(td) / "thumb.jpg"
        try:
            _extract_thumbnail(video_path, duration / 2.0, thumb)
            from matplotlib.image import imread
            img = imread(str(thumb))
            ax_left.imshow(img)
        except Exception as e:
            ax_left.text(0.5, 0.5, f"(thumbnail unavailable)\n{e}",
                         color=fg, ha="center", va="center", transform=ax_left.transAxes)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_title("Input Movie", color=label_green, fontsize=20,
                      fontweight="bold", pad=12, loc="center", y=-0.12)

    # === 우: V/A 시계열 ===
    ax_right.set_facecolor(bg)
    ax_right.plot(t, v, color="#3a6fe0", linewidth=2.2, label="Valence")
    ax_right.plot(t, a, color="#2ca02c", linewidth=2.2, label="Arousal")
    ax_right.axhline(0, color=fg, linewidth=0.6, alpha=0.35)
    ax_right.set_xlim(0, max(duration, float(t[-1]) if len(t) else 1.0))
    ax_right.set_ylim(-1.0, 1.0)
    ax_right.set_xlabel("Time-stamps (sec)", color=fg, fontsize=14)
    ax_right.set_ylabel("Values", color=fg, fontsize=14)
    ax_right.tick_params(colors=fg, labelsize=11)
    for spine in ax_right.spines.values():
        spine.set_color(fg)
    ax_right.grid(True, color=fg, linestyle=":", alpha=0.25)
    leg = ax_right.legend(loc="upper right", fontsize=11, facecolor=bg,
                          edgecolor=fg, labelcolor=fg)
    leg.get_frame().set_alpha(0.7)
    ax_right.set_title("Output", color=label_green, fontsize=20,
                       fontweight="bold", pad=12, y=-0.22)

    mode_tag = "window-level" if is_window else "scene-level"
    fig.suptitle(
        f"MoodEQ V/A Timeline  —  {video_path.name}  ({duration:.1f}s, {mode_tag}, n={len(t)})",
        color=fg, fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def plot_scene_overlay(
    orig: np.ndarray,
    proc: np.ndarray,
    sr: int,
    timeline: dict,
    out_png: Path,
    show: bool = True,
) -> None:
    """모든 scene 의 difference spectrum 을 mood 색상으로 한 장에 overlay.

    Scene 마다 그 구간의 원본·처리 오디오를 각각 FFT → smooth → 두 선의 차이
    (proc − orig) 를 그린다. 같은 mood 끼리는 색이 같아서 mood 별 EQ 방향 경향이
    드러난다.
    """
    fig, (ax_ov, ax_mean) = plt.subplots(
        2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35}
    )

    mood_curves: dict[str, list[np.ndarray]] = {}
    grid_ref: np.ndarray | None = None
    plotted_moods = set()

    for sc in timeline["scenes"]:
        s = int(round(sc["start_sec"] * sr))
        e = int(round(sc["end_sec"] * sr))
        if e - s < 2048:
            continue  # FFT 에 필요한 최소 샘플 수 확보
        o = orig[s:e]
        p = proc[s:e]
        f_o, db_o = fft_magnitude_db(o, sr)
        f_p, db_p = fft_magnitude_db(p, sr)
        g, s_o = smooth_db(f_o, db_o)
        _, s_p = smooth_db(f_p, db_p)
        diff = s_p - s_o
        grid_ref = g if grid_ref is None else grid_ref

        mood = sc["mood"]["name"]
        color = MOOD_COLORS.get(mood, "#888888")
        label = f"{mood}" if mood not in plotted_moods else None
        plotted_moods.add(mood)
        ax_ov.semilogx(
            g, diff, color=color, alpha=0.55, linewidth=1.4, label=label,
        )
        mood_curves.setdefault(mood, []).append(diff)

    ax_ov.axhline(0, color="black", linewidth=0.9, alpha=0.6)
    ax_ov.set_xlim(20, 22050)
    ax_ov.set_xlabel("Frequency (Hz)")
    ax_ov.set_ylabel("Δ dB  (MoodEQ − Original)")
    ax_ov.set_title(
        f"All-scene Difference Spectrum  —  {len(timeline['scenes'])} scenes, "
        f"{len(plotted_moods)} moods (colored)"
    )
    ax_ov.grid(True, which="both", alpha=0.3)
    ax_ov.legend(loc="upper left", fontsize=9)
    for fc in BAND_CENTERS_HZ:
        ax_ov.axvline(fc, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)

    # === mood 별 평균 diff ===
    if grid_ref is not None:
        for mood, curves in mood_curves.items():
            avg = np.mean(np.stack(curves), axis=0)
            ax_mean.semilogx(
                grid_ref, avg,
                color=MOOD_COLORS.get(mood, "#888888"),
                linewidth=2.0, label=f"{mood} (n={len(curves)})",
            )
    ax_mean.axhline(0, color="black", linewidth=0.9, alpha=0.6)
    ax_mean.set_xlim(20, 22050)
    ax_mean.set_xlabel("Frequency (Hz)")
    ax_mean.set_ylabel("mean Δ dB")
    ax_mean.set_title("Per-mood mean Δ spectrum (bold)")
    ax_mean.grid(True, which="both", alpha=0.3)
    ax_mean.legend(loc="upper left", fontsize=9)
    for fc in BAND_CENTERS_HZ:
        ax_mean.axvline(fc, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)

    fig.suptitle(
        f"MoodEQ Scene-level Difference Overview  —  sr={sr}Hz",
        fontsize=13, fontweight="bold", y=1.0,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def plot_three_way(
    orig: np.ndarray,
    proc1x: np.ndarray,
    proc3x: np.ndarray,
    sr: int,
    out_png: Path,
    show: bool = True,
) -> None:
    """Original / 1x MoodEQ (spec) / 3x exaggerated (demo) 3-way 비교.

    3 rows:
      (1) Full-range spectrum overlay  (3 선: Original / 1x / 3x)
      (2) 10-band power bar chart      (3 그룹: Orig / 1x / 3x)
      (3) Difference spectrum          (2 선: 1x-Orig / 3x-Orig)
    """
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.2, 1.4], hspace=0.45)

    # === (1) Full-range overlay ===
    f_o, db_o = fft_magnitude_db(orig, sr)
    f_1, db_1 = fft_magnitude_db(proc1x, sr)
    f_3, db_3 = fft_magnitude_db(proc3x, sr)
    g, s_o = smooth_db(f_o, db_o)
    _, s_1 = smooth_db(f_1, db_1)
    _, s_3 = smooth_db(f_3, db_3)

    ax1 = fig.add_subplot(gs[0])
    ax1.semilogx(g, s_o, label="Original", color="#1f77b4", linewidth=1.5, alpha=0.85)
    ax1.semilogx(g, s_1, label="MoodEQ 1× (spec)", color="#d62728", linewidth=1.5, alpha=0.85)
    ax1.semilogx(g, s_3, label="MoodEQ 3× (demo, no dialog protection)",
                 color="#9467bd", linewidth=1.8, alpha=0.9)
    ax1.set_xlim(20, 22050)
    aud = (g >= 20) & (g <= 16000)
    all_vals = np.concatenate([s_o[aud], s_1[aud], s_3[aud]])
    y_lo = float(np.percentile(all_vals, 2)) - 3.0
    y_hi = float(np.percentile(all_vals, 98)) + 3.0
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("(1) Full-range Spectrum — Original vs MoodEQ 1× vs 3×")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower left")
    for fc in BAND_CENTERS_HZ:
        ax1.axvline(fc, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

    # === (2) 10-band grouped bars ===
    p_o = band_power_db(orig, sr)
    p_1 = band_power_db(proc1x, sr)
    p_3 = band_power_db(proc3x, sr)
    d_1 = [a - b for a, b in zip(p_1, p_o)]
    d_3 = [a - b for a, b in zip(p_3, p_o)]

    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(BAND_CENTERS_HZ))
    w = 0.27
    ax2.bar(x - w, p_o, w, label="Original", color="#1f77b4", alpha=0.85)
    ax2.bar(x,      p_1, w, label="1× (spec)", color="#d62728", alpha=0.85)
    ax2.bar(x + w,  p_3, w, label="3× (demo)", color="#9467bd", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{int(f)}Hz" if f < 1000 else f"{int(f/1000)}kHz" for f in BAND_CENTERS_HZ]
    )
    ax2.set_ylabel("Band Power (dB)")
    ax2.set_title(
        f"(2) 10-band Power  —  mean |Δ| 1×={np.mean(np.abs(d_1)):.2f}dB, "
        f"3×={np.mean(np.abs(d_3)):.2f}dB"
    )
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    # === (3) Difference spectra ===
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    diff_1 = s_1 - s_o
    diff_3 = s_3 - s_o
    ax3.semilogx(g, diff_1, color="#d62728", linewidth=1.5, label="1× Δ (spec)", alpha=0.85)
    ax3.semilogx(g, diff_3, color="#9467bd", linewidth=1.8, label="3× Δ (demo)", alpha=0.9)
    ax3.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax3.axhspan(-1.0, 1.0, color="gray", alpha=0.12, label="±1 dB (sub-JND)")
    ax3.axhspan(-3.0, -1.0, color="#2ca02c", alpha=0.05)
    ax3.axhspan( 1.0,  3.0, color="#2ca02c", alpha=0.05)
    d_max = float(max(abs(diff_1).max(), abs(diff_3).max()))
    ax3.set_ylim(-d_max * 1.15 - 0.5, d_max * 1.15 + 0.5)
    ax3.set_xlim(20, 22050)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Δ dB")
    ax3.set_title(
        f"(3) Difference Spectra   —  1× max |Δ|={max(abs(diff_1)):.2f}dB, "
        f"3× max |Δ|={max(abs(diff_3)):.2f}dB   (JND band shaded gray)"
    )
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(loc="upper left", fontsize=9)
    for fc in BAND_CENTERS_HZ:
        ax3.axvline(fc, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.suptitle(
        f"MoodEQ 3-way Comparison (sr={sr}Hz, duration={len(orig)/sr:.1f}s)  —  "
        f"Spec 1× vs Demo 3× (no-dialog-protection)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(
    orig: np.ndarray,
    proc: np.ndarray,
    sr: int,
    timeline: dict | None = None,
) -> None:
    print(f"\n=== Numeric summary ===")
    print(f"  sample rate     : {sr} Hz")
    print(f"  duration        : {len(orig)/sr:.2f} s")
    print(f"  samples (orig)  : {len(orig):,}")
    print(f"  samples (proc)  : {len(proc):,}")
    print(f"  length match    : {'YES' if len(orig)==len(proc) else 'NO (A/V sync risk)'}")

    p_o = band_power_db(orig, sr)
    p_p = band_power_db(proc, sr)
    print(f"\n  {'band':>8}  {'orig dB':>9}  {'proc dB':>9}  {'delta':>8}")
    for fc, o, p in zip(BAND_CENTERS_HZ, p_o, p_p):
        lab = f"{int(fc)}Hz" if fc < 1000 else f"{int(fc/1000)}kHz"
        d = p - o
        print(f"  {lab:>8}  {o:>+9.2f}  {p:>+9.2f}  {d:>+8.2f}")

    if timeline:
        scenes = timeline["scenes"]
        print(f"\n  timeline scenes : {len(scenes)}")
        moods = {}
        for sc in scenes:
            moods[sc["mood"]["name"]] = moods.get(sc["mood"]["name"], 0) + 1
        print(f"  mood counts     : {moods}")
        g = timeline["global"]
        print(f"  global mean V/A : {g['mean_va']['valence']:+.3f} / {g['mean_va']['arousal']:+.3f}")
        print(f"  avg dialog dens : {g['avg_dialogue_density']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description="A/B spectrum comparison — Original vs MoodEQ")
    p.add_argument(
        "--original", default=str(
            Path.home() / "Downloads/LIRIS-ACCEDE-movies/movies/Iron_Sky_Teaser_3_We_Come_In_Peace.mp4"
        ),
    )
    p.add_argument("--processed", default="runs/demo_iron_sky/iron_sky_eq_applied.mp4")
    p.add_argument("--timeline", default="runs/demo_iron_sky/timeline.json")
    p.add_argument("--output", default="runs/demo_iron_sky/spectrum_compare.png")
    p.add_argument("--no-show", action="store_true", help="PNG 만 저장, GUI 창 띄우지 않음")
    p.add_argument(
        "--scene", type=int, default=None,
        help="특정 scene idx 만 확대 비교 (timeline.json 필요)",
    )
    p.add_argument(
        "--overlay", action="store_true",
        help="모든 scene 의 Δ spectrum 을 mood 색으로 overlay (timeline.json 필요)",
    )
    p.add_argument(
        "--three-way", action="store_true",
        help="Original / 1x MoodEQ (spec) / 3x (demo) 3-way 비교",
    )
    p.add_argument(
        "--processed-demo", default="runs/demo_iron_sky/iron_sky_eq_demo3x.mp4",
        help="3-way 모드에서 3x exaggerated mp4 경로",
    )
    p.add_argument(
        "--va-timeline", action="store_true",
        help="LIRIS-ACCEDE 스타일: 영상 프레임 + 시간축 V/A 곡선 (timeline.json 필요, processed 불필요)",
    )
    p.add_argument(
        "--no-split", action="store_true",
        help="기본 모드에서 heatmap 을 별도 PNG 가 아닌 한 그림에 포함 (기존 동작)",
    )
    p.add_argument(
        "--light", action="store_true",
        help="--va-timeline 에서 검은 배경 대신 흰 배경 사용",
    )
    args = p.parse_args()

    orig_path = Path(args.original)
    if not orig_path.exists():
        print(f"[error] original not found: {orig_path}", file=sys.stderr)
        return 1

    timeline = None
    if args.timeline and Path(args.timeline).exists():
        timeline = json.loads(Path(args.timeline).read_text())

    # === V/A timeline: processed audio 가 필요 없음. 빠르게 분기 ===
    if args.va_timeline:
        if not timeline:
            print("[error] --va-timeline requires timeline.json", file=sys.stderr)
            return 1
        out = Path(args.output)
        if out.name == "spectrum_compare.png":
            out = out.with_name("va_timeline.png")
        plot_va_timeline(orig_path, timeline, out,
                         show=not args.no_show, dark=not args.light)
        return 0

    # === 이하는 오디오 A/B 비교 모드들 — processed 필수 ===
    proc_path = Path(args.processed)
    if not proc_path.exists():
        print(f"[error] processed not found: {proc_path}", file=sys.stderr)
        return 1

    print(f"[audio] extracting from {orig_path.name} and {proc_path.name}...")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        orig, sr_o = extract_audio_to_wav(orig_path, td_path / "orig.wav")
        proc, sr_p = extract_audio_to_wav(proc_path, td_path / "proc.wav")

    if sr_o != sr_p:
        print(f"[warn] sample rate differs: {sr_o} vs {sr_p} — resampling processed")
        from scipy.signal import resample_poly
        proc = resample_poly(proc, sr_o, sr_p)
    sr = sr_o

    # equal-length clip for fair comparison
    n = min(len(orig), len(proc))
    orig = orig[:n]
    proc = proc[:n]

    print_summary(orig, proc, sr, timeline)

    # === Mode dispatch ===
    if args.three_way:
        demo_path = Path(args.processed_demo)
        if not demo_path.exists():
            print(f"[error] --processed-demo not found: {demo_path}", file=sys.stderr)
            return 1
        print(f"[audio] extracting 3x demo from {demo_path.name}...")
        with tempfile.TemporaryDirectory() as td:
            proc3x, sr3 = extract_audio_to_wav(demo_path, Path(td) / "demo.wav")
        assert sr3 == sr, "sample rate mismatch in 3x demo"
        n3 = min(len(orig), len(proc), len(proc3x))
        out = Path(args.output)
        if out.name == "spectrum_compare.png":
            out = out.with_name("spectrum_compare_threeway.png")
        plot_three_way(orig[:n3], proc[:n3], proc3x[:n3], sr, out, show=not args.no_show)
        return 0

    if args.overlay:
        if not timeline:
            print("[error] --overlay requires timeline.json", file=sys.stderr)
            return 1
        out = Path(args.output)
        if out.name == "spectrum_compare.png":
            out = out.with_name("spectrum_compare_overlay.png")
        plot_scene_overlay(orig, proc, sr, timeline, out, show=not args.no_show)
        return 0

    if args.scene is not None:
        if not timeline:
            print("[error] --scene requires timeline.json", file=sys.stderr)
            return 1
        scenes = timeline["scenes"]
        if not 0 <= args.scene < len(scenes):
            print(f"[error] scene idx must be in [0, {len(scenes)})", file=sys.stderr)
            return 1
        sc = scenes[args.scene]
        s_start = int(round(sc["start_sec"] * sr))
        s_end = int(round(sc["end_sec"] * sr))
        o_slice = orig[s_start:s_end]
        p_slice = proc[s_start:s_end]
        sub_timeline = {
            "scenes": [sc],
            "global": timeline.get("global", {}),
            "metadata": timeline.get("metadata", {}),
            "config": timeline.get("config", {}),
        }
        out = Path(args.output)
        mood_tag = sc["mood"]["name"]
        if out.name == "spectrum_compare.png":
            out = out.with_name(f"spectrum_compare_scene{args.scene:02d}_{mood_tag}.png")
        print(f"[scene {args.scene}] {mood_tag}  [{sc['start_sec']:.1f}s, {sc['end_sec']:.1f}s]  "
              f"V={sc['va']['valence']:+.3f}  A={sc['va']['arousal']:+.3f}")
        plot_comparison(o_slice, p_slice, sr, out, timeline=sub_timeline,
                        show=not args.no_show, embed_heatmap=True)
        return 0

    # Default: full-track comparison
    out_main = Path(args.output)
    if args.no_split:
        plot_comparison(orig, proc, sr, out_main, timeline=timeline,
                        show=not args.no_show, embed_heatmap=True)
    else:
        plot_comparison(orig, proc, sr, out_main, timeline=timeline,
                        show=not args.no_show, embed_heatmap=False)
        if timeline:
            heat_out = out_main.with_name(
                out_main.stem + "_heatmap" + out_main.suffix
            )
            plot_scene_heatmap(timeline, heat_out, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
