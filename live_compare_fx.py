"""Live A/B/C 3-way 비교 뷰어 — 1× EQ-only vs 1× EQ+FX vs Original.

좌: 1× EQ-only (명세) / 중: 1× EQ + Mood FX (full system) / 우: Original
하단 3개 실시간 스펙트럼 + 씬 바 (클릭 점프) + 우측 컨트롤 패널

live_compare.py 와 동일한 기능 셋. 중간 패널이 3× demo 대신 문헌-근거
mood-driven FX (sub-bass/low/high shelf + binary reverb) 를 보여줌.
FX 근거: Juslin & Västfjäll 2008 (BRECVEM), Rumsey 2002 (spatial),
         Zentner 2008 (GEMS), Eerola & Vuoskoski 2011 (brightness).
레시피 정의: generate_fx_demo.py MOOD_FX_RECIPE 참고.

컨트롤:
    ▶/⏸ Play/Pause         (버튼 or SPACE 키)
    ← / →  prev/next scene  (scene bar 클릭도 가능)
    Show videos 체크박스 3개  (or F/G/H 키)
    Audio source 라디오 4개   (or 1/2/3/0 키)
    종료: Q 키 또는 창 닫기

Usage:
    venv/bin/python live_compare_fx.py                    # 기본 Kakao demo
    venv/bin/python live_compare_fx.py --audio demo       # EQ+FX 오디오로 시작
    venv/bin/python live_compare_fx.py --speed 0.5        # 0.5× 재생
    venv/bin/python live_compare_fx.py \\
        --processed my_eq.mp4 --demo my_eq_fx.mp4 \\
        --original my_orig.mp4 --timeline my_timeline.json

오디오는 선택된 채널 하나만 재생. 영상은 각 패널을 독립적으로 freeze/resume 가능.

내부 구조상 'demo' 식별자가 중간 패널을 가리키지만, 본 파일에선 실제로
**1× EQ + Mood FX** 를 로드/표시함.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons

BAND_CENTERS_HZ = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
FFT_WINDOW_SEC = 0.25

MOOD_COLORS = {
    "Tension":          "#d62728",
    "Sadness":          "#4b6cb7",
    "Tenderness":       "#ffa500",
    "Wonder":           "#2ca02c",
    "Peacefulness":     "#17becf",
    "Power":            "#9467bd",
    "JoyfulActivation": "#e377c2",
}

PANEL_COLORS = {
    "proc": "#d62728",   # 1× EQ-only
    "demo": "#2ca02c",   # 1× EQ + FX (green — "full system")
    "orig": "#1f3a68",   # Original
}
PANEL_LABELS = {
    "proc": "1× EQ (spec)",
    "demo": "1× EQ + FX",
    "orig": "Original",
}

AUDIO_LABELS = ["1× EQ", "1× EQ + FX", "Original", "Mute"]
AUDIO_KEYS = {"proc": 0, "demo": 1, "orig": 2, "none": 3}
AUDIO_FROM_IDX = {0: "proc", 1: "demo", 2: "orig", 3: "none"}


def extract_audio(video: Path, dst: Path) -> tuple[np.ndarray, int]:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-acodec", "pcm_s16le", str(dst)],
        check=True,
    )
    audio, sr = sf.read(str(dst), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def compute_spectrum(audio: np.ndarray, sr: int, n_out: int = 256) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < 64:
        g = np.geomspace(20.0, 22050.0, n_out)
        return g, np.full(n_out, -100.0)
    window = np.hanning(len(audio))
    X = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    mag = np.abs(X) / len(audio) * 2.0
    db = 20.0 * np.log10(mag + 1e-10)
    f_lo = max(20.0, freqs[1])
    f_hi = min(freqs[-1], 22050.0)
    grid = np.geomspace(f_lo, f_hi, n_out)
    return grid, np.interp(grid, freqs, db)


def band_power_db(audio: np.ndarray, sr: int) -> list[float]:
    if len(audio) < 64:
        return [-80.0] * 10
    X = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    out: list[float] = []
    for fc in BAND_CENTERS_HZ:
        lo = fc / (2 ** 0.5)
        hi = fc * (2 ** 0.5)
        mask = (freqs >= lo) & (freqs < hi)
        p = float(np.mean(np.abs(X[mask]) ** 2)) if mask.any() else 1e-12
        out.append(10.0 * np.log10(p + 1e-12))
    return out


def find_current_scene(timeline: dict, t_sec: float) -> dict | None:
    for sc in timeline.get("scenes", []):
        if sc["start_sec"] <= t_sec < sc["end_sec"]:
            return sc
    return None


class AudioClock:
    """Sample-accurate audio clock with pause/resume + source switching.

    - audio playing: position driven by sounddevice callback
    - muted: wall-clock advances position
    - paused: position frozen
    - source switch: position preserved across streams
    """

    def __init__(self, sr: int, speed: float = 1.0) -> None:
        self.sr = sr
        self.speed = speed
        self._pos = 0
        self._paused = False
        self._source: np.ndarray | None = None
        self._stream = None
        self._wall_anchor_time = time.time()
        self._wall_anchor_pos = 0

    def set_source(self, audio: np.ndarray | None) -> None:
        self._stop_stream()
        self._source = audio
        if not self._paused and audio is not None:
            self._start_stream()
        else:
            # muted / paused: re-anchor wall-clock
            self._wall_anchor_time = time.time()
            self._wall_anchor_pos = self._pos

    def set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        if paused:
            self._stop_stream()
            self._paused = True
        else:
            self._paused = False
            self._wall_anchor_time = time.time()
            self._wall_anchor_pos = self._pos
            if self._source is not None:
                self._start_stream()

    def is_paused(self) -> bool:
        return self._paused

    def _stop_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _start_stream(self) -> None:
        if self._source is None:
            return
        try:
            import sounddevice as sd

            src = self._source

            def callback(outdata, frames, time_info, status):
                s = self._pos
                e = min(s + frames, len(src))
                n = e - s
                if n > 0:
                    outdata[:n, 0] = src[s:e]
                outdata[n:, 0] = 0.0
                self._pos = e
                if e >= len(src):
                    raise sd.CallbackStop()

            self._stream = sd.OutputStream(
                samplerate=int(self.sr), channels=1, dtype="float32",
                callback=callback, blocksize=1024,
            )
            self._stream.start()
        except Exception as e:
            print(f"[audio] stream start failed: {e}", file=sys.stderr)
            self._stream = None

    def current_sec(self) -> float:
        if self._paused:
            return self._pos / self.sr
        if self._source is None or self._stream is None:
            # wall-clock advance (muted path)
            now = time.time()
            advanced = self._wall_anchor_pos + int(
                (now - self._wall_anchor_time) * self.sr * self.speed
            )
            if advanced > self._pos:
                self._pos = advanced
        return self._pos / self.sr

    def seek(self, sec: float) -> None:
        """주어진 시각으로 점프. 오디오 스트림은 새 위치에서 재시작."""
        sec = max(0.0, sec)
        new_pos = int(sec * self.sr)
        if self._source is not None:
            new_pos = min(new_pos, max(0, len(self._source) - 1))
        self._pos = new_pos
        self._wall_anchor_time = time.time()
        self._wall_anchor_pos = new_pos
        if self._source is not None:
            self._stop_stream()
            if not self._paused:
                self._start_stream()

    def stop(self) -> None:
        self._stop_stream()


def main() -> int:
    p = argparse.ArgumentParser(description="Live A/B/C 3-way MoodEQ comparison")
    p.add_argument(
        "--original",
        default="KakaoTalk_Video_2026-04-23-00-05-15.mp4",
    )
    p.add_argument("--processed", default="runs/demo_kakao/kakao_eq_applied.mp4",
                   help="1× EQ-only (spec) 영상")
    p.add_argument("--demo", default="runs/demo_kakao/kakao_eq_fx.mp4",
                   help="1× EQ + Mood FX (full system) 영상")
    p.add_argument("--timeline", default="runs/demo_kakao/timeline.json")
    p.add_argument("--audio", choices=["proc", "demo", "orig", "none"], default="demo",
                   help="시작 시 재생할 오디오 (기본 demo=EQ+FX)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=20.0, help="화면 refresh rate")
    args = p.parse_args()

    paths = {
        "proc": Path(args.processed),
        "demo": Path(args.demo),
        "orig": Path(args.original),
    }
    for key, pth in paths.items():
        if not pth.exists():
            print(f"[error] {key} video not found: {pth}", file=sys.stderr)
            return 1

    tl_path = Path(args.timeline)
    timeline = json.loads(tl_path.read_text()) if tl_path.exists() else None
    if timeline is None:
        print(f"[warn] timeline.json 없음 — scene info 표시 안 됨")

    # --- Video capture ---
    caps = {k: cv2.VideoCapture(str(p)) for k, p in paths.items()}
    for k, c in caps.items():
        if not c.isOpened():
            print(f"[error] {k} 열기 실패: {paths[k]}", file=sys.stderr)
            return 1

    fps_video = caps["proc"].get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    duration = total_frames / fps_video
    print(f"[video] fps={fps_video:.1f}, frames={total_frames}, duration={duration:.1f}s")

    # --- Audio extraction ---
    print("[audio] extracting 3 tracks (ffmpeg) ...")
    audios: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        sr_ref: int | None = None
        for k, p_ in paths.items():
            a, sr = extract_audio(p_, tdp / f"{k}.wav")
            if sr_ref is None:
                sr_ref = sr
            elif sr != sr_ref:
                from scipy.signal import resample_poly
                a = resample_poly(a, sr_ref, sr).astype(np.float32)
            audios[k] = a
    sr = sr_ref
    n_audio = min(len(a) for a in audios.values())
    for k in audios:
        audios[k] = audios[k][:n_audio]
    window_samples = int(FFT_WINDOW_SEC * sr)
    print(f"[audio] sr={sr}Hz, len={n_audio / sr:.1f}s")

    # --- Figure layout ---
    fig = plt.figure(figsize=(17, 10), facecolor="#f0f0f0")
    try:
        fig.canvas.manager.set_window_title("MoodEQ Live A/B/C Compare")
    except Exception:
        pass

    # Main 3-col grid (info / video / EQ / spectrum)
    # 우측 컨트롤 패널용으로 0.78까지만 사용
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[0.95, 2.5, 0.9, 1.4],
        hspace=0.45, wspace=0.08,
        left=0.035, right=0.77, top=0.95, bottom=0.06,
    )
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis("off")
    info_text = ax_info.text(
        0.02, 0.85, "", ha="left", va="top",
        fontsize=11, family="monospace", transform=ax_info.transAxes,
    )

    # Scene bar (click-to-seek)
    # info row 하단에 별도 axes 로 배치 — 이벤트 픽킹 용이
    # 위치는 ax_info 와 같은 수평 범위, 아래쪽에 얇게
    _pos = ax_info.get_position()
    ax_scene = fig.add_axes([_pos.x0, _pos.y0 + 0.005, _pos.width, 0.045])
    ax_scene.set_xlim(0, max(duration, 1.0))
    ax_scene.set_ylim(0, 1)
    ax_scene.set_yticks([])
    ax_scene.set_xticks([])
    for spine in ax_scene.spines.values():
        spine.set_linewidth(0.5)
    ax_scene.set_title("", fontsize=0)

    scene_rects: list[Rectangle] = []
    scene_labels: list[plt.Text] = []
    if timeline:
        for sc in timeline.get("scenes", []):
            mood = sc["mood"]["name"]
            col = MOOD_COLORS.get(mood, "#666666")
            w = sc["end_sec"] - sc["start_sec"]
            rect = Rectangle(
                (sc["start_sec"], 0.0), w, 1.0,
                facecolor=col, edgecolor="white", linewidth=0.8, alpha=0.55,
            )
            ax_scene.add_patch(rect)
            scene_rects.append(rect)
            # Scene 번호 라벨 (너무 좁으면 생략)
            if w > 1.5:
                lbl = ax_scene.text(
                    sc["start_sec"] + w / 2, 0.5,
                    f"s{sc['scene_idx']}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
                scene_labels.append(lbl)
    # 현재 시각 표시 마커 (세로선)
    scene_cursor = ax_scene.axvline(0, color="black", linewidth=2.0, zorder=10)

    # 3 video / EQ / spectrum axes
    ax_vid: dict[str, plt.Axes] = {}
    ax_eq: dict[str, plt.Axes] = {}
    ax_sp: dict[str, plt.Axes] = {}
    for col, key in enumerate(["proc", "demo", "orig"]):
        ax_vid[key] = fig.add_subplot(gs[1, col])
        ax_eq[key] = fig.add_subplot(gs[2, col])
        ax_sp[key] = fig.add_subplot(gs[3, col])

    # Video panel titles
    title_alignment = {"proc": "left", "demo": "center", "orig": "right"}
    for key in ["proc", "demo", "orig"]:
        ax_vid[key].set_title(
            PANEL_LABELS[key], fontsize=13, color=PANEL_COLORS[key],
            fontweight="bold", loc=title_alignment[key],
        )
        ax_vid[key].set_xticks([])
        ax_vid[key].set_yticks([])

    # EQ panels (video 와 spectrum 사이)
    # 1×, 3× 는 timeline gain 막대, Original 은 bypass 수평선
    EQ_Y_RANGE = (-8.0, 8.0)
    # 중간 패널 (demo) 이 1× EQ + FX 이므로 EQ gain 은 1× (FX 는 EQ 외 별도 체인)
    EQ_GAIN_MULT = {"proc": 1.0, "demo": 1.0, "orig": 0.0}
    # 로그 x축에서 각 band 의 bar 폭 (band 중심 좌우 half-octave 가까이)
    BAND_BAR_WIDTHS = [
        fc * (2 ** (1 / 6)) - fc * (2 ** (-1 / 6))
        for fc in BAND_CENTERS_HZ
    ]

    eq_bars: dict[str, object] = {}
    eq_title_text: dict[str, plt.Text] = {}
    for key in ["proc", "demo", "orig"]:
        ax = ax_eq[key]
        label_suffix = (
            "EQ gain (1× spec)  + mood FX chain" if key == "demo"
            else "EQ gain (bypass)" if key == "orig"
            else "EQ gain (1× spec)"
        )
        ax.set_title(label_suffix, fontsize=10, color=PANEL_COLORS[key], loc="left")
        ax.set_xscale("log")
        ax.set_xlim(20, 22050)
        ax.set_ylim(*EQ_Y_RANGE)
        ax.set_ylabel("gain dB", fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=False)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="both", alpha=0.25)
        for fc in BAND_CENTERS_HZ:
            ax.axvline(fc, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)
        # ±1 dB JND 영역 음영
        ax.axhspan(-1.0, 1.0, color="gray", alpha=0.12)

        # 초기 bar (0 gain)
        bars = ax.bar(
            BAND_CENTERS_HZ,
            [0.0] * 10,
            width=BAND_BAR_WIDTHS,
            align="center",
            color=PANEL_COLORS[key],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        eq_bars[key] = bars
        # scene 라벨 (오른쪽 위)
        eq_title_text[key] = ax.text(
            0.98, 0.93, "", ha="right", va="top",
            fontsize=9, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
        )

    # Spectrum panels
    for key in ["proc", "demo", "orig"]:
        ax = ax_sp[key]
        ax.set_title(f"Spectrum — {PANEL_LABELS[key]}",
                     fontsize=10, color=PANEL_COLORS[key], loc="left")
        ax.set_xscale("log")
        ax.set_xlim(20, 22050)
        ax.set_ylim(-110, 0)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("dB", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        for fc in BAND_CENTERS_HZ:
            ax.axvline(fc, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    sp_lines: dict[str, plt.Line2D] = {}
    sp_marks = {}
    for key in ["proc", "demo", "orig"]:
        sp_lines[key], = ax_sp[key].plot([], [], color=PANEL_COLORS[key], linewidth=1.5)
        sp_marks[key] = ax_sp[key].scatter(
            [], [], s=26, color=PANEL_COLORS[key],
            edgecolors="white", linewidths=0.5, zorder=5,
        )

    # 초기 프레임
    ims: dict[str, object] = {}
    for key in ["proc", "demo", "orig"]:
        ret, fr = caps[key].read()
        if not ret:
            print(f"[error] 첫 프레임 읽기 실패: {key}", file=sys.stderr)
            return 1
        ims[key] = ax_vid[key].imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

    # Freeze overlay (off 상태 표시용)
    freeze_overlays: dict[str, plt.Text] = {}
    for key in ["proc", "demo", "orig"]:
        t = ax_vid[key].text(
            0.5, 0.5, "(paused)", transform=ax_vid[key].transAxes,
            ha="center", va="center", fontsize=20, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.0),
        )
        t.set_visible(False)
        freeze_overlays[key] = t

    # === 컨트롤 패널 (우측) ===
    # Play/Pause 버튼
    ax_btn_play = fig.add_axes([0.80, 0.82, 0.17, 0.07])
    btn_play = Button(ax_btn_play, "⏸  Pause", color="#e0e0e0", hovercolor="#cccccc")

    # Show videos 체크박스
    ax_check = fig.add_axes([0.80, 0.54, 0.17, 0.22])
    ax_check.set_title("Show videos\n(F / G / H)", fontsize=10, pad=8)
    check_videos = CheckButtons(
        ax_check,
        labels=[PANEL_LABELS["proc"], PANEL_LABELS["demo"], PANEL_LABELS["orig"]],
        actives=[True, True, True],
    )

    # Audio source radio
    ax_radio = fig.add_axes([0.80, 0.18, 0.17, 0.30])
    ax_radio.set_title("Audio source\n(1 / 2 / 3 / 0)", fontsize=10, pad=8)
    radio_audio = RadioButtons(
        ax_radio,
        labels=AUDIO_LABELS,
        active=AUDIO_KEYS[args.audio],
    )

    # 키보드 안내
    ax_help = fig.add_axes([0.80, 0.04, 0.17, 0.09])
    ax_help.axis("off")
    ax_help.text(
        0.0, 1.0,
        "SPACE  play/pause\n"
        "←/→    prev/next scene\n"
        "click scene bar: jump\n"
        "F/G/H  toggle video\n"
        "1/2/3/0  audio\n"
        "Q  quit",
        fontsize=9, family="monospace", va="top",
        transform=ax_help.transAxes,
    )

    # --- State ---
    state = {
        "done": False,
        "frame_idx": {"proc": 0, "demo": 0, "orig": 0},
        "video_enabled": {"proc": True, "demo": True, "orig": True},
        "audio_source": args.audio,
        "last_spectra": {},
    }

    clock = AudioClock(sr=sr, speed=args.speed)
    if args.audio != "none":
        clock.set_source(audios[args.audio])
    print(f"[audio] initial source: {args.audio.upper()}")

    # --- Control handlers ---
    def on_play_clicked(event=None):
        new_paused = not clock.is_paused()
        clock.set_paused(new_paused)
        btn_play.label.set_text("▶  Play" if new_paused else "⏸  Pause")
        fig.canvas.draw_idle()

    btn_play.on_clicked(on_play_clicked)

    def on_check_clicked(label):
        key = {v: k for k, v in PANEL_LABELS.items()}[label]
        state["video_enabled"][key] = not state["video_enabled"][key]
        freeze_overlays[key].set_visible(not state["video_enabled"][key])
        freeze_overlays[key].get_bbox_patch().set_alpha(
            0.55 if not state["video_enabled"][key] else 0.0
        )
        fig.canvas.draw_idle()

    check_videos.on_clicked(on_check_clicked)

    def on_radio_clicked(label):
        idx = AUDIO_LABELS.index(label)
        new_src = AUDIO_FROM_IDX[idx]
        state["audio_source"] = new_src
        clock.set_source(audios[new_src] if new_src != "none" else None)
        print(f"[audio] -> {new_src.upper()}")

    radio_audio.on_clicked(on_radio_clicked)

    def seek_all(target_sec: float) -> None:
        """오디오 클록과 모든 비디오 캡처를 target_sec 로 동기 이동."""
        target_sec = max(0.0, min(target_sec, duration - 0.05))
        clock.seek(target_sec)
        target_frame = int(target_sec * fps_video)
        for k, c in caps.items():
            c.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame))
            state["frame_idx"][k] = max(0, target_frame - 1)  # next update() reads this frame
        print(f"[seek] -> {target_sec:.2f}s (frame {target_frame})")

    def on_scene_click(event):
        if event.inaxes is not ax_scene or event.xdata is None:
            return
        if event.button != 1:  # left click only
            return
        click_sec = float(event.xdata)
        # 클릭 좌표 → 해당 scene 의 start_sec 로 점프
        if timeline:
            sc = find_current_scene(timeline, click_sec)
            if sc is not None:
                seek_all(sc["start_sec"])
                return
        seek_all(click_sec)

    fig.canvas.mpl_connect("button_press_event", on_scene_click)

    def on_key(event):
        if event.key == " ":
            on_play_clicked()
        elif event.key and event.key.lower() in ("f", "g", "h"):
            idx = "fgh".index(event.key.lower())
            check_videos.set_active(idx)  # triggers on_check_clicked
        elif event.key in ("1", "2", "3", "0"):
            idx = {"1": 0, "2": 1, "3": 2, "0": 3}[event.key]
            radio_audio.set_active(idx)
        elif event.key in ("left", "right") and timeline:
            # 이전/다음 scene 으로 이동
            cur_t = clock.current_sec()
            scenes = timeline["scenes"]
            if event.key == "right":
                target = next(
                    (sc["start_sec"] for sc in scenes if sc["start_sec"] > cur_t + 0.1),
                    scenes[-1]["start_sec"],
                )
            else:  # left
                target = 0.0
                for sc in scenes:
                    if sc["start_sec"] < cur_t - 0.5:
                        target = sc["start_sec"]
                    else:
                        break
            seek_all(target)
        elif event.key and event.key.lower() == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Update loop ---
    def seek_and_read(key: str, target: int) -> np.ndarray | None:
        """caps[key] 를 target 프레임까지 전진시켜 최신 프레임 반환."""
        cur = state["frame_idx"][key]
        delta = target - cur
        if delta <= 0:
            return None
        if delta > 5:
            caps[key].set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, fr = caps[key].read()
        else:
            fr = None
            ret = False
            for _ in range(delta):
                ret, fr = caps[key].read()
                if not ret:
                    break
        state["frame_idx"][key] = target
        return fr if ret else None

    def update(_):
        if state["done"]:
            return []

        t_sec = clock.current_sec()
        target_frame = int(t_sec * fps_video)
        if target_frame >= total_frames:
            state["done"] = True
            ani.event_source.stop()
            return []

        # Video frames (per-panel toggle 반영)
        for key in ["proc", "demo", "orig"]:
            if not state["video_enabled"][key]:
                continue  # frozen
            if target_frame == state["frame_idx"][key] and target_frame > 0:
                continue  # same frame, skip
            fr = seek_and_read(key, target_frame)
            if fr is not None:
                ims[key].set_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

        # Spectra: 비활성 패널은 마지막 값 유지
        sc_center = int(t_sec * sr)
        half = window_samples // 2
        s = max(0, sc_center - half)
        e = min(n_audio, sc_center + half)
        if e - s >= 512:
            for key in ["proc", "demo", "orig"]:
                if not state["video_enabled"][key]:
                    continue
                g, db = compute_spectrum(audios[key][s:e], sr)
                sp_lines[key].set_data(g, db)
                bp = band_power_db(audios[key][s:e], sr)
                sp_marks[key].set_offsets(list(zip(BAND_CENTERS_HZ, bp)))

        # EQ bars (scene 단위로만 바뀜)
        cur_scene = find_current_scene(timeline, t_sec) if timeline else None
        if cur_scene is not None:
            base_gains = [
                b["gain_db"]
                for b in cur_scene.get("eq_preset", {}).get("effective_bands", [])
            ]
        else:
            base_gains = [0.0] * 10
        if len(base_gains) != 10:
            base_gains = (base_gains + [0.0] * 10)[:10]

        for key in ["proc", "demo", "orig"]:
            if not state["video_enabled"][key]:
                continue
            mult = EQ_GAIN_MULT[key]
            gains = [g * mult for g in base_gains]
            for bar, gain in zip(eq_bars[key], gains):
                bar.set_height(gain)
                # boost=green, cut=red, ≈0=gray
                if gain > 0.2:
                    bar.set_color("#2ca02c")
                elif gain < -0.2:
                    bar.set_color("#d62728")
                else:
                    bar.set_color("#888888")
            # scene 요약
            if key == "orig":
                eq_title_text[key].set_text("(bypass — EQ 미적용)")
            elif cur_scene is not None:
                mood = cur_scene["mood"]["name"]
                max_abs = max(abs(g) for g in gains) if gains else 0.0
                eq_title_text[key].set_text(f"{mood} · max |Δ|={max_abs:.1f}dB")
            else:
                eq_title_text[key].set_text("")

        # Info text
        info_lines = [
            f"time {t_sec:6.2f} / {duration:6.2f} s     "
            f"frame {target_frame:5d} / {total_frames:5d}     "
            f"audio={state['audio_source']:>4s}     "
            f"speed={args.speed:.2f}×     "
            f"{'⏸ PAUSED' if clock.is_paused() else '▶ playing'}"
        ]
        if timeline:
            sc = find_current_scene(timeline, t_sec)
            if sc:
                mood = sc["mood"]["name"]
                va = sc.get("va", {})
                info_lines.append(
                    f"scene {sc.get('scene_idx', '?'):>2}  "
                    f"[{sc['start_sec']:5.1f}–{sc['end_sec']:5.1f}s]  "
                    f"mood={mood:<18}  V={va.get('valence', 0):+.2f}  "
                    f"A={va.get('arousal', 0):+.2f}"
                )
                bands = sc.get("eq_preset", {}).get("effective_bands", [])
                if bands:
                    nonzero = [(b["freq_hz"], b["gain_db"])
                               for b in bands if abs(b["gain_db"]) > 0.05]
                    if nonzero:
                        parts = []
                        for f, g in nonzero:
                            tag = f"{int(f)}Hz" if f < 1000 else f"{int(f/1000)}k"
                            parts.append(f"{tag}:{g:+.1f}")
                        info_lines.append("EQ(1×) " + "   ".join(parts))
                    else:
                        info_lines.append("EQ(1×) (flat)")
        info_text.set_text("\n".join(info_lines))
        scene_cursor.set_xdata([t_sec, t_sec])
        # 현재 scene 강조
        if timeline and scene_rects:
            for i, sc in enumerate(timeline["scenes"]):
                is_current = sc["start_sec"] <= t_sec < sc["end_sec"]
                scene_rects[i].set_alpha(0.95 if is_current else 0.55)
                scene_rects[i].set_linewidth(2.0 if is_current else 0.8)
                scene_rects[i].set_edgecolor("black" if is_current else "white")

        eq_artists: list[object] = []
        for bars in eq_bars.values():
            eq_artists.extend(bars)
        eq_artists.extend(eq_title_text.values())
        return (list(ims.values()) + list(sp_lines.values())
                + list(sp_marks.values()) + eq_artists + scene_rects
                + [info_text, scene_cursor] + list(freeze_overlays.values()))

    interval_ms = 1000.0 / args.fps
    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)

    def on_close(event):
        state["done"] = True
        clock.stop()
        for c in caps.values():
            c.release()
        print("[live] closed")

    fig.canvas.mpl_connect("close_event", on_close)

    print()
    print("=== Controls ===")
    print("  SPACE          play / pause")
    print("  ← / →          prev / next scene")
    print("  click scene bar  jump to that scene")
    print("  F / G / H      toggle 1× / 3× / Original video")
    print("  1 / 2 / 3 / 0  audio source: 1× / 3× / Original / Mute")
    print("  Q              quit")
    print()

    try:
        plt.show()
    finally:
        state["done"] = True
        clock.stop()
        for c in caps.values():
            c.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
