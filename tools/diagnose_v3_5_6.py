"""diagnose_v3_5_6.py — V3.5.6 처리 효과 정량 진단.

Phase D: V3.5.6 청취 시 "차이 체감 어려움" 피드백 → 코드상 처리(시변 EQ + M/S +
Sidechain ducking + Compressor)가 실제 출력에 반영됐는지 / 매칭으로 상쇄됐는지
정량 측정.

측정 항목:
  1. 전체 차이 (diff RMS, diff peak) — matched & 비매칭 양쪽
  2. 대역별 spectrum 변화 (10 octave bands, dB)
  3. M/S decomposition — side/mid energy ratio 비교
  4. Sidechain ducking — vocals envelope vs no_vocals 2-5kHz 대역 envelope 상관
  5. LUFS 차이 (pyloudnorm)
  6. 씬별 / 카테고리별 diff (timeline.json 기반)

출력: 콘솔 리포트 + tmp/v3_5_6_diagnosis.json

사용:
  PYTHONIOENCODING=utf-8 D:/Homecinema/venv/Scripts/python.exe \\
      tools/diagnose_v3_5_6.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


REPO = Path(__file__).resolve().parent.parent
TMP = REPO / "tmp" / "full_trailer" / "topgun"
JOB = REPO / "backend" / "data" / "jobs" / "fe2ecad8-dc25-4131-adfe-ffeea6d977a1"

# 10 octave bands (V3.3 EQ band 중심 주파수와 일치)
BANDS = [
    (22, 45),     (45, 90),     (90, 180),    (180, 355),   (355, 710),
    (710, 1400),  (1400, 2800), (2800, 5600), (5600, 11200), (11200, 22000),
]
BAND_LABELS = ["31.5 Hz", "63 Hz", "125 Hz", "250 Hz", "500 Hz",
               "1 kHz", "2 kHz", "4 kHz", "8 kHz", "16 kHz"]

DUCK_BAND = (2000.0, 5000.0)  # V3.5.6 sidechain ducking 대상 대역


# ────────────────────────────────────────────────────────
# I/O
# ────────────────────────────────────────────────────────
def load(path: Path) -> tuple[int, np.ndarray]:
    sr, a = wavfile.read(str(path))
    if a.dtype == np.int16:
        a = a.astype(np.float64) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float64) / 2147483648.0
    else:
        a = a.astype(np.float64)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    return sr, a


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def peak(x: np.ndarray) -> float:
    return float(np.abs(x).max())


# ────────────────────────────────────────────────────────
# 분석
# ────────────────────────────────────────────────────────
def measure_lufs(sr: int, a: np.ndarray) -> float:
    return float(pyln.Meter(sr).integrated_loudness(a))


def diff_stats(ref: np.ndarray, test: np.ndarray) -> dict:
    L = min(ref.shape[0], test.shape[0])
    d = test[:L] - ref[:L]
    return {
        "rms_db":  db(rms(d)),
        "peak_db": db(peak(d)),
        "ref_rms_db":  db(rms(ref[:L])),
        "test_rms_db": db(rms(test[:L])),
    }


def band_spectrum_db(sr: int, a: np.ndarray) -> list[float]:
    """모노 합산 후 FFT, 각 octave band 평균 magnitude (dB)."""
    mono = a.mean(axis=1)
    n = len(mono)
    spec = np.fft.rfft(mono * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.abs(spec) / (n / 2)
    out = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            out.append(db(float(np.mean(mag[m]))))
        else:
            out.append(float("nan"))
    return out


def ms_decompose(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, 2] stereo → (mid, side) 모노."""
    L, R = a[:, 0], a[:, 1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)
    return mid, side


def ms_energy(a: np.ndarray) -> dict:
    mid, side = ms_decompose(a)
    e_m, e_s = float(np.mean(mid ** 2)), float(np.mean(side ** 2))
    return {
        "mid_rms_db":  db(np.sqrt(e_m)),
        "side_rms_db": db(np.sqrt(e_s)),
        "side_over_mid_db": db(np.sqrt(e_s / max(e_m, 1e-20))),
    }


def bandpass(sr: int, a: np.ndarray, lo: float, hi: float, order: int = 6) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, a, axis=0)


def envelope(x: np.ndarray, sr: int, smooth_ms: float = 50.0) -> np.ndarray:
    """간단한 abs + 단방향 1-pole smoothing 엔벨로프 (모노 또는 stereo→mono)."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    abs_x = np.abs(x)
    n_smp = max(1, int(sr * smooth_ms / 1000.0))
    # 단순 boxcar EMA
    alpha = 1.0 / n_smp
    out = np.empty_like(abs_x)
    acc = 0.0
    for i in range(len(abs_x)):
        acc = (1 - alpha) * acc + alpha * abs_x[i]
        out[i] = acc
    return out


def sidechain_diagnosis(
    sr: int, ref: np.ndarray, test: np.ndarray, vocals: np.ndarray,
) -> dict:
    """2-5 kHz 대역에서 vocals envelope 와 (ref-test) gain reduction 의 상관."""
    L = min(ref.shape[0], test.shape[0], vocals.shape[0])
    ref_b = bandpass(sr, ref[:L], DUCK_BAND[0], DUCK_BAND[1])
    test_b = bandpass(sr, test[:L], DUCK_BAND[0], DUCK_BAND[1])
    voc_b = bandpass(sr, vocals[:L], DUCK_BAND[0], DUCK_BAND[1])

    # downsample 100 Hz envelope
    env_ref = envelope(ref_b, sr, smooth_ms=50.0)
    env_test = envelope(test_b, sr, smooth_ms=50.0)
    env_voc = envelope(voc_b, sr, smooth_ms=50.0)

    step = max(1, int(sr / 100))
    er = env_ref[::step]
    et = env_test[::step]
    ev = env_voc[::step]

    # gain reduction in dB (양수 = 감쇠 발생)
    gr_db = 20.0 * np.log10(np.maximum(er, 1e-9) / np.maximum(et, 1e-9))
    # vocals 활동 dB
    voc_db = 20.0 * np.log10(np.maximum(ev, 1e-9))

    # vocals 활동 큰 구간 / 작은 구간 평균 GR
    voc_high_mask = voc_db > np.percentile(voc_db, 75)
    voc_low_mask = voc_db < np.percentile(voc_db, 25)
    gr_high = float(np.mean(gr_db[voc_high_mask])) if voc_high_mask.any() else float("nan")
    gr_low = float(np.mean(gr_db[voc_low_mask])) if voc_low_mask.any() else float("nan")

    # Pearson correlation
    if len(gr_db) > 10:
        corr = float(np.corrcoef(voc_db, gr_db)[0, 1])
    else:
        corr = float("nan")

    return {
        "band_hz": list(DUCK_BAND),
        "gr_mean_db_when_vocals_loud":  gr_high,
        "gr_mean_db_when_vocals_quiet": gr_low,
        "gr_diff_loud_minus_quiet_db":  gr_high - gr_low,
        "vocals_db_to_gr_correlation":  corr,
        "ref_band_rms_db":  db(rms(ref_b)),
        "test_band_rms_db": db(rms(test_b)),
    }


def per_scene_diff(
    sr: int, ref: np.ndarray, test: np.ndarray, scenes: list[dict],
) -> list[dict]:
    L = min(ref.shape[0], test.shape[0])
    out = []
    for s in scenes:
        i0 = max(0, int(s["start_sec"] * sr))
        i1 = min(L, int(s["end_sec"] * sr))
        if i1 <= i0:
            continue
        d = test[i0:i1] - ref[i0:i1]
        out.append({
            "scene_id":   s["scene_id"],
            "category":   s["aggregated"]["category"],
            "start_sec":  s["start_sec"],
            "end_sec":    s["end_sec"],
            "diff_rms_db": db(rms(d)),
            "diff_peak_db": db(peak(d)),
        })
    return out


def category_summary(per_scene: list[dict]) -> dict:
    by_cat: dict[str, list[float]] = {}
    for s in per_scene:
        by_cat.setdefault(s["category"], []).append(s["diff_rms_db"])
    return {
        c: {
            "n_scenes": len(v),
            "mean_diff_rms_db": float(np.mean(v)),
            "min_diff_rms_db":  float(np.min(v)),
            "max_diff_rms_db":  float(np.max(v)),
        }
        for c, v in sorted(by_cat.items())
    }


# ────────────────────────────────────────────────────────
# 리포트
# ────────────────────────────────────────────────────────
def fmt_db(v: float, sign: bool = True) -> str:
    if not np.isfinite(v):
        return "  N/A "
    return f"{v:+7.2f} dB" if sign else f"{v:7.2f} dB"


def render_report(d: dict) -> str:
    L = []
    L.append("=" * 68)
    L.append("V3.5.6 처리 효과 정량 진단 (Phase D)")
    L.append("=" * 68)
    L.append("")
    L.append("[1] 입력 파일")
    for k, p in d["files"].items():
        L.append(f"    {k:24s}  {p}")
    L.append("")

    L.append("[2] 전체 차이 (diff = test − reference)")
    L.append("    --- matched 비교 (평가에 사용된 파일) ---")
    m = d["overall"]["matched"]
    L.append(f"    diff RMS:     {fmt_db(m['rms_db'], False)}     (0 dBFS 기준)")
    L.append(f"    diff peak:    {fmt_db(m['peak_db'], False)}")
    L.append(f"    ref  RMS:     {fmt_db(m['ref_rms_db'], False)}")
    L.append(f"    test RMS:     {fmt_db(m['test_rms_db'], False)}")
    if "unmatched" in d["overall"]:
        u = d["overall"]["unmatched"]
        L.append("    --- unmatched 비교 (LUFS 매칭 전, 원본 처리 효과) ---")
        L.append(f"    diff RMS:     {fmt_db(u['rms_db'], False)}")
        L.append(f"    diff peak:    {fmt_db(u['peak_db'], False)}")
    L.append("")

    L.append("[3] LUFS")
    lu = d["lufs"]
    L.append(f"    ref  matched: {lu['ref_matched']:+7.2f} LUFS")
    L.append(f"    test matched: {lu['test_matched']:+7.2f} LUFS"
             f"     Δ={lu['test_matched']-lu['ref_matched']:+.2f} LU")
    if "ref_unmatched" in lu:
        L.append(f"    ref  unmatch: {lu['ref_unmatched']:+7.2f} LUFS")
        L.append(f"    test unmatch: {lu['test_unmatched']:+7.2f} LUFS"
                 f"     Δ={lu['test_unmatched']-lu['ref_unmatched']:+.2f} LU")
        match_gain = (lu['test_matched'] - lu['test_unmatched'])
        L.append(f"    매칭 게인 (test): {match_gain:+.2f} dB"
                 f"   {'(상쇄 효과 있음)' if abs(match_gain) > 0.5 else '(매칭 영향 미미)'}")
    L.append("")

    L.append("[4] 대역별 spectrum (test − ref, dB)")
    L.append(f"    {'Band':<10}{'ref dB':>10}{'test dB':>10}{'Δ dB':>10}")
    for label, r, t in zip(BAND_LABELS, d["spectrum"]["ref_db"], d["spectrum"]["test_db"]):
        L.append(f"    {label:<10}{r:10.2f}{t:10.2f}{(t-r):+10.2f}")
    L.append("")

    L.append("[5] M/S decomposition")
    L.append(f"    {'':<6}{'mid dB':>10}{'side dB':>10}{'side/mid dB':>14}")
    L.append(f"    ref   {d['ms']['ref']['mid_rms_db']:10.2f}{d['ms']['ref']['side_rms_db']:10.2f}"
             f"{d['ms']['ref']['side_over_mid_db']:+14.2f}")
    L.append(f"    test  {d['ms']['test']['mid_rms_db']:10.2f}{d['ms']['test']['side_rms_db']:10.2f}"
             f"{d['ms']['test']['side_over_mid_db']:+14.2f}")
    L.append(f"    Δ     {d['ms']['delta']['mid_db']:+10.2f}{d['ms']['delta']['side_db']:+10.2f}"
             f"{d['ms']['delta']['side_over_mid_db']:+14.2f}")
    L.append(f"    (V3.5.6 기대: side/mid Δ ≈ +1.5 ~ +2.5 dB)")
    L.append("")

    if "sidechain" in d:
        sc = d["sidechain"]
        L.append("[6] Sidechain ducking (2–5 kHz, vocals envelope 기반)")
        L.append(f"    band ref  RMS: {sc['ref_band_rms_db']:+.2f} dB")
        L.append(f"    band test RMS: {sc['test_band_rms_db']:+.2f} dB")
        L.append(f"    GR (vocals 큼):    {sc['gr_mean_db_when_vocals_loud']:+.2f} dB")
        L.append(f"    GR (vocals 작음):  {sc['gr_mean_db_when_vocals_quiet']:+.2f} dB")
        L.append(f"    GR diff (loud-quiet): {sc['gr_diff_loud_minus_quiet_db']:+.2f} dB"
                 f"   (>0.5 면 ducking 동작 신호)")
        L.append(f"    Pearson r (voc → GR): {sc['vocals_db_to_gr_correlation']:+.3f}"
                 f"   (>0.2 면 vocals 추종)")
        L.append("")

    L.append("[7] 카테고리별 평균 diff RMS")
    L.append(f"    {'Category':<22}{'n':>4}{'mean':>10}{'min':>10}{'max':>10}")
    for c, v in d["category"].items():
        L.append(f"    {c:<22}{v['n_scenes']:4d}{v['mean_diff_rms_db']:10.2f}"
                 f"{v['min_diff_rms_db']:10.2f}{v['max_diff_rms_db']:10.2f}")
    L.append("")

    # 종합 진단
    diag = d["diagnosis"]
    L.append("[8] 종합 진단")
    for key, (mark, msg) in diag.items():
        L.append(f"    [{mark}] {key:<22} — {msg}")
    L.append("")
    L.append("추천 다음 액션:")
    for line in d["recommendations"]:
        L.append(f"    · {line}")
    L.append("=" * 68)
    return "\n".join(L)


def synthesize_diagnosis(d: dict) -> tuple[dict, list[str]]:
    """수치 → ✓/○/⚠️ 상태 변환."""
    diag: dict[str, tuple[str, str]] = {}
    rec: list[str] = []

    # 전체 차이
    rms_m = d["overall"]["matched"]["rms_db"]
    if rms_m < -45:
        diag["overall_diff"] = ("⚠", f"matched diff RMS {rms_m:+.1f} dB 매우 작음 — 매칭이 처리를 상쇄했을 가능성")
        rec.append("unmatched diff RMS 와 비교해 매칭 효과 분리 확인")
    elif rms_m < -30:
        diag["overall_diff"] = ("○", f"matched diff RMS {rms_m:+.1f} dB — 보통 (청감 차이 미묘)")
    else:
        diag["overall_diff"] = ("✓", f"matched diff RMS {rms_m:+.1f} dB — 충분히 큼")

    # M/S (side/mid Δ ≈ +1.5 dB 기대)
    sm_delta = d["ms"]["delta"]["side_over_mid_db"]
    if sm_delta >= 1.0:
        diag["ms_processing"] = ("✓", f"side/mid Δ {sm_delta:+.2f} dB — M/S 적용 확인")
    elif sm_delta >= 0.3:
        diag["ms_processing"] = ("○", f"side/mid Δ {sm_delta:+.2f} dB — 효과 있으나 약함")
    else:
        diag["ms_processing"] = ("⚠", f"side/mid Δ {sm_delta:+.2f} dB — M/S 효과 미미 (기대 +1.5 dB)")
        rec.append("apply_mid_side_processing 호출 여부 / mid_gain_db 파라미터 확인")

    # Sidechain (vocals 큰 구간 GR > vocals 작은 구간 + corr > 0.2)
    if "sidechain" in d:
        sc = d["sidechain"]
        gr_diff = sc["gr_diff_loud_minus_quiet_db"]
        corr = sc["vocals_db_to_gr_correlation"]
        if gr_diff > 0.5 and corr > 0.2:
            diag["sidechain_ducking"] = ("✓", f"GR diff +{gr_diff:.2f} dB, r={corr:+.2f} — ducking 작동")
        elif gr_diff > 0.2 or corr > 0.1:
            diag["sidechain_ducking"] = ("○", f"GR diff +{gr_diff:.2f} dB, r={corr:+.2f} — 약한 효과")
        else:
            diag["sidechain_ducking"] = ("⚠", f"GR diff +{gr_diff:.2f} dB, r={corr:+.2f} — ducking 미작동 의심")
            rec.append("apply_sidechain_ducking 호출 여부 / threshold/ratio 파라미터 확인")

    # EQ 효과 (대역별 변화 max abs)
    deltas = [t - r for r, t in zip(d["spectrum"]["ref_db"], d["spectrum"]["test_db"])]
    max_abs = max(abs(x) for x in deltas if np.isfinite(x))
    if max_abs >= 1.5:
        diag["eq_changes"] = ("✓", f"대역별 변화 최대 {max_abs:.2f} dB — EQ 동작 확인")
    elif max_abs >= 0.7:
        diag["eq_changes"] = ("○", f"대역별 변화 최대 {max_abs:.2f} dB — EQ 효과 약함")
    else:
        diag["eq_changes"] = ("⚠", f"대역별 변화 최대 {max_abs:.2f} dB — EQ 효과 미미")
        rec.append("scene EQ preset 강도 / dialogue protection 설정 점검")

    # 카테고리별 차이 분산
    cat_means = [v["mean_diff_rms_db"] for v in d["category"].values()]
    cat_spread = max(cat_means) - min(cat_means)
    if cat_spread > 3.0:
        diag["temporal_variation"] = ("✓", f"카테고리별 spread {cat_spread:.2f} dB — 시변 EQ 작동")
    elif cat_spread > 1.0:
        diag["temporal_variation"] = ("○", f"카테고리별 spread {cat_spread:.2f} dB — 시변 약함")
    else:
        diag["temporal_variation"] = ("⚠", f"카테고리별 spread {cat_spread:.2f} dB — 시변성 거의 없음")

    if not rec:
        rec.append("처리 효과 충분히 적용됨 — 청감 미미 시 청취 환경/볼륨 점검")
        rec.append("개별 파라미터 강화: mid_gain_db -1.0 → -2.0, side +1.5 → +2.5, ducking max 6→9 dB")
    rec.append(f"진단 JSON: tmp/v3_5_6_diagnosis.json")
    return diag, rec


# ────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-matched", default=str(TMP / "original_matched.wav"))
    ap.add_argument("--test-matched", default=str(TMP / "v3_5_6_matched.wav"))
    ap.add_argument("--ref-unmatched", default=str(TMP / "original.wav"))
    ap.add_argument("--test-unmatched", default=str(TMP / "v3_5_6.wav"))
    ap.add_argument("--vocals", default=str(TMP / "stems" / "vocals.wav"))
    ap.add_argument("--timeline", default=str(JOB / "timeline.json"))
    ap.add_argument("--out", default=str(REPO / "tmp" / "v3_5_6_diagnosis.json"))
    args = ap.parse_args()

    print("로딩...")
    sr_r, ref_m  = load(Path(args.ref_matched))
    sr_t, test_m = load(Path(args.test_matched))
    if sr_r != sr_t:
        raise ValueError(f"SR mismatch: ref={sr_r} test={sr_t}")
    sr = sr_r

    has_unmatched = Path(args.ref_unmatched).exists() and Path(args.test_unmatched).exists()
    has_vocals    = Path(args.vocals).exists()
    has_timeline  = Path(args.timeline).exists()

    result: dict = {
        "files": {
            "ref_matched":  args.ref_matched,
            "test_matched": args.test_matched,
        },
        "sr": sr,
        "samples": int(min(ref_m.shape[0], test_m.shape[0])),
        "duration_sec": float(min(ref_m.shape[0], test_m.shape[0]) / sr),
    }

    # [2] overall diff
    print("  · overall diff")
    overall = {"matched": diff_stats(ref_m, test_m)}
    if has_unmatched:
        sr_u, ref_u  = load(Path(args.ref_unmatched))
        _,    test_u = load(Path(args.test_unmatched))
        overall["unmatched"] = diff_stats(ref_u, test_u)
        result["files"]["ref_unmatched"]  = args.ref_unmatched
        result["files"]["test_unmatched"] = args.test_unmatched
    result["overall"] = overall

    # [3] LUFS
    print("  · LUFS")
    lufs = {
        "ref_matched":  measure_lufs(sr, ref_m),
        "test_matched": measure_lufs(sr, test_m),
    }
    if has_unmatched:
        lufs["ref_unmatched"]  = measure_lufs(sr, ref_u)
        lufs["test_unmatched"] = measure_lufs(sr, test_u)
    result["lufs"] = lufs

    # [4] band spectrum
    print("  · band spectrum")
    result["spectrum"] = {
        "bands_hz": BANDS,
        "labels":   BAND_LABELS,
        "ref_db":   band_spectrum_db(sr, ref_m),
        "test_db":  band_spectrum_db(sr, test_m),
    }

    # [5] M/S
    print("  · M/S decomposition")
    ms_r = ms_energy(ref_m)
    ms_t = ms_energy(test_m)
    result["ms"] = {
        "ref":  ms_r,
        "test": ms_t,
        "delta": {
            "mid_db":  ms_t["mid_rms_db"]  - ms_r["mid_rms_db"],
            "side_db": ms_t["side_rms_db"] - ms_r["side_rms_db"],
            "side_over_mid_db": ms_t["side_over_mid_db"] - ms_r["side_over_mid_db"],
        },
    }

    # [6] sidechain (vocals 필요)
    if has_vocals:
        print("  · sidechain ducking (envelope analysis)")
        _, voc = load(Path(args.vocals))
        result["sidechain"] = sidechain_diagnosis(sr, ref_m, test_m, voc)
        result["files"]["vocals"] = args.vocals

    # [7] per-scene
    if has_timeline:
        print("  · per-scene diff")
        timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
        scenes = timeline["scenes"]
        per_scene = per_scene_diff(sr, ref_m, test_m, scenes)
        result["per_scene"] = per_scene
        result["category"] = category_summary(per_scene)
        result["files"]["timeline"] = args.timeline
    else:
        result["per_scene"] = []
        result["category"]  = {}

    # [8] 종합
    diag, rec = synthesize_diagnosis(result)
    result["diagnosis"] = diag
    result["recommendations"] = rec

    # 출력 + JSON
    print()
    print(render_report(result))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n진단 JSON 저장: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
