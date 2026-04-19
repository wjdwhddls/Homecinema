"""objective_metrics.py — V3.5.6 처리 객관 지표 (Phase O).

표준 지표:
  · STOI  — Short-Time Objective Intelligibility (Taal et al., 2010)
  · ESTOI — Extended STOI (Jensen & Taal, 2016)
  · SI-SDR — Scale-Invariant Signal-to-Distortion Ratio (Le Roux et al., 2019)
  · IACC  — Inter-Aural Cross-Correlation (concert hall spatial impression)
  · LSD   — Log-Spectral Distance (per-frame, dB)

커스텀 지표 (V3.5.6 평가용):
  · DPR — Dialogue Protection Ratio
        대사 활성 구간(vocals envelope > -30 dB)의 1-4 kHz 대역 RMS 차이 (test - ref).
        양수면 대사가 처리 후 더 강조됨 (protection 효과).
  · MRI — Music Restoration Index
        대사 비활성 구간의 high-frequency content 변화 + side/mid 변화 종합.
  · SEI — Spatial Expansion Index
        Δ(side/mid) ratio in dB. V3.5.6 기대 +1.5~+2.5.
  · CAS — Compression Aggressiveness Score
        crest factor (peak/RMS) 변화. test_crest - ref_crest. 음수면 압축 강해짐.

입력: stereo WAV (LUFS 매칭된 ref/test). vocals stem 있으면 DPR 정확.
출력: JSON + Markdown 리포트.

사용:
  PYTHONIOENCODING=utf-8 D:/Homecinema/venv/Scripts/python.exe \\
      tools/objective_metrics.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from pystoi import stoi
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt


REPO = Path(__file__).resolve().parent.parent
TMP = REPO / "tmp" / "full_trailer" / "topgun"
JOB = REPO / "backend" / "data" / "jobs" / "fe2ecad8-dc25-4131-adfe-ffeea6d977a1"


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


def to_mono(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=1) if a.ndim == 2 else a


def db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


# ────────────────────────────────────────────────────────
# 표준 지표
# ────────────────────────────────────────────────────────
def metric_stoi(sr: int, ref: np.ndarray, test: np.ndarray, extended: bool) -> float:
    """STOI / ESTOI. 모노 입력. pystoi 내부에서 10kHz 리샘플."""
    L = min(ref.shape[0], test.shape[0])
    return float(stoi(to_mono(ref[:L]), to_mono(test[:L]), sr, extended=extended))


def metric_si_sdr(ref: np.ndarray, test: np.ndarray) -> float:
    """SI-SDR (Le Roux 2019). 모노 입력 가정.

    s_target = (<s_hat, s> / ||s||²) * s
    e = s_hat - s_target
    SI-SDR = 10 log10(||s_target||² / ||e||²)
    """
    s     = to_mono(ref).astype(np.float64)
    s_hat = to_mono(test).astype(np.float64)
    L = min(len(s), len(s_hat))
    s = s[:L]; s_hat = s_hat[:L]
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    alpha = float(np.dot(s_hat, s) / (np.dot(s, s) + 1e-12))
    s_tar = alpha * s
    e     = s_hat - s_tar
    num = float(np.dot(s_tar, s_tar))
    den = float(np.dot(e, e)) + 1e-12
    return 10.0 * np.log10(num / den + 1e-12)


def metric_iacc(stereo: np.ndarray, sr: int, max_lag_ms: float = 1.0) -> float:
    """IACC: ±1ms lag 범위 내 L/R cross-correlation 최대값.

    값 범위 [-1, 1]. 1에 가까우면 거의 mono(L=R), 0이면 무상관(완전 분리).
    공간감(spaciousness) ↔ 1-IACC.
    """
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        return float("nan")
    L = stereo[:, 0]; R = stereo[:, 1]
    L = L - L.mean(); R = R - R.mean()
    n_lag = int(sr * max_lag_ms / 1000.0)
    norm = np.sqrt(np.dot(L, L) * np.dot(R, R) + 1e-12)
    best = -1.0
    for k in range(-n_lag, n_lag + 1):
        if k >= 0:
            c = float(np.dot(L[:len(L)-k], R[k:]))
        else:
            c = float(np.dot(L[-k:], R[:len(R)+k]))
        v = c / norm
        if v > best:
            best = v
    return best


def metric_lsd(sr: int, ref: np.ndarray, test: np.ndarray, frame_ms: float = 25.0) -> float:
    """Log-Spectral Distance (dB). per-frame STFT magnitude log 차이의 RMS."""
    a = to_mono(ref); b = to_mono(test)
    L = min(len(a), len(b))
    n = int(sr * frame_ms / 1000.0)
    hop = n // 2
    win = np.hanning(n)
    out = []
    for i in range(0, L - n, hop):
        A = np.abs(np.fft.rfft(a[i:i+n] * win))
        B = np.abs(np.fft.rfft(b[i:i+n] * win))
        log_diff = 20.0 * (np.log10(np.maximum(A, 1e-10)) - np.log10(np.maximum(B, 1e-10)))
        out.append(float(np.sqrt(np.mean(log_diff ** 2))))
    return float(np.mean(out)) if out else float("nan")


def metric_lufs(sr: int, a: np.ndarray) -> float:
    return float(pyln.Meter(sr).integrated_loudness(a))


def metric_crest_factor_db(a: np.ndarray) -> float:
    """Crest factor: peak / rms in dB. 동적 범위 지표."""
    p = float(np.abs(a).max())
    r = rms(a)
    return db(p) - db(r)


# ────────────────────────────────────────────────────────
# 커스텀 지표
# ────────────────────────────────────────────────────────
def vocals_envelope_db(sr: int, vocals: np.ndarray, smooth_ms: float = 50.0) -> np.ndarray:
    """Mono vocals envelope (1-pole EMA on |x|), dB scale."""
    mono = to_mono(vocals)
    abs_x = np.abs(mono).astype(np.float64)
    n = max(1, int(sr * smooth_ms / 1000.0))
    alpha = 1.0 / n
    out = np.empty_like(abs_x)
    acc = 0.0
    for i in range(len(abs_x)):
        acc = (1 - alpha) * acc + alpha * abs_x[i]
        out[i] = acc
    return 20 * np.log10(np.maximum(out, 1e-10))


def bandpass(sr: int, a: np.ndarray, lo: float, hi: float, order: int = 6) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, a, axis=0)


def metric_dpr(
    sr: int,
    ref: np.ndarray,
    test: np.ndarray,
    vocals_env_db: np.ndarray,
    threshold_db: float = -30.0,
) -> dict:
    """Dialogue Protection Ratio.

    대사 활성 구간의 1-4 kHz 대역 RMS 차이 (test - ref).
    양수 = 대사 강조. 음수 = 대사 약화. 0 근처 = 보존.
    """
    L = min(len(ref), len(test), len(vocals_env_db))
    ref_b = bandpass(sr, ref[:L], 1000.0, 4000.0)
    test_b = bandpass(sr, test[:L], 1000.0, 4000.0)
    mask = vocals_env_db[:L] > threshold_db
    if not mask.any():
        return {"dpr_db": float("nan"), "dialogue_pct": 0.0}
    ref_active = ref_b[mask]
    test_active = test_b[mask]
    return {
        "dpr_db": db(rms(test_active)) - db(rms(ref_active)),
        "dialogue_pct": float(mask.mean() * 100),
        "ref_dialogue_band_rms_db": db(rms(ref_active)),
        "test_dialogue_band_rms_db": db(rms(test_active)),
    }


def metric_mri(
    sr: int,
    ref: np.ndarray,
    test: np.ndarray,
    vocals_env_db: np.ndarray,
    threshold_db: float = -30.0,
) -> dict:
    """Music Restoration Index.

    대사 비활성 구간 (vocals env ≤ threshold)에서:
      · high-band (5-15 kHz) RMS 변화 (음악의 air/sparkle)
      · 전체 RMS 변화
    양수면 음악이 처리 후 더 활성/풍부.
    """
    L = min(len(ref), len(test), len(vocals_env_db))
    mask = vocals_env_db[:L] <= threshold_db
    if not mask.any():
        return {"mri_db": float("nan"), "music_pct": 0.0}
    ref_quiet = ref[:L][mask]
    test_quiet = test[:L][mask]
    ref_hi = bandpass(sr, ref[:L], 5000.0, 15000.0)[mask]
    test_hi = bandpass(sr, test[:L], 5000.0, 15000.0)[mask]
    high_db_delta = db(rms(test_hi)) - db(rms(ref_hi))
    full_db_delta = db(rms(test_quiet)) - db(rms(ref_quiet))
    return {
        "mri_db": high_db_delta,
        "music_pct": float(mask.mean() * 100),
        "high_band_db_delta": high_db_delta,
        "full_band_db_delta": full_db_delta,
    }


def metric_sei(ref: np.ndarray, test: np.ndarray) -> dict:
    """Spatial Expansion Index. Δ(side/mid in dB)."""
    def sm(a):
        L, R = a[:, 0], a[:, 1]
        m = 0.5 * (L + R); s = 0.5 * (L - R)
        return db(rms(s)) - db(rms(m))
    s_ref = sm(ref); s_test = sm(test)
    return {
        "sei_db":            s_test - s_ref,
        "ref_side_over_mid": s_ref,
        "test_side_over_mid": s_test,
    }


def metric_cas(ref: np.ndarray, test: np.ndarray) -> dict:
    """Compression Aggressiveness Score. test crest − ref crest (in dB).
    음수 = test 가 더 압축됨 (peak/RMS 비율 작음)."""
    cr_ref = metric_crest_factor_db(to_mono(ref))
    cr_test = metric_crest_factor_db(to_mono(test))
    return {
        "cas_db":          cr_test - cr_ref,
        "ref_crest_db":   cr_ref,
        "test_crest_db":  cr_test,
    }


# ────────────────────────────────────────────────────────
# 종합
# ────────────────────────────────────────────────────────
def compute_all(
    sr: int,
    ref_stereo: np.ndarray,
    test_stereo: np.ndarray,
    vocals_stereo: np.ndarray | None,
) -> dict:
    print("  · STOI")
    stoi_v  = metric_stoi(sr, ref_stereo, test_stereo, extended=False)
    print("  · ESTOI")
    estoi_v = metric_stoi(sr, ref_stereo, test_stereo, extended=True)
    print("  · SI-SDR")
    sisdr_v = metric_si_sdr(ref_stereo, test_stereo)
    print("  · IACC")
    iacc_ref  = metric_iacc(ref_stereo, sr)
    iacc_test = metric_iacc(test_stereo, sr)
    print("  · LSD")
    lsd_v = metric_lsd(sr, ref_stereo, test_stereo)
    print("  · LUFS / crest")
    lufs_ref  = metric_lufs(sr, ref_stereo)
    lufs_test = metric_lufs(sr, test_stereo)
    crest_ref  = metric_crest_factor_db(to_mono(ref_stereo))
    crest_test = metric_crest_factor_db(to_mono(test_stereo))

    result = {
        "standard": {
            "STOI":            stoi_v,
            "ESTOI":           estoi_v,
            "SI_SDR_dB":       sisdr_v,
            "IACC_ref":        iacc_ref,
            "IACC_test":       iacc_test,
            "IACC_delta":      iacc_test - iacc_ref,
            "LSD_dB":          lsd_v,
            "LUFS_ref":        lufs_ref,
            "LUFS_test":       lufs_test,
            "LUFS_delta_LU":   lufs_test - lufs_ref,
            "crest_ref_dB":    crest_ref,
            "crest_test_dB":   crest_test,
        }
    }

    if vocals_stereo is not None:
        print("  · vocals envelope (DPR/MRI prep)")
        env_db = vocals_envelope_db(sr, vocals_stereo, smooth_ms=50.0)
        print("  · DPR (Dialogue Protection)")
        result["custom_DPR"] = metric_dpr(sr, ref_stereo, test_stereo, env_db)
        print("  · MRI (Music Restoration)")
        result["custom_MRI"] = metric_mri(sr, ref_stereo, test_stereo, env_db)

    print("  · SEI (Spatial Expansion)")
    result["custom_SEI"] = metric_sei(ref_stereo, test_stereo)
    print("  · CAS (Compression Aggressiveness)")
    result["custom_CAS"] = metric_cas(ref_stereo, test_stereo)
    return result


# ────────────────────────────────────────────────────────
# 리포트 (Markdown)
# ────────────────────────────────────────────────────────
def render_markdown(d: dict, meta: dict) -> str:
    s = d["standard"]
    L = []
    L.append(f"# V3.5.6 객관 지표 리포트 (Phase O)")
    L.append("")
    L.append(f"- 생성: {meta['timestamp']}")
    L.append(f"- Reference: `{meta['ref']}`")
    L.append(f"- Test:      `{meta['test']}`")
    if "vocals" in meta:
        L.append(f"- Vocals stem (DPR/MRI 용): `{meta['vocals']}`")
    L.append(f"- 길이: {meta['duration_sec']:.2f} s, sr {meta['sr']} Hz")
    L.append("")

    L.append("## 표준 지표")
    L.append("")
    L.append("| 지표 | 값 | 해석 |")
    L.append("|---|---:|---|")
    L.append(f"| STOI  | {s['STOI']:.4f}  | 0~1, 1=완전 일치. 0.9↑ 양호 |")
    L.append(f"| ESTOI | {s['ESTOI']:.4f} | extended, 비정상 신호 강건 |")
    L.append(f"| SI-SDR | {s['SI_SDR_dB']:+.2f} dB | 양수=신호 우세, ≥10 dB 양호 |")
    L.append(f"| IACC ref  | {s['IACC_ref']:+.4f}  | 1=mono, 낮을수록 stereo 폭 |")
    L.append(f"| IACC test | {s['IACC_test']:+.4f} | 처리 후 |")
    L.append(f"| **IACC Δ** | **{s['IACC_delta']:+.4f}** | 음수=stereo 확장 |")
    L.append(f"| LSD | {s['LSD_dB']:.2f} dB | 0=동일, 작을수록 spectrum 유사 |")
    L.append(f"| LUFS ref  | {s['LUFS_ref']:+.2f} | ITU-R BS.1770-4 |")
    L.append(f"| LUFS test | {s['LUFS_test']:+.2f} | |")
    L.append(f"| LUFS Δ | {s['LUFS_delta_LU']:+.2f} LU | \\|Δ\\|<1 매칭 양호 |")
    L.append(f"| Crest ref  | {s['crest_ref_dB']:.2f} dB  | peak/RMS, 동적 범위 |")
    L.append(f"| Crest test | {s['crest_test_dB']:.2f} dB | |")
    L.append("")

    L.append("## 커스텀 지표 (V3.5.6 평가용)")
    L.append("")
    if "custom_DPR" in d:
        dpr = d["custom_DPR"]
        L.append(f"### DPR — Dialogue Protection Ratio")
        L.append(f"- DPR: **{dpr['dpr_db']:+.2f} dB** (1-4 kHz, vocals 활성 구간)")
        L.append(f"- 대사 활성 시간: {dpr['dialogue_pct']:.1f} %")
        L.append(f"- 양수=대사 강조, 음수=약화, 0 근처=보존")
        L.append("")
    if "custom_MRI" in d:
        mri = d["custom_MRI"]
        L.append(f"### MRI — Music Restoration Index")
        L.append(f"- MRI (5-15 kHz Δ in music regions): **{mri['mri_db']:+.2f} dB**")
        L.append(f"- 음악 비대사 구간: {mri['music_pct']:.1f} %")
        L.append(f"- 양수=음악 air/sparkle 강조")
        L.append("")
    sei = d["custom_SEI"]
    L.append(f"### SEI — Spatial Expansion Index")
    L.append(f"- SEI: **{sei['sei_db']:+.2f} dB** (Δ side/mid)")
    L.append(f"- ref side/mid: {sei['ref_side_over_mid']:+.2f} dB → test: {sei['test_side_over_mid']:+.2f} dB")
    L.append(f"- V3.5.6 기대 +1.5 ~ +2.5 dB. 양수=공간 확장")
    L.append("")
    cas = d["custom_CAS"]
    L.append(f"### CAS — Compression Aggressiveness Score")
    L.append(f"- CAS: **{cas['cas_db']:+.2f} dB** (Δ crest factor)")
    L.append(f"- ref crest: {cas['ref_crest_db']:.2f} dB → test crest: {cas['test_crest_db']:.2f} dB")
    L.append(f"- 음수=test 더 압축됨 (peak/RMS 비율 감소)")
    L.append("")

    # 종합 평가
    L.append("## 종합 해석")
    L.append("")
    notes = []
    if s["STOI"] >= 0.95:
        notes.append("- ✓ STOI ≥ 0.95: 명료도 거의 손실 없음")
    elif s["STOI"] >= 0.85:
        notes.append(f"- ○ STOI {s['STOI']:.3f}: 명료도 약간 변화")
    else:
        notes.append(f"- ⚠ STOI {s['STOI']:.3f}: 명료도 손실 의심")
    if abs(s["SI_SDR_dB"]) > 30:
        notes.append(f"- ✓ SI-SDR 큼: ref와 test 매우 일치 (글로벌 distortion 작음)")
    elif s["SI_SDR_dB"] > 10:
        notes.append(f"- ○ SI-SDR {s['SI_SDR_dB']:.1f} dB: 적정 처리")
    else:
        notes.append(f"- ⚠ SI-SDR {s['SI_SDR_dB']:.1f} dB: 처리 강도 큼 (의도적 enhancement)")
    if s["IACC_delta"] < -0.05:
        notes.append(f"- ✓ IACC Δ {s['IACC_delta']:+.3f}: stereo 폭 명확히 확장")
    elif s["IACC_delta"] > 0.05:
        notes.append(f"- ⚠ IACC Δ {s['IACC_delta']:+.3f}: stereo 폭 축소 (의도와 다름?)")
    else:
        notes.append(f"- ○ IACC Δ {s['IACC_delta']:+.3f}: stereo 폭 변화 미미")
    if "custom_SEI" in d and d["custom_SEI"]["sei_db"] >= 1.0:
        notes.append(f"- ✓ SEI {d['custom_SEI']['sei_db']:+.2f} dB: M/S processing 효과 측정됨")
    if "custom_DPR" in d:
        dpr_v = d["custom_DPR"]["dpr_db"]
        if abs(dpr_v) < 1.0:
            notes.append(f"- ○ DPR {dpr_v:+.2f} dB: 대사 대역 보존")
        elif dpr_v > 0:
            notes.append(f"- ⚠ DPR {dpr_v:+.2f} dB: 대사 대역 강조")
        else:
            notes.append(f"- ⚠ DPR {dpr_v:+.2f} dB: 대사 대역 약화 (의도 검토)")
    L.extend(notes)
    L.append("")

    return "\n".join(L)


# ────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",  default=str(TMP / "original_matched.wav"))
    ap.add_argument("--test", default=str(TMP / "v3_5_6_matched.wav"))
    ap.add_argument("--vocals", default=str(TMP / "stems" / "vocals.wav"))
    ap.add_argument("--out-json", default=str(REPO / "tmp" / "objective_metrics.json"))
    ap.add_argument("--out-md",   default=str(REPO / "tmp" / "objective_metrics_report.md"))
    ap.add_argument("--label", default="V3.5.6", help="test label (예: V3.5.6_fixed)")
    args = ap.parse_args()

    print(f"=== 객관 지표 측정 ({args.label}) ===")
    print(f"  ref  = {args.ref}")
    print(f"  test = {args.test}")

    sr_r, ref  = load(Path(args.ref))
    sr_t, test = load(Path(args.test))
    if sr_r != sr_t:
        raise ValueError(f"SR mismatch: ref={sr_r} test={sr_t}")
    sr = sr_r
    L = min(ref.shape[0], test.shape[0])
    ref = ref[:L]; test = test[:L]

    voc = None
    if Path(args.vocals).exists():
        sr_v, voc = load(Path(args.vocals))
        if sr_v != sr:
            print(f"  ⚠ vocals SR {sr_v} ≠ {sr}, skipping DPR/MRI")
            voc = None
        else:
            print(f"  vocals = {args.vocals}")

    print()
    metrics = compute_all(sr, ref, test, voc)

    meta = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "ref":          args.ref,
        "test":         args.test,
        "label":        args.label,
        "sr":           sr,
        "samples":      L,
        "duration_sec": L / sr,
    }
    if voc is not None:
        meta["vocals"] = args.vocals

    out = {"meta": meta, "metrics": metrics}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(metrics, meta)
    Path(args.out_md).write_text(md, encoding="utf-8")

    print()
    print(md)
    print()
    print(f"JSON:     {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
