"""objective_metrics_compare.py — 다조건 객관 지표 비교 (Phase P2).

Reference 대비 여러 test 조건의 객관 지표를 한 번에 측정해 비교 표 생성.

Phase P 5조건:
  · Reference (original_matched.wav)
  · Anchor    (anchor_matched.wav, mid 7kHz LPF)
  · V3.5.5    (v3_5_5_matched.wav, baseline V3.5.5)
  · V3.5.6    (v3_5_6_matched.wav, subtle pro)
  · V3.5.7    (v3_5_7_matched.wav, aggressive consumer)

각 condition vs Reference 로 측정. Reference 자체는 self-comparison (sanity).

출력:
  · tmp/objective_metrics_compare.json  — 모든 수치
  · tmp/objective_metrics_report_v2.md  — 발표용 비교 표

사용:
  PYTHONIOENCODING=utf-8 D:/Homecinema/venv/Scripts/python.exe \\
      tools/objective_metrics_compare.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# 동일 모듈의 metric 함수 재사용
from tools.objective_metrics import (
    load, to_mono, compute_all,
)


REPO = Path(__file__).resolve().parent.parent
TMP_TRAILER = REPO / "tmp" / "full_trailer" / "topgun"
EVAL_DIR    = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio" / "full_trailer" / "topgun"


# 비교 대상 — 평가 디렉토리 우선, 없으면 tmp
def _resolve(name: str) -> Path | None:
    for base in (EVAL_DIR, TMP_TRAILER):
        p = base / name
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="reference wav path (default: original_matched.wav)")
    ap.add_argument("--vocals", default=str(TMP_TRAILER / "stems" / "vocals.wav"))
    ap.add_argument("--out-json", default=str(REPO / "tmp" / "objective_metrics_compare.json"))
    ap.add_argument("--out-md",   default=str(REPO / "tmp" / "objective_metrics_report_v2.md"))
    args = ap.parse_args()

    ref_path = Path(args.ref) if args.ref else _resolve("original_matched.wav")
    if ref_path is None:
        raise FileNotFoundError("original_matched.wav 미발견 (--ref 지정)")

    # 조건 → 파일명 매핑
    candidates = [
        ("Anchor", _resolve("anchor_matched.wav")),
        ("V3.5.5", _resolve("v3_5_5_matched.wav")),
        ("V3.5.6", _resolve("v3_5_6_matched.wav")),
        ("V3.5.7", _resolve("v3_5_7_matched.wav")),
    ]
    test_conditions = [(name, p) for name, p in candidates if p is not None]

    if not test_conditions:
        raise FileNotFoundError("test 조건 wav 미발견")

    print("=" * 64)
    print("Phase P2 — 객관 지표 다조건 비교")
    print("=" * 64)
    print(f"Reference: {ref_path}")
    for name, p in test_conditions:
        print(f"  + {name:8s}: {p}")
    print()

    sr_r, ref = load(ref_path)
    voc = None
    voc_path = Path(args.vocals)
    if voc_path.exists():
        sr_v, voc = load(voc_path)
        if sr_v != sr_r:
            print(f"  ⚠ vocals SR {sr_v} ≠ ref SR {sr_r} → DPR/MRI 스킵")
            voc = None
        else:
            print(f"  vocals: {voc_path}\n")

    # 각 조건 측정
    results: dict[str, dict] = {}
    for name, test_path in test_conditions:
        print(f"--- {name} ---")
        sr_t, test = load(test_path)
        L = min(ref.shape[0], test.shape[0])
        results[name] = compute_all(sr_r, ref[:L], test[:L], voc)
        print()

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference": str(ref_path),
        "vocals_stem": str(voc_path) if voc is not None else None,
        "sr": sr_r,
        "n_conditions": len(test_conditions),
        "conditions": [name for name, _ in test_conditions],
    }
    out = {"meta": meta, "results": results}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 비교 표 (Markdown) ──
    md = render_compare_md(results, meta)
    Path(args.out_md).write_text(md, encoding="utf-8")

    print()
    print(md)
    print()
    print(f"JSON:     {args.out_json}")
    print(f"Markdown: {args.out_md}")


def render_compare_md(results: dict, meta: dict) -> str:
    names = list(results.keys())
    L: list[str] = []
    L.append("# Phase P — 객관 지표 다조건 비교 리포트")
    L.append("")
    L.append(f"- 생성: {meta['timestamp']}")
    L.append(f"- Reference: `{meta['reference']}`")
    L.append(f"- 조건: {' / '.join(names)}")
    L.append(f"- 길이/sr: {meta['sr']} Hz stereo")
    L.append("")

    L.append("## 표준 지표 (Reference 대비)")
    L.append("")
    header = "| 지표 | " + " | ".join(f"**{n}**" for n in names) + " | 해석 |"
    L.append(header)
    L.append("|---|" + "---:|" * len(names) + "---|")

    def row(label: str, key: str, fmt: str, hint: str, getter=None):
        if getter is None:
            getter = lambda r: r["standard"][key]
        cells = " | ".join(fmt.format(getter(results[n])) for n in names)
        L.append(f"| {label} | {cells} | {hint} |")

    row("STOI",        "STOI",        "{:.3f}",   "0~1, 1=완전 일치")
    row("ESTOI",       "ESTOI",       "{:.3f}",   "0~1, extended")
    row("SI-SDR (dB)", "SI_SDR_dB",   "{:+.2f}",  "양수=신호 우세, ≥10 dB 양호")
    row("IACC ref",    "IACC_ref",    "{:+.3f}",  "1=mono")
    row("IACC test",   "IACC_test",   "{:+.3f}",  "처리 후")
    row("**IACC Δ**",  "IACC_delta",  "{:+.3f}",  "음수=stereo 확장")
    row("LSD (dB)",    "LSD_dB",      "{:.2f}",   "0=동일, 작을수록 spectrum 유사")
    row("LUFS test",   "LUFS_test",   "{:+.2f}",  "ITU-R BS.1770-4")
    row("LUFS Δ (LU)", "LUFS_delta_LU", "{:+.2f}", "<1 LU 양호")
    row("Crest test",  "crest_test_dB","{:.2f}",  "peak/RMS, 동적 범위")

    L.append("")
    L.append("## 커스텀 지표 (Reference 대비)")
    L.append("")
    L.append(header)
    L.append("|---|" + "---:|" * len(names) + "---|")

    def cust_row(label: str, group: str, key: str, fmt: str, hint: str):
        cells = []
        for n in names:
            v = results[n].get(group, {}).get(key)
            cells.append(fmt.format(v) if v is not None and not (isinstance(v, float) and not np.isfinite(v)) else "—")
        L.append(f"| {label} | {' | '.join(cells)} | {hint} |")

    cust_row("DPR (dB)",  "custom_DPR", "dpr_db",   "{:+.2f}", "대사 1-4kHz 보존, 0근처=이상")
    cust_row("MRI (dB)",  "custom_MRI", "mri_db",   "{:+.2f}", "음악 5-15kHz Δ, 양수=강조")
    cust_row("**SEI (dB)**", "custom_SEI", "sei_db",   "{:+.2f}", "M/S spatial expansion")
    cust_row("CAS (dB)",  "custom_CAS", "cas_db",   "{:+.2f}", "Δ crest factor (음수=압축)")

    L.append("")
    L.append("## 시나리오 판정")
    L.append("")

    # V3.5.7 vs V3.5.6 자동 시나리오 판정
    if "V3.5.6" in results and "V3.5.7" in results:
        s6 = results["V3.5.6"]; s7 = results["V3.5.7"]
        sei6 = s6.get("custom_SEI", {}).get("sei_db", float("nan"))
        sei7 = s7.get("custom_SEI", {}).get("sei_db", float("nan"))
        stoi7 = s7["standard"]["STOI"]
        cas7  = s7.get("custom_CAS", {}).get("cas_db", float("nan"))
        iacc6 = s6["standard"]["IACC_delta"]
        iacc7 = s7["standard"]["IACC_delta"]

        L.append(f"### V3.5.6 vs V3.5.7 (aggressive 강화 효과)")
        L.append(f"- SEI Δ: {sei7 - sei6:+.2f} dB  (V3.5.7 가 spatial 더 확장)")
        L.append(f"- IACC Δ Δ: {iacc7 - iacc6:+.3f}  (음수일수록 V3.5.7 더 wider)")
        L.append(f"- V3.5.7 STOI: {stoi7:.3f}  ({'양호' if stoi7 >= 0.85 else '⚠ 명료도 손상'})")
        L.append(f"- V3.5.7 CAS: {cas7:+.2f} dB  ({'양호' if cas7 >= -6 else '⚠ 과도 압축'})")
        L.append("")
        if stoi7 >= 0.85 and cas7 >= -6.0:
            L.append("**시나리오 A**: V3.5.7 강화 효과 측정 가능 + artifact 없음 → MUSHRA 4조건 채택")
        else:
            L.append("**시나리오 B**: V3.5.7 artifact 발생 → 검토 필요")
        L.append("")

    if "Anchor" in results:
        anc = results["Anchor"]["standard"]
        L.append(f"### Anchor 검증 (Mid 7 kHz LPF)")
        L.append(f"- STOI: {anc['STOI']:.3f}  (Reference 대비)")
        L.append(f"- LSD: {anc['LSD_dB']:.2f} dB  (큰 spectrum 차이 = 정상 anchor)")
        L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
