"""Phase 4-A — COGNIMUSE OOD 일반화 측정 (Track A only).

BASE Model (X-CLIP + PANNs + Gate + K=7, 3-seed frozen weights at
`runs/phase2a/2a2_A_K7_s{42,123,2024}/best.pt`)을 COGNIMUSE 12편 Hollywood
영화의 experienced emotion annotation에 대해 inference 만 수행하는 패키지.

OAT 원칙 (BASE 파일 0개 수정):
    - BASE weights: read-only load, re-save 금지
    - 모든 출력은 `runs/cognimuse/phase4a/` 하위
    - 모든 feature는 `data/features/cognimuse_panns_v5spec/` 하위
    - precompute 상수 (4s crop / 2s stride / 10s pad / 8 frames / 224px) BASE와 bit-identical

구성:
    preprocessing.py         — .dat 14 traces → per-frame median → 10s window → CSV
    analyze_distribution.py  — Phase 0 gate (LIRIS vs COGNIMUSE distribution)
    precompute_cognimuse.py  — precompute_liris.py fork, start_sec/end_sec 지원
    run_eval_ood.py          — run_test_eval.py fork, Raw CCC + per-film z-score CCC
    tests/                   — parser · segmentation · ckpt isolation

Annotation 채널: experienced primary (12 traces per film, per-frame median).
영상 범위: 12 Hollywood films (Travel docs/GWW는 emotion annotation 없음).

해석 가이드 (per-film z-score CCC 기준):
    ≥ 0.30 — 강한 일반화
    0.15~0.30 — 부분 일반화 (표준)
    0.05~0.15 — 약한 일반화
    < 0.05 — 일반화 실패
"""
