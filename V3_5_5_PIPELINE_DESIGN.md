# V3.5.5 파이프라인 설계 문서

**작성일**: Day 9 (2026-04-18)
**상태**: Step 4 구현 직전 확정
**근거 Step**: Step 1 (Demucs 설치), Step 2 (탑건 분리, HALT 2 통과), Step 3 (라라랜드 분리, HALT 3 "보통" 판정)

---

## 1. 목표

V3.3 풀믹스 EQ의 한계를 극복하기 위해 Demucs 2-stem 분리를 도입:

- **음악/SFX (no_vocals)**: V3.3 카테고리 프리셋 그대로 적용 → 정서 강조
- **대사 (vocals)**: 명료도 전용 EQ → 대사 선명도 향상
- **합산 + Compressor**: V3.3와 동일

이로써 "음악 정서 강조 vs 대사 명료도" trade-off를 구조적으로 분리.

---

## 2. 검증된 전제 (Step 1~3 결과)

| 항목 | 상태 | 값 |
|---|---|---|
| Demucs 모델 | 확정 | htdemucs (4-stem → vocals/no_vocals 합성) |
| 처리 환경 | 확정 | CPU, torchcodec 비활성 (scipy wavfile 저장) |
| 리샘플 경로 | 확정 | 48000 → 44100 → 48000 Hz |
| 탑건 재결합 SNR | 검증 | +28.39 dB (residual -47.46 dBFS) |
| 탑건 청취 판정 | HALT 2 GO | 대사/음악 분리 양호 |
| 라라랜드 청취 판정 | HALT 3 GO with 주의 | 허밍 leak 보통 수준 |
| 분리 소요 시간 | 기준치 | ~20초 / 11초 클립 (CPU 전체 파이프라인) |

**한계 인지**: 라라랜드는 뮤지컬이라 vocals leak이 no_vocals에 남음. no_vocals에 강한 EQ 걸면 유령 목소리가 같이 부스트될 위험.

---

## 3. 파이프라인 전체 흐름

```
Input: original.wav (48 kHz stereo)
  │
  ├─► Demucs htdemucs 분리
  │     │
  │     ├─► vocals.wav         (대사 stem)
  │     └─► no_vocals.wav      (drums + bass + other 합산)
  │
  ├─► Vocals 처리:
  │     └─► 명료도 EQ 체인 적용 → vocals_eq.wav
  │
  ├─► No_vocals 처리:
  │     └─► V3.3 카테고리 EQ 적용 (씬의 지배 mood) → no_vocals_eq.wav
  │
  ├─► 합산: vocals_eq + no_vocals_eq → mixed.wav
  │
  ├─► Compressor (V3.3와 동일 파라미터) → compressed.wav
  │
  └─► 길이 보장 (crop/pad to original length) → v3_5_5.wav
```

---

## 4. 주요 설계 결정

### 4-1. Vocals 전용 EQ 체인

**목적**: 대사 자음(consonant)·마찰음(fricative) 선명도 향상, 저역 머플링 제거.

| 밴드 | 주파수 | Gain | 근거 |
|---|---|---|---|
| B3 | 125 Hz | **-2.0 dB** | boomy 제거 |
| B4 | 250 Hz | -1.0 dB | 저역 머플링 완화 |
| B6 | 1 kHz | +1.0 dB | 음성 presence |
| B7 | 2 kHz | **+2.5 dB** | consonant (ANSI SII 핵심 대역) |
| B8 | 4 kHz | **+2.0 dB** | fricative / sibilance |
| 나머지 | — | 0 dB | 보존 |

Q는 V3.x 기본값 사용 (B3=1.0, B4=1.2, B6=1.4, B7=1.4, B8=1.2).

**주의**: 이 EQ는 **모든 씬 공통**. 씬 카테고리와 무관. 대사 명료도는 장르 독립적이므로 고정 값으로 처리.

---

### 4-2. No_vocals EQ 체인

**핵심 결정: 1차는 V3.3 EQ 그대로 적용 (±6dB 강도 유지)**

**근거**:
- PoC 로직: 변수 하나(Demucs 분리)만 바꿔서 효과 isolation
- V3.3 대비 V3.5.5의 개선이 "분리 덕분"인지 "EQ 완화 덕분"인지 구분하려면 EQ는 동일 조건
- 라라랜드 vocal leak이 유령 목소리로 증폭될 위험은 **인지하되 실측 확인**
- 만약 실제로 유령 목소리 문제 발생 시 → 2차 이터레이션에서 `no_vocals_gain_coeff = 0.7 or 0.5` 도입

**V3.3 카테고리 EQ 프리셋**: `model/autoEQ/inference/eq_engine.py`의 `EQ_PRESETS_V3_3` 참조. 씬 mood의 지배 카테고리(`scene.aggregated.category`)로 선택.

**2차 이터레이션 조건 (참고)**:
- 라라랜드 청취에서 "반주 부스트 시 허밍이 유령처럼 또렷해짐" 체감되면
- Conservative 모드 추가: `no_vocals_eq_coeff ∈ {0.5, 0.7}`
- 이번 Step 4 구현에서는 **구현만 하지 않음. 단, 함수 인자로 `eq_intensity` 파라미터 두어 향후 확장 용이하게.**

---

### 4-3. Compressor (V3.3 계승, 변경 없음)

| 파라미터 | 값 |
|---|---|
| threshold | -12 dB |
| ratio | 3 : 1 |
| attack | 10 ms |
| release | 100 ms |
| makeup gain | +3 dB |

**주의**: Compressor는 합산 **후에만** 적용. vocals/no_vocals 각각에 걸지 않음. Dynamic range 최종 조정 목적.

---

### 4-4. 길이 보장

Demucs 내부 리샘플(48→44.1→48kHz)로 인해 sample 단위 drift 가능. 원본 길이 기준으로:
- drift < 10 samples: crop (초과분 제거)
- drift ≥ 10 samples: warning 로그 후 crop
- 출력이 원본보다 짧은 경우: zero-pad

**기준**: 원본 파일의 sample 개수를 target으로 모든 처리 결과 정렬.

---

### 4-5. Peak / 클리핑 관리

V3.3 이미 Compressor + makeup 조합으로 peak 관리 검증됨. V3.5.5도 동일 체인이므로 추가 limiter 불필요.

**단, 분리 artifact가 clip 트리거할 경우 대비**:
- 최종 출력 peak > -0.1 dBFS면 warning 로그
- 실제 clip 발생 시 makeup gain -1dB 일시 축소 (fallback)

---

## 5. 씬별 테스트 계획

### 5-1. 테스트 대상 (Step 4에서 생성할 파일)

| 씬 | 카테고리 | 출력 파일 |
|---|---|---|
| scene_topgun_category_eq | Tension/Power (씬 mood에 따름) | v3_5_5.wav |
| scene_lalaland_wonder | Wonder | v3_5_5.wav |

**재사용 자산**:
- `tmp/demucs_test/htdemucs/topgun_cat_eq/{vocals,no_vocals}.wav`
- `tmp/demucs_test/htdemucs/lalaland_wonder/{vocals,no_vocals}.wav`

이미 분리된 stem 재사용. 재분리하지 않음 (시간 절약).

### 5-2. 씬 mood 카테고리 확인

timeline.json 또는 기존 job 결과에서 각 씬의 지배 카테고리 조회:
- 탑건 category_eq: `backend/data/jobs/fe2ecad8-.../timeline.json`
- 라라랜드 wonder: `backend/data/jobs/lalaland-demo/timeline.json`

씬 시간 구간에 해당하는 `aggregated.category` 또는 `category_index` 읽기.

---

## 6. 청취 비교 판정 기준 (HALT 5 / Step 6)

**비교**: v3_3.wav vs v3_5_5.wav (동일 씬 A/B)

| 항목 | 개선 | 동일 | 악화 |
|---|---|---|---|
| 대사 명료도 | ★ | · | ✗ |
| 음악 정서 강조 | ★ | · | ✗ |
| 전체 자연스러움 | ★ | · | ✗ |

**판정 매트릭스**:

| 결과 | 판정 |
|---|---|
| 대사 ★ + 음악 ★ + 자연 ★ | **채택**: V3.5.5 정식 도입 |
| 대사 ★ + 음악 · / 자연 · | 채택: 부작용 없는 개선 |
| 대사 ★ + 음악 ✗ | **재조정**: no_vocals EQ 강도 완화 (2차 이터레이션) |
| 대사 · + 음악 · | 보류: Demucs 오버헤드 대비 효용 부족 → V3.3 최종 유지 |
| 대사 ✗ or 자연 ✗ | **거부**: Demucs 포기, V3.3 최종 |

---

## 7. 구현 체크리스트 (Step 4 요구사항)

- [ ] `tools/run_v3_5_5_pipeline.py` 작성
- [ ] 함수 시그니처: `apply_v3_5_5(original_wav, scene_category, output_wav, eq_intensity=1.0)`
- [ ] V3.3 EQ 프리셋 로드 (`from model.autoEQ.inference.eq_engine import EQ_PRESETS_V3_3`)
- [ ] Vocals 명료도 EQ 체인 상수 정의 (4-1 섹션 값)
- [ ] Compressor 설정 상수 정의 (4-3 섹션 값)
- [ ] Demucs 분리 결과 재사용 (`tmp/demucs_test/htdemucs/*`)
- [ ] 길이 정렬 로직 (4-4 섹션)
- [ ] Peak 검증 로직 (4-5 섹션)
- [ ] 탑건 / 라라랜드 두 씬에 대해 실행
- [ ] 출력 경로: `evaluation/webmushra/configs/resources/audio/scene_*/v3_5_5.wav`

---

## 8. 위험 / 한계 정리

### 8-1. 라라랜드 유령 목소리 위험 (주의)
허밍 leak이 no_vocals에 남아있고 V3.3 Wonder 프리셋 (고역 +1.5 / +2.0 dB) 적용되면 유령 목소리가 더 또렷해질 가능성. **청취 검증에서 이 현상 여부 집중 체크**.

### 8-2. Compressor가 artifact 증폭 가능성
분리 경계에서 발생할 수 있는 저레벨 noise를 Compressor의 makeup gain이 끌어올릴 수 있음. 대사 구간 사이 침묵에서 "쉬익" 소리 잔존 여부 체크.

### 8-3. CPU 처리 비용
~20초 / 11초 클립은 real-time factor 약 1.8x. 실시간 스트리밍 불가. **분석 단계에서 프리컴퓨트하여 저장하는 방식만 지원**. 재생 파이프라인 아키텍처 영향 없음 (V3.2 명세 부합).

### 8-4. 프리셋 버전 관리
V3.5.5는 "Demucs + V3.3 EQ + 명료도 EQ + V3.3 Compressor"의 조합명. `PRESET_VERSIONS` 딕셔너리에 `"v3_5_5"` 엔트리 추가 필요 (Step 4에서 함께 처리).

---

## 9. 다음 Step 간 흐름

```
Step 4 [지금]
  ├─ run_v3_5_5_pipeline.py 작성
  ├─ 탑건 / 라라랜드 v3_5_5.wav 생성
  └─ HALT 4: 파일 생성 확인 → 청취 검증 전 STOP

Step 5
  ├─ simple_player_v3 확장 (5 condition → 6 condition, v3_5_5 추가)
  └─ HALT 5: 비교 가능 상태 확인

Step 6
  ├─ A/B 청취: v3_3 vs v3_5_5 (탑건 / 라라랜드 각각)
  ├─ 판정 매트릭스 적용
  └─ 결론: 채택 / 재조정 / 거부
```

**문서 끝.**
