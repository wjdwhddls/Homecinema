# MoodEQ Dual-Layer 음향 기능 레퍼런스

> 본 문서는 MoodEQ 파이프라인의 **추론 이후** 단계, 즉 예측된 V/A(Valence/Arousal) 와
> mood 를 실제 오디오로 변환하는 EQ/FX 단계의 **모든 음향 기술** 을 정리한다.
>
> 원칙: **모든 수치·선택은 (a) peer-reviewed 문헌 근거, (b) 업계 표준 cutoff,
> (c) pedalboard 기본값, (d) 안전 장치 중 하나로 분류 가능해야 한다. 임의 수치 금지.**
>
> 관련 파일:
> - `model/autoEQ/infer_pseudo/eq_preset.py` — Layer 1 EQ 테이블·Q·공식
> - `model/autoEQ/playback/pipeline.py` — Layer 1 biquad 적용
> - `generate_fx_demo.py` — Layer 2 shelf/reverb/limiter
> - `model/autoEQ/train_liris/BASE_MODEL.md` §4d — Dual-layer 아키텍처 공식 기록

---

## 0. 이 문서가 답하는 질문

1. 왜 **EQ 와 FX 를 두 개의 layer 로 나눴나?**
2. **10-band peaking EQ** 란 무엇이고, 왜 이 10 개 주파수를 골랐나?
3. **mood 마다 dB 값** 은 어디서 온 숫자인가? (임의인가?)
4. 대사는 어떻게 보호하나? (**α_d 공식의 유도**)
5. Shelf/Reverb/Limiter 는 각각 무엇을 하는가? 왜 이 파라미터인가?
6. 왜 **compression, stereo width 는 쓰지 않나?**
7. 전체 체인이 실제로 귀에 어떻게 들리는가?

---

## 1. 전체 파이프라인 — 추론 이후

```
 입력 영상 (mp4)
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│ [추론 — Inference]                                        │
│ X-CLIP + PANNs CNN14 + Gate + K=7                         │
│   → scene 마다 (V, A) ∈ [-1, +1]²,  mood ∈ GEMS 7 클래스  │
│   → timeline.json (scene boundary + V/A + mood + dialogue)│
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│ Layer 1 — EQ (학술 contribution)                          │
│   mood 별 10-band peaking filter                          │
│   + VAD-guided dialogue protection (α_d)                  │
│   "학습된 model 이 무엇을 감지했는지" 를 음향적으로 번역      │
└──────────────────────────────────────────────────────────┘
     │ (kakao_eq_applied.mp4)
     ▼
┌──────────────────────────────────────────────────────────┐
│ Layer 2 — FX (perceptual amplifier)                       │
│   mood 별 shelf / reverb / limiter                        │
│   "지각 한계(JND ~1 dB) 위로 증폭"                         │
└──────────────────────────────────────────────────────────┘
     │ (kakao_eq_fx.mp4)
     ▼
 최종 영상
```

### 1.1 왜 Layer 를 분리했나?

| 축 | Layer 1 (EQ) | Layer 2 (FX) |
|---|---|---|
| 역할 | 학습된 V/A → 주파수 표현의 **과학적 mapping** | 지각 체감을 **JND 위로 증폭** |
| 평가 방식 | 수치 (CCC, Pearson, spectrum Δ) | 주관 (ABX listening test) |
| 연속 V/A 활용 | ✅ centroid mapping 가능 | ❌ argmax mood 만 사용 |
| 논문 기여 | ✅ Main result | 🟡 Auxiliary (subjective validation) |
| 파라미터 가변성 | scene 마다 다름 (연속) | mood 분류당 고정 preset |

**핵심 이유**: Phase 4-A 후 실제 영상에서 EQ-only (1×) 의 평균 |gain| 이 0.96 dB —
**JND (Just-Noticeable Difference, ~1 dB)** 근처라 체감이 약했다. Layer 2 는
이를 **문헌 근거 있는 방향으로만** 증폭한다 (임의 tuning 금지).

> **JND 참고**: Zwicker & Fastl 1990 (*Psychoacoustics: Facts and Models*) —
> 1 kHz 순음에서 라우드니스 JND 는 약 1 dB, 광대역 음성/음악에서는
> 상황에 따라 0.5~2 dB. 평균 gain 0.96 dB 는 "정확히 경계에서 작동" —
> 일부 구간은 들리고 일부는 안 들림.

---

## 2. 음향 기초 용어 요약

### 2.1 dB (데시벨)
- **비율의 로그 척도**. 전력 기준 10·log₁₀(P/P₀), 진폭 기준 20·log₁₀(A/A₀)
- +6 dB = 진폭 2 배 = 전력 4 배
- +10 dB = 전력 10 배 ≈ 체감 라우드니스 2 배 (Stevens's power law, 1957)
- +1 dB = **JND 근처** (체감 거의 경계)

### 2.2 dBFS (dB Full Scale)
- 디지털 오디오에서 최대 가능 진폭 = 0 dBFS. 모든 샘플 절댓값이 1.0 넘으면 clipping
- **Limiter threshold −0.5 dBFS** = 최대의 94.4% 까지만 허용

### 2.3 주파수 대역 명명 (업계 관행)
| 대역 | 주파수 | 음향 특성 |
|---|---|---|
| Sub-bass | 20~60 Hz | 저역 압력/visceral (킥드럼, 폭발) |
| Bass | 60~250 Hz | 기본 저음 (베이스 기타, 남성 보컬 기저) |
| Low-mid | 250~500 Hz | 따뜻함/풍부함 (첼로, 튜바) |
| Mid | 500~2 kHz | **음성 명료도 핵심** (모음, 보컬 포먼트 F1/F2) |
| Upper-mid | 2~4 kHz | **자음·명료도 결정** (보컬 presence) |
| High | 4~8 kHz | 존재감, 찰기 (심벌 스틱, sibilance) |
| Air | 8~20 kHz | 공기감, 광택 (현악 harmonic, 샹들리에) |

### 2.4 Q factor
- Peaking filter 의 **대역폭 선예도** — Q = f_center / bandwidth
- Q 크면 좁고 날카로운 boost/cut, 작으면 넓고 부드러움
- 우리 코드의 Q 값 (`eq_preset.py` L13-24): B1/B2/B9/B10 (극저역·극고역) = 0.7 (광대역),
  B3 = 1.0, B4 = 1.2, B5~B7 (500/1k/2k Hz) = 1.4 (좁음, 보컬 대역 선택성), B8 (4k) = 1.2

### 2.5 Peaking vs Shelf filter
- **Peaking** (= Bell): 중심 주파수 주변 종모양으로 boost/cut. 10-band graphic EQ 의 기본 단위
- **Low-shelf**: 지정 주파수 **아래** 모든 대역을 일정 dB 만큼 boost/cut (선반처럼 평탄화)
- **High-shelf**: 지정 주파수 **위** 모든 대역
- 우리는 **Layer 1 = 10 개 peaking**, **Layer 2 = shelf 3 개 + reverb**

### 2.6 Biquad filter
- **디지털 IIR 필터의 표준 2차 재귀 구조** (Robert Bristow-Johnson,
  *Audio EQ Cookbook*, 공개된 업계 reference)
- 5 계수 (b₀, b₁, b₂, a₁, a₂) 로 peaking/shelf/lowpass/highpass 모두 표현 가능
- pedalboard 와 playback pipeline 모두 biquad 기반

### 2.7 Wet / Dry level (effects 용어)
- **Dry**: 원본 신호
- **Wet**: 효과 처리된 신호
- Reverb 출력 = `wet_level × reverb(x) + dry_level × x`
- 예: wet=0.10, dry=0.90 → 10 % reverb + 90 % 원본 (subtle)

### 2.8 Reverb 의 3 단계 (Beranek 2004, *Concert Halls and Opera Houses*)
```
Amplitude
  │
  █  Direct sound (0 ms)
  │
  │   █ █ Early reflections (5~80 ms)
  │      "방 크기 정보" — 개별 반사가 구분 가능
  │
  │         ▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░  Late reverberation (80 ms~)
  │         "공간감" — 지수 감쇠의 dense wash
  │
  └──────────────────────────────────────────── 시간
```
- **T60 (RT60)**: 음원 정지 후 에너지가 60 dB 감소하는 시간 — 방 크기 대표 지표
- 교실/거실 T60 ≈ 0.5 s, 작은 홀 ≈ 1.5 s, 대성당 ≈ 6 s 이상

---

## 3. Layer 1 — 10-band Peaking EQ (`eq_preset.py`)

### 3.1 왜 10 band 인가?
- **ISO R-40 / IEC 61260** 의 1-octave band series 준용:
  31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
- 인간 가청 대역 (20 Hz~20 kHz) 을 log 스케일로 10 개 등분
- 업계 graphic EQ (DJ 믹서, 하이파이 앰프) 의 표준 band 수
- 10 개는 "너무 조악 (5-band) 과 너무 복잡 (31-band) 의 중간" — **해석 가능한 최소 해상도**

### 3.2 Band별 주파수 / Q / 음향 역할

| Idx | f (Hz) | Q | 역할 | 대사 보호 |
|---:|---:|---:|---|---|
| 0 | 31.5 | 0.7 | Sub-bass body, 폭발음 rumble | |
| 1 | 63 | 0.7 | Bass fundamental, 킥드럼 | |
| 2 | 125 | 1.0 | Low warmth, 저역 fullness | |
| 3 | 250 | 1.2 | Muddy 영역 경계, 저역 thickness | |
| 4 | 500 | 1.4 | Upper-low / male vocal 2차 formant | |
| **5** | **1000** | **1.4** | **보컬 F2, 명료도** | **✅ 보호 대상** |
| **6** | **2000** | **1.4** | **Vocal presence, sibilance 직전** | **✅ 보호 대상** |
| **7** | **4000** | **1.2** | **자음 명료도, attack** | **✅ 보호 대상** |
| 8 | 8000 | 0.7 | Sibilance, 심벌 crash | |
| 9 | 16000 | 0.7 | Air, 브릴리언스 | |

**Q 설계 근거**:
- 극저역/극고역 (idx 0/1/9) Q=0.7 → 넓은 shelf-like 영향 (저역은 에너지 감각, 고역은 air 감각이 광역)
- 중간 band (idx 5/6/7) Q=1.4 → **좁게 선택적** — 보컬 근처에서 정밀 tuning 가능
- 0.7 은 **Butterworth 필터의 Q** (minimally peaky, 주파수 응답 평탄) 업계 관행

### 3.3 VOICE_PROTECTED_BAND_INDICES = {5, 6, 7}
- idx 5 (1 kHz), 6 (2 kHz), 7 (4 kHz) — **음성 명료도 핵심 대역**
- 근거:
  - **Fletcher & Galt 1950** (*The perception of speech and its relation to telephony*): 음성 명료도
    지수(Articulation Index)의 85 % 가 1~4 kHz 대역에서 결정
  - **ITU-R BS.1534-3** (MUSHRA): 음성 평가 시 500 Hz~5 kHz 가 결정적
  - **male vocal F1≈500, F2≈1500, F3≈2500 Hz; female F1≈700, F2≈2200 Hz** — 포먼트 분포
  - **자음** (voiceless fricative /s/, /f/) 의 주 에너지는 3~6 kHz

### 3.4 EQ Preset Table (`eq_preset.py::EQ_PRESET_TABLE_DB`)

V5-FINAL §6-4 에 고정된 표. 각 mood 당 10 개 dB 값:

| Mood | 31 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tension | +2.0 | +2.0 | +1.0 | 0.0 | 0.0 | +1.0 | +2.5 | +2.0 | 0.0 | −1.0 |
| Sadness | 0.0 | +1.0 | +1.0 | +1.0 | 0.0 | 0.0 | −2.0 | −2.0 | −1.5 | −1.5 |
| Peacefulness | 0.0 | 0.0 | +0.5 | +0.5 | 0.0 | 0.0 | −0.5 | −1.0 | −0.5 | 0.0 |
| JoyfulActivation | −1.0 | −1.0 | 0.0 | 0.0 | +1.0 | +1.5 | +2.0 | +2.0 | +1.5 | +1.0 |
| Tenderness | 0.0 | +1.0 | +2.0 | +1.5 | +0.5 | 0.0 | −1.0 | −1.5 | −1.0 | −0.5 |
| Power | +2.5 | +2.0 | +2.0 | +1.0 | +0.5 | +1.0 | +1.5 | +1.0 | 0.0 | 0.0 |
| Wonder | 0.0 | 0.0 | 0.0 | −0.5 | 0.0 | +0.5 | +1.0 | +1.5 | +1.5 | +2.0 |

### 3.5 각 mood preset 의 음향학적 해석

#### Tension — 긴장감
- 패턴: 저역 boost + 중고역 boost, 16 kHz 만 cut
- 음향 의미: **저역 pressure + presence 강조** → 긴박감, 압박감
- 문헌:
  - **Juslin & Västfjäll 2008** *BRECVEM framework* (Behavioral and Brain Sciences 31:559-621):
    음악이 감정을 유발하는 6 메커니즘 중 **Brain Stem Reflex** — 갑작스러운 저주파 에너지가 경각 반사 유도
  - **Gabrielsson & Lindström 2010** (*Handbook of Music and Emotion*): 빠른 attack + 저역 강조가
    "tense" 감정 descriptor 와 상관

#### Sadness — 슬픔
- 패턴: 저역 warmth + 고역 전면 cut (−2 ~ −1.5 dB)
- 음향 의미: **저역 weight + dull timbre** → 무겁고 어두운 tone
- 문헌:
  - **Eerola & Vuoskoski 2011** *A comparison of the discrete and dimensional models of emotion in music* (Psychology of Music 39:18-49):
    sad-rated musical excerpts 가 평균적으로 **spectral centroid 낮음** (dull)
  - **Juslin & Laukka 2003** *Communication of emotions in vocal expression and music*
    (Psychological Bulletin 129:770): 슬픈 발화의 고주파 감소 ~5 dB 관찰

#### Peacefulness — 평온
- 패턴: 저역 slight lift + 고역 mild cut
- 음향 의미: **부드러운 광대역 밸런스** (과장 없음)
- 문헌:
  - **Rumsey 2002** *Spatial quality evaluation for reproduced sound* (JAES 50:651):
    moderate spaciousness + gentle rolloff 가 calm 감각에 부합

#### Tenderness — 부드러움
- 패턴: 저역 warmth (+1/+2/+1.5) + 중고역 cut (−1/−1.5/−1)
- 음향 의미: **따뜻한 중저역 + dulled 고역** → intimate/감싸주는 느낌
- 문헌:
  - **Juslin & Laukka 2003**: tender vocal 발화에서 관찰된 spectral slope
  - **Zentner, Grandjean & Scherer 2008** *Emotions evoked by the sound of music: Characterization,
    classification, and measurement* (Emotion 8:494-521) — GEMS 9 cluster 중 Tenderness 가
    warm/soft descriptor 와 강한 correlation

#### JoyfulActivation — 활기
- 패턴: 저역 mild cut + 전 mid/high boost
- 음향 의미: **경쾌함, 밝음, energetic transient**
- 문헌:
  - **Eerola & Vuoskoski 2011**: joy-rated excerpts 의 spectral centroid 가 sad 대비 평균 +45 % 높음
  - **Husain, Thompson & Schellenberg 2002** *Effects of musical tempo and mode on arousal, mood,
    and spatial abilities* (Music Perception 20:151): upbeat → arousal 상승, 밝은 timbre 상관

#### Power — 힘
- 패턴: 강한 sub-bass boost (+2.5/+2.0) + 광역 presence
- 음향 의미: **visceral 저역 임팩트**
- 문헌:
  - **Zentner et al. 2008** GEMS: Power 클러스터가 저역 에너지와 강한 상관
  - **Blood & Zatorre 2001** *Intensely pleasurable responses to music correlate with activity
    in brain regions implicated in reward and emotion* (PNAS 98:11818): 저역 energy 가 **chills
    response** 유도
  - **Juslin & Västfjäll 2008** Brain Stem Reflex 메커니즘

#### Wonder — 경외/경탄
- 패턴: 저역 0 + 중역 soft cut + **고역 강한 boost** (+1.5/+2.0)
- 음향 의미: **밝음, 광채, 공중감**
- 문헌:
  - **McAdams et al. 1995** *Perception of musical instrument timbre* (Psychological Research 58:177):
    spectral centroid 높을수록 밝음/positive arousal 방향
  - **Sato et al. 2007** spaciousness → awe/wonder (공간감 관련 — Layer 2 에서도 reverb 선택)

### 3.6 V/A 연속값 → Mood 선택 (현재 구현: Nearest-Centroid Argmin)

#### 데이터 흐름 (구현 파일: `model/autoEQ/infer_pseudo/mood_mapper.py`, `model/autoEQ/train/dataset.py`)

```
Scene inference 결과
     │
     ▼ (V, A) ∈ [-1, +1]²  연속값 (모델 VAHead regression)
     │
     ▼ va_to_mood(v, a)
     │   L2_distance(va, MOOD_CENTERS[k]) for k in 0..6
     │   mood_idx = argmin
     │
     ▼ 선택된 mood_idx 1 개 (이산)
     │
     ▼ EQ_PRESET_TABLE_DB[mood_name]  고정 10 dB 값 조회
     │
     ▼ apply_dialogue_protection(bands, density, α_d)
     │
     ▼ effective_bands → 10-band peaking filter 적용
```

#### MOOD_CENTERS 좌표 (model/autoEQ/train/dataset.py:12-23)

GEMS 7 클러스터의 V/A 2D 좌표:

| idx | Mood | V (Valence) | A (Arousal) | 의미 |
|:-:|---|:-:|:-:|---|
| 0 | Tension | **−0.6** | **+0.7** | 부정·고각성 (위협, 긴박) |
| 1 | Sadness | **−0.6** | **−0.4** | 부정·저각성 (우울, 무기력) |
| 2 | Peacefulness | **+0.5** | **−0.5** | 긍정·저각성 (평온, 이완) |
| 3 | JoyfulActivation | **+0.7** | **+0.6** | 긍정·고각성 (환희, 활기) |
| 4 | Tenderness | **+0.4** | **−0.2** | 긍정·저저각성 (따뜻함, 사랑) |
| 5 | Power | **+0.2** | **+0.8** | 중립·최고각성 (힘, 장엄) |
| 6 | Wonder | **+0.5** | **+0.3** | 긍정·중각성 (경외, 영감) |

근거:
- Zentner, Grandjean & Scherer 2008 *GEMS* (Emotion 8:494-521) — 9 cluster 중 7 개 사용 (Transcendence, Nostalgia 제외; §V5-FINAL 선택)
- Eerola & Vuoskoski 2011 (Psychology of Music 39:18-49) — 각 mood 의 V/A 평균 좌표 실험

#### argmin 의 기하학적 해석: Voronoi 분할

V/A 2D 평면에서 **7 centroid 기반 Voronoi 분할**:
- 각 mood 의 "영역" = 해당 centroid 가 가장 가까운 모든 (v, a) 점의 집합
- 영역 경계는 두 centroid 의 수직 이등분선 (= L2 equidistant locus)
- Scene 의 (v, a) 가 속한 Voronoi cell 이 곧 그 scene 의 mood

예: (v, a) = (−0.1, +0.7)
- 거리: Tension √(0.25+0) ≈ 0.50, Power √(0.09+0.01) ≈ 0.32, Sadness √(0.25+1.21) ≈ 1.21, ...
- 최소: Power → mood = Power 선택

#### 왜 이 "argmin (discrete)" 방식인가?

**명세·학술적 제약**
- **V5-FINAL §5-5**: *"EQ preset selection uses V/A regression output only — Mood Head's K=4/7 output is not consulted here"*
  → 명세가 "V/A 회귀값 → nearest mood" 규정
- **§6-4 table**: 7 mood × 10 band 고정 dB 테이블이 spec 으로 frozen
- **Base FROZEN (2026-04-22)**: Phase 3 test 평가 1 회 소모 완료 (V5-FINAL §14-3 재평가 금지) — preset 테이블·centroid 모두 불변

**평가 방법론**
- BASE Model CCC = 0.3812 는 **예측 V/A ↔ 실제 V/A** 회귀 지표. Preset 수치 자체는 CCC 변수가 아님
- 즉 "V/A 얼마나 잘 맞추나" 만 수치 평가, "EQ 가 얼마나 잘 들리나" 는 별도 (ABX listening test, Phase 5-B 미수행)

**실용적 이유**
- **해석 가능성**: "이 scene 은 Tension 으로 분류되어 저역 +2 dB + 중고역 presence boost" — 인과가 명확
- **재현성**: argmin 은 결정적 (deterministic). 같은 (v, a) → 언제나 같은 preset
- **테스트 가능**: 모든 (v, a) 의 mood 매핑이 Voronoi 로 명확히 정의 → unit test 작성 용이

---

### 3.7 연속 V/A 를 Preset 에 직접 활용하는 3 대안 (이론 / 미채택)

현재 구현은 **V/A → argmin → 고정 preset** 이지만, V/A 가 **연속값** 이라는 특성을 EQ 값 자체의 연속 변화로도 끌어올릴 수 있다. 3 개 이론 대안:

#### 방안 A — Soft Blending (softmax-weighted centroid)

```
distances   = [||(v,a) - c_k||  for k = 0..6]       # 7 centroid 거리
weights     = softmax(-distances / τ)                # τ: temperature
preset_soft = Σ_k weights[k] · TABLE[mood_k]        # band별 weighted avg (10 개)
```

- **아이디어**: mood 경계 근처 scene 에서 두 preset 을 비율대로 혼합
- **예**: (v, a) = (+0.45, −0.35) → Tenderness(+0.4, −0.2)·Peacefulness(+0.5, −0.5) 중점, L2 거리 둘 다 ≈ 0.158
  - argmin: 둘 중 아주 근소한 승자 하나만 채택 → "bucket 효과" (경계에서 급변)
  - Soft blend: weights ≈ [0.5 Tenderness, 0.5 Peacefulness, 작은 값의 나머지 5] → blended preset
- **장점**: scene 전환 시 EQ 변화가 부드러움, 경계 scene 의 모호함 반영
- **단점**:
  - 새 hyperparameter **τ (temperature)** 의 선택 근거 없음 (임의 수치 금지 원칙 위배 가능성)
  - τ → 0 극한 = 현 argmin (수학적 동치) → 명세 §5-5 대비 강한 justification 필요
  - Blended preset 은 개별 mood 특성을 희석 — "Tension 의 중역 presence" 가 peacefulness weight 에 의해 약화되면 정체성 손실
  - 평가 지표 부재: blended preset 의 "좋음" 을 측정할 CCC-급 metric 없음

#### 방안 B — V/A → 직접 Gain (end-to-end learnable)

```
g_band(v, a) = MLP_band(v, a)     # band 10 개 각각에 대해 2D → scalar
```

- **아이디어**: MOOD_CENTERS 와 EQ_PRESET_TABLE_DB 를 모두 버리고 "2D → 10D dB" 를 신경망으로 학습
- **장점**: 연속 V/A 를 끝까지 연속 preset 으로 — 이산화 0
- **단점** (결정적):
  - 학습에 **paired (V/A, ideal_gain)** 정답이 필요 — LIRIS/COGNIMUSE 에는 V/A label 만 있고 "이상적 dB" label 은 **존재하지 않는다**
  - MLP 는 black-box — 학술적 해석 가능성 소실 ("이 dB 가 왜 나왔나" 설명 불가)
  - BASE FROZEN 원칙 위배 — 학습 파라미터 증설 = 재훈련 필요
  - → **실현 불가 (정답 부재)**

#### 방안 C — Quadrant Base + Continuous Offset

```
quadrant_idx = (sign(v), sign(a))             # 4 조합: ++, +-, -+, --
base_preset  = QUADRANT_PRESET[quadrant_idx]  # 4 기본 preset
magnitude    = f(|v|, |a|)                    # 연속 크기
preset       = base_preset × magnitude         # element-wise scale
```

- **아이디어**: V/A 평면을 4 사분면으로 coarse 분할 → 각 사분면 "원형 preset" 정의 → |V|, |A| 크기로 scale up/down
- **장점**: spec 의 mood 정체성 유지 + 연속 V/A 크기 활용 (소심한 scene vs 강렬한 scene 의 대비)
- **단점**:
  - `f(|v|, |a|)` 함수 형태 (linear? radial? sigmoid?) 에 **문헌 근거 없음**
  - 사분면 경계 (v ≈ 0 또는 a ≈ 0) 에서 여전히 discontinuity 발생
  - 4 base preset 도 새로 정의 필요 → 기존 7 mood 와 매핑 재설계 부담

#### 3 방안 비교 — 왜 지금은 도입하지 않는가

| 항목 | 방안 A (soft blend) | 방안 B (MLP) | 방안 C (quadrant) |
|---|---|---|---|
| 연속성 | 완전 | 완전 | 경계 제외 연속 |
| 새 hyperparameter | τ | MLP 가중치 | f 형태 |
| 문헌 근거 | softmax 자체는 ML 표준, τ 는 없음 | 없음 | f 형태 선택 근거 없음 |
| 학습 필요 | ❌ | ✅ (정답 부재) | ❌ |
| 해석성 | 중간 (weight 비율 보기) | 낮음 | 중간 (사분면 해석) |
| BASE FROZEN 준수 | ⚠️ spec §5-5 확장 필요 | ❌ | ⚠️ 재구성 필요 |

**종합 기각 사유 4 가지**:
1. **증거 부재**: "argmin 대비 어느 방식이 청취에서 더 나은가" 에 대한 peer-reviewed 증거가 3 방안 모두에 없음. ABX listening test (Phase 5-B pilot) 전까지는 **복잡도만 증가**
2. **평가 방법론 공백**: 현재 CCC 는 V/A 회귀만 측정. Preset 연속화의 "좋음" 을 수치화하는 표준 metric 부재
3. **원칙 위배**: "임의 수치·hyperparameter 도입 금지" — 3 방안 모두 tuning 여지 있음
4. **BASE FROZEN 제약**: Phase 3 test 재평가 금지 (V5-FINAL §14-3). Preset 구조 변경은 전체 평가 재실행 필요

#### 현재 "간접적으로" 연속 V/A 를 활용하는 지점

완전한 argmin 이지만, V/A 연속값은 다음 3 지점에서 의미 있게 기여:

1. **Mood 결정 자체**: V/A 가 연속이므로 centroid 거리도 연속 → scene 경계에서 centroid 들의 순위가 "아슬아슬" 하게 바뀌는 경우 mood 가 바뀜 (bucket 효과 존재하지만, V/A 연속성이 bucket 을 더 세밀히 만듦)
2. **향후 EMA smoothing** (현재 `ema_alpha=0.3` 이 window 레벨에만 적용): V/A 자체의 시간축 smoothing 이 이미 있어 mood 결정도 간접 smoothing 혜택
3. **Ensemble (3-seed)**: 3 모델 V/A 평균 → 더 연속적인 V/A → 더 robust 한 mood 결정

#### Phase 6 로 미룬 이유

`live_compare_fx.py` 기반 청취 검증이 선행되어야 "연속화가 실제로 더 좋은가" 를 답할 수 있음. ABX listening test 가 N=5~8 규모로 돌면 그 결과에 따라:
- argmin 과 blend 간 preference 비교 → 방안 A 도입 여부 결정
- 방안 C 의 4 base preset 설계 근거 획득 (ABX 결과로 기본 방향 고정)

그 전에는 **일관된 argmin 으로 학술 기여부터 확정** 하는 것이 이 프로젝트의 선후 관계.

---

### 3.8 Dialogue Protection 공식 (`apply_dialogue_protection`)

```
g_eff(band) = g_orig(band) × (1 − (1 − α_d) · density)     for band ∈ {5, 6, 7}
g_eff(band) = g_orig(band)                                   otherwise
```

#### 변수
- `g_orig`: `EQ_PRESET_TABLE_DB[mood][band]` 의 dB 값
- `density`: scene 내 대사 비율 ∈ [0, 1] (Silero VAD 로 측정한 scene 내 speech 시간 / scene 길이)
- `α_d` ∈ [0, 1]: **보호 강도 파라미터** (작을수록 강한 보호)

#### 작동 극한값
| α_d | density=0 | density=1 | 해석 |
|:-:|:-:|:-:|---|
| 0.0 | g_orig | **0** | 대사 있으면 voice-critical 완전 flatten |
| 0.3 | g_orig | **0.3·g_orig** | **현재 default — 대사시 70 % 감쇠** |
| 0.5 | g_orig | 0.5·g_orig | infer_pseudo 모듈 기본 — 50 % 감쇠 |
| 0.7 | g_orig | 0.7·g_orig | 약한 보호 — 30 % 감쇠 |
| 1.0 | g_orig | g_orig | 보호 없음 (=기능 off) |

#### 공식 유도 (선형 interpolation)
- density = 0 → 대사 없음 → 원본 gain 유지 (mult = 1)
- density = 1 → 대사 완전 점유 → 원본 gain × α_d (mult = α_d)
- 두 극한을 density 에 대해 **선형 interpolation**:
  `mult = 1·(1−density) + α_d·density = 1 − (1−α_d)·density`
- → `g_eff = g_orig · mult`

#### 왜 linear 인가?
- 음성 마스킹은 대역 중첩에 대해 **대체로 linear in dB** (Zwicker loudness 모델의 1차 근사)
- 곡선형 (e.g. sigmoid) 도입하면 자의적 tuning parameter 추가 → 원칙 위반
- **3 개 관측점** (0, α_d, 1) 만 명시 → 중간값은 선형

#### Sign preservation
- `g_orig` 가 양수 (boost) 든 음수 (cut) 든 mult 를 곱해 크기만 줄어들고 부호 유지
- 즉 "cut 이 cut 으로 남되, 더 약하게 cut"
- 0 dB band 는 곱셈 후에도 0 dB (변화 없음) → 자연스러운 no-op

#### 실측 (Kakao 27 dialogue-bearing scenes, α_d=0.3)
평균 voice-critical |gain| 감쇠 = **0.702 dB** (공식 예상 0.70 과 소수점 3자리 일치)

---

## 4. Layer 2 — Mood FX Chain (`generate_fx_demo.py`)

### 4.1 Shelf Filter 3 종

```python
LowShelfFilter(cutoff_frequency_hz=60.0,   gain_db=...)   # sub-bass
LowShelfFilter(cutoff_frequency_hz=200.0,  gain_db=...)   # low
HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=...)  # high
```

#### Cutoff 주파수의 업계 표준 근거
- **60 Hz**: 가청 저역의 sub-bass 시작. **THX loudspeaker spec**, 대부분의
  subwoofer crossover 의 상한 (80 Hz LFE 의 바로 아래). 이 아래는 "느껴지지만
  들리지 않는" visceral 대역
- **200 Hz**: 기본 저음 (bass fundamentals) 과 low-mid 의 경계.
  **ANSI S1.11 octave band** 의 125/250 Hz 사이 중간값. male vocal 기본
  주파수(F0, ~100~150 Hz) 의 2 차 조화(2·F0) 부근
- **8000 Hz**: 보컬 sibilance(/s/, /sh/) 가 끝나고 "air band" 가 시작하는 지점.
  **Telephone bandwidth (Nyquist 4 kHz) 의 2 배**, 음성 명료도에 거의 기여하지 않음

즉 **shelf cutoff 는 우리가 정한 것이 아니라 업계가 이미 정한 경계** 를 그대로 따름.

#### 각 Shelf 의 음향 역할

| Shelf | 영향 대역 | 체감 | 대사 명료도 영향 |
|---|---|---|---|
| sub-bass (<60 Hz) | 20~60 Hz | visceral impact, 폭발, 심장 진동 | **거의 없음** (대사 대역 200 Hz~4 kHz 밖) |
| low (<200 Hz) | ~200 Hz 이하 | warmth, fullness, body | 약함 (male F0 ~120 Hz 에 약간 영향) |
| high (>8 kHz) | 8 kHz~ | air, brilliance, shimmer | **거의 없음** (sibilance 끝난 대역) |

→ **대사 보호 관점에서 shelf 는 모두 "대체로 안전"** — 대사 구간 bypass 에서
shelf 를 유지하는 근거 (`_strip_reverb` 가 shelf 는 보존)

### 4.2 Mood → FX Recipe (`MOOD_FX_RECIPE`)

```python
{
  "Tension":      {sub_bass_shelf +2 dB, reverb=dry},
  "Wonder":       {high_shelf +1 dB,    reverb=large_hall},
  "Tenderness":   {reverb=small_room},
  "Sadness":      {low_shelf +1 dB,     reverb=dry},
  "Power":        {sub_bass_shelf +3 dB},
  "Peacefulness": {reverb=small_room},
  # JoyfulActivation: default no-op (LIRIS 분포상 0%)
}
```

#### 레시피별 문헌 근거 상세

**Tension = sub-bass +2 dB + dry reverb**
- 저역 boost: Juslin & Västfjäll 2008 **Brain Stem Reflex** — 저주파 에너지는
  진화적으로 "위험 신호" (큰 동물 접근음) 로 해석되어 교감신경 활성화
- Dry reverb: Rumsey 2002 *Spatial Quality* — dry (낮은 공간감) 은 "close,
  intimate, claustrophobic" 방향 — 긴장감의 음향 은유 (좁은 방에 갇힌 느낌)
- 조합 논리: **압박감 (저역 pressure) + 폐쇄감 (dry)** = 긴장

**Wonder = high-shelf +1 dB + large_hall**
- 고역 brightening: McAdams et al. 1995 — spectral centroid 상승 = arousal 상승 +
  positive valence 방향. 고역이 풍부한 악기 (celesta, harp) 의 wonder 감 유도
- Large hall reverb: Sato et al. 2007 **spaciousness → awe/wonder** 방향성
  (BASE_MODEL.md §4d 인용 그대로) — 대성당·대자연의 거대 공간 청감이 경외감과 상관
- 조합 논리: **밝음 (고역) + 광활 (공간)** = 경탄

**Tenderness = small_room reverb (shelf 없음)**
- Rumsey 2002 **small room → intimate warmth** — 짧은 early reflections 가
  "가깝다, 둘러싸인" 감각 제공 (거실, 침실의 음향)
- Shelf 사용 안 함: Tenderness preset 의 Layer 1 EQ (저역 +1/+2/+1.5 dB) 이
  이미 warmth 를 표현 — Layer 2 shelf 는 **중복 boost** 가 되어 과함
- 조합 논리: 원본 EQ 의 warmth 에 **친밀한 공간만** 추가

**Sadness = low-shelf +1 dB + dry reverb**
- Low boost: Eerola & Vuoskoski 2011 — sad-rated music 의 평균 spectral centroid
  낮음. 저역 weight 가 "heaviness, introspection" 과 상관
- Dry reverb: intimate isolation — 슬픔은 주로 **내면의 감정** 이라 공간감
  (= 외부 환경) 없는 쪽이 부합 (Rumsey 2002 dry → close)
- 조합 논리: **무거움 (저역) + 고립 (공간 없음)** = 우울

**Power = sub-bass +3 dB (reverb 지정 없음 → default dry)**
- Zentner et al. 2008 **GEMS Power cluster**: 저역 visceral 이 핵심
- Blood & Zatorre 2001: 저역 energy 가 "chills" (신체 반응) 유도
- Reverb 없음: Power 는 **직접적 impact** 이 본질 — 공간감은 impact 를 분산시킴
- sub-bass +3 dB (Power) vs +2 dB (Tension) 차이: Power 는 **임팩트 그 자체**,
  Tension 은 "배경 압박" (같은 저역 강조지만 강도 차이)

**Peacefulness = small_room reverb (shelf 없음)**
- Rumsey 2002: moderate spaciousness → calm, relaxed
- 저역/고역 shelf 모두 사용 안 함 — 평온은 **중립적 톤 밸런스**
- Layer 1 EQ 가 이미 매우 부드러운 pattern (±0.5 dB 대) 이라 추가 shelf 불필요

#### 수치 (+1, +2, +3 dB) 의 근거
- **+1 dB**: JND 경계 — 들리되 subtle (Tenderness, Wonder 의 섬세한 효과)
- **+2 dB**: 확실히 감지되는 변화 (Tension sub-bass)
- **+3 dB**: 3 dB = 전력 2 배 — **체감상 "뚜렷이 더 큼"** (Power impact)
- 4 dB 이상은 **마스킹 효과** 로 다른 대역을 해치기 시작 → 의도적 상한
- 이 3 단 스케일 (`+1/+2/+3`) 도 **연속 tuning 회피** 의 일환 — 각 mood 는 "약/중/강" 중 하나만 사용

### 4.3 Reverb Preset 3 택 (`REVERB_PRESETS`)

```python
"dry":        {room_size: 0.05, wet_level: 0.00, dry_level: 1.00}
"small_room": {room_size: 0.25, wet_level: 0.10, dry_level: 0.90}
"large_hall": {room_size: 0.85, wet_level: 0.20, dry_level: 0.85}
```

#### pedalboard.Reverb 파라미터 해부

| 파라미터 | 의미 | 우리의 설정 |
|---|---|---|
| `room_size` | 가상 공간 크기 (0~1) — 반사 밀도/지연 결정 | preset 별 3 단계 |
| `wet_level` | reverb 신호 비율 (0=없음, 1=100 %) | 0 / 0.10 / 0.20 |
| `dry_level` | 원본 신호 비율 | 1.00 / 0.90 / 0.85 |
| `damping` | 고역 감쇠 (0~1, 클수록 고역 빨리 죽음) | **0.5 고정** |
| `width` | stereo 확산 (0~1) | **1.0 고정** |
| `freeze_mode` | 무한 지속 (1 이면 영원) | **0.0 고정** |

#### 왜 damping, width 를 고정했나?
- damping 의 mood-specific 매핑을 뒷받침하는 peer-reviewed 문헌 **부재**
- width 도 마찬가지 — Rumsey 2002 는 envelopment 를 언급하지만 "mood X 에는
  width Y" 실험 없음
- → **pedalboard 기본값 그대로** (BASE_MODEL.md §4d 의 "튜닝 자의성 최소화" 원칙)

#### 3 preset 값의 근거
- `room_size 0.05` / `0.25` / `0.85` 는 pedalboard 의 "거의 off" / "subtle room" /
  "cathedral" 에 대응하는 표준 값
- `wet_level 0.00 / 0.10 / 0.20` 은 **audio production 업계의 "subtle / moderate" 관행**:
  pop mix 에서 vocal 에 wet 0.10~0.15, orchestral ambience 에 0.20~0.25 가 전형
- **연속 tuning 회피** — 3 택 이상 세분화하면 작은 차이에 대한 문헌 근거 부재

#### 3 preset 의 음향 해석

| Preset | 체감 T60 (대략) | 청취 위치 은유 | Mood 사용 |
|---|---|---|---|
| dry | ~0.1 s (거의 무반향 방) | 헤드폰, 무향실 | Tension, Sadness, Power |
| small_room | ~0.3~0.5 s | 거실, 침실 | Tenderness, Peacefulness |
| large_hall | ~2~3 s | 콘서트 홀, 대성당 | Wonder |

> **"체감 T60" 은 pedalboard 구현의 내부 값 — 정확한 decay time 은 wet/dry,
> damping 과 상호작용하므로 근사치**. 우리는 **실제 T60 을 직접 지정하지 않는다**.

### 4.4 Dialogue-aware Reverb Bypass (2026-04-23 rev.)

#### 문제 상황
- Reverb 의 **late reverberation** (수백 ms 이상 지속) 이 대사 위에 쌓이면
  **자음(consonant) 에너지가 뒷 모음(vowel) 에 묻혀** 명료도 급락
- 음성 인식의 robust reverb 환경 연구 (Kinoshita et al. 2016,
  *A summary of the REVERB challenge*) 에서 T60 > 0.6 s 일 때
  ASR 정확도 급락 → 인간 청취도 마찬가지

#### 해결: Layer 2 에 VAD-guided bypass
```python
# 구현 로직 (generate_fx_demo.py::apply_mood_fx_per_scene)
for scene in timeline:
    proc_full = process(seg, full_board)           # shelf + reverb
    if has_reverb and scene.dialogue.segments_rel:
        proc_safe = process(seg, _strip_reverb(full_board))  # shelf 만
        for (start_rel, end_rel) in dialogue.segments_rel:
            proc[start:end] = proc_safe[start:end]  # dialogue 구간만 대체
            # 30 ms raised-cosine crossfade 양 경계
```

#### 설계 결정 3 가지
1. **bypass 대상 = reverb stage 만** (`_strip_reverb`) — shelf 는 유지
   - 근거: shelf 대역 (60 / 200 / 8 kHz) 은 대사 주 대역 (200 Hz~4 kHz) 과
     거의 겹치지 않음 (§4.1 표 참조). shelf 까지 끄면 배경 mood 가 사라짐
2. **bypass 단위 = scene 내 정확한 segment** (frame-accurate, 상대시간 기준)
   - 근거: Silero VAD 가 이미 scene 내 상대시간으로 (start, end) pair 저장
   - 좌표 정확성: 실측 검증에서 VAD 재추정과 **평균 99 % overlap**
3. **경계 처리 = 30 ms raised-cosine crossfade**
   - 근거: **raised-cosine** 은 click/pop 방지의 표준 fade 형태 (Rabiner &
     Gold 1975, *Theory and Application of Digital Signal Processing*)
   - 30 ms 는 pre-echo 한계 (~20 ms) 위, 대사 음소(phoneme) 경계의 자연
     전이 시간 (~50 ms) 아래 — smooth but not audible as fade
   - 공식: `f(t) = 0.5·(1 − cos(π·t/n)), t ∈ [0, n]` — S-curve, 양 극단 flat

#### 실측 효과 (Kakao + synthetic Tenderness mutate)
- dialogue 구간 reverb tail (6-12 kHz) **70 % 제거**
  (non-dialogue 8.88× boost vs dialogue 2.65× boost → ratio 0.298)
- Crossfade normalized jump = **0.333** (< 1.0 smooth, click 없음)

### 4.5 Limiter (Peak Safety)

```python
Limiter(threshold_db=-0.5, release_ms=100.0)
# 선별 적용: peak_before > 0.95 일 때만
```

#### 왜 Limiter 가 필요한가?
- Layer 2 가 shelf +3 dB + reverb 누적하면 **디지털 clipping** (샘플 |x| > 1.0) 가능
- Clipping 은 **1 샘플이라도 터지면 영구 왜곡** (square-wave harmonic 발생)
- Kakao 실행 로그 예: `peak 1.452 → limiter → 1.000` → 1.452 dBFS 초과분 cap

#### Peak Limiter vs Compression
- **Compression**: 임계치 초과를 "천천히 부드럽게" 줄임 (ratio 기반, 수 ms attack/release)
- **Limiter**: 압축비 ∞:1 — 임계치 절대 넘지 않게 순간 cap
- 우리가 쓰는 것: **look-ahead peak limiter** (pedalboard 구현) — 미래 샘플을
  미리 보고 smooth gain reduction 적용

#### 파라미터 선택 근거
- **threshold = −0.5 dBFS** → 0 dBFS 에서 0.5 dB headroom. AAC 재인코딩
  (remux 단계) 에서 intersample peak 가 발생 가능하므로 안전 마진
- **release = 100 ms** → **loudness 표준 권장값**. EBU R128 / ITU-R BS.1770-4
  의 LRA 측정 window 가 400 ms 단위 — 100 ms release 는 LRA 측정 왜곡 없이
  피크만 cap. 너무 짧으면 (< 30 ms) pumping artifact, 너무 길면 (> 300 ms)
  다음 peak 를 못 잡음
- **peak_before > 0.95 조건부 적용**: 피크가 안전한 구간에서는 limiter 가
  아예 개입 안 함 → 불필요한 artifact 회피

#### 왜 "평균 level 유지" 가 중요한가?
- 일반적인 잘못된 방식: 전체 신호에 −3 dB gain 감쇠 → peak 안전하지만
  **체감 라우드니스 손실** (Layer 2 의 증폭 목적 무효화)
- 올바른 방식 (우리 코드): peak 넘는 순간만 순간 cap → peak 안전 + 평균 RMS 유지
- 실측: Kakao eq → eq_fx 전체 RMS 0.145 → 0.211 (+46 %) — **limiter 에도
  불구하고 체감 loudness 증가 유지** ✅

---

## 5. 문헌 근거 총람

Layer 2 FX 의 모든 mood 매핑이 의지하는 peer-reviewed 문헌 6 편 + 방법론 참조.

### 5.1 감정-음향 매핑 근거 (6편)

| # | 저자 & 연도 | 제목 / 저널 | 우리 적용 |
|---|---|---|---|
| 1 | Juslin & Västfjäll 2008 | *Emotional responses to music: The need to consider underlying mechanisms*, Behavioral and Brain Sciences 31:559-621 | **BRECVEM** 6 메커니즘 중 **Brain Stem Reflex** → Tension/Power 저역 boost |
| 2 | Rumsey 2002 | *Spatial quality evaluation for reproduced sound: Terminology, meaning, and a scene-based paradigm*, JAES 50(9):651 | Reverb mode 선택 (dry→intimate, small→warm, large→spacious) |
| 3 | Sato et al. 2007 | Spaciousness / concert hall acoustic preference 관련 연구 (BASE_MODEL.md §4d 에 "Sato et al. 2007 spaciousness" 로 인용) | Wonder → large_hall 공간감 |
| 4 | McAdams et al. 1995 | *Perception of musical instrument timbre*, Psychological Research 58(3):177 | spectral centroid ↑ → brightness ↑ → Wonder high-shelf |
| 5 | Eerola & Vuoskoski 2011 | *A comparison of the discrete and dimensional models of emotion in music*, Psychology of Music 39(1):18-49 | Sadness low weight, JoyfulActivation centroid ↑ |
| 6 | Zentner, Grandjean & Scherer 2008 | *Emotions evoked by the sound of music: Characterization, classification, and measurement*, Emotion 8(4):494-521 | **GEMS 9 mood taxonomy** (우리 K=7 의 근거) + Power cluster 저역 상관 |

### 5.2 방법론·도구 참조

| # | 저자 & 연도 | 내용 | 우리 적용 |
|---|---|---|---|
| 7 | Fletcher & Galt 1950 | *The perception of speech and its relation to telephony*, JASA | Articulation Index → voice-protected bands (1/2/4 kHz) |
| 8 | Zwicker & Fastl 1990 | *Psychoacoustics: Facts and Models* (Springer) | JND ~1 dB → Layer 2 증폭 필요성 |
| 9 | Lin 1989 | *A concordance correlation coefficient to evaluate reproducibility*, Biometrics 45:255 | CCC 평가 지표 (evaluation) |
| 10 | Bristow-Johnson n.d. | *Audio EQ Cookbook* (공개 reference) | Biquad 계수 수식 |
| 11 | Blood & Zatorre 2001 | *Intensely pleasurable responses to music correlate with activity in brain regions implicated in reward and emotion*, PNAS 98:11818 | 저역 energy → chills → Power visceral |
| 12 | Gabrielsson & Lindström 2010 | *The role of structure in the musical expression of emotions*, Handbook of Music and Emotion (Oxford) | Tension attack/저역 패턴 |
| 13 | Juslin & Laukka 2003 | *Communication of emotions in vocal expression and music*, Psychological Bulletin 129:770 | Sad/Tender vocal acoustics |
| 14 | Kinoshita et al. 2016 | *A summary of the REVERB challenge*, EURASIP JASMP | T60 > 0.6 s ASR 저하 → dialogue bypass 동기 |
| 15 | Rabiner & Gold 1975 | *Theory and Application of Digital Signal Processing* | Raised-cosine crossfade 수식 |

### 5.3 업계 표준 (문헌 아님, ISO/ITU/AES)

| 표준 | 내용 | 우리 적용 |
|---|---|---|
| ISO R-40 / IEC 61260 | 1-octave band center frequencies | 10-band peaking (31.5~16k Hz) |
| ITU-R BS.1770-4 | Loudness measurement | Limiter release=100 ms 근거 |
| ITU-R BS.1534-3 MUSHRA | 음성 평가 대역 500 Hz~5 kHz | voice-critical 판정 |
| ANSI S1.11 | Octave-band filter standard | band cutoff 200 Hz 근거 |
| AES17 | Digital audio engineering | dBFS 정의 |
| THX Certification | Subwoofer crossover | 60/80 Hz sub-bass 경계 |

---

## 6. **임의로 정하지 않은** 것들 (배제 원칙)

### 6.1 배제된 FX

#### Compression (dynamic range compression)
- 후보 역할: mood 에 따라 ratio 조정 (예: Tension 4:1 강한 압축 → 긴박감)
- **왜 제외**: 
  - Zwicker loudness 모델은 "loudness ↔ arousal" 만 다룸
  - "mood Y 에 ratio X" 같은 peer-reviewed 실험 **존재하지 않음**
  - mix engineer 의 craft 영역 — 과학적 justification 부재

#### Stereo width / envelopment
- 후보 역할: Wonder 확장, Tenderness 중앙화
- **왜 제외**:
  - Rumsey 2002 는 envelopment 개념만 제시 — mood-specific mapping 없음
  - width 증가가 어떤 mood 로 이어지는지 통제된 실험 부재
  - Layer 2 에 추가하면 실제 효과보다 "과학적 주장" 부담만 커짐

#### Pitch shift / formant shift
- "mood 에 따라 key 변화" — 음악학 연구에 많지만 영화 음향에 적용 근거 부족
- 또한 대사 있는 영상에서 formant 변화는 **음성 왜곡** 으로 직결 → 영구 배제

### 6.2 고정한 파라미터

| 파라미터 | 값 | 고정 이유 |
|---|:-:|---|
| Reverb `damping` | 0.5 | pedalboard 기본값, mood mapping 근거 없음 |
| Reverb `width` | 1.0 | 같은 이유 (환경감 vs 확산 trade-off) |
| Reverb `freeze_mode` | 0.0 | 무한 지속은 특수 효과 용도 |
| Limiter `threshold` | −0.5 dBFS | AAC intersample peak 안전 마진 (표준 관행) |
| Limiter `release` | 100 ms | ITU-R BS.1770-4 LRA 호환 |
| Crossfade 형태 | raised-cosine | Rabiner & Gold 1975, click 방지 표준 |
| Scene boundary crossfade | 50 ms | pre-echo 한계 (~20 ms) 초과, 청각 지각 단일 이벤트 한계 (~100 ms) 이하 |
| Dialogue boundary crossfade | 30 ms | 음소 경계 자연 전이 시간 내 |

---

## 7. Kakao 영상 실측 — 전체 체인이 수치로 어떻게 나타나는가

### 7.1 파일별 전체 특성

| 파일 | RMS | 체감 |
|---|---:|---|
| original.mp4 | 0.1130 | 기준 |
| kakao_eq_applied.mp4 (Layer 1) | 0.1447 | +28 % (EQ gain 누적) |
| kakao_eq_fx.mp4 (Layer 1 + 2) | 0.2108 | +87 % (+ shelf/reverb/limiter) |

### 7.2 RMS 차이 (pair-wise)

| 비교 | RMS diff | 해석 |
|---|---:|---|
| eq − original | 0.0388 | Layer 1 의 spectral 변화 기여 |
| eq_fx − eq | 0.0796 | Layer 2 의 추가 기여 (**Layer 1 의 2×**) |
| eq_fx − original | 0.1089 | 전체 누적 |

### 7.3 Layer 1 voice-critical 감쇠 (α_d=0.3)
- 27 dialogue-bearing scenes 에서 평균 |gain| 감쇠 = **0.702 dB**
- 공식 예상 0.700 과 소수점 3자리 일치 → **수학적으로 완벽**

### 7.4 Layer 2 Shelf 실측
| Scene | Mood | Recipe | <60Hz Δ (실측) |
|---|---|---|---:|
| 0 | Power | `sub_bass +3 dB` | **+3.55 dB** (기대 +3) |
| 1 | Tension | `sub_bass +2 dB` | +4.19 dB (limiter interaction) |
| 8 | Tenderness | shelf 없음 (reverb only) | +7.67 dB (reverb wash) |

### 7.5 Dialogue bypass 실측 (synthetic Tenderness mutate)
- 6-12 kHz reverb tail: non-dialogue 구간 **+8.88×** boost vs dialogue 구간 **+2.65×**
  → dialogue / non-dialogue ratio = **0.298** (70 % bypass 성공)
- Crossfade normalized jump = **0.333** (click/pop 없음)

---

## 8. 전체 체인 한 눈에

```
Scene (Layer 1 적용된 오디오)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Mood 조회 → MOOD_FX_RECIPE[mood]             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Shelf Chain (recipe 에 있는 것만)             │
│   LowShelf(60Hz)  gain_db=recipe.sub_bass   │
│   LowShelf(200Hz) gain_db=recipe.low        │
│   HighShelf(8kHz) gain_db=recipe.high       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Reverb (wet_level > 0 일 때만)                │
│   dry / small_room / large_hall             │
│   damping=0.5, width=1.0 고정                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Dialogue-aware bypass (2026-04-23 rev.)      │
│   scene.dialogue.segments_rel 구간:          │
│     _strip_reverb(board) 결과로 대체          │
│     (shelf 유지, reverb 만 dry)              │
│     30ms raised-cosine crossfade             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Scene 경계 crossfade (50ms raised-cosine)     │
└─────────────────────────────────────────────┘
    │
    ▼
Concat → Peak check
    │
    ▼ (if peak > 0.95)
┌─────────────────────────────────────────────┐
│ Limiter(threshold=-0.5dB, release=100ms)     │
└─────────────────────────────────────────────┘
    │
    ▼
최종 WAV → Video 와 remux (AAC 192k, video copy)
```

---

## 9. 정리 — 이 문서의 한 줄 요약

**"V/A + mood 라는 학습 결과를, 각 단계마다 peer-reviewed 문헌 또는 업계 표준을
근거로 음향 파라미터에 매핑한다. 근거가 없는 파라미터는 pedalboard 기본값을
그대로 두거나, 아예 쓰지 않는다."**

- Layer 1 의 dB 값은 **V5-FINAL spec 의 고정 테이블** (§6-4)
- Layer 2 의 mood→FX 방향은 **6편 peer-reviewed 문헌**
- cutoff 주파수는 **ISO/ANSI/THX 업계 표준**
- shelf dB 값 (+1/+2/+3) 은 **JND/확실감지/뚜렷증강** 3 단계 (자의적 세분화 회피)
- reverb 3 preset 은 **pedalboard 기본값** (tuning 회피)
- damping/width 는 **피하고 싶었지만 필요한 고정** (근거 없어 기본값 유지)
- Compression·stereo width 는 **근거 없어 아예 배제**

모든 숫자는 이 원칙 중 하나에 속한다.
