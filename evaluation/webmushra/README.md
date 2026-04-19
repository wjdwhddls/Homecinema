# Mood-EQ MUSHRA 평가 인프라

트랙 5+ 평가 설정. 두 평가 페이지 + 자동 저장 서버 + 분석 스크립트로 구성.

---

## 1. 서버 시작

```bash
cd evaluation/webmushra
python save_server.py
```

- 기본 포트 **8765** (변경: `--port 8000`)
- 바인딩 기본 `0.0.0.0` → LAN 상 다른 기기도 접근 가능
- 엔드포인트:
  - `GET /<any>` — 정적 파일 서빙 (HTML/JS/WAV 모두)
  - `POST /save_results` — 평가 JSON 자동 저장
  - `GET /health` — 서버 헬스 체크

Ctrl+C로 종료. Threaded TCP 서버라 여러 평가자 동시 제출 안전.

---

## 2. 평가 페이지 2종

### 2.1 Segment MUSHRA

```
http://localhost:8765/simple_player_v3.html
```

- **4 segment × 4 condition** (단일 점수 / 0~100)
- Segments (탑건 풀 트레일러 자동 추출):
  - Joyful (11~21s) / Power (41~51s) / Peacefulness (80~90s) / Sadness (99~109s)
- Conditions: Reference / Anchor / V3.3 / V3.5.5
- 블라인드 A~D 매 세션 셔플
- 즉시 전환 모드 (5ms crossfade) / 키보드 1~4

### 2.2 Full Trailer 비교

```
http://localhost:8765/full_trailer_comparison.html
```

- **1 트레일러(144s) × 5 condition × 3 axis** = 15 슬라이더
- Conditions: Reference / Anchor / V3.3 / V3.5.5 / V3.5.6
- Axes: 선호도(preference) / 명료도(clarity) / 자연스러움(naturalness)
- 5조건 동기 재생 (SyncEngine, Web Audio BufferSource + GainNode muting)
- 키보드 1~5 / Space = play/pause / 진행바 클릭 = 동시 seek
- 참가자 정보 form (이름 / 이어폰|헤드폰|스피커 / 초심|중간|숙련)

---

## 3. 결과 저장

### 자동 (서버 실행 중)
평가 완료 후 "📤 결과 제출" 버튼 → POST `/save_results` →
`evaluation/webmushra/results/{timestamp}_{page_type}_{evaluator}_{uuid}.json`

예: `2026-04-18T23-10-52_full_trailer_yun_a1b2c3.json`

### Fallback (서버 미실행)
자동으로 브라우저 다운로드 폴더에 JSON 저장 + "⚠ 서버 저장 실패" 토스트.

### Post-screening 자동 기록
BS.1534-3 기반 플래그:
- `anchor.preference > 90` → 저품질을 고품질로 오인 → suspicious
- `reference < 90` (15%+) → 레퍼런스 식별 실패 → suspicious

JSON 저장 시 `post_screening.validity` 필드 자동 기록. 분석 시 기본 제외.

---

## 4. 결과 분석

```bash
# 전체 page_type stdout
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tools/analyze_mushra_results.py

# Markdown 보고서 저장
venv/Scripts/python.exe tools/analyze_mushra_results.py \
    --output evaluation/webmushra/results/report.md

# Full Trailer만 (Segment 제외)
venv/Scripts/python.exe tools/analyze_mushra_results.py --page-type full_trailer

# Post-screening 플래그 세션 포함
venv/Scripts/python.exe tools/analyze_mushra_results.py --include-flagged
```

출력:
- 조건별 / 씬별 / 축별 집계표 (N, mean, std, min, max)
- Wilcoxon signed-rank (n ≥ 5):
  - V3.3 vs V3.5.5, V3.5.5 vs V3.5.6, V3.3 vs V3.5.6
  - (full_trailer) Reference vs V3.5.6, V3.5.6 vs Anchor (sanity)
- 유의성 표시 (p < 0.05)

의존성: `scipy`, `numpy` (venv 기존 설치).

---

## 5. 원격 평가 가이드 (참가자 모집)

### 옵션 A: 같은 LAN (동일 Wi-Fi)
1. 본인 IP 확인:
   - Windows: `ipconfig` → IPv4 주소 (예: 192.168.0.123)
   - macOS/Linux: `ifconfig` 또는 `ip addr`
2. 평가자에게 URL 전달:
   ```
   http://192.168.0.123:8765/full_trailer_comparison.html
   ```
3. 평가자 평가 → 본인 컴퓨터 `results/` 에 자동 저장
4. 방화벽이 외부 접속 차단 시 Windows Defender에서 Python에 공용 네트워크 허용

### 옵션 B: 원격 (서버 미운영)
1. 평가자에게 페이지 URL 또는 zip 전달 (평가 페이지 + 매칭 wav 파일)
2. 평가자 평가 → 파일 다운로드됨 (fallback 모드)
3. 평가자가 JSON 파일 이메일/카톡으로 전송
4. 본인이 `evaluation/webmushra/results/` 에 직접 배치

### 옵션 C: 클라우드 (예: ngrok 터널링)
```bash
# 별도 터미널
ngrok http 8765
# https://abcd.ngrok.io/full_trailer_comparison.html → 평가자에게 전달
```
주의: ngrok 무료 티어는 연결 수/대역폭 제한. 공개 URL이므로 민감 정보 주의.

---

## 6. 파일 / 디렉토리 요약

| 경로 | 내용 |
|---|---|
| `simple_player_v3.html` | Segment MUSHRA 평가 페이지 (4 segment × 4 cond) |
| `full_trailer_comparison.html` | Full Trailer 비교 페이지 (1 × 5 cond × 3 axis) |
| `save_server.py` | 정적 서빙 + POST /save_results |
| `configs/resources/audio/trailer_segments/topgun/` | 4 segment × 4 cond matched wav (16 파일) |
| `configs/resources/audio/full_trailer/topgun/` | 5 풀 트레일러 matched wav |
| `results/*.json` | 저장된 평가 세션 (gitignore) |
| `../../tools/analyze_mushra_results.py` | 통계 분석 스크립트 |
| `../../tools/loudness_match.py` | LUFS 매칭 (평가용 wav 준비 시) |
| `../../tools/generate_anchor.py` | MUSHRA 표준 low anchor (3.5 kHz LPF) |
| `../../tools/auto_select_segments.py` | diversity 기반 segment 자동 선정 |

---

## 7. webMUSHRA 상류 (참조)

본 디렉토리의 `lib/`, `configs/` 일부는 [audiolabs/webMUSHRA](https://github.com/audiolabs/webMUSHRA) 프로젝트 기반.
본 프로젝트에서는 주로 `simple_player_v3.html` / `full_trailer_comparison.html` 의 독자 평가
페이지를 사용. 웹MUSHRA 기본 YAML 기반 평가는 `configs/mood_eq_curated.yaml` 참조.

webMUSHRA 상류 문서/소스: https://github.com/audiolabs/webMUSHRA

---

## 8. 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| POST /save_results 실패 (CORS error) | 서버가 실행 중인지 확인, `--bind 0.0.0.0` 기본 설정인지 확인 |
| 5 wav 로딩 느림 (full_trailer_comparison) | 초기 fetch 135MB — 10초 내외. 오버레이 표시됨 |
| 결과 저장 파일 평가자 이름이 "anon" | simple_player_v3의 `evaluatorId` 입력 또는 full_trailer_comparison의 `participant.name` 비어있음 |
| 서버 시작 시 `Address already in use` | 기존 서버 프로세스가 포트 보유. `--port 8766` 등 다른 포트로 실행 |
| 분석 스크립트 "scipy 미설치" | `venv/Scripts/python.exe -m pip install scipy numpy` |

---

작성: 트랙 5+ Phase Y (MUSHRA 저장 인프라).
관련 커밋: `save_server.py`(Y2), client integration(Y3), `analyze_mushra_results.py`(Y5), 본 문서(Y6).
