# Phase 1 (공간 환경 스캐닝) - 빌드 및 테스트 가이드

## 자동 완료된 작업

다음은 코드로 자동 처리되었습니다:
- [x] iOS Deployment Target 16.0으로 변경 (Podfile)
- [x] Info.plist에 카메라 권한 추가
- [x] Swift 네이티브 모듈 파일 생성 (RoomScanner.swift, RoomScanViewController.swift)
- [x] Objective-C 브릿지 파일 생성 (RoomScanner.m)
- [x] Bridging Header 설정
- [x] TypeScript 인터페이스 작성 (src/native/RoomScanner.ts)
- [x] SpeakerPlacementScreen 실제 스캔 화면으로 교체
- [x] pod install 완료

## 사용자가 수동으로 해야 할 작업 (Xcode)

### 1. Xcode 워크스페이스 열기

```bash
cd mobile/ios
open MoodEQ.xcworkspace
```

⚠️ 반드시 `.xcworkspace`를 열어야 합니다 (`.xcodeproj` 아님).

### 2. Swift/Obj-C 파일을 Xcode 타깃에 추가

좌측 Project Navigator에서 `MoodEQ` 폴더(노란 폴더) 우클릭 → **"Add Files to MoodEQ..."**

다음 4개 파일 선택:
- `RoomScanner.swift`
- `RoomScanViewController.swift`
- `RoomScanner.m`
- `MoodEQ-Bridging-Header.h`

다이얼로그에서 **"Add to targets: MoodEQ"가 체크**되어 있는지 반드시 확인.

### 3. RoomPlan 프레임워크 추가

좌측에서 **MoodEQ 프로젝트(파란 아이콘)** 클릭 → **TARGETS: MoodEQ** → **General** 탭
→ **Frameworks, Libraries, and Embedded Content** → **+** 버튼
→ "RoomPlan" 검색 → 추가

### 4. Bridging Header 경로 설정

같은 화면에서 **Build Settings** 탭 → 검색창에 "Bridging Header" 입력
→ **Objective-C Bridging Header** 값에 다음 입력:
```
MoodEQ/MoodEQ-Bridging-Header.h
```

### 5. iOS Deployment Target 확인

**General** 탭 → **Minimum Deployments** → **iOS 16.0** 확인 (Podfile에서 이미 설정했지만 Xcode 프로젝트 설정도 일치해야 함)

### 6. Signing 설정

**Signing & Capabilities** 탭 → **Team** 드롭다운에서 본인 Apple ID 선택
→ Bundle Identifier를 본인 고유값으로 변경 (예: `com.본인이름.moodeq`)

## 실기기 테스트

1. iPhone Pro 모델을 USB로 Mac에 연결
2. iPhone: 설정 → 개인정보 보호 및 보안 → 개발자 모드 → 켬 (재시작 필요)
3. Xcode 상단에서 디바이스 선택 → ▶️ 빌드
4. 첫 실행 시 iPhone에서: 설정 → 일반 → VPN 및 기기 관리 → 본인 Apple ID → 신뢰

## 검증 체크리스트

- [ ] 앱 실행 후 홈에서 "📍 스피커 위치 자동 배정" 탭
- [ ] "디바이스 호환성 확인 중..." 잠시 표시 후 "🎯 방 스캔 시작" 버튼 표시
- [ ] 화면 상단에 "📍 스캔 시작 위치 = 청취 위치" 안내 박스 표시
- [ ] 시뮬레이터나 일반 iPhone에서는 "지원되지 않는 기기" 화면 표시 확인
- [ ] 실기기(Pro 모델)에서 "방 스캔 시작" 누르면 카메라 권한 팝업
- [ ] 권한 허용 후 RoomPlan UI 모달 표시 (카메라 영상 + 파란 와이어프레임)
- [ ] 방을 스캔하면 가구 위에 흰 박스와 영문 라벨 표시
- [ ] "완료" 누르면 모달 닫히고 React Native 화면에 결과 박스 표시
- [ ] 결과 박스에 한글 카테고리(소파, 탁자 등)와 크기, 신뢰도 표시
- [ ] 벽 1~2개만 인식된 상태에서 완료 시 "스캔이 충분하지 않아요" Alert 표시 (재시도 유도)
- [ ] Xcode 콘솔에 `========== ROOM SCAN RESULT ==========` 로그와 객체 목록 출력
- [ ] Metro 콘솔에 `=== ROOM SCAN RESULT (JS) ===` 로그와 JSON 출력

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `RoomScanner is null` 또는 LINKING_ERROR | Swift 파일이 Xcode 타깃에 미포함 | 위 2번 단계 다시 확인. 파일 선택 후 우측 패널 File Inspector → Target Membership에서 MoodEQ 체크 |
| 빌드 실패: "No such module 'RoomPlan'" | RoomPlan 프레임워크 미추가 또는 iOS 15 이하 타깃 | 위 3번, 5번 단계 확인 |
| 빌드 실패: Bridging Header 관련 | 헤더 경로 오류 | 위 4번 단계의 경로 정확히 입력 |
| 빌드 실패: "Use of unresolved identifier 'RCTPromiseResolveBlock'" | Bridging Header에 React import 누락 | MoodEQ-Bridging-Header.h 내용 확인 |
| isSupported가 false 반환 | LiDAR 미탑재 기기 또는 iOS 15 | Pro 모델 + iOS 16+ 인지 확인 (시뮬레이터는 항상 false) |
| 시뮬레이터에서 테스트하려고 함 | RoomPlan은 시뮬레이터 미지원 | 반드시 실기기 사용 |
| 항상 "스캔이 충분하지 않아요" 에러 | 방을 충분히 둘러보지 않음 | 360도 회전하며 모든 벽을 비춰야 함. 좁은 영역만 스캔 시 벽이 3개 미만으로 인식됨 |
