# Mood EQ Mobile

영화 분위기 자동 EQ 적용 프로젝트의 모바일 앱입니다. React Native CLI 기반으로 영상 업로드, 분석 상태 확인, 다운로드 및 A/B 비교 재생 기능을 제공합니다. 현재는 ML 분석 없이 업로드와 다운로드/재생 UI만 동작하는 skeleton 버전입니다.

## 사전 요구사항 (환경 셋업)

### 공통
- Node.js 18 이상
- npm 또는 yarn

### Android 개발용
- JDK 17
- Android Studio + Android SDK (API Level 33 이상)
- Android 에뮬레이터 또는 실기기
- `ANDROID_HOME` 환경 변수 설정:
  ```bash
  export ANDROID_HOME=$HOME/Library/Android/sdk
  export PATH=$PATH:$ANDROID_HOME/emulator
  export PATH=$PATH:$ANDROID_HOME/platform-tools
  ```

### iOS 개발용 (Mac만)
- Xcode 최신 버전
- CocoaPods
- iOS Simulator

**공식 가이드**: https://reactnative.dev/docs/environment-setup (React Native CLI Quickstart 탭)

## React Native 버전 및 Architecture 결정

- **React Native 0.74.x 고정**
- **New architecture (Fabric/TurboModules) 비활성화**

이유: 서드파티 라이브러리(`react-native-document-picker`, `react-native-video`, `react-native-fs`) 호환성 보장 + 학사 일정에서 native 빌드 이슈 회피.

## 설치 방법

```bash
cd mobile
npm install
cd ios && pod install && cd ..    # Mac만
```

## 백엔드 URL 설정

- **iOS 시뮬레이터**: 기본 (`localhost`) — 변경 불필요
- **Android 에뮬레이터**: 기본 (`10.0.2.2`) — 변경 불필요
- **실기기**: `src/constants/config.ts`의 `DEVICE_BACKEND_URL`을 본인 컴퓨터의 LAN IP로 변경

## 실행 방법

```bash
npx react-native start                # 터미널 1: Metro bundler
npx react-native run-android          # 터미널 2 (또는 run-ios)
```

## 화면 흐름

```
HomeScreen
   ├─ "영상 업로드" → UploadScreen
   │                     ├─ 영상 선택 → "업로드"
   │                     └─ 업로드 성공 → ResultScreen (replace)
   │                                         ├─ 상태 polling
   │                                         ├─ "분석 완료" 시 PlaybackScreen 자동 이동
   │                                         └─ 개발 테스트용 "강제 이동" 버튼
   │                                                ↓
   │                                         PlaybackScreen
   │                                            ├─ 원본/EQ 적용본 순차 다운로드
   │                                            ├─ react-native-video로 재생
   │                                            ├─ A/B 토글 (원본 ↔ EQ 적용)
   │                                            └─ 삭제 버튼 (서버 + 로컬)
   │
   └─ "스피커 위치 자동 배정" → SpeakerPlacementScreen (다른 팀 구현)
```

## 개발 테스트 방법

현재 skeleton 단계에서는 서버의 ML/EQ 처리가 없기 때문에 `processed.mp4`가 실제 존재하지 않습니다. A/B 토글 UI를 테스트하려면:

1. 백엔드 `.env`에 `DEV_FAKE_PROCESSED=true` 설정 후 서버 재시작
2. 앱에서 영상 업로드
3. ResultScreen의 "테스트용: 강제 재생 화면으로 이동" 버튼 누름
4. PlaybackScreen이 원본을 두 번 다운로드 (original + processed 엔드포인트 둘 다 원본 반환)
5. A/B 토글을 눌러보면 UI가 동작하지만 소리는 동일 (둘 다 원본이므로)

실제 EQ 적용본은 Phase 2 이후 서버에 ML과 pedalboard 처리가 통합되면 정상 반환됩니다.

## 로컬 저장 공간 관리

앱은 다운로드한 영상을 `{DocumentDirectoryPath}/jobs/{id}/`에 저장합니다. 한 영상당 약 원본 크기의 2배가 필요합니다 (original + processed). 사용자는 PlaybackScreen의 삭제 버튼으로 개별 job을 제거할 수 있습니다.

향후 확장: HomeScreen에 로컬 저장 job 목록 + 일괄 정리 기능 추가 예정.

## 자주 발생하는 문제와 해결

### Metro bundler 캐시 문제
```bash
npx react-native start --reset-cache
```

### iOS pod install 실패
```bash
cd ios
pod deintegrate
pod install --repo-update
cd ..
```

### Android 빌드 실패 (Gradle)
```bash
cd android
./gradlew clean
cd ..
```

### `react-native-document-picker` iOS 파일 접근 실패
- v9+ API 사용 확인: `import DocumentPicker, { types } from 'react-native-document-picker'`
- `types.video` 사용 (`DocumentPicker.types.video` 아님)
- `copyTo: 'cachesDirectory'` 옵션 확인
- FormData에 `result.fileCopyUri` 사용 (iOS)

### `react-native-video` 재생 실패
- iOS: `pod install` 재실행
- Android: `gradle.properties`의 `newArchEnabled=false` 확인
- 로컬 파일 경로가 `file://` 스킴으로 시작하는지 확인

### `react-native-fs` 다운로드 실패
- iOS: Info.plist의 `NSAppTransportSecurity` 설정 확인 (개발 시 localhost HTTP 허용)
- Android: `AndroidManifest.xml`의 `android:usesCleartextTraffic="true"` 확인

### HTTP cleartext 이슈 (개발 환경)
- 개발 서버가 HTTP이므로 Android/iOS 모두 기본적으로 차단될 수 있음
- Android: `android/app/src/main/AndroidManifest.xml`의 `<application>`에 `android:usesCleartextTraffic="true"` (이미 설정됨)
- iOS: `ios/MoodEQ/Info.plist`에 `NSAllowsArbitraryLoads` 추가 필요 시
- **운영 배포 시 반드시 HTTPS + 설정 제거**

### New architecture 빌드 오류
- `gradle.properties`의 `newArchEnabled=false` 확인
- iOS Podfile에 `:fabric_enabled => true`가 있다면 제거 또는 false
- `cd ios && pod deintegrate && pod install` 후 재빌드
