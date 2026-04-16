# Native Modules

## RoomScanner (iOS 전용)

iOS RoomPlan API를 React Native에서 사용하기 위한 브릿지.

### 요구사항
- iOS 16.0 이상
- LiDAR 탑재 디바이스 (iPhone 12 Pro 이상의 Pro 모델, iPad Pro 2020+)

### Xcode 수동 작업 필요

`mobile/ios/MoodEQ/` 폴더에 다음 파일들이 있어야 합니다:
- `RoomScanner.swift`
- `RoomScanViewController.swift`
- `RoomScanner.m`
- `MoodEQ-Bridging-Header.h`

**중요**: 이 파일들은 Xcode 프로젝트 빌드 타깃에 수동으로 추가해야 합니다.

1. `open mobile/ios/MoodEQ.xcworkspace`
2. 좌측 Project Navigator에서 `MoodEQ` 폴더 우클릭 → "Add Files to MoodEQ..."
3. 위 4개 파일 선택, "Add to targets: MoodEQ" 체크 확인 후 추가
4. 좌측에서 MoodEQ 프로젝트(파란 아이콘) 클릭 → TARGETS: MoodEQ → General 탭
5. "Frameworks, Libraries, and Embedded Content" 섹션에서 + 버튼 → "RoomPlan" 검색 후 추가
6. Build Settings → "Objective-C Bridging Header" 검색 → 값에 `MoodEQ/MoodEQ-Bridging-Header.h` 입력
7. General 탭 → Minimum Deployments → iOS 16.0으로 설정

### 사용법

```typescript
import { isRoomScanSupported, startRoomScan } from './RoomScanner';

const supported = await isRoomScanSupported();
if (supported) {
  const room = await startRoomScan();
  console.log(room.objects); // [{ category: 'sofa', dimensions: [...] }, ...]
}
```
