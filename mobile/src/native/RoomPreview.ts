// native/RoomPreview.ts — RoomPreview Swift 브릿지 래퍼
// 3D USDZ 뷰어를 모달로 띄워 청취자/스피커 위치를 시각화
//
// 좌표계 규약:
// - JS → Swift: xRIR 백엔드 좌표 (Z-up, y 앞뒤 반전) 그대로 넘김
// - Swift 내부에서 SceneKit(Y-up)으로 변환
import { NativeModules, Platform } from 'react-native';

/// xRIR 좌표 (백엔드 API가 반환하는 원본)
/// - x: 좌우 (m)
/// - y: 앞뒤 (m, - 부호)
/// - z: 높이 (m)
export interface XRIRPosition {
  x: number;
  y: number;
  z: number;
}

export interface PreviewSpeaker extends XRIRPosition {
  label: string;     // 화면에 표시될 텍스트
  color: string;     // "#RRGGBB" 형식 hex
}

export interface PreviewOptions {
  usdzUri: string;                 // "file:///..." RoomPlan USDZ
  listener?: XRIRPosition;         // 청취자 위치 (xRIR 좌표)
  speakers?: PreviewSpeaker[];     // 스피커 마커들 (xRIR 좌표)
}

interface RoomPreviewNative {
  show(options: PreviewOptions): Promise<null>;
}

const LINKING_ERROR =
  `RoomPreview 네이티브 모듈을 찾을 수 없습니다.\n` +
  `- iOS: Xcode 타깃에 RoomPreview.swift, RoomPreview.m, RoomPreviewViewController.swift 추가 필요\n` +
  `- pod install 후 앱 재빌드`;

const RoomPreview = NativeModules.RoomPreview
  ? (NativeModules.RoomPreview as RoomPreviewNative)
  : (new Proxy({}, {
      get() { throw new Error(LINKING_ERROR); }
    }) as RoomPreviewNative);

/// 3D 뷰어 모달 표시
export const showRoomPreview = (options: PreviewOptions): Promise<null> => {
  if (Platform.OS !== 'ios') {
    return Promise.reject(new Error('3D 미리보기는 iOS 전용입니다'));
  }
  return RoomPreview.show(options);
};

/// 임시 스피커 배치용 마커 색상 (탑뷰 생성기와 동일 팔레트)
export const PREVIEW_COLORS = {
  listener: '#00d4ff',   // 청취자 (파랑)
  initial:  '#ffd700',   // 임시 스피커 (노랑)
  left:     '#00ff88',   // L 채널 (녹색)
  right:    '#ff6b6b',   // R 채널 (빨강)
} as const;
