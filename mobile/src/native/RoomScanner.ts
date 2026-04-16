import { NativeModules, Platform } from 'react-native';

export type SurfaceCategory = 'wall' | 'door' | 'window' | 'opening' | 'floor' | 'unknown';

export type ObjectCategory =
  | 'storage' | 'refrigerator' | 'stove' | 'bed' | 'sink'
  | 'washerDryer' | 'toilet' | 'bathtub' | 'oven' | 'dishwasher'
  | 'table' | 'sofa' | 'chair' | 'fireplace' | 'television' | 'stairs'
  | 'unknown';

export type Confidence = 'low' | 'medium' | 'high' | 'unknown';

export interface CapturedSurface {
  id: string;
  category: SurfaceCategory;
  dimensions: [number, number, number];
  transform: number[];
  confidence: Confidence;
}

export interface CapturedObject {
  id: string;
  category: ObjectCategory;
  dimensions: [number, number, number];
  transform: number[];
  confidence: Confidence;
}

export interface CapturedRoom {
  walls: CapturedSurface[];
  doors: CapturedSurface[];
  windows: CapturedSurface[];
  openings: CapturedSurface[];
  objects: CapturedObject[];
  scannedAt: string;
}

interface RoomScannerNative {
  isSupported(): Promise<boolean>;
  startScan(): Promise<CapturedRoom>;
}

const LINKING_ERROR =
  `RoomScanner 네이티브 모듈을 찾을 수 없습니다.\n` +
  `- iOS: cd ios && pod install 실행 후 앱 재빌드\n` +
  `- Xcode에서 RoomScanner.swift, RoomScanViewController.swift, RoomScanner.m 파일이 빌드 타깃에 포함되어 있는지 확인`;

const RoomScanner = NativeModules.RoomScanner
  ? (NativeModules.RoomScanner as RoomScannerNative)
  : (new Proxy({}, {
      get() { throw new Error(LINKING_ERROR); }
    }) as RoomScannerNative);

export const isRoomScanSupported = async (): Promise<boolean> => {
  if (Platform.OS !== 'ios') return false;
  try {
    return await RoomScanner.isSupported();
  } catch {
    return false;
  }
};

export const startRoomScan = (): Promise<CapturedRoom> => {
  if (Platform.OS !== 'ios') {
    return Promise.reject(new Error('이 기능은 iOS 전용입니다'));
  }
  return RoomScanner.startScan();
};

// 한글 라벨 매핑
export const categoryLabelKR: Record<string, string> = {
  wall: '벽',
  door: '문',
  window: '창문',
  opening: '개구부',
  floor: '바닥',
  sofa: '소파',
  chair: '의자',
  table: '탁자',
  bed: '침대',
  storage: '수납장',
  refrigerator: '냉장고',
  stove: '가스레인지',
  oven: '오븐',
  dishwasher: '식기세척기',
  sink: '싱크대',
  washerDryer: '세탁기',
  toilet: '변기',
  bathtub: '욕조',
  fireplace: '벽난로',
  television: 'TV',
  stairs: '계단',
  unknown: '미분류',
};
