// native/SweepRecorder.ts — SweepRecorder Swift 브릿지 래퍼
import { NativeModules, Platform } from 'react-native';

// ──────────────────────────────────────────────
// 타입 정의
// ──────────────────────────────────────────────

export interface SweepRecordResult {
  uri: string;       // 녹음된 ref_rir.wav 로컬 경로
  durationMs: number; // 실제 녹음 시간 (ms)
}

interface SweepRecorderNative {
  /** 마이크 권한 요청 */
  requestPermission(): Promise<boolean>;
  /** sweep 재생 + 동시 녹음 → wav 파일 경로 반환 */
  recordSweep(durationSec: number, sampleRate: number): Promise<SweepRecordResult>;
  /** 녹음 파일 삭제 (다음 측정 전 초기화용) */
  deleteRecording(uri: string): Promise<void>;
}

const LINKING_ERROR =
  `SweepRecorder 네이티브 모듈을 찾을 수 없습니다.\n` +
  `- ios/SweepRecorder.swift, SweepRecorder.m 파일이 빌드 타깃에 포함되어 있는지 확인\n` +
  `- cd ios && pod install 실행 후 앱 재빌드`;

const SweepRecorderNative: SweepRecorderNative =
  NativeModules.SweepRecorder
    ? (NativeModules.SweepRecorder as SweepRecorderNative)
    : (new Proxy({}, {
        get() { throw new Error(LINKING_ERROR); }
      }) as SweepRecorderNative);

// ──────────────────────────────────────────────
// 공개 API
// ──────────────────────────────────────────────

/**
 * 마이크 권한 요청
 * 최초 1회 호출, 이후엔 시스템이 캐시
 */
export const requestMicPermission = async (): Promise<boolean> => {
  if (Platform.OS !== 'ios') {
    throw new Error('SweepRecorder는 iOS 전용입니다');
  }
  return SweepRecorderNative.requestPermission();
};

/**
 * Sweep 재생 + 동시 녹음
 *
 * @param durationSec  sweep 길이 (기본 5초)
 * @param sampleRate   샘플레이트 (기본 48000 Hz)
 * @returns            녹음 파일 uri + 실제 녹음 시간
 *
 * 사용 예:
 *   const { uri } = await recordSweep();
 *   // uri를 ref_rir.wav로 서버에 전송
 */
export const recordSweep = async (
  durationSec: number = 5,
  sampleRate: number = 48000,
): Promise<SweepRecordResult> => {
  if (Platform.OS !== 'ios') {
    throw new Error('SweepRecorder는 iOS 전용입니다');
  }

  const hasPermission = await requestMicPermission();
  if (!hasPermission) {
    throw new Error('마이크 권한이 없습니다. 설정에서 허용해주세요.');
  }

  return SweepRecorderNative.recordSweep(durationSec, sampleRate);
};

/**
 * 이전 녹음 파일 삭제
 * 다음 측정 전에 호출해서 저장공간 정리
 */
export const deleteRecording = async (uri: string): Promise<void> => {
  if (Platform.OS !== 'ios') return;
  return SweepRecorderNative.deleteRecording(uri);
};
