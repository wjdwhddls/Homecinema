// native/SweepRecorder.ts — SweepRecorder Swift 브릿지 래퍼
import { NativeModules, Platform } from 'react-native';

// ──────────────────────────────────────────────
// 타입 정의
// ──────────────────────────────────────────────

export interface SweepRecordResult {
  recordedUri: string;  // 마이크 녹음 wav 경로
  sweepUri: string;     // sweep 원본 wav 경로
  durationMs: number;   // 실제 녹음 시간 (ms)
}

interface SweepRecorderNative {
  requestPermission(): Promise<boolean>;
  recordSweep(durationSec: number, sampleRate: number): Promise<SweepRecordResult>;
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

export const requestMicPermission = async (): Promise<boolean> => {
  if (Platform.OS !== 'ios') throw new Error('SweepRecorder는 iOS 전용입니다');
  return SweepRecorderNative.requestPermission();
};

/**
 * Sweep 재생 + 동시 녹음
 * @returns recordedUri (마이크 녹음), sweepUri (sweep 원본)
 */
export const recordSweep = async (
  durationSec: number = 5,
  sampleRate: number = 48000,
): Promise<SweepRecordResult> => {
  if (Platform.OS !== 'ios') throw new Error('SweepRecorder는 iOS 전용입니다');

  const hasPermission = await requestMicPermission();
  if (!hasPermission) {
    throw new Error('마이크 권한이 없습니다. 설정에서 허용해주세요.');
  }

  return SweepRecorderNative.recordSweep(durationSec, sampleRate);
};

export const deleteRecording = async (uri: string): Promise<void> => {
  if (Platform.OS !== 'ios') return;
  return SweepRecorderNative.deleteRecording(uri);
};
