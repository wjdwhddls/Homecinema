// src/native/SweepRecorder.ts
import {NativeModules, Platform} from 'react-native';

interface SweepRecorderNative {
  getSweepUri(): Promise<string>;
  record(sweepAssetName: string): Promise<string>;
}

const LINKING_ERROR =
  `SweepRecorder 네이티브 모듈을 찾을 수 없습니다.\n` +
  `- cd ios && pod install 실행 후 앱 재빌드\n` +
  `- Xcode에서 SweepRecorder.swift, SweepRecorder.m 이 빌드 타깃에 있는지 확인`;

const Native = NativeModules.SweepRecorder
  ? (NativeModules.SweepRecorder as SweepRecorderNative)
  : (new Proxy({}, {get() { throw new Error(LINKING_ERROR); }}) as SweepRecorderNative);

/**
 * 번들에 내장된 sweep.wav의 file:// URI 반환
 * 서버 전송 시 sweep 원본 파일로 사용
 */
export async function getSweepUri(): Promise<string> {
  if (Platform.OS !== 'ios') throw new Error('iOS 전용입니다.');
  return Native.getSweepUri();
}

/**
 * sweep 신호 재생 + 동시 마이크 녹음
 * @returns 녹음된 recorded.wav 파일 URI (file://...)
 */
export async function recordSweep(sweepAssetName = 'sweep'): Promise<string> {
  if (Platform.OS !== 'ios') throw new Error('iOS 전용입니다.');
  return Native.record(sweepAssetName);
}
