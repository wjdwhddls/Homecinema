// src/api/eq.ts — EQ 분석 API 클라이언트

import {apiClient} from './client';

export interface EQBand {
  freq: number;           // 중심 주파수 (Hz)
  theory_gain_db: number; // 이론적 보정값
  actual_gain_db: number; // 실제 적용값 (안전 제한 적용)
}

export interface SimpleEQ {
  bass:   {gain_db: number; label: 'Strong' | 'Normal' | 'Weak'};
  mid:    {gain_db: number; label: 'Strong' | 'Normal' | 'Weak'};
  treble: {gain_db: number; label: 'Strong' | 'Normal' | 'Weak'};
}

export interface ParametricFilter {
  freq:    number;  // 중심 주파수 (Hz)
  gain_db: number;  // gain (dB)
  Q:       number;  // Q 팩터
}

export interface EQCurve {
  freqs:        number[]; // log-spaced 20~20000 Hz, 96 points
  measured_db:  number[]; // 측정된 transfer function H(f), 평균 0 정규화
  corrected_db: number[]; // 보정 게인 적용 후 H'(f)
  y_range_db:   number;   // 차트 권장 ±범위 (보통 9)
}

export interface EQAnalysisResponse {
  status:     'success' | 'error';
  bands:      EQBand[];           // 23밴드 보정값
  simple:     SimpleEQ;           // Bass/Mid/Treble 요약
  parametric: ParametricFilter[]; // Parametric EQ 필터 (최대 5개)
  curve?:     EQCurve;            // 시각화용 곡선 (백엔드 호환성 위해 옵셔널)
}

/**
 * sweep + 녹음 파일 → EQ 분석 요청
 *
 * @param sweepUri    번들의 sweep.wav URI (getSweepUri()로 얻은 값)
 * @param recordedUri 최적 위치에서 녹음한 recorded.wav URI
 */
export async function analyzeEQ(
  sweepUri: string,
  recordedUri: string,
): Promise<EQAnalysisResponse> {
  const formData = new FormData();

  formData.append('sweep', {
    uri:  sweepUri,
    type: 'audio/wav',
    name: 'sweep.wav',
  } as any);

  formData.append('recorded', {
    uri:  recordedUri,
    type: 'audio/wav',
    name: 'recorded.wav',
  } as any);

  const res = await apiClient.post('/api/eq/analyze', formData, {
    headers: {'Content-Type': 'multipart/form-data'},
    timeout: 60000,
  });

  return res.data;
}
