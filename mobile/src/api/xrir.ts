// api/xrir.ts — xRIR 스피커 최적화 API
import { Platform } from 'react-native';
import { apiClient } from './client';

export interface Position {
  x: number;
  y: number;
  z: number;
}

export interface InitialPositionResponse {
  listener_position: Position;
  initial_speaker_position: Position;
}

export interface SpeakerResult {
  position: Position;
  score: number;
  rt60: number;
  c80: number;
  drr: number;
}

export interface SpeakersResponse {
  job_id: string;
  status: string;
  poll_url: string;
}

export interface JobStatusResponse {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: {
    best: SpeakerResult;
    top_alternatives: SpeakerResult[];
  };
  error?: string;
}

export async function getInitialPosition(
  roomplanScan: object,
  listenerHeightM: number = 1.2,
  speakerHeightM: number = 1.2,
): Promise<InitialPositionResponse> {
  const formData = new FormData();
  formData.append('roomplan_scan', JSON.stringify(roomplanScan));
  formData.append('listener_height_m', String(listenerHeightM));
  formData.append('speaker_height_m', String(speakerHeightM));

  const response = await apiClient.post<InitialPositionResponse>(
    '/api/xrir/initial-position',
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return response.data;
}

/**
 * 최적 스피커 위치 추론 요청
 * recorded.wav + sweep.wav → 서버에서 deconvolution → xRIR 추론
 * mesh.bin은 선택사항 (있으면 더 정확한 depth.npy 생성)
 */
export async function requestSpeakerOptimization(
  roomplanScan: object,
  recordedUri: string,
  sweepUri: string,
  meshBinUri?: string,         // ← 선택사항 (LiDAR mesh.bin)
  listenerHeightM: number = 1.2,
  speakerHeightM: number = 1.2,
  topK: number = 5,
): Promise<SpeakersResponse> {
  const formData = new FormData();
  formData.append('roomplan_scan', JSON.stringify(roomplanScan));
  formData.append('recorded', {
    uri: recordedUri,
    type: 'audio/wav',
    name: 'recorded.wav',
  } as any);
  formData.append('sweep', {
    uri: sweepUri,
    type: 'audio/wav',
    name: 'sweep.wav',
  } as any);

  // mesh.bin 있을 때만 추가
  if (meshBinUri) {
    formData.append('mesh', {
      uri: meshBinUri,
      type: 'application/octet-stream',
      name: 'mesh.bin',
    } as any);
  }

  formData.append('listener_height_m', String(listenerHeightM));
  formData.append('speaker_height_m', String(speakerHeightM));
  formData.append('top_k', String(topK));

  const response = await apiClient.post<SpeakersResponse>(
    '/api/xrir/speakers',
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return response.data;
}

export async function pollJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await apiClient.get<JobStatusResponse>(
    `/api/xrir/status/${jobId}`,
  );
  return response.data;
}

export async function waitForJobCompletion(
  jobId: string,
  onStatusUpdate?: (status: string) => void,
  intervalMs: number = 2000,
  maxAttempts: number = 60,
): Promise<JobStatusResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    const result = await pollJobStatus(jobId);
    onStatusUpdate?.(result.status);

    if (result.status === 'completed') return result;
    if (result.status === 'failed') {
      throw new Error(result.error || '추론 작업이 실패했습니다.');
    }

    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }
  throw new Error('추론 작업이 시간 초과되었습니다.');
}
