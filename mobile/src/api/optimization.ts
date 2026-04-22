// api/optimization.ts — 스피커 최적 배치 API 클라이언트 (xRIR 버전)
import {apiClient} from './client';
import {CapturedRoom} from '../native/RoomScanner';

// ── 요청 타입 ────────────────────────────────────────────────────

export interface InitialPositionRequest {
  roomplan_scan: CapturedRoom;
  listener_height_m?: number;
  speaker_height_m?: number;
}

export interface InitialPositionResponse {
  listener_position: {x: number; y: number; z: number};
  initial_speaker_position: {x: number; y: number; z: number};
}

// ── 응답 타입 ────────────────────────────────────────────────────

export interface SpeakerPosition {
  x: number;
  y: number;
  z: number;
}

export interface StereoPlacements {
  left: SpeakerPosition;
  right: SpeakerPosition;
  listener: SpeakerPosition;
}

export interface AcousticMetrics {
  rt60_seconds: number;
  c80_db: number;
  drr_db: number;
  rt60_score: number;
  c80_score: number;
  drr_score: number;
}

export interface OptimalResult {
  placement: StereoPlacements;
  score: number;
  metrics: AcousticMetrics;
  rank: number;
}

export interface OptimizeResponse {
  status: 'success' | 'error';
  job_id: string;
  best: OptimalResult | null;
  top_alternatives: OptimalResult[];
  room_summary: null;
  computation_time_seconds: number;
  warnings: string[];
  error_message?: string | null;
}

export interface StartOptimizationResponse {
  job_id: string;
  status: string;
  estimated_seconds: number;
  poll_url: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress_percent?: number | null;
  result?: OptimizeResponse | null;
}

// ── API 함수 ────────────────────────────────────────────────────

/**
 * STEP 1: 스캔 직후 임시 스피커 위치 요청
 * sweep 측정 전에 스피커를 어디에 둘지 알려줌
 */
export async function getInitialSpeakerPosition(
  req: InitialPositionRequest,
): Promise<InitialPositionResponse> {
  const formData = new FormData();
  formData.append('roomplan_scan', JSON.stringify(req.roomplan_scan));
  if (req.listener_height_m !== undefined) {
    formData.append('listener_height_m', String(req.listener_height_m));
  }
  if (req.speaker_height_m !== undefined) {
    formData.append('speaker_height_m', String(req.speaker_height_m));
  }

  const res = await apiClient.post('/api/xrir/initial-position', formData, {
    headers: {'Content-Type': 'multipart/form-data'},
  });
  return res.data;
}

/**
 * STEP 2: xRIR 최적 스피커 위치 계산 시작
 * recorded.wav (마이크 녹음), sweep.wav (sweep 원본), 임시 스피커 위치 필요
 */
export async function startXRirOptimization(
  roomplan: CapturedRoom,
  recordedWavUri: string,                   // 녹음 파일 URI (file://...)
  sweepWavUri: string,                      // sweep 원본 파일 URI (file://...)
  initialSpeakerPosition: SpeakerPosition,  // 녹음 시 스피커를 놓았던 임시 위치
  options?: {
    listener_height_m?: number;
    speaker_height_m?: number;
    top_k?: number;
  },
): Promise<StartOptimizationResponse> {
  const formData = new FormData();

  formData.append('roomplan_scan', JSON.stringify(roomplan));

  // wav 파일 첨부 (React Native FormData 방식)
  formData.append('recorded', {
    uri: recordedWavUri,
    type: 'audio/wav',
    name: 'recorded.wav',
  } as any);

  formData.append('sweep', {
    uri: sweepWavUri,
    type: 'audio/wav',
    name: 'sweep.wav',
  } as any);

  // 백엔드 필수 파라미터: sweep 측정 시 스피커를 두었던 임시 위치
  formData.append('initial_speaker_x', String(initialSpeakerPosition.x));
  formData.append('initial_speaker_y', String(initialSpeakerPosition.y));
  formData.append('initial_speaker_z', String(initialSpeakerPosition.z));

  if (options?.listener_height_m !== undefined) {
    formData.append('listener_height_m', String(options.listener_height_m));
  }
  if (options?.speaker_height_m !== undefined) {
    formData.append('speaker_height_m', String(options.speaker_height_m));
  }
  if (options?.top_k !== undefined) {
    formData.append('top_k', String(options.top_k));
  }

  const res = await apiClient.post('/api/xrir/speakers', formData, {
    headers: {'Content-Type': 'multipart/form-data'},
    timeout: 30000, // 업로드는 30초 (추론은 폴링으로 기다림)
  });
  return res.data;
}

/**
 * STEP 3: 최적화 결과 폴링 (/api/xrir/status)
 */
export async function getXRirStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  // ✅ xRIR 전용 status 엔드포인트 사용
  const res = await apiClient.get(`/api/xrir/status/${jobId}`, {signal});
  return res.data;
}

export class OptimizationAbortedError extends Error {
  constructor(message = '최적화가 취소되었습니다.') {
    super(message);
    this.name = 'OptimizationAbortedError';
  }
}

/**
 * 폴링 래퍼 — job 완료까지 대기
 */
export async function waitForXRirOptimization(
  jobId: string,
  onProgress?: (percent: number) => void,
  pollIntervalMs: number = 3000,
  timeoutMs: number = 300_000,
  signal?: AbortSignal,
): Promise<OptimizeResponse> {
  const throwIfAborted = () => {
    if (signal?.aborted) throw new OptimizationAbortedError();
  };

  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    throwIfAborted();

    let status: JobStatusResponse;
    try {
      status = await getXRirStatus(jobId, signal);
    } catch (e) {
      if (signal?.aborted) throw new OptimizationAbortedError();
      throw e;
    }

    if (onProgress && typeof status.progress_percent === 'number') {
      onProgress(status.progress_percent);
    }

    if (status.status === 'completed' && status.result) {
      if (status.result.status !== 'success') {
        throw new Error(status.result.error_message ?? '최적화를 완료하지 못했습니다.');
      }
      return status.result;
    }

    if (status.status === 'failed') {
      throw new Error(status.result?.error_message ?? '최적화 작업이 실패했습니다.');
    }

    // 다음 폴링까지 대기 (abort 시 즉시 종료)
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      }, pollIntervalMs);
      const onAbort = () => {
        clearTimeout(timer);
        reject(new OptimizationAbortedError());
      };
      if (signal?.aborted) {
        clearTimeout(timer);
        reject(new OptimizationAbortedError());
        return;
      }
      signal?.addEventListener('abort', onAbort, {once: true});
    });
  }

  throw new Error('최적화 시간이 초과되었습니다.');
}
