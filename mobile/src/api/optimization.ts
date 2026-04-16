// api/optimization.ts — 스피커 최적 배치 API 클라이언트
import {apiClient} from './client';
import {CapturedRoom} from '../native/RoomScanner';

export type ConfigType = 'single' | 'stereo';

export interface UserMaterialSelection {
  object_id: string;
  material_key: string;
}

export interface SpeakerDimensionsPayload {
  width_m: number;
  height_m: number;
  depth_m: number;
}

export interface OptimizeRequest {
  roomplan_scan: CapturedRoom;
  speaker_dimensions: SpeakerDimensionsPayload;
  listener_height_m?: number;
  config_type?: ConfigType;
  user_material_selections?: UserMaterialSelection[];
}

export interface SpeakerPosition {
  x: number;
  y: number;
  z: number;
}

export interface AcousticMetrics {
  rt60_seconds: number;
  rt60_low: number;
  rt60_mid: number;
  standing_wave_severity_db: number;
  flatness_db: number;
  early_reflection_ratio: number;
  direct_to_reverb_ratio_db: number;
}

export interface OptimalResult {
  placement: {
    left: SpeakerPosition;
    right: SpeakerPosition;
    listener: SpeakerPosition;
  };
  score: number;
  metrics: AcousticMetrics;
  rank: number;
}

export interface RoomSummary {
  wall_count: number;
  object_count: number;
  floor_area_m2: number;
  height_m: number;
  volume_m3: number;
}

export interface OptimizeResponse {
  status: 'success' | 'insufficient_data' | 'error';
  job_id: string;
  best: OptimalResult | null;
  top_alternatives: OptimalResult[];
  room_summary: RoomSummary | null;
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
  estimated_remaining_seconds?: number | null;
  result?: OptimizeResponse | null;
}

export async function startOptimization(
  request: OptimizeRequest,
): Promise<StartOptimizationResponse> {
  const res = await apiClient.post('/api/optimize/speakers', request);
  return res.data;
}

// 취소 전용 에러. 호출측에서 사용자 취소(언마운트/뒤로가기)를
// 일반 실패와 구분하기 위해 사용.
export class OptimizationAbortedError extends Error {
  constructor(message = '최적화가 취소되었습니다.') {
    super(message);
    this.name = 'OptimizationAbortedError';
  }
}

export async function getOptimizationStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  const res = await apiClient.get(`/api/optimize/status/${jobId}`, {signal});
  return res.data;
}

export async function waitForOptimization(
  jobId: string,
  onProgress?: (percent: number) => void,
  pollIntervalMs: number = 3000,
  timeoutMs: number = 300_000,
  signal?: AbortSignal,
): Promise<OptimizeResponse> {
  const throwIfAborted = () => {
    if (signal?.aborted) {
      throw new OptimizationAbortedError();
    }
  };

  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    throwIfAborted();
    let status: JobStatusResponse;
    try {
      status = await getOptimizationStatus(jobId, signal);
    } catch (e) {
      // axios 인터셉터가 abort를 네트워크 에러로 바꿔버리므로 signal로 재판별
      if (signal?.aborted) {
        throw new OptimizationAbortedError();
      }
      throw e;
    }
    if (onProgress && typeof status.progress_percent === 'number') {
      onProgress(status.progress_percent);
    }
    if (status.status === 'completed' && status.result) {
      if (status.result.status !== 'success') {
        throw new Error(
          status.result.error_message ?? '최적화를 완료하지 못했습니다.',
        );
      }
      return status.result;
    }
    if (status.status === 'failed') {
      throw new Error(
        status.result?.error_message ?? '최적화 작업이 실패했습니다.',
      );
    }
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
