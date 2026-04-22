// types/index.ts — 프로젝트 전역 타입 정의
import type {OptimizeResponse} from '../api/optimization';

// 스피커 물리 치수 (사용자 입력, 단위: cm)
export interface SpeakerDimensions {
  width_cm: number;
  height_cm: number;
  depth_cm: number;
}

// 네비게이션 (V3: Playback 추가, Phase 3: OptimizationResult 추가)
export type RootStackParamList = {
  Home: undefined;
  Upload: undefined;
  Result: {jobId: string};
  Playback: {jobId: string};
  SpeakerSize: undefined;
  SpeakerPlacement: {speakerDimensions: SpeakerDimensions};
  OptimizationResult: {result: OptimizeResponse};
  EQMeasurement: {                                  // ← 추가
    optimalPosition: {
      left:  {x: number; y: number; z: number};
      right: {x: number; y: number; z: number};
    };
  };
};

// Job 상태 (V3: 분석 phase 확장)
export type JobStatus =
  | 'uploaded'
  | 'queued'
  | 'analyzing'
  | 'eq_processing'
  | 'completed'
  | 'failed';

// API 응답 타입
export interface UploadResponse {
  status: 'success';
  job_id: string;
  original_filename: string;
  saved_filename: string;
  size_bytes: number;
  message: string;
}

export interface HealthResponse {
  status: 'ok';
  service: string;
  version: string;
  started_at: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  message: string | null;
  created_at: string;
  updated_at: string;
  original_size_bytes: number;
  processed_size_bytes: number | null;
  error_message: string | null;
}

export interface TimelineData {
  schema_version: string;
  metadata: {
    video_filename: string;
    video_duration_sec: number;
    analyzed_at: string;
    model_version: string;
  };
  scenes: TimelineScene[];
  global?: {
    mean_valence?: number;
    mean_arousal?: number;
    total_scenes?: number;
    mean_dialogue_density?: number;
  };
}

export interface TimelineScene {
  scene_id: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  aggregated: {
    valence: number;
    arousal: number;
    category: string;
  };
}

// 파일 정보 타입 (DocumentPicker 결과)
export interface SelectedFile {
  uri: string;
  fileCopyUri: string | null;
  name: string | null;
  size: number | null;
  type: string | null;
}

// 로컬 job 메타 (V3 신규)
export interface LocalJobMeta {
  job_id: string;
  original_filename: string;
  original_ext: string;
  downloaded_at: string;
  original_local_path: string;
  processed_local_path: string;
  original_size_bytes: number;
  processed_size_bytes: number;
}

// RoomPlan 관련 타입은 src/native/RoomScanner.ts에서 직접 import해서 사용
// 여기서는 재export만 제공
export type {
  CapturedRoom,
  CapturedSurface,
  CapturedObject,
  SurfaceCategory,
  ObjectCategory,
  Confidence,
} from '../native/RoomScanner';
