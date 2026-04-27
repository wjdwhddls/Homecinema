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
  OptimizationResult: {
    result: OptimizeResponse;
    usdzUri?: string;
    speakerDimensions?: SpeakerDimensions;
  };
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

// Timeline 타입 — backend timeline_writer.py::build_timeline_dict 과 1:1 대응
// (schema_version "1.0", BASE_MODEL.md §4d / 음향기능.md §3.4 참조)

export type MoodName =
  | 'Tension'
  | 'Sadness'
  | 'Peacefulness'
  | 'JoyfulActivation'
  | 'Tenderness'
  | 'Power'
  | 'Wonder';

export interface EQBand {
  freq_hz: number;
  gain_db: number;
  q: number;
}

export interface TimelineScene {
  scene_idx: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  va: {
    valence: number; // [-1, +1]
    arousal: number; // [-1, +1]
  };
  gate: {
    mean_w_v: number;
    mean_w_a: number;
  };
  mood: {
    name: MoodName;
    idx: number;
  };
  dialogue: {
    density: number; // [0, 1] scene 내 대사 시간 비율
    segments_rel: [number, number][]; // scene 내 상대시간 [start, end]
  };
  eq_preset: {
    original_bands: EQBand[]; // 10-band, dialogue protection 전
    effective_bands: EQBand[]; // dialogue protection 적용 후
  };
}

export interface SpectrogramData {
  hop_ms: number;          // 프레임 간격 (예: 100ms)
  freqs: number[];         // log-spaced 주파수 축 (Hz)
  frames_db: number[][];   // (n_frames, freqs.length) — peak 정규화된 dB
  ref_db: number;          // 0dB ceiling
  floor_db: number;        // -60dB floor
}

export interface TimelineData {
  schema_version: string;
  metadata: {
    video: string;
    duration_sec: number;
    model_version: string;
    analyzed_at: string;
    n_scenes: number;
  };
  config: {
    window_sec: number;
    stride_sec: number;
    ema_alpha: number;
    alpha_d: number;
    num_mood_classes: number;
    batch_size: number;
  };
  scenes: TimelineScene[];
  global: {
    mean_va: {valence: number; arousal: number};
    mood_distribution: Partial<Record<MoodName, number>>;
    avg_dialogue_density: number;
  };
  spectrogram?: SpectrogramData;
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
