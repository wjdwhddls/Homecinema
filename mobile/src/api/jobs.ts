// api/jobs.ts — job 상태 조회, 다운로드 URL, 삭제
import {apiClient} from './client';
import {BACKEND_URL} from '../constants/config';
import {JobStatusResponse, TimelineData} from '../types';

/** Job 상태 조회 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await apiClient.get<JobStatusResponse>(
    `/api/jobs/${jobId}/status`,
  );
  return response.data;
}

/** 분석 타임라인 조회 */
export async function getJobTimeline(jobId: string): Promise<TimelineData> {
  const response = await apiClient.get<TimelineData>(
    `/api/jobs/${jobId}/timeline`,
  );
  return response.data;
}

/** Job 삭제 (서버측) */
export async function deleteJob(jobId: string): Promise<void> {
  await apiClient.delete(`/api/jobs/${jobId}`);
}

/** 원본 다운로드 URL 생성 (RNFS.downloadFile에서 직접 사용) */
export function getOriginalDownloadUrl(jobId: string): string {
  return `${BACKEND_URL}/api/jobs/${jobId}/download/original`;
}

/** EQ 적용본 다운로드 URL 생성 (RNFS.downloadFile에서 직접 사용) */
export function getProcessedDownloadUrl(jobId: string): string {
  return `${BACKEND_URL}/api/jobs/${jobId}/download/processed`;
}
