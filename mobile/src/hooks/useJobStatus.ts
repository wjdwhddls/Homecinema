// hooks/useJobStatus.ts — job 상태 polling hook
import {useEffect, useRef, useState, useCallback} from 'react';
import {getJobStatus} from '../api/jobs';
import {JobStatus} from '../types';

interface UseJobStatusOptions {
  /** polling 간격 (ms). 기본 2000 */
  pollInterval?: number;
  /** 최대 재시도 횟수. 기본 300 (10분) */
  maxRetries?: number;
  /** 이 상태에 도달하면 polling 중지 */
  stopOnStatuses?: JobStatus[];
}

interface UseJobStatusResult {
  status: JobStatus | null;
  progress: number;
  message: string | null;
  errorMessage: string | null;
  isPolling: boolean;
}

/**
 * job 상태를 주기적으로 polling하는 hook.
 * stopOnStatuses에 해당하는 상태에 도달하면 자동 중지합니다.
 */
export function useJobStatus(
  jobId: string,
  options?: UseJobStatusOptions,
): UseJobStatusResult {
  const {
    pollInterval = 2000,
    maxRetries = 300,
    stopOnStatuses = ['completed', 'failed'],
  } = options || {};

  const [status, setStatus] = useState<JobStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(true);

  const retryCountRef = useRef(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    setIsPolling(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    try {
      const data = await getJobStatus(jobId);
      setStatus(data.status);
      setProgress(data.progress);
      setMessage(data.message);
      setErrorMessage(data.error_message);

      // 목표 상태에 도달하면 polling 중지
      if (stopOnStatuses.includes(data.status)) {
        stopPolling();
        return;
      }

      retryCountRef.current += 1;
      if (retryCountRef.current >= maxRetries) {
        setErrorMessage('상태 확인 시간이 초과되었습니다.');
        stopPolling();
      }
    } catch (err: any) {
      setErrorMessage(err.message || '상태 조회 중 오류가 발생했습니다.');
      stopPolling();
    }
  }, [jobId, maxRetries, stopOnStatuses, stopPolling]);

  useEffect(() => {
    // 최초 1회 즉시 호출
    poll();

    // 이후 주기적 polling
    intervalRef.current = setInterval(poll, pollInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [poll, pollInterval]);

  return {status, progress, message, errorMessage, isPolling};
}
