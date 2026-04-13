// hooks/useDownloadJob.ts — 영상 다운로드 관리 hook (V3)
import {useEffect, useRef, useState, useCallback} from 'react';
import RNFS from 'react-native-fs';
import {getJobStatus} from '../api/jobs';
import {getOriginalDownloadUrl, getProcessedDownloadUrl} from '../api/jobs';
import {
  createLocalJobDir,
  getLocalOriginalPath,
  getLocalProcessedPath,
  saveLocalMeta,
  localJobExists,
  loadLocalMeta,
} from '../utils/localStorage';

export type DownloadPhase =
  | 'idle'
  | 'downloading_original'
  | 'downloading_processed'
  | 'ready'
  | 'error';

export interface UseDownloadJobResult {
  phase: DownloadPhase;
  originalProgress: number;
  processedProgress: number;
  originalLocalPath: string | null;
  processedLocalPath: string | null;
  error: string | null;
  retry: () => void;
  cancel: () => void;
}

/**
 * 원본과 EQ 적용본을 순차 다운로드하는 hook.
 * 이미 로컬에 존재하면 skip하고 바로 ready 상태.
 */
export function useDownloadJob(jobId: string): UseDownloadJobResult {
  const [phase, setPhase] = useState<DownloadPhase>('idle');
  const [originalProgress, setOriginalProgress] = useState(0);
  const [processedProgress, setProcessedProgress] = useState(0);
  const [originalLocalPath, setOriginalLocalPath] = useState<string | null>(
    null,
  );
  const [processedLocalPath, setProcessedLocalPath] = useState<string | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  // RNFS.downloadFile의 jobId (cancel용)
  const downloadJobIdRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const mountedRef = useRef(true);

  const startDownload = useCallback(async () => {
    if (!mountedRef.current) {
      return;
    }
    cancelledRef.current = false;
    setError(null);

    try {
      // 이미 로컬에 존재하는지 확인
      const exists = await localJobExists(jobId);
      if (exists) {
        const meta = await loadLocalMeta(jobId);
        if (meta) {
          setOriginalLocalPath(meta.original_local_path);
          setProcessedLocalPath(meta.processed_local_path);
          setOriginalProgress(1);
          setProcessedProgress(1);
          setPhase('ready');
          return;
        }
      }

      // 서버에서 job meta 조회 (원본 확장자 확인)
      const jobMeta = await getJobStatus(jobId);
      // meta에서 original_ext를 직접 가져올 수 없으므로 기본값 사용
      const ext = 'mp4'; // skeleton에서는 mp4 기본

      // 로컬 디렉토리 생성
      await createLocalJobDir(jobId);

      // --- 원본 다운로드 ---
      if (cancelledRef.current) {
        return;
      }
      setPhase('downloading_original');
      const origPath = getLocalOriginalPath(jobId, ext);

      const origResult = RNFS.downloadFile({
        fromUrl: getOriginalDownloadUrl(jobId),
        toFile: origPath,
        progress: res => {
          if (res.contentLength > 0) {
            setOriginalProgress(res.bytesWritten / res.contentLength);
          }
        },
        progressDivider: 5,
      });
      downloadJobIdRef.current = origResult.jobId;

      const origResponse = await origResult.promise;
      if (origResponse.statusCode !== 200) {
        throw new Error(`원본 다운로드 실패: HTTP ${origResponse.statusCode}`);
      }
      setOriginalProgress(1);
      setOriginalLocalPath(origPath);

      // --- EQ 적용본 다운로드 ---
      if (cancelledRef.current) {
        return;
      }
      setPhase('downloading_processed');
      const procPath = getLocalProcessedPath(jobId);

      const procResult = RNFS.downloadFile({
        fromUrl: getProcessedDownloadUrl(jobId),
        toFile: procPath,
        progress: res => {
          if (res.contentLength > 0) {
            setProcessedProgress(res.bytesWritten / res.contentLength);
          }
        },
        progressDivider: 5,
      });
      downloadJobIdRef.current = procResult.jobId;

      const procResponse = await procResult.promise;
      if (procResponse.statusCode !== 200) {
        throw new Error(
          `EQ 적용본 다운로드 실패: HTTP ${procResponse.statusCode}`,
        );
      }
      setProcessedProgress(1);
      setProcessedLocalPath(procPath);

      // 로컬 meta 저장
      await saveLocalMeta(jobId, {
        job_id: jobId,
        original_filename: jobMeta.job_id,
        original_ext: ext,
        downloaded_at: new Date().toISOString(),
        original_local_path: origPath,
        processed_local_path: procPath,
        original_size_bytes: origResponse.bytesWritten,
        processed_size_bytes: procResponse.bytesWritten,
      });

      if (mountedRef.current) {
        setPhase('ready');
      }
    } catch (err: any) {
      if (cancelledRef.current) {
        return;
      }
      if (mountedRef.current) {
        setError(err.message || '다운로드 중 오류가 발생했습니다.');
        setPhase('error');
      }
    }
  }, [jobId]);

  // 마운트 시 자동 시작
  useEffect(() => {
    mountedRef.current = true;
    startDownload();

    return () => {
      mountedRef.current = false;
    };
  }, [startDownload]);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    if (downloadJobIdRef.current !== null) {
      RNFS.stopDownload(downloadJobIdRef.current);
    }
    setPhase('idle');
    setOriginalProgress(0);
    setProcessedProgress(0);
  }, []);

  const retry = useCallback(() => {
    setPhase('idle');
    setOriginalProgress(0);
    setProcessedProgress(0);
    setError(null);
    startDownload();
  }, [startDownload]);

  return {
    phase,
    originalProgress,
    processedProgress,
    originalLocalPath,
    processedLocalPath,
    error,
    retry,
    cancel,
  };
}
