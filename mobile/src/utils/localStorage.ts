// utils/localStorage.ts — 로컬 파일 관리 (react-native-fs 래핑)
import RNFS from 'react-native-fs';
import {LocalJobMeta} from '../types';

const JOBS_DIR = `${RNFS.DocumentDirectoryPath}/jobs`;

/** 로컬 job 디렉토리 경로 (생성 없이 반환) */
export function getLocalJobDir(jobId: string): string {
  return `${JOBS_DIR}/${jobId}`;
}

/** 로컬 job 디렉토리 생성 후 경로 반환 */
export async function createLocalJobDir(jobId: string): Promise<string> {
  const dir = getLocalJobDir(jobId);
  await RNFS.mkdir(dir);
  return dir;
}

/** 원본 영상 로컬 경로 */
export function getLocalOriginalPath(jobId: string, ext: string): string {
  return `${getLocalJobDir(jobId)}/original.${ext}`;
}

/** EQ 적용본 로컬 경로 */
export function getLocalProcessedPath(jobId: string): string {
  return `${getLocalJobDir(jobId)}/processed.mp4`;
}

/** 로컬 meta.json 경로 */
export function getLocalMetaPath(jobId: string): string {
  return `${getLocalJobDir(jobId)}/meta.json`;
}

/** 로컬 meta.json 저장 */
export async function saveLocalMeta(
  jobId: string,
  meta: LocalJobMeta,
): Promise<void> {
  const metaPath = getLocalMetaPath(jobId);
  await RNFS.writeFile(metaPath, JSON.stringify(meta, null, 2), 'utf8');
}

/** 로컬 meta.json 읽기. 없으면 null */
export async function loadLocalMeta(
  jobId: string,
): Promise<LocalJobMeta | null> {
  const metaPath = getLocalMetaPath(jobId);
  const exists = await RNFS.exists(metaPath);
  if (!exists) {
    return null;
  }
  const content = await RNFS.readFile(metaPath, 'utf8');
  return JSON.parse(content) as LocalJobMeta;
}

/** 로컬 job 존재 여부 (meta.json 기준) */
export async function localJobExists(jobId: string): Promise<boolean> {
  const metaPath = getLocalMetaPath(jobId);
  return RNFS.exists(metaPath);
}

/** 로컬 job 삭제 (디렉토리 전체) */
export async function deleteLocalJob(jobId: string): Promise<void> {
  const dir = getLocalJobDir(jobId);
  const exists = await RNFS.exists(dir);
  if (exists) {
    await RNFS.unlink(dir);
  }
}

/** 모든 로컬 job 목록 조회 */
export async function listLocalJobs(): Promise<LocalJobMeta[]> {
  const exists = await RNFS.exists(JOBS_DIR);
  if (!exists) {
    return [];
  }

  const items = await RNFS.readDir(JOBS_DIR);
  const metas: LocalJobMeta[] = [];

  for (const item of items) {
    if (item.isDirectory()) {
      const metaPath = `${item.path}/meta.json`;
      const metaExists = await RNFS.exists(metaPath);
      if (metaExists) {
        const content = await RNFS.readFile(metaPath, 'utf8');
        metas.push(JSON.parse(content) as LocalJobMeta);
      }
    }
  }

  return metas;
}

/** 전체 로컬 저장 공간 사용량 (bytes) */
export async function getLocalStorageUsage(): Promise<number> {
  const exists = await RNFS.exists(JOBS_DIR);
  if (!exists) {
    return 0;
  }

  let totalSize = 0;
  const items = await RNFS.readDir(JOBS_DIR);

  for (const item of items) {
    if (item.isDirectory()) {
      const files = await RNFS.readDir(item.path);
      for (const file of files) {
        if (file.isFile()) {
          totalSize += file.size;
        }
      }
    }
  }

  return totalSize;
}
