// api/upload.ts — 영상 파일 업로드
import {Platform} from 'react-native';
import {apiClient} from './client';
import {SelectedFile, UploadResponse} from '../types';

/**
 * 서버에 영상 파일을 업로드합니다.
 * multipart/form-data로 전송하며, 업로드 중 진행률 콜백을 제공합니다.
 */
export async function uploadVideo(
  file: SelectedFile,
  onProgress?: (progress: number) => void,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', {
    uri: Platform.OS === 'ios' ? file.fileCopyUri : file.uri,
    type: file.type || 'video/mp4',
    name: file.name || 'upload.mp4',
  } as any);

  const response = await apiClient.post<UploadResponse>(
    '/api/upload',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: progressEvent => {
        if (onProgress && progressEvent.total) {
          onProgress(progressEvent.loaded / progressEvent.total);
        }
      },
    },
  );

  return response.data;
}
