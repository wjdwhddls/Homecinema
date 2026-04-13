// api/client.ts — axios 기반 API 클라이언트
import axios, {AxiosError} from 'axios';
import {BACKEND_URL, API_TIMEOUT} from '../constants/config';

export const apiClient = axios.create({
  baseURL: BACKEND_URL,
  timeout: API_TIMEOUT,
  headers: {
    Accept: 'application/json',
  },
});

// 응답 인터셉터: 에러 메시지 통일
apiClient.interceptors.response.use(
  response => response,
  (error: AxiosError) => {
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(
        new Error('요청 시간이 초과되었습니다. 다시 시도해주세요.'),
      );
    }
    if (!error.response) {
      return Promise.reject(
        new Error(
          '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.',
        ),
      );
    }
    const data = error.response.data as any;
    const detail = data?.detail || data?.message;
    if (detail) {
      return Promise.reject(new Error(detail));
    }
    return Promise.reject(new Error(`서버 오류 (${error.response.status})`));
  },
);
