// constants/config.ts — 백엔드 URL 및 앱 설정
import {Platform} from 'react-native';

const PORT = 8000;

// 실기기 사용 시 이 값을 본인 컴퓨터의 LAN IP로 변경
const DEVICE_BACKEND_URL = `http://192.168.0.10:${PORT}`;

const getBackendURL = (): string => {
  if (Platform.OS === 'android') {
    // Android 에뮬레이터는 10.0.2.2로 호스트 localhost 접근
    return `http://10.0.2.2:${PORT}`;
  } else if (Platform.OS === 'ios') {
    return `http://localhost:${PORT}`;
  }
  return DEVICE_BACKEND_URL;
};

export const BACKEND_URL = getBackendURL();
export const API_TIMEOUT = 120000; // 2분 (대용량 업로드 고려)
