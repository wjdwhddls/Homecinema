// constants/config.ts — 백엔드 URL 및 앱 설정
import {Platform, NativeModules} from 'react-native';

const PORT = 8000;

// Release 빌드나 Metro IP 추출 실패 시 사용할 수동 폴백
// (Wi-Fi가 바뀌면 `ipconfig getifaddr en0`로 재확인 후 수정)
const FALLBACK_DEVICE_BACKEND_URL = `http://192.168.0.22:${PORT}`;

// Dev 빌드: Metro의 scriptURL에서 Mac 호스트를 추출해 재활용
// → Wi-Fi가 바뀌어도 Metro만 재기동하면 자동으로 새 IP 적용
const getMetroHost = (): string | null => {
  try {
    const scriptURL: string | undefined = (NativeModules as any).SourceCode
      ?.scriptURL;
    if (!scriptURL) return null;
    const match = scriptURL.match(/^https?:\/\/([^:/]+)/);
    const host = match?.[1];
    if (!host || host === 'localhost' || host === '127.0.0.1') return null;
    return host;
  } catch {
    return null;
  }
};

const getBackendURL = (): string => {
  if (Platform.OS === 'android') {
    // Android 에뮬레이터는 10.0.2.2로 호스트 localhost 접근
    return `http://10.0.2.2:${PORT}`;
  }
  // iOS 실기기(Debug): Metro 호스트 IP 재활용
  const metroHost = getMetroHost();
  if (metroHost) {
    return `http://${metroHost}:${PORT}`;
  }
  // Release 또는 추출 실패 시 폴백
  return FALLBACK_DEVICE_BACKEND_URL;
};

export const BACKEND_URL = getBackendURL();
export const API_TIMEOUT = 120000; // 2분 (대용량 업로드 고려)
