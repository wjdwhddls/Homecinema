// constants/config.ts — 백엔드 URL 및 앱 설정
import {Platform, NativeModules} from 'react-native';

const PORT = 8000;

// ── 백엔드 모드 선택 ────────────────────────────────────────────────
// 'local'  : 본인 Mac 서버 (Metro IP 자동 감지). xRIR 추론 제외한 모든 기능 동작.
// 'remote' : 원격 서버 (도현 ngrok URL). xRIR 포함 전체 기능.
//
// 모드 전환은 아래 BACKEND_MODE 상수를 직접 수정하고 앱 재빌드.
// (Info.plist 기반 동적 전환은 NativeModules.SettingsManager가 임의 plist 키를
//  노출하지 않아 동작하지 않음 — 네이티브 브릿지 추가 전엔 이 파일 직접 수정.)
type BackendMode = 'local' | 'remote';

const BACKEND_MODE = 'local' as BackendMode;

// 'remote' 모드에서 사용할 ngrok URL.
// ngrok URL이 바뀌면 이 상수를 갱신 후 재빌드.
const REMOTE_BACKEND_URL = 'https://mark-investigative-equinely.ngrok-free.dev';

// 'local' 모드 iOS 폴백 IP (Metro 호스트 감지 실패 시 사용).
// Wi-Fi 바뀌었는데 자동 감지가 안 되면 `ipconfig getifaddr en0` 로 확인 후 갱신.
const FALLBACK_LOCAL_IP = '192.168.0.21';

// ── Metro 호스트 자동 감지 ──────────────────────────────────────────
// RN Dev 빌드: Metro 번들러가 내려준 scriptURL 에서 Mac 호스트 IP 뽑아 백엔드 포트로 재활용
// → Wi-Fi 바뀌어도 Metro 만 재기동하면 자동으로 새 IP 적용
const getMetroHost = (): string | null => {
  try {
    const scriptURL: string | undefined = (NativeModules as any).SourceCode?.scriptURL;
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
  if (BACKEND_MODE === 'remote') {
    return REMOTE_BACKEND_URL;
  }

  // 'local' 모드
  if (Platform.OS === 'android') {
    // Android 에뮬레이터는 10.0.2.2 로 호스트 머신 localhost 접근
    return `http://10.0.2.2:${PORT}`;
  }

  // iOS 실기기: Metro 호스트 IP 재활용 → 실패 시 수동 폴백
  const metroHost = getMetroHost();
  if (metroHost) return `http://${metroHost}:${PORT}`;
  return `http://${FALLBACK_LOCAL_IP}:${PORT}`;
};

export const BACKEND_URL = getBackendURL();
export const BACKEND_MODE_ACTIVE = BACKEND_MODE;
export const API_TIMEOUT = 120000; // 2분 (대용량 업로드 고려)
