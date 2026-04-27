// screens/UploadScreen.tsx — EQ 설정 진입 (시네마틱 라이브 스펙트럼)
//
// 메인 페이지와의 대비 전략
//   · 메인  = 가로 flowing waveform (hairline curve)
//   · 업로드 = 세로 live spectrum analyzer (32 band bars)
// 공통 디자인 시스템
//   · 흑색 배경 + 중앙 radial glow
//   · 오렌지/퍼플/시안 3-점 액센트
//   · glass pill 컨트롤
//
// 외부 의존성 추가 없이 react-native-svg 만 사용.
import React, {useEffect, useRef, useState} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  Dimensions,
  Animated,
  Easing,
  Alert,
} from 'react-native';
import DocumentPicker, {types} from 'react-native-document-picker';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  RadialGradient,
  Stop,
  Rect,
  Path,
  G,
} from 'react-native-svg';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList, SelectedFile} from '../types';
import {uploadVideo} from '../api/upload';

type Props = NativeStackScreenProps<RootStackParamList, 'Upload'>;

const {width: SCREEN_W} = Dimensions.get('window');

// ── Equalizer 파라미터 ───────────────────────────────────────
const NUM_BARS = 32;
const VIZ_W = SCREEN_W - 48;
const VIZ_H = 220;
const BAR_SPACING = 2;
const BAR_W = (VIZ_W - (NUM_BARS - 1) * BAR_SPACING) / NUM_BARS;
const FLOOR_Y = VIZ_H * 0.82; // 바 하단(= reflection 시작점)

// ──────────────────────────────────────────────────────────────
// 메인 스크린
// ──────────────────────────────────────────────────────────────
export default function UploadScreen({navigation}: Props) {
  const [selectedFile, setSelectedFile] = useState<SelectedFile | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [phase, setPhase] = useState(0);

  const fade = useRef(new Animated.Value(0)).current;

  // 진입 페이드인 + 주파수 위상 애니메이션
  useEffect(() => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 1400,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();

    let raf: number;
    let last = Date.now();
    const tick = () => {
      const now = Date.now();
      const dt = (now - last) / 1000;
      last = now;
      setPhase(p => p + dt);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [fade]);

  // ── 파일 선택
  const handlePickFile = async () => {
    try {
      const result = await DocumentPicker.pickSingle({
        type: [types.video],
        copyTo: 'cachesDirectory',
      });
      setSelectedFile({
        uri: result.uri,
        fileCopyUri: result.fileCopyUri ?? null,
        name: result.name ?? null,
        size: result.size ?? null,
        type: result.type ?? null,
      });
    } catch (err) {
      if (DocumentPicker.isCancel(err)) {
        return;
      }
      Alert.alert('오류', '파일 선택 중 오류가 발생했습니다.');
    }
  };

  // ── 업로드
  const handleUpload = async () => {
    if (!selectedFile) {
      return;
    }
    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await uploadVideo(selectedFile, p =>
        setUploadProgress(p),
      );
      // 업로드 성공 시 Alert 없이 바로 분석 화면으로 — 시네마틱한 끊김 없는 전환
      navigation.replace('Result', {jobId: response.job_id});
    } catch (err: any) {
      Alert.alert(
        '업로드 실패',
        err.message || '업로드 중 오류가 발생했습니다.',
      );
    } finally {
      setIsUploading(false);
    }
  };

  const formatSize = (bytes: number | null): string => {
    if (bytes === null) {
      return '크기 알 수 없음';
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const canSubmit = !!selectedFile && !isUploading;

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <BackgroundGlow />

      <SafeAreaView style={styles.safe}>
        {/* ── Header ── */}
        <Animated.View style={[styles.header, {opacity: fade}]}>
          <Text style={styles.eyebrow}>EQ SETTINGS</Text>
          <Text style={styles.hero}>
            영상의 감정을{'\n'}주파수로 번역합니다
          </Text>
        </Animated.View>

        {/* ── Live Spectrum ── */}
        <Animated.View style={[styles.vizContainer, {opacity: fade}]}>
          <Equalizer phase={phase} active={isUploading} />
          <View style={styles.freqLabels}>
            {['20', '100', '1k', '10k', '20k'].map(f => (
              <Text key={f} style={styles.freqLabel}>
                {f}
              </Text>
            ))}
          </View>
        </Animated.View>

        {/* ── Controls ── */}
        <Animated.View style={[styles.controls, {opacity: fade}]}>
          {/* 파일 선택 */}
          <TouchableOpacity
            activeOpacity={0.7}
            onPress={handlePickFile}
            disabled={isUploading}
            style={[styles.pickBtn, isUploading && styles.dim]}>
            <View style={styles.pickBtnInner}>
              <PlusIcon />
              <Text style={styles.pickText}>
                {selectedFile ? '다른 영상 선택' : '영상 선택'}
              </Text>
            </View>
          </TouchableOpacity>

          {/* 파일 정보 (카드 대신 타이포그래피 한 줄) */}
          {selectedFile && (
            <View style={styles.fileRow}>
              <View style={styles.fileDot} />
              <Text style={styles.fileName} numberOfLines={1}>
                {selectedFile.name || '파일명 없음'}
              </Text>
              <Text style={styles.fileSize}>
                {formatSize(selectedFile.size)}
              </Text>
            </View>
          )}

          {/* 분석 시작 / 업로드 진행 */}
          <TouchableOpacity
            activeOpacity={0.85}
            onPress={handleUpload}
            disabled={!canSubmit}
            style={[styles.primaryBtn, !canSubmit && styles.primaryDisabled]}>
            {isUploading ? (
              <UploadProgress progress={uploadProgress} />
            ) : (
              <View style={styles.primaryInner}>
                <Text
                  style={[
                    styles.primaryText,
                    !selectedFile && styles.primaryTextDisabled,
                  ]}>
                  분석 시작
                </Text>
                <Text
                  style={[
                    styles.primaryArrow,
                    !selectedFile && styles.primaryTextDisabled,
                  ]}>
                  →
                </Text>
              </View>
            )}
          </TouchableOpacity>
        </Animated.View>
      </SafeAreaView>
    </View>
  );
}

// ──────────────────────────────────────────────────────────────
// Background — 중앙에서 부드럽게 번지는 radial glow
// ──────────────────────────────────────────────────────────────
function BackgroundGlow() {
  return (
    <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
      <Defs>
        <RadialGradient id="uploadBg" cx="50%" cy="55%" r="70%">
          <Stop offset="0%" stopColor="#1C1530" stopOpacity="1" />
          <Stop offset="50%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#uploadBg)" />
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// Equalizer — 32 band live spectrum analyzer
// ──────────────────────────────────────────────────────────────
interface EqualizerProps {
  phase: number;
  active: boolean; // 업로드 중이면 bars 가 더 격렬하게 움직임
}

function Equalizer({phase, active}: EqualizerProps) {
  const baseIntensity = active ? 1.0 : 0.58;
  // 업로드 중엔 phase 속도도 체감상 빠르게 — freq 계수를 살짝 부스트
  const speedBoost = active ? 1.35 : 1.0;

  // bar 하나씩 계산
  const bars: Array<{
    x: number;
    h: number;
    color: string;
  }> = [];

  for (let i = 0; i < NUM_BARS; i++) {
    const t = i / (NUM_BARS - 1); // 0(low) ~ 1(high)

    // 저역은 느리게 (0.8), 고역은 빠르게 (4.0)
    const freq = (0.8 + t * 3.2) * speedBoost;
    const bump =
      Math.sin(phase * freq + i * 0.4) * 0.55 +
      Math.sin(phase * (freq * 0.5) + i * 0.25) * 0.35 +
      Math.sin(phase * (freq * 1.7) - i * 0.15) * 0.25;

    // 좌우 끝이 살짝 죽도록 envelope (V자가 아니라 cos-bell)
    const env = 0.55 + 0.45 * Math.cos((t - 0.5) * Math.PI * 1.6);
    const norm = (bump * 0.5 + 0.55) * env * baseIntensity;
    const h = Math.max(6, Math.min(FLOOR_Y - 6, norm * VIZ_H * 0.62));

    const x = i * (BAR_W + BAR_SPACING);

    // 대역별 색상: 저역(오렌지) → 중역(퍼플) → 고역(시안)
    let color: string;
    if (t < 0.33) {
      color = '#FF8A5B';
    } else if (t < 0.66) {
      color = '#9E7BE0';
    } else {
      color = '#3DC8FF';
    }

    bars.push({x, h, color});
  }

  // EQ curve — 바 상단을 부드럽게 잇는 path
  let curveD = '';
  bars.forEach((bar, i) => {
    const cxBar = bar.x + BAR_W / 2;
    const cyBar = FLOOR_Y - bar.h;
    curveD +=
      (i === 0 ? 'M ' : 'L ') + cxBar.toFixed(1) + ' ' + cyBar.toFixed(1) + ' ';
  });

  return (
    <Svg width={VIZ_W} height={VIZ_H}>
      <Defs>
        <SvgLinearGradient id="gridFade" x1="0%" y1="0%" x2="100%" y2="0%">
          <Stop offset="0%" stopColor="#FFFFFF" stopOpacity="0" />
          <Stop offset="50%" stopColor="#FFFFFF" stopOpacity="0.15" />
          <Stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
        </SvgLinearGradient>
      </Defs>

      {/* 바닥 기준선 */}
      <Rect
        x={0}
        y={FLOOR_Y}
        width={VIZ_W}
        height={0.5}
        fill="url(#gridFade)"
      />

      {/* ── 1. Reflection (아래로 미러링) */}
      <G>
        {bars.map((bar, i) => (
          <Rect
            key={`r-${i}`}
            x={bar.x}
            y={FLOOR_Y + 2}
            width={BAR_W}
            height={Math.min(bar.h * 0.45, VIZ_H - FLOOR_Y - 4)}
            rx={BAR_W / 2}
            ry={BAR_W / 2}
            fill={bar.color}
            opacity={0.16}
          />
        ))}
      </G>

      {/* ── 2. Main bars */}
      {bars.map((bar, i) => (
        <Rect
          key={`b-${i}`}
          x={bar.x}
          y={FLOOR_Y - bar.h}
          width={BAR_W}
          height={bar.h}
          rx={BAR_W / 2}
          ry={BAR_W / 2}
          fill={bar.color}
          opacity={0.82}
        />
      ))}

      {/* ── 3. Peak dots (각 바 상단) */}
      {bars.map((bar, i) => (
        <Rect
          key={`p-${i}`}
          x={bar.x}
          y={FLOOR_Y - bar.h - 4}
          width={BAR_W}
          height={2}
          rx={1}
          ry={1}
          fill="#FFFFFF"
          opacity={0.88}
        />
      ))}

      {/* ── 4. EQ curve overlay */}
      <Path
        d={curveD}
        stroke="#FFFFFF"
        strokeWidth={1.1}
        strokeLinejoin="round"
        strokeLinecap="round"
        fill="none"
        opacity={0.55}
      />
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// 업로드 진행 바 (primary 버튼 내부에서 좌→우로 채워짐)
// ──────────────────────────────────────────────────────────────
interface UploadProgressProps {
  progress: number;
}

function UploadProgress({progress}: UploadProgressProps) {
  const pct = Math.round(progress * 100);
  return (
    <View style={styles.progressWrap}>
      <View style={[styles.progressFill, {width: `${pct}%`}]} />
      <View style={styles.progressContent}>
        <Text style={styles.progressText}>분석 중</Text>
        <Text style={styles.progressPct}>{pct}%</Text>
      </View>
    </View>
  );
}

// ──────────────────────────────────────────────────────────────
// Icon
// ──────────────────────────────────────────────────────────────
function PlusIcon() {
  return (
    <Svg width={16} height={16} viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 5v14 M5 12h14"
        stroke="#F5F5F7"
        strokeWidth={1.5}
        strokeLinecap="round"
      />
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// Styles
// ──────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#000000',
  },
  safe: {
    flex: 1,
    justifyContent: 'space-between',
  },
  header: {
    alignItems: 'center',
    paddingTop: 28,
    paddingHorizontal: 24,
  },
  eyebrow: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.42)',
    letterSpacing: 6,
    fontWeight: '600',
    marginBottom: 18,
  },
  hero: {
    fontSize: 28,
    fontWeight: '200',
    color: '#F5F5F7',
    letterSpacing: -0.5,
    lineHeight: 38,
    textAlign: 'center',
  },
  vizContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  freqLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: VIZ_W,
    marginTop: 8,
    paddingHorizontal: 4,
  },
  freqLabel: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.4)',
    letterSpacing: 1.5,
    fontWeight: '500',
  },
  controls: {
    paddingHorizontal: 24,
    paddingBottom: 28,
    gap: 14,
  },
  pickBtn: {
    height: 72,
    borderRadius: 18,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.16)',
    borderStyle: 'dashed',
    backgroundColor: 'rgba(255,255,255,0.03)',
    overflow: 'hidden',
  },
  pickBtnInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  pickText: {
    fontSize: 15,
    color: '#F5F5F7',
    letterSpacing: 1,
    fontWeight: '500',
  },
  fileRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 4,
  },
  fileDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#FF8A5B',
    shadowColor: '#FF8A5B',
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 5,
    elevation: 5,
  },
  fileName: {
    flex: 1,
    fontSize: 13,
    color: '#F5F5F7',
    fontWeight: '500',
  },
  fileSize: {
    fontSize: 12,
    color: 'rgba(245,245,247,0.5)',
    letterSpacing: 0.4,
  },
  primaryBtn: {
    height: 62,
    borderRadius: 100,
    overflow: 'hidden',
    backgroundColor: '#FF8A5B',
    shadowColor: '#FF8A5B',
    shadowOffset: {width: 0, height: 8},
    shadowOpacity: 0.45,
    shadowRadius: 16,
    elevation: 8,
  },
  primaryDisabled: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    shadowOpacity: 0,
    elevation: 0,
  },
  primaryInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  primaryText: {
    fontSize: 16,
    color: '#0A0A12',
    fontWeight: '600',
    letterSpacing: 1.2,
  },
  primaryArrow: {
    fontSize: 18,
    color: '#0A0A12',
    fontWeight: '300',
  },
  primaryTextDisabled: {
    color: 'rgba(245,245,247,0.35)',
  },
  progressWrap: {
    flex: 1,
    overflow: 'hidden',
    borderRadius: 100,
    justifyContent: 'center',
  },
  progressFill: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    backgroundColor: 'rgba(10,10,18,0.35)',
  },
  progressContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  progressText: {
    fontSize: 14,
    color: '#0A0A12',
    letterSpacing: 1,
    fontWeight: '600',
  },
  progressPct: {
    fontSize: 14,
    color: '#0A0A12',
    fontWeight: '700',
  },
  dim: {
    opacity: 0.4,
  },
});
