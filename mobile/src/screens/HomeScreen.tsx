// screens/HomeScreen.tsx — Homecinema 진입 게이트 (시네마틱 사운드웨이브 버전)
//
// 디자인 원칙
//   · 극흑 배경(#000) + 중심부 극미한 퍼플-워밍 radial glow
//   · 28 레이어의 hairline 사운드웨이브를 Gaussian envelope 로 중앙에 집중
//   · requestAnimationFrame 으로 다중 sin 주파수를 위상 진행시켜 "호흡하는" 움직임
//   · Chromatic aberration (오렌지/시안) — 아날로그 영사기 느낌
//   · 두 개의 glass pill 버튼에 LED 닷 액센트
//
// 외부 의존성 없이 react-native-svg 만 사용.
import React, {useEffect, useMemo, useRef, useState} from 'react';
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
} from 'react-native';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  RadialGradient,
  Stop,
  Rect,
  Circle,
  Path,
  G,
} from 'react-native-svg';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList} from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

const {width: SCREEN_W} = Dimensions.get('window');

// ── Waveform 파라미터 ────────────────────────────────────────
const WAVE_W = SCREEN_W;
const WAVE_H = 300;
const NUM_LINES = 28; // 층수 — 너무 많으면 iOS SVG 재페인트가 부담된다
const NUM_POINTS = 60; // x 샘플 수
const ENV_SIGMA = WAVE_W * 0.17; // 중심 집중도 (작을수록 뾰족)
const AMP = 52; // 최대 진폭(px)
const CENTER_GHOST_COUNT = 8; // chromatic aberration 을 입힐 중앙 라인 개수

// ──────────────────────────────────────────────────────────────
// 메인 스크린
// ──────────────────────────────────────────────────────────────
export default function HomeScreen({navigation}: Props) {
  // phase 는 애니메이션 위상 (초 단위 누적)
  const [phase, setPhase] = useState(0);
  // 진입 페이드인 (waveform / text 가 천천히 깨어나는 느낌)
  const fade = useRef(new Animated.Value(0)).current;

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

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <BackgroundGlow />

      <SafeAreaView style={styles.safe}>
        {/* ── Header ── */}
        <Animated.View style={[styles.header, {opacity: fade}]}>
          <Text style={styles.eyebrow}>HOMECINEMA</Text>
          <Text style={styles.hero}>공간의 파형을{'\n'}깨우다</Text>
        </Animated.View>

        {/* ── Sound Wave ── */}
        <Animated.View style={[styles.waveContainer, {opacity: fade}]}>
          <SoundWave phase={phase} />
        </Animated.View>

        {/* ── Actions ── */}
        <Animated.View style={[styles.actions, {opacity: fade}]}>
          <ActionButton
            label="EQ 설정"
            accent="#FF8A5B"
            Icon={EqIcon}
            onPress={() => navigation.navigate('Upload')}
          />
          <ActionButton
            label="스피커 배치"
            accent="#9E7BE0"
            Icon={SpeakerIcon}
            onPress={() => navigation.navigate('SpeakerSize')}
          />
        </Animated.View>
      </SafeAreaView>
    </View>
  );
}

// ──────────────────────────────────────────────────────────────
// Background — 중심에서 아주 미묘하게 번지는 radial glow
// ──────────────────────────────────────────────────────────────
function BackgroundGlow() {
  return (
    <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
      <Defs>
        <RadialGradient id="bgGlow" cx="50%" cy="52%" r="60%">
          <Stop offset="0%" stopColor="#1C1530" stopOpacity="1" />
          <Stop offset="45%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
        <RadialGradient id="bgWarm" cx="50%" cy="52%" r="35%">
          <Stop offset="0%" stopColor="#FF7849" stopOpacity="0.05" />
          <Stop offset="100%" stopColor="#FF7849" stopOpacity="0" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#bgGlow)" />
      <Rect width="100%" height="100%" fill="url(#bgWarm)" />
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// SoundWave — NUM_LINES 개의 hairline 경로를 동적으로 계산
// ──────────────────────────────────────────────────────────────
interface SoundWaveProps {
  phase: number;
}

function SoundWave({phase}: SoundWaveProps) {
  const cx = WAVE_W / 2;
  const cy = WAVE_H / 2;

  // 라인 인덱스 배열 — 렌더마다 재생성 막기 위해 memo
  const lines = useMemo(() => Array.from({length: NUM_LINES}, (_, i) => i), []);

  // 라인 하나의 SVG path 생성
  // yOffset: chromatic aberration 용 ghost 복제시 y 방향 미세 shift
  const makePath = (lineIdx: number, yOffset: number = 0): string => {
    const layerPhase = lineIdx * 0.11; // 층마다 위상 어긋남 → 쌓인 파도
    const normalized = (lineIdx - NUM_LINES / 2) / (NUM_LINES / 2); // -1 ~ 1
    const layerAmp = 1 - Math.abs(normalized) * 0.22; // 가운데가 살짝 더 진폭 큼
    const layerY = normalized * 1.3; // 레이어별 수직 오프셋 (촘촘하게 쌓음)

    let d = '';
    for (let p = 0; p <= NUM_POINTS; p++) {
      const x = (WAVE_W / NUM_POINTS) * p;
      const dx = x - cx;
      // Gaussian envelope — 중앙에서 최대, 양끝으로 갈수록 0
      const env = Math.exp(-(dx * dx) / (2 * ENV_SIGMA * ENV_SIGMA));
      // 3 개의 서로 다른 주파수/방향 sin 을 믹스해서 "자연스러운" 움직임
      const wave =
        Math.sin(x * 0.018 - phase * 1.15 + layerPhase) * 0.55 +
        Math.sin(x * 0.033 + phase * 0.85 + layerPhase * 2.1) * 0.28 +
        Math.sin(x * 0.008 - phase * 0.45 + layerPhase * 0.6) * 0.42;
      const y = cy + layerY + env * AMP * wave * layerAmp + yOffset;
      d += (p === 0 ? 'M ' : 'L ') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
    }
    return d;
  };

  // 중앙 라인은 진하게, 바깥 라인은 흐리게
  const opacityFor = (i: number) => {
    const n = Math.abs(i - NUM_LINES / 2) / (NUM_LINES / 2);
    return 0.3 + (1 - n) * 0.55;
  };

  // chromatic aberration 대상 (중앙 8 개 라인)
  const isGhostLayer = (i: number) =>
    Math.abs(i - NUM_LINES / 2) < CENTER_GHOST_COUNT / 2;

  return (
    <Svg width={WAVE_W} height={WAVE_H}>
      <Defs>
        {/* 양끝으로 갈수록 투명해지는 가로 마스크 — "사라지는" 느낌 강화 */}
        <SvgLinearGradient id="waveFade" x1="0%" y1="0%" x2="100%" y2="0%">
          <Stop offset="0%" stopColor="#FFFFFF" stopOpacity="0" />
          <Stop offset="20%" stopColor="#FFFFFF" stopOpacity="0.6" />
          <Stop offset="50%" stopColor="#FFFFFF" stopOpacity="1" />
          <Stop offset="80%" stopColor="#FFFFFF" stopOpacity="0.6" />
          <Stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
        </SvgLinearGradient>
      </Defs>

      {/* ── 1. Warm orange ghost (아래로 shift) — 중앙 라인만 */}
      <G opacity="0.55">
        {lines.filter(isGhostLayer).map(i => (
          <Path
            key={`og-${i}`}
            d={makePath(i, 1.8)}
            stroke="#FF6A3D"
            strokeWidth={0.6}
            strokeLinecap="round"
            fill="none"
          />
        ))}
      </G>

      {/* ── 2. Cool cyan ghost (위로 shift) — 중앙 라인만 */}
      <G opacity="0.55">
        {lines.filter(isGhostLayer).map(i => (
          <Path
            key={`cg-${i}`}
            d={makePath(i, -1.8)}
            stroke="#3DC8FF"
            strokeWidth={0.6}
            strokeLinecap="round"
            fill="none"
          />
        ))}
      </G>

      {/* ── 3. 메인 흰색 레이어 (전체) */}
      {lines.map(i => (
        <Path
          key={`w-${i}`}
          d={makePath(i)}
          stroke="url(#waveFade)"
          strokeWidth={0.6}
          strokeLinecap="round"
          fill="none"
          opacity={opacityFor(i)}
        />
      ))}

      {/* ── 4. 중심 스파크 (스피커의 "소리가 터지는" 점) */}
      <Circle cx={cx} cy={cy} r={2} fill="#FFFFFF" opacity={0.85} />
      <Circle cx={cx} cy={cy} r={6} fill="#FFFFFF" opacity={0.1} />
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// 라인 아트 아이콘 — 이모지 대신 일관된 톤으로
// ──────────────────────────────────────────────────────────────
function EqIcon({color}: {color: string}) {
  return (
    <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
      <Path
        d="M5 3v18 M12 3v18 M19 3v18"
        stroke={color}
        strokeWidth={1.3}
        strokeLinecap="round"
        opacity={0.5}
      />
      <Circle cx={5} cy={8} r={2.2} fill="#000" stroke={color} strokeWidth={1.4} />
      <Circle cx={12} cy={14} r={2.2} fill="#000" stroke={color} strokeWidth={1.4} />
      <Circle cx={19} cy={10} r={2.2} fill="#000" stroke={color} strokeWidth={1.4} />
    </Svg>
  );
}

function SpeakerIcon({color}: {color: string}) {
  return (
    <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
      <Path
        d="M6 3.5h12a1.5 1.5 0 0 1 1.5 1.5v14a1.5 1.5 0 0 1-1.5 1.5H6a1.5 1.5 0 0 1-1.5-1.5V5A1.5 1.5 0 0 1 6 3.5z"
        stroke={color}
        strokeWidth={1.3}
      />
      <Circle cx={12} cy={8.5} r={1.6} stroke={color} strokeWidth={1.3} />
      <Circle cx={12} cy={15} r={3.4} stroke={color} strokeWidth={1.3} />
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// ActionButton — 글래스 필(pill) + LED 닷 액센트
// ──────────────────────────────────────────────────────────────
interface ActionButtonProps {
  label: string;
  accent: string;
  Icon: React.FC<{color: string}>;
  onPress: () => void;
}

function ActionButton({label, accent, Icon, onPress}: ActionButtonProps) {
  return (
    <TouchableOpacity
      activeOpacity={0.7}
      onPress={onPress}
      style={styles.btn}>
      <View style={styles.btnInner}>
        <View
          style={[
            styles.btnDot,
            {
              backgroundColor: accent,
              shadowColor: accent,
            },
          ]}
        />
        <Icon color="#F5F5F7" />
        <Text style={styles.btnText}>{label}</Text>
      </View>
    </TouchableOpacity>
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
  },
  header: {
    alignItems: 'center',
    paddingTop: 32,
    paddingHorizontal: 24,
  },
  eyebrow: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.42)',
    letterSpacing: 6,
    fontWeight: '600',
    marginBottom: 22,
  },
  hero: {
    fontSize: 36,
    fontWeight: '200',
    color: '#F5F5F7',
    letterSpacing: -0.8,
    lineHeight: 46,
    textAlign: 'center',
  },
  waveContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actions: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingBottom: 28,
    gap: 12,
  },
  btn: {
    flex: 1,
    height: 62,
    borderRadius: 100,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.14)',
    backgroundColor: 'rgba(255,255,255,0.03)',
    // 아주 미세한 외부 shadow 로 "떠 있는" 느낌
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 6},
    shadowOpacity: 0.4,
    shadowRadius: 14,
    elevation: 6,
  },
  btnInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingHorizontal: 16,
  },
  btnDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 2,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 5,
    elevation: 6,
  },
  btnText: {
    fontSize: 15,
    color: '#F5F5F7',
    letterSpacing: 1.2,
    fontWeight: '500',
  },
});
