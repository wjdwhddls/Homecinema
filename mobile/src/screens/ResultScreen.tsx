// screens/ResultScreen.tsx — 분석 중 시네마틱 UI
//
// 설계 원칙
//   · 메인/업로드 화면과 동일한 다크 + 3-색 액센트 (오렌지/퍼플/시안)
//   · 분석이 "호흡"하는 느낌 — waveform 을 메인보다 느리게 펄싱 + 중심 flicker
//   · 좌→우 scan line 으로 "스캐닝하는 시선" 표현
//   · 4-stage pipeline dot (장면 · 감정 · EQ · 완성) — 백엔드 progress 실시간 반영
//   · hairline progress bar + 부드러운 숫자 카운터 (Animated interpolation)
//
// 상태 매핑 (pipeline_runner _STAGE_MARKERS)
//   0.00 ~ 0.10 → 장면 분할 · window slicing
//   0.10 ~ 0.45 → 3-seed ensemble 감정 추론
//   0.45 ~ 0.75 → Layer 1 EQ 적용 (대사 보호 포함)
//   0.75 ~ 1.00 → Layer 2 FX (VAD-guided reverb bypass) + 머지
//
// 외부 의존성 없이 react-native-svg + Animated.
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
import {RootStackParamList, JobStatus} from '../types';
import {useJobStatus} from '../hooks/useJobStatus';

type Props = NativeStackScreenProps<RootStackParamList, 'Result'>;

// ⚠️ 반드시 컴포넌트 외부 상수로 유지할 것.
//    컴포넌트 안에서 { ... } 리터럴로 넘기면 매 렌더마다 새 참조가 되어
//    useJobStatus 내부의 poll useCallback + polling useEffect 가 재실행되고
//    setInterval 이 즉시 cleanup/재생성되며 HTTP 요청이 폭주한다.
//    (rAF 루프로 매 프레임 re-render 되는 이 화면에서는 치명적)
const JOB_STATUS_OPTIONS = {
  pollInterval: 2000,
  maxRetries: 300,
  stopOnStatuses: ['completed', 'failed'] as JobStatus[],
};

const {width: SCREEN_W} = Dimensions.get('window');

// ── Waveform (analyzing mode — 메인보다 느리고 chromatic aberration 증폭) ──
const WAVE_W = SCREEN_W;
const WAVE_H = 240;
const NUM_LINES = 26;
const NUM_POINTS = 56;
const ENV_SIGMA = WAVE_W * 0.14; // 중심 더 집중
const AMP = 44;
const CENTER_GHOST_COUNT = 10;
const SCAN_PERIOD_SEC = 4.0;

// ── 4-stage pipeline (threshold 이하일 동안 해당 stage 가 current) ──
interface StageDef {
  label: string;
  threshold: number;
}
const STAGES: StageDef[] = [
  {label: '장면', threshold: 0.10},
  {label: '감정', threshold: 0.45},
  {label: 'EQ', threshold: 0.75},
  {label: '완성', threshold: 1.00},
];

function getCurrentStageIdx(progress: number): number {
  for (let i = 0; i < STAGES.length; i++) {
    if (progress < STAGES[i].threshold) return i;
  }
  return STAGES.length; // 모두 완료
}

function getHeroText(progress: number, status: string | null): string {
  if (status === 'failed') return '분석에 실패했습니다';
  if (status === 'completed' || progress >= 1.0) return '분석을 마쳤습니다';
  const idx = getCurrentStageIdx(progress);
  const phrases = [
    '영상을 분해하고\n있습니다',
    '장면의 감정을\n읽고 있습니다',
    '이퀄라이저를\n조율하고 있습니다',
    '최종 영상을\n만들고 있습니다',
  ];
  return phrases[Math.min(idx, phrases.length - 1)];
}

// ──────────────────────────────────────────────────────────────
// 메인 스크린
// ──────────────────────────────────────────────────────────────
export default function ResultScreen({route, navigation}: Props) {
  const {jobId} = route.params;
  const {status, progress, errorMessage} = useJobStatus(
    jobId,
    JOB_STATUS_OPTIONS,
  );

  // waveform / scan line 용 phase (초 단위 누적)
  const [phase, setPhase] = useState(0);
  // 진입 페이드인
  const fade = useRef(new Animated.Value(0)).current;
  // 백엔드 progress 는 0.05/0.10/0.45/0.75/1.0 로 거칠게 점프 → 부드럽게 보간
  const smoothProgress = useRef(new Animated.Value(0)).current;
  const [displayedPct, setDisplayedPct] = useState(0);

  // ── 페이드인 + rAF 루프
  useEffect(() => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 1200,
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

  // ── 백엔드 progress → smooth 보간
  useEffect(() => {
    Animated.timing(smoothProgress, {
      toValue: progress,
      duration: 1000,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  }, [progress, smoothProgress]);

  // ── 숫자 카운터 — animated value 를 listen
  useEffect(() => {
    const id = smoothProgress.addListener(({value}) => {
      setDisplayedPct(Math.round(value * 100));
    });
    return () => smoothProgress.removeListener(id);
  }, [smoothProgress]);

  // ── completed 시 Playback 으로 전환
  useEffect(() => {
    if (status === 'completed') {
      const t = setTimeout(() => {
        navigation.replace('Playback', {jobId});
      }, 1200);
      return () => clearTimeout(t);
    }
  }, [status, jobId, navigation]);

  const isFailed = status === 'failed';
  const currentStageIdx = getCurrentStageIdx(progress);
  const heroText = getHeroText(progress, status);

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <BackgroundGlow failed={isFailed} />

      <SafeAreaView style={styles.safe}>
        {/* ── Header */}
        <Animated.View style={[styles.header, {opacity: fade}]}>
          <Text style={[styles.eyebrow, isFailed && styles.eyebrowFailed]}>
            {isFailed ? 'F A I L E D' : 'A N A L Y Z I N G'}
          </Text>
          <Text style={styles.hero}>{heroText}</Text>
          {isFailed && errorMessage ? (
            <Text style={styles.errorText} numberOfLines={4}>
              {errorMessage}
            </Text>
          ) : null}
        </Animated.View>

        {/* ── Analyzing waveform */}
        <Animated.View style={[styles.vizContainer, {opacity: fade}]}>
          <AnalyzingWave phase={phase} frozen={isFailed} />
        </Animated.View>

        {/* ── Bottom: stage pipeline + progress + action */}
        <Animated.View style={[styles.bottom, {opacity: fade}]}>
          {!isFailed && <StagePipeline currentIdx={currentStageIdx} />}
          {!isFailed && (
            <SmoothProgress
              animatedValue={smoothProgress}
              pct={displayedPct}
            />
          )}

          <TouchableOpacity
            activeOpacity={0.7}
            onPress={() => navigation.replace('Home', undefined)}
            style={[styles.actionBtn, isFailed && styles.actionBtnPrimary]}>
            <Text
              style={[
                styles.actionText,
                isFailed && styles.actionTextPrimary,
              ]}>
              {isFailed ? '처음으로' : '취소'}
            </Text>
          </TouchableOpacity>
        </Animated.View>
      </SafeAreaView>
    </View>
  );
}

// ──────────────────────────────────────────────────────────────
// Background — 실패시 붉은 기운 살짝
// ──────────────────────────────────────────────────────────────
const BackgroundGlow = React.memo(function BackgroundGlow({
  failed,
}: {
  failed: boolean;
}) {
  const core = failed ? '#2A1515' : '#1C1530';
  return (
    <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
      <Defs>
        <RadialGradient id="anBg" cx="50%" cy="50%" r="65%">
          <Stop offset="0%" stopColor={core} stopOpacity="1" />
          <Stop offset="50%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#anBg)" />
    </Svg>
  );
});

// ──────────────────────────────────────────────────────────────
// AnalyzingWave — 메인 waveform 의 "분석 모드" 파생
// ──────────────────────────────────────────────────────────────
interface WaveProps {
  phase: number;
  frozen: boolean; // 실패 시 파형 동결 + 감쇠
}

function AnalyzingWave({phase, frozen}: WaveProps) {
  const cx = WAVE_W / 2;
  const cy = WAVE_H / 2;
  const pEff = frozen ? 0 : phase * 0.65; // 메인보다 느리게 — "집중" 감성

  const lines = useMemo(
    () => Array.from({length: NUM_LINES}, (_, i) => i),
    [],
  );

  const makePath = (lineIdx: number, yOffset: number = 0): string => {
    const layerPhase = lineIdx * 0.11;
    const normalized = (lineIdx - NUM_LINES / 2) / (NUM_LINES / 2);
    const layerAmp = 1 - Math.abs(normalized) * 0.25;
    const layerY = normalized * 1.2;
    let d = '';
    for (let p = 0; p <= NUM_POINTS; p++) {
      const x = (WAVE_W / NUM_POINTS) * p;
      const dx = x - cx;
      const env = Math.exp(-(dx * dx) / (2 * ENV_SIGMA * ENV_SIGMA));
      const wave =
        Math.sin(x * 0.02 - pEff * 1.15 + layerPhase) * 0.55 +
        Math.sin(x * 0.037 + pEff * 0.85 + layerPhase * 2.1) * 0.3 +
        Math.sin(x * 0.009 - pEff * 0.45 + layerPhase * 0.6) * 0.42;
      const y = cy + layerY + env * AMP * wave * layerAmp + yOffset;
      d += (p === 0 ? 'M ' : 'L ') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
    }
    return d;
  };

  const opacityFor = (i: number) => {
    const n = Math.abs(i - NUM_LINES / 2) / (NUM_LINES / 2);
    return 0.3 + (1 - n) * 0.55;
  };
  const isGhost = (i: number) =>
    Math.abs(i - NUM_LINES / 2) < CENTER_GHOST_COUNT / 2;

  // 중심 스파크 flicker — 두 sin 을 다른 주파수로 곱해 "불규칙"한 깜빡임
  const flicker = frozen
    ? 0.3
    : (0.55 + 0.45 * (0.5 + 0.5 * Math.sin(phase * 14))) *
      (0.85 + 0.15 * Math.sin(phase * 31.7));

  // scan line x — 주기적으로 좌→우
  const scanNorm = frozen ? -1 : (phase % SCAN_PERIOD_SEC) / SCAN_PERIOD_SEC;
  const scanX = scanNorm * WAVE_W;

  return (
    <Svg width={WAVE_W} height={WAVE_H}>
      <Defs>
        <SvgLinearGradient id="anWaveFade" x1="0%" y1="0%" x2="100%" y2="0%">
          <Stop offset="0%" stopColor="#FFFFFF" stopOpacity="0" />
          <Stop offset="25%" stopColor="#FFFFFF" stopOpacity="0.55" />
          <Stop offset="50%" stopColor="#FFFFFF" stopOpacity="1" />
          <Stop offset="75%" stopColor="#FFFFFF" stopOpacity="0.55" />
          <Stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
        </SvgLinearGradient>
        <SvgLinearGradient id="anScan" x1="50%" y1="0%" x2="50%" y2="100%">
          <Stop offset="0%" stopColor="#9E7BE0" stopOpacity="0" />
          <Stop offset="50%" stopColor="#9E7BE0" stopOpacity="0.55" />
          <Stop offset="100%" stopColor="#9E7BE0" stopOpacity="0" />
        </SvgLinearGradient>
      </Defs>

      {/* warm orange ghost — chromatic aberration 강화 (±2.5px) */}
      <G opacity="0.6">
        {lines.filter(isGhost).map(i => (
          <Path
            key={`og-${i}`}
            d={makePath(i, 2.5)}
            stroke="#FF6A3D"
            strokeWidth={0.6}
            fill="none"
          />
        ))}
      </G>
      {/* cool cyan ghost */}
      <G opacity="0.6">
        {lines.filter(isGhost).map(i => (
          <Path
            key={`cg-${i}`}
            d={makePath(i, -2.5)}
            stroke="#3DC8FF"
            strokeWidth={0.6}
            fill="none"
          />
        ))}
      </G>
      {/* 메인 흰색 */}
      {lines.map(i => (
        <Path
          key={`w-${i}`}
          d={makePath(i)}
          stroke="url(#anWaveFade)"
          strokeWidth={0.6}
          fill="none"
          opacity={opacityFor(i) * (frozen ? 0.4 : 1)}
        />
      ))}

      {/* 중심 스파크 (flicker) */}
      <Circle cx={cx} cy={cy} r={2.2} fill="#FFFFFF" opacity={flicker} />
      <Circle cx={cx} cy={cy} r={7} fill="#FFFFFF" opacity={0.12 * flicker} />

      {/* Scan line */}
      {!frozen && (
        <Rect
          x={scanX - 1}
          y={0}
          width={2}
          height={WAVE_H}
          fill="url(#anScan)"
        />
      )}
    </Svg>
  );
}

// ──────────────────────────────────────────────────────────────
// StagePipeline — 4 dot, current 는 pulse ring
// ──────────────────────────────────────────────────────────────
interface StagePipelineProps {
  currentIdx: number;
}

// rAF 루프가 부모(ResultScreen)를 매 프레임 re-render 시키므로
// currentIdx 가 그대로면 통째로 스킵되도록 memo.
const StagePipeline = React.memo(function StagePipeline({
  currentIdx,
}: StagePipelineProps) {
  const pulse = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          toValue: 1,
          duration: 900,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: false,
        }),
        Animated.timing(pulse, {
          toValue: 0,
          duration: 900,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: false,
        }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [pulse]);

  return (
    <View style={styles.pipelineRow}>
      {STAGES.map((s, i) => {
        const state: StageState =
          i < currentIdx ? 'done' : i === currentIdx ? 'current' : 'pending';
        return (
          <StageDot key={s.label} label={s.label} state={state} pulse={pulse} />
        );
      })}
    </View>
  );
});

type StageState = 'done' | 'current' | 'pending';

interface StageDotProps {
  label: string;
  state: StageState;
  pulse: Animated.Value;
}

function StageDot({label, state, pulse}: StageDotProps) {
  const color =
    state === 'done'
      ? '#FF8A5B'
      : state === 'current'
      ? '#9E7BE0'
      : '#555A68';
  const ringScale = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: [1, 2.2],
  });
  const ringOpacity = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: [0.55, 0],
  });

  return (
    <View style={styles.dotCol}>
      <View style={styles.dotWrap}>
        {state === 'current' && (
          <Animated.View
            style={[
              styles.dotRing,
              {
                borderColor: color,
                transform: [{scale: ringScale}],
                opacity: ringOpacity,
              },
            ]}
          />
        )}
        <View
          style={[
            styles.dot,
            {
              backgroundColor: color,
              shadowColor: color,
              shadowOpacity: state === 'pending' ? 0 : 0.9,
            },
          ]}
        />
      </View>
      <Text
        style={[
          styles.dotLabel,
          state === 'pending' && styles.dotLabelPending,
          state === 'current' && styles.dotLabelCurrent,
        ]}>
        {label}
      </Text>
    </View>
  );
}

// ──────────────────────────────────────────────────────────────
// Smooth progress — hairline bar + 부드럽게 올라가는 %
// ──────────────────────────────────────────────────────────────
interface SmoothProgressProps {
  animatedValue: Animated.Value;
  pct: number;
}

// pct 가 바뀌지 않는 한 rAF 리렌더에 합류하지 않도록 memo.
// animatedValue 의 변경은 Animated 내부에서 native bridge 를 통해 width 만 갱신.
const SmoothProgress = React.memo(function SmoothProgress({
  animatedValue,
  pct,
}: SmoothProgressProps) {
  const width = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });
  return (
    <View style={styles.progressSection}>
      <View style={styles.progressTrack}>
        <Animated.View style={[styles.progressFill, {width}]} />
      </View>
      <Text style={styles.progressPct}>{pct}%</Text>
    </View>
  );
});

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
  eyebrowFailed: {
    color: 'rgba(228,88,88,0.85)',
  },
  hero: {
    fontSize: 28,
    fontWeight: '200',
    color: '#F5F5F7',
    letterSpacing: -0.5,
    lineHeight: 38,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 13,
    color: 'rgba(228,88,88,0.85)',
    marginTop: 18,
    textAlign: 'center',
    paddingHorizontal: 16,
    fontWeight: '400',
    lineHeight: 20,
  },
  vizContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bottom: {
    paddingHorizontal: 24,
    paddingBottom: 28,
    gap: 22,
  },
  // ── Pipeline ──
  pipelineRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
  },
  dotCol: {
    alignItems: 'center',
  },
  dotWrap: {
    width: 18,
    height: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dotRing: {
    position: 'absolute',
    width: 14,
    height: 14,
    borderRadius: 7,
    borderWidth: 1.5,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    shadowOffset: {width: 0, height: 0},
    shadowRadius: 6,
    elevation: 4,
  },
  dotLabel: {
    fontSize: 11,
    color: '#F5F5F7',
    marginTop: 6,
    letterSpacing: 1.2,
    fontWeight: '500',
  },
  dotLabelPending: {
    color: 'rgba(245,245,247,0.38)',
    fontWeight: '400',
  },
  dotLabelCurrent: {
    color: '#9E7BE0',
    fontWeight: '600',
  },
  // ── Progress ──
  progressSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  progressTrack: {
    flex: 1,
    height: 3,
    backgroundColor: 'rgba(245,245,247,0.08)',
    borderRadius: 1.5,
    overflow: 'hidden',
  },
  progressFill: {
    // Animated.View 의 width 가 매 프레임 변하므로 shadow 를 얹으면
    // native 쪽에서 shadow 를 매번 재계산 → "shadow set but cannot calculate
    // efficiently" advice 및 프레임 드롭 유발. 배경 색만 유지한다.
    height: '100%',
    backgroundColor: '#FF8A5B',
  },
  progressPct: {
    fontSize: 12,
    color: '#F5F5F7',
    fontWeight: '600',
    letterSpacing: 0.5,
    minWidth: 42,
    textAlign: 'right',
  },
  // ── Action button ──
  actionBtn: {
    height: 52,
    borderRadius: 100,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.14)',
    backgroundColor: 'rgba(255,255,255,0.02)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnPrimary: {
    backgroundColor: '#FF8A5B',
    borderColor: 'transparent',
    shadowColor: '#FF8A5B',
    shadowOffset: {width: 0, height: 6},
    shadowOpacity: 0.5,
    shadowRadius: 14,
    elevation: 6,
  },
  actionText: {
    fontSize: 14,
    color: '#F5F5F7',
    letterSpacing: 1.2,
    fontWeight: '500',
  },
  actionTextPrimary: {
    color: '#0A0A12',
    fontWeight: '600',
  },
});
