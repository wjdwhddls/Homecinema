// components/SweepRippleOverlay.tsx — sweep 측정 중 풀스크린 음파(waveform) 오버레이
//
// 디자인 컨셉 (audio vector waveform)
//   · 검은 BG 위에 가는 흰 라인 수십 줄이 겹쳐서 흐르는 오디오 음파.
//   · 다중 sine harmonic 합성 → 자연스러운 비대칭 envelope.
//   · 가운데가 amplitude 강하고 좌우 가장자리로 갈수록 잔잔해지는 bell envelope.
//   · 라인마다 phase / amplitude 미세 shift → 입체감 (image #1 의 multi-line texture).
//   · 시간 t 가 흐르며 phase 가 진행해 음파가 좌→우로 흘러가는 듯한 모션.
//
// prop 인터페이스는 기존 ripple 버전과 호환 유지.
import React, {useEffect, useMemo, useRef, useState} from 'react';
import {
  Animated,
  Dimensions,
  Easing,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Svg, {
  Defs,
  LinearGradient,
  RadialGradient,
  Stop,
  Rect,
  Path,
  G,
} from 'react-native-svg';
import {SpeakerDimensions} from '../types';

// ── 파라미터 ────────────────────────────────────────────────
const N_LAYERS = 28; // 동시 렌더 라인 수 (성능 ↔ 깊이감 균형)
const N_SAMPLES = 56; // 한 라인의 path point 수
const HIGHLIGHT = '#F5F5F7';

// 합성 sine harmonics — (frequency cycles/screen, amplitude, phase-speed rad/s)
const HARMONICS: ReadonlyArray<{f: number; a: number; ps: number}> = [
  {f: 0.55, a: 1.0, ps: 0.55},
  {f: 1.3, a: 0.55, ps: 1.1},
  {f: 2.45, a: 0.32, ps: 1.85},
  {f: 4.1, a: 0.18, ps: 2.5},
  {f: 6.8, a: 0.09, ps: 3.3},
];
// 정규화 분모 (= Σ amplitude)
const HARMONIC_SUM = HARMONICS.reduce((s, h) => s + h.a, 0);

interface Props {
  visible: boolean;
  caption?: string;
  subcaption?: string;
  // 호환성 위해 prop 유지 (시각엔 영향 없음 — 풀스크린 단일 waveform)
  speakerCount?: 1 | 2;
  speakerDimensions?: SpeakerDimensions;
}

// ─────────────────────────────────────────────────────────────────
// 메인 오버레이
// ─────────────────────────────────────────────────────────────────
export default function SweepRippleOverlay({
  visible,
  caption = '측정 중',
  subcaption = '스피커에서 소리가 나오면 움직이지 마세요',
}: Props) {
  const {width: SW, height: SH} = Dimensions.get('window');
  const cy = SH / 2;
  const ampScale = SH * 0.22; // 화면 높이 대비 진폭 비율

  // mount 유지 (페이드 아웃 동안 unmount 막음)
  const [mounted, setMounted] = useState(visible);

  // 페이드 in/out
  const fade = useRef(new Animated.Value(0)).current;

  // rAF phase
  const [phase, setPhase] = useState(0);

  // ── visible 변화 시 fade ─────────────────────────────────────
  useEffect(() => {
    if (visible) {
      setMounted(true);
      Animated.timing(fade, {
        toValue: 1,
        duration: 600,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(fade, {
        toValue: 0,
        duration: 500,
        easing: Easing.in(Easing.cubic),
        useNativeDriver: true,
      }).start(({finished}) => {
        if (finished) {
          setMounted(false);
        }
      });
    }
  }, [visible, fade]);

  // ── rAF phase 진행 ───────────────────────────────────────────
  useEffect(() => {
    if (!mounted) {
      return;
    }
    let raf = 0;
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
  }, [mounted]);

  // pre-compute x positions (변하지 않음)
  const xs = useMemo(() => {
    const a = new Array<number>(N_SAMPLES + 1);
    for (let i = 0; i <= N_SAMPLES; i++) {
      a[i] = (i / N_SAMPLES) * SW;
    }
    return a;
  }, [SW]);

  // pre-compute bell envelope (가운데 진하고 가장자리로 약해짐)
  const envelope = useMemo(() => {
    const a = new Array<number>(N_SAMPLES + 1);
    for (let i = 0; i <= N_SAMPLES; i++) {
      const xn = i / N_SAMPLES;
      const e = Math.exp(-Math.pow((xn - 0.5) * 2.4, 2));
      a[i] = e * 0.92 + 0.08;
    }
    return a;
  }, []);

  if (!mounted) {
    return null;
  }

  // amplitude 호흡 — 라이브한 느낌
  const ampPulse = 0.88 + 0.12 * Math.sin(phase * 0.75);

  // 각 layer 별 path build
  const center = (N_LAYERS - 1) / 2;
  const paths: {d: string; opacity: number; sw: number}[] = [];

  for (let l = 0; l < N_LAYERS; l++) {
    const distFromCenter = Math.abs(l - center) / center; // 0 ~ 1
    // 중심에 가까운 라인이 진함, 가장자리는 흐림
    const opacity = (1 - Math.pow(distFromCenter, 1.5) * 0.85) * 0.55;
    // 가장자리 라인은 더 가늘게
    const sw = 0.55 + (1 - distFromCenter) * 0.55;
    // layer 별 phase shift / amplitude shift
    const layerPhase = (l - center) * 0.05;
    const layerAmpShift = 1 - Math.pow(distFromCenter, 1.3) * 0.42;

    let d = '';
    for (let i = 0; i <= N_SAMPLES; i++) {
      const xn = i / N_SAMPLES;
      let y = 0;
      for (let k = 0; k < HARMONICS.length; k++) {
        const h = HARMONICS[k];
        y +=
          h.a *
          Math.sin(h.f * Math.PI * 2 * xn + h.ps * phase + layerPhase * h.f * 1.5);
      }
      y = y / HARMONIC_SUM;
      const yPx =
        cy + y * envelope[i] * ampScale * ampPulse * layerAmpShift;
      d +=
        i === 0
          ? `M${xs[i].toFixed(1)},${yPx.toFixed(1)}`
          : `L${xs[i].toFixed(1)},${yPx.toFixed(1)}`;
    }
    paths.push({d, opacity, sw});
  }

  return (
    <Animated.View
      pointerEvents={visible ? 'auto' : 'none'}
      style={[StyleSheet.absoluteFill, styles.root, {opacity: fade}]}>
      <Svg width={SW} height={SH} style={StyleSheet.absoluteFill}>
        <Defs>
          <RadialGradient id="wfBg" cx="50%" cy="50%" r="75%">
            <Stop offset="0%" stopColor="#1A1A22" stopOpacity={1} />
            <Stop offset="60%" stopColor="#0A0A0F" stopOpacity={1} />
            <Stop offset="100%" stopColor="#000000" stopOpacity={1} />
          </RadialGradient>
          <LinearGradient id="wfEdgeFade" x1="0" y1="0" x2="1" y2="0">
            <Stop offset="0%" stopColor="#000000" stopOpacity={1} />
            <Stop offset="9%" stopColor="#000000" stopOpacity={0} />
            <Stop offset="91%" stopColor="#000000" stopOpacity={0} />
            <Stop offset="100%" stopColor="#000000" stopOpacity={1} />
          </LinearGradient>
        </Defs>

        {/* ① BG */}
        <Rect width={SW} height={SH} fill="url(#wfBg)" />

        {/* ② waveform layers */}
        <G>
          {paths.map((p, i) => (
            <Path
              key={i}
              d={p.d}
              stroke={HIGHLIGHT}
              strokeWidth={p.sw}
              strokeLinecap="round"
              fill="none"
              opacity={p.opacity}
            />
          ))}
        </G>

        {/* ③ 좌우 가장자리 부드럽게 페이드 */}
        <Rect
          width={SW}
          height={SH}
          fill="url(#wfEdgeFade)"
          pointerEvents="none"
        />
      </Svg>

      {/* 중앙 라벨 — 화면 위쪽 */}
      <View style={styles.labelWrap} pointerEvents="none">
        <Text style={styles.eyebrow}>MEASURING</Text>
        <Text style={styles.title}>{caption}</Text>
        <Text style={styles.subtitle}>{subcaption}</Text>
      </View>
    </Animated.View>
  );
}

// ─────────────────────────────────────────────────────────────────
// 스타일
// ─────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: {
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  labelWrap: {
    position: 'absolute',
    top: '14%',
    alignItems: 'center',
    paddingHorizontal: 36,
  },
  eyebrow: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.55)',
    letterSpacing: 6,
    fontWeight: '600',
    marginBottom: 14,
  },
  title: {
    fontSize: 32,
    color: '#F5F5F7',
    fontWeight: '200',
    letterSpacing: -0.6,
    marginBottom: 14,
  },
  subtitle: {
    fontSize: 12,
    color: 'rgba(245,245,247,0.6)',
    letterSpacing: 0.5,
    textAlign: 'center',
    lineHeight: 18,
  },
});
