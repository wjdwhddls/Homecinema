// components/EQResponseCurve.tsx
//
// 현재 씬의 10-band biquad peaking EQ 의 주파수 응답을 곡선으로 시각화 (Pro-Q4 풍).
// - 각 밴드의 biquad transfer function 을 계산해 모두 합산 → 실제 백엔드 EQ 와 동일한 곡선
// - 0dB 위 (boost): 오렌지 그라디언트 영역
// - 0dB 아래 (cut): 퍼플 그라디언트 영역
// - 곡선 라인: 노란 임팩트 라인 / 10개 밴드 포인트 표시
// - x축: 로그 스케일 20Hz ~ 20kHz
// - y축: ±6dB
// - 씬 전환 시 짧은 페이드로 변화 인지 강조
// - mode='original' 이면 평탄 라인만, 그라디언트 영역 미표시
import React, {useEffect, useMemo, useRef} from 'react';
import {Animated, StyleSheet, Text, View} from 'react-native';
import Svg, {
  Circle,
  ClipPath,
  Defs,
  LinearGradient,
  Line,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';
import {EQBand, MoodName} from '../types';

// ── 상수 ────────────────────────────────────────────────────────
// 영상 오디오는 일반적으로 48kHz. 가정값이지만 |H(f)| 모양은 f/fs 비율로
// 결정되므로 44.1kHz 라도 가청 영역에서의 시각 차이는 무시할 수준.
const FS = 48000;
const F_MIN = 20;
const F_MAX = 20000;
const N_SAMPLES = 256; // 곡선 샘플 포인트 수
const VB_W = 1000;
const VB_H = 180;
const ZERO_Y = VB_H / 2;
const MAX_DB = 6;
const Y_AXIS_WIDTH = 28;

const POS_COLOR = '#FF8A5B'; // boost — 오렌지 (SceneEQChart 와 동일)
const NEG_COLOR = '#9E7BE0'; // cut   — 퍼플   (SceneEQChart 와 동일)
const LINE_COLOR = '#FFD89E'; // 곡선 임팩트 라인 (옅은 골드)
const ZERO_LINE = 'rgba(245,245,247,0.35)';
const GRID_LINE = 'rgba(245,245,247,0.08)';

const MOOD_LABELS_KO: Record<MoodName, string> = {
  Tension: '긴장',
  Sadness: '슬픔',
  Peacefulness: '평온',
  JoyfulActivation: '활기',
  Tenderness: '부드러움',
  Power: '힘',
  Wonder: '경이',
};

// ── 수학 ────────────────────────────────────────────────────────
// RBJ Audio Cookbook 의 biquad peaking filter 응답을 dB 로 반환.
// dBgain≈0 이면 응답은 거의 0이므로 빠른 경로로 단축.
function biquadPeakingDb(
  f: number,
  fc: number,
  gainDb: number,
  q: number,
): number {
  if (Math.abs(gainDb) < 1e-3) {
    return 0;
  }
  const omega0 = (2 * Math.PI * fc) / FS;
  const A = Math.pow(10, gainDb / 40);
  const sinW0 = Math.sin(omega0);
  const cosW0 = Math.cos(omega0);
  const alpha = sinW0 / (2 * q);

  const b0 = 1 + alpha * A;
  const b1 = -2 * cosW0;
  const b2 = 1 - alpha * A;
  const a0 = 1 + alpha / A;
  const a1 = -2 * cosW0;
  const a2 = 1 - alpha / A;

  const omega = (2 * Math.PI * f) / FS;
  const cosW = Math.cos(omega);
  const sinW = Math.sin(omega);
  const cos2W = Math.cos(2 * omega);
  const sin2W = Math.sin(2 * omega);

  // z^-1 = e^{-jω} = cosω - j sinω
  // N(z) = b0 + b1 z^-1 + b2 z^-2
  const numRe = b0 + b1 * cosW + b2 * cos2W;
  const numIm = -b1 * sinW - b2 * sin2W;
  const numMag2 = numRe * numRe + numIm * numIm;

  const denRe = a0 + a1 * cosW + a2 * cos2W;
  const denIm = -a1 * sinW - a2 * sin2W;
  const denMag2 = denRe * denRe + denIm * denIm;

  return 10 * Math.log10(numMag2 / denMag2);
}

function totalDbAt(f: number, bands: EQBand[]): number {
  let sum = 0;
  for (const b of bands) {
    sum += biquadPeakingDb(f, b.freq_hz, b.gain_db, b.q);
  }
  return sum;
}

// 주파수 → viewBox x 좌표 (로그 스케일)
function fxLog(f: number): number {
  return (Math.log10(f / F_MIN) / Math.log10(F_MAX / F_MIN)) * VB_W;
}

// dB → viewBox y 좌표 (위쪽이 양수)
function fyDb(db: number): number {
  const clamped = Math.max(-MAX_DB, Math.min(MAX_DB, db));
  return ZERO_Y - (clamped / MAX_DB) * ZERO_Y;
}

// ── 컴포넌트 ────────────────────────────────────────────────────
interface Props {
  bands: EQBand[];
  moodName: MoodName;
  mode: 'original' | 'processed';
}

export default function EQResponseCurve({bands, moodName, mode}: Props) {
  const isBypass = mode === 'original';

  // 각 sample frequency 에서 누적 dB 계산
  const samples = useMemo(() => {
    const out: {x: number; y: number}[] = new Array(N_SAMPLES);
    const logRatio = Math.log10(F_MAX / F_MIN);
    for (let i = 0; i < N_SAMPLES; i++) {
      const t = i / (N_SAMPLES - 1);
      const f = F_MIN * Math.pow(10, t * logRatio);
      const db = isBypass ? 0 : totalDbAt(f, bands);
      out[i] = {x: t * VB_W, y: fyDb(db)};
    }
    return out;
  }, [bands, isBypass]);

  // 곡선 path
  const curvePath = useMemo(() => {
    if (samples.length === 0) {
      return '';
    }
    let d = `M ${samples[0].x.toFixed(2)} ${samples[0].y.toFixed(2)}`;
    for (let i = 1; i < samples.length; i++) {
      d += ` L ${samples[i].x.toFixed(2)} ${samples[i].y.toFixed(2)}`;
    }
    return d;
  }, [samples]);

  // 0dB 라인까지 닫힌 path (boost / cut 영역 채우기용)
  const filledPath = useMemo(() => {
    if (samples.length === 0) {
      return '';
    }
    const first = samples[0];
    const last = samples[samples.length - 1];
    let d = `M ${first.x.toFixed(2)} ${ZERO_Y}`;
    for (const s of samples) {
      d += ` L ${s.x.toFixed(2)} ${s.y.toFixed(2)}`;
    }
    d += ` L ${last.x.toFixed(2)} ${ZERO_Y} Z`;
    return d;
  }, [samples]);

  // 밴드 포인트 (곡선 위에 정확히 앉도록 누적 dB 사용)
  const bandPoints = useMemo(() => {
    if (isBypass) {
      return [];
    }
    return bands.map(b => ({
      cx: fxLog(b.freq_hz),
      cy: fyDb(totalDbAt(b.freq_hz, bands)),
    }));
  }, [bands, isBypass]);

  // 씬 전환 시 짧은 페이드 (사용자가 변화를 인지하도록)
  const opacity = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    if (isBypass) {
      opacity.setValue(1);
      return;
    }
    Animated.sequence([
      Animated.timing(opacity, {
        toValue: 0.35,
        duration: 120,
        useNativeDriver: true,
      }),
      Animated.timing(opacity, {
        toValue: 1,
        duration: 220,
        useNativeDriver: true,
      }),
    ]).start();
  }, [bands, isBypass, opacity]);

  const moodKo = MOOD_LABELS_KO[moodName] ?? moodName;
  const maxAbsDelta = useMemo(
    () =>
      isBypass
        ? 0
        : bands.reduce((m, b) => Math.max(m, Math.abs(b.gain_db)), 0),
    [bands, isBypass],
  );

  // x축 라벨 위치 (로그 스케일 기반)
  const xTicks = [20, 100, 1000, 10000, 20000];

  return (
    <View style={styles.card}>
      {/* 상단 헤더 */}
      <View style={styles.headerRow}>
        <Text
          style={[
            styles.title,
            isBypass ? styles.titleBypass : styles.titleActive,
          ]}>
          {isBypass ? 'EQ response (bypass)' : 'EQ response'}
        </Text>
        <View
          style={[styles.tag, isBypass ? styles.tagBypass : styles.tagActive]}>
          <Text
            style={[
              styles.tagText,
              isBypass ? styles.tagTextBypass : styles.tagTextActive,
            ]}>
            {isBypass
              ? 'flat — 0 dB'
              : `${moodKo} · biquad peaking · |Δ|≤${maxAbsDelta.toFixed(1)}dB`}
          </Text>
        </View>
      </View>

      {/* 차트 */}
      <View style={styles.chartArea}>
        <View style={styles.yAxis}>
          <Text style={styles.yAxisLabel}>+6</Text>
          <Text style={styles.yAxisLabel}>+3</Text>
          <Text style={styles.yAxisLabel}>0</Text>
          <Text style={styles.yAxisLabel}>-3</Text>
          <Text style={styles.yAxisLabel}>-6</Text>
        </View>

        <View style={styles.plot}>
          <Animated.View style={{opacity, width: '100%', height: VB_H}}>
            <Svg
              width="100%"
              height={VB_H}
              viewBox={`0 0 ${VB_W} ${VB_H}`}
              preserveAspectRatio="none">
              <Defs>
                <LinearGradient
                  id="boostGrad"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2={ZERO_Y}
                  gradientUnits="userSpaceOnUse">
                  <Stop offset="0" stopColor={POS_COLOR} stopOpacity="0.55" />
                  <Stop offset="1" stopColor={POS_COLOR} stopOpacity="0" />
                </LinearGradient>
                <LinearGradient
                  id="cutGrad"
                  x1="0"
                  y1={ZERO_Y}
                  x2="0"
                  y2={VB_H}
                  gradientUnits="userSpaceOnUse">
                  <Stop offset="0" stopColor={NEG_COLOR} stopOpacity="0" />
                  <Stop offset="1" stopColor={NEG_COLOR} stopOpacity="0.55" />
                </LinearGradient>
                <ClipPath id="aboveZero">
                  <Rect x="0" y="0" width={VB_W} height={ZERO_Y} />
                </ClipPath>
                <ClipPath id="belowZero">
                  <Rect x="0" y={ZERO_Y} width={VB_W} height={ZERO_Y} />
                </ClipPath>
              </Defs>

              {/* 그리드: ±6, ±3, 0dB */}
              <Line
                x1="0"
                y1="0"
                x2={VB_W}
                y2="0"
                stroke={GRID_LINE}
                strokeWidth={0.5}
              />
              <Line
                x1="0"
                y1={fyDb(3)}
                x2={VB_W}
                y2={fyDb(3)}
                stroke={GRID_LINE}
                strokeWidth={0.5}
              />
              <Line
                x1="0"
                y1={ZERO_Y}
                x2={VB_W}
                y2={ZERO_Y}
                stroke={ZERO_LINE}
                strokeWidth={1}
              />
              <Line
                x1="0"
                y1={fyDb(-3)}
                x2={VB_W}
                y2={fyDb(-3)}
                stroke={GRID_LINE}
                strokeWidth={0.5}
              />
              <Line
                x1="0"
                y1={VB_H}
                x2={VB_W}
                y2={VB_H}
                stroke={GRID_LINE}
                strokeWidth={0.5}
              />

              {/* 채움 영역 — 동일 닫힘 path 를 두 번, clip 으로 위/아래 분리 */}
              {!isBypass && (
                <>
                  <Path
                    d={filledPath}
                    fill="url(#boostGrad)"
                    clipPath="url(#aboveZero)"
                  />
                  <Path
                    d={filledPath}
                    fill="url(#cutGrad)"
                    clipPath="url(#belowZero)"
                  />
                </>
              )}

              {/* 곡선 라인 */}
              <Path
                d={curvePath}
                stroke={isBypass ? 'rgba(245,245,247,0.35)' : LINE_COLOR}
                strokeWidth={1.6}
                fill="none"
              />

              {/* 10개 밴드 포인트 */}
              {bandPoints.map((p, i) => (
                <Circle
                  key={i}
                  cx={p.cx}
                  cy={p.cy}
                  r="4"
                  fill="#ffffff"
                  stroke="rgba(0,0,0,0.45)"
                  strokeWidth={0.6}
                />
              ))}
            </Svg>
          </Animated.View>
        </View>
      </View>

      {/* x축 라벨 (로그 스케일 위치) */}
      <View style={styles.xAxis}>
        <View style={styles.xAxisSpacer} />
        <View style={styles.xLabelsTrack}>
          {xTicks.map(f => {
            const pct = (fxLog(f) / VB_W) * 100;
            return (
              <Text
                key={f}
                style={[
                  styles.xLabel,
                  {left: `${pct}%`},
                ]}>
                {f >= 1000 ? `${f / 1000}k` : `${f}`}
              </Text>
            );
          })}
        </View>
      </View>
    </View>
  );
}

// ── 스타일 (SceneEQChart 와 톤 통일) ─────────────────────────────
const styles = StyleSheet.create({
  card: {
    marginHorizontal: 16,
    marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingTop: 14,
    paddingBottom: 10,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  title: {
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  titleActive: {
    color: '#FFD89E',
  },
  titleBypass: {
    color: '#9E7BE0',
  },
  tag: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    maxWidth: '70%',
  },
  tagActive: {
    backgroundColor: 'rgba(255,216,158,0.10)',
    borderColor: 'rgba(255,216,158,0.40)',
  },
  tagBypass: {
    backgroundColor: 'rgba(158,123,224,0.10)',
    borderColor: 'rgba(158,123,224,0.40)',
  },
  tagText: {
    fontSize: 11,
    fontWeight: '600',
  },
  tagTextActive: {
    color: '#FFE3B5',
  },
  tagTextBypass: {
    color: '#C6B1F0',
  },
  chartArea: {
    flexDirection: 'row',
    height: VB_H,
  },
  yAxis: {
    width: Y_AXIS_WIDTH,
    height: VB_H,
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingRight: 4,
  },
  yAxisLabel: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.45)',
    lineHeight: 11,
    marginTop: -5,
  },
  plot: {
    flex: 1,
    height: VB_H,
    position: 'relative',
  },
  xAxis: {
    flexDirection: 'row',
    marginTop: 4,
    height: 14,
  },
  xAxisSpacer: {
    width: Y_AXIS_WIDTH,
  },
  xLabelsTrack: {
    flex: 1,
    height: 14,
    position: 'relative',
  },
  xLabel: {
    position: 'absolute',
    fontSize: 9,
    color: 'rgba(245,245,247,0.5)',
    transform: [{translateX: -10}],
  },
});
