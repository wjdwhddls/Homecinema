// components/MeasuredResponseCurve.tsx — 시네마틱 다크 리스킨
//
// sweep 측정 결과의 transfer function H(f) 곡선 시각화.
// 입력: 백엔드 run_eq_pipeline의 curve.{freqs, measured_db | corrected_db}
// - 단일 곡선 polyline (96 log-spaced points)
// - 0dB = 평탄한 이상적 응답 (sweep 자체)
// - variant='measured' (오렌지) 또는 'corrected' (시안)
// - 곡선 아래쪽으로 fade fill (광채 느낌)
// - 우상단 라벨에 max abs dB 표기

import React, {useMemo} from 'react';
import {StyleSheet, Text, View} from 'react-native';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  Stop,
  Line,
  Path,
} from 'react-native-svg';

const F_MIN = 20;
const F_MAX = 20000;
const VB_W = 1000;
const VB_H = 180;
const ZERO_Y = VB_H / 2;
const Y_AXIS_WIDTH = 28;

const MEASURED_COLOR = '#FF8A5B'; // 오렌지 (시네마틱 액센트)
const CORRECTED_COLOR = '#3DC8FF'; // 시안

const GRID_LINE = 'rgba(245,245,247,0.07)';
const ZERO_LINE = 'rgba(245,245,247,0.32)';

function fxLog(f: number): number {
  return (Math.log10(f / F_MIN) / Math.log10(F_MAX / F_MIN)) * VB_W;
}

interface Props {
  freqs: number[];
  values_db: number[];
  title: string;
  variant: 'measured' | 'corrected';
  yRangeDb?: number;
}

export default function MeasuredResponseCurve({
  freqs,
  values_db,
  title,
  variant,
  yRangeDb = 9,
}: Props) {
  const color = variant === 'measured' ? MEASURED_COLOR : CORRECTED_COLOR;
  const gradientId =
    variant === 'measured' ? 'measuredGlow' : 'correctedGlow';

  // dB → viewBox y 좌표 (위쪽이 양수)
  const fyDb = useMemo(() => {
    return (db: number): number => {
      const clamped = Math.max(-yRangeDb, Math.min(yRangeDb, db));
      return ZERO_Y - (clamped / yRangeDb) * ZERO_Y;
    };
  }, [yRangeDb]);

  // 곡선 stroke path
  const curvePath = useMemo(() => {
    const n = Math.min(freqs.length, values_db.length);
    if (n === 0) {
      return '';
    }
    let d = `M ${fxLog(freqs[0]).toFixed(2)} ${fyDb(values_db[0]).toFixed(2)}`;
    for (let i = 1; i < n; i++) {
      d += ` L ${fxLog(freqs[i]).toFixed(2)} ${fyDb(values_db[i]).toFixed(2)}`;
    }
    return d;
  }, [freqs, values_db, fyDb]);

  // 곡선 + 0dB 기준선 사이 area fill path
  const areaPath = useMemo(() => {
    const n = Math.min(freqs.length, values_db.length);
    if (n === 0) {
      return '';
    }
    let d = `M ${fxLog(freqs[0]).toFixed(2)} ${ZERO_Y}`;
    for (let i = 0; i < n; i++) {
      d += ` L ${fxLog(freqs[i]).toFixed(2)} ${fyDb(values_db[i]).toFixed(2)}`;
    }
    d += ` L ${fxLog(freqs[n - 1]).toFixed(2)} ${ZERO_Y} Z`;
    return d;
  }, [freqs, values_db, fyDb]);

  // 변동량 (max abs dB) — "얼마나 평탄한가"의 정량 지표
  const maxAbsDb = useMemo(() => {
    let m = 0;
    for (const v of values_db) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
    return m;
  }, [values_db]);

  // ±yRangeDb 안에서 3 단위 그리드
  const gridDbs = useMemo(() => {
    const out: number[] = [];
    const step = 3;
    for (let v = -yRangeDb + step; v < yRangeDb; v += step) {
      if (Math.abs(v) > 0.01) out.push(v);
    }
    return out;
  }, [yRangeDb]);

  const xTicks = [20, 100, 1000, 10000, 20000];

  return (
    <View style={styles.card}>
      {/* 헤더 */}
      <View style={styles.headerRow}>
        <View style={styles.titleRow}>
          <View style={[styles.dot, {backgroundColor: color, shadowColor: color}]} />
          <Text style={styles.title}>{title}</Text>
        </View>
        <View style={[styles.tag, {borderColor: color + '66'}]}>
          <Text style={[styles.tagText, {color}]}>
            ±{maxAbsDb.toFixed(1)} dB
          </Text>
        </View>
      </View>

      {/* 차트 */}
      <View style={styles.chartArea}>
        <View style={styles.yAxis}>
          <Text style={styles.yAxisLabel}>+{yRangeDb}</Text>
          <Text style={styles.yAxisLabelZero}>0</Text>
          <Text style={styles.yAxisLabel}>-{yRangeDb}</Text>
        </View>

        <View style={styles.plot}>
          <Svg
            width="100%"
            height={VB_H}
            viewBox={`0 0 ${VB_W} ${VB_H}`}
            preserveAspectRatio="none">
            <Defs>
              <SvgLinearGradient
                id={gradientId}
                x1="0%"
                y1="0%"
                x2="0%"
                y2="100%">
                <Stop offset="0%" stopColor={color} stopOpacity="0.28" />
                <Stop offset="100%" stopColor={color} stopOpacity="0" />
              </SvgLinearGradient>
            </Defs>

            {/* 위/아래 경계 */}
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
              y1={VB_H}
              x2={VB_W}
              y2={VB_H}
              stroke={GRID_LINE}
              strokeWidth={0.5}
            />

            {/* 보조 그리드 (±3, ±6) */}
            {gridDbs.map(db => (
              <Line
                key={db}
                x1="0"
                y1={fyDb(db)}
                x2={VB_W}
                y2={fyDb(db)}
                stroke={GRID_LINE}
                strokeWidth={0.5}
              />
            ))}

            {/* 0dB 기준선 (강조) — 평탄한 이상적 응답 */}
            <Line
              x1="0"
              y1={ZERO_Y}
              x2={VB_W}
              y2={ZERO_Y}
              stroke={ZERO_LINE}
              strokeWidth={1}
              strokeDasharray="4 4"
            />

            {/* Area glow */}
            <Path d={areaPath} fill={`url(#${gradientId})`} />

            {/* 측정/보정 곡선 */}
            <Path
              d={curvePath}
              stroke={color}
              strokeWidth={1.8}
              strokeLinejoin="round"
              strokeLinecap="round"
              fill="none"
              opacity={0.95}
            />
          </Svg>
        </View>
      </View>

      {/* x축 라벨 */}
      <View style={styles.xAxis}>
        <View style={styles.xAxisSpacer} />
        <View style={styles.xLabelsTrack}>
          {xTicks.map(f => {
            const pct = (fxLog(f) / VB_W) * 100;
            return (
              <Text key={f} style={[styles.xLabel, {left: `${pct}%`}]}>
                {f >= 1000 ? `${f / 1000}k` : `${f}`}
              </Text>
            );
          })}
        </View>
      </View>

      {/* 부제: 0dB 의미 안내 */}
      <Text style={styles.caption}>
        0 dB = 평탄한 이상적 응답 (sweep 자체)
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: 'rgba(255,255,255,0.03)',
    padding: 16,
    borderRadius: 14,
    marginBottom: 16,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.12)',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: 10,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 5,
    elevation: 5,
  },
  title: {
    fontSize: 14,
    fontWeight: '600',
    color: '#F5F5F7',
    letterSpacing: 0.3,
    flex: 1,
  },
  tag: {
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 100,
    borderWidth: 1,
    backgroundColor: 'rgba(255,255,255,0.03)',
  },
  tagText: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.4,
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
    fontSize: 9.5,
    color: 'rgba(245,245,247,0.42)',
    lineHeight: 11,
    marginTop: -5,
    fontVariant: ['tabular-nums'],
  },
  yAxisLabelZero: {
    fontSize: 9.5,
    color: '#F5F5F7',
    fontWeight: '600',
    lineHeight: 11,
    marginTop: -5,
    fontVariant: ['tabular-nums'],
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
    color: 'rgba(245,245,247,0.4)',
    transform: [{translateX: -10}],
    letterSpacing: 0.3,
  },
  caption: {
    marginTop: 10,
    fontSize: 10.5,
    color: 'rgba(245,245,247,0.4)',
    textAlign: 'center',
    letterSpacing: 0.3,
  },
});
