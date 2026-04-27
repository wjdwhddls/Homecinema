// components/MeasuredResponseCurve.tsx
//
// sweep 측정 결과의 transfer function H(f) 곡선 시각화 (light 테마).
// 입력: 백엔드 run_eq_pipeline의 curve.{freqs, measured_db | corrected_db}
// - 단일 곡선 polyline (96 log-spaced points)
// - 0dB = 평탄한 이상적 응답 (sweep 자체)
// - variant='measured' (빨강) 또는 'corrected' (시안)
// - 우상단 라벨에 max abs dB 표기
//
// EQResponseCurve.tsx의 fxLog/fyDb/SVG 그리드 패턴을 차용하되
// EQMeasurementScreen의 light 테마(흰 카드)에 맞춰 색상을 재구성함.

import React, {useMemo} from 'react';
import {StyleSheet, Text, View} from 'react-native';
import Svg, {Line, Path} from 'react-native-svg';

const F_MIN = 20;
const F_MAX = 20000;
const VB_W = 1000;
const VB_H = 180;
const ZERO_Y = VB_H / 2;
const Y_AXIS_WIDTH = 28;

const MEASURED_COLOR = '#FF6B6B';
const CORRECTED_COLOR = '#0EA5A4';
const GRID_LINE = '#e5e7eb';
const ZERO_LINE = '#6b7280';

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

  // dB → viewBox y 좌표 (위쪽이 양수)
  const fyDb = useMemo(() => {
    return (db: number): number => {
      const clamped = Math.max(-yRangeDb, Math.min(yRangeDb, db));
      return ZERO_Y - (clamped / yRangeDb) * ZERO_Y;
    };
  }, [yRangeDb]);

  // 곡선 path
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

  // 변동량 (max abs dB) — "얼마나 평탄한가"의 정량 지표
  const maxAbsDb = useMemo(() => {
    let m = 0;
    for (const v of values_db) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
    return m;
  }, [values_db]);

  // ±yRangeDb 기준의 그리드 라인 위치 (정수 dB 기준 동적 생성)
  const gridDbs = useMemo(() => {
    // ±yRangeDb 안에서 3 단위 그리드 (예: ±9 → -6, -3, +3, +6)
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
        <Text style={styles.title}>{title}</Text>
        <View style={[styles.tag, {borderColor: color}]}>
          <Text style={[styles.tagText, {color}]}>±{maxAbsDb.toFixed(1)} dB</Text>
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
            {/* 위/아래 경계 */}
            <Line x1="0" y1="0" x2={VB_W} y2="0" stroke={GRID_LINE} strokeWidth={0.5} />
            <Line x1="0" y1={VB_H} x2={VB_W} y2={VB_H} stroke={GRID_LINE} strokeWidth={0.5} />

            {/* 보조 그리드 (±3, ±6 등) */}
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

            {/* 측정/보정 곡선 */}
            <Path
              d={curvePath}
              stroke={color}
              strokeWidth={1.8}
              strokeLinejoin="round"
              strokeLinecap="round"
              fill="none"
              opacity={0.92}
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
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 4,
    shadowOffset: {width: 0, height: 2},
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  title: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1f2937',
    flex: 1,
  },
  tag: {
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 10,
    borderWidth: 1,
    backgroundColor: '#f9fafb',
  },
  tagText: {
    fontSize: 12,
    fontWeight: '600',
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
    color: '#9ca3af',
    lineHeight: 11,
    marginTop: -5,
  },
  yAxisLabelZero: {
    fontSize: 10,
    color: '#374151',
    fontWeight: '600',
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
    color: '#9ca3af',
    transform: [{translateX: -10}],
  },
  caption: {
    marginTop: 8,
    fontSize: 11,
    color: '#9ca3af',
    textAlign: 'center',
  },
});
