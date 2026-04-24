// components/SceneEQChart.tsx
//
// 현재 씬의 10-band EQ gain 을 세로 바 차트로 시각화.
// - y축 ±MAX_GAIN_DB, 0dB 중심선, ±3dB / ±6dB 그리드
// - 양수 bar: 오렌지(#FF8A5B)  · 부스트(warming)
// - 음수 bar: 퍼플(#9E7BE0)   · 컷(cooling)
// - mode='original' 이면 모든 값을 0 으로 flat 렌더 (bypass)
// - 씬이 바뀌면 bands 참조가 바뀌어 자연스럽게 재렌더
import React, {useMemo} from 'react';
import {View, Text, StyleSheet} from 'react-native';
import {EQBand, MoodName} from '../types';

const MAX_GAIN_DB = 6;
const CHART_HEIGHT = 180;
const HALF = CHART_HEIGHT / 2;
const BAR_WIDTH_RATIO = 0.55; // 슬롯 폭 대비 바 폭 비율

const POS_COLOR = '#FF8A5B'; // 오렌지 — 부스트
const NEG_COLOR = '#9E7BE0'; // 퍼플 — 컷
const ZERO_LINE = 'rgba(245,245,247,0.35)';
const GRID_LINE = 'rgba(245,245,247,0.08)';
const SOFT_BAND = 'rgba(245,245,247,0.04)';

const MOOD_LABELS_KO: Record<MoodName, string> = {
  Tension: '긴장',
  Sadness: '슬픔',
  Peacefulness: '평온',
  JoyfulActivation: '활기',
  Tenderness: '부드러움',
  Power: '힘',
  Wonder: '경이',
};

interface Props {
  bands: EQBand[];
  moodName: MoodName;
  mode: 'original' | 'processed';
}

export default function SceneEQChart({bands, moodName, mode}: Props) {
  const isBypass = mode === 'original';

  const maxAbsDelta = useMemo(() => {
    if (isBypass) {
      return 0;
    }
    return bands.reduce((m, b) => Math.max(m, Math.abs(b.gain_db)), 0);
  }, [bands, isBypass]);

  const moodKo = MOOD_LABELS_KO[moodName] ?? moodName;

  return (
    <View style={styles.card}>
      {/* 상단: 좌 라벨 · 우 태그 */}
      <View style={styles.headerRow}>
        <Text
          style={[
            styles.title,
            isBypass ? styles.titleBypass : styles.titleActive,
          ]}>
          {isBypass ? 'EQ gain (bypass)' : 'EQ gain'}
        </Text>
        <View
          style={[styles.tag, isBypass ? styles.tagBypass : styles.tagActive]}>
          <Text
            style={[
              styles.tagText,
              isBypass ? styles.tagTextBypass : styles.tagTextActive,
            ]}>
            {isBypass
              ? 'bypass — EQ 미적용'
              : `${moodKo} · max |Δ|=${maxAbsDelta.toFixed(1)}dB`}
          </Text>
        </View>
      </View>

      {/* 차트 영역 */}
      <View style={styles.chartArea}>
        {/* y축 라벨 */}
        <View style={styles.yAxis}>
          <Text style={styles.yAxisLabel}>+6</Text>
          <Text style={styles.yAxisLabel}>+3</Text>
          <Text style={styles.yAxisLabel}>0</Text>
          <Text style={styles.yAxisLabel}>-3</Text>
          <Text style={styles.yAxisLabel}>-6</Text>
        </View>

        {/* 플롯 */}
        <View style={styles.plot}>
          {/* ±1dB 음영 (투명존 힌트) */}
          <View
            style={[
              styles.softBand,
              {
                top: HALF - (HALF * 1) / MAX_GAIN_DB,
                height: (HALF * 2) / MAX_GAIN_DB,
              },
            ]}
          />
          {/* 그리드: ±6, ±3 */}
          <View style={[styles.gridLine, {top: 0}]} />
          <View
            style={[styles.gridLine, {top: HALF - (HALF * 3) / MAX_GAIN_DB}]}
          />
          <View
            style={[styles.gridLine, {top: HALF + (HALF * 3) / MAX_GAIN_DB}]}
          />
          <View style={[styles.gridLine, {top: CHART_HEIGHT - 1}]} />
          {/* 0dB 중심선 */}
          <View style={[styles.zeroLine, {top: HALF - 0.5}]} />

          {/* 바 */}
          <View style={styles.barsRow}>
            {bands.map((b, i) => {
              const gain = isBypass ? 0 : b.gain_db;
              const h = Math.min(1, Math.abs(gain) / MAX_GAIN_DB) * HALF;
              const isPos = gain >= 0;
              return (
                <View key={i} style={styles.barSlot}>
                  {isBypass ? (
                    <View style={styles.flatDot} />
                  ) : h < 0.5 ? (
                    // gain≈0: 얇은 점으로만 표시
                    <View style={[styles.flatDot, {top: HALF - 1}]} />
                  ) : (
                    <View
                      style={[
                        styles.bar,
                        isPos
                          ? {
                              bottom: HALF,
                              height: h,
                              backgroundColor: POS_COLOR,
                            }
                          : {
                              top: HALF,
                              height: h,
                              backgroundColor: NEG_COLOR,
                            },
                      ]}
                    />
                  )}
                </View>
              );
            })}
          </View>
        </View>
      </View>

      {/* x축 freq 라벨 */}
      <View style={styles.xAxis}>
        <View style={styles.xAxisSpacer} />
        <View style={styles.xLabelsRow}>
          {bands.map((b, i) => (
            <View key={i} style={styles.xLabelSlot}>
              <Text style={styles.xLabel} numberOfLines={1}>
                {fmtFreq(b.freq_hz)}
              </Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );
}

function fmtFreq(hz: number): string {
  if (hz >= 1000) {
    const k = hz / 1000;
    return Number.isInteger(k) ? `${k}k` : `${k.toFixed(1)}k`;
  }
  return `${Math.round(hz)}`;
}

const Y_AXIS_WIDTH = 28;

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
    color: '#FF8A5B',
  },
  titleBypass: {
    color: '#9E7BE0',
  },
  tag: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
  },
  tagActive: {
    backgroundColor: 'rgba(255,138,91,0.10)',
    borderColor: 'rgba(255,138,91,0.40)',
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
    color: '#FFB995',
  },
  tagTextBypass: {
    color: '#C6B1F0',
  },
  chartArea: {
    flexDirection: 'row',
    height: CHART_HEIGHT,
  },
  yAxis: {
    width: Y_AXIS_WIDTH,
    height: CHART_HEIGHT,
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingRight: 4,
  },
  yAxisLabel: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.45)',
    lineHeight: 11,
    marginTop: -5, // 첫/끝 라벨을 그리드 라인에 정렬
  },
  plot: {
    flex: 1,
    height: CHART_HEIGHT,
    position: 'relative',
  },
  softBand: {
    position: 'absolute',
    left: 0,
    right: 0,
    backgroundColor: SOFT_BAND,
  },
  gridLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: GRID_LINE,
  },
  zeroLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: ZERO_LINE,
  },
  barsRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'stretch',
  },
  barSlot: {
    flex: 1,
    position: 'relative',
    marginHorizontal: 2,
  },
  bar: {
    position: 'absolute',
    left: `${((1 - BAR_WIDTH_RATIO) / 2) * 100}%`,
    right: `${((1 - BAR_WIDTH_RATIO) / 2) * 100}%`,
    borderRadius: 2,
  },
  flatDot: {
    position: 'absolute',
    left: `${((1 - BAR_WIDTH_RATIO) / 2) * 100}%`,
    right: `${((1 - BAR_WIDTH_RATIO) / 2) * 100}%`,
    top: HALF - 1,
    height: 2,
    borderRadius: 1,
    backgroundColor: 'rgba(245,245,247,0.25)',
  },
  xAxis: {
    flexDirection: 'row',
    marginTop: 4,
  },
  xAxisSpacer: {
    width: Y_AXIS_WIDTH,
  },
  xLabelsRow: {
    flex: 1,
    flexDirection: 'row',
  },
  xLabelSlot: {
    flex: 1,
    marginHorizontal: 2,
    alignItems: 'center',
  },
  xLabel: {
    fontSize: 9,
    color: 'rgba(245,245,247,0.5)',
  },
});
