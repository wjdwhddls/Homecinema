// components/SpectrumDualLine.tsx
//
// 영상 재생 시 currentTime의 오디오 스펙트럼(원본)과 EQ 적용 후 스펙트럼을
// 한 차트에 dual-line 으로 오버레이. 백엔드 STFT spectrogram(timeline.spectrogram)
// 의 frame을 currentTime에 맞춰 조회하고, 현재 씬의 EQ band 응답을 곱해
// "보정 후"를 클라이언트에서 합성.
//
// 색상 정책 (대비 명확 + EQ 적용 임팩트 우위):
//  - 원본:    진한 빨강 #E63946, 1.8px, opacity 0.8  → 다크 배경에서 또렷하지만
//             EQ의 line+glow+fill 3중 구조 대비 시각 무게는 양보
//  - EQ 적용: 강렬한 호박색 #FFB347, 2.6px, opacity 1.0
//             + 외곽 glow halo (반투명 더 두꺼운 stroke 위에 메인 stroke)
//             + 곡선 아래 영역 그라디언트 fill (호박색 → 투명)
//             → 단순 빨강 line 1개 vs 호박색 line+glow+fill 3중 구조 → 임팩트 우위 유지
//
// EQ gain 막대(SceneEQChart) 와 톤이 통일되도록 호박색은 SceneEQChart의
// 부스트 컬러 #FF8A5B 보다 약간 밝은 #FFB347 사용 (yellow-amber).

import React, {useMemo} from 'react';
import {Pressable, StyleSheet, Text, View} from 'react-native';
import Svg, {
  Defs,
  LinearGradient,
  Path,
  Stop,
  Line,
} from 'react-native-svg';
import {EQBand, MoodName, SpectrogramData} from '../types';
import {biquadPeakingDb} from './EQResponseCurve';

const F_MIN = 20;
const F_MAX = 20000;
const VB_W = 1000;
// compact = 인라인 카드 (현재와 동일), expanded = 풀스크린 모달 (의미 동일, 큰 캔버스)
const VB_H_COMPACT = 180;
const VB_H_EXPANDED = 480;
const Y_AXIS_WIDTH = 28;

const ORIG_COLOR = '#E63946';
const EQ_COLOR = '#FFB347';
const EQ_GLOW_COLOR = '#FFD089';
const ZERO_LINE = 'rgba(245,245,247,0.18)';
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

function fxLog(f: number): number {
  return (Math.log10(f / F_MIN) / Math.log10(F_MAX / F_MIN)) * VB_W;
}

interface Props {
  spectrogram: SpectrogramData;
  currentTime: number;       // 영상 currentTime (초)
  bands: EQBand[];           // 현재 씬의 EQ 밴드
  moodName: MoodName;
  mode: 'original' | 'processed';
  variant?: 'compact' | 'expanded';   // compact = 인라인 카드, expanded = 풀스크린 모달
  onPress?: () => void;               // 카드 탭 핸들러 (탭하면 부모가 모달 열기)
}

export default function SpectrumDualLine({
  spectrogram,
  currentTime,
  bands,
  moodName,
  mode,
  variant = 'compact',
  onPress,
}: Props) {
  const isBypass = mode === 'original';
  const isExpanded = variant === 'expanded';
  const VB_H = isExpanded ? VB_H_EXPANDED : VB_H_COMPACT;
  const {hop_ms, freqs, frames_db, floor_db} = spectrogram;

  // currentTime → frame index. 범위 클램프.
  const frameIdx = useMemo(() => {
    const idx = Math.floor((currentTime * 1000) / hop_ms);
    return Math.max(0, Math.min(frames_db.length - 1, idx));
  }, [currentTime, hop_ms, frames_db.length]);

  const originalFrame = frames_db[frameIdx] ?? [];

  // 원본 spectrum + EQ 응답 = 보정 후 spectrum
  // mode=original (bypass) 이면 EQ 효과 미적용
  const correctedFrame = useMemo(() => {
    if (isBypass || originalFrame.length === 0) {
      return originalFrame;
    }
    const out = new Array<number>(originalFrame.length);
    for (let i = 0; i < freqs.length; i++) {
      let eqDb = 0;
      for (const b of bands) {
        eqDb += biquadPeakingDb(freqs[i], b.freq_hz, b.gain_db, b.q);
      }
      out[i] = originalFrame[i] + eqDb;
    }
    return out;
  }, [isBypass, originalFrame, freqs, bands]);

  // dB → viewBox y 좌표. floor_db ~ ceiling(0dB+10) 범위를 차트 높이에 맞춤.
  // EQ 적용 시 +값이 0dB을 넘을 수 있으므로 상단 10dB 여유.
  const Y_MAX_DB = 10;
  const Y_MIN_DB = floor_db; // -60
  const fyDb = (db: number): number => {
    const clamped = Math.max(Y_MIN_DB, Math.min(Y_MAX_DB, db));
    return ((Y_MAX_DB - clamped) / (Y_MAX_DB - Y_MIN_DB)) * VB_H;
  };

  const buildPath = (frame: number[]): string => {
    if (frame.length === 0 || frame.length !== freqs.length) {
      return '';
    }
    let d = `M ${fxLog(freqs[0]).toFixed(2)} ${fyDb(frame[0]).toFixed(2)}`;
    for (let i = 1; i < freqs.length; i++) {
      d += ` L ${fxLog(freqs[i]).toFixed(2)} ${fyDb(frame[i]).toFixed(2)}`;
    }
    return d;
  };

  const buildFilledPath = (frame: number[]): string => {
    const line = buildPath(frame);
    if (!line) return '';
    const lastX = fxLog(freqs[freqs.length - 1]).toFixed(2);
    const firstX = fxLog(freqs[0]).toFixed(2);
    return `${line} L ${lastX} ${VB_H} L ${firstX} ${VB_H} Z`;
  };

  // expanded variant에서는 stroke을 굵게 + 글자/패딩도 키움 (캔버스 비례 가독성)
  const strokeScale = isExpanded ? 2.0 : 1.0;
  const fontScale = isExpanded ? 1.6 : 1.0;

  const origPath = useMemo(() => buildPath(originalFrame), [originalFrame]);
  const corrPath = useMemo(() => buildPath(correctedFrame), [correctedFrame]);
  const corrFilledPath = useMemo(
    () => buildFilledPath(correctedFrame),
    [correctedFrame],
  );

  const moodKo = MOOD_LABELS_KO[moodName] ?? moodName;
  const xTicks = [20, 100, 1000, 10000];

  // 0dB 라인 위치
  const zeroY = fyDb(0);

  const cardContent = (
    <>
      {/* 헤더 */}
      <View style={styles.headerRow}>
        <Text
          style={[
            styles.title,
            isBypass ? styles.titleBypass : styles.titleActive,
            {fontSize: 14 * fontScale},
          ]}>
          {isBypass ? 'Spectrum (bypass)' : 'Spectrum'}
        </Text>
        <View style={[styles.tag, isBypass ? styles.tagBypass : styles.tagActive]}>
          <Text
            style={[
              styles.tagText,
              isBypass ? styles.tagTextBypass : styles.tagTextActive,
              {fontSize: 11 * fontScale},
            ]}>
            {isBypass
              ? 'EQ 미적용 — 원본만'
              : `${moodKo} · 원본 → EQ 적용${isExpanded ? '' : '  · tap ⤢'}`}
          </Text>
        </View>
      </View>

      {/* 차트 — y축 width와 글자 높이를 fontScale에 맞춰 동적 적용 (잘림 방지) */}
      <View style={[styles.chartArea, {height: VB_H}]}>
        <View
          style={[
            styles.yAxis,
            {height: VB_H, width: Y_AXIS_WIDTH * fontScale},
          ]}>
          {(['+10', '0', '-30', '-60'] as const).map(label => (
            <Text
              key={label}
              numberOfLines={1}
              style={[
                styles.yAxisLabel,
                {
                  fontSize:   10 * fontScale,
                  lineHeight: 13 * fontScale,
                  marginTop:  -5 * fontScale,
                },
              ]}>
              {label}
            </Text>
          ))}
        </View>

        <View style={[styles.plot, {height: VB_H}]}>
          <Svg
            width="100%"
            height={VB_H}
            viewBox={`0 0 ${VB_W} ${VB_H}`}
            preserveAspectRatio="none">
            <Defs>
              <LinearGradient
                id={`eqFill-${variant}`}
                x1="0"
                y1="0"
                x2="0"
                y2={VB_H}
                gradientUnits="userSpaceOnUse">
                <Stop offset="0" stopColor={EQ_COLOR} stopOpacity="0.35" />
                <Stop offset="1" stopColor={EQ_COLOR} stopOpacity="0" />
              </LinearGradient>
            </Defs>

            {/* 0dB 기준선 */}
            <Line
              x1="0"
              y1={zeroY}
              x2={VB_W}
              y2={zeroY}
              stroke={ZERO_LINE}
              strokeWidth={1}
              strokeDasharray="4 4"
            />
            {/* 위/아래 외곽 */}
            <Line x1="0" y1="0" x2={VB_W} y2="0" stroke={GRID_LINE} strokeWidth={0.5} />
            <Line x1="0" y1={VB_H} x2={VB_W} y2={VB_H} stroke={GRID_LINE} strokeWidth={0.5} />

            {/* 원본 곡선 — 진한 빨강 */}
            <Path
              d={origPath}
              stroke={ORIG_COLOR}
              strokeWidth={1.8 * strokeScale}
              strokeLinejoin="round"
              strokeLinecap="round"
              fill="none"
              opacity={0.8}
            />

            {!isBypass && (
              <>
                {/* EQ 적용 fill — 호박색 그라디언트 */}
                <Path d={corrFilledPath} fill={`url(#eqFill-${variant})`} />

                {/* EQ 적용 glow halo */}
                <Path
                  d={corrPath}
                  stroke={EQ_GLOW_COLOR}
                  strokeWidth={5 * strokeScale}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  fill="none"
                  opacity={0.35}
                />

                {/* EQ 적용 메인 stroke */}
                <Path
                  d={corrPath}
                  stroke={EQ_COLOR}
                  strokeWidth={2.6 * strokeScale}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  fill="none"
                  opacity={1}
                />
              </>
            )}
          </Svg>
        </View>
      </View>

      {/* x축 라벨 — spacer 폭은 y축 width와 동기화 */}
      <View style={[styles.xAxis, {height: 14 * fontScale, marginTop: 4 * fontScale}]}>
        <View style={[styles.xAxisSpacer, {width: Y_AXIS_WIDTH * fontScale}]} />
        <View style={[styles.xLabelsTrack, {height: 14 * fontScale}]}>
          {xTicks.map(f => {
            const pct = (fxLog(f) / VB_W) * 100;
            return (
              <Text
                key={f}
                numberOfLines={1}
                style={[
                  styles.xLabel,
                  {
                    left:       `${pct}%`,
                    fontSize:   9 * fontScale,
                    lineHeight: 12 * fontScale,
                  },
                ]}>
                {f >= 1000 ? `${f / 1000}k` : `${f}`}
              </Text>
            );
          })}
        </View>
      </View>
    </>
  );

  // expanded는 모달 안에 직접 풀너비 컨테이너로 렌더되므로 카드 margin/border 제거.
  const cardStyle = isExpanded ? styles.cardExpanded : styles.card;

  if (onPress && !isExpanded) {
    return (
      <Pressable
        onPress={onPress}
        android_ripple={{color: 'rgba(255,255,255,0.06)'}}
        style={({pressed}) => [cardStyle, pressed && {opacity: 0.85}]}>
        {cardContent}
      </Pressable>
    );
  }
  return <View style={cardStyle}>{cardContent}</View>;
}

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
  titleActive: {color: '#FFB347'},
  titleBypass: {color: '#9E7BE0'},
  tag: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    maxWidth: '70%',
  },
  tagActive: {
    backgroundColor: 'rgba(255,179,71,0.10)',
    borderColor: 'rgba(255,179,71,0.40)',
  },
  tagBypass: {
    backgroundColor: 'rgba(158,123,224,0.10)',
    borderColor: 'rgba(158,123,224,0.40)',
  },
  tagText: {fontSize: 11, fontWeight: '600'},
  tagTextActive: {color: '#FFD089'},
  tagTextBypass: {color: '#C6B1F0'},
  cardExpanded: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 12,
  },
  chartArea: {
    flexDirection: 'row',
  },
  yAxis: {
    width: Y_AXIS_WIDTH,
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingRight: 4,
  },
  yAxisLabel: {
    color: 'rgba(245,245,247,0.45)',
  },
  plot: {
    flex: 1,
    position: 'relative',
  },
  xAxis: {
    flexDirection: 'row',
    marginTop: 4,
    height: 14,
  },
  xAxisSpacer: {width: Y_AXIS_WIDTH},
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
