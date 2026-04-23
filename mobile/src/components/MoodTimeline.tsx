// components/MoodTimeline.tsx — scene 별 mood 색띠 + 재생 위치 indicator
//
// Timeline 타입은 backend timeline_writer.py 스키마 1.0 과 1:1 대응.
// Mood 7 클래스 (GEMS) 색 매핑은 감정 V/A 2D 배치를 시각적으로 반영:
//   negative·high arousal → 빨강, negative·low → 남색, positive·low → 하늘, ...
import React, {useMemo} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  LayoutChangeEvent,
} from 'react-native';
import {MoodName, TimelineScene} from '../types';

const MOOD_COLORS: Record<MoodName, string> = {
  Tension: '#D32F2F', // 진빨강 · neg·high
  Sadness: '#1976D2', // 남색 · neg·low
  Peacefulness: '#4FC3F7', // 하늘 · pos·low
  JoyfulActivation: '#FBC02D', // 노랑 · pos·high
  Tenderness: '#F8BBD0', // 연핑크 · warm
  Power: '#6A1B9A', // 진보라 · max arousal
  Wonder: '#7E57C2', // 보라 · spacious
};

const MOOD_LABELS_SHORT: Record<MoodName, string> = {
  Tension: '긴장',
  Sadness: '슬픔',
  Peacefulness: '평온',
  JoyfulActivation: '활기',
  Tenderness: '부드러움',
  Power: '힘',
  Wonder: '경이',
};

interface Props {
  scenes: TimelineScene[];
  durationSec: number;
  currentTimeSec: number;
  /** 탭하면 해당 씬 시작 초로 이동 (optional — 주면 터치 활성화) */
  onSeek?: (sec: number) => void;
  /** 전체 높이 (기본 64) */
  height?: number;
}

export default function MoodTimeline({
  scenes,
  durationSec,
  currentTimeSec,
  onSeek,
  height = 64,
}: Props) {
  const [barWidth, setBarWidth] = React.useState(0);

  const handleLayout = (e: LayoutChangeEvent) => {
    setBarWidth(e.nativeEvent.layout.width);
  };

  // 씬별 좌표 (px) 미리 계산
  const placed = useMemo(() => {
    if (durationSec <= 0 || barWidth <= 0) return [];
    return scenes.map(sc => {
      const left = (sc.start_sec / durationSec) * barWidth;
      const width = Math.max(1, ((sc.end_sec - sc.start_sec) / durationSec) * barWidth);
      return {scene: sc, left, width};
    });
  }, [scenes, durationSec, barWidth]);

  // 재생 indicator 위치
  const indicatorLeft =
    durationSec > 0 && barWidth > 0
      ? Math.min(
          barWidth - 2,
          Math.max(0, (currentTimeSec / durationSec) * barWidth),
        )
      : 0;

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Scene mood timeline</Text>
      <View
        style={[styles.bar, {height}]}
        onLayout={handleLayout}>
        {placed.map(({scene, left, width}) => {
          const color = MOOD_COLORS[scene.mood.name] ?? '#999';
          const short = MOOD_LABELS_SHORT[scene.mood.name] ?? scene.mood.name;
          const seg = (
            <View
              key={scene.scene_idx}
              style={[
                styles.segment,
                {
                  left,
                  width,
                  height,
                  backgroundColor: color,
                },
              ]}>
              {width > 28 && (
                <Text style={styles.segmentLabel} numberOfLines={1}>
                  {short}
                </Text>
              )}
            </View>
          );
          if (onSeek) {
            return (
              <TouchableOpacity
                key={scene.scene_idx}
                style={[
                  styles.segmentTouch,
                  {left, width, height},
                ]}
                onPress={() => onSeek(scene.start_sec)}
                activeOpacity={0.7}>
                <View
                  style={[
                    styles.segment,
                    {
                      left: 0,
                      width,
                      height,
                      backgroundColor: color,
                    },
                  ]}>
                  {width > 28 && (
                    <Text style={styles.segmentLabel} numberOfLines={1}>
                      {short}
                    </Text>
                  )}
                </View>
              </TouchableOpacity>
            );
          }
          return seg;
        })}

        {/* 재생 위치 indicator — 두꺼운 흰 선 + 얇은 검정 outline */}
        {barWidth > 0 && (
          <View
            pointerEvents="none"
            style={[
              styles.indicator,
              {left: indicatorLeft, height},
            ]}
          />
        )}
      </View>

      {/* 범례 — 현재 영상에 등장한 mood 만 */}
      <Legend scenes={scenes} />
    </View>
  );
}

function Legend({scenes}: {scenes: TimelineScene[]}) {
  const uniqueMoods = useMemo(() => {
    const seen = new Set<MoodName>();
    const out: MoodName[] = [];
    for (const s of scenes) {
      if (!seen.has(s.mood.name)) {
        seen.add(s.mood.name);
        out.push(s.mood.name);
      }
    }
    return out;
  }, [scenes]);

  return (
    <View style={styles.legend}>
      {uniqueMoods.map(m => (
        <View key={m} style={styles.legendItem}>
          <View style={[styles.legendSwatch, {backgroundColor: MOOD_COLORS[m]}]} />
          <Text style={styles.legendText}>{MOOD_LABELS_SHORT[m]}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingTop: 12,
  },
  label: {
    fontSize: 12,
    color: '#666',
    marginBottom: 6,
  },
  bar: {
    position: 'relative',
    width: '100%',
    borderRadius: 4,
    overflow: 'hidden',
    backgroundColor: '#eee',
  },
  segment: {
    position: 'absolute',
    top: 0,
    justifyContent: 'center',
    alignItems: 'center',
    borderRightWidth: 1,
    borderRightColor: 'rgba(0,0,0,0.15)',
  },
  segmentTouch: {
    position: 'absolute',
    top: 0,
  },
  segmentLabel: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '600',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowRadius: 2,
  },
  indicator: {
    position: 'absolute',
    top: 0,
    width: 2,
    backgroundColor: '#fff',
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#000',
  },
  legend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
    gap: 10,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendSwatch: {
    width: 10,
    height: 10,
    borderRadius: 2,
  },
  legendText: {
    fontSize: 11,
    color: '#555',
  },
});
