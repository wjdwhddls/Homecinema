/**
 * OptimizationResultScreen
 *
 * Phase 3 결과 표시: 최적 스피커 배치 좌표 + 음향 지표.
 * (3D/AR 시각화는 향후 Phase.)
 */
import React from 'react';
import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import {
  useRoute,
  useNavigation,
  RouteProp,
} from '@react-navigation/native';
import {RootStackParamList} from '../types';

type OptimizationResultRouteProp = RouteProp<
  RootStackParamList,
  'OptimizationResult'
>;

export default function OptimizationResultScreen() {
  const route = useRoute<OptimizationResultRouteProp>();
  const navigation = useNavigation<any>();
  const result = route.params?.result;

  if (!result || !result.best || !result.room_summary) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.center}>
          <Text style={styles.emptyTitle}>결과가 없습니다</Text>
          <TouchableOpacity
            style={styles.homeBtn}
            onPress={() => navigation.navigate('Home')}>
            <Text style={styles.homeBtnText}>홈으로</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const {
    best,
    room_summary,
    computation_time_seconds,
    top_alternatives = [],
    warnings = [],
  } = result;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>🎯 최적 스피커 위치</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>방 정보</Text>
          <Text style={styles.cardLine}>
            바닥 면적: {room_summary.floor_area_m2.toFixed(1)} m²
          </Text>
          <Text style={styles.cardLine}>
            높이: {room_summary.height_m.toFixed(2)} m
          </Text>
          <Text style={styles.cardLine}>
            체적: {room_summary.volume_m3.toFixed(1)} m³
          </Text>
          <Text style={styles.cardLine}>
            벽 {room_summary.wall_count}개 · 가구 {room_summary.object_count}개
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>🥇 권장 배치</Text>
          <CoordRow label="왼쪽 스피커" pos={best.placement.left} />
          <CoordRow label="오른쪽 스피커" pos={best.placement.right} />
          <CoordRow label="청취 위치" pos={best.placement.listener} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>📊 예상 음향 특성</Text>
          <MetricRow
            label="잔향 시간 (RT60)"
            value={`${best.metrics.rt60_seconds.toFixed(2)} 초`}
          />
          <MetricRow
            label="저주파 RT60"
            value={`${best.metrics.rt60_low.toFixed(2)} 초`}
          />
          <MetricRow
            label="중주파 RT60"
            value={`${best.metrics.rt60_mid.toFixed(2)} 초`}
          />
          <MetricRow
            label="부밍 심각도"
            value={`${best.metrics.standing_wave_severity_db.toFixed(1)} dB`}
            suffix={
              best.metrics.standing_wave_severity_db < 3 ? '✅ 양호' : '⚠️ 주의'
            }
          />
          <MetricRow
            label="주파수 평탄도"
            value={`${best.metrics.flatness_db.toFixed(1)} dB`}
          />
          <MetricRow
            label="직접/잔향 비율"
            value={`${best.metrics.direct_to_reverb_ratio_db.toFixed(1)} dB`}
          />
          <MetricRow
            label="초기 반사 비율"
            value={`${(best.metrics.early_reflection_ratio * 100).toFixed(0)}%`}
          />
        </View>

        {top_alternatives.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>
              대안 위치 ({top_alternatives.length}개)
            </Text>
            {top_alternatives.map((alt, i) => (
              <View key={i} style={styles.altRow}>
                <Text style={styles.altRank}>#{alt.rank}</Text>
                <Text style={styles.altDesc}>
                  L: ({alt.placement.left.x.toFixed(1)},{' '}
                  {alt.placement.left.y.toFixed(1)}) · RT60{' '}
                  {alt.metrics.rt60_seconds.toFixed(2)}s
                </Text>
              </View>
            ))}
          </View>
        )}

        {warnings.length > 0 && (
          <View style={[styles.card, styles.warnCard]}>
            <Text style={styles.cardTitle}>⚠️ 참고</Text>
            {warnings.map((w, i) => (
              <Text key={i} style={styles.warnLine}>
                • {w}
              </Text>
            ))}
          </View>
        )}

        <Text style={styles.footer}>
          계산 시간: {computation_time_seconds.toFixed(1)}초
        </Text>

        <TouchableOpacity
          style={styles.homeBtn}
          onPress={() => navigation.navigate('Home')}>
          <Text style={styles.homeBtnText}>홈으로</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const CoordRow = ({
  label,
  pos,
}: {
  label: string;
  pos: {x: number; y: number; z: number};
}) => (
  <View style={styles.coordRow}>
    <Text style={styles.coordLabel}>{label}</Text>
    <Text style={styles.coordValue}>
      ({pos.x.toFixed(2)}, {pos.y.toFixed(2)}, {pos.z.toFixed(2)}) m
    </Text>
  </View>
);

const MetricRow = ({
  label,
  value,
  suffix,
}: {
  label: string;
  value: string;
  suffix?: string;
}) => (
  <View style={styles.metricRow}>
    <Text style={styles.metricLabel}>{label}</Text>
    <Text style={styles.metricValue}>
      {value}
      {suffix ? ` ${suffix}` : ''}
    </Text>
  </View>
);

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#f9fafb'},
  content: {padding: 20, paddingBottom: 40},
  center: {flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24},
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  emptyTitle: {fontSize: 18, color: '#6b7280', marginBottom: 20},
  card: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 4,
    shadowOffset: {width: 0, height: 2},
    elevation: 2,
  },
  warnCard: {backgroundColor: '#fffbeb'},
  cardTitle: {
    fontSize: 17,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#111',
  },
  cardLine: {fontSize: 14, color: '#374151', marginBottom: 4},
  coordRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  coordLabel: {fontSize: 14, color: '#6b7280'},
  coordValue: {fontSize: 14, fontWeight: '500', color: '#111'},
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  metricLabel: {fontSize: 14, color: '#374151'},
  metricValue: {fontSize: 14, fontWeight: '500', color: '#111'},
  altRow: {flexDirection: 'row', paddingVertical: 6},
  altRank: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2563eb',
    marginRight: 10,
    width: 30,
  },
  altDesc: {fontSize: 13, color: '#374151', flex: 1},
  warnLine: {fontSize: 13, color: '#92400e', marginBottom: 4},
  footer: {
    fontSize: 12,
    color: '#9ca3af',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 20,
  },
  homeBtn: {
    backgroundColor: '#2563eb',
    padding: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  homeBtnText: {color: '#fff', fontSize: 16, fontWeight: '600'},
});
