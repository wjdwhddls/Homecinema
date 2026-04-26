/**
 * OptimizationResultScreen
 *
 * xRIR 최적 스피커 배치 결과 표시
 * - left/right 스테레오 좌표
 * - 음향 점수 (RT60, C80, DRR)
 * - 대안 위치 목록
 * - EQ 보정 측정으로 이동 버튼
 */
import React from 'react';
import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Image,
  Alert,
} from 'react-native';
import {useRoute, useNavigation, RouteProp} from '@react-navigation/native';
import {RootStackParamList} from '../types';
import {showRoomPreview, PREVIEW_COLORS} from '../native/RoomPreview';

type OptimizationResultRouteProp = RouteProp<
  RootStackParamList,
  'OptimizationResult'
>;

export default function OptimizationResultScreen() {
  const route      = useRoute<OptimizationResultRouteProp>();
  const navigation = useNavigation<any>();
  const result     = route.params?.result;

  if (!result || !result.best) {
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

  const {best, top_alternatives = [], warnings = [], computation_time_seconds} = result;
  const usdzUri = route.params?.usdzUri;
  const speakerDimensions = route.params?.speakerDimensions;

  const scorePercent = Math.round((best.score ?? 0) * 100);

  const handleShow3D = async () => {
    if (!usdzUri) {
      Alert.alert('3D 미리보기 불가', '방 스캔 데이터(USDZ)가 없어 3D 보기를 사용할 수 없습니다.');
      return;
    }
    const dimensions = speakerDimensions
      ? {
          width_m:  speakerDimensions.width_cm  / 100,
          height_m: speakerDimensions.height_cm / 100,
          depth_m:  speakerDimensions.depth_cm  / 100,
        }
      : undefined;
    try {
      await showRoomPreview({
        usdzUri,
        listener: best.placement.listener,
        speakers: [
          {label: '왼쪽 스피커',  color: PREVIEW_COLORS.left,  ...best.placement.left,  dimensions},
          {label: '오른쪽 스피커', color: PREVIEW_COLORS.right, ...best.placement.right, dimensions},
        ],
      });
    } catch (err: any) {
      Alert.alert('3D 미리보기 실패', err?.message || '알 수 없는 오류');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>최적 스피커 위치</Text>
        {result.topview_image && (
          <Image
            source={{uri: `data:image/png;base64,${result.topview_image}`}}
            style={{width: '100%', aspectRatio: 1, borderRadius: 12, marginBottom: 12}}
            resizeMode="contain"
          />
        )}

        {usdzUri && (
          <TouchableOpacity
            style={styles.preview3dBtn}
            onPress={handleShow3D}
            activeOpacity={0.8}>
            <Text style={styles.preview3dBtnText}>🏠 3D로 자세히 보기</Text>
          </TouchableOpacity>
        )}

        {/* 종합 점수 */}
        <View style={styles.scoreCard}>
          <Text style={styles.scoreLabel}>음향 종합 점수</Text>
          <Text style={styles.scoreValue}>{scorePercent}</Text>
          <Text style={styles.scoreUnit}>/ 100</Text>
        </View>

        {/* 권장 배치 좌표 */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>권장 배치</Text>
          <CoordRow label="왼쪽 스피커 (L)" pos={best.placement.left} />
          <CoordRow label="오른쪽 스피커 (R)" pos={best.placement.right} />
          <CoordRow label="청취 위치" pos={best.placement.listener} />
          <Text style={styles.coordNote}>
            * 방 좌표 (단위: m) — 청취 위치 행을 기준으로 좌우(x)·앞뒤(y)·높이(z) 비교하세요
          </Text>
        </View>

        {/* 음향 지표 */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>예상 음향 특성</Text>
          
          <MetricRow
            label="잔향 시간 (RT60)"
            value={`${best.metrics.rt60_seconds.toFixed(2)} 초`}
            badge={
              best.metrics.rt60_seconds < 0.3 ? '짧음'
              : best.metrics.rt60_seconds < 0.5 ? '양호'
              : best.metrics.rt60_seconds < 0.7 ? '보통'
              : '길음'
            }
          />
          <MetricRow
            label="명료도 (C80)"
            value={`${best.metrics.c80_db.toFixed(1)} dB`}
            badge={
              best.metrics.c80_db >= 3  ? '매우 명료'
              : best.metrics.c80_db >= 0  ? '양호'
              : best.metrics.c80_db >= -3 ? '보통'
              : '흐림'
            }
          />
          <MetricRow
            label="직접음 비율 (DRR)"
            value={`${best.metrics.drr_db.toFixed(1)} dB`}
            badge={
              best.metrics.drr_db >= 6 ? '강함'
              : best.metrics.drr_db >= 0 ? '양호'
              : '잔향 우세'
            }
          />

          <View style={styles.scoreBreakdown}>
            <Text style={styles.breakdownTitle}>세부 점수</Text>
            <View style={styles.breakdownRow}>
              <ScoreBar label="RT60" value={best.metrics.rt60_score} />
              <ScoreBar label="C80"  value={best.metrics.c80_score} />
              <ScoreBar label="DRR"  value={best.metrics.drr_score} />
            </View>
          </View>
        </View>

        {/* 대안 위치 */}
        {top_alternatives.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>
              대안 위치 ({top_alternatives.length}개)
            </Text>
            {top_alternatives.map((alt, i) => (
              <View key={i} style={styles.altRow}>
                <View style={styles.altRankBadge}>
                  <Text style={styles.altRankText}>#{alt.rank + 1}</Text>
                </View>
                <View style={styles.altInfo}>
                  <Text style={styles.altCoord}>
                    L: ({alt.placement.left.x.toFixed(2)}, {alt.placement.left.y.toFixed(2)}) m
                  </Text>
                  <Text style={styles.altCoord}>
                    R: ({alt.placement.right.x.toFixed(2)}, {alt.placement.right.y.toFixed(2)}) m
                  </Text>
                  <Text style={styles.altScore}>
                    점수: {Math.round((alt.score ?? 0) * 100)} · RT60 {alt.metrics.rt60_seconds.toFixed(2)}s
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* 경고 */}
        {warnings.length > 0 && (
          <View style={[styles.card, styles.warnCard]}>
            <Text style={styles.cardTitle}>참고</Text>
            {warnings.map((w, i) => (
              <Text key={i} style={styles.warnLine}>• {w}</Text>
            ))}
          </View>
        )}

        <Text style={styles.footer}>
          계산 시간: {(computation_time_seconds ?? 0).toFixed(1)}초
        </Text>

        {/* ── EQ 측정 버튼 (추가) ──────────────────────────────── */}
        <TouchableOpacity
          style={styles.eqBtn}
          onPress={() =>
            navigation.navigate('EQMeasurement', {
              optimalPosition: best.placement,
            })
          }>
          <Text style={styles.homeBtnText}>EQ 자동 보정 측정하기</Text>
        </TouchableOpacity>

        {/* 홈으로 버튼 */}
        <TouchableOpacity
          style={styles.homeBtn}
          onPress={() => navigation.navigate('Home')}>
          <Text style={styles.homeBtnText}>홈으로</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

// ── 서브 컴포넌트 ────────────────────────────────────────────────

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
      ({pos.x.toFixed(2)}, {pos.y.toFixed(2)}, {pos.z.toFixed(2)})
    </Text>
  </View>
);

const MetricRow = ({
  label,
  value,
  badge,
}: {
  label: string;
  value: string;
  badge?: string;
}) => (
  <View style={styles.metricRow}>
    <Text style={styles.metricLabel}>{label}</Text>
    <View style={styles.metricRight}>
      <Text style={styles.metricValue}>{value}</Text>
      {badge ? <Text style={styles.metricBadge}>{badge}</Text> : null}
    </View>
  </View>
);

const ScoreBar = ({label, value}: {label: string; value: number}) => {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <View style={styles.barWrap}>
      <Text style={styles.barLabel}>{label}</Text>
      <View style={styles.barBg}>
        <View style={[styles.barFill, {width: `${pct}%`}]} />
      </View>
      <Text style={styles.barPct}>{pct}</Text>
    </View>
  );
};

// ── 스타일 ───────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container:      {flex: 1, backgroundColor: '#f9fafb'},
  content:        {padding: 20, paddingBottom: 40},
  center:         {flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24},
  title:          {fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginBottom: 20},
  emptyTitle:     {fontSize: 18, color: '#6b7280', marginBottom: 20},

  scoreCard: {
    backgroundColor: '#2563eb', borderRadius: 16, padding: 20,
    alignItems: 'center', marginBottom: 16, flexDirection: 'row',
    justifyContent: 'center', gap: 8,
  },
  scoreLabel: {color: '#bfdbfe', fontSize: 15, marginRight: 12},
  scoreValue: {color: '#fff', fontSize: 48, fontWeight: 'bold'},
  scoreUnit:  {color: '#bfdbfe', fontSize: 20, alignSelf: 'flex-end', paddingBottom: 6},

  card: {
    backgroundColor: '#fff', padding: 16, borderRadius: 12, marginBottom: 16,
    shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4,
    shadowOffset: {width: 0, height: 2}, elevation: 2,
  },
  warnCard:  {backgroundColor: '#fffbeb'},
  cardTitle: {fontSize: 17, fontWeight: 'bold', marginBottom: 12, color: '#111'},

  coordRow:  {flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6},
  coordLabel:{fontSize: 14, color: '#6b7280'},
  coordValue:{fontSize: 14, fontWeight: '500', color: '#111'},
  coordNote: {fontSize: 11, color: '#9ca3af', marginTop: 8},

  metricRow:   {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#f3f4f6'},
  metricLabel: {fontSize: 14, color: '#374151', flex: 1},
  metricRight: {flexDirection: 'row', alignItems: 'center', gap: 6},
  metricValue: {fontSize: 14, fontWeight: '500', color: '#111'},
  metricBadge: {fontSize: 12},

  scoreBreakdown: {marginTop: 12},
  breakdownTitle: {fontSize: 13, color: '#6b7280', marginBottom: 8},
  breakdownRow:   {flexDirection: 'row', gap: 8},
  barWrap:        {flex: 1, alignItems: 'center'},
  barLabel:       {fontSize: 12, color: '#6b7280', marginBottom: 4},
  barBg:          {width: '100%', height: 6, backgroundColor: '#e5e7eb', borderRadius: 3},
  barFill:        {height: 6, backgroundColor: '#2563eb', borderRadius: 3},
  barPct:         {fontSize: 11, color: '#374151', marginTop: 2},

  altRow:       {flexDirection: 'row', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#f3f4f6'},
  altRankBadge: {width: 32, height: 32, borderRadius: 16, backgroundColor: '#eff6ff', justifyContent: 'center', alignItems: 'center', marginRight: 10},
  altRankText:  {fontSize: 12, fontWeight: 'bold', color: '#2563eb'},
  altInfo:      {flex: 1},
  altCoord:     {fontSize: 13, color: '#374151'},
  altScore:     {fontSize: 12, color: '#9ca3af', marginTop: 2},

  warnLine: {fontSize: 13, color: '#92400e', marginBottom: 4},
  footer:   {fontSize: 12, color: '#9ca3af', textAlign: 'center', marginTop: 8, marginBottom: 20},

  eqBtn:       {backgroundColor: '#10b981', padding: 14, borderRadius: 10, alignItems: 'center', marginBottom: 12},
  homeBtn:     {backgroundColor: '#2563eb', padding: 14, borderRadius: 10, alignItems: 'center'},
  homeBtnText: {color: '#fff', fontSize: 16, fontWeight: '600'},
  preview3dBtn:    {backgroundColor: '#1f2937', paddingVertical: 12, borderRadius: 10, alignItems: 'center', marginBottom: 16},
  preview3dBtnText:{color: '#fff', fontSize: 15, fontWeight: '600'},
});
