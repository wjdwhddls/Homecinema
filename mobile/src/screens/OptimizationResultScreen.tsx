/**
 * OptimizationResultScreen — 시네마틱 리스킨
 *
 * xRIR 최적 스피커 배치 결과 표시
 * - left/right 스테레오 좌표
 * - 음향 점수 (T60, C50, EDT)
 * - 대안 위치 목록
 * - EQ 보정 측정으로 이동 버튼
 *
 * 디자인 시스템
 *   · 다크 BG + radial glow + glass cards
 *   · 종합 점수 = 큰 숫자 + 얇은 typography (HomeScreen hero 와 동일 톤)
 *   · 3-색 액센트: T60(오렌지) / C50(퍼플) / EDT(시안)
 *   · 2D topview PNG 는 흰색 그대로 dark 액자 안에 (가독성 우선)
 *   · primary glass pill: EQ 측정으로 진입 (orange)
 */
import React, {useEffect, useRef} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Image,
  Alert,
  StatusBar,
  Animated,
  Easing,
} from 'react-native';
import Svg, {Defs, RadialGradient, Stop, Rect} from 'react-native-svg';
import {useRoute, useNavigation, RouteProp} from '@react-navigation/native';
import {RootStackParamList} from '../types';
import {showRoomPreview, PREVIEW_COLORS} from '../native/RoomPreview';

type OptimizationResultRouteProp = RouteProp<
  RootStackParamList,
  'OptimizationResult'
>;

const safeNum = (n: unknown, fallback = 0): number =>
  typeof n === 'number' && Number.isFinite(n) ? n : fallback;
const clamp = (n: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, n));

// ─────────────────────────────────────────────────────────────────
// 메인 스크린
// ─────────────────────────────────────────────────────────────────
export default function OptimizationResultScreen() {
  const route = useRoute<OptimizationResultRouteProp>();
  const navigation = useNavigation<any>();
  const result = route.params?.result;

  const fade = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 1100,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [fade]);

  if (!result || !result.best) {
    return (
      <View style={styles.root}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <BackgroundGlow />
        <SafeAreaView style={styles.centerScreen}>
          <Text style={styles.emptyEyebrow}>NO RESULT</Text>
          <Text style={styles.emptyTitle}>결과가 없습니다</Text>
          <TouchableOpacity
            style={styles.emptyHomeBtn}
            activeOpacity={0.8}
            onPress={() => navigation.navigate('Home')}>
            <Text style={styles.emptyHomeText}>홈으로</Text>
          </TouchableOpacity>
        </SafeAreaView>
      </View>
    );
  }

  const {
    best,
    top_alternatives = [],
    warnings = [],
    computation_time_seconds,
  } = result;
  const usdzUri = route.params?.usdzUri;
  const speakerDimensions = route.params?.speakerDimensions;

  const scorePercent = clamp(Math.round(safeNum(best.score) * 100), 0, 100);
  const t60 = safeNum(best.metrics?.t60_seconds);
  const c50 = safeNum(best.metrics?.c50_db);
  const edt = safeNum(best.metrics?.edt_seconds);
  const t60Score = safeNum(best.metrics?.t60_score);
  const c50Score = safeNum(best.metrics?.c50_score);
  const edtScore = safeNum(best.metrics?.edt_score);

  const handleShow3D = async () => {
    if (!usdzUri) {
      Alert.alert(
        '3D 미리보기 불가',
        '방 스캔 데이터(USDZ)가 없어 3D 보기를 사용할 수 없습니다.',
      );
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
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <BackgroundGlow />
      <SafeAreaView style={styles.safe}>
        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}>
          <Animated.View style={{opacity: fade}}>
            {/* Header */}
            <View style={styles.header}>
              <Text style={styles.eyebrow}>OPTIMAL PLACEMENT</Text>
              <Text style={styles.hero}>
                최적 스피커{'\n'}위치를 찾았습니다
              </Text>
            </View>

            {/* topview PNG */}
            {result.topview_image && (
              <View style={styles.topviewFrame}>
                <View style={styles.topviewLabelRow}>
                  <View style={styles.topviewDot} />
                  <Text style={styles.topviewLabel}>TOP VIEW</Text>
                </View>
                <Image
                  source={{
                    uri: `data:image/png;base64,${result.topview_image}`,
                  }}
                  style={styles.topviewImage}
                  resizeMode="contain"
                />
              </View>
            )}

            {/* 3D 미리보기 */}
            {usdzUri && (
              <TouchableOpacity
                style={styles.preview3dBtn}
                activeOpacity={0.7}
                onPress={handleShow3D}>
                <View style={styles.preview3dInner}>
                  <View style={styles.preview3dDot} />
                  <Text style={styles.preview3dText}>3 D 로 자세히 보기</Text>
                </View>
              </TouchableOpacity>
            )}

            {/* 종합 점수 */}
            <View style={styles.scoreCard}>
              <Text style={styles.scoreEyebrow}>ACOUSTIC SCORE</Text>
              <View style={styles.scoreRow}>
                <Text style={styles.scoreValue}>{scorePercent}</Text>
                <Text style={styles.scoreUnit}>/ 100</Text>
              </View>
              <View style={styles.scoreBarBg}>
                <View
                  style={[
                    styles.scoreBarFill,
                    {width: `${scorePercent}%`},
                  ]}
                />
              </View>
            </View>

            {/* 권장 배치 좌표 */}
            <View style={styles.glassCard}>
              <Text style={styles.cardEyebrow}>PLACEMENT</Text>
              <CoordRow
                label="왼쪽 스피커"
                pos={best.placement.left}
                accent="#FF8A5B"
              />
              <CoordRow
                label="오른쪽 스피커"
                pos={best.placement.right}
                accent="#9E7BE0"
              />
              <CoordRow
                label="청취 위치"
                pos={best.placement.listener}
                accent="#3DC8FF"
              />
              <Text style={styles.coordNote}>
                좌표 (x, y, z) m · 청취 위치 기준 좌우 / 앞뒤 / 높이
              </Text>
            </View>

            {/* 음향 지표 */}
            <View style={styles.glassCard}>
              <Text style={styles.cardEyebrow}>ACOUSTIC METRICS</Text>

              <MetricRow
                label="잔향 시간 (T60)"
                value={`${t60.toFixed(2)} s`}
                badge={
                  t60 < 0.30
                    ? '짧음'
                    : t60 < 0.50
                    ? '양호'
                    : t60 < 0.70
                    ? '보통'
                    : '길음'
                }
                accent="#FF8A5B"
              />
              <MetricRow
                label="명료도 (C50)"
                value={`${c50.toFixed(1)} dB`}
                badge={
                  c50 >= 5
                    ? '매우 명료'
                    : c50 >= 2
                    ? '양호'
                    : c50 >= -2
                    ? '보통'
                    : '흐림'
                }
                accent="#9E7BE0"
              />
              <MetricRow
                label="초기 감쇠 (EDT)"
                value={`${edt.toFixed(2)} s`}
                badge={
                  edt < 0.25
                    ? '매우 짧음'
                    : edt < 0.45
                    ? '양호'
                    : edt < 0.60
                    ? '보통'
                    : '길음'
                }
                accent="#3DC8FF"
              />

              <Text style={styles.breakdownTitle}>세부 점수</Text>
              <View style={styles.breakdownRow}>
                <ScoreBar
                  label="T60"
                  value={clamp(t60Score, 0, 1)}
                  accent="#FF8A5B"
                />
                <ScoreBar
                  label="C50"
                  value={clamp(c50Score, 0, 1)}
                  accent="#9E7BE0"
                />
                <ScoreBar
                  label="EDT"
                  value={clamp(edtScore, 0, 1)}
                  accent="#3DC8FF"
                />
              </View>
            </View>

            {/* 대안 위치 */}
            {top_alternatives.length > 0 && (
              <View style={styles.glassCard}>
                <Text style={styles.cardEyebrow}>
                  ALTERNATIVES · {top_alternatives.length}
                </Text>
                {top_alternatives.map((alt, i) => {
                  const altScore = clamp(
                    Math.round(safeNum(alt.score) * 100),
                    0,
                    100,
                  );
                  const altT60 = safeNum(alt.metrics?.t60_seconds);
                  const lx = safeNum(alt.placement?.left?.x);
                  const ly = safeNum(alt.placement?.left?.y);
                  const rx = safeNum(alt.placement?.right?.x);
                  const ry = safeNum(alt.placement?.right?.y);
                  return (
                    <View key={i} style={styles.altRow}>
                      <View style={styles.altRankBadge}>
                        <Text style={styles.altRankText}>
                          #{(alt.rank ?? i) + 1}
                        </Text>
                      </View>
                      <View style={styles.altInfo}>
                        <Text style={styles.altCoord}>
                          L ({lx.toFixed(2)}, {ly.toFixed(2)})  ·  R (
                          {rx.toFixed(2)}, {ry.toFixed(2)}) m
                        </Text>
                        <Text style={styles.altScore}>
                          점수 {altScore} · T60 {altT60.toFixed(2)} s
                        </Text>
                      </View>
                    </View>
                  );
                })}
              </View>
            )}

            {/* 경고 */}
            {warnings.length > 0 && (
              <View style={styles.warnCard}>
                <Text style={[styles.cardEyebrow, {color: '#FFC850'}]}>
                  NOTE
                </Text>
                {warnings.map((w, i) => (
                  <View key={i} style={styles.warnRow}>
                    <View style={styles.warnDot} />
                    <Text style={styles.warnLine}>{w}</Text>
                  </View>
                ))}
              </View>
            )}

            <Text style={styles.footer}>
              계산 시간 {(computation_time_seconds ?? 0).toFixed(1)} s
            </Text>

            {/* Primary: EQ 측정 진입 */}
            <TouchableOpacity
              style={styles.eqBtn}
              activeOpacity={0.85}
              onPress={() =>
                navigation.navigate('EQMeasurement', {
                  optimalPosition: best.placement,
                  speakerDimensions,
                })
              }>
              <View style={styles.eqBtnInner}>
                <Text style={styles.eqBtnText}>EQ 자동 보정 측정</Text>
                <Text style={styles.eqBtnArrow}>→</Text>
              </View>
            </TouchableOpacity>

            {/* Secondary: 홈으로 */}
            <TouchableOpacity
              style={styles.homeBtn}
              activeOpacity={0.7}
              onPress={() => navigation.navigate('Home')}>
              <Text style={styles.homeBtnText}>홈으로</Text>
            </TouchableOpacity>
          </Animated.View>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────
// 서브 컴포넌트
// ─────────────────────────────────────────────────────────────────
function CoordRow({
  label,
  pos,
  accent,
}: {
  label: string;
  pos: {x: number; y: number; z: number};
  accent: string;
}) {
  return (
    <View style={styles.coordRow}>
      <View style={styles.coordLeft}>
        <View
          style={[
            styles.coordDot,
            {backgroundColor: accent, shadowColor: accent},
          ]}
        />
        <Text style={styles.coordLabel}>{label}</Text>
      </View>
      <Text style={styles.coordValue}>
        ({pos.x.toFixed(2)}, {pos.y.toFixed(2)}, {pos.z.toFixed(2)})
      </Text>
    </View>
  );
}

function MetricRow({
  label,
  value,
  badge,
  accent,
}: {
  label: string;
  value: string;
  badge?: string;
  accent: string;
}) {
  return (
    <View style={styles.metricRow}>
      <View style={styles.metricLeft}>
        <View
          style={[
            styles.metricDot,
            {backgroundColor: accent, shadowColor: accent},
          ]}
        />
        <Text style={styles.metricLabel}>{label}</Text>
      </View>
      <View style={styles.metricRight}>
        <Text style={styles.metricValue}>{value}</Text>
        {badge ? (
          <View style={[styles.metricBadge, {borderColor: accent + '66'}]}>
            <Text style={[styles.metricBadgeText, {color: accent}]}>
              {badge}
            </Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

function ScoreBar({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
  accent: string;
}) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <View style={styles.barWrap}>
      <Text style={styles.barLabel}>{label}</Text>
      <View style={styles.barBg}>
        <View
          style={[
            styles.barFill,
            {width: `${pct}%`, backgroundColor: accent},
          ]}
        />
      </View>
      <Text style={[styles.barPct, {color: accent}]}>{pct}</Text>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────
// Background glow
// ─────────────────────────────────────────────────────────────────
function BackgroundGlow() {
  return (
    <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
      <Defs>
        <RadialGradient id="resBg" cx="50%" cy="32%" r="70%">
          <Stop offset="0%" stopColor="#1C1530" stopOpacity="1" />
          <Stop offset="50%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
        <RadialGradient id="resPurpleGlow" cx="50%" cy="22%" r="42%">
          <Stop offset="0%" stopColor="#9E7BE0" stopOpacity="0.05" />
          <Stop offset="100%" stopColor="#9E7BE0" stopOpacity="0" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#resBg)" />
      <Rect width="100%" height="100%" fill="url(#resPurpleGlow)" />
    </Svg>
  );
}

// ─────────────────────────────────────────────────────────────────
// 스타일
// ─────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: '#000'},
  safe: {flex: 1},
  scroll: {paddingHorizontal: 24, paddingBottom: 40, paddingTop: 56},

  centerScreen: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 28,
  },

  // Header
  header: {alignItems: 'center', marginBottom: 24},
  eyebrow: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.42)',
    letterSpacing: 5.5,
    fontWeight: '600',
    marginBottom: 18,
  },
  hero: {
    fontSize: 28,
    fontWeight: '200',
    color: '#F5F5F7',
    letterSpacing: -0.5,
    lineHeight: 38,
    textAlign: 'center',
  },

  // Topview frame
  topviewFrame: {
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderRadius: 16,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.14)',
    padding: 12,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 6},
    shadowOpacity: 0.5,
    shadowRadius: 14,
    elevation: 6,
  },
  topviewLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    marginLeft: 4,
    gap: 8,
  },
  topviewDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#9E7BE0',
    shadowColor: '#9E7BE0',
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 5,
    elevation: 5,
  },
  topviewLabel: {
    fontSize: 9.5,
    color: 'rgba(245,245,247,0.55)',
    letterSpacing: 3.5,
    fontWeight: '600',
  },
  topviewImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
  },

  // 3D preview button
  preview3dBtn: {
    height: 48,
    borderRadius: 100,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.18)',
    backgroundColor: 'rgba(255,255,255,0.04)',
    marginBottom: 18,
  },
  preview3dInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  preview3dDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#3DC8FF',
    shadowColor: '#3DC8FF',
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 5,
    elevation: 5,
  },
  preview3dText: {
    fontSize: 13,
    color: '#F5F5F7',
    letterSpacing: 2,
    fontWeight: '500',
  },

  // Score card
  scoreCard: {
    alignItems: 'center',
    paddingVertical: 22,
    paddingHorizontal: 20,
    borderRadius: 18,
    marginBottom: 18,
    backgroundColor: 'rgba(158,123,224,0.08)',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(158,123,224,0.3)',
  },
  scoreEyebrow: {
    fontSize: 9.5,
    color: '#9E7BE0',
    letterSpacing: 4,
    fontWeight: '600',
    marginBottom: 12,
  },
  scoreRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 6,
    marginBottom: 14,
  },
  scoreValue: {
    fontSize: 64,
    fontWeight: '100',
    color: '#F5F5F7',
    letterSpacing: -2,
    lineHeight: 68,
  },
  scoreUnit: {
    fontSize: 16,
    color: 'rgba(245,245,247,0.5)',
    fontWeight: '300',
    paddingBottom: 12,
    letterSpacing: 0.6,
  },
  scoreBarBg: {
    width: '100%',
    height: 2,
    backgroundColor: 'rgba(245,245,247,0.08)',
    borderRadius: 1,
    overflow: 'hidden',
  },
  scoreBarFill: {
    height: 2,
    backgroundColor: '#9E7BE0',
  },

  // Glass card (general)
  glassCard: {
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.12)',
  },
  cardEyebrow: {
    fontSize: 9.5,
    color: 'rgba(245,245,247,0.5)',
    letterSpacing: 3.5,
    fontWeight: '600',
    marginBottom: 14,
  },

  // Coord row
  coordRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 7,
  },
  coordLeft: {flexDirection: 'row', alignItems: 'center', gap: 10},
  coordDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.9,
    shadowRadius: 4,
    elevation: 4,
  },
  coordLabel: {
    fontSize: 13,
    color: 'rgba(245,245,247,0.7)',
    letterSpacing: 0.4,
  },
  coordValue: {
    fontSize: 13.5,
    color: '#F5F5F7',
    fontWeight: '400',
    letterSpacing: 0.3,
    fontVariant: ['tabular-nums'],
  },
  coordNote: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.4)',
    marginTop: 10,
    letterSpacing: 0.3,
  },

  // Metric row
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 9,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(245,245,247,0.06)',
  },
  metricLeft: {flexDirection: 'row', alignItems: 'center', gap: 10, flex: 1},
  metricDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.9,
    shadowRadius: 4,
    elevation: 4,
  },
  metricLabel: {
    fontSize: 13,
    color: 'rgba(245,245,247,0.7)',
    letterSpacing: 0.3,
    flex: 1,
  },
  metricRight: {flexDirection: 'row', alignItems: 'center', gap: 8},
  metricValue: {
    fontSize: 13.5,
    color: '#F5F5F7',
    fontWeight: '500',
    letterSpacing: 0.3,
    fontVariant: ['tabular-nums'],
  },
  metricBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 100,
    borderWidth: 1,
  },
  metricBadgeText: {
    fontSize: 10,
    fontWeight: '600',
    letterSpacing: 0.4,
  },

  // Score breakdown
  breakdownTitle: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.45)',
    letterSpacing: 1,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 10,
  },
  breakdownRow: {
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 4,
  },
  barWrap: {flex: 1, alignItems: 'center'},
  barLabel: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.55)',
    letterSpacing: 0.6,
    marginBottom: 6,
    fontWeight: '600',
  },
  barBg: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(245,245,247,0.08)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  barFill: {height: 4, borderRadius: 2},
  barPct: {
    fontSize: 11,
    marginTop: 4,
    fontWeight: '600',
    letterSpacing: 0.3,
  },

  // Alternatives
  altRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(245,245,247,0.06)',
  },
  altRankBadge: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: 'rgba(158,123,224,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(158,123,224,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  altRankText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#9E7BE0',
    letterSpacing: 0.3,
  },
  altInfo: {flex: 1},
  altCoord: {
    fontSize: 12,
    color: 'rgba(245,245,247,0.75)',
    letterSpacing: 0.2,
    fontVariant: ['tabular-nums'],
  },
  altScore: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.42)',
    marginTop: 2,
    letterSpacing: 0.4,
  },

  // Warning
  warnCard: {
    backgroundColor: 'rgba(255,200,80,0.05)',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(255,200,80,0.3)',
  },
  warnRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 6,
  },
  warnDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#FFC850',
    marginTop: 7,
  },
  warnLine: {
    flex: 1,
    fontSize: 12.5,
    color: 'rgba(245,245,247,0.78)',
    lineHeight: 18,
    letterSpacing: 0.2,
  },

  footer: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.35)',
    textAlign: 'center',
    marginVertical: 14,
    letterSpacing: 0.4,
  },

  // Buttons
  eqBtn: {
    height: 60,
    borderRadius: 100,
    overflow: 'hidden',
    backgroundColor: '#FF8A5B',
    shadowColor: '#FF8A5B',
    shadowOffset: {width: 0, height: 8},
    shadowOpacity: 0.42,
    shadowRadius: 16,
    elevation: 8,
    marginBottom: 12,
  },
  eqBtnInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  eqBtnText: {
    fontSize: 15,
    color: '#0A0A12',
    fontWeight: '600',
    letterSpacing: 1,
  },
  eqBtnArrow: {
    fontSize: 18,
    color: '#0A0A12',
    fontWeight: '300',
  },

  homeBtn: {
    height: 56,
    borderRadius: 100,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.16)',
    backgroundColor: 'rgba(255,255,255,0.03)',
    marginBottom: 18,
  },
  homeBtnText: {
    color: '#F5F5F7',
    fontSize: 14,
    fontWeight: '500',
    letterSpacing: 1.4,
  },

  // Empty state
  emptyEyebrow: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.42)',
    letterSpacing: 5.5,
    fontWeight: '600',
    marginBottom: 14,
  },
  emptyTitle: {
    fontSize: 22,
    color: '#F5F5F7',
    fontWeight: '300',
    letterSpacing: -0.4,
    marginBottom: 28,
  },
  emptyHomeBtn: {
    paddingVertical: 14,
    paddingHorizontal: 38,
    borderRadius: 100,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.16)',
    backgroundColor: 'rgba(255,255,255,0.03)',
  },
  emptyHomeText: {
    color: '#F5F5F7',
    fontSize: 14,
    fontWeight: '500',
    letterSpacing: 1.4,
  },
});
