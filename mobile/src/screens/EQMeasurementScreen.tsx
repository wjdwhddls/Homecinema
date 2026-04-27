/**
 * EQMeasurementScreen — 시네마틱 리스킨
 *
 * 최적 위치에 스피커 배치 후 EQ 보정 측정 화면.
 *
 * 흐름:
 * 1. 사용자가 스피커를 최적 위치에 배치
 * 2. sweep 재생 + 마이크 녹음 (SweepRecorder)
 * 3. POST /api/eq/analyze → EQ 보정값 수신
 * 4. Bass/Mid/Treble 요약 + 8밴드 결과 표시
 *
 * 디자인 시스템
 *   · 다크 BG + radial glow (오렌지 tint — 측정/액티브)
 *   · 3-step pipeline: 측정(오렌지) / 분석(퍼플) / 결과(시안)
 *   · 8밴드 보정값 = freq label + gain 숫자 + 미니 vertical bar (0dB 기준 +/− 색상 분기)
 *   · MeasuredResponseCurve 2 매 (이미 다크 리스킨 완료) — 측정 / 보정 후
 */
import React, {useEffect, useRef, useState} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
  StatusBar,
  Animated,
  Easing,
} from 'react-native';
import Svg, {Defs, RadialGradient, Stop, Rect} from 'react-native-svg';
import {useNavigation, useRoute, RouteProp} from '@react-navigation/native';
import {recordSweep, getSweepUri} from '../native/SweepRecorder';
import {analyzeEQ, EQAnalysisResponse} from '../api/eq';
import {RootStackParamList} from '../types';
import MeasuredResponseCurve from '../components/MeasuredResponseCurve';

type EQRouteProp = RouteProp<RootStackParamList, 'EQMeasurement'>;

type Step = 'ready' | 'recording' | 'analyzing' | 'done';

const POS_COLOR_LEFT = '#FF8A5B';
const POS_COLOR_RIGHT = '#9E7BE0';
const GAIN_POS = '#FF8A5B'; // 부스트
const GAIN_NEG = '#3DC8FF'; // 컷
const GAIN_FLAT = 'rgba(245,245,247,0.55)';

// ─────────────────────────────────────────────────────────────────
// 메인 스크린
// ─────────────────────────────────────────────────────────────────
export default function EQMeasurementScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<EQRouteProp>();
  const {optimalPosition} = route.params;

  const [step, setStep] = useState<Step>('ready');
  const [result, setResult] = useState<EQAnalysisResponse | null>(null);
  const mountedRef = useRef(true);

  const fade = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 1000,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
    return () => {
      mountedRef.current = false;
    };
  }, [fade]);

  const safe = <T,>(setter: (v: T) => void) => (v: T) => {
    if (mountedRef.current) setter(v);
  };

  const handleMeasure = async () => {
    try {
      safe(setStep)('recording');
      const recordedUri = await recordSweep('sweep');
      if (!mountedRef.current) return;

      const sweepUri = await getSweepUri();
      if (!mountedRef.current) return;

      safe(setStep)('analyzing');
      const eqResult = await analyzeEQ(sweepUri, recordedUri);
      if (!mountedRef.current) return;

      safe(setResult)(eqResult);
      safe(setStep)('done');
    } catch (e: any) {
      if (!mountedRef.current) return;
      safe(setStep)('ready');
      Alert.alert('측정 실패', e.message ?? '오류가 발생했습니다.');
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
              <Text style={styles.eyebrow}>EQ CALIBRATION</Text>
              <Text style={styles.hero}>
                사운드를 측정하고{'\n'}보정합니다
              </Text>
            </View>

            {/* 최적 위치 안내 */}
            <View style={styles.posCard}>
              <Text style={styles.cardEyebrow}>SPEAKER POSITION</Text>
              <Text style={styles.posDesc}>
                아래 좌표에 스피커를 배치한 뒤 측정을 시작해주세요.
              </Text>
              <CoordRow
                label="L  왼쪽"
                x={optimalPosition.left.x}
                y={optimalPosition.left.y}
                accent={POS_COLOR_LEFT}
              />
              <CoordRow
                label="R  오른쪽"
                x={optimalPosition.right.x}
                y={optimalPosition.right.y}
                accent={POS_COLOR_RIGHT}
              />
              <Text style={styles.posNote}>청취 위치 기준 좌우(x), 앞뒤(y) · 단위 m</Text>
            </View>

            {/* 3-step pipeline */}
            <StepIndicator current={step} />

            {/* 측정 버튼 */}
            {(step === 'ready' || step === 'recording') && (
              <>
                <PrimaryGlassButton
                  label={step === 'recording' ? 'sweep 재생 + 녹음 중...' : 'EQ 측정 시작'}
                  accent={GAIN_POS}
                  loading={step === 'recording'}
                  disabled={step === 'recording'}
                  onPress={handleMeasure}
                />
                {step === 'recording' && (
                  <Text style={styles.hint}>
                    스피커에서 소리가 나오면 움직이지 마세요. 약 7~8초 소요됩니다.
                  </Text>
                )}
              </>
            )}

            {/* 분석 중 */}
            {step === 'analyzing' && (
              <View style={styles.loadingBox}>
                <ActivityIndicator size="large" color="#9E7BE0" />
                <Text style={styles.loadingText}>EQ 보정값 계산 중...</Text>
                <Text style={styles.hint}>
                  주파수 응답 분석 + 안전 게인 산출 진행 중.
                </Text>
              </View>
            )}

            {/* 결과 */}
            {step === 'done' && result && (
              <>
                {/* 8밴드 보정값 — 권장 적용값 (가장 먼저) */}
                <View style={styles.glassCard}>
                  <Text style={styles.cardEyebrow}>RECOMMENDED BANDS</Text>
                  <Text style={styles.bandsHint}>
                    아래 값을 EQ 앱 / 리시버에 직접 입력하세요.
                  </Text>
                  <BandsRow bands={result.bands} />
                  <View style={styles.bandsLegendRow}>
                    <View style={styles.bandsLegendItem}>
                      <View style={[styles.legendDot, {backgroundColor: GAIN_POS}]} />
                      <Text style={styles.legendText}>boost</Text>
                    </View>
                    <View style={styles.bandsLegendItem}>
                      <View style={[styles.legendDot, {backgroundColor: GAIN_NEG}]} />
                      <Text style={styles.legendText}>cut</Text>
                    </View>
                    <Text style={styles.legendCaption}>Hz / dB</Text>
                  </View>
                </View>

                {/* 주파수 응답 곡선 */}
                {result.curve && (
                  <>
                    <MeasuredResponseCurve
                      freqs={result.curve.freqs}
                      values_db={result.curve.measured_db}
                      title="측정된 주파수 응답"
                      variant="measured"
                      yRangeDb={result.curve.y_range_db}
                    />
                    <MeasuredResponseCurve
                      freqs={result.curve.freqs}
                      values_db={result.curve.corrected_db}
                      title="EQ 보정 후 (예상)"
                      variant="corrected"
                      yRangeDb={result.curve.y_range_db}
                    />
                  </>
                )}

                {/* Bass / Mid / Treble */}
                <View style={styles.glassCard}>
                  <Text style={styles.cardEyebrow}>SIMPLE EQ</Text>
                  <View style={styles.simpleRow}>
                    {(['bass', 'mid', 'treble'] as const).map((band, idx) => {
                      const info = result.simple[band];
                      const accent =
                        idx === 0 ? '#FF8A5B' : idx === 1 ? '#9E7BE0' : '#3DC8FF';
                      const labelKr =
                        band === 'bass' ? '저음' : band === 'mid' ? '중음' : '고음';
                      const sign = info.gain_db > 0 ? '+' : '';
                      const gainColor =
                        info.gain_db > 1
                          ? GAIN_POS
                          : info.gain_db < -1
                          ? GAIN_NEG
                          : GAIN_FLAT;
                      return (
                        <View key={band} style={styles.simpleItem}>
                          <View
                            style={[
                              styles.simpleDot,
                              {backgroundColor: accent, shadowColor: accent},
                            ]}
                          />
                          <Text style={styles.simpleLabel}>{labelKr}</Text>
                          <Text style={[styles.simpleGain, {color: gainColor}]}>
                            {sign}
                            {info.gain_db.toFixed(1)}
                          </Text>
                          <Text style={styles.simpleUnit}>dB</Text>
                          <Text style={styles.simpleStrength}>{info.label}</Text>
                        </View>
                      );
                    })}
                  </View>
                </View>

                {/* Parametric EQ */}
                {result.parametric.length > 0 && (
                  <View style={styles.glassCard}>
                    <Text style={styles.cardEyebrow}>PARAMETRIC</Text>
                    {result.parametric.map((f, i) => {
                      const isPos = f.gain_db > 0;
                      const accent = isPos ? GAIN_POS : GAIN_NEG;
                      const sign = isPos ? '+' : '';
                      return (
                        <View key={i} style={styles.paramRow}>
                          <View style={styles.paramLeft}>
                            <View
                              style={[
                                styles.paramDot,
                                {backgroundColor: accent, shadowColor: accent},
                              ]}
                            />
                            <Text style={styles.paramFreq}>
                              {f.freq.toLocaleString()}{' '}
                              <Text style={styles.paramUnit}>Hz</Text>
                            </Text>
                          </View>
                          <Text style={[styles.paramGain, {color: accent}]}>
                            {sign}
                            {f.gain_db.toFixed(1)}{' '}
                            <Text style={styles.paramUnit}>dB</Text>
                          </Text>
                          <Text style={styles.paramQ}>
                            Q <Text style={styles.paramQVal}>{f.Q}</Text>
                          </Text>
                        </View>
                      );
                    })}
                  </View>
                )}

                {/* 다시 측정 */}
                <TouchableOpacity
                  style={styles.outlineBtn}
                  activeOpacity={0.7}
                  onPress={() => {
                    setStep('ready');
                    setResult(null);
                  }}>
                  <Text style={styles.outlineBtnText}>다시 측정</Text>
                </TouchableOpacity>

                {/* 홈으로 */}
                <TouchableOpacity
                  style={styles.homeBtn}
                  activeOpacity={0.7}
                  onPress={() => navigation.navigate('Home')}>
                  <Text style={styles.homeBtnText}>홈으로</Text>
                </TouchableOpacity>
              </>
            )}
          </Animated.View>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────
// 서브 컴포넌트
// ─────────────────────────────────────────────────────────────────

const STEPS: Array<{label: string; accent: string}> = [
  {label: '측정', accent: GAIN_POS},
  {label: '분석', accent: '#9E7BE0'},
  {label: '결과', accent: '#3DC8FF'},
];

const stepIndex = (s: Step) =>
  ({ready: 0, recording: 0, analyzing: 1, done: 2}[s] ?? 0);

function StepIndicator({current}: {current: Step}) {
  const cur = stepIndex(current);
  return (
    <View style={styles.stepRow}>
      {STEPS.map((s, i) => {
        const done = i < cur;
        const active = i === cur;
        return (
          <View key={i} style={styles.stepItem}>
            <View
              style={[
                styles.stepDot,
                done && {backgroundColor: s.accent + 'AA', borderColor: s.accent},
                active && {backgroundColor: s.accent, borderColor: s.accent},
              ]}>
              <Text
                style={[
                  styles.stepDotText,
                  (active || done) && {color: '#0A0A12'},
                ]}>
                {done ? '✓' : i + 1}
              </Text>
            </View>
            <Text
              style={[
                styles.stepLabel,
                active && {color: s.accent, fontWeight: '600'},
              ]}>
              {s.label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

interface PrimaryGlassButtonProps {
  label: string;
  accent: string;
  loading?: boolean;
  disabled?: boolean;
  onPress: () => void;
}

function PrimaryGlassButton({
  label,
  accent,
  loading,
  disabled,
  onPress,
}: PrimaryGlassButtonProps) {
  return (
    <TouchableOpacity
      activeOpacity={0.85}
      onPress={onPress}
      disabled={disabled}
      style={[
        styles.primaryBtn,
        {backgroundColor: accent, shadowColor: accent},
        disabled && styles.primaryBtnDim,
      ]}>
      <View style={styles.primaryBtnInner}>
        {loading && (
          <ActivityIndicator color="#0A0A12" style={{marginRight: 8}} />
        )}
        <Text style={styles.primaryBtnText}>{label}</Text>
        {!loading && <Text style={styles.primaryBtnArrow}>→</Text>}
      </View>
    </TouchableOpacity>
  );
}

function CoordRow({
  label,
  x,
  y,
  accent,
}: {
  label: string;
  x: number;
  y: number;
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
        ({x.toFixed(2)}, {y.toFixed(2)})
      </Text>
    </View>
  );
}

// 8 밴드 row — 각 밴드 = freq label + gain 숫자 + mini vertical bar
function BandsRow({bands}: {bands: EQAnalysisResponse['bands']}) {
  // 표시 가능한 max abs (정규화용) — 최소 6dB 보장
  let absMax = 6;
  for (const b of bands) {
    const a = Math.abs(b.actual_gain_db);
    if (a > absMax) absMax = a;
  }
  const BAR_H = 38; // 한 방향 최대 바 높이

  return (
    <View style={styles.bandsRow}>
      {bands.map(b => {
        const g = b.actual_gain_db;
        const isPos = g > 0.05;
        const isNeg = g < -0.05;
        const accent = isPos ? GAIN_POS : isNeg ? GAIN_NEG : GAIN_FLAT;
        const sign = g > 0 ? '+' : '';
        const heightPx =
          Math.min(BAR_H, (Math.abs(g) / absMax) * BAR_H);

        return (
          <View key={b.freq} style={styles.bandCell}>
            {/* 게인 숫자 */}
            <Text style={[styles.bandGain, {color: accent}]}>
              {sign}
              {g.toFixed(1)}
            </Text>

            {/* mini bar — 0dB 가운데 라인에서 +/− 방향 */}
            <View style={styles.bandBarTrack}>
              <View style={styles.bandBarTopHalf}>
                {isPos && (
                  <View
                    style={[
                      styles.bandBarFill,
                      {height: heightPx, backgroundColor: accent},
                    ]}
                  />
                )}
              </View>
              <View style={styles.bandBarMidline} />
              <View style={styles.bandBarBottomHalf}>
                {isNeg && (
                  <View
                    style={[
                      styles.bandBarFill,
                      {height: heightPx, backgroundColor: accent},
                    ]}
                  />
                )}
              </View>
            </View>

            {/* freq label */}
            <Text style={styles.bandFreq}>
              {b.freq >= 1000 ? `${b.freq / 1000}k` : `${b.freq}`}
            </Text>
          </View>
        );
      })}
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
        <RadialGradient id="eqBg" cx="50%" cy="38%" r="70%">
          <Stop offset="0%" stopColor="#1C1530" stopOpacity="1" />
          <Stop offset="50%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
        <RadialGradient id="eqOrangeGlow" cx="50%" cy="28%" r="38%">
          <Stop offset="0%" stopColor="#FF8A5B" stopOpacity="0.05" />
          <Stop offset="100%" stopColor="#FF8A5B" stopOpacity="0" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#eqBg)" />
      <Rect width="100%" height="100%" fill="url(#eqOrangeGlow)" />
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

  // Header
  header: {alignItems: 'center', marginBottom: 22},
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

  // Position card
  posCard: {
    backgroundColor: 'rgba(158,123,224,0.06)',
    borderRadius: 14,
    padding: 16,
    marginBottom: 22,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(158,123,224,0.25)',
  },
  cardEyebrow: {
    fontSize: 9.5,
    color: 'rgba(245,245,247,0.5)',
    letterSpacing: 3.5,
    fontWeight: '600',
    marginBottom: 12,
  },
  posDesc: {
    fontSize: 12.5,
    color: 'rgba(245,245,247,0.7)',
    marginBottom: 14,
    lineHeight: 18,
  },
  posNote: {
    fontSize: 10.5,
    color: 'rgba(245,245,247,0.42)',
    marginTop: 10,
    letterSpacing: 0.4,
  },

  // Coord row
  coordRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 6,
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
    fontSize: 14,
    color: '#F5F5F7',
    fontWeight: '500',
    letterSpacing: 0.3,
    fontVariant: ['tabular-nums'],
  },

  // Step indicator
  stepRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 36,
    marginBottom: 22,
  },
  stepItem: {alignItems: 'center', flex: 1},
  stepDot: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: 'rgba(245,245,247,0.2)',
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 6,
  },
  stepDotText: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.55)',
    fontWeight: '700',
  },
  stepLabel: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.4)',
    letterSpacing: 0.6,
  },

  // Loading
  loadingBox: {
    alignItems: 'center',
    paddingVertical: 36,
    paddingHorizontal: 16,
  },
  loadingText: {
    marginTop: 18,
    fontSize: 14,
    color: '#F5F5F7',
    letterSpacing: 0.4,
  },
  hint: {
    marginTop: 10,
    fontSize: 11.5,
    color: 'rgba(245,245,247,0.5)',
    textAlign: 'center',
    letterSpacing: 0.3,
    lineHeight: 17,
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

  // 8 bands
  bandsHint: {
    fontSize: 12,
    color: 'rgba(245,245,247,0.55)',
    marginBottom: 14,
    lineHeight: 17,
  },
  bandsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingHorizontal: 4,
  },
  bandCell: {
    alignItems: 'center',
    flex: 1,
    minWidth: 0,
  },
  bandGain: {
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 4,
    letterSpacing: 0.2,
    fontVariant: ['tabular-nums'],
  },
  bandBarTrack: {
    width: 4,
    height: 78,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bandBarTopHalf: {
    width: 4,
    flex: 1,
    justifyContent: 'flex-end',
  },
  bandBarMidline: {
    width: 14,
    height: 1,
    backgroundColor: 'rgba(245,245,247,0.22)',
  },
  bandBarBottomHalf: {
    width: 4,
    flex: 1,
    justifyContent: 'flex-start',
  },
  bandBarFill: {
    width: 4,
    borderRadius: 2,
  },
  bandFreq: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.55)',
    marginTop: 6,
    letterSpacing: 0.4,
    fontVariant: ['tabular-nums'],
  },
  bandsLegendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 14,
    gap: 14,
  },
  bandsLegendItem: {flexDirection: 'row', alignItems: 'center', gap: 5},
  legendDot: {width: 6, height: 6, borderRadius: 3},
  legendText: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.55)',
    letterSpacing: 0.5,
  },
  legendCaption: {
    flex: 1,
    textAlign: 'right',
    fontSize: 10,
    color: 'rgba(245,245,247,0.35)',
    letterSpacing: 0.4,
  },

  // Simple EQ (Bass/Mid/Treble)
  simpleRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 4,
  },
  simpleItem: {alignItems: 'center', flex: 1},
  simpleDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginBottom: 8,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 5,
    elevation: 5,
  },
  simpleLabel: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.55)',
    letterSpacing: 1,
    marginBottom: 6,
    fontWeight: '600',
  },
  simpleGain: {
    fontSize: 30,
    fontWeight: '200',
    letterSpacing: -0.8,
    fontVariant: ['tabular-nums'],
  },
  simpleUnit: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.45)',
    letterSpacing: 0.6,
    marginTop: -2,
  },
  simpleStrength: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.42)',
    marginTop: 8,
    letterSpacing: 0.4,
  },

  // Parametric
  paramRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 9,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(245,245,247,0.06)',
  },
  paramLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    width: 110,
  },
  paramDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.9,
    shadowRadius: 4,
    elevation: 4,
  },
  paramFreq: {
    fontSize: 13,
    color: '#F5F5F7',
    fontWeight: '500',
    letterSpacing: 0.3,
    fontVariant: ['tabular-nums'],
  },
  paramUnit: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.45)',
    fontWeight: '400',
    letterSpacing: 0.4,
  },
  paramGain: {
    fontSize: 13.5,
    fontWeight: '700',
    width: 80,
    textAlign: 'center',
    letterSpacing: 0.3,
    fontVariant: ['tabular-nums'],
  },
  paramQ: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.55)',
    width: 50,
    textAlign: 'right',
    letterSpacing: 0.4,
  },
  paramQVal: {
    color: '#F5F5F7',
    fontWeight: '500',
    fontVariant: ['tabular-nums'],
  },

  // Primary glass button (filled)
  primaryBtn: {
    height: 60,
    borderRadius: 100,
    overflow: 'hidden',
    shadowOffset: {width: 0, height: 8},
    shadowOpacity: 0.42,
    shadowRadius: 16,
    elevation: 8,
    marginBottom: 6,
  },
  primaryBtnDim: {opacity: 0.55, shadowOpacity: 0},
  primaryBtnInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  primaryBtnText: {
    fontSize: 15,
    color: '#0A0A12',
    fontWeight: '600',
    letterSpacing: 1,
  },
  primaryBtnArrow: {
    fontSize: 18,
    color: '#0A0A12',
    fontWeight: '300',
  },

  // Outline secondary button
  outlineBtn: {
    height: 56,
    borderRadius: 100,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.18)',
    backgroundColor: 'rgba(255,255,255,0.04)',
    marginTop: 8,
    marginBottom: 12,
  },
  outlineBtnText: {
    color: '#F5F5F7',
    fontSize: 14,
    fontWeight: '500',
    letterSpacing: 1.4,
  },

  // Home button (more transparent, terminal action)
  homeBtn: {
    height: 56,
    borderRadius: 100,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.12)',
    backgroundColor: 'transparent',
    marginBottom: 18,
  },
  homeBtnText: {
    color: 'rgba(245,245,247,0.7)',
    fontSize: 13,
    fontWeight: '500',
    letterSpacing: 1.4,
  },
});
