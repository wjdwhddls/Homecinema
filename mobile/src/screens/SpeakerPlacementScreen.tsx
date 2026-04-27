/**
 * SpeakerPlacementScreen — 시네마틱 리스킨
 *
 * 디자인 시스템
 *   · 다크 BG (#000) + radial glow + glass card
 *   · 4-stage hairline pipeline (스캔 / 임시배치 / 녹음 / 계산)
 *   · 단계별 액센트: cyan(스캔) → cyan(배치) → orange(녹음) → purple(계산)
 *   · primary glass pill 버튼 + LED dot
 *
 * 2D topview PNG 는 light 톤이 가독성 우위라는 결정에 따라
 * 흰색 PNG 그대로 dark glass card 안에 박아 "갤러리 액자" 컨셉으로 처리.
 *
 * 파이프라인 흐름 (기존 그대로)
 * 1. 방 스캔 (RoomPlan)
 * 2. 임시 스피커 위치 수신 → 표시
 * 3. 사용자가 스피커를 임시 위치에 배치
 * 4. sweep 재생 + 마이크 녹음 → recorded.wav
 * 5. recorded.wav + sweep.wav(번들) → 서버로 전송 → xRIR 최적화
 * 6. 결과 화면으로 이동
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
  Platform,
  Image,
  StatusBar,
  Animated,
  Easing,
  Dimensions,
} from 'react-native';
import Svg, {Defs, RadialGradient, Stop, Rect} from 'react-native-svg';
import {useNavigation, useRoute, RouteProp} from '@react-navigation/native';
import {
  isRoomScanSupported,
  startRoomScan,
  CapturedRoom,
  categoryLabelKR,
} from '../native/RoomScanner';
import {recordSweep, getSweepUri} from '../native/SweepRecorder';
import {showRoomPreview, PREVIEW_COLORS} from '../native/RoomPreview';
import {
  getInitialSpeakerPosition,
  startXRirOptimization,
  waitForXRirOptimization,
  OptimizationAbortedError,
  InitialPositionResponse,
} from '../api/optimization';
import {RootStackParamList} from '../types';

type SpeakerPlacementRouteProp = RouteProp<RootStackParamList, 'SpeakerPlacement'>;

type Step =
  | 'ready'         // 스캔 전
  | 'scanning'      // 스캔 중
  | 'fetchingPos'   // 임시 위치 요청 중
  | 'waitPlacement' // 임시 위치 표시 → 사용자 배치 대기
  | 'recording'     // sweep 재생 + 녹음 중
  | 'optimizing';   // xRIR 추론 중

// ─────────────────────────────────────────────────────────────────
// 메인 스크린
// ─────────────────────────────────────────────────────────────────
export default function SpeakerPlacementScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<SpeakerPlacementRouteProp>();
  const {speakerDimensions} = route.params;

  const [checking, setChecking]       = useState(true);
  const [supported, setSupported]     = useState(false);
  const [step, setStep]               = useState<Step>('ready');
  const [progress, setProgress]       = useState(0);
  const [room, setRoom]               = useState<CapturedRoom | null>(null);
  const [initialPos, setInitialPos]   = useState<InitialPositionResponse | null>(null);

  const abortRef   = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  const fade = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 1000,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [fade]);

  useEffect(() => {
    (async () => {
      const ok = await isRoomScanSupported();
      setSupported(ok);
      setChecking(false);
    })();
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  const safe = <T,>(setter: (v: T) => void) => (v: T) => {
    if (mountedRef.current) setter(v);
  };

  // ── STEP 1: 방 스캔 ─────────────────────────────────────────
  const handleScan = async () => {
    try {
      safe(setStep)('scanning');
      safe(setRoom)(null);
      safe(setInitialPos)(null);

      const scanned = await startRoomScan();
      if (!mountedRef.current) return;
      safe(setRoom)(scanned);
      await fetchInitialPosition(scanned);
    } catch (e: any) {
      if (!mountedRef.current) return;
      safe(setStep)('ready');
      if (e.code === 'CANCELLED' || e.message?.includes('취소')) return;
      if (e.code === 'SCAN_INSUFFICIENT') {
        Alert.alert('스캔이 충분하지 않아요', e.message ?? '구석구석 다시 스캔해주세요.', [{text: '다시 시도'}]);
        return;
      }
      Alert.alert('스캔 실패', e.message ?? '알 수 없는 오류');
    }
  };

  // ── STEP 2: 임시 스피커 위치 요청 ───────────────────────────
  const fetchInitialPosition = async (scanned: CapturedRoom) => {
    try {
      safe(setStep)('fetchingPos');
      const pos = await getInitialSpeakerPosition({
        roomplan_scan: scanned,
        listener_height_m: 1.2,
      });
      if (!mountedRef.current) return;
      safe(setInitialPos)(pos);
      safe(setStep)('waitPlacement');
    } catch (e: any) {
      if (!mountedRef.current) return;
      safe(setStep)('ready');
      Alert.alert('위치 계산 실패', e.message ?? '서버 오류가 발생했습니다.');
    }
  };

  // ── STEP 3: sweep 재생 + 녹음 ───────────────────────────────
  const handleRecord = async () => {
    try {
      safe(setStep)('recording');
      const recordedUri = await recordSweep('sweep');
      if (!mountedRef.current) return;
      const sweepUri = await getSweepUri();
      if (!mountedRef.current) return;

      Alert.alert(
        '녹음 완료',
        '이제 AI가 최적 위치를 계산합니다.',
        [{text: '계산 시작', onPress: () => handleOptimize(recordedUri, sweepUri)}],
      );
    } catch (e: any) {
      if (!mountedRef.current) return;
      safe(setStep)('waitPlacement');
      Alert.alert('녹음 실패', e.message ?? '마이크 권한을 확인해주세요.');
    }
  };

  // ── STEP 4: xRIR 최적화 요청 ────────────────────────────────
  const handleOptimize = async (recordedUri: string, sweepUri: string) => {
    if (!room || !initialPos) return;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      safe(setStep)('optimizing');
      safe(setProgress)(0);

      const {job_id} = await startXRirOptimization(
        room,
        recordedUri,
        sweepUri,
        initialPos.initial_speaker_position,
        {
          listener_height_m: 1.2,
          speaker_dimensions: speakerDimensions,
          top_k: 2,
        },
      );

      const response = await waitForXRirOptimization(
        job_id,
        safe(setProgress),
        3000,
        300_000,
        controller.signal,
      );

      if (!mountedRef.current) return;
      navigation.navigate('OptimizationResult', {
        result: response,
        usdzUri: room.usdzUri,
        speakerDimensions,
      });
    } catch (e: any) {
      if (e instanceof OptimizationAbortedError || controller.signal.aborted) return;
      if (!mountedRef.current) return;
      safe(setStep)('waitPlacement');
      Alert.alert('최적화 실패', e.message ?? '알 수 없는 오류가 발생했습니다.');
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  // ── 로딩/미지원 화면 ─────────────────────────────────────────
  if (checking) {
    return (
      <View style={styles.root}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <BackgroundGlow />
        <SafeAreaView style={styles.centerScreen}>
          <ActivityIndicator size="large" color="#9E7BE0" />
          <Text style={styles.checkingText}>디바이스 호환성 확인 중...</Text>
        </SafeAreaView>
      </View>
    );
  }

  if (!supported) {
    return (
      <View style={styles.root}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <BackgroundGlow />
        <SafeAreaView style={styles.centerScreen}>
          <View style={styles.warnGlassCard}>
            <Text style={styles.warnEyebrow}>UNSUPPORTED</Text>
            <Text style={styles.warnTitle}>지원되지 않는 기기</Text>
            <Text style={styles.warnDesc}>
              {Platform.OS === 'android'
                ? '이 기능은 iOS 전용입니다.'
                : 'LiDAR 가 탑재된 iPhone Pro 또는 iPad Pro 에서만 사용할 수 있습니다.'}
            </Text>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  // ── 메인 화면 ────────────────────────────────────────────────
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
              <Text style={styles.eyebrow}>SPEAKER PLACEMENT</Text>
              <Text style={styles.hero}>
                방의 음향을{'\n'}측정합니다
              </Text>
            </View>

            {/* 입력한 스피커 정보 */}
            <View style={styles.specRow}>
              <Text style={styles.specLabel}>입력한 스피커</Text>
              <Text style={styles.specValue}>
                {speakerDimensions.width_cm} × {speakerDimensions.height_cm} ×{' '}
                {speakerDimensions.depth_cm} cm
              </Text>
            </View>

            {/* 4-step pipeline */}
            <StepIndicator current={step} />

            {/* STEP 1: 스캔 가이드 + 버튼 */}
            {(step === 'ready' || step === 'scanning') && (
              <View style={styles.section}>
                <NoticeCard
                  accent="#3DC8FF"
                  title="스캔 시작 위치 = 청취 위치"
                  desc={
                    '평소 감상하는 자리에 앉아 스크린(TV)을 정확히 바라보며 시작해주세요.\n시작 시 본 방향이 정면(스크린)으로 인식되며, 이 방향이 어긋나면 최적 위치가 부정확해집니다.'
                  }
                />
                <PrimaryGlassButton
                  label={step === 'scanning' ? '스캔 진행 중...' : '방 스캔 시작'}
                  accent="#3DC8FF"
                  loading={step === 'scanning'}
                  disabled={step === 'scanning'}
                  onPress={handleScan}
                />
              </View>
            )}

            {/* 스캔 결과 요약 */}
            {room && step !== 'scanning' && (
              <View style={styles.glassCard}>
                <Text style={styles.cardEyebrow}>SCAN RESULT</Text>
                <View style={styles.summaryRow}>
                  <SummaryItem label="벽" count={room.walls.length} accent="#F5F5F7" />
                  <SummaryItem label="문" count={room.doors.length} accent="#F5F5F7" />
                  <SummaryItem label="창문" count={room.windows.length} accent="#F5F5F7" />
                  <SummaryItem label="가구" count={room.objects.length} accent="#F5F5F7" />
                </View>
                {room.objects.slice(0, 3).map(o => (
                  <View key={o.id} style={styles.objLineRow}>
                    <View style={styles.objLineDot} />
                    <Text style={styles.objLine}>
                      {categoryLabelKR[o.category] ?? o.category}{'  '}
                      <Text style={styles.objLineDim}>
                        {o.dimensions[0].toFixed(1)}×{o.dimensions[1].toFixed(1)} m
                      </Text>
                    </Text>
                  </View>
                ))}
              </View>
            )}

            {/* 임시 위치 계산 중 */}
            {step === 'fetchingPos' && (
              <View style={styles.loadingBox}>
                <ActivityIndicator size="large" color="#3DC8FF" />
                <Text style={styles.loadingText}>임시 스피커 위치 계산 중...</Text>
              </View>
            )}

            {/* STEP 2+3: 임시 위치 표시 + 녹음 버튼 */}
            {(step === 'waitPlacement' || step === 'recording') && initialPos && (
              <View style={styles.section}>
                {/* topview PNG — 흰 배경 그대로, 다크 액자 안에 */}
                {initialPos.topview_image && (
                  <View style={styles.topviewFrame}>
                    <View style={styles.topviewLabelRow}>
                      <View style={styles.topviewDot} />
                      <Text style={styles.topviewLabel}>TOP VIEW</Text>
                    </View>
                    <Image
                      source={{
                        uri: `data:image/png;base64,${initialPos.topview_image}`,
                      }}
                      style={styles.topviewImage}
                      resizeMode="contain"
                    />
                  </View>
                )}

                {/* 3D 미리보기 */}
                {room?.usdzUri && (
                  <TouchableOpacity
                    style={styles.preview3dBtn}
                    activeOpacity={0.7}
                    onPress={async () => {
                      try {
                        await showRoomPreview({
                          usdzUri: room.usdzUri!,
                          listener: initialPos.listener_position,
                          speakers: [
                            {
                              label: '임시 스피커',
                              color: PREVIEW_COLORS.initial,
                              ...initialPos.initial_speaker_position,
                              dimensions: {
                                width_m:  speakerDimensions.width_cm  / 100,
                                height_m: speakerDimensions.height_cm / 100,
                                depth_m:  speakerDimensions.depth_cm  / 100,
                              },
                            },
                          ],
                        });
                      } catch (err: any) {
                        Alert.alert('3D 미리보기 실패', err?.message || '알 수 없는 오류');
                      }
                    }}>
                    <View style={styles.preview3dInner}>
                      <View style={styles.preview3dDot} />
                      <Text style={styles.preview3dText}>3 D 로 자세히 보기</Text>
                    </View>
                  </TouchableOpacity>
                )}

                {/* 임시 위치 좌표 카드 */}
                <View style={styles.posCard}>
                  <Text style={styles.cardEyebrow}>TEMP POSITION</Text>
                  <Text style={styles.posDesc}>
                    아래 좌표에 스피커를 배치한 뒤 sweep 녹음을 시작해주세요.
                  </Text>
                  <CoordRow
                    label="x  좌우"
                    value={initialPos.initial_speaker_position.x}
                    accent="#FF8A5B"
                  />
                  <CoordRow
                    label="y  앞뒤"
                    value={initialPos.initial_speaker_position.y}
                    accent="#9E7BE0"
                  />
                  <Text style={styles.posNote}>청취 위치(원점) 기준 · 단위: m</Text>
                </View>

                <PrimaryGlassButton
                  label={
                    step === 'recording' ? 'sweep 재생 + 녹음 중...' : 'sweep 재생 + 녹음 시작'
                  }
                  accent="#FF8A5B"
                  loading={step === 'recording'}
                  disabled={step === 'recording'}
                  onPress={handleRecord}
                />
                {step === 'recording' && (
                  <Text style={styles.hint}>
                    스피커에서 소리가 나오면 움직이지 마세요. 약 7~8초 소요됩니다.
                  </Text>
                )}
              </View>
            )}

            {/* STEP 4: 최적화 중 */}
            {step === 'optimizing' && (
              <View style={styles.loadingBox}>
                <ActivityIndicator size="large" color="#9E7BE0" />
                <Text style={styles.loadingText}>
                  최적 위치 계산 중...{' '}
                  <Text style={styles.loadingPct}>{progress}%</Text>
                </Text>
                <Text style={styles.hint}>
                  AI 음향 분석 실행 중. 약 1~3분 소요됩니다.
                </Text>

                {/* hairline progress bar */}
                <View style={styles.optProgressBg}>
                  <View
                    style={[styles.optProgressFill, {width: `${progress}%`}]}
                  />
                </View>
              </View>
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

const STEPS = ['스캔', '배치', '녹음', '계산'];
const STEP_ACCENTS = ['#3DC8FF', '#3DC8FF', '#FF8A5B', '#9E7BE0'];

const stepIndex = (s: Step) =>
  ({
    ready: 0,
    scanning: 0,
    fetchingPos: 1,
    waitPlacement: 1,
    recording: 2,
    optimizing: 3,
  }[s] ?? 0);

function StepIndicator({current}: {current: Step}) {
  const cur = stepIndex(current);
  return (
    <View style={styles.stepRow}>
      {STEPS.map((label, i) => {
        const done = i < cur;
        const active = i === cur;
        const accent = STEP_ACCENTS[i];
        return (
          <View key={i} style={styles.stepItem}>
            <View
              style={[
                styles.stepDot,
                done && {backgroundColor: accent + 'AA', borderColor: accent},
                active && {borderColor: accent, backgroundColor: accent},
              ]}>
              <Text
                style={[
                  styles.stepDotText,
                  active && {color: '#0A0A12'},
                  done && {color: '#0A0A12'},
                ]}>
                {done ? '✓' : i + 1}
              </Text>
            </View>
            <Text
              style={[
                styles.stepLabel,
                active && {color: accent, fontWeight: '600'},
              ]}>
              {label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function NoticeCard({
  title,
  desc,
  accent,
}: {
  title: string;
  desc: string;
  accent: string;
}) {
  return (
    <View style={[styles.noticeCard, {borderLeftColor: accent}]}>
      <View style={[styles.noticeDot, {backgroundColor: accent, shadowColor: accent}]} />
      <View style={styles.noticeBody}>
        <Text style={styles.noticeTitle}>{title}</Text>
        <Text style={styles.noticeDesc}>{desc}</Text>
      </View>
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
        {loading && <ActivityIndicator color="#0A0A12" style={{marginRight: 8}} />}
        <Text style={styles.primaryBtnText}>{label}</Text>
        {!loading && <Text style={styles.primaryBtnArrow}>→</Text>}
      </View>
    </TouchableOpacity>
  );
}

function SummaryItem({
  label,
  count,
  accent,
}: {
  label: string;
  count: number;
  accent: string;
}) {
  return (
    <View style={styles.summaryItem}>
      <Text style={[styles.summaryCount, {color: accent}]}>{count}</Text>
      <Text style={styles.summaryLabel}>{label}</Text>
    </View>
  );
}

function CoordRow({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
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
      <Text style={styles.coordValue}>{value.toFixed(2)} m</Text>
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
        <RadialGradient id="placementBg" cx="50%" cy="40%" r="70%">
          <Stop offset="0%" stopColor="#1C1530" stopOpacity="1" />
          <Stop offset="50%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
        <RadialGradient id="placementCyan" cx="50%" cy="32%" r="42%">
          <Stop offset="0%" stopColor="#3DC8FF" stopOpacity="0.04" />
          <Stop offset="100%" stopColor="#3DC8FF" stopOpacity="0" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#placementBg)" />
      <Rect width="100%" height="100%" fill="url(#placementCyan)" />
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
  header: {alignItems: 'center', marginBottom: 16},
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

  // Spec row
  specRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.1)',
    marginTop: 18,
    marginBottom: 22,
  },
  specLabel: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.5)',
    letterSpacing: 1.5,
    fontWeight: '500',
  },
  specValue: {
    fontSize: 14,
    color: '#F5F5F7',
    fontWeight: '400',
    letterSpacing: 0.4,
  },

  // Step indicator
  stepRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    marginBottom: 28,
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

  // Section
  section: {marginBottom: 18},

  // Notice
  noticeCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderLeftWidth: 2,
    borderRadius: 12,
    padding: 14,
    paddingLeft: 16,
    marginBottom: 16,
    gap: 12,
    borderTopWidth: StyleSheet.hairlineWidth * 2,
    borderRightWidth: StyleSheet.hairlineWidth * 2,
    borderBottomWidth: StyleSheet.hairlineWidth * 2,
    borderTopColor: 'rgba(245,245,247,0.08)',
    borderRightColor: 'rgba(245,245,247,0.08)',
    borderBottomColor: 'rgba(245,245,247,0.08)',
  },
  noticeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 5,
    shadowOffset: {width: 0, height: 0},
    shadowOpacity: 0.95,
    shadowRadius: 6,
    elevation: 6,
  },
  noticeBody: {flex: 1},
  noticeTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#F5F5F7',
    marginBottom: 6,
    letterSpacing: 0.3,
  },
  noticeDesc: {
    fontSize: 12.5,
    color: 'rgba(245,245,247,0.65)',
    lineHeight: 19,
    letterSpacing: 0.2,
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
    color: 'rgba(245,245,247,0.45)',
    letterSpacing: 3.5,
    fontWeight: '600',
    marginBottom: 14,
  },

  // Scan summary
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 14,
    paddingTop: 4,
  },
  summaryItem: {alignItems: 'center'},
  summaryCount: {
    fontSize: 26,
    fontWeight: '200',
    letterSpacing: -0.6,
  },
  summaryLabel: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.5)',
    marginTop: 2,
    letterSpacing: 0.6,
  },
  objLineRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
    paddingHorizontal: 4,
  },
  objLineDot: {
    width: 3,
    height: 3,
    borderRadius: 1.5,
    backgroundColor: 'rgba(245,245,247,0.45)',
    marginRight: 10,
  },
  objLine: {
    fontSize: 12.5,
    color: 'rgba(245,245,247,0.7)',
    letterSpacing: 0.3,
  },
  objLineDim: {color: 'rgba(245,245,247,0.42)', fontSize: 11.5},

  // Topview frame (white PNG inside dark glass card)
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

  // 3D preview button (secondary glass pill)
  preview3dBtn: {
    height: 48,
    borderRadius: 100,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.18)',
    backgroundColor: 'rgba(255,255,255,0.04)',
    marginBottom: 16,
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

  // Position card
  posCard: {
    backgroundColor: 'rgba(158,123,224,0.06)',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(158,123,224,0.25)',
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
    letterSpacing: 0.5,
  },
  coordValue: {
    fontSize: 15,
    color: '#F5F5F7',
    fontWeight: '500',
    letterSpacing: 0.3,
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

  // Loading box
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
  loadingPct: {
    color: '#9E7BE0',
    fontWeight: '700',
  },
  hint: {
    marginTop: 10,
    fontSize: 11.5,
    color: 'rgba(245,245,247,0.5)',
    textAlign: 'center',
    letterSpacing: 0.3,
    lineHeight: 17,
  },
  optProgressBg: {
    width: '100%',
    height: 2,
    marginTop: 22,
    backgroundColor: 'rgba(245,245,247,0.08)',
    borderRadius: 1,
    overflow: 'hidden',
  },
  optProgressFill: {
    height: 2,
    backgroundColor: '#9E7BE0',
  },

  // Checking
  checkingText: {
    marginTop: 14,
    color: 'rgba(245,245,247,0.6)',
    fontSize: 13,
    letterSpacing: 0.5,
  },

  // Unsupported
  warnGlassCard: {
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 16,
    padding: 24,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(255,200,80,0.35)',
    alignItems: 'center',
  },
  warnEyebrow: {
    fontSize: 10,
    color: '#FFC850',
    letterSpacing: 5.5,
    fontWeight: '600',
    marginBottom: 12,
  },
  warnTitle: {
    fontSize: 18,
    color: '#F5F5F7',
    fontWeight: '300',
    letterSpacing: -0.3,
    marginBottom: 12,
  },
  warnDesc: {
    fontSize: 13,
    color: 'rgba(245,245,247,0.65)',
    textAlign: 'center',
    lineHeight: 20,
    letterSpacing: 0.2,
  },
});
