/**
 * SpeakerPlacementScreen
 *
 * Phase 1: 공간 환경 스캐닝
 * - LiDAR + RoomPlan으로 방의 3D 구조 스캔
 * - 가구/벽/문/창문 자동 분류 및 체적 데이터 수집
 * - 청취 위치를 원점으로 사용 (스캔 시작 위치 = 원점)
 *
 * Phase 2: 임시 스피커 배치 + Sweep 녹음
 * - 서버에서 임시 스피커 위치 계산
 * - 사용자가 임시 위치에 스피커 배치
 * - sweep 재생 + 마이크 녹음 → ref_rir.wav
 *
 * Phase 3: xRIR 추론 → 최적 스피커 위치 반환
 *
 * iOS 전용. Android는 미지원 안내.
 */
import React, { useEffect, useState } from 'react';
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
} from 'react-native';
import {
  isRoomScanSupported,
  startRoomScan,
  CapturedRoom,
  categoryLabelKR,
} from '../native/RoomScanner';
import { recordSweep } from '../native/SweepRecorder';
import {
  getInitialPosition,
  requestSpeakerOptimization,
  waitForJobCompletion,
  Position,
  SpeakerResult,
} from '../api/xrir';

// ──────────────────────────────────────────────
// 단계 정의
// ──────────────────────────────────────────────
type Step =
  | 'idle'             // 시작 전
  | 'scanning'         // 방 스캔 중
  | 'place_initial'    // 임시 위치에 스피커 배치 대기
  | 'recording'        // sweep 재생 + 녹음 중
  | 'optimizing'       // xRIR 추론 중
  | 'done';            // 완료

export default function SpeakerPlacementScreen() {
  const [checking, setChecking] = useState(true);
  const [supported, setSupported] = useState(false);
  const [step, setStep] = useState<Step>('idle');

  const [room, setRoom] = useState<CapturedRoom | null>(null);
  const [initialPos, setInitialPos] = useState<Position | null>(null);
  const [listenerPos, setListenerPos] = useState<Position | null>(null);
  const [optimizeStatus, setOptimizeStatus] = useState('');
  const [bestSpeaker, setBestSpeaker] = useState<SpeakerResult | null>(null);
  const [alternatives, setAlternatives] = useState<SpeakerResult[]>([]);

  useEffect(() => {
    (async () => {
      const ok = await isRoomScanSupported();
      setSupported(ok);
      setChecking(false);
    })();
  }, []);

  // ── Phase 1: 방 스캔 ──────────────────────────
  const handleScan = async () => {
    try {
      setStep('scanning');
      setRoom(null);
      setInitialPos(null);
      setBestSpeaker(null);

      const scannedRoom = await startRoomScan();
      setRoom(scannedRoom);

      // 스캔 완료 즉시 임시 위치 계산 요청
      const posResult = await getInitialPosition(scannedRoom);
      setInitialPos(posResult.initial_speaker_position);
      setListenerPos(posResult.listener_position);

      setStep('place_initial');
    } catch (e: any) {
      setStep('idle');
      if (e.code === 'CANCELLED' || e.message === '사용자가 스캔을 취소했습니다') return;
      if (e.code === 'SCAN_INSUFFICIENT') {
        Alert.alert(
          '스캔이 충분하지 않아요',
          e.message ?? '방을 더 천천히, 구석구석 둘러보며 다시 스캔해주세요.',
          [{ text: '다시 시도' }]
        );
        return;
      }
      Alert.alert('오류', e.message ?? '알 수 없는 오류');
    }
  };

  // ── Phase 2: Sweep 녹음 ───────────────────────
  const handleRecord = async () => {
    if (!room) return;
    try {
      setStep('recording');

      const { uri } = await recordSweep(5, 48000);

      // 녹음 완료 → xRIR 추론 요청
      setStep('optimizing');
      setOptimizeStatus('추론 요청 중...');

      const { job_id } = await requestSpeakerOptimization(room, uri);

      const jobResult = await waitForJobCompletion(
        job_id,
        status => setOptimizeStatus(statusLabel(status)),
      );

      if (jobResult.result) {
        setBestSpeaker(jobResult.result.best);
        setAlternatives(jobResult.result.top_alternatives ?? []);
      }
      setStep('done');
    } catch (e: any) {
      setStep('place_initial');
      Alert.alert('오류', e.message ?? '녹음 또는 추론 중 오류가 발생했습니다.');
    }
  };

  // ──────────────────────────────────────────────
  // 렌더링
  // ──────────────────────────────────────────────
  if (checking) {
    return (
      <SafeAreaView style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.checkingText}>디바이스 호환성 확인 중...</Text>
      </SafeAreaView>
    );
  }

  if (!supported) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.center}>
          <Text style={styles.warnIcon}>⚠️</Text>
          <Text style={styles.warnTitle}>지원되지 않는 기기입니다</Text>
          <Text style={styles.warnDesc}>
            {Platform.OS === 'android'
              ? '이 기능은 현재 iOS 전용입니다. (RoomPlan API)'
              : 'LiDAR가 탑재된 iPhone Pro 모델 또는 iPad Pro에서만 사용할 수 있습니다.'}
          </Text>
          <Text style={styles.warnSub}>
            지원 기기: iPhone 12 Pro 이상의 Pro 모델, iPad Pro (2020년 이후)
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>스피커 위치 자동 배정</Text>

        {/* Step 진행 표시 */}
        <StepIndicator step={step} />

        {/* ── idle / scanning ── */}
        {(step === 'idle' || step === 'scanning') && (
          <>
            <View style={styles.originNotice}>
              <Text style={styles.originIcon}>📍</Text>
              <View style={styles.originTextWrap}>
                <Text style={styles.originTitle}>스캔 시작 위치 = 청취 위치</Text>
                <Text style={styles.originDesc}>
                  평소 음악/영화를 감상하는 자리에 앉아서 시작 버튼을 눌러주세요.
                  그 위치가 음향 분석의 기준점이 됩니다.
                </Text>
              </View>
            </View>

            <Text style={styles.desc}>
              시작 후에는 기기를 천천히 움직이며{'\n'}
              벽, 천장, 가구 전체를 비춰주세요.{'\n'}
              (권장 소요 시간: 1~2분)
            </Text>

            <TouchableOpacity
              style={[styles.primaryBtn, step === 'scanning' && styles.btnDisabled]}
              onPress={handleScan}
              disabled={step === 'scanning'}
            >
              {step === 'scanning'
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.primaryBtnText}>🎯 방 스캔 시작</Text>
              }
            </TouchableOpacity>
          </>
        )}

        {/* ── 스캔 결과 요약 (place_initial 이후 항상 표시) ── */}
        {room && step !== 'idle' && step !== 'scanning' && (
          <View style={styles.resultBox}>
            <Text style={styles.resultTitle}>📊 스캔 결과</Text>
            <View style={styles.summaryRow}>
              <SummaryItem label="벽" count={room.walls.length} />
              <SummaryItem label="문" count={room.doors.length} />
              <SummaryItem label="창문" count={room.windows.length} />
              <SummaryItem label="가구" count={room.objects.length} />
            </View>
          </View>
        )}

        {/* ── place_initial: 임시 위치 안내 ── */}
        {step === 'place_initial' && initialPos && (
          <View style={styles.positionBox}>
            <Text style={styles.positionTitle}>📦 임시 스피커 위치</Text>
            <Text style={styles.positionDesc}>
              아래 위치에 스피커를 배치한 후{'\n'}
              "Sweep 측정 시작" 버튼을 눌러주세요.
            </Text>
            <PositionCard label="임시 위치" pos={initialPos} color="#f59e0b" />
            {listenerPos && (
              <PositionCard label="청취 위치 (기준)" pos={listenerPos} color="#2563eb" />
            )}
            <TouchableOpacity style={styles.primaryBtn} onPress={handleRecord}>
              <Text style={styles.primaryBtnText}>🎙️ Sweep 측정 시작</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* ── recording / optimizing: 진행 중 ── */}
        {(step === 'recording' || step === 'optimizing') && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#2563eb" />
            <Text style={styles.loadingText}>
              {step === 'recording'
                ? '🎙️ Sweep 신호 측정 중...\n5초간 조용히 해주세요.'
                : `🧠 최적 위치 계산 중...\n${optimizeStatus}`}
            </Text>
          </View>
        )}

        {/* ── done: 결과 ── */}
        {step === 'done' && bestSpeaker && (
          <View style={styles.resultBox}>
            <Text style={styles.resultTitle}>✅ 최적 스피커 위치</Text>
            <PositionCard label="최적 위치" pos={bestSpeaker.position} color="#16a34a" />
            <View style={styles.metricsRow}>
              <MetricItem label="RT60" value={`${bestSpeaker.rt60.toFixed(2)}s`} />
              <MetricItem label="C80" value={`${bestSpeaker.c80.toFixed(1)}dB`} />
              <MetricItem label="DRR" value={`${bestSpeaker.drr.toFixed(1)}dB`} />
            </View>

            {alternatives.length > 0 && (
              <>
                <Text style={styles.sectionTitle}>대안 위치</Text>
                {alternatives.map((alt, i) => (
                  <PositionCard
                    key={i}
                    label={`대안 ${i + 1} (점수: ${alt.score.toFixed(2)})`}
                    pos={alt.position}
                    color="#6b7280"
                  />
                ))}
              </>
            )}

            <TouchableOpacity
              style={[styles.primaryBtn, { marginTop: 24 }]}
              onPress={() => setStep('idle')}
            >
              <Text style={styles.primaryBtnText}>🔄 처음부터 다시</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ──────────────────────────────────────────────
// 서브 컴포넌트
// ──────────────────────────────────────────────

const STEPS: { key: Step; label: string }[] = [
  { key: 'scanning',      label: '스캔' },
  { key: 'place_initial', label: '임시 배치' },
  { key: 'recording',     label: '측정' },
  { key: 'optimizing',    label: '계산' },
  { key: 'done',          label: '완료' },
];
const STEP_ORDER: Step[] = ['idle', 'scanning', 'place_initial', 'recording', 'optimizing', 'done'];

function StepIndicator({ step }: { step: Step }) {
  const currentIdx = STEP_ORDER.indexOf(step);
  return (
    <View style={styles.stepRow}>
      {STEPS.map((s, i) => {
        const sIdx = STEP_ORDER.indexOf(s.key);
        const active = sIdx === currentIdx;
        const done = sIdx < currentIdx;
        return (
          <View key={s.key} style={styles.stepItem}>
            <View style={[
              styles.stepDot,
              done && styles.stepDotDone,
              active && styles.stepDotActive,
            ]}>
              <Text style={styles.stepDotText}>{done ? '✓' : i + 1}</Text>
            </View>
            <Text style={[styles.stepLabel, active && styles.stepLabelActive]}>
              {s.label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function SummaryItem({ label, count }: { label: string; count: number }) {
  return (
    <View style={styles.summaryItem}>
      <Text style={styles.summaryCount}>{count}</Text>
      <Text style={styles.summaryLabel}>{label}</Text>
    </View>
  );
}

function PositionCard({ label, pos, color }: { label: string; pos: Position; color: string }) {
  return (
    <View style={[styles.posCard, { borderLeftColor: color }]}>
      <Text style={[styles.posLabel, { color }]}>{label}</Text>
      <Text style={styles.posCoord}>
        X: {pos.x.toFixed(2)}m &nbsp; Y: {pos.y.toFixed(2)}m &nbsp; Z: {pos.z.toFixed(2)}m
      </Text>
    </View>
  );
}

function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.metricItem}>
      <Text style={styles.metricValue}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  );
}

function statusLabel(s: string): string {
  switch (s) {
    case 'pending':    return '대기 중...';
    case 'processing': return '추론 중...';
    case 'completed':  return '완료!';
    case 'failed':     return '실패';
    default:           return s;
  }
}

// ──────────────────────────────────────────────
// 스타일
// ──────────────────────────────────────────────
const styles = StyleSheet.create({
  container:      { flex: 1, backgroundColor: '#fff' },
  center:         { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  checkingText:   { marginTop: 12, color: '#6b7280' },
  content:        { padding: 20, paddingBottom: 48 },
  title:          { fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginVertical: 16 },
  desc:           { fontSize: 15, color: '#555', textAlign: 'center', marginBottom: 28, lineHeight: 22 },

  // Step indicator
  stepRow:        { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 28 },
  stepItem:       { alignItems: 'center', flex: 1 },
  stepDot:        { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e5e7eb', justifyContent: 'center', alignItems: 'center' },
  stepDotActive:  { backgroundColor: '#2563eb' },
  stepDotDone:    { backgroundColor: '#16a34a' },
  stepDotText:    { color: '#fff', fontSize: 12, fontWeight: 'bold' },
  stepLabel:      { fontSize: 11, color: '#9ca3af', marginTop: 4 },
  stepLabelActive:{ color: '#2563eb', fontWeight: '600' },

  // Origin notice
  originNotice:   { flexDirection: 'row', backgroundColor: '#eff6ff', borderLeftWidth: 4, borderLeftColor: '#2563eb', padding: 14, borderRadius: 10, marginBottom: 20 },
  originIcon:     { fontSize: 24, marginRight: 12 },
  originTextWrap: { flex: 1 },
  originTitle:    { fontSize: 15, fontWeight: 'bold', color: '#1e40af', marginBottom: 4 },
  originDesc:     { fontSize: 13, color: '#1e3a8a', lineHeight: 19 },

  // Buttons
  primaryBtn:     { backgroundColor: '#2563eb', padding: 18, borderRadius: 14, alignItems: 'center', marginTop: 8 },
  btnDisabled:    { backgroundColor: '#93c5fd' },
  primaryBtnText: { color: '#fff', fontSize: 17, fontWeight: '600' },

  // Result / position boxes
  resultBox:      { marginTop: 20, padding: 16, backgroundColor: '#f3f4f6', borderRadius: 12 },
  resultTitle:    { fontSize: 18, fontWeight: 'bold', marginBottom: 12 },
  summaryRow:     { flexDirection: 'row', justifyContent: 'space-around', marginBottom: 8 },
  summaryItem:    { alignItems: 'center' },
  summaryCount:   { fontSize: 22, fontWeight: 'bold', color: '#2563eb' },
  summaryLabel:   { fontSize: 13, color: '#6b7280', marginTop: 2 },
  sectionTitle:   { fontSize: 15, fontWeight: '600', marginTop: 16, marginBottom: 8, color: '#374151' },

  positionBox:    { marginTop: 20, padding: 16, backgroundColor: '#fffbeb', borderRadius: 12, borderWidth: 1, borderColor: '#fcd34d' },
  positionTitle:  { fontSize: 18, fontWeight: 'bold', marginBottom: 8, color: '#92400e' },
  positionDesc:   { fontSize: 14, color: '#78350f', marginBottom: 16, lineHeight: 20 },
  posCard:        { backgroundColor: '#fff', borderLeftWidth: 4, padding: 12, borderRadius: 8, marginBottom: 10 },
  posLabel:       { fontSize: 13, fontWeight: '700', marginBottom: 4 },
  posCoord:       { fontSize: 14, color: '#374151', fontFamily: 'Menlo' },

  // Metrics
  metricsRow:     { flexDirection: 'row', justifyContent: 'space-around', marginTop: 12 },
  metricItem:     { alignItems: 'center' },
  metricValue:    { fontSize: 18, fontWeight: 'bold', color: '#16a34a' },
  metricLabel:    { fontSize: 12, color: '#6b7280', marginTop: 2 },

  // Loading
  loadingBox:     { marginTop: 40, alignItems: 'center', padding: 24 },
  loadingText:    { marginTop: 20, fontSize: 16, color: '#374151', textAlign: 'center', lineHeight: 24 },

  // Warn
  warnIcon:       { fontSize: 48, marginBottom: 16 },
  warnTitle:      { fontSize: 18, fontWeight: 'bold', color: '#b45309', marginBottom: 12 },
  warnDesc:       { fontSize: 15, color: '#374151', textAlign: 'center', lineHeight: 22 },
  warnSub:        { fontSize: 13, color: '#6b7280', textAlign: 'center', marginTop: 16 },
});


// /**
//  * SpeakerPlacementScreen
//  *
//  * Phase 1: 공간 환경 스캐닝
//  * - LiDAR + RoomPlan으로 방의 3D 구조 스캔
//  * - 가구/벽/문/창문 자동 분류 및 체적 데이터 수집
//  * - 청취 위치를 원점으로 사용 (스캔 시작 위치 = 원점)
//  *
//  * iOS 전용. Android는 미지원 안내.
//  */
// import React, { useEffect, useRef, useState } from 'react';
// import {
//   SafeAreaView,
//   View,
//   Text,
//   TouchableOpacity,
//   ActivityIndicator,
//   Alert,
//   StyleSheet,
//   ScrollView,
//   Platform,
// } from 'react-native';
// import {useNavigation, useRoute, RouteProp} from '@react-navigation/native';
// import {
//   isRoomScanSupported,
//   startRoomScan,
//   CapturedRoom,
//   categoryLabelKR,
// } from '../native/RoomScanner';
// import {
//   startOptimization,
//   waitForOptimization,
//   OptimizationAbortedError,
// } from '../api/optimization';
// import {RootStackParamList} from '../types';

// type SpeakerPlacementRouteProp = RouteProp<RootStackParamList, 'SpeakerPlacement'>;

// export default function SpeakerPlacementScreen() {
//   const navigation = useNavigation<any>();
//   const route = useRoute<SpeakerPlacementRouteProp>();
//   const {speakerDimensions} = route.params;
//   const [checking, setChecking] = useState(true);
//   const [supported, setSupported] = useState(false);
//   const [scanning, setScanning] = useState(false);
//   const [result, setResult] = useState<CapturedRoom | null>(null);
//   const [optimizing, setOptimizing] = useState(false);
//   const [optimizationProgress, setOptimizationProgress] = useState(0);
//   const abortRef = useRef<AbortController | null>(null);
//   const mountedRef = useRef(true);

//   useEffect(() => {
//     (async () => {
//       const ok = await isRoomScanSupported();
//       setSupported(ok);
//       setChecking(false);
//     })();
//   }, []);

//   useEffect(() => {
//     return () => {
//       mountedRef.current = false;
//       abortRef.current?.abort();
//     };
//   }, []);

//   const safeSetOptimizing = (v: boolean) => {
//     if (mountedRef.current) setOptimizing(v);
//   };
//   const safeSetProgress = (v: number) => {
//     if (mountedRef.current) setOptimizationProgress(v);
//   };

//   const handleOptimize = async () => {
//     if (!result) return;
//     const controller = new AbortController();
//     abortRef.current = controller;
//     try {
//       safeSetOptimizing(true);
//       safeSetProgress(0);

//       const {job_id} = await startOptimization({
//         roomplan_scan: result,
//         speaker_dimensions: {
//           width_m: speakerDimensions.width_cm / 100,
//           height_m: speakerDimensions.height_cm / 100,
//           depth_m: speakerDimensions.depth_cm / 100,
//         },
//         listener_height_m: 1.2,
//         config_type: 'stereo',
//       });

//       const response = await waitForOptimization(
//         job_id,
//         safeSetProgress,
//         3000,
//         300_000,
//         controller.signal,
//       );

//       if (!mountedRef.current) return;
//       navigation.navigate('OptimizationResult', {result: response});
//     } catch (e: any) {
//       if (e instanceof OptimizationAbortedError || controller.signal.aborted) {
//         return; // 사용자 취소/언마운트: 조용히 종료
//       }
//       if (!mountedRef.current) return;
//       Alert.alert('최적화 실패', e.message ?? '알 수 없는 오류가 발생했습니다.');
//     } finally {
//       if (abortRef.current === controller) {
//         abortRef.current = null;
//       }
//       safeSetOptimizing(false);
//       safeSetProgress(0);
//     }
//   };

//   const handleScan = async () => {
//     try {
//       setScanning(true);
//       setResult(null);
//       const room = await startRoomScan();
//       console.log('=== ROOM SCAN RESULT (JS) ===');
//       console.log(JSON.stringify(room, null, 2));
//       setResult(room);
//     } catch (e: any) {
//       // 사용자 취소는 조용히 무시
//       if (e.code === 'CANCELLED' || e.message === '사용자가 스캔을 취소했습니다') {
//         return;
//       }
//       // 스캔 데이터 부족 - 재시도 유도
//       if (e.code === 'SCAN_INSUFFICIENT') {
//         Alert.alert(
//           '스캔이 충분하지 않아요',
//           e.message ?? '방을 더 천천히, 구석구석 둘러보며 다시 스캔해주세요.',
//           [{ text: '다시 시도', style: 'default' }]
//         );
//         return;
//       }
//       // 기타 에러
//       Alert.alert('스캔 실패', e.message ?? '알 수 없는 오류');
//     } finally {
//       setScanning(false);
//     }
//   };

//   if (checking) {
//     return (
//       <SafeAreaView style={styles.center}>
//         <ActivityIndicator size="large" />
//         <Text style={styles.checkingText}>디바이스 호환성 확인 중...</Text>
//       </SafeAreaView>
//     );
//   }

//   if (!supported) {
//     return (
//       <SafeAreaView style={styles.container}>
//         <View style={styles.center}>
//           <Text style={styles.warnIcon}>⚠️</Text>
//           <Text style={styles.warnTitle}>지원되지 않는 기기입니다</Text>
//           <Text style={styles.warnDesc}>
//             {Platform.OS === 'android'
//               ? '이 기능은 현재 iOS 전용입니다. (RoomPlan API)'
//               : 'LiDAR가 탑재된 iPhone Pro 모델 또는 iPad Pro에서만 사용할 수 있습니다.'}
//           </Text>
//           <Text style={styles.warnSub}>
//             지원 기기: iPhone 12 Pro 이상의 Pro 모델, iPad Pro (2020년 이후)
//           </Text>
//         </View>
//       </SafeAreaView>
//     );
//   }

//   return (
//     <SafeAreaView style={styles.container}>
//       <ScrollView contentContainerStyle={styles.content}>
//         <Text style={styles.title}>공간 환경 스캐닝</Text>

//         <View style={styles.speakerInfo}>
//           <Text style={styles.speakerInfoLabel}>입력한 스피커</Text>
//           <Text style={styles.speakerInfoValue}>
//             {speakerDimensions.width_cm} × {speakerDimensions.height_cm} × {speakerDimensions.depth_cm} cm
//           </Text>
//         </View>

//         {/* 청취 위치 = 좌표 원점이라는 핵심 UX를 강조 */}
//         <View style={styles.originNotice}>
//           <Text style={styles.originIcon}>📍</Text>
//           <View style={styles.originTextWrap}>
//             <Text style={styles.originTitle}>스캔 시작 위치 = 청취 위치</Text>
//             <Text style={styles.originDesc}>
//               평소 음악/영화를 감상하는 자리(소파 가운데 등)에 앉아서 시작 버튼을 눌러주세요.
//               그 위치가 음향 분석의 기준점이 됩니다.
//             </Text>
//           </View>
//         </View>

//         <Text style={styles.desc}>
//           시작 후에는 기기를 천천히 움직이며{'\n'}
//           벽, 천장, 가구 전체를 비춰주세요.{'\n'}
//           (권장 소요 시간: 1~2분)
//         </Text>

//         <TouchableOpacity
//           style={[styles.scanBtn, scanning && styles.scanBtnDisabled]}
//           onPress={handleScan}
//           disabled={scanning}
//         >
//           <Text style={styles.scanBtnText}>
//             {scanning ? '스캔 진행 중...' : '🎯 방 스캔 시작'}
//           </Text>
//         </TouchableOpacity>

//         {result && (
//           <View style={styles.resultBox}>
//             <Text style={styles.resultTitle}>📊 스캔 결과</Text>
//             <View style={styles.summaryRow}>
//               <SummaryItem label="벽" count={result.walls.length} />
//               <SummaryItem label="문" count={result.doors.length} />
//               <SummaryItem label="창문" count={result.windows.length} />
//               <SummaryItem label="가구" count={result.objects.length} />
//             </View>

//             {result.objects.length > 0 && (
//               <>
//                 <Text style={styles.sectionTitle}>인식된 가구/객체</Text>
//                 {result.objects.map(o => (
//                   <View key={o.id} style={styles.objRow}>
//                     <Text style={styles.objCategory}>
//                       • {categoryLabelKR[o.category] ?? o.category}
//                     </Text>
//                     <Text style={styles.objDim}>
//                       {o.dimensions[0].toFixed(2)} × {o.dimensions[1].toFixed(2)} × {o.dimensions[2].toFixed(2)} m
//                     </Text>
//                     <Text style={styles.objConf}>신뢰도: {o.confidence}</Text>
//                   </View>
//                 ))}
//               </>
//             )}

//             <Text style={styles.timestamp}>
//               스캔 완료: {new Date(result.scannedAt).toLocaleString('ko-KR')}
//             </Text>
//           </View>
//         )}

//         {result && (
//           <View style={styles.optimizeSection}>
//             <TouchableOpacity
//               style={[styles.optimizeBtn, optimizing && styles.optimizeBtnDisabled]}
//               onPress={handleOptimize}
//               disabled={optimizing}>
//               {optimizing ? (
//                 <>
//                   <ActivityIndicator color="#fff" />
//                   <Text style={styles.optimizeBtnText}>
//                     {'  '}최적 위치 계산 중... {optimizationProgress}%
//                   </Text>
//                 </>
//               ) : (
//                 <Text style={styles.optimizeBtnText}>
//                   🎯 스피커 최적 위치 계산하기
//                 </Text>
//               )}
//             </TouchableOpacity>
//             <Text style={styles.optimizeHint}>
//               서버에서 음향 시뮬레이션을 실행합니다. 약 1~3분 소요됩니다.
//             </Text>
//           </View>
//         )}
//       </ScrollView>
//     </SafeAreaView>
//   );
// }

// const SummaryItem = ({ label, count }: { label: string; count: number }) => (
//   <View style={styles.summaryItem}>
//     <Text style={styles.summaryCount}>{count}</Text>
//     <Text style={styles.summaryLabel}>{label}</Text>
//   </View>
// );

// const styles = StyleSheet.create({
//   container: { flex: 1, backgroundColor: '#fff' },
//   center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
//   checkingText: { marginTop: 12, color: '#6b7280' },
//   content: { padding: 20 },
//   title: { fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginVertical: 16 },
//   desc: {
//     fontSize: 15,
//     color: '#555',
//     textAlign: 'center',
//     marginBottom: 28,
//     lineHeight: 22,
//   },
//   originNotice: {
//     flexDirection: 'row',
//     backgroundColor: '#eff6ff',
//     borderLeftWidth: 4,
//     borderLeftColor: '#2563eb',
//     padding: 14,
//     borderRadius: 10,
//     marginBottom: 20,
//   },
//   originIcon: { fontSize: 24, marginRight: 12 },
//   originTextWrap: { flex: 1 },
//   originTitle: { fontSize: 15, fontWeight: 'bold', color: '#1e40af', marginBottom: 4 },
//   originDesc: { fontSize: 13, color: '#1e3a8a', lineHeight: 19 },
//   scanBtn: {
//     backgroundColor: '#2563eb',
//     padding: 18,
//     borderRadius: 14,
//     alignItems: 'center',
//   },
//   scanBtnDisabled: { backgroundColor: '#93c5fd' },
//   scanBtnText: { color: '#fff', fontSize: 17, fontWeight: '600' },
//   resultBox: {
//     marginTop: 24,
//     padding: 16,
//     backgroundColor: '#f3f4f6',
//     borderRadius: 12,
//   },
//   resultTitle: { fontSize: 18, fontWeight: 'bold', marginBottom: 12 },
//   summaryRow: { flexDirection: 'row', justifyContent: 'space-around', marginBottom: 16 },
//   summaryItem: { alignItems: 'center' },
//   summaryCount: { fontSize: 22, fontWeight: 'bold', color: '#2563eb' },
//   summaryLabel: { fontSize: 13, color: '#6b7280', marginTop: 2 },
//   sectionTitle: { fontSize: 15, fontWeight: '600', marginTop: 8, marginBottom: 8, color: '#374151' },
//   objRow: {
//     backgroundColor: '#fff',
//     padding: 10,
//     borderRadius: 8,
//     marginBottom: 6,
//   },
//   objCategory: { fontSize: 15, fontWeight: '600' },
//   objDim: { fontSize: 13, color: '#374151', marginTop: 2 },
//   objConf: { fontSize: 12, color: '#9ca3af', marginTop: 2 },
//   timestamp: { fontSize: 11, color: '#9ca3af', textAlign: 'right', marginTop: 12 },
//   warnIcon: { fontSize: 48, marginBottom: 16 },
//   warnTitle: { fontSize: 18, fontWeight: 'bold', color: '#b45309', marginBottom: 12 },
//   warnDesc: { fontSize: 15, color: '#374151', textAlign: 'center', lineHeight: 22 },
//   warnSub: { fontSize: 13, color: '#6b7280', textAlign: 'center', marginTop: 16 },
//   optimizeSection: { marginTop: 20 },
//   optimizeBtn: {
//     backgroundColor: '#10b981',
//     padding: 18,
//     borderRadius: 14,
//     alignItems: 'center',
//     flexDirection: 'row',
//     justifyContent: 'center',
//   },
//   optimizeBtnDisabled: { backgroundColor: '#6ee7b7' },
//   optimizeBtnText: { color: '#fff', fontSize: 17, fontWeight: '600' },
//   optimizeHint: {
//     marginTop: 10,
//     fontSize: 12,
//     color: '#6b7280',
//     textAlign: 'center',
//   },
//   speakerInfo: {
//     flexDirection: 'row',
//     justifyContent: 'space-between',
//     alignItems: 'center',
//     backgroundColor: '#f3f4f6',
//     padding: 12,
//     borderRadius: 10,
//     marginBottom: 16,
//   },
//   speakerInfoLabel: {fontSize: 13, color: '#6b7280', fontWeight: '500'},
//   speakerInfoValue: {fontSize: 15, color: '#111827', fontWeight: '600'},
// });
