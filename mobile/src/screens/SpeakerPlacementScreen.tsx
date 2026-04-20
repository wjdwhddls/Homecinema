/**
 * SpeakerPlacementScreen
 *
 * 전체 흐름:
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
} from 'react-native';
import {useNavigation, useRoute, RouteProp} from '@react-navigation/native';
import {
  isRoomScanSupported,
  startRoomScan,
  CapturedRoom,
  categoryLabelKR,
} from '../native/RoomScanner';
import {recordSweep, getSweepUri} from '../native/SweepRecorder';
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

      // 스캔 완료 즉시 임시 위치 요청
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
        speaker_height_m: 1.2,
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

      // sweep.wav 재생 + 동시 마이크 녹음
      const recordedUri = await recordSweep('sweep');
      if (!mountedRef.current) return;

      // 번들의 sweep.wav URI도 가져오기 (서버 deconvolution용)
      const sweepUri = await getSweepUri();
      if (!mountedRef.current) return;

      Alert.alert(
        '녹음 완료 ✅',
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
    if (!room) return;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      safe(setStep)('optimizing');
      safe(setProgress)(0);

      // multipart/form-data로 전송
      //   recorded = 마이크 녹음 wav
      //   sweep    = 번들 원본 sweep wav (서버가 deconvolution에 사용)
      const {job_id} = await startXRirOptimization(
        room,
        recordedUri,
        sweepUri,
        {
          listener_height_m: 1.2,
          speaker_height_m: 1.2,
          top_k: 5,
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
      navigation.navigate('OptimizationResult', {result: response});
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
      <SafeAreaView style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.subText}>디바이스 호환성 확인 중...</Text>
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
              ? '이 기능은 iOS 전용입니다.'
              : 'LiDAR가 탑재된 iPhone Pro 또는 iPad Pro에서만 사용할 수 있습니다.'}
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // ── 메인 화면 ────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>스피커 자동 배치</Text>

        {/* 스피커 정보 */}
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>입력한 스피커</Text>
          <Text style={styles.infoValue}>
            {speakerDimensions.width_cm} × {speakerDimensions.height_cm} × {speakerDimensions.depth_cm} cm
          </Text>
        </View>

        {/* 단계 표시 */}
        <StepIndicator current={step} />

        {/* STEP 1: 스캔 버튼 */}
        {(step === 'ready' || step === 'scanning') && (
          <View style={styles.section}>
            <View style={styles.notice}>
              <Text style={styles.noticeIcon}>📍</Text>
              <View style={styles.noticeText}>
                <Text style={styles.noticeTitle}>스캔 시작 위치 = 청취 위치</Text>
                <Text style={styles.noticeDesc}>
                  평소 감상하는 자리에 앉아서 버튼을 눌러주세요.
                </Text>
              </View>
            </View>
            <TouchableOpacity
              style={[styles.btn, styles.btnBlue, step === 'scanning' && styles.btnDisabled]}
              onPress={handleScan}
              disabled={step === 'scanning'}>
              {step === 'scanning' && <ActivityIndicator color="#fff" />}
              <Text style={styles.btnText}>
                {step === 'scanning' ? '  스캔 진행 중...' : '🎯 방 스캔 시작'}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        {/* 스캔 결과 요약 */}
        {room && step !== 'scanning' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>📊 스캔 결과</Text>
            <View style={styles.summaryRow}>
              <SummaryItem label="벽" count={room.walls.length} />
              <SummaryItem label="문" count={room.doors.length} />
              <SummaryItem label="창문" count={room.windows.length} />
              <SummaryItem label="가구" count={room.objects.length} />
            </View>
            {room.objects.slice(0, 3).map(o => (
              <Text key={o.id} style={styles.objLine}>
                • {categoryLabelKR[o.category] ?? o.category}{' '}
                {o.dimensions[0].toFixed(1)}×{o.dimensions[1].toFixed(1)}m
              </Text>
            ))}
          </View>
        )}

        {/* 임시 위치 계산 중 */}
        {step === 'fetchingPos' && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#2563eb" />
            <Text style={styles.loadingText}>임시 스피커 위치 계산 중...</Text>
          </View>
        )}

        {/* STEP 2+3: 임시 위치 표시 + 녹음 버튼 */}
        {(step === 'waitPlacement' || step === 'recording') && initialPos && (
          <View style={styles.section}>
            <View style={styles.posCard}>
              <Text style={styles.cardTitle}>📌 임시 스피커 위치</Text>
              <Text style={styles.posDesc}>
                아래 좌표에 스피커를 배치한 후 sweep 녹음을 시작해주세요.
              </Text>
              <CoordRow label="x  (좌우)" value={initialPos.initial_speaker_position.x} />
              <CoordRow label="y  (앞뒤)" value={initialPos.initial_speaker_position.y} />
              <CoordRow label="z  (높이)" value={initialPos.initial_speaker_position.z} />
              <Text style={styles.posNote}>청취 위치(원점) 기준 · 단위: m</Text>
            </View>

            <TouchableOpacity
              style={[styles.btn, styles.btnGreen, step === 'recording' && styles.btnDisabled]}
              onPress={handleRecord}
              disabled={step === 'recording'}>
              {step === 'recording' && <ActivityIndicator color="#fff" />}
              <Text style={styles.btnText}>
                {step === 'recording' ? '  sweep 재생 + 녹음 중...' : '🎙️ sweep 재생 + 녹음 시작'}
              </Text>
            </TouchableOpacity>
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
            <ActivityIndicator size="large" color="#10b981" />
            <Text style={styles.loadingText}>
              최적 위치 계산 중... {progress}%
            </Text>
            <Text style={styles.hint}>AI 음향 분석 실행 중. 약 1~3분 소요됩니다.</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ── 서브 컴포넌트 ────────────────────────────────────────────────

const STEPS = ['① 스캔', '② 임시 배치', '③ 녹음', '④ 계산'];

const stepIndex = (s: Step) =>
  ({ready: 0, scanning: 0, fetchingPos: 1, waitPlacement: 1, recording: 2, optimizing: 3}[s] ?? 0);

const StepIndicator = ({current}: {current: Step}) => {
  const cur = stepIndex(current);
  return (
    <View style={styles.stepRow}>
      {STEPS.map((label, i) => (
        <View key={i} style={styles.stepItem}>
          <View style={[styles.stepDot, i < cur && styles.stepDone, i === cur && styles.stepActive]}>
            <Text style={styles.stepDotText}>{i < cur ? '✓' : i + 1}</Text>
          </View>
          <Text style={[styles.stepLabel, i === cur && styles.stepLabelActive]}>{label}</Text>
        </View>
      ))}
    </View>
  );
};

const SummaryItem = ({label, count}: {label: string; count: number}) => (
  <View style={styles.summaryItem}>
    <Text style={styles.summaryCount}>{count}</Text>
    <Text style={styles.summaryLabel}>{label}</Text>
  </View>
);

const CoordRow = ({label, value}: {label: string; value: number}) => (
  <View style={styles.coordRow}>
    <Text style={styles.coordLabel}>{label}</Text>
    <Text style={styles.coordValue}>{value.toFixed(2)} m</Text>
  </View>
);

// ── 스타일 ───────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container:       {flex: 1, backgroundColor: '#fff'},
  center:          {flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24},
  content:         {padding: 20, paddingBottom: 40},
  title:           {fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginVertical: 16},
  subText:         {marginTop: 12, color: '#6b7280'},
  infoRow:         {flexDirection: 'row', justifyContent: 'space-between', backgroundColor: '#f3f4f6', padding: 12, borderRadius: 10, marginBottom: 16},
  infoLabel:       {fontSize: 13, color: '#6b7280', fontWeight: '500'},
  infoValue:       {fontSize: 15, color: '#111827', fontWeight: '600'},
  stepRow:         {flexDirection: 'row', justifyContent: 'space-around', marginBottom: 24},
  stepItem:        {alignItems: 'center'},
  stepDot:         {width: 28, height: 28, borderRadius: 14, backgroundColor: '#e5e7eb', justifyContent: 'center', alignItems: 'center', marginBottom: 4},
  stepDone:        {backgroundColor: '#10b981'},
  stepActive:      {backgroundColor: '#2563eb'},
  stepDotText:     {fontSize: 12, color: '#fff', fontWeight: 'bold'},
  stepLabel:       {fontSize: 11, color: '#9ca3af'},
  stepLabelActive: {color: '#2563eb', fontWeight: '600'},
  section:         {marginBottom: 20},
  notice:          {flexDirection: 'row', backgroundColor: '#eff6ff', borderLeftWidth: 4, borderLeftColor: '#2563eb', padding: 14, borderRadius: 10, marginBottom: 16},
  noticeIcon:      {fontSize: 24, marginRight: 12},
  noticeText:      {flex: 1},
  noticeTitle:     {fontSize: 15, fontWeight: 'bold', color: '#1e40af', marginBottom: 4},
  noticeDesc:      {fontSize: 13, color: '#1e3a8a', lineHeight: 19},
  btn:             {padding: 18, borderRadius: 14, alignItems: 'center', flexDirection: 'row', justifyContent: 'center'},
  btnBlue:         {backgroundColor: '#2563eb'},
  btnGreen:        {backgroundColor: '#10b981'},
  btnDisabled:     {opacity: 0.6},
  btnText:         {color: '#fff', fontSize: 17, fontWeight: '600'},
  card:            {backgroundColor: '#f3f4f6', borderRadius: 12, padding: 16, marginBottom: 16},
  posCard:         {backgroundColor: '#f0fdf4', borderRadius: 12, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: '#86efac'},
  cardTitle:       {fontSize: 17, fontWeight: 'bold', marginBottom: 10},
  posDesc:         {fontSize: 13, color: '#374151', marginBottom: 12},
  posNote:         {fontSize: 11, color: '#9ca3af', marginTop: 8},
  coordRow:        {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4},
  coordLabel:      {fontSize: 14, color: '#6b7280'},
  coordValue:      {fontSize: 14, fontWeight: '600', color: '#111'},
  summaryRow:      {flexDirection: 'row', justifyContent: 'space-around', marginBottom: 12},
  summaryItem:     {alignItems: 'center'},
  summaryCount:    {fontSize: 22, fontWeight: 'bold', color: '#2563eb'},
  summaryLabel:    {fontSize: 13, color: '#6b7280', marginTop: 2},
  objLine:         {fontSize: 13, color: '#374151', marginBottom: 2},
  loadingBox:      {alignItems: 'center', padding: 32},
  loadingText:     {marginTop: 16, fontSize: 16, color: '#374151', fontWeight: '500'},
  hint:            {marginTop: 10, fontSize: 12, color: '#6b7280', textAlign: 'center'},
  warnIcon:        {fontSize: 48, marginBottom: 16},
  warnTitle:       {fontSize: 18, fontWeight: 'bold', color: '#b45309', marginBottom: 12},
  warnDesc:        {fontSize: 15, color: '#374151', textAlign: 'center', lineHeight: 22},
});
