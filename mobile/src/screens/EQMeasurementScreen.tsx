/**
 * EQMeasurementScreen
 *
 * 최적 위치에 스피커 배치 후 EQ 보정 측정 화면.
 *
 * 흐름:
 * 1. 사용자가 스피커를 최적 위치에 배치
 * 2. sweep 재생 + 마이크 녹음 (SweepRecorder)
 * 3. POST /api/eq/analyze → EQ 보정값 수신
 * 4. Bass/Mid/Treble 요약 + 23밴드 결과 표시
 */
import React, {useRef, useState} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
} from 'react-native';
import {useNavigation, useRoute, RouteProp} from '@react-navigation/native';
import {recordSweep, getSweepUri} from '../native/SweepRecorder';
import {analyzeEQ, EQAnalysisResponse} from '../api/eq';
import {RootStackParamList} from '../types';

type EQRouteProp = RouteProp<RootStackParamList, 'EQMeasurement'>;

type Step = 'ready' | 'recording' | 'analyzing' | 'done';

export default function EQMeasurementScreen() {
  const navigation = useNavigation<any>();
  const route      = useRoute<EQRouteProp>();
  const {optimalPosition} = route.params;  // OptimizationResult에서 넘어온 최적 위치

  const [step, setStep]       = useState<Step>('ready');
  const [result, setResult]   = useState<EQAnalysisResponse | null>(null);
  const mountedRef             = useRef(true);

  const safe = <T,>(setter: (v: T) => void) => (v: T) => {
    if (mountedRef.current) setter(v);
  };

  const handleMeasure = async () => {
    try {
      // ── Step 1: sweep 재생 + 녹음 ──────────────────────────────
      safe(setStep)('recording');
      const recordedUri = await recordSweep('sweep');
      if (!mountedRef.current) return;

      const sweepUri = await getSweepUri();
      if (!mountedRef.current) return;

      // ── Step 2: EQ 분석 ─────────────────────────────────────────
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
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>🎛️ EQ 자동 보정</Text>

        {/* 최적 위치 안내 */}
        <View style={styles.posCard}>
          <Text style={styles.posTitle}>✅ 스피커를 최적 위치에 배치했나요?</Text>
          <Text style={styles.posLine}>
            L: ({optimalPosition.left.x.toFixed(2)}, {optimalPosition.left.y.toFixed(2)}) m
          </Text>
          <Text style={styles.posLine}>
            R: ({optimalPosition.right.x.toFixed(2)}, {optimalPosition.right.y.toFixed(2)}) m
          </Text>
          <Text style={styles.posNote}>위 좌표에 스피커를 배치한 후 측정을 시작하세요.</Text>
        </View>

        {/* 측정 버튼 */}
        {(step === 'ready' || step === 'recording') && (
          <TouchableOpacity
            style={[styles.btn, step === 'recording' && styles.btnDisabled]}
            onPress={handleMeasure}
            disabled={step === 'recording'}>
            {step === 'recording' && <ActivityIndicator color="#fff" />}
            <Text style={styles.btnText}>
              {step === 'recording' ? '  sweep 재생 + 녹음 중...' : '🎙️ EQ 측정 시작'}
            </Text>
          </TouchableOpacity>
        )}
        {step === 'recording' && (
          <Text style={styles.hint}>스피커에서 소리가 나오면 움직이지 마세요.</Text>
        )}

        {/* 분석 중 */}
        {step === 'analyzing' && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#2563eb" />
            <Text style={styles.loadingText}>EQ 보정값 계산 중...</Text>
          </View>
        )}

        {/* 결과 표시 */}
        {step === 'done' && result && (
          <>
            {/* Bass / Mid / Treble 요약 */}
            <View style={styles.card}>
              <Text style={styles.cardTitle}>📊 일반 설정</Text>
              <View style={styles.simpleRow}>
                {(['bass', 'mid', 'treble'] as const).map(band => {
                  const info = result.simple[band];
                  return (
                    <View key={band} style={styles.simpleItem}>
                      <Text style={styles.simpleLabel}>
                        {band === 'bass' ? '저음' : band === 'mid' ? '중음' : '고음'}
                      </Text>
                      <Text style={[
                        styles.simpleGain,
                        info.gain_db > 1 ? styles.gainPos :
                        info.gain_db < -1 ? styles.gainNeg : styles.gainZero,
                      ]}>
                        {info.gain_db > 0 ? '+' : ''}{info.gain_db.toFixed(1)} dB
                      </Text>
                      <Text style={styles.simpleStrength}>{info.label}</Text>
                    </View>
                  );
                })}
              </View>
            </View>

            {/* Parametric EQ */}
            {result.parametric.length > 0 && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>🎚️ 고급 설정</Text>
                {result.parametric.map((f, i) => (
                  <View key={i} style={styles.paramRow}>
                    <Text style={styles.paramFreq}>{f.freq.toLocaleString()} Hz</Text>
                    <Text style={[
                      styles.paramGain,
                      f.gain_db > 0 ? styles.gainPos : styles.gainNeg,
                    ]}>
                      {f.gain_db > 0 ? '+' : ''}{f.gain_db.toFixed(1)} dB
                    </Text>
                    <Text style={styles.paramQ}>Q {f.Q}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* 재측정 버튼 */}
            <TouchableOpacity
              style={[styles.btn, styles.btnOutline]}
              onPress={() => { setStep('ready'); setResult(null); }}>
              <Text style={styles.btnOutlineText}>🔄 다시 측정</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.btnHome}
              onPress={() => navigation.navigate('Home')}>
              <Text style={styles.btnText}>홈으로</Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ── 스타일 ───────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container:   {flex: 1, backgroundColor: '#f9fafb'},
  content:     {padding: 20, paddingBottom: 40},
  title:       {fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginBottom: 20},
  posCard:     {backgroundColor: '#eff6ff', borderRadius: 12, padding: 16, marginBottom: 20, borderLeftWidth: 4, borderLeftColor: '#2563eb'},
  posTitle:    {fontSize: 15, fontWeight: 'bold', color: '#1e40af', marginBottom: 8},
  posLine:     {fontSize: 14, color: '#1e3a8a', marginBottom: 2},
  posNote:     {fontSize: 12, color: '#6b7280', marginTop: 6},
  btn:         {backgroundColor: '#10b981', padding: 18, borderRadius: 14, alignItems: 'center', flexDirection: 'row', justifyContent: 'center', marginBottom: 12},
  btnDisabled: {opacity: 0.6},
  btnText:     {color: '#fff', fontSize: 17, fontWeight: '600'},
  btnOutline:  {backgroundColor: 'transparent', borderWidth: 2, borderColor: '#6b7280', padding: 14, borderRadius: 14, alignItems: 'center', marginBottom: 12},
  btnOutlineText: {color: '#374151', fontSize: 15, fontWeight: '600'},
  btnHome:     {backgroundColor: '#2563eb', padding: 14, borderRadius: 10, alignItems: 'center'},
  hint:        {fontSize: 12, color: '#6b7280', textAlign: 'center', marginBottom: 16},
  loadingBox:  {alignItems: 'center', padding: 32},
  loadingText: {marginTop: 16, fontSize: 16, color: '#374151', fontWeight: '500'},
  card:        {backgroundColor: '#fff', padding: 16, borderRadius: 12, marginBottom: 16, elevation: 2, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, shadowOffset: {width: 0, height: 2}},
  cardTitle:   {fontSize: 17, fontWeight: 'bold', marginBottom: 12},
  simpleRow:   {flexDirection: 'row', justifyContent: 'space-around'},
  simpleItem:  {alignItems: 'center'},
  simpleLabel: {fontSize: 14, color: '#6b7280', marginBottom: 4},
  simpleGain:  {fontSize: 20, fontWeight: 'bold', marginBottom: 2},
  simpleStrength: {fontSize: 12, color: '#9ca3af'},
  gainPos:     {color: '#2563eb'},
  gainNeg:     {color: '#dc2626'},
  gainZero:    {color: '#6b7280'},
  paramRow:    {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#f3f4f6'},
  paramFreq:   {fontSize: 14, color: '#374151', width: 90},
  paramGain:   {fontSize: 14, fontWeight: '600', width: 70, textAlign: 'center'},
  paramQ:      {fontSize: 14, color: '#6b7280', width: 50, textAlign: 'right'},
});
