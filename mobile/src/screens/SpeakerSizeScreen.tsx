// screens/SpeakerSizeScreen.tsx — 스피커 치수 입력 (시네마틱 리스킨)
//
// 디자인 원칙
//   · HomeScreen / UploadScreen 과 동일한 다크 + 3-색 액센트 시스템
//   · 입력 트리오 (W/H/D) 를 풍성한 글래스 input + cm unit chip 으로 표현
//   · 스피커 박스 미니 illustration (SVG) — 입력값에 따라 비율이 미세하게 변함
//   · purple 계열 primary glass pill (메인의 "스피커 배치" 액센트)
import React, {useEffect, useMemo, useRef, useState} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  StatusBar,
  Animated,
  Easing,
  Dimensions,
} from 'react-native';
import Svg, {
  Defs,
  RadialGradient,
  LinearGradient as SvgLinearGradient,
  Stop,
  Rect,
  Path,
  G,
} from 'react-native-svg';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList, SpeakerDimensions} from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'SpeakerSize'>;

const {width: SCREEN_W} = Dimensions.get('window');

const MIN_CM = 1;
const MAX_CM = 200;

type FieldKey = 'width_cm' | 'height_cm' | 'depth_cm';

interface FieldConfig {
  key: FieldKey;
  label: string;
  abbr: string;
  placeholder: string;
  hint: string;
  accent: string;
}

const FIELDS: FieldConfig[] = [
  {
    key: 'width_cm',
    label: '가로',
    abbr: 'W',
    placeholder: '20',
    hint: '정면에서 본 너비',
    accent: '#FF8A5B',
  },
  {
    key: 'height_cm',
    label: '세로',
    abbr: 'H',
    placeholder: '35',
    hint: '바닥에서 위까지 높이',
    accent: '#9E7BE0',
  },
  {
    key: 'depth_cm',
    label: '깊이',
    abbr: 'D',
    placeholder: '28',
    hint: '앞뒤 두께',
    accent: '#3DC8FF',
  },
];

function parseDim(raw: string): number | null {
  if (raw.trim() === '') return null;
  const n = Number(raw.replace(',', '.'));
  if (!Number.isFinite(n)) return null;
  if (n < MIN_CM || n > MAX_CM) return null;
  return n;
}

// ─────────────────────────────────────────────────────────────────
// 메인 스크린
// ─────────────────────────────────────────────────────────────────
export default function SpeakerSizeScreen({navigation}: Props) {
  const [values, setValues] = useState<Record<FieldKey, string>>({
    width_cm: '',
    height_cm: '',
    depth_cm: '',
  });
  const [focusedKey, setFocusedKey] = useState<FieldKey | null>(null);

  const fade = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 1100,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [fade]);

  const parsed = useMemo(
    () => ({
      width_cm: parseDim(values.width_cm),
      height_cm: parseDim(values.height_cm),
      depth_cm: parseDim(values.depth_cm),
    }),
    [values],
  );

  const isValid =
    parsed.width_cm !== null &&
    parsed.height_cm !== null &&
    parsed.depth_cm !== null;

  const handleNext = () => {
    if (!isValid) return;
    const dimensions: SpeakerDimensions = {
      width_cm: parsed.width_cm as number,
      height_cm: parsed.height_cm as number,
      depth_cm: parsed.depth_cm as number,
    };
    navigation.navigate('SpeakerPlacement', {speakerDimensions: dimensions});
  };

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <BackgroundGlow />
      <SafeAreaView style={styles.safe}>
        <KeyboardAvoidingView
          style={styles.flex}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <ScrollView
            contentContainerStyle={styles.scroll}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}>
            {/* Header */}
            <Animated.View style={[styles.header, {opacity: fade}]}>
              <Text style={styles.eyebrow}>SPEAKER SIZE</Text>
              <Text style={styles.hero}>
                스피커의 형태를{'\n'}알려주세요
              </Text>
              <Text style={styles.subline}>
                벽·모서리 여유 공간 계산에 사용됩니다.
              </Text>
            </Animated.View>

            {/* Speaker box illustration */}
            <Animated.View style={[styles.illust, {opacity: fade}]}>
              <SpeakerBoxIllust
                w={parsed.width_cm}
                h={parsed.height_cm}
                d={parsed.depth_cm}
              />
            </Animated.View>

            {/* Inputs */}
            <Animated.View style={[styles.fields, {opacity: fade}]}>
              {FIELDS.map(field => {
                const raw = values[field.key];
                const value = parsed[field.key];
                const showError = raw.trim() !== '' && value === null;
                const focused = focusedKey === field.key;
                return (
                  <View key={field.key} style={styles.fieldRow}>
                    <View style={styles.labelRow}>
                      <View
                        style={[
                          styles.abbrBadge,
                          {backgroundColor: field.accent + '22', borderColor: field.accent},
                        ]}>
                        <Text style={[styles.abbrText, {color: field.accent}]}>
                          {field.abbr}
                        </Text>
                      </View>
                      <Text style={styles.fieldLabel}>{field.label}</Text>
                      <Text style={styles.fieldHint}>· {field.hint}</Text>
                    </View>

                    <View
                      style={[
                        styles.inputWrap,
                        focused && {borderColor: field.accent + '99'},
                        showError && styles.inputWrapError,
                      ]}>
                      <TextInput
                        style={styles.input}
                        value={raw}
                        onChangeText={txt =>
                          setValues(prev => ({...prev, [field.key]: txt}))
                        }
                        onFocus={() => setFocusedKey(field.key)}
                        onBlur={() => setFocusedKey(null)}
                        placeholder={field.placeholder}
                        placeholderTextColor="rgba(245,245,247,0.25)"
                        keyboardType="decimal-pad"
                        maxLength={6}
                        returnKeyType="done"
                      />
                      <View style={styles.unitPill}>
                        <Text style={styles.unitText}>cm</Text>
                      </View>
                    </View>

                    {showError && (
                      <Text style={styles.errorText}>
                        {MIN_CM}–{MAX_CM} 범위로 입력해주세요
                      </Text>
                    )}
                  </View>
                );
              })}
            </Animated.View>

            {/* Submit */}
            <Animated.View style={[styles.submitWrap, {opacity: fade}]}>
              <TouchableOpacity
                activeOpacity={0.85}
                onPress={handleNext}
                disabled={!isValid}
                style={[
                  styles.submitBtn,
                  !isValid && styles.submitBtnDisabled,
                ]}>
                <View style={styles.submitInner}>
                  <Text
                    style={[
                      styles.submitText,
                      !isValid && styles.submitTextDisabled,
                    ]}>
                    다음 단계
                  </Text>
                  <Text
                    style={[
                      styles.submitArrow,
                      !isValid && styles.submitTextDisabled,
                    ]}>
                    →
                  </Text>
                </View>
              </TouchableOpacity>
            </Animated.View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────
// Background glow (메인/업로드와 동일 톤)
// ─────────────────────────────────────────────────────────────────
function BackgroundGlow() {
  return (
    <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
      <Defs>
        <RadialGradient id="sizeBg" cx="50%" cy="42%" r="68%">
          <Stop offset="0%" stopColor="#1C1530" stopOpacity="1" />
          <Stop offset="50%" stopColor="#0A0A12" stopOpacity="1" />
          <Stop offset="100%" stopColor="#000000" stopOpacity="1" />
        </RadialGradient>
        <RadialGradient id="sizePurpleGlow" cx="50%" cy="42%" r="32%">
          <Stop offset="0%" stopColor="#9E7BE0" stopOpacity="0.06" />
          <Stop offset="100%" stopColor="#9E7BE0" stopOpacity="0" />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#sizeBg)" />
      <Rect width="100%" height="100%" fill="url(#sizePurpleGlow)" />
    </Svg>
  );
}

// ─────────────────────────────────────────────────────────────────
// SpeakerBoxIllust — 입력값으로 비율이 변하는 hairline 박스
// ─────────────────────────────────────────────────────────────────
interface IllustProps {
  w: number | null;
  h: number | null;
  d: number | null;
}

function SpeakerBoxIllust({w, h, d}: IllustProps) {
  const ILLUST_W = SCREEN_W - 80;
  const ILLUST_H = 140;

  // 기본 비율 — 입력값이 있으면 그 비율로 계산
  const defaults = {w: 20, h: 35, d: 28};
  const rw = w ?? defaults.w;
  const rh = h ?? defaults.h;
  const rd = d ?? defaults.d;

  // 가장 큰 변을 90px 로 정규화
  const maxV = Math.max(rw, rh, rd);
  const unit = 80 / maxV;

  const bw = rw * unit;
  const bh = rh * unit;
  const bd = rd * unit;

  const cx = ILLUST_W / 2;
  const cy = ILLUST_H / 2;

  // 정면 사각형 (W × H), 측면 평행사변형 (D 깊이)
  const frontX = cx - bw / 2 - bd * 0.3;
  const frontY = cy - bh / 2 + bd * 0.15;
  const skewX = bd * 0.6;
  const skewY = -bd * 0.3;

  return (
    <Svg width={ILLUST_W} height={ILLUST_H}>
      <Defs>
        <SvgLinearGradient id="boxFace" x1="0%" y1="0%" x2="100%" y2="100%">
          <Stop offset="0%" stopColor="#9E7BE0" stopOpacity="0.18" />
          <Stop offset="100%" stopColor="#9E7BE0" stopOpacity="0.04" />
        </SvgLinearGradient>
      </Defs>

      <G opacity={w === null && h === null && d === null ? 0.45 : 1}>
        {/* 측면 (D 평행사변형) — 뒤 윗면 */}
        <Path
          d={`M ${frontX} ${frontY}
              L ${frontX + skewX} ${frontY + skewY}
              L ${frontX + skewX + bw} ${frontY + skewY}
              L ${frontX + bw} ${frontY} Z`}
          fill="rgba(245,245,247,0.04)"
          stroke="rgba(245,245,247,0.5)"
          strokeWidth={1}
        />
        {/* 측면 (D 평행사변형) — 우측면 */}
        <Path
          d={`M ${frontX + bw} ${frontY}
              L ${frontX + bw + skewX} ${frontY + skewY}
              L ${frontX + bw + skewX} ${frontY + skewY + bh}
              L ${frontX + bw} ${frontY + bh} Z`}
          fill="rgba(245,245,247,0.04)"
          stroke="rgba(245,245,247,0.5)"
          strokeWidth={1}
        />
        {/* 정면 (W × H 사각형) */}
        <Rect
          x={frontX}
          y={frontY}
          width={bw}
          height={bh}
          fill="url(#boxFace)"
          stroke="#F5F5F7"
          strokeWidth={1.3}
        />
        {/* 정면 우퍼/트위터 동그라미 (스피커 느낌) */}
        <Path
          d={`M ${frontX + bw / 2 - bw * 0.18} ${frontY + bh * 0.35}
              a ${bw * 0.18} ${bw * 0.18} 0 1 0 ${bw * 0.36} 0
              a ${bw * 0.18} ${bw * 0.18} 0 1 0 ${-bw * 0.36} 0`}
          fill="none"
          stroke="rgba(245,245,247,0.45)"
          strokeWidth={1}
        />
        <Path
          d={`M ${frontX + bw / 2 - bw * 0.1} ${frontY + bh * 0.72}
              a ${bw * 0.1} ${bw * 0.1} 0 1 0 ${bw * 0.2} 0
              a ${bw * 0.1} ${bw * 0.1} 0 1 0 ${-bw * 0.2} 0`}
          fill="none"
          stroke="rgba(245,245,247,0.45)"
          strokeWidth={1}
        />
      </G>
    </Svg>
  );
}

// ─────────────────────────────────────────────────────────────────
// 스타일
// ─────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: '#000'},
  safe: {flex: 1},
  flex: {flex: 1},
  scroll: {paddingHorizontal: 24, paddingBottom: 32, paddingTop: 56},

  // Header
  header: {alignItems: 'center', marginBottom: 6},
  eyebrow: {
    fontSize: 10,
    color: 'rgba(245,245,247,0.42)',
    letterSpacing: 6,
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
  subline: {
    fontSize: 12,
    color: 'rgba(245,245,247,0.5)',
    letterSpacing: 0.4,
    marginTop: 12,
    textAlign: 'center',
  },

  // Illust
  illust: {alignItems: 'center', marginVertical: 18},

  // Fields
  fields: {marginTop: 6, gap: 18},
  fieldRow: {},
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
    paddingHorizontal: 4,
  },
  abbrBadge: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  abbrText: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  fieldLabel: {
    fontSize: 14,
    color: '#F5F5F7',
    fontWeight: '500',
    letterSpacing: 0.5,
  },
  fieldHint: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.45)',
    letterSpacing: 0.3,
  },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 56,
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderColor: 'rgba(245,245,247,0.16)',
    backgroundColor: 'rgba(255,255,255,0.03)',
    paddingHorizontal: 16,
  },
  inputWrapError: {
    borderColor: 'rgba(255,90,90,0.7)',
    backgroundColor: 'rgba(255,90,90,0.05)',
  },
  input: {
    flex: 1,
    fontSize: 18,
    color: '#F5F5F7',
    fontWeight: '300',
    letterSpacing: 0.4,
    padding: 0,
  },
  unitPill: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 100,
    backgroundColor: 'rgba(245,245,247,0.08)',
  },
  unitText: {
    fontSize: 11,
    color: 'rgba(245,245,247,0.7)',
    letterSpacing: 1,
    fontWeight: '600',
  },
  errorText: {
    fontSize: 11,
    color: '#FF6A6A',
    marginTop: 6,
    marginLeft: 4,
    letterSpacing: 0.3,
  },

  // Submit
  submitWrap: {marginTop: 28},
  submitBtn: {
    height: 62,
    borderRadius: 100,
    overflow: 'hidden',
    backgroundColor: '#9E7BE0',
    shadowColor: '#9E7BE0',
    shadowOffset: {width: 0, height: 8},
    shadowOpacity: 0.45,
    shadowRadius: 16,
    elevation: 8,
  },
  submitBtnDisabled: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    shadowOpacity: 0,
    elevation: 0,
  },
  submitInner: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  submitText: {
    fontSize: 16,
    color: '#0A0A12',
    fontWeight: '600',
    letterSpacing: 1.2,
  },
  submitArrow: {
    fontSize: 18,
    color: '#0A0A12',
    fontWeight: '300',
  },
  submitTextDisabled: {
    color: 'rgba(245,245,247,0.35)',
  },
});
