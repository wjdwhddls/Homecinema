/**
 * SpeakerSizeScreen
 *
 * 방 스캔 전에 사용자 스피커의 물리 치수(W × H × D, cm)를 입력받는 화면.
 * 입력값은 백엔드 최적화기의 벽/모서리 여유 마진 계산에 사용된다.
 */
import React, {useMemo, useState} from 'react';
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
} from 'react-native';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList, SpeakerDimensions} from '../types';

type Props = NativeStackScreenProps<RootStackParamList, 'SpeakerSize'>;

const MIN_CM = 1;
const MAX_CM = 200;

type FieldKey = 'width_cm' | 'height_cm' | 'depth_cm';

interface FieldConfig {
  key: FieldKey;
  label: string;
  placeholder: string;
  hint: string;
}

const FIELDS: FieldConfig[] = [
  {key: 'width_cm', label: '가로 (W)', placeholder: '예: 20', hint: '정면에서 본 너비'},
  {key: 'height_cm', label: '세로 (H)', placeholder: '예: 35', hint: '바닥에서 위까지 높이'},
  {key: 'depth_cm', label: '깊이 (D)', placeholder: '예: 28', hint: '앞뒤 두께'},
];

function parseDim(raw: string): number | null {
  if (raw.trim() === '') return null;
  const n = Number(raw.replace(',', '.'));
  if (!Number.isFinite(n)) return null;
  if (n < MIN_CM || n > MAX_CM) return null;
  return n;
}

export default function SpeakerSizeScreen({navigation}: Props) {
  const [values, setValues] = useState<Record<FieldKey, string>>({
    width_cm: '',
    height_cm: '',
    depth_cm: '',
  });

  const parsed = useMemo(() => {
    return {
      width_cm: parseDim(values.width_cm),
      height_cm: parseDim(values.height_cm),
      depth_cm: parseDim(values.depth_cm),
    };
  }, [values]);

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
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>스피커 치수 입력</Text>

          <View style={styles.infoBox}>
            <Text style={styles.infoIcon}>📏</Text>
            <View style={styles.infoTextWrap}>
              <Text style={styles.infoTitle}>왜 필요한가요?</Text>
              <Text style={styles.infoDesc}>
                스피커 크기에 맞춰 벽·모서리 여유 공간을 계산해
                실제로 놓을 수 있는 위치만 추천합니다.
              </Text>
            </View>
          </View>

          {FIELDS.map(field => {
            const raw = values[field.key];
            const value = parsed[field.key];
            const showError = raw.trim() !== '' && value === null;
            return (
              <View key={field.key} style={styles.fieldRow}>
                <Text style={styles.fieldLabel}>{field.label}</Text>
                <View style={[styles.inputWrap, showError && styles.inputWrapError]}>
                  <TextInput
                    style={styles.input}
                    value={raw}
                    onChangeText={txt =>
                      setValues(prev => ({...prev, [field.key]: txt}))
                    }
                    placeholder={field.placeholder}
                    placeholderTextColor="#9ca3af"
                    keyboardType="decimal-pad"
                    maxLength={6}
                    returnKeyType="done"
                  />
                  <Text style={styles.unit}>cm</Text>
                </View>
                <Text style={styles.fieldHint}>{field.hint}</Text>
                {showError && (
                  <Text style={styles.errorText}>
                    {MIN_CM}~{MAX_CM} 사이의 숫자를 입력해주세요.
                  </Text>
                )}
              </View>
            );
          })}

          <TouchableOpacity
            style={[styles.nextBtn, !isValid && styles.nextBtnDisabled]}
            onPress={handleNext}
            disabled={!isValid}>
            <Text style={styles.nextBtnText}>다음 →</Text>
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#fff'},
  flex: {flex: 1},
  content: {padding: 20, paddingBottom: 40},
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 16,
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: '#eff6ff',
    borderLeftWidth: 4,
    borderLeftColor: '#2563eb',
    padding: 14,
    borderRadius: 10,
    marginBottom: 24,
  },
  infoIcon: {fontSize: 24, marginRight: 12},
  infoTextWrap: {flex: 1},
  infoTitle: {fontSize: 15, fontWeight: 'bold', color: '#1e40af', marginBottom: 4},
  infoDesc: {fontSize: 13, color: '#1e3a8a', lineHeight: 19},
  fieldRow: {marginBottom: 20},
  fieldLabel: {fontSize: 16, fontWeight: '600', color: '#111827', marginBottom: 6},
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 10,
    backgroundColor: '#f9fafb',
    paddingHorizontal: 14,
  },
  inputWrapError: {borderColor: '#dc2626', backgroundColor: '#fef2f2'},
  input: {
    flex: 1,
    fontSize: 18,
    paddingVertical: 14,
    color: '#111827',
  },
  unit: {fontSize: 16, color: '#6b7280', marginLeft: 8},
  fieldHint: {fontSize: 12, color: '#6b7280', marginTop: 4},
  errorText: {fontSize: 12, color: '#dc2626', marginTop: 4},
  nextBtn: {
    marginTop: 12,
    backgroundColor: '#2563eb',
    padding: 18,
    borderRadius: 14,
    alignItems: 'center',
  },
  nextBtnDisabled: {backgroundColor: '#93c5fd'},
  nextBtnText: {color: '#fff', fontSize: 17, fontWeight: '600'},
});
