// screens/HomeScreen.tsx — 홈 화면
import React from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList} from '../types';
import {COLORS} from '../constants/colors';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({navigation}: Props) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {/* 앱 제목 */}
        <View style={styles.titleSection}>
          <Text style={styles.title}>Mood EQ</Text>
          <Text style={styles.subtitle}>영상 분위기 자동 EQ</Text>
        </View>

        {/* 메인 버튼 */}
        <View style={styles.buttonSection}>
          <TouchableOpacity
            style={[styles.button, styles.primaryButton]}
            onPress={() => navigation.navigate('Upload')}
            activeOpacity={0.8}>
            <Text style={styles.buttonText}>🎬 새 영상 업로드</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.secondaryButton]}
            onPress={() => navigation.navigate('SpeakerSize')}
            activeOpacity={0.8}>
            <Text style={styles.buttonText}>📍 스피커 위치 자동 배정</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  titleSection: {
    alignItems: 'center',
    marginBottom: 60,
  },
  title: {
    fontSize: 40,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    color: COLORS.textSecondary,
  },
  buttonSection: {
    width: '100%',
    gap: 16,
  },
  button: {
    paddingVertical: 18,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: COLORS.buttonPrimary,
  },
  secondaryButton: {
    backgroundColor: COLORS.buttonSecondary,
  },
  buttonText: {
    color: COLORS.textInverse,
    fontSize: 18,
    fontWeight: '600',
  },
});
