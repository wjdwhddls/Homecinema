// screens/SpeakerPlacementScreen.tsx — 스피커 위치 자동 배정 (placeholder)
// 이 화면은 별도 팀에서 ARKit/ARCore를 사용하여 구현할 예정입니다.
import React from 'react';
import {SafeAreaView, View, Text, StyleSheet} from 'react-native';
import {COLORS} from '../constants/colors';

export default function SpeakerPlacementScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.icon}>📍</Text>
        <Text style={styles.title}>스피커 위치 자동 배정</Text>
        <Text style={styles.description}>
          ARKit/ARCore를 이용한 스피커 위치 자동 배정 기능은 별도 팀에서 구현
          예정입니다.
        </Text>
        <View style={styles.infoBox}>
          <Text style={styles.infoText}>
            이 기능은 사용자의 방 구조를 AR로 스캔한 후, 최적의 스피커 배치를
            추천합니다. 구현 시 src/native/ 폴더에 native 모듈이 추가됩니다.
          </Text>
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
  icon: {
    fontSize: 64,
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 12,
  },
  description: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 24,
  },
  infoBox: {
    backgroundColor: COLORS.surface,
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  infoText: {
    fontSize: 14,
    color: COLORS.textSecondary,
    lineHeight: 20,
  },
});
