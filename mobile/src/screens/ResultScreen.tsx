/**
 * ResultScreen
 *
 * 업로드 완료 후 job의 분석 상태를 polling하고, 완료되면 PlaybackScreen으로
 * 자동 이동합니다.
 *
 * 현재 skeleton 단계:
 * - 백엔드 /api/jobs/{id}/status가 항상 "uploaded"를 반환하므로
 *   자동 이동이 일어나지 않습니다.
 * - 개발 테스트를 위해 "테스트용: 강제 재생 화면으로 이동" 버튼 제공.
 *   이 버튼은 백엔드의 DEV_FAKE_PROCESSED=true와 함께 사용하면
 *   원본 영상을 processed로 받아 A/B 토글 UI를 테스트할 수 있습니다.
 *
 * Phase 2 이후 예정:
 * - 백엔드에서 실제 ML 분석과 EQ 처리가 돌아가면 status가 순차적으로
 *   "analyzing" → "eq_processing" → "completed"로 변경됩니다.
 * - completed가 되면 자동으로 PlaybackScreen으로 navigate합니다.
 */
import React, {useEffect} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList} from '../types';
import {useJobStatus} from '../hooks/useJobStatus';
import {COLORS} from '../constants/colors';

type Props = NativeStackScreenProps<RootStackParamList, 'Result'>;

export default function ResultScreen({route, navigation}: Props) {
  const {jobId} = route.params;

  const {status, progress, errorMessage} = useJobStatus(jobId, {
    pollInterval: 2000,
    maxRetries: 300,
    stopOnStatuses: ['completed', 'failed'],
  });

  // 분석 완료 시 자동으로 PlaybackScreen으로 이동
  useEffect(() => {
    if (status === 'completed') {
      const timer = setTimeout(() => {
        navigation.replace('Playback', {jobId});
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [status, jobId, navigation]);

  // 상태별 UI 렌더링
  const renderStatusContent = () => {
    switch (status) {
      case 'uploaded':
      case 'queued':
        return (
          <>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.statusText}>분석 대기 중...</Text>
          </>
        );
      case 'analyzing':
        return (
          <>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.statusText}>
              영상을 분석하고 있습니다... ({Math.round(progress * 100)}%)
            </Text>
          </>
        );
      case 'eq_processing':
        return (
          <>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.statusText}>EQ를 적용하고 있습니다...</Text>
          </>
        );
      case 'completed':
        return (
          <>
            <Text style={styles.completedIcon}>✅</Text>
            <Text style={styles.statusText}>분석 완료</Text>
            <Text style={styles.subText}>재생 화면으로 이동합니다...</Text>
          </>
        );
      case 'failed':
        return (
          <>
            <Text style={styles.failedIcon}>❌</Text>
            <Text style={styles.statusText}>분석 실패</Text>
            {errorMessage && (
              <Text style={styles.errorText}>{errorMessage}</Text>
            )}
            <TouchableOpacity
              style={styles.homeButton}
              onPress={() => navigation.replace('Home', undefined)}>
              <Text style={styles.homeButtonText}>처음으로</Text>
            </TouchableOpacity>
          </>
        );
      default:
        return (
          <>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.statusText}>상태 확인 중...</Text>
          </>
        );
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {/* Job ID 표시 */}
        <Text style={styles.jobId}>Job: {jobId}</Text>

        {/* 상태 표시 */}
        <View style={styles.statusSection}>{renderStatusContent()}</View>

        {/* 개발 중 알림 박스 */}
        <View style={styles.warningBox}>
          <Text style={styles.warningText}>
            ⚠️ 현재 분석 기능은 개발 중입니다. 업로드된 영상은 서버에
            저장되었지만, 자동 분석과 EQ 적용은 Phase 2 이후 추가 예정입니다.
          </Text>
        </View>

        {/* 테스트용 강제 이동 버튼 */}
        <TouchableOpacity
          style={styles.devButton}
          onPress={() => navigation.replace('Playback', {jobId})}
          activeOpacity={0.8}>
          <Text style={styles.devButtonText}>
            테스트용: 강제 재생 화면으로 이동
          </Text>
        </TouchableOpacity>
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
    paddingHorizontal: 24,
    paddingTop: 16,
  },
  jobId: {
    fontSize: 12,
    color: COLORS.textLight,
    textAlign: 'center',
    marginBottom: 8,
  },
  statusSection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusText: {
    fontSize: 18,
    fontWeight: '600',
    color: COLORS.text,
    marginTop: 16,
  },
  subText: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginTop: 8,
  },
  completedIcon: {
    fontSize: 48,
  },
  failedIcon: {
    fontSize: 48,
  },
  errorText: {
    fontSize: 14,
    color: COLORS.error,
    marginTop: 8,
    textAlign: 'center',
  },
  homeButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
    backgroundColor: COLORS.buttonPrimary,
  },
  homeButtonText: {
    color: COLORS.textInverse,
    fontSize: 16,
    fontWeight: '600',
  },
  warningBox: {
    backgroundColor: COLORS.warningBackground,
    borderWidth: 1,
    borderColor: COLORS.warningBorder,
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
  },
  warningText: {
    fontSize: 13,
    color: COLORS.warningText,
    lineHeight: 20,
  },
  devButton: {
    paddingVertical: 14,
    borderRadius: 8,
    backgroundColor: COLORS.secondary,
    alignItems: 'center',
    marginBottom: 24,
  },
  devButtonText: {
    color: COLORS.textInverse,
    fontSize: 14,
    fontWeight: '500',
  },
});
