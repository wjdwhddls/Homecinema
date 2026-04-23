/**
 * PlaybackScreen
 *
 * 서버에서 원본과 EQ 적용된 영상을 다운로드하여 로컬에 저장한 후,
 * A/B 토글로 비교 재생합니다.
 *
 * 현재 skeleton 단계:
 * - 백엔드의 processed.mp4가 아직 실제 EQ 적용본이 아니기 때문에
 *   (ML/EQ 처리 미통합), DEV_FAKE_PROCESSED=true로 설정된 경우에만
 *   원본 영상이 processed로 반환되어 재생 UI를 테스트할 수 있습니다.
 * - 이 경우 A/B 토글은 "원본 vs 원본"을 재생하므로 청각적 차이가 없지만,
 *   UI 플로우와 다운로드/저장/재생 로직은 정상 동작합니다.
 *
 * Phase 2 이후:
 * - 서버가 실제 EQ 적용본을 반환하므로 A/B 토글이 청각적 차이를 드러냅니다.
 *
 * 재생 기술 참고:
 * - react-native-video 사용
 * - 로컬 파일 재생 (file:// URI)
 * - A/B 토글 시 currentTime 저장 후 소스 교체 + seek
 */
import React, {useState, useRef, useCallback, useEffect} from 'react';
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
import Video, {type VideoRef, type OnProgressData, type OnLoadData} from 'react-native-video';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList, TimelineData} from '../types';
import {useDownloadJob} from '../hooks/useDownloadJob';
import {deleteJob, getJobTimeline} from '../api/jobs';
import {deleteLocalJob} from '../utils/localStorage';
import {COLORS} from '../constants/colors';
import MoodTimeline from '../components/MoodTimeline';

type Props = NativeStackScreenProps<RootStackParamList, 'Playback'>;

export default function PlaybackScreen({route, navigation}: Props) {
  const {jobId} = route.params;

  // 다운로드 hook
  const {
    phase,
    originalProgress,
    processedProgress,
    originalLocalPath,
    processedLocalPath,
    error,
    retry,
    cancel,
  } = useDownloadJob(jobId);

  // 재생 상태
  const [currentSource, setCurrentSource] = useState<
    'original' | 'processed'
  >('processed');
  const [currentTime, setCurrentTime] = useState(0);
  const [paused, setPaused] = useState(false);
  const videoRef = useRef<VideoRef>(null);

  // Timeline (mood 타임라인 UI 용). completed 된 job 만 endpoint 가 200 반환.
  const [timeline, setTimeline] = useState<TimelineData | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (phase !== 'ready') return;
    getJobTimeline(jobId)
      .then(t => {
        if (!cancelled) setTimeline(t);
      })
      .catch(() => {
        // timeline 없어도 재생은 가능 — silent fail
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, phase]);

  // MoodTimeline 탭 seek
  const handleSeek = useCallback((sec: number) => {
    setCurrentTime(sec);
    videoRef.current?.seek(sec);
  }, []);

  // A/B 토글
  const handleToggle = useCallback(
    (target: 'original' | 'processed') => {
      if (target === currentSource) {
        return;
      }
      setCurrentSource(target);
    },
    [currentSource],
  );

  // 소스 변경 후 이전 위치로 seek
  const handleLoad = useCallback(
    (_data: OnLoadData) => {
      if (videoRef.current && currentTime > 0) {
        videoRef.current.seek(currentTime);
      }
    },
    [currentTime],
  );

  // 재생 위치 업데이트
  const handleProgress = useCallback((data: OnProgressData) => {
    setCurrentTime(data.currentTime);
  }, []);

  // 삭제 처리
  const handleDelete = useCallback(() => {
    Alert.alert(
      '삭제 확인',
      '이 영상을 삭제하시겠습니까? 로컬 파일과 서버 데이터가 모두 삭제됩니다.',
      [
        {text: '취소', style: 'cancel'},
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            try {
              await deleteJob(jobId);
              await deleteLocalJob(jobId);
              navigation.replace('Home', undefined);
            } catch (err: any) {
              Alert.alert(
                '삭제 실패',
                err.message || '삭제 중 오류가 발생했습니다.',
              );
            }
          },
        },
      ],
    );
  }, [jobId, navigation]);

  // 헤더에 삭제 버튼 설정
  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <TouchableOpacity onPress={handleDelete} style={styles.headerButton}>
          <Text style={styles.headerButtonText}>🗑️</Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation, handleDelete]);

  // --- downloading 상태 ---
  if (phase === 'downloading_original' || phase === 'downloading_processed') {
    const isOriginal = phase === 'downloading_original';
    const progress = isOriginal ? originalProgress : processedProgress;
    const label = isOriginal ? '원본' : 'EQ 적용본';

    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.downloadText}>
            {label} 다운로드 중... {Math.round(progress * 100)}%
          </Text>
          <TouchableOpacity
            style={styles.cancelButton}
            onPress={() => {
              cancel();
              navigation.replace('Home', undefined);
            }}>
            <Text style={styles.cancelButtonText}>취소</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // --- error 상태 ---
  if (phase === 'error') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centerContent}>
          <Text style={styles.errorIcon}>❌</Text>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={retry}>
            <Text style={styles.retryButtonText}>다시 시도</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // --- idle (로딩 중) ---
  if (phase === 'idle' || !originalLocalPath || !processedLocalPath) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.downloadText}>준비 중...</Text>
        </View>
      </SafeAreaView>
    );
  }

  // --- ready/playing 상태 ---
  const videoSource =
    currentSource === 'original'
      ? {uri: `file://${originalLocalPath}`}
      : {uri: `file://${processedLocalPath}`};

  return (
    <SafeAreaView style={styles.container}>
      {/* 영상 플레이어 */}
      <View style={styles.videoContainer}>
        <Video
          ref={videoRef}
          source={videoSource}
          style={styles.video}
          paused={paused}
          onLoad={handleLoad}
          onProgress={handleProgress}
          resizeMode="contain"
          controls={true}
        />
      </View>

      {/* Mood 타임라인 — timeline fetch 됐을 때만 */}
      {timeline && (
        <MoodTimeline
          scenes={timeline.scenes}
          durationSec={timeline.metadata.duration_sec}
          currentTimeSec={currentTime}
          onSeek={handleSeek}
        />
      )}

      {/* A/B 토글 섹션 */}
      <View style={styles.toggleSection}>
        <Text style={styles.toggleLabel}>재생 소스</Text>
        <View style={styles.toggleRow}>
          <TouchableOpacity
            style={[
              styles.toggleButton,
              currentSource === 'original' && styles.toggleButtonActive,
            ]}
            onPress={() => handleToggle('original')}
            activeOpacity={0.8}>
            <Text
              style={[
                styles.toggleButtonText,
                currentSource === 'original' && styles.toggleButtonTextActive,
              ]}>
              🔇 원본
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.toggleButton,
              currentSource === 'processed' && styles.toggleButtonActive,
            ]}
            onPress={() => handleToggle('processed')}
            activeOpacity={0.8}>
            <Text
              style={[
                styles.toggleButtonText,
                currentSource === 'processed' && styles.toggleButtonTextActive,
              ]}>
              🎵 EQ 적용
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* 재생/일시정지 버튼 */}
      <View style={styles.controlSection}>
        <TouchableOpacity
          style={styles.playPauseButton}
          onPress={() => setPaused(!paused)}
          activeOpacity={0.8}>
          <Text style={styles.playPauseText}>{paused ? '▶️ 재생' : '⏸️ 일시정지'}</Text>
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
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  videoContainer: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#000',
  },
  video: {
    width: '100%',
    height: '100%',
  },
  toggleSection: {
    paddingHorizontal: 24,
    paddingTop: 20,
  },
  toggleLabel: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginBottom: 12,
    textAlign: 'center',
  },
  toggleRow: {
    flexDirection: 'row',
    gap: 12,
  },
  toggleButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 10,
    backgroundColor: COLORS.toggleInactive,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  toggleButtonActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primaryDark,
  },
  toggleButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.textSecondary,
  },
  toggleButtonTextActive: {
    color: COLORS.textInverse,
  },
  controlSection: {
    paddingHorizontal: 24,
    paddingTop: 16,
    alignItems: 'center',
  },
  playPauseButton: {
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
    backgroundColor: COLORS.buttonPrimary,
  },
  playPauseText: {
    color: COLORS.textInverse,
    fontSize: 16,
    fontWeight: '600',
  },
  downloadText: {
    fontSize: 16,
    color: COLORS.text,
    marginTop: 16,
  },
  cancelButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
    backgroundColor: COLORS.buttonSecondary,
  },
  cancelButtonText: {
    color: COLORS.textInverse,
    fontSize: 16,
    fontWeight: '600',
  },
  errorIcon: {
    fontSize: 48,
  },
  errorText: {
    fontSize: 14,
    color: COLORS.error,
    marginTop: 12,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
    backgroundColor: COLORS.buttonPrimary,
  },
  retryButtonText: {
    color: COLORS.textInverse,
    fontSize: 16,
    fontWeight: '600',
  },
  headerButton: {
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  headerButtonText: {
    fontSize: 20,
  },
});
