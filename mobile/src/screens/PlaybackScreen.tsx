/**
 * PlaybackScreen
 *
 * 서버에서 원본과 EQ 적용된 영상을 다운로드하여 로컬에 저장한 후,
 * A/B 토글로 비교 재생합니다.
 *
 * UI 톤: Home/Upload/Result 와 동일한 시네마틱 다크 테마.
 *  · 배경 #000, rgba glass 카드, 오렌지(#FF8A5B) accent
 *  · 10-band EQ gain 은 현재 씬만 SceneEQChart 로 세로 바 표시
 *    - 양수(+dB, boost) = 오렌지
 *    - 음수(-dB, cut)   = 퍼플(#9E7BE0)
 *    - mode='original'  = bypass, flat 으로 정적 표시
 *
 * 재생 기술 참고:
 * - react-native-video 사용
 * - 로컬 파일 재생 (file:// URI)
 * - A/B 토글 시 currentTime 저장 후 소스 교체 + seek
 */
import React, {useState, useRef, useCallback, useEffect, useMemo} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
  StatusBar,
} from 'react-native';
import Video, {
  type VideoRef,
  type OnProgressData,
  type OnLoadData,
} from 'react-native-video';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {
  MoodName,
  RootStackParamList,
  TimelineData,
  TimelineScene,
} from '../types';
import {useDownloadJob} from '../hooks/useDownloadJob';
import {deleteJob, getJobTimeline} from '../api/jobs';
import {deleteLocalJob} from '../utils/localStorage';
import SceneEQChart from '../components/SceneEQChart';
import EQResponseCurve from '../components/EQResponseCurve';

const BG = '#000000';
const CARD_BG = 'rgba(255,255,255,0.04)';
const CARD_BORDER = 'rgba(255,255,255,0.08)';
const TEXT_PRIMARY = '#f5f5f7';
const TEXT_SECONDARY = 'rgba(245,245,247,0.6)';
const TEXT_MUTED = 'rgba(245,245,247,0.45)';
const ACCENT_ORANGE = '#FF8A5B';
const ACCENT_ORANGE_DIM = 'rgba(255,138,91,0.25)';
const ACCENT_PURPLE = '#9E7BE0';
const ERROR_COLOR = '#ff6b6b';

const MOOD_LABELS_KO: Record<MoodName, string> = {
  Tension: '긴장',
  Sadness: '슬픔',
  Peacefulness: '평온',
  JoyfulActivation: '활기',
  Tenderness: '부드러움',
  Power: '힘',
  Wonder: '경이',
};

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
  const [currentSource, setCurrentSource] = useState<'original' | 'processed'>(
    'processed',
  );
  const [currentTime, setCurrentTime] = useState(0);
  const [paused, setPaused] = useState(false);
  const videoRef = useRef<VideoRef>(null);
  // A/B 토글 시 react-native-video 가 새 source 로딩 동안 onProgress=0 을
  // transient 하게 방출하는 것을 차단. onLoad 후 seek 하며 해제.
  const ignoreProgressRef = useRef(false);

  // Timeline
  const [timeline, setTimeline] = useState<TimelineData | null>(null);

  // 현재 재생 위치에 해당하는 씬
  const currentScene = useMemo<TimelineScene | null>(() => {
    if (!timeline || timeline.scenes.length === 0) {
      return null;
    }
    const scenes = timeline.scenes;
    const hit = scenes.find(
      s => currentTime >= s.start_sec && currentTime < s.end_sec,
    );
    if (hit) {
      return hit;
    }
    // 영상 끝 (currentTime >= last.end_sec) → 마지막 씬 유지 (과거: scenes[0] 으로 플래시)
    const last = scenes[scenes.length - 1];
    if (currentTime >= last.end_sec) {
      return last;
    }
    return scenes[0];
  }, [timeline, currentTime]);

  useEffect(() => {
    let cancelled = false;
    if (phase !== 'ready') {
      return;
    }
    getJobTimeline(jobId)
      .then(t => {
        if (!cancelled) {
          setTimeline(t);
        }
      })
      .catch(() => {
        // timeline 없어도 재생은 가능 — silent fail
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, phase]);

  // A/B 토글
  const handleToggle = useCallback(
    (target: 'original' | 'processed') => {
      if (target === currentSource) {
        return;
      }
      ignoreProgressRef.current = true;
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
      ignoreProgressRef.current = false;
    },
    [currentTime],
  );

  // 재생 위치 업데이트
  const handleProgress = useCallback((data: OnProgressData) => {
    if (ignoreProgressRef.current) {
      return;
    }
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

  // 네비게이션 헤더를 다크 테마로 + 삭제 버튼
  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerStyle: {backgroundColor: BG},
      headerTintColor: TEXT_PRIMARY,
      headerTitleStyle: {color: TEXT_PRIMARY, fontWeight: '600'},
      headerShadowVisible: false,
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
        <StatusBar barStyle="light-content" backgroundColor={BG} />
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color={ACCENT_ORANGE} />
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
        <StatusBar barStyle="light-content" backgroundColor={BG} />
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

  // --- idle ---
  if (phase === 'idle' || !originalLocalPath || !processedLocalPath) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor={BG} />
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color={ACCENT_ORANGE} />
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
      <StatusBar barStyle="light-content" backgroundColor={BG} />

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

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* 현재 씬 분위기 헤더 카드 */}
        {currentScene && timeline && (
          <MoodHeaderCard
            scene={currentScene}
            totalScenes={timeline.metadata.n_scenes}
            mode={currentSource}
          />
        )}

        {/* 10-band EQ Bar Chart */}
        {currentScene && (
          <SceneEQChart
            bands={currentScene.eq_preset.effective_bands}
            moodName={currentScene.mood.name}
            mode={currentSource}
          />
        )}

        {/* 전문 EQ 주파수 응답 곡선 (Pro-Q4 풍, biquad peaking 합성) */}
        {currentScene && (
          <EQResponseCurve
            bands={currentScene.eq_preset.effective_bands}
            moodName={currentScene.mood.name}
            mode={currentSource}
          />
        )}

        {/* A/B 토글 + 재생 카드 */}
        <View style={styles.controlCard}>
          <Text style={styles.cardLabel}>재생 소스</Text>
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
                  currentSource === 'processed' &&
                    styles.toggleButtonTextActive,
                ]}>
                🎵 EQ 적용
              </Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            style={styles.playPauseButton}
            onPress={() => setPaused(!paused)}
            activeOpacity={0.85}>
            <View style={styles.playPauseDot} />
            <Text style={styles.playPauseText}>
              {paused ? '재생' : '일시정지'}
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// ── 현재 씬 분위기 헤더 카드 ─────────────────────────────────────
function MoodHeaderCard({
  scene,
  totalScenes,
  mode,
}: {
  scene: TimelineScene;
  totalScenes: number;
  mode: 'original' | 'processed';
}) {
  const ko = MOOD_LABELS_KO[scene.mood.name] ?? scene.mood.name;
  const v = scene.va.valence;
  const a = scene.va.arousal;
  const fmt = (n: number) => (n >= 0 ? `+${n.toFixed(2)}` : n.toFixed(2));
  return (
    <View style={headerStyles.card}>
      <View style={headerStyles.topRow}>
        <Text style={headerStyles.sceneIdx}>
          SCENE {scene.scene_idx + 1} / {totalScenes}
        </Text>
        <View
          style={[
            headerStyles.modeBadge,
            mode === 'processed'
              ? headerStyles.modeBadgeOn
              : headerStyles.modeBadgeOff,
          ]}>
          <Text
            style={[
              headerStyles.modeBadgeText,
              mode === 'processed'
                ? headerStyles.modeBadgeTextOn
                : headerStyles.modeBadgeTextOff,
            ]}>
            {mode === 'processed' ? 'EQ 적용' : '원본'}
          </Text>
        </View>
      </View>
      <Text style={headerStyles.moodKo}>{ko}</Text>
      <Text style={headerStyles.moodEn}>{scene.mood.name}</Text>
      <View style={headerStyles.vaRow}>
        <View style={headerStyles.vaItem}>
          <Text style={headerStyles.vaLabel}>Valence</Text>
          <Text style={headerStyles.vaValue}>{fmt(v)}</Text>
        </View>
        <View style={headerStyles.vaDivider} />
        <View style={headerStyles.vaItem}>
          <Text style={headerStyles.vaLabel}>Arousal</Text>
          <Text style={headerStyles.vaValue}>{fmt(a)}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BG,
  },
  scrollContent: {
    paddingBottom: 40,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
    backgroundColor: BG,
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
  controlCard: {
    marginHorizontal: 16,
    marginTop: 12,
    backgroundColor: CARD_BG,
    borderWidth: 1,
    borderColor: CARD_BORDER,
    borderRadius: 14,
    padding: 14,
  },
  cardLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.6,
    color: TEXT_MUTED,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  toggleRow: {
    flexDirection: 'row',
    gap: 10,
  },
  toggleButton: {
    flex: 1,
    paddingVertical: 13,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.05)',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  toggleButtonActive: {
    backgroundColor: 'rgba(255,138,91,0.14)',
    borderColor: ACCENT_ORANGE,
  },
  toggleButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: TEXT_SECONDARY,
  },
  toggleButtonTextActive: {
    color: ACCENT_ORANGE,
  },
  playPauseButton: {
    marginTop: 12,
    paddingVertical: 13,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  playPauseDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
    backgroundColor: ACCENT_ORANGE,
  },
  playPauseText: {
    color: TEXT_PRIMARY,
    fontSize: 15,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
  downloadText: {
    fontSize: 15,
    color: TEXT_PRIMARY,
    marginTop: 18,
  },
  cancelButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  cancelButtonText: {
    color: TEXT_PRIMARY,
    fontSize: 15,
    fontWeight: '600',
  },
  errorIcon: {
    fontSize: 42,
    marginBottom: 4,
  },
  errorText: {
    fontSize: 14,
    color: ERROR_COLOR,
    marginTop: 8,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 20,
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 10,
    backgroundColor: ACCENT_ORANGE_DIM,
    borderWidth: 1,
    borderColor: ACCENT_ORANGE,
  },
  retryButtonText: {
    color: ACCENT_ORANGE,
    fontSize: 15,
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

const headerStyles = StyleSheet.create({
  card: {
    marginHorizontal: 16,
    marginTop: 12,
    backgroundColor: CARD_BG,
    borderWidth: 1,
    borderColor: CARD_BORDER,
    borderLeftWidth: 3,
    borderLeftColor: ACCENT_ORANGE,
    borderRadius: 14,
    padding: 14,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  sceneIdx: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.8,
    color: TEXT_MUTED,
  },
  modeBadge: {
    paddingHorizontal: 9,
    paddingVertical: 3,
    borderRadius: 10,
    borderWidth: 1,
  },
  modeBadgeOn: {
    backgroundColor: 'rgba(255,138,91,0.12)',
    borderColor: ACCENT_ORANGE,
  },
  modeBadgeOff: {
    backgroundColor: 'rgba(158,123,224,0.12)',
    borderColor: ACCENT_PURPLE,
  },
  modeBadgeText: {
    fontSize: 10,
    fontWeight: '700',
  },
  modeBadgeTextOn: {
    color: ACCENT_ORANGE,
  },
  modeBadgeTextOff: {
    color: ACCENT_PURPLE,
  },
  moodKo: {
    fontSize: 24,
    fontWeight: '800',
    color: TEXT_PRIMARY,
    marginTop: 2,
    letterSpacing: 0.5,
  },
  moodEn: {
    fontSize: 11,
    color: TEXT_MUTED,
    marginBottom: 10,
    letterSpacing: 2,
    textTransform: 'uppercase',
  },
  vaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
  },
  vaItem: {
    flex: 1,
    alignItems: 'center',
  },
  vaDivider: {
    width: 1,
    height: 24,
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  vaLabel: {
    fontSize: 10,
    color: TEXT_MUTED,
    marginBottom: 3,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  vaValue: {
    fontSize: 14,
    fontWeight: '700',
    color: TEXT_PRIMARY,
    fontVariant: ['tabular-nums'],
  },
});
