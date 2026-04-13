// screens/UploadScreen.tsx — 영상 업로드 화면
import React, {useState} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
} from 'react-native';
import DocumentPicker, {types} from 'react-native-document-picker';
import {NativeStackScreenProps} from '@react-navigation/native-stack';
import {RootStackParamList, SelectedFile} from '../types';
import {uploadVideo} from '../api/upload';
import {COLORS} from '../constants/colors';

type Props = NativeStackScreenProps<RootStackParamList, 'Upload'>;

export default function UploadScreen({navigation}: Props) {
  const [selectedFile, setSelectedFile] = useState<SelectedFile | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // 파일 선택 (document-picker v9+ API)
  const handlePickFile = async () => {
    try {
      const result = await DocumentPicker.pickSingle({
        type: [types.video],
        copyTo: 'cachesDirectory', // iOS 필수
      });
      setSelectedFile({
        uri: result.uri,
        fileCopyUri: result.fileCopyUri ?? null,
        name: result.name ?? null,
        size: result.size ?? null,
        type: result.type ?? null,
      });
    } catch (err) {
      if (DocumentPicker.isCancel(err)) {
        return; // 사용자 취소
      }
      Alert.alert('오류', '파일 선택 중 오류가 발생했습니다.');
    }
  };

  // 업로드 실행
  const handleUpload = async () => {
    if (!selectedFile) {
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await uploadVideo(selectedFile, progress => {
        setUploadProgress(progress);
      });

      Alert.alert('업로드 완료', response.message, [
        {
          text: '확인',
          onPress: () => {
            navigation.replace('Result', {jobId: response.job_id});
          },
        },
      ]);
    } catch (err: any) {
      Alert.alert(
        '업로드 실패',
        err.message || '업로드 중 오류가 발생했습니다.',
      );
    } finally {
      setIsUploading(false);
    }
  };

  // 파일 크기 포맷 (MB)
  const formatSize = (bytes: number | null): string => {
    if (bytes === null) {
      return '크기 알 수 없음';
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {/* 안내 텍스트 */}
        <Text style={styles.guide}>
          분위기 분석을 위한 영상을 선택해주세요
        </Text>

        {/* 영상 선택 버튼 */}
        <TouchableOpacity
          style={[styles.pickButton, isUploading && styles.disabledButton]}
          onPress={handlePickFile}
          disabled={isUploading}
          activeOpacity={0.8}>
          <Text style={styles.pickButtonText}>영상 선택</Text>
        </TouchableOpacity>

        {/* 선택된 파일 정보 */}
        {selectedFile && (
          <View style={styles.fileInfo}>
            <Text style={styles.fileName} numberOfLines={1}>
              {selectedFile.name || '파일명 없음'}
            </Text>
            <Text style={styles.fileSize}>
              {formatSize(selectedFile.size)}
            </Text>
          </View>
        )}

        {/* 업로드 버튼 */}
        <TouchableOpacity
          style={[
            styles.uploadButton,
            (!selectedFile || isUploading) && styles.disabledButton,
          ]}
          onPress={handleUpload}
          disabled={!selectedFile || isUploading}
          activeOpacity={0.8}>
          {isUploading ? (
            <View style={styles.uploadingRow}>
              <ActivityIndicator color={COLORS.textInverse} size="small" />
              <Text style={styles.uploadButtonText}>
                업로드 중... {Math.round(uploadProgress * 100)}%
              </Text>
            </View>
          ) : (
            <Text style={styles.uploadButtonText}>업로드</Text>
          )}
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
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  guide: {
    fontSize: 16,
    color: COLORS.textSecondary,
    marginBottom: 32,
    textAlign: 'center',
  },
  pickButton: {
    width: '100%',
    paddingVertical: 40,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: COLORS.primary,
    borderStyle: 'dashed',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
  },
  pickButtonText: {
    fontSize: 20,
    fontWeight: '600',
    color: COLORS.primary,
  },
  fileInfo: {
    marginTop: 20,
    padding: 16,
    backgroundColor: COLORS.surface,
    borderRadius: 8,
    width: '100%',
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  fileName: {
    fontSize: 14,
    color: COLORS.text,
    fontWeight: '500',
  },
  fileSize: {
    fontSize: 13,
    color: COLORS.textSecondary,
    marginTop: 4,
  },
  uploadButton: {
    marginTop: 24,
    width: '100%',
    paddingVertical: 16,
    borderRadius: 12,
    backgroundColor: COLORS.buttonPrimary,
    alignItems: 'center',
  },
  uploadButtonText: {
    color: COLORS.textInverse,
    fontSize: 18,
    fontWeight: '600',
  },
  uploadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  disabledButton: {
    backgroundColor: COLORS.buttonDisabled,
    borderColor: COLORS.buttonDisabled,
  },
});
