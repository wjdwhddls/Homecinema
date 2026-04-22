// navigation/AppNavigator.tsx — 앱 네비게이션 구조
import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {RootStackParamList} from '../types';
import {COLORS} from '../constants/colors';

import HomeScreen from '../screens/HomeScreen';
import UploadScreen from '../screens/UploadScreen';
import ResultScreen from '../screens/ResultScreen';
import PlaybackScreen from '../screens/PlaybackScreen';
import SpeakerSizeScreen from '../screens/SpeakerSizeScreen';
import SpeakerPlacementScreen from '../screens/SpeakerPlacementScreen';
import OptimizationResultScreen from '../screens/OptimizationResultScreen';
import EQMeasurementScreen from '../screens/EQMeasurementScreen';  // ← 추가

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <Stack.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerStyle: {
          backgroundColor: COLORS.surface,
        },
        headerTintColor: COLORS.text,
        headerTitleStyle: {
          fontWeight: '600',
        },
        contentStyle: {
          backgroundColor: COLORS.background,
        },
      }}>
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{headerShown: false}}
      />
      <Stack.Screen
        name="Upload"
        component={UploadScreen}
        options={{title: '영상 업로드'}}
      />
      <Stack.Screen
        name="Result"
        component={ResultScreen}
        options={{
          title: '분석 진행 상황',
          headerBackVisible: false,
        }}
      />
      <Stack.Screen
        name="Playback"
        component={PlaybackScreen}
        options={{title: '영상 재생'}}
      />
      <Stack.Screen
        name="SpeakerSize"
        component={SpeakerSizeScreen}
        options={{title: '스피커 정보 입력'}}
      />
      <Stack.Screen
        name="SpeakerPlacement"
        component={SpeakerPlacementScreen}
        options={{title: '스피커 위치 자동 배정'}}
      />
      <Stack.Screen
        name="OptimizationResult"
        component={OptimizationResultScreen}
        options={{title: '최적 배치 결과'}}
      />
      <Stack.Screen
        name="EQMeasurement"
        component={EQMeasurementScreen}
        options={{title: 'EQ 자동 보정'}}
      />
    </Stack.Navigator>
  );
}
