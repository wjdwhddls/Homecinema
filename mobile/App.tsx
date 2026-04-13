// App.tsx — 앱 진입점
import React from 'react';
import {StatusBar} from 'react-native';
import {NavigationContainer} from '@react-navigation/native';
import AppNavigator from './src/navigation/AppNavigator';
import {COLORS} from './src/constants/colors';

function App(): React.JSX.Element {
  return (
    <NavigationContainer>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.surface} />
      <AppNavigator />
    </NavigationContainer>
  );
}

export default App;
