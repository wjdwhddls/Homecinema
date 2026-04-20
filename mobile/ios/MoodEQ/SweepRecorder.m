#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(SweepRecorder, NSObject)

// 번들 내 sweep.wav의 file:// URI 반환
RCT_EXTERN_METHOD(getSweepUri:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

// sweep 재생 + 마이크 녹음
RCT_EXTERN_METHOD(record:(NSString *)sweepAssetName
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
