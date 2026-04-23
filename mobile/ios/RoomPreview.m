#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(RoomPreview, NSObject)

// 3D 미리보기 모달 표시
// options: { usdzUri: String, listener: {x,y,z}, speakers: [{label,color,x,y,z}] }
RCT_EXTERN_METHOD(show:(NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
