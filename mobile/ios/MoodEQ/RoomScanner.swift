import Foundation
import RoomPlan
import React
import UIKit

@available(iOS 16.0, *)
@objc(RoomScanner)
class RoomScanner: RCTEventEmitter {

  override func supportedEvents() -> [String]! {
    return ["onScanProgress", "onScanComplete", "onScanError"]
  }

  override static func requiresMainQueueSetup() -> Bool { true }

  // MARK: - 디바이스 지원 여부 확인
  @objc(isSupported:rejecter:)
  func isSupported(_ resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
    if #available(iOS 16.0, *) {
      resolve(RoomCaptureSession.isSupported)
    } else {
      resolve(false)
    }
  }

  // MARK: - 스캔 시작
  @objc(startScan:rejecter:)
  func startScan(_ resolve: @escaping RCTPromiseResolveBlock,
                 rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async {
      guard #available(iOS 16.0, *), RoomCaptureSession.isSupported else {
        reject("UNSUPPORTED", "이 기기는 LiDAR를 지원하지 않습니다", nil)
        return
      }

      guard let rootVC = UIApplication.shared.windows.first(where: { $0.isKeyWindow })?.rootViewController else {
        reject("NO_VC", "ViewController를 찾을 수 없습니다", nil)
        return
      }

      // 이미 모달이 떠 있으면 그 위에 띄우기
      var topVC = rootVC
      while let presented = topVC.presentedViewController {
        topVC = presented
      }

      let scanVC = RoomScanViewController()
      scanVC.modalPresentationStyle = .fullScreen
      scanVC.onComplete = { [weak self] capturedRoom in
        guard let self = self else { return }

        // 스캔 결과 충분성 검증 (Phase 2/3 음향 분석 정확도를 위한 최소 조건)
        if capturedRoom.walls.count < 3 {
          reject(
            "SCAN_INSUFFICIENT",
            "벽이 충분히 인식되지 않았습니다 (\(capturedRoom.walls.count)개 / 최소 3개 필요). 방을 더 둘러보며 다시 스캔해주세요.",
            nil
          )
          return
        }

        let json = self.encodeRoom(capturedRoom)
        print("========== ROOM SCAN RESULT ==========")
        print("Walls: \(capturedRoom.walls.count)")
        print("Doors: \(capturedRoom.doors.count)")
        print("Windows: \(capturedRoom.windows.count)")
        print("Objects: \(capturedRoom.objects.count)")
        for obj in capturedRoom.objects {
          print("• \(obj.category) | size: \(obj.dimensions.x)×\(obj.dimensions.y)×\(obj.dimensions.z)m | confidence: \(obj.confidence)")
        }
        print("======================================")
        resolve(json)
      }
      scanVC.onCancel = {
        reject("CANCELLED", "사용자가 스캔을 취소했습니다", nil)
      }
      scanVC.onError = { error in
        reject("SCAN_ERROR", error.localizedDescription, error)
      }
      topVC.present(scanVC, animated: true)
    }
  }

  // MARK: - CapturedRoom → JSON 직렬화
  @available(iOS 16.0, *)
  private func encodeRoom(_ room: CapturedRoom) -> [String: Any] {
    return [
      "walls": room.walls.map(encodeSurface),
      "doors": room.doors.map(encodeSurface),
      "windows": room.windows.map(encodeSurface),
      "openings": room.openings.map(encodeSurface),
      "objects": room.objects.map(encodeObject),
      "scannedAt": ISO8601DateFormatter().string(from: Date())
    ]
  }

  @available(iOS 16.0, *)
  private func encodeSurface(_ s: CapturedRoom.Surface) -> [String: Any] {
    let t = s.transform
    return [
      "id": s.identifier.uuidString,
      "category": surfaceCategoryString(s.category),
      "dimensions": [s.dimensions.x, s.dimensions.y, s.dimensions.z],
      "transform": [
        t.columns.0.x, t.columns.0.y, t.columns.0.z, t.columns.0.w,
        t.columns.1.x, t.columns.1.y, t.columns.1.z, t.columns.1.w,
        t.columns.2.x, t.columns.2.y, t.columns.2.z, t.columns.2.w,
        t.columns.3.x, t.columns.3.y, t.columns.3.z, t.columns.3.w
      ],
      "confidence": confidenceString(s.confidence)
    ]
  }

  @available(iOS 16.0, *)
  private func encodeObject(_ o: CapturedRoom.Object) -> [String: Any] {
    let t = o.transform
    return [
      "id": o.identifier.uuidString,
      "category": objectCategoryString(o.category),
      "dimensions": [o.dimensions.x, o.dimensions.y, o.dimensions.z],
      "transform": [
        t.columns.0.x, t.columns.0.y, t.columns.0.z, t.columns.0.w,
        t.columns.1.x, t.columns.1.y, t.columns.1.z, t.columns.1.w,
        t.columns.2.x, t.columns.2.y, t.columns.2.z, t.columns.2.w,
        t.columns.3.x, t.columns.3.y, t.columns.3.z, t.columns.3.w
      ],
      "confidence": confidenceString(o.confidence)
    ]
  }

  @available(iOS 16.0, *)
  private func surfaceCategoryString(_ c: CapturedRoom.Surface.Category) -> String {
    switch c {
    case .wall: return "wall"
    case .door: return "door"
    case .window: return "window"
    case .opening: return "opening"
    case .floor: return "floor"
    @unknown default: return "unknown"
    }
  }

  @available(iOS 16.0, *)
  private func objectCategoryString(_ c: CapturedRoom.Object.Category) -> String {
    switch c {
    case .storage: return "storage"
    case .refrigerator: return "refrigerator"
    case .stove: return "stove"
    case .bed: return "bed"
    case .sink: return "sink"
    case .washerDryer: return "washerDryer"
    case .toilet: return "toilet"
    case .bathtub: return "bathtub"
    case .oven: return "oven"
    case .dishwasher: return "dishwasher"
    case .table: return "table"
    case .sofa: return "sofa"
    case .chair: return "chair"
    case .fireplace: return "fireplace"
    case .television: return "television"
    case .stairs: return "stairs"
    @unknown default: return "unknown"
    }
  }

  @available(iOS 16.0, *)
  private func confidenceString(_ c: CapturedRoom.Confidence) -> String {
    switch c {
    case .low: return "low"
    case .medium: return "medium"
    case .high: return "high"
    @unknown default: return "unknown"
    }
  }

}
