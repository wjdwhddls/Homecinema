import Foundation
import RoomPlan
import React
import simd

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

      var topVC = rootVC
      while let presented = topVC.presentedViewController {
        topVC = presented
      }

      let scanVC = RoomScanViewController()
      scanVC.modalPresentationStyle = .fullScreen

      // onComplete: CapturedRoom + mesh.bin Data 받음
      scanVC.onComplete = { [weak self] capturedRoom, meshData in
        guard let self = self else { return }

        if capturedRoom.walls.count < 3 {
          reject(
            "SCAN_INSUFFICIENT",
            "벽이 충분히 인식되지 않았습니다 (\(capturedRoom.walls.count)개 / 최소 3개 필요). 방을 더 둘러보며 다시 스캔해주세요.",
            nil
          )
          return
        }

        var json = self.encodeRoom(capturedRoom)

        // mesh.bin 임시 파일로 저장 → URI 반환
        if let data = meshData {
          let meshURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mesh_\(Int(Date().timeIntervalSince1970)).bin")
          do {
            try data.write(to: meshURL)
            json["meshBinUri"] = meshURL.absoluteString
            print("mesh.bin 저장 완료: \(meshURL.path) (\(data.count) bytes)")
          } catch {
            print("mesh.bin 저장 실패: \(error)")
            // mesh.bin 저장 실패해도 스캔 결과는 반환 (fallback 가능)
          }
        } else {
          print("mesh.bin 없음 (ARMeshAnchor 수집 실패 또는 미지원 기기)")
        }

        print("========== ROOM SCAN RESULT ==========")
        print("Walls: \(capturedRoom.walls.count)")
        print("Doors: \(capturedRoom.doors.count)")
        print("Windows: \(capturedRoom.windows.count)")
        print("Objects: \(capturedRoom.objects.count)")
        print("meshBinUri: \(json["meshBinUri"] ?? "없음")")
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
    return [
      "id": s.identifier.uuidString,
      "category": surfaceCategoryString(s.category),
      "dimensions": [s.dimensions.x, s.dimensions.y, s.dimensions.z],
      "transform": flattenMatrix(s.transform),
      "confidence": confidenceString(s.confidence)
    ]
  }

  @available(iOS 16.0, *)
  private func encodeObject(_ o: CapturedRoom.Object) -> [String: Any] {
    return [
      "id": o.identifier.uuidString,
      "category": objectCategoryString(o.category),
      "dimensions": [o.dimensions.x, o.dimensions.y, o.dimensions.z],
      "transform": flattenMatrix(o.transform),
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

  private func flattenMatrix(_ m: simd_float4x4) -> [Float] {
    return [
      m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w,
      m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
      m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w,
      m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w
    ]
  }
}
