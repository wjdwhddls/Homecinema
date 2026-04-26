import Foundation
import UIKit
import React

/// React Native → Swift 브릿지
/// JS에서 RoomPreview.show({...})로 호출하면 USDZ 3D 뷰어를 모달로 띄움
///
/// 좌표계: JS는 xRIR(Z-up) 좌표로 넘김 → Swift가 SceneKit(Y-up)으로 변환
@objc(RoomPreview)
class RoomPreview: NSObject {

  @objc static func requiresMainQueueSetup() -> Bool { true }

  /// JS 호출: RoomPreview.show({ usdzUri, listener, speakers })
  ///
  /// - usdzUri  : "file:///..." 문자열
  /// - listener : { x, y, z } (xRIR Z-up)
  /// - speakers : [{ label, color, x, y, z }]  (xRIR Z-up)
  @objc(show:resolver:rejecter:)
  func show(
    _ options: NSDictionary,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async {
      // 1. usdzUri 파싱
      guard let uriString = options["usdzUri"] as? String,
            let url = URL(string: uriString) else {
        reject("INVALID_URI", "usdzUri가 유효하지 않습니다", nil)
        return
      }
      guard FileManager.default.fileExists(atPath: url.path) else {
        reject("FILE_NOT_FOUND", "USDZ 파일을 찾을 수 없습니다: \(url.path)", nil)
        return
      }

      // 2. 마커 파싱 (xRIR 좌표 → SceneKit 좌표 변환)
      var markers: [RoomPreviewViewController.Marker] = []

      if let listener = options["listener"] as? NSDictionary,
         let pos = Self.parseXRIRPosition(listener) {
        markers.append(.init(
          label: "청취자",
          color: UIColor(red: 0.00, green: 0.83, blue: 1.00, alpha: 1.0),
          position: RoomPreviewViewController.xrirToScene(x: pos.x, y: pos.y, z: pos.z),
          isSphere: true,
          dimensions: nil
        ))
      }

      if let speakers = options["speakers"] as? [NSDictionary] {
        for sp in speakers {
          guard let pos = Self.parseXRIRPosition(sp) else { continue }
          let label = (sp["label"] as? String) ?? "스피커"
          let colorHex = (sp["color"] as? String) ?? "#ffd700"
          markers.append(.init(
            label: label,
            color: UIColor(hex: colorHex) ?? UIColor.yellow,
            position: RoomPreviewViewController.xrirToScene(x: pos.x, y: pos.y, z: pos.z),
            isSphere: false,
            dimensions: Self.parseDimensions(sp["dimensions"])
          ))
        }
      }

      // 3. 최상위 ViewController 찾기
      guard let rootVC = UIApplication.shared.windows
              .first(where: { $0.isKeyWindow })?
              .rootViewController else {
        reject("NO_VC", "ViewController를 찾을 수 없습니다", nil)
        return
      }
      var topVC = rootVC
      while let presented = topVC.presentedViewController {
        topVC = presented
      }

      // 4. 3D 뷰어 presnet
      let vc = RoomPreviewViewController()
      vc.usdzURL = url
      vc.markers = markers
      vc.modalPresentationStyle = .fullScreen
      vc.onClose = { resolve(nil) }
      topVC.present(vc, animated: true)
    }
  }

  // MARK: - 헬퍼

  private static func parseXRIRPosition(_ dict: NSDictionary) -> (x: Float, y: Float, z: Float)? {
    guard let x = (dict["x"] as? NSNumber)?.floatValue,
          let y = (dict["y"] as? NSNumber)?.floatValue,
          let z = (dict["z"] as? NSNumber)?.floatValue else {
      return nil
    }
    return (x, y, z)
  }

  /// JS 측 { width_m, height_m, depth_m } → Marker.dimensions
  private static func parseDimensions(_ raw: Any?) -> (width: CGFloat, height: CGFloat, depth: CGFloat)? {
    guard let dict = raw as? NSDictionary,
          let w = (dict["width_m"]  as? NSNumber)?.doubleValue,
          let h = (dict["height_m"] as? NSNumber)?.doubleValue,
          let d = (dict["depth_m"]  as? NSNumber)?.doubleValue,
          w > 0, h > 0, d > 0 else {
      return nil
    }
    return (CGFloat(w), CGFloat(h), CGFloat(d))
  }
}

// MARK: - UIColor hex 확장

private extension UIColor {
  /// "#RRGGBB" / "RRGGBB" → UIColor
  convenience init?(hex: String) {
    var s = hex.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
    if s.hasPrefix("#") { s.removeFirst() }
    guard s.count == 6, let v = UInt32(s, radix: 16) else { return nil }
    let r = CGFloat((v >> 16) & 0xFF) / 255.0
    let g = CGFloat((v >>  8) & 0xFF) / 255.0
    let b = CGFloat( v        & 0xFF) / 255.0
    self.init(red: r, green: g, blue: b, alpha: 1.0)
  }
}
