import UIKit
import RoomPlan
import ARKit

@available(iOS 16.0, *)
class RoomScanViewController: UIViewController, RoomCaptureViewDelegate, RoomCaptureSessionDelegate, ARSessionDelegate {

  var onComplete: ((CapturedRoom, Data?) -> Void)?  // mesh.bin Data 추가
  var onCancel: (() -> Void)?
  var onError: ((Error) -> Void)?

  private var roomCaptureView: RoomCaptureView!
  private var captureConfig = RoomCaptureSession.Configuration()
  private var didFinish = false

  // ARSession (ARMeshAnchor 수집용)
  private var arSession: ARSession!
  private var meshAnchors: [ARMeshAnchor] = []

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .black
    setupARSession()
    setupRoomCaptureView()
    setupButtons()
  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    roomCaptureView?.captureSession.run(configuration: captureConfig)
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    roomCaptureView?.captureSession.stop()
    arSession?.pause()
  }

  // MARK: - ARSession 설정

  private func setupARSession() {
    arSession = ARSession()
    arSession.delegate = self

    let config = ARWorldTrackingConfiguration()
    config.sceneReconstruction = .mesh  // LiDAR 메쉬 수집
    config.environmentTexturing = .none
    arSession.run(config)
  }

  // MARK: - ARSessionDelegate (메쉬 누적)

  func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
    for anchor in anchors.compactMap({ $0 as? ARMeshAnchor }) {
      meshAnchors.append(anchor)
    }
  }

  func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
    for anchor in anchors.compactMap({ $0 as? ARMeshAnchor }) {
      if let idx = meshAnchors.firstIndex(where: { $0.identifier == anchor.identifier }) {
        meshAnchors[idx] = anchor  // 기존 앵커 업데이트
      } else {
        meshAnchors.append(anchor)
      }
    }
  }

  // MARK: - mesh.bin 생성

  private func buildMeshBin() -> Data? {
    guard !meshAnchors.isEmpty else { return nil }

    var allVertices: [SIMD3<Float>] = []
    var allFaces: [SIMD3<Int32>] = []
    var vertexOffset: Int32 = 0

    for anchor in meshAnchors {
      let geometry = anchor.geometry
      let transform = anchor.transform

      // 버텍스 추출 + world space 변환
      let vertexCount = geometry.vertices.count
      let vertexBuffer = geometry.vertices.buffer.contents()
      let vertexStride = geometry.vertices.stride

      for i in 0..<vertexCount {
        let ptr = vertexBuffer.advanced(by: i * vertexStride)
        var localPos = ptr.load(as: SIMD3<Float>.self)

        // world space 변환 (anchor.transform 적용)
        let worldPos = transform * SIMD4<Float>(localPos.x, localPos.y, localPos.z, 1.0)
        allVertices.append(SIMD3<Float>(worldPos.x, worldPos.y, worldPos.z))
      }

      // 페이스(삼각형) 추출
      let faceCount = geometry.faces.count
      let faceBuffer = geometry.faces.buffer.contents()
      let faceStride = geometry.faces.bytesPerIndex

      for i in 0..<faceCount {
        let base = i * 3
        func idx(_ j: Int) -> Int32 {
          let ptr = faceBuffer.advanced(by: (base + j) * faceStride)
          if faceStride == 4 {
            return Int32(ptr.load(as: UInt32.self)) + vertexOffset
          } else {
            return Int32(ptr.load(as: UInt16.self)) + vertexOffset
          }
        }
        allFaces.append(SIMD3<Int32>(idx(0), idx(1), idx(2)))
      }

      vertexOffset += Int32(vertexCount)
    }

    // ── mesh.bin 직렬화 ──────────────────────────────────────
    // [vertex_count: int32][face_count: int32]
    // [vertices: float32 x,y,z * N]
    // [faces: int32 i0,i1,i2 * M]
    var data = Data()

    var vCount = Int32(allVertices.count)
    var fCount = Int32(allFaces.count)
    data.append(Data(bytes: &vCount, count: 4))
    data.append(Data(bytes: &fCount, count: 4))

    for var v in allVertices {
      data.append(Data(bytes: &v.x, count: 4))
      data.append(Data(bytes: &v.y, count: 4))
      data.append(Data(bytes: &v.z, count: 4))
    }

    for var f in allFaces {
      data.append(Data(bytes: &f.x, count: 4))
      data.append(Data(bytes: &f.y, count: 4))
      data.append(Data(bytes: &f.z, count: 4))
    }

    print("mesh.bin 생성 완료: 버텍스 \(allVertices.count)개, 페이스 \(allFaces.count)개")
    return data
  }

  // MARK: - UI 설정

  private func setupRoomCaptureView() {
    roomCaptureView = RoomCaptureView(frame: view.bounds)
    roomCaptureView.captureSession.delegate = self
    roomCaptureView.delegate = self
    roomCaptureView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    view.addSubview(roomCaptureView)
  }

  private func setupButtons() {
    let doneBtn = UIButton(type: .system)
    doneBtn.setTitle("완료", for: .normal)
    doneBtn.titleLabel?.font = .boldSystemFont(ofSize: 18)
    doneBtn.setTitleColor(.white, for: .normal)
    doneBtn.backgroundColor = .systemBlue
    doneBtn.layer.cornerRadius = 12
    doneBtn.addTarget(self, action: #selector(doneTapped), for: .touchUpInside)
    doneBtn.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(doneBtn)

    let cancelBtn = UIButton(type: .system)
    cancelBtn.setTitle("취소", for: .normal)
    cancelBtn.titleLabel?.font = .boldSystemFont(ofSize: 18)
    cancelBtn.setTitleColor(.white, for: .normal)
    cancelBtn.backgroundColor = .systemGray
    cancelBtn.layer.cornerRadius = 12
    cancelBtn.addTarget(self, action: #selector(cancelTapped), for: .touchUpInside)
    cancelBtn.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(cancelBtn)

    NSLayoutConstraint.activate([
      doneBtn.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
      doneBtn.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
      doneBtn.widthAnchor.constraint(equalToConstant: 100),
      doneBtn.heightAnchor.constraint(equalToConstant: 50),

      cancelBtn.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
      cancelBtn.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
      cancelBtn.widthAnchor.constraint(equalToConstant: 100),
      cancelBtn.heightAnchor.constraint(equalToConstant: 50),
    ])
  }

  @objc private func doneTapped() {
    guard !didFinish else { return }
    roomCaptureView.captureSession.stop()
  }

  @objc private func cancelTapped() {
    guard !didFinish else { return }
    didFinish = true
    roomCaptureView.captureSession.stop()
    arSession.pause()
    dismiss(animated: true) { [weak self] in
      self?.onCancel?()
    }
  }

  // MARK: - RoomCaptureViewDelegate

  func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
    return true
  }

  func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
    guard !didFinish else { return }
    didFinish = true
    arSession.pause()

    if let error = error {
      dismiss(animated: true) { [weak self] in self?.onError?(error) }
      return
    }

    // mesh.bin 생성
    let meshData = buildMeshBin()

    dismiss(animated: true) { [weak self] in
      self?.onComplete?(processedResult, meshData)
    }
  }
}
