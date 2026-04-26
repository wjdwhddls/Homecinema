import UIKit
import SceneKit
import QuartzCore

/// 3D 방 미리보기 (RoomPlan USDZ + 청취자/스피커 마커 오버레이)
///
/// 좌표계 규약
/// - SceneKit: Y-up 오른손좌표 (x=좌우, y=높이, z=앞뒤)
/// - RoomPlan / USDZ: 동일 Y-up → 그대로 로드
/// - xRIR 백엔드: Z-up (x=좌우, y=앞뒤, z=높이) → `xrirToScene(_:)`으로 변환
class RoomPreviewViewController: UIViewController {

  // MARK: - 데이터 모델

  /// 씬에 표시할 마커
  struct Marker {
    let label: String
    let color: UIColor
    let position: SCNVector3  // SceneKit 좌표 (Y-up)
    let isSphere: Bool        // true=구(청취자), false=박스(스피커)
    /// 박스 크기 (m, SceneKit 단위). nil=기본 크기 사용. isSphere=true면 무시.
    let dimensions: (width: CGFloat, height: CGFloat, depth: CGFloat)?
  }

  // MARK: - 입력

  var usdzURL: URL!
  var markers: [Marker] = []
  var onClose: (() -> Void)?

  // MARK: - View

  private var scnView: SCNView!

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = UIColor(red: 0.10, green: 0.10, blue: 0.18, alpha: 1.0)
    setupSceneView()
    loadSceneAndMarkers()
    setupCloseButton()
  }

  // MARK: - 좌표 변환

  /// xRIR 좌표(Z-up, y 앞뒤 반전) → SceneKit 좌표(Y-up)
  /// sceneX = xRIR.x, sceneY = xRIR.z(높이), sceneZ = -xRIR.y(앞뒤 부호 반전)
  static func xrirToScene(x: Float, y: Float, z: Float) -> SCNVector3 {
    return SCNVector3(x: x, y: z, z: -y)
  }

  // MARK: - Setup

  private func setupSceneView() {
    scnView = SCNView(frame: view.bounds)
    scnView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    scnView.backgroundColor = UIColor(red: 0.10, green: 0.10, blue: 0.18, alpha: 1.0)
    scnView.allowsCameraControl = true         // 기본 제스처(회전/줌/팬)
    scnView.autoenablesDefaultLighting = true
    scnView.antialiasingMode = .multisampling4X
    view.addSubview(scnView)
  }

  private func loadSceneAndMarkers() {
    let scene: SCNScene
    do {
      scene = try SCNScene(url: usdzURL, options: [
        .checkConsistency: true,
        .convertToYUp: true,  // RoomPlan USDZ가 이미 Y-up이지만 안전장치
      ])
    } catch {
      print("USDZ 로드 실패: \(error)")
      // 빈 씬으로 폴백 (마커만 표시)
      scene = SCNScene()
      scene.background.contents = UIColor(red: 0.10, green: 0.10, blue: 0.18, alpha: 1.0)
    }

    // 마커 추가 (USDZ 원본 재질은 그대로 유지 — Apple RoomPlan 공식 렌더)
    for marker in markers {
      scene.rootNode.addChildNode(makeMarkerNode(marker))
    }

    // 바닥 기준 그리드 (옵션 - 시각적 참조용)
    scene.rootNode.addChildNode(makeFloorGrid())

    // 카메라 세팅 (마커/방을 적당히 담도록 초기 시점)
    if scene.rootNode.childNodes.contains(where: { $0.camera != nil }) == false {
      let cameraNode = SCNNode()
      cameraNode.camera = SCNCamera()
      cameraNode.camera?.zFar = 100
      cameraNode.position = SCNVector3(0, 3.0, 5.0)  // 청취자 뒤쪽 위에서
      cameraNode.look(at: SCNVector3(0, 1.0, 0))
      scene.rootNode.addChildNode(cameraNode)
    }

    scnView.scene = scene
  }

  // MARK: - Marker / Material

  private func makeMarkerNode(_ m: Marker) -> SCNNode {
    let node = SCNNode()
    node.position = m.position

    // 지오메트리
    if m.isSphere {
      let sphere = SCNSphere(radius: 0.12)
      sphere.firstMaterial?.diffuse.contents = m.color
      sphere.firstMaterial?.emission.contents = m.color.withAlphaComponent(0.35)
      node.geometry = sphere
    } else {
      // 사용자가 입력한 스피커 치수가 있으면 그 크기로, 없으면 기본 0.22×0.32×0.22
      // SCNBox: width(좌우=x) × height(높이=y) × length(앞뒤=z)
      let w: CGFloat = m.dimensions?.width  ?? 0.22
      let h: CGFloat = m.dimensions?.height ?? 0.32
      let d: CGFloat = m.dimensions?.depth  ?? 0.22
      // chamferRadius 는 박스의 가장 짧은 변보다 작아야 안전 (5% 정도로 축소)
      let chamfer = min(w, h, d) * 0.05
      let box = SCNBox(width: w, height: h, length: d, chamferRadius: chamfer)
      box.firstMaterial?.diffuse.contents = m.color
      box.firstMaterial?.emission.contents = m.color.withAlphaComponent(0.35)
      node.geometry = box
    }

    // 라벨 (카메라를 따라 회전 - billboard)
    // 박스 정수리 위 약간 띄움. 큰 스피커도 라벨이 묻히지 않도록.
    let labelY: Float = m.isSphere
      ? 0.35
      : Float((m.dimensions?.height ?? 0.32) * 0.5) + 0.15
    let label = makeLabelNode(m.label, color: m.color)
    label.position = SCNVector3(0, labelY, 0)
    node.addChildNode(label)

    // 바닥 링 (위치 인식 보조)
    let ring = makeGroundRing(color: m.color)
    ring.position = SCNVector3(0, -m.position.y + 0.01, 0)  // 월드 바닥(y=0) 바로 위
    node.addChildNode(ring)

    return node
  }

  private func makeLabelNode(_ text: String, color: UIColor) -> SCNNode {
    let scnText = SCNText(string: text, extrusionDepth: 0.0)
    scnText.font = .boldSystemFont(ofSize: 6)
    scnText.flatness = 0.2
    scnText.firstMaterial?.diffuse.contents = UIColor.white
    scnText.firstMaterial?.isDoubleSided = true

    let node = SCNNode(geometry: scnText)
    // SCNText는 기본 단위가 포인트라 SceneKit 미터에 맞게 축소
    node.scale = SCNVector3(0.02, 0.02, 0.02)

    // 텍스트 중심 정렬
    let (minV, maxV) = scnText.boundingBox
    let dx = (maxV.x - minV.x) * 0.5
    node.pivot = SCNMatrix4MakeTranslation(dx + minV.x, 0, 0)

    // 카메라를 항상 바라보도록
    let billboard = SCNBillboardConstraint()
    billboard.freeAxes = .Y
    node.constraints = [billboard]

    // 배경 패널 (가독성)
    let panel = SCNPlane(width: CGFloat((maxV.x - minV.x) * 0.02 + 0.12),
                         height: 0.09)
    panel.cornerRadius = 0.02
    panel.firstMaterial?.diffuse.contents = color.withAlphaComponent(0.85)
    panel.firstMaterial?.isDoubleSided = true
    let panelNode = SCNNode(geometry: panel)
    panelNode.position = SCNVector3(0, 0.03, -0.001)  // 텍스트 뒤에
    panelNode.constraints = [billboard]

    let container = SCNNode()
    container.addChildNode(panelNode)
    container.addChildNode(node)
    return container
  }

  private func makeGroundRing(color: UIColor) -> SCNNode {
    let torus = SCNTorus(ringRadius: 0.18, pipeRadius: 0.015)
    torus.firstMaterial?.diffuse.contents = color
    torus.firstMaterial?.emission.contents = color.withAlphaComponent(0.5)
    return SCNNode(geometry: torus)
  }

  private func makeFloorGrid() -> SCNNode {
    // 1m x 1m 반투명 격자 (10x10m)
    let plane = SCNPlane(width: 10, height: 10)
    let mat = SCNMaterial()
    mat.diffuse.contents = UIColor(red: 0.15, green: 0.15, blue: 0.25, alpha: 0.3)
    mat.isDoubleSided = true
    plane.materials = [mat]

    let node = SCNNode(geometry: plane)
    node.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)  // 바닥에 깔기
    node.position = SCNVector3(0, 0, 0)
    return node
  }

  /// USDZ 내부 재질을 반투명 화이트로 덮어써서 마커가 잘 보이게
  private func applyRoomMaterial(_ node: SCNNode) {
    if let geometry = node.geometry {
      let mat = SCNMaterial()
      mat.diffuse.contents = UIColor(white: 0.85, alpha: 0.55)
      mat.isDoubleSided = true
      mat.blendMode = .alpha
      geometry.materials = [mat]
    }
    for child in node.childNodes {
      applyRoomMaterial(child)
    }
  }

  // MARK: - Close button

  private func setupCloseButton() {
    let closeBtn = UIButton(type: .system)
    closeBtn.setTitle("닫기", for: .normal)
    closeBtn.titleLabel?.font = .boldSystemFont(ofSize: 16)
    closeBtn.setTitleColor(.white, for: .normal)
    closeBtn.backgroundColor = UIColor.black.withAlphaComponent(0.6)
    closeBtn.layer.cornerRadius = 10
    closeBtn.translatesAutoresizingMaskIntoConstraints = false
    closeBtn.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)
    view.addSubview(closeBtn)

    NSLayoutConstraint.activate([
      closeBtn.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
      closeBtn.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
      closeBtn.widthAnchor.constraint(equalToConstant: 72),
      closeBtn.heightAnchor.constraint(equalToConstant: 40),
    ])
  }

  @objc private func closeTapped() {
    dismiss(animated: true) { [weak self] in
      self?.onClose?()
    }
  }
}
