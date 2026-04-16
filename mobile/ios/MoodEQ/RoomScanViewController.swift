import UIKit
import RoomPlan

@available(iOS 16.0, *)
class RoomScanViewController: UIViewController, RoomCaptureViewDelegate, RoomCaptureSessionDelegate {

  var onComplete: ((CapturedRoom) -> Void)?
  var onCancel: (() -> Void)?
  var onError: ((Error) -> Void)?

  private var roomCaptureView: RoomCaptureView!
  private var captureConfig = RoomCaptureSession.Configuration()
  private var didFinish = false

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .black
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
  }

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
    if let error = error {
      dismiss(animated: true) { [weak self] in self?.onError?(error) }
      return
    }
    dismiss(animated: true) { [weak self] in
      self?.onComplete?(processedResult)
    }
  }
}
