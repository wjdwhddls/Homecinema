import Foundation
import AVFoundation
import React

/// sweep 신호 재생 + 동시 마이크 녹음 → recorded.wav 저장
/// JS에서 NativeModules.SweepRecorder.record() 로 호출
@objc(SweepRecorder)
class SweepRecorder: NSObject {

  private var audioEngine  = AVAudioEngine()
  private var playerNode   = AVAudioPlayerNode()
  private var recorderFile : AVAudioFile?

  // MARK: - JS 인터페이스

  /// 번들 내 sweep.wav의 file:// URI 반환 (JS에서 서버 업로드에 사용)
  @objc(getSweepUri:rejecter:)
  func getSweepUri(
    _ resolve: RCTPromiseResolveBlock,
    rejecter reject: RCTPromiseRejectBlock
  ) {
    guard let url = Bundle.main.url(forResource: "sweep", withExtension: "wav") else {
      reject("NO_SWEEP", "sweep.wav를 번들에서 찾을 수 없습니다. Xcode 빌드 타깃에 추가했는지 확인하세요.", nil)
      return
    }
    resolve(url.absoluteString)   // "file:///..."
  }

  /// sweep 재생 + 녹음 시작
  /// - sweepAssetName: 앱 번들 내 sweep 파일명 (확장자 제외)
  /// - Returns: 녹음된 recorded.wav 경로 (file://...)
  @objc(record:resolver:rejecter:)
  func record(
    _ sweepAssetName: String,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
      guard let self = self else { return }
      do {
        let outputURL = try self.startRecording(sweepAssetName: sweepAssetName)
        resolve(outputURL.absoluteString)
      } catch {
        reject("SWEEP_ERROR", error.localizedDescription, error)
      }
    }
  }

  // MARK: - 내부 구현

  private func startRecording(sweepAssetName: String) throws -> URL {

    // ── 1. sweep 파일 로드 ────────────────────────────────────────
    guard let sweepURL = Bundle.main.url(
      forResource: sweepAssetName, withExtension: "wav"
    ) else {
      throw NSError(
        domain: "SweepRecorder", code: 1,
        userInfo: [NSLocalizedDescriptionKey:
          "\(sweepAssetName).wav 파일을 찾을 수 없습니다. Xcode 번들 타깃에 추가했는지 확인하세요."]
      )
    }

    let sweepFile = try AVAudioFile(forReading: sweepURL)
    let format    = sweepFile.processingFormat

    // ── 2. AVAudioSession 설정 (재생 + 녹음) ─────────────────────
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord,
                            mode: .measurement,
                            options: [.allowBluetooth, .allowBluetoothA2DP])
    try session.setActive(true)

    // ── 3. 출력 파일 준비 ─────────────────────────────────────────
    let tmpDir    = FileManager.default.temporaryDirectory
    let outputURL = tmpDir.appendingPathComponent("recorded_\(UUID().uuidString).wav")

    let settings: [String: Any] = [
      AVFormatIDKey:          Int(kAudioFormatLinearPCM),
      AVSampleRateKey:        format.sampleRate,
      AVNumberOfChannelsKey:  1,
      AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey:  false,
    ]
    recorderFile = try AVAudioFile(forWriting: outputURL, settings: settings)

    // ── 4. 오디오 그래프 구성 ─────────────────────────────────────
    audioEngine  = AVAudioEngine()
    playerNode   = AVAudioPlayerNode()
    audioEngine.attach(playerNode)
    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)

    let inputNode   = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)

    guard let recFile = recorderFile else {
      throw NSError(domain: "SweepRecorder", code: 2, userInfo: nil)
    }

    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
      try? recFile.write(from: buffer)
    }

    // ── 5. 엔진 시작 + sweep 재생 ────────────────────────────────
    try audioEngine.start()

    let buffer = AVAudioPCMBuffer(
      pcmFormat: format,
      frameCapacity: AVAudioFrameCount(sweepFile.length)
    )!
    try sweepFile.read(into: buffer)

    let semaphore = DispatchSemaphore(value: 0)
    playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
      semaphore.signal()
    }
    playerNode.play()

    // sweep 재생 완료 대기 (최대 30초)
    _ = semaphore.wait(timeout: .now() + 30)

    // 잔향 캡처를 위한 여유 시간
    Thread.sleep(forTimeInterval: 0.5)

    // ── 6. 정리 ──────────────────────────────────────────────────
    inputNode.removeTap(onBus: 0)
    audioEngine.stop()
    try? AVAudioSession.sharedInstance().setActive(false)

    print("SweepRecorder: 녹음 완료 → \(outputURL.path)")
    return outputURL
  }
}
