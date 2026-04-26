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
  private var firstTapWriteError: Error?

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
    let sweepFormat = sweepFile.processingFormat

    // ── 2. AVAudioSession 설정 (재생 + 녹음) ─────────────────────
    // .measurement 모드는 내장 하드웨어를 강제해 블루투스 라우팅을 차단한다.
    // .allowBluetooth(HFP)는 녹음 활성화 시 HFP로 폴백돼 대부분의 BT 스피커에서 무음이 된다.
    // A2DP 단독으로 두어 홈시어터 경로(BT 스피커 출력 + 핸드폰 내장 마이크)를 측정한다.
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord,
                            mode: .default,
                            options: [.allowBluetoothA2DP])
    try session.setActive(true)

    // 실제 라우팅 확인 로그
    for output in session.currentRoute.outputs {
      print("SweepRecorder: active output = \(output.portType.rawValue) (\(output.portName))")
    }

    // 블루투스 A2DP 미연결 시 측정 중단 (내장 스피커 → 내장 마이크는 무의미한 측정)
    let hasBluetoothOutput = session.currentRoute.outputs.contains {
      $0.portType == .bluetoothA2DP
    }
    if !hasBluetoothOutput {
      try? session.setActive(false)
      throw NSError(
        domain: "SweepRecorder", code: 3,
        userInfo: [NSLocalizedDescriptionKey:
          "블루투스 스피커가 연결되어 있지 않습니다. 홈시어터 측정을 위해 블루투스 스피커를 먼저 연결하세요."]
      )
    }

    // ── 3. 오디오 그래프 구성 (inputFormat을 알아야 출력 파일을 만들 수 있음) ──
    audioEngine  = AVAudioEngine()
    playerNode   = AVAudioPlayerNode()
    audioEngine.attach(playerNode)
    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: sweepFormat)

    let inputNode   = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)
    print("SweepRecorder: inputFormat sr=\(inputFormat.sampleRate) ch=\(inputFormat.channelCount)")

    // ── 4. 출력 파일 준비 — 마이크 입력 포맷에 맞춤 ─────────────────
    // 이전 버전은 sweep 포맷(SR/1ch)을 강제해서 inputFormat과 mismatch 발생 →
    // tap 안의 try? recFile.write(...) 가 매번 silent fail → 0-frame WAV 생성.
    // SR/채널 수를 inputFormat과 맞추면 buffer.format 과 file.processingFormat 호환.
    let tmpDir    = FileManager.default.temporaryDirectory
    let outputURL = tmpDir.appendingPathComponent("recorded_\(UUID().uuidString).wav")

    let settings: [String: Any] = [
      AVFormatIDKey:          Int(kAudioFormatLinearPCM),
      AVSampleRateKey:        inputFormat.sampleRate,
      AVNumberOfChannelsKey:  inputFormat.channelCount,
      AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey:  false,
    ]
    recorderFile = try AVAudioFile(forWriting: outputURL, settings: settings)

    guard recorderFile != nil else {
      throw NSError(domain: "SweepRecorder", code: 2, userInfo: nil)
    }

    // ── 5. 녹음 tap 설치 — write 실패를 무시하지 않음 ───────────────
    firstTapWriteError = nil
    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
      guard let self = self, let file = self.recorderFile else { return }
      do {
        try file.write(from: buffer)
      } catch {
        if self.firstTapWriteError == nil {
          self.firstTapWriteError = error
          print("SweepRecorder: 녹음 write 실패 - \(error.localizedDescription)")
        }
      }
    }

    // ── 6. 엔진 시작 + sweep 재생 ────────────────────────────────
    try audioEngine.start()

    let buffer = AVAudioPCMBuffer(
      pcmFormat: sweepFormat,
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

    // ── 7. 정리 ──────────────────────────────────────────────────
    inputNode.removeTap(onBus: 0)
    audioEngine.stop()
    try? AVAudioSession.sharedInstance().setActive(false)

    // ── 8. 녹음 검증 ─────────────────────────────────────────────
    if let err = firstTapWriteError {
      recorderFile = nil
      throw NSError(
        domain: "SweepRecorder", code: 4,
        userInfo: [NSLocalizedDescriptionKey:
          "녹음 데이터 기록 중 오류가 발생했습니다: \(err.localizedDescription)"]
      )
    }
    let recordedFrames = recorderFile?.length ?? 0
    print("SweepRecorder: 녹음 프레임 = \(recordedFrames) (\(Double(recordedFrames) / inputFormat.sampleRate)s)")
    if recordedFrames < AVAudioFramePosition(inputFormat.sampleRate * 0.5) {
      recorderFile = nil
      throw NSError(
        domain: "SweepRecorder", code: 5,
        userInfo: [NSLocalizedDescriptionKey:
          "녹음 길이가 너무 짧습니다 (\(recordedFrames) 프레임). 마이크 권한과 블루투스 스피커 연결을 확인한 뒤 다시 시도해주세요."]
      )
    }

    // ── 9. AVAudioFile 강제 close (WAV 헤더 flush) ──────────────────
    // AVAudioFile은 deinit 시점에만 RIFF/data chunk size를 헤더에 최종 기록한다.
    // instance var로 잡고 있으면 다음 record() 호출 전까지 close되지 않아
    // 디스크의 WAV 헤더가 frame=0 상태로 남고 backend가 빈 파일로 인식한다.
    recorderFile = nil

    let fileAttrs = try? FileManager.default.attributesOfItem(atPath: outputURL.path)
    let fileSize = (fileAttrs?[.size] as? Int) ?? 0
    print("SweepRecorder: 파일 크기 = \(fileSize) bytes")

    // ── 10. 디스크 헤더 검증 — 다시 열어서 frame 수 확인 ──────────────
    // 메모리상 length가 526592여도 디스크 WAV 헤더가 0-frame이면 backend는 빈 파일로 읽음.
    // 검증용으로 다시 열어 length를 확인하면 헤더 flush가 실제로 일어났는지 알 수 있다.
    let diskFrames: AVAudioFramePosition
    do {
      let verify = try AVAudioFile(forReading: outputURL)
      diskFrames = verify.length
    } catch {
      print("SweepRecorder: 검증용 파일 열기 실패 - \(error.localizedDescription)")
      throw NSError(
        domain: "SweepRecorder", code: 6,
        userInfo: [NSLocalizedDescriptionKey:
          "녹음 파일을 검증할 수 없습니다: \(error.localizedDescription)"]
      )
    }
    print("SweepRecorder: 디스크 frame = \(diskFrames) (\(Double(diskFrames) / inputFormat.sampleRate)s)")
    if diskFrames < AVAudioFramePosition(inputFormat.sampleRate * 0.5) {
      throw NSError(
        domain: "SweepRecorder", code: 7,
        userInfo: [NSLocalizedDescriptionKey:
          "녹음 파일 헤더가 정상 기록되지 않았습니다 (디스크 frame=\(diskFrames), 메모리 frame=\(recordedFrames)). 다시 시도해주세요."]
      )
    }

    print("SweepRecorder: 녹음 완료 → \(outputURL.path)")
    return outputURL
  }
}
