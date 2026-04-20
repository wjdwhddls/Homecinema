import Foundation
import AVFoundation
import React

@objc(SweepRecorder)
class SweepRecorder: NSObject {

  private var audioEngine: AVAudioEngine?
  private var playerNode: AVAudioPlayerNode?

  override static func requiresMainQueueSetup() -> Bool { false }

  // MARK: - 마이크 권한 요청

  @objc(requestPermission:rejecter:)
  func requestPermission(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    AVAudioSession.sharedInstance().requestRecordPermission { granted in
      resolve(granted)
    }
  }

  // MARK: - Sweep 재생 + 동시 녹음

  @objc(recordSweep:sampleRate:resolver:rejecter:)
  func recordSweep(
    _ durationSec: Double,
    sampleRate: Double,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    do {
      // 1. 오디오 세션 설정 (재생 + 녹음 동시)
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
      try session.setActive(true)

      let sr = Double(sampleRate)
      let frameCount = AVAudioFrameCount(sr * durationSec)
      let format = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 1)!

      // 2. Sine sweep 버퍼 생성 (20Hz → 20kHz, log sweep)
      let sweepBuffer = makeSweepBuffer(format: format, frameCount: frameCount, sampleRate: sr)

      // 3. sweep.wav 저장 (서버 deconvolution용 원본 신호)
      let sweepURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("sweep_\(Int(Date().timeIntervalSince1970)).wav")
      try savePCMBufferToWav(buffer: sweepBuffer, url: sweepURL, sampleRate: sr)

      // 4. AVAudioEngine 구성
      let engine = AVAudioEngine()
      let player = AVAudioPlayerNode()
      self.audioEngine = engine
      self.playerNode = player

      engine.attach(player)
      engine.connect(player, to: engine.mainMixerNode, format: format)

      // 5. recorded.wav 경로
      let recordedURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("recorded_\(Int(Date().timeIntervalSince1970)).wav")

      // 6. 마이크 탭 설치 → recorded.wav 저장
      let inputFormat = engine.inputNode.outputFormat(forBus: 0)
      var audioFile: AVAudioFile? = try AVAudioFile(
        forWriting: recordedURL,
        settings: [
          AVFormatIDKey: kAudioFormatLinearPCM,
          AVSampleRateKey: sampleRate,
          AVNumberOfChannelsKey: 1,
          AVLinearPCMBitDepthKey: 24,
          AVLinearPCMIsFloatKey: false,
        ]
      )

      engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
        try? audioFile?.write(from: buffer)
      }

      // 7. 엔진 시작 + sweep 재생
      try engine.start()
      player.scheduleBuffer(sweepBuffer, completionCallbackType: .dataPlayedBack) { _ in
        // 재생 완료 후 0.5초 여유 (잔향 캡처)
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) { [weak self] in
          engine.inputNode.removeTap(onBus: 0)
          engine.stop()
          audioFile = nil
          self?.audioEngine = nil
          self?.playerNode = nil

          try? session.setActive(false)

          // recorded.wav + sweep.wav 둘 다 반환
          resolve([
            "recordedUri": recordedURL.absoluteString,
            "sweepUri": sweepURL.absoluteString,
            "durationMs": Int((durationSec + 0.5) * 1000),
          ])
        }
      }
      player.play()

    } catch {
      reject("SWEEP_ERROR", "Sweep 녹음 중 오류 발생: \(error.localizedDescription)", error)
    }
  }

  // MARK: - 녹음 파일 삭제

  @objc(deleteRecording:resolver:rejecter:)
  func deleteRecording(
    _ uri: String,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    guard let url = URL(string: uri) else {
      reject("INVALID_URI", "유효하지 않은 파일 경로입니다", nil)
      return
    }
    do {
      try FileManager.default.removeItem(at: url)
      resolve(nil)
    } catch {
      reject("DELETE_ERROR", "파일 삭제 실패: \(error.localizedDescription)", error)
    }
  }

  // MARK: - PCM 버퍼 → wav 파일 저장

  private func savePCMBufferToWav(buffer: AVAudioPCMBuffer, url: URL, sampleRate: Double) throws {
    let audioFile = try AVAudioFile(
      forWriting: url,
      settings: [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: sampleRate,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 24,
        AVLinearPCMIsFloatKey: false,
      ]
    )
    try audioFile.write(from: buffer)
  }

  // MARK: - Sine Sweep 버퍼 생성 (log sweep, 20Hz → 20kHz)

  private func makeSweepBuffer(
    format: AVAudioFormat,
    frameCount: AVAudioFrameCount,
    sampleRate: Double
  ) -> AVAudioPCMBuffer {
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
    buffer.frameLength = frameCount

    let f0: Double = 20.0
    let f1: Double = 20000.0
    let T: Double = Double(frameCount) / sampleRate
    let K = T / log(f1 / f0)

    let channelData = buffer.floatChannelData![0]
    for i in 0..<Int(frameCount) {
      let t = Double(i) / sampleRate
      let phase = 2.0 * Double.pi * f0 * K * (exp(t / K) - 1.0)
      channelData[i] = Float(0.8 * sin(phase))
    }

    return buffer
  }
}
