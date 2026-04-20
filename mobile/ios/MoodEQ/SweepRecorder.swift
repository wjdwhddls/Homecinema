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

      // 3. AVAudioEngine 구성
      let engine = AVAudioEngine()
      let player = AVAudioPlayerNode()
      self.audioEngine = engine
      self.playerNode = player

      engine.attach(player)
      engine.connect(player, to: engine.mainMixerNode, format: format)

      // 4. 녹음 파일 경로
      let outputURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("ref_rir_\(Int(Date().timeIntervalSince1970)).wav")

      // 5. 입력(마이크) 탭 설치 → wav 파일로 저장
      let inputFormat = engine.inputNode.outputFormat(forBus: 0)
      var audioFile: AVAudioFile? = try AVAudioFile(
        forWriting: outputURL,
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

      // 6. 엔진 시작 + sweep 재생
      try engine.start()
      player.scheduleBuffer(sweepBuffer, completionCallbackType: .dataPlayedBack) { _ in
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) { [weak self] in
          // 재생 완료 후 0.3초 여유를 두고 종료 (잔향 캡처)
          engine.inputNode.removeTap(onBus: 0)
          engine.stop()
          audioFile = nil  // 파일 닫기
          self?.audioEngine = nil
          self?.playerNode = nil

          try? session.setActive(false)

          resolve([
            "uri": outputURL.absoluteString,
            "durationMs": Int((durationSec + 0.3) * 1000),
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

  // MARK: - Sine Sweep 버퍼 생성 (log sweep, 20Hz → 20kHz)

  private func makeSweepBuffer(
    format: AVAudioFormat,
    frameCount: AVAudioFrameCount,
    sampleRate: Double
  ) -> AVAudioPCMBuffer {
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
    buffer.frameLength = frameCount

    let f0: Double = 20.0      // 시작 주파수 (Hz)
    let f1: Double = 20000.0   // 끝 주파수 (Hz)
    let T: Double = Double(frameCount) / sampleRate
    let K = T / log(f1 / f0)

    let channelData = buffer.floatChannelData![0]
    for i in 0..<Int(frameCount) {
      let t = Double(i) / sampleRate
      // log sweep 위상 공식
      let phase = 2.0 * Double.pi * f0 * K * (exp(t / K) - 1.0)
      channelData[i] = Float(0.8 * sin(phase))
    }

    return buffer
  }
}
