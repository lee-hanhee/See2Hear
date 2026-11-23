import AVFoundation
import Foundation
import Speech
import UIKit

func requestSpeechPermissions() async throws {
    let speechStatus = await withCheckedContinuation { continuation in
        SFSpeechRecognizer.requestAuthorization { status in
            continuation.resume(returning: status)
        }
    }
    let micGranted = await AVAudioApplication.requestRecordPermission()
    guard speechStatus == .authorized, micGranted else {
        throw SpeechError.notAuthorized
    }
}

final class ListenSession: @unchecked Sendable {
    private let recognizer: SFSpeechRecognizer
    private let audioEngine = AVAudioEngine()
    private let request = SFSpeechAudioBufferRecognitionRequest()

    private var recognitionTask: SFSpeechRecognitionTask?
    private var silenceTimerTask: Task<Void, Never>?
    private var maxDurationTask: Task<Void, Never>?
    private var continuation: CheckedContinuation<String, Error>?
    private var latestTranscript = ""
    private var isCompleted = false

    init(recognizer: SFSpeechRecognizer) {
        self.recognizer = recognizer
        self.request.shouldReportPartialResults = true
        self.request.requiresOnDeviceRecognition = true
    }

    func run() async throws -> String {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        await MainActor.run {
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [request] buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()

        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<String, Error>) in
            self.continuation = cont

            scheduleSilenceTimer(seconds: 5.0)
            maxDurationTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(10.0))
                self?.finish()
            }

            recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
                guard let self else { return }

                if let result {
                    let text = result.bestTranscription.formattedString
                    self.latestTranscript = text
                    if result.isFinal {
                        self.finish()
                        return
                    }
                    self.scheduleSilenceTimer(seconds: silenceDelay(for: text))
                }

                if let error {
                    self.complete(with: self.latestTranscript.isEmpty ? error : nil)
                }
            }
        }
    }

    func finish() {
        complete(with: nil)
    }

    func cancel() {
        complete(with: CancellationError())
    }

    private func scheduleSilenceTimer(seconds: TimeInterval) {
        silenceTimerTask?.cancel()
        silenceTimerTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self?.finish()
        }
    }

    private func complete(with error: Error?) {
        guard !isCompleted else { return }
        isCompleted = true

        silenceTimerTask?.cancel()
        maxDurationTask?.cancel()
        request.endAudio()
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionTask?.cancel()
        recognitionTask = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        Task { @MainActor in
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }

        if let error {
            continuation?.resume(throwing: error)
        } else {
            continuation?.resume(returning: latestTranscript)
        }
        continuation = nil
    }
}

final class SpeakSession: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Error>?
    private var isCompleted = false

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String, locale: Locale) async throws {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: locale.identifier)

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.continuation = cont
            synthesizer.speak(utterance)
        }
    }

    func cancel() {
        synthesizer.stopSpeaking(at: .immediate)
        complete(with: CancellationError())
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        complete(with: nil)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        complete(with: CancellationError())
    }

    private func complete(with error: Error?) {
        guard !isCompleted else { return }
        isCompleted = true

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        if let error {
            continuation?.resume(throwing: error)
        } else {
            continuation?.resume()
        }
        continuation = nil
    }
}
