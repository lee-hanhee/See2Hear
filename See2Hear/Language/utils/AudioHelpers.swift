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
    private let lock = NSLock()

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

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [request] buffer, _ in
            request.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }

        await MainActor.run {
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        }

        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<String, Error>) in
            lock.lock()
            if isCompleted {
                lock.unlock()
                cont.resume(throwing: CancellationError())
                return
            }
            self.continuation = cont
            lock.unlock()

            scheduleSilenceTimer(seconds: 5.0)
            maxDurationTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(10.0))
                guard !Task.isCancelled else { return }
                self?.finish()
            }

            recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
                guard let self else { return }

                if let result {
                    let text = result.bestTranscription.formattedString
                    self.lock.lock()
                    self.latestTranscript = text
                    self.lock.unlock()

                    if result.isFinal {
                        self.finish()
                        return
                    }
                    self.scheduleSilenceTimer(seconds: silenceDelay(for: text))
                }

                if let error {
                    self.lock.lock()
                    let hasTranscript = !self.latestTranscript.isEmpty
                    self.lock.unlock()
                    self.complete(with: hasTranscript ? nil : error)
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
        lock.lock()
        silenceTimerTask?.cancel()
        silenceTimerTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self?.finish()
        }
        lock.unlock()
    }

    private func complete(with error: Error?) {
        lock.lock()
        guard !isCompleted else {
            lock.unlock()
            return
        }
        isCompleted = true
        let transcript = latestTranscript
        let cont = continuation
        continuation = nil
        silenceTimerTask?.cancel()
        maxDurationTask?.cancel()
        lock.unlock()

        request.endAudio()
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionTask?.cancel()
        recognitionTask = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        if error == nil {
            Task { @MainActor in
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
        }

        if let error {
            cont?.resume(throwing: error)
        } else {
            cont?.resume(returning: transcript)
        }
    }
}

final class SpeakSession: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()
    private let lock = NSLock()
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
            lock.lock()
            if isCompleted {
                lock.unlock()
                try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
                cont.resume(throwing: CancellationError())
                return
            }
            self.continuation = cont
            lock.unlock()
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
        lock.lock()
        guard !isCompleted else {
            lock.unlock()
            return
        }
        isCompleted = true
        let cont = continuation
        continuation = nil
        lock.unlock()

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        if let error {
            cont?.resume(throwing: error)
        } else {
            cont?.resume()
        }
    }
}
