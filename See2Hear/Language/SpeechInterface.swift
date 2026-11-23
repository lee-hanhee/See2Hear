import AVFoundation
import Foundation
import MLXLLM
import MLXLMCommon
import Speech
import UIKit

protocol SpeechInterfacing {
    /// Listens for one spoken request. Return nil for an unsupported request.
    /// Throw if speech capture or recognition fails. Release the microphone
    /// when the call ends, including on cancellation.
    func listenForQuery() async throws -> SignQuery?

    /// Speaks the supplied feedback. Return once playback finishes. 
    /// Throw on failure and stop playback on cancellation.
    func speak(_ feedback: SpeechFeedback) async throws
}

enum SpeechError: Error {
    case recognizerUnavailable
    case notAuthorized
}

enum QueryParseOutcome {
    case matched(SignQuery)
    case unsupported
    case unmatched
}

final class SpeechInterface: SpeechInterfacing {
    static let defaultModelID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    private static let llmInstructions = """
    Classify the airport passenger's spoken request into exactly one label:
    - washroom
    - restaurant
    - gate
    - gate 1
    - gate 2
    - gate 3
    - gate 4
    - gate 5
    - gate 6
    - none
    Reply with only the label.
    """

    private let locale: Locale
    private let modelID: String
    private var modelContainer: ModelContainer?
    private var activeListenSession: ListenSession?
    private var activeSpeakSession: SpeakSession?

    init(
        locale: Locale = Locale(identifier: "en-US"),
        modelID: String = SpeechInterface.defaultModelID
    ) {
        self.locale = locale
        self.modelID = modelID
    }

    func finishListening() {
        activeListenSession?.finish()
    }

    func listenForQuery() async throws -> SignQuery? {
        try await requestPermissions()

        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
            throw SpeechError.recognizerUnavailable
        }

        let session = ListenSession(recognizer: recognizer)
        activeListenSession = session
        defer { activeListenSession = nil }

        let transcript = try await withTaskCancellationHandler {
            try await session.run()
        } onCancel: {
            session.cancel()
        }

        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        switch parseQuery(trimmed) {
        case .matched(let query):
            return query
        case .unsupported:
            return nil
        case .unmatched:
            return try await classifyWithLLM(trimmed)
        }
    }

    func speak(_ feedback: SpeechFeedback) async throws {
        let text = formatFeedback(feedback)
        let session = SpeakSession()
        activeSpeakSession = session
        defer { activeSpeakSession = nil }

        try await withTaskCancellationHandler {
            try await session.speak(text, locale: locale)
        } onCancel: {
            session.cancel()
        }
    }

    private func classifyWithLLM(_ transcript: String) async throws -> SignQuery? {
        if modelContainer == nil {
            modelContainer = try await loadModelContainer(id: modelID)
        }
        let session = ChatSession(
            modelContainer!,
            instructions: Self.llmInstructions,
            generateParameters: GenerateParameters(maxTokens: 8, temperature: 0.0)
        )
        let reply = try await session.respond(to: transcript)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        guard reply != "none", case .matched(let query) = parseQuery(reply) else {
            return nil
        }
        return query
    }

    private func requestPermissions() async throws {
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
}

// MARK: - Query Parsing & Adaptive Silence Helpers

private let washroomKeywords: Set<String> = [
    "washroom", "washrooms", "bathroom", "bathrooms",
    "restroom", "restrooms", "toilet", "toilets",
    "lavatory", "wc", "mens", "womens", "ladies", "gents"
]

private let restaurantKeywords: Set<String> = [
    "restaurant", "restaurants", "food", "eat",
    "dining", "cafe", "coffee", "lunch", "dinner",
    "breakfast", "snack", "snacks"
]

private let gateKeywords: Set<String> = ["gate", "gates"]

private let numberWords: [String: Int] = [
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
]

private let postGateHomophones: [String: Int] = [
    "to": 2, "too": 2, "for": 4
]

func parseQuery(_ transcript: String) -> QueryParseOutcome {
    let tokens = transcript
        .lowercased()
        .replacingOccurrences(of: "'", with: "")
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }

    guard !tokens.isEmpty else { return .unmatched }

    if let gateIndex = tokens.firstIndex(where: { gateKeywords.contains($0) }) {
        if let gateNumber = extractGateNumber(from: tokens, gateIndex: gateIndex) {
            guard let query = SignQuery(queryID: UUID(), label: .gate, requestedGateID: gateNumber) else {
                return .unsupported
            }
            return .matched(query)
        }
        return .matched(SignQuery(queryID: UUID(), label: .gate)!)
    }

    if tokens.contains(where: { washroomKeywords.contains($0) }) {
        return .matched(SignQuery(queryID: UUID(), label: .washroom)!)
    }

    if tokens.contains(where: { restaurantKeywords.contains($0) }) {
        return .matched(SignQuery(queryID: UUID(), label: .restaurant)!)
    }

    return .unmatched
}

private func extractGateNumber(from tokens: [String], gateIndex: Int) -> Int? {
    for (offset, token) in tokens[(gateIndex + 1)...].enumerated() {
        if let value = Int(token) ?? numberWords[token] {
            return value
        }
        if offset == 0, let value = postGateHomophones[token] {
            return value
        }
    }
    for token in tokens[..<gateIndex] {
        if let value = Int(token) ?? numberWords[token] {
            return value
        }
    }
    return nil
}

func silenceDelay(for transcript: String) -> TimeInterval {
    switch parseQuery(transcript) {
    case .matched(let query):
        return (query.label == .gate && query.requestedGateID == nil) ? 2.0 : 0.8
    case .unsupported:
        return 0.8
    case .unmatched:
        return 2.0
    }
}

// MARK: - Feedback Formatting Helper

func formatFeedback(_ feedback: SpeechFeedback) -> String {
    let target: String
    switch feedback.query.label {
    case .washroom:
        target = "Washroom"
    case .restaurant:
        target = "Restaurant"
    case .gate:
        target = feedback.query.requestedGateID.map { "Gate \($0)" } ?? "Gate"
    }

    guard feedback.outcome == .targetVisible else {
        return "\(target) sign is not visible."
    }

    var parts: [String] = []
    switch feedback.distanceToSign {
    case .near:
        parts.append("\(target) sign is less than 5 meters away.")
    case .medium:
        parts.append("\(target) sign is 5 to 15 meters away.")
    case .far:
        parts.append("\(target) sign is more than 15 meters away.")
    case nil:
        parts.append("\(target) sign is visible.")
    }

    switch feedback.arrowDirection {
    case .left:
        parts.append("Follow the arrow to the left.")
    case .right:
        parts.append("Follow the arrow to the right.")
    case .forward:
        parts.append("Head straight ahead.")
    case nil:
        break
    }

    return parts.joined(separator: " ")
}

// MARK: - Audio Capture & Synthesis Sessions

private final class ListenSession: @unchecked Sendable {
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

private final class SpeakSession: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {
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
