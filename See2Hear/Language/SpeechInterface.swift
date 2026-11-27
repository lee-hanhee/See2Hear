import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Speech

@MainActor
protocol SpeechInterfacing {
    /// Resolves one user request, including clarification turns within this call.
    /// Returns nil if the interaction doesn't yield a valid query, even after potentially
    /// multiple conversational rounds.
    /// Throw on failure or cancellation and release the microphone on exit.
    func listenForQuery() async throws -> SignQuery?

    /// Announces one event for the supplied query. Return once playback finishes.
    /// Throw on failure and stop playback on cancellation.
    func speak(_ feedback: SpeechFeedback) async throws
}

enum SpeechError: LocalizedError {
    case recognizerUnavailable
    case notAuthorized

    var errorDescription: String? {
        switch self {
        case .recognizerUnavailable: "Speech recognition is unavailable."
        case .notAuthorized: "Allow microphone and speech recognition access in Settings to use See2Hear."
        }
    }
}

final class SpeechInterface: SpeechInterfacing {
    static let defaultModelID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    private static let mlxCacheLimitBytes = 20 * 1024 * 1024

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
    private let listeningHaptics: Bool
    private let modelID: String
    private var modelContainer: ModelContainer?
    private var activeListenSession: ListenSession?
    private var activeSpeakSession: SpeakSession?

    init(
        locale: Locale = Locale(identifier: "en-US"),
        modelID: String = SpeechInterface.defaultModelID,
        listeningHaptics: Bool = true
    ) {
        self.locale = locale
        self.modelID = modelID
        self.listeningHaptics = listeningHaptics
    }

    func finishListening() {
        activeListenSession?.finish()
    }

    func listenForQuery() async throws -> SignQuery? {
        try await requestSpeechPermissions()
        try Task.checkCancellation()

        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
            throw SpeechError.recognizerUnavailable
        }

        let session = ListenSession(recognizer: recognizer, listeningHaptics: listeningHaptics)
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
        case .unsupported, .missingGateNumber:
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
            Task { @MainActor in session.cancel() }
        }
    }

    /// Preloads and caches the on-device fallback LLM (e.g. while on Wi-Fi before offline use).
    func preloadModel() async throws {
        _ = try await loadOrReuseModelContainer()
    }

    private func loadOrReuseModelContainer() async throws -> ModelContainer {
        try Task.checkCancellation()
        if let existing = modelContainer {
            return existing
        }
        Memory.cacheLimit = Self.mlxCacheLimitBytes
        let loaded = try await loadModelContainer(id: modelID)
        try Task.checkCancellation()
        modelContainer = loaded
        return loaded
    }

    private func classifyWithLLM(_ transcript: String) async throws -> SignQuery? {
        let container = try await loadOrReuseModelContainer()
        defer { Memory.clearCache() }

        try Task.checkCancellation()
        let session = ChatSession(
            container,
            instructions: Self.llmInstructions,
            generateParameters: GenerateParameters(maxTokens: 8, temperature: 0.0)
        )
        let reply = try await session.respond(to: transcript)
        try Task.checkCancellation()
        return parseLLMReply(reply)
    }
}
