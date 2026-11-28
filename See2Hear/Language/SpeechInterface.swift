import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Observation
import Speech

enum SpeechActivity {
    case idle, listening, processing, speaking

    var status: String {
        switch self {
        case .idle: "Preparing to listen"
        case .listening: "Listening for your request"
        case .processing: "Understanding your request"
        case .speaking: "Speaking to you"
        }
    }
}

@MainActor
protocol SpeechInterfacing {
    var isLoadingModel: Bool { get }
    var activity: SpeechActivity { get }
    /// The live transcript of the current or most recent utterance, for on-screen display.
    var heardText: String { get }

    /// Resolves one user request, including clarification turns within this call.
    /// Waits through silence and clarification turns without a conversation timeout.
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

@Observable
final class SpeechInterface: SpeechInterfacing {
    static let defaultModelID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    private static let mlxCacheLimitBytes = 20 * 1024 * 1024

    private static let llmInstructions = """
    You help airport passengers find a washroom, a restaurant, or a gate numbered 1 through 6.
    Reply with one JSON object only.
    For a washroom or bathroom request, reply {"kind":"ready","label":"washroom"}.
    For a restaurant or food request, reply {"kind":"ready","label":"restaurant"}.
    For a gate request with a number, reply {"kind":"ready","label":"gate","gateID":1}, using the number the passenger gave.
    If a gate number is missing, reply {"kind":"clarify","message":"Which gate number?"}.
    If the request is unclear or unsupported, reply {"kind":"clarify","message":"Would you like a washroom, a restaurant, or a gate from 1 to 6?"}.
    Remember the conversation. Use a short answer to fill in missing information. Honor corrections.
    A washroom or restaurant request is already complete. Answer the passenger's actual request.
    """

    private let locale: Locale
    private let listeningHaptics: Bool
    private let modelID: String
    private var modelContainer: ModelContainer?
    private var activeListenSession: ListenSession?
    private var activeSpeakSession: SpeakSession?
    private(set) var isLoadingModel = false
    private(set) var heardText = ""
    private(set) var activity: SpeechActivity = .idle

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
        defer { activity = .idle }
        try await requestSpeechPermissions()
        try Task.checkCancellation()
        return try await resolveConversation(instructions: Self.llmInstructions)
    }

    // Keep conversation history local and apply the chat template to all turns together.
    // Reuse the model weights while rebuilding each turn's context from explicit messages.
    private nonisolated func resolveConversation(instructions: String) async throws -> SignQuery {
        var history: [Chat.Message] = []
        defer { Memory.clearCache() }
        return try await resolveQueryConversation(
            capture: { try await self.listenToUtterance() },
            respond: { utterance in
                let container = try await self.loadOrReuseModelContainer(announceLoading: true)
                let session = ChatSession(
                    container,
                    instructions: instructions,
                    history: history,
                    generateParameters: GenerateParameters(maxTokens: 256, temperature: 0.0)
                )
                await MainActor.run { self.activity = .processing }
                do {
                    let reply = try await session.respond(to: utterance)
                    await session.synchronize()
                    history.append(.user(utterance))
                    history.append(.assistant(reply))
                    return reply
                } catch {
                    await session.synchronize()
                    throw error
                }
            },
            speak: { try await self.speakText($0) }
        )
    }

    private func listenToUtterance() async throws -> String {
        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
            throw SpeechError.recognizerUnavailable
        }

        heardText = ""
        activity = .listening
        let session = ListenSession(recognizer: recognizer, listeningHaptics: listeningHaptics) { text in
            Task { @MainActor [weak self] in self?.heardText = text }
        }
        activeListenSession = session
        defer { activeListenSession = nil }

        return try await withTaskCancellationHandler {
            try await session.run()
        } onCancel: {
            session.cancel()
        }
    }

    func speak(_ feedback: SpeechFeedback) async throws {
        try await speakText(formatFeedback(feedback))
    }

    private func speakText(_ text: String) async throws {
        activity = .speaking
        defer { activity = .idle }
        let session = SpeakSession()
        activeSpeakSession = session
        defer { activeSpeakSession = nil }

        try await withTaskCancellationHandler {
            try await session.speak(text, locale: locale)
        } onCancel: {
            Task { @MainActor in session.cancel() }
        }
    }

    /// Preloads and caches the on-device LLM (e.g. while on Wi-Fi before offline use).
    func preloadModel() async throws {
        _ = try await loadOrReuseModelContainer()
    }

    private func loadOrReuseModelContainer(announceLoading: Bool = false) async throws -> ModelContainer {
        try Task.checkCancellation()
        if let existing = modelContainer {
            return existing
        }
        isLoadingModel = true
        defer { isLoadingModel = false }
        if announceLoading {
            try await speakText("Loading the language model. Please wait.")
            try Task.checkCancellation()
        }
        Memory.cacheLimit = Self.mlxCacheLimitBytes
        let loaded = try await loadModelContainer(id: modelID)
        try Task.checkCancellation()
        modelContainer = loaded
        return loaded
    }
}
