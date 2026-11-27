import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Speech

@MainActor
protocol SpeechInterfacing {
    /// Resolves one user request, including clarification turns within this call.
    /// Keeps listening through silence until a complete query is resolved.
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
    You help an airport passenger choose a destination for a sign-search application.

    Supported destinations:
    - washroom
    - restaurant
    - gate with a number from 1 through 6

    Your job is to resolve one complete destination request through conversation.
    You do not search for signs, provide navigation, or look up flight information.

    Use the passenger's messages and the earlier conversation to understand their
    request. A short answer may supply information you asked for previously.
    Honor corrections and changes of destination. Do not keep information that
    the passenger has replaced or contradicted.

    Rules:
    - A washroom or restaurant request is complete when the intended destination
      is clear.
    - A gate request is complete only when the passenger has explicitly supplied
      a gate number from 1 through 6, either now or in an earlier message.
    - Understand gate numbers expressed as digits or words.
    - Never invent a gate number or infer it from a flight number, time, or
      unrelated number.
    - If the passenger wants a gate but has not supplied its number, ask which
      gate number they need.
    - If the request is vague or has multiple possible destinations, ask one
      short question to resolve it.
    - If the destination is unsupported, briefly explain the supported choices
      and ask what the passenger wants.
    - If the passenger requests flight information, explain that you can help
      find a gate sign but need their gate number.
    - Do not claim that you have found a sign or know its location.
    - Do not ask for confirmation once the request is complete.

    Return exactly one JSON object using one of these forms:

    For a clarification:
    {"kind":"clarify","message":"One short spoken question, with a brief explanation if needed."}

    For a completed washroom request:
    {"kind":"ready","label":"washroom"}

    For a completed restaurant request:
    {"kind":"ready","label":"restaurant"}

    For a completed gate request:
    {"kind":"ready","label":"gate","gateID":5}

    Output requirements:
    - Return JSON only, without Markdown, code fences, or surrounding text.
    - Use only the fields shown for the selected form.
    - The clarification message must be nonempty and suitable for speaking aloud.
    - The ready label must be "washroom", "restaurant", or "gate".
    - gateID must be an integer from 1 through 6 for a gate.
    - Omit gateID for washroom and restaurant requests.

    Examples:

    Passenger: I need my flight.
    Reply: {"kind":"clarify","message":"Which gate number are you looking for?"}
    Passenger: Five.
    Reply: {"kind":"ready","label":"gate","gateID":5}

    Passenger: Find a bathroom.
    Reply: {"kind":"ready","label":"washroom"}

    Passenger: I want somewhere to eat.
    Reply: {"kind":"ready","label":"restaurant"}

    Passenger: Find gate seven.
    Reply: {"kind":"clarify","message":"I can help with gates 1 through 6. Which supported destination would you like?"}

    Passenger: I need help.
    Reply: {"kind":"clarify","message":"Are you looking for a washroom, a restaurant, or a gate?"}

    Passenger: A restaurant or maybe a washroom.
    Reply: {"kind":"clarify","message":"Which would you like to find first, a restaurant or a washroom?"}
    Passenger: Actually, gate three.
    Reply: {"kind":"ready","label":"gate","gateID":3}
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

        var conversation: ChatSession?
        defer { Memory.clearCache() }
        return try await resolveQueryConversation(
            capture: { try await self.listenToUtterance() },
            respond: { utterance in
                let session: ChatSession
                if let conversation {
                    session = conversation
                } else {
                    let container = try await self.loadOrReuseModelContainer()
                    session = ChatSession(
                        container,
                        instructions: Self.llmInstructions,
                        generateParameters: GenerateParameters(maxTokens: 256, temperature: 0.0)
                    )
                    conversation = session
                }
                do {
                    let reply = try await session.respond(to: utterance)
                    await session.synchronize()
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

        let session = ListenSession(recognizer: recognizer, listeningHaptics: listeningHaptics)
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
}
