import AVFoundation
import Foundation
import MLXLLM
import MLXLMCommon
import Speech

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
        try await requestSpeechPermissions()

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
}
