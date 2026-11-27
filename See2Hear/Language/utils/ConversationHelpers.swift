import Foundation

enum ConversationError: LocalizedError {
    case invalidResponse

    var errorDescription: String? { "The language model returned an invalid response." }
}

enum ConversationReply {
    case clarify(String)
    case ready(SignQuery)
}

func decodeConversationReply(_ text: String) throws -> ConversationReply {
    struct Reply: Decodable {
        let kind: String
        let message: String?
        let label: String?
        let gateID: Int?
    }

    guard let reply = try? JSONDecoder().decode(Reply.self, from: Data(text.utf8)) else {
        throw ConversationError.invalidResponse
    }
    switch reply.kind {
    case "clarify":
        guard let message = reply.message?.trimmingCharacters(in: .whitespacesAndNewlines), !message.isEmpty else {
            throw ConversationError.invalidResponse
        }
        return .clarify(message)
    case "ready":
        guard let label = reply.label.flatMap(SignLabel.init(rawValue:)),
              let query = SignQuery(queryID: UUID(), label: label, requestedGateID: reply.gateID) else {
            throw ConversationError.invalidResponse
        }
        return .ready(query)
    default:
        throw ConversationError.invalidResponse
    }
}

@MainActor
func resolveQueryConversation(
    capture: () async throws -> String,
    respond: (String) async throws -> String,
    speak: (String) async throws -> Void
) async throws -> SignQuery {
    while true {
        try Task.checkCancellation()
        let utterance = try await capture().trimmingCharacters(in: .whitespacesAndNewlines)
        try Task.checkCancellation()
        if utterance.isEmpty { continue }
        let response = try await respond(utterance)
        try Task.checkCancellation()
        switch try decodeConversationReply(response) {
        case .clarify(let message):
            try await speak(message)
        case .ready(let query):
            return query
        }
    }
}
