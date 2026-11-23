import Foundation

enum QueryParseOutcome {
    case matched(SignQuery)
    case unsupported
    case unmatched
}

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

/// Adaptive silence delay in seconds after the most recent word:
/// - 0.8s once a complete query is matched (washroom, restaurant, or gate with number)
/// - 2.0s when waiting for a gate number after "gate" or for full sentence before LLM fallback
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
