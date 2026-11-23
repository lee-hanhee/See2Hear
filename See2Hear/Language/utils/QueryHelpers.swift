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

    let gateIndex = tokens.firstIndex(where: { gateKeywords.contains($0) })
    let hasWashroom = tokens.contains(where: { washroomKeywords.contains($0) })
    let hasRestaurant = tokens.contains(where: { restaurantKeywords.contains($0) })

    let matchedCategories = [gateIndex != nil, hasWashroom, hasRestaurant].filter { $0 }.count
    guard matchedCategories == 1 else {
        return .unmatched
    }

    if let gateIndex {
        if let gateNumber = extractGateNumber(from: tokens, gateIndex: gateIndex) {
            guard let query = SignQuery(queryID: UUID(), label: .gate, requestedGateID: gateNumber) else {
                return .unsupported
            }
            return .matched(query)
        }
        return .matched(SignQuery(queryID: UUID(), label: .gate)!)
    }

    if hasWashroom {
        return .matched(SignQuery(queryID: UUID(), label: .washroom)!)
    }

    return .matched(SignQuery(queryID: UUID(), label: .restaurant)!)
}

/// Extracts a gate number only when it immediately follows "gate" (or "gate number <N>").
private func extractGateNumber(from tokens: [String], gateIndex: Int) -> Int? {
    let nextIndex = gateIndex + 1
    guard nextIndex < tokens.count else { return nil }

    let firstToken = tokens[nextIndex]
    if let value = Int(firstToken) ?? numberWords[firstToken] ?? postGateHomophones[firstToken] {
        return value
    }

    if (firstToken == "number" || firstToken == "no"), nextIndex + 1 < tokens.count {
        let secondToken = tokens[nextIndex + 1]
        return Int(secondToken) ?? numberWords[secondToken]
    }

    return nil
}

/// Strictly parses the constrained 10-label LLM output without heuristic keyword scanning.
func parseLLMReply(_ reply: String) -> SignQuery? {
    let normalized = reply
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
        .trimmingCharacters(in: .punctuationCharacters)

    switch normalized {
    case "washroom":
        return SignQuery(queryID: UUID(), label: .washroom)
    case "restaurant":
        return SignQuery(queryID: UUID(), label: .restaurant)
    case "gate":
        return SignQuery(queryID: UUID(), label: .gate)
    default:
        let parts = normalized.split(separator: " ")
        if parts.count == 2, parts[0] == "gate", let id = Int(parts[1]) {
            return SignQuery(queryID: UUID(), label: .gate, requestedGateID: id)
        }
        return nil
    }
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
