import Foundation

func formatFeedback(_ feedback: SpeechFeedback) -> String? {
    let target: String
    switch feedback.query.label {
    case .washroom:
        target = "Washroom"
    case .restaurant:
        target = "Restaurant"
    case .gate:
        target = feedback.query.requestedGateID.map { "Gate \($0)" } ?? "Gate"
    }

    switch feedback.event {
    case .stillSearching:
        return "\(target) sign is not visible."
    case .recognized(let arrow, let distance):
        return formatRecognizedSign(target: target, arrow: arrow, distance: distance)
    case .searchStarted, .lostSightOfSign, .wrongGate, .retryingRecognition, .recognitionFailed:
        return nil
    }
}

private func formatRecognizedSign(target: String, arrow: ArrowDirection?, distance: SignDistanceBand?) -> String {
    var parts: [String] = []
    switch distance {
    case .near:
        parts.append("\(target) sign is less than 5 meters away.")
    case .medium:
        parts.append("\(target) sign is 5 to 15 meters away.")
    case .far:
        parts.append("\(target) sign is more than 15 meters away.")
    case nil:
        parts.append("\(target) sign is visible.")
    }

    switch arrow {
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
