import Foundation

func formatFeedback(_ feedback: SpeechFeedback) -> String {
    let target: String
    let destination: String
    switch feedback.query.label {
    case .washroom:
        target = "Washroom"
        destination = "the washroom"
    case .restaurant:
        target = "Restaurant"
        destination = "a restaurant"
    case .gate:
        target = feedback.query.requestedGateID.map { "Gate \($0)" } ?? "Gate"
        destination = target
    }

    switch feedback.event {
    case .searchStarted:
        return "Confirmed. Let’s look for \(destination)."
    case .stillSearching:
        return "Still looking for \(destination). Continue to move your phone slowly."
    case .lostSightOfSign:
        return "I lost sight of the sign. Pan back slowly."
    case .wrongGate(let observedGateID):
        return "This sign says Gate \(observedGateID). Keep panning slowly to look for \(target)."
    case .retryingRecognition(.first):
        return "I couldn’t read the sign. Hold your phone still while I try again."
    case .retryingRecognition(.second):
        return "I still couldn’t read the sign. Hold your phone still while I try once more."
    case .recognitionFailed:
        return "I couldn’t read the sign. Please tell me what you’d like me to look for."
    case .recognized(let arrow, let distance):
        return formatRecognizedSign(target: target, arrow: arrow, distance: distance)
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
