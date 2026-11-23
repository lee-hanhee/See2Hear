import Foundation

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
