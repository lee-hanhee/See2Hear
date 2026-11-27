import Foundation

struct CapturedFrame {
    let frame: CameraFrame
    let timestamp: TimeInterval
}

struct TrackedSign {
    let box: SignBoundingBox
    let timestamp: TimeInterval

    func overlap(with next: SignBoundingBox, at time: TimeInterval) -> Double {
        guard time > timestamp, time - timestamp <= 0.5 else { return 0 }
        return overlap(with: next)
    }

    func overlap(with next: SignBoundingBox) -> Double {
        guard box.frameWidth == next.frameWidth, box.frameHeight == next.frameHeight else { return 0 }
        let intersection = box.rect.intersection(next.rect)
        let area = intersection.width * intersection.height
        let union = box.rect.width * box.rect.height + next.rect.width * next.rect.height - area
        return area / union
    }

    func matching(_ candidates: [SignBoundingBox], at time: TimeInterval,
                  afterPlayback: Bool = false) -> TrackedSign? {
        var best: SignBoundingBox?
        var bestOverlap = 0.0
        for candidate in candidates {
            let score = afterPlayback ? overlap(with: candidate) : overlap(with: candidate, at: time)
            if score >= 0.40 && score > bestOverlap {
                best = candidate
                bestOverlap = score
            }
        }
        return best.map { TrackedSign(box: $0, timestamp: time) }
    }
}

struct RejectedSigns {
    var tracks: [TrackedSign] = []

    mutating func eligible(_ candidates: [SignBoundingBox], at time: TimeInterval) -> [SignBoundingBox] {
        tracks = tracks.compactMap { $0.matching(candidates, at: time, afterPlayback: true) }
        return candidates.filter { candidate in
            !tracks.contains { $0.overlap(with: candidate) >= 0.40 }
        }
    }
}

private extension SignBoundingBox {
    var rect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
    var centered: Bool {
        abs(rect.midX / Double(frameWidth) - 0.5) <= 0.12
            && abs(rect.midY / Double(frameHeight) - 0.5) <= 0.12
    }
}

struct CenteringStreak {
    private var last: TrackedSign?
    private var count = 0

    mutating func observe(_ box: SignBoundingBox?, at time: TimeInterval) -> TrackedSign? {
        guard let box, box.centered else {
            last = nil
            count = 0
            return nil
        }
        let overlap = last?.overlap(with: box, at: time) ?? 0
        count = overlap >= 0.40 ? count + 1 : 1
        last = TrackedSign(box: box, timestamp: time)
        return count >= 3 ? last : nil
    }
}

/// A slot with no matching box abstains on every field. All votes still require three of five.
struct SignBatch {
    static let size = 5
    struct Observation {
        let reading: SignReading?
        let distance: SignDistanceBand
    }
    enum Outcome {
        case speak(SpeechFeedback)
        case lostCandidate
        case unreadableGate
        case wrongGate(Int)
    }
    var observations: [Observation] = []

    private func majority<T: Equatable>(_ values: [T]) -> T? {
        values.first { value in values.filter { $0 == value }.count >= 3 }
    }

    func outcome(for query: SignQuery) -> Outcome {
        guard observations.count >= 3 else { return .lostCandidate }
        let voters: [Observation]
        var arrow: ArrowDirection?
        switch query.label {
        case .gate:
            guard case .gate(let number) = majority(observations.compactMap(\.reading)) else {
                return .unreadableGate
            }
            guard number == query.requestedGateID else {
                return .wrongGate(number)
            }
            voters = observations.filter { $0.reading == .gate(number) }
        case .restaurant, .washroom:
            voters = observations
            if case .arrow(let direction) = majority(observations.compactMap(\.reading)) { arrow = direction }
        }
        let event = SearchFeedbackEvent.recognized(arrowDirection: arrow, distanceToSign: majority(voters.map(\.distance)))
        return .speak(SpeechFeedback(query: query, event: event))
    }
}
