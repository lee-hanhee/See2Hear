import CoreGraphics
import Foundation
import Testing
@testable import See2Hear

@MainActor
struct OrchestrationTests {
    @Test("Guidance resets on a miss, reads five fresh frames, and waits for high haptics to finish before speaking")
    func successfulRound() async throws {
        let rig = try RoundFixture(label: .washroom)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        let feedback = try #require(rig.language.spoken.first)
        #expect(feedback.query.queryID == rig.query.queryID)
        #expect(feedback.arrowDirection == .left)
        #expect(feedback.distanceToSign == nil)
        #expect(rig.language.spoken.count == 1)
        #expect(await rig.vision.readBoxes == [32, 33, 34, 35, 36])
        #expect(rig.haptics.changes.map(\.strength).starts(with: [.low, .off, .low, .high, .off]))
        let high = try #require(rig.haptics.changes.first { $0.strength == .high })
        let end = try #require(rig.haptics.changes.first { $0.strength == .off && $0.time > high.time })
        #expect(high.time.duration(to: end.time) >= rig.pulse)
        #expect(try #require(rig.language.spokeAt) >= end.time)
    }

    @Test("Pulse expires independently of slow inference, and cancellation discards the late result")
    func cancelDuringRead() async throws {
        let rig = try RoundFixture(label: .washroom, delayReaders: true)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.haptics.changes.contains { $0.strength == .high } && rig.haptics.changes.last?.strength == .off }
        #expect(rig.coordinator.phase == .reading)
        #expect(rig.language.spoken.isEmpty)
        rig.coordinator.cancelRound()
        await rig.vision.releaseReaders()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.spoken.isEmpty)
        #expect(rig.haptics.changes.last?.strength == .off)
    }

    @Test("An unsupported query aborts without guiding or speaking")
    func invalidQuery() async throws {
        let rig = try RoundFixture(label: .restaurant)
        rig.language.invalidInput = true
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        #expect(rig.coordinator.phase == .listening)
        #expect(rig.haptics.changes.allSatisfy { $0.strength == .off })
        #expect(rig.language.spoken.isEmpty)
    }

    @Test("The cancel button interrupts playback before the next listen")
    func cancelDuringPlayback() async throws {
        let rig = try RoundFixture(label: .restaurant)
        rig.language.holdPlayback = true
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.coordinator.phase == .speaking && rig.language.spoken.count == 1 }
        rig.coordinator.cancelRound()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.playbackWasCancelled)
        #expect(rig.haptics.changes.last?.strength == .off)
    }

    @Test("A wrong gate silently aborts and the next round listens again")
    func wrongGate() async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 2)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.spoken.isEmpty)
        #expect(rig.haptics.changes.contains { $0.strength == .high })
    }

    @Test("An interruption preempts a read and resumption starts with a fresh query")
    func interruption() async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 1, delayReaders: true)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.haptics.changes.last?.strength == .high }
        rig.coordinator.stop(reason: "Interrupted")
        #expect(rig.haptics.changes.last?.strength == .off)
        #expect(rig.coordinator.phase == .suspended("Interrupted"))
        rig.coordinator.start()
        await rig.vision.releaseReaders()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.spoken.isEmpty)
    }

    @Test("A numbered gate only uses distance votes from matching digits")
    func gateVotes() throws {
        let query = try #require(SignQuery(queryID: UUID(), label: .gate, requestedGateID: 4))
        var batch = SignBatch(observations: [
            .init(reading: .gate(4), distance: .medium), .init(reading: .gate(4), distance: .near),
            .init(reading: .gate(4), distance: .medium), .init(reading: .gate(1), distance: .medium),
        ])
        guard case .speak(let feedback) = batch.outcome(for: query) else {
            Issue.record("Expected a confirmed Gate 4")
            return
        }
        #expect(feedback.distanceToSign == nil)
        batch.observations.removeFirst()
        guard case .abort = batch.outcome(for: query) else { Issue.record("Unreadable gate must abort"); return }
    }

    @Test("A stable destination with an unreadable arrow can be announced, while a lost candidate returns to search")
    func missingMetadata() throws {
        let query = try #require(SignQuery(queryID: UUID(), label: .restaurant))
        var batch = SignBatch(observations: Array(repeating: .init(reading: nil, distance: .near), count: 3))
        guard case .speak(let feedback) = batch.outcome(for: query) else {
            Issue.record("Expected a visible restaurant")
            return
        }
        #expect(feedback.arrowDirection == nil)
        #expect(feedback.distanceToSign == .near)
        batch.observations.removeLast()
        guard case .searchAgain = batch.outcome(for: query) else { Issue.record("Expected reacquisition"); return }
    }

    private func until(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(4))
        while !condition() && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        try #require(condition())
    }
}

private struct CameraFixture: FrameCapturing {
    let image: CGImage
    func start() async throws {}
    func stop() {}
    func discardFrames() {}

    func frames(_ delivery: FrameDelivery) -> AsyncThrowingStream<CapturedFrame, Error> {
        AsyncThrowingStream { continuation in
            let indices = delivery == .latest ? 0..<6 : 6..<11
            for i in indices {
                continuation.yield(CapturedFrame(frame: CameraFrame(image: image), timestamp: Double(i) / 30))
            }
            continuation.finish()
        }
    }
}

private actor VisionFixture: VisionModelInvoking {
    private var index = 0
    private var delay: Bool
    private var pending: CheckedContinuation<Void, Never>?
    private(set) var readBoxes: [Int] = []

    init(delayReaders: Bool) { delay = delayReaders }

    func releaseReaders() {
        delay = false
        pending?.resume()
        pending = nil
    }

    func detect(frame: CameraFrame, category: SignLabel) -> [SignBoundingBox] {
        defer { index += 1 }
        // Two centered observations, a miss, then a fresh streak of three.
        guard index != 2 else { return [] }
        let x = index < 6 ? 30 : 32 + index - 6
        let sign = SignBoundingBox(x: x, y: 30, width: 40, height: 40, frameWidth: 100, frameHeight: 100)
        let other = SignBoundingBox(x: 0, y: 0, width: 12, height: 12, frameWidth: 100, frameHeight: 100)
        return index < 6 ? [sign] : [other, sign]
    }

    func readSign(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) async -> SignReading? {
        if delay { await withCheckedContinuation { pending = $0 } }
        readBoxes.append(box.x)
        if category == .gate { return .gate(1) }
        return readBoxes.count <= 3 ? .arrow(.left) : nil
    }

    func estimateDistance(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) -> SignDistanceBand {
        [.near, .near, .medium, .medium, .far][readBoxes.count - 1]
    }
}

@MainActor
private final class LanguageFixture: SpeechInterfacing {
    let query: SignQuery
    var invalidInput = false
    var holdPlayback = false
    var playbackWasCancelled = false
    var listenCount = 0
    var spoken: [SpeechFeedback] = []
    var spokeAt: ContinuousClock.Instant?

    init(query: SignQuery) { self.query = query }

    func listenForQuery() async throws -> SignQuery? {
        listenCount += 1
        if listenCount == 1 { return invalidInput ? nil : query }
        try await Task.sleep(for: .seconds(60))
        return nil
    }

    func speak(_ feedback: SpeechFeedback) async throws {
        spoken.append(feedback)
        spokeAt = .now
        if holdPlayback {
            defer { playbackWasCancelled = Task.isCancelled }
            try await Task.sleep(for: .seconds(60))
        }
    }
}

private final class HapticFixture: HapticControlling {
    var changes: [(strength: HapticFeedback.Strength, time: ContinuousClock.Instant)] = []
    func apply(_ feedback: HapticFeedback) {
        if feedback.strength != changes.last?.strength { changes.append((feedback.strength, .now)) }
    }
}

@MainActor
private struct RoundFixture {
    let query: SignQuery
    let coordinator: SearchCoordinator
    let language: LanguageFixture
    let haptics = HapticFixture()
    let vision: VisionFixture
    let pulse = Duration.milliseconds(50)

    init(label: SignLabel, requestedGate: Int? = nil, delayReaders: Bool = false) throws {
        query = try #require(SignQuery(queryID: UUID(), label: label, requestedGateID: requestedGate))
        language = LanguageFixture(query: query)
        vision = VisionFixture(delayReaders: delayReaders)
        let context = try #require(CGContext(data: nil, width: 100, height: 100, bitsPerComponent: 8,
                                             bytesPerRow: 400, space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let camera = try CameraFixture(image: #require(context.makeImage()))
        coordinator = SearchCoordinator(camera: camera, speech: language, vision: vision, haptics: haptics, pulseDuration: pulse)
    }

    func start() { coordinator.start() }
}
