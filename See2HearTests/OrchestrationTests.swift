import CoreGraphics
import Foundation
import Testing
@testable import See2Hear

@MainActor
struct OrchestrationTests {
    @Test("Search start precedes detection; result playback follows five fresh frames and the pulse")
    func successfulRound() async throws {
        let rig = try RoundFixture(label: .washroom)
        rig.language.heldEvent = .searchStarted
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.events == [.searchStarted] }
        #expect(rig.camera.batchTimes.isEmpty)
        #expect(await rig.vision.readBoxes.isEmpty)
        #expect(await rig.vision.detectCount == 0)
        rig.language.releasePlayback()
        try await until { rig.language.listenCount == 2 }
        let feedback = try #require(rig.language.spoken.last)
        #expect(feedback.query.queryID == rig.query.queryID)
        #expect(feedback.event == .recognized(arrowDirection: .left, distanceToSign: nil))
        #expect(rig.language.spoken.count == 2)
        #expect(await rig.vision.readBoxes == [32, 33, 34, 35, 36])
        #expect(rig.haptics.changes.map(\.strength).starts(with: [.off, .low, .off, .low, .high, .off]))
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
        #expect(rig.language.events == [.searchStarted])
        rig.coordinator.cancelRound()
        await rig.vision.releaseReaders()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.events == [.searchStarted])
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

    @Test("Round cancellation interrupts result playback before the next listen")
    func cancelDuringPlayback() async throws {
        let rig = try RoundFixture(label: .restaurant)
        rig.language.heldEvent = .recognized(arrowDirection: .left, distanceToSign: nil)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.coordinator.phase == .speaking && rig.language.spoken.count == 2 }
        rig.coordinator.cancelRound()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.playbackWasCancelled)
        #expect(rig.haptics.changes.last?.strength == .off)
    }

    @Test("Wrong gates are announced and suppressed while the same query continues")
    func wrongGate() async throws {
        let wrong = box(20)
        let requested = box(40)
        let rig = try RoundFixture(label: .gate, requestedGate: 2, scripts: [
            .search([wrong]), .read(Array(repeating: .gate(1), count: 5), boxes: [wrong]),
            .search([wrong, requested], gap: 2),
            .read(Array(repeating: .gate(2), count: 5), boxes: [wrong, requested])
        ])
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.events == [
            .searchStarted, .wrongGate(observedGateID: 1), .recognized(arrowDirection: nil, distanceToSign: nil)
        ])
        #expect(rig.language.spoken.allSatisfy { $0.query.queryID == rig.query.queryID })
        #expect(await rig.vision.readBoxes == Array(repeating: 20, count: 5) + Array(repeating: 40, count: 5))
    }

    @Test("Suppressed candidates follow movement and become eligible after disappearing")
    func rejectedCandidateFiltering() {
        var rejected = RejectedSigns(tracks: [TrackedSign(box: box(20), timestamp: 0)])
        #expect(rejected.eligible([box(22), box(45)], at: 2).map(\.x) == [45])
        #expect(rejected.eligible([], at: 3).isEmpty)
        #expect(rejected.eligible([box(22)], at: 4).map(\.x) == [22])
    }

    @Test("Unreadable attempts use fresh votes and can succeed on the third attempt")
    func retrySuccess() async throws {
        let unreadable: [SignReading?] = [.gate(1), .gate(1), nil, nil, nil]
        let rig = try RoundFixture(label: .gate, requestedGate: 1, scripts: [
            .search(), .read(unreadable), .read(unreadable, gap: 1),
            .read(Array(repeating: .gate(1), count: 5), gap: 1)
        ])
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.events == [
            .searchStarted, .retryingRecognition(retryNumber: .first), .retryingRecognition(retryNumber: .second),
            .recognized(arrowDirection: nil, distanceToSign: nil)
        ])
        #expect(rig.camera.batchTimes.count == 3)
        #expect(await rig.vision.readBoxes.count == 15)
    }

    @Test("The third unreadable attempt announces failure and returns to listening")
    func retryFailure() async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 1, scripts: [
            .search(), .read(), .read(gap: 1), .read(gap: 1)
        ])
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.events == [
            .searchStarted, .retryingRecognition(retryNumber: .first),
            .retryingRecognition(retryNumber: .second), .recognitionFailed
        ])
        #expect(rig.camera.batchTimes.count == 3)
    }

    @Test("A retry waits for speech and follows the candidate after a playback-length gap")
    func retryPlaybackOrdering() async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 1, scripts: [
            .search(), .read(), .read(Array(repeating: .gate(1), count: 5), gap: 1)
        ])
        rig.language.heldEvent = .retryingRecognition(retryNumber: .first)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.events.last == .retryingRecognition(retryNumber: .first) }
        #expect(rig.camera.batchTimes.count == 1)
        #expect(rig.haptics.changes.last?.strength == .off)
        let releasedAt = ContinuousClock.now
        rig.language.releasePlayback()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.camera.batchTimes.count == 2)
        #expect(rig.camera.batchTimes[1] >= releasedAt)
        #expect(rig.language.events.last == .recognized(arrowDirection: nil, distanceToSign: nil))
    }

    @Test("Candidate loss announces once and resets attempts before reacquisition")
    func candidateLoss() async throws {
        let lost = StreamScript(delivery: .fiveConsecutive, frames: [
            FramePrediction(boxes: [box(30)], reading: .gate(1)),
            FramePrediction(boxes: [box(30)], reading: .gate(1)),
            FramePrediction(boxes: []), FramePrediction(boxes: []), FramePrediction(boxes: [])
        ], gap: 1)
        let rig = try RoundFixture(label: .gate, requestedGate: 1, scripts: [
            .search(), .read(), lost, .search([box(40)]), .read(boxes: [box(40)]),
            .read(Array(repeating: .gate(1), count: 5), boxes: [box(40)], gap: 1)
        ])
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.events == [
            .searchStarted, .retryingRecognition(retryNumber: .first), .lostSightOfSign,
            .retryingRecognition(retryNumber: .first), .recognized(arrowDirection: nil, distanceToSign: nil)
        ])
        #expect(rig.camera.batchTimes.count == 4)
        #expect(rig.language.spoken.allSatisfy { $0.query.queryID == rig.query.queryID })
    }

    @Test("Stop during search-start or retry speech prevents further capture",
          arguments: [SearchFeedbackEvent.searchStarted, .retryingRecognition(retryNumber: .first)])
    func stopDuringAnnouncement(event: SearchFeedbackEvent) async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 1, scripts: [.search(), .read()])
        rig.language.heldEvent = event
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.events.last == event }
        let batchesBeforeStop = rig.camera.batchTimes.count
        rig.coordinator.stop()
        try await until { rig.coordinator.phase == .stopped }
        #expect(rig.language.playbackWasCancelled)
        #expect(rig.camera.batchTimes.count == batchesBeforeStop)
        #expect(!rig.coordinator.isEnabled)
        #expect(rig.haptics.changes.last?.strength == .off)
    }

    @Test("An interruption preempts a read and resumption starts with a fresh query")
    func interruption() async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 1, delayReaders: true)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.haptics.changes.last?.strength == .high }
        rig.coordinator.stop(reason: "Interrupted")
        #expect(rig.haptics.changes.last?.strength == .off)
        #expect(rig.coordinator.phase == .stopping)
        rig.coordinator.start()
        #expect(rig.language.listenCount == 1)
        await rig.vision.releaseReaders()
        try await until { rig.coordinator.phase == .suspended("Interrupted") }
        rig.coordinator.resumeIfEnabled()
        try await until { rig.language.listenCount == 2 }
        #expect(rig.language.events == [.searchStarted])
    }

    @Test("Launch waits for Start; automatic resume cannot undo button Stop")
    func explicitStartAndStop() async throws {
        let rig = try RoundFixture(label: .washroom)
        rig.coordinator.resumeIfEnabled()
        #expect(rig.language.listenCount == 0)
        #expect(rig.coordinator.phase == .stopped)
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.listenCount == 2 }
        rig.coordinator.stop()
        #expect(!rig.coordinator.isEnabled)
        #expect(rig.coordinator.phase == .stopping)
        try await until { rig.coordinator.phase == .stopped }
        rig.coordinator.resumeIfEnabled()
        #expect(rig.language.listenCount == 2)
        rig.coordinator.start()
        try await until { rig.language.listenCount == 3 }
        #expect(rig.coordinator.isEnabled)
    }

    @Test("Button Stop disables the session before delayed model work finishes")
    func stopDuringModelWork() async throws {
        let rig = try RoundFixture(label: .gate, requestedGate: 5)
        rig.language.delayQuery = true
        rig.start()
        defer { rig.coordinator.stop() }
        try await until { rig.language.queryIsPending }
        rig.coordinator.stop()
        #expect(rig.coordinator.phase == .stopping)
        #expect(!rig.coordinator.isEnabled)
        #expect(rig.haptics.changes.last?.strength == .off)
        rig.coordinator.start()
        rig.coordinator.resumeIfEnabled()
        #expect(rig.language.listenCount == 1)
        rig.language.releaseQuery()
        try await until { rig.coordinator.phase == .stopped }
        #expect(rig.language.spoken.isEmpty)
        rig.language.delayQuery = false
        rig.coordinator.start()
        try await until { rig.language.listenCount == 2 }
    }

    @Test("Button Stop preempts reading or result playback", arguments: ["read", "playback"])
    func stopDuringSearch(stage: String) async throws {
        let rig = try RoundFixture(label: .restaurant, delayReaders: stage == "read")
        rig.language.heldEvent = stage == "playback" ? .recognized(arrowDirection: .left, distanceToSign: nil) : nil
        rig.start()
        defer { rig.coordinator.stop() }
        if stage == "read" {
            try await until { rig.haptics.changes.last?.strength == .high }
        } else {
            try await until { rig.language.events.last == .recognized(arrowDirection: .left, distanceToSign: nil) }
        }
        rig.coordinator.stop()
        #expect(rig.coordinator.phase == .stopping)
        #expect(!rig.coordinator.isEnabled)
        #expect(rig.haptics.changes.last?.strength == .off)
        await rig.vision.releaseReaders()
        try await until { rig.coordinator.phase == .stopped }
        if stage == "playback" {
            #expect(rig.language.playbackWasCancelled)
        } else {
            #expect(rig.language.events == [.searchStarted])
        }
        rig.coordinator.resumeIfEnabled()
        #expect(rig.language.listenCount == 1)
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
        #expect(feedback.event == .recognized(arrowDirection: nil, distanceToSign: nil))
        batch.observations.removeFirst()
        guard case .unreadableGate = batch.outcome(for: query) else { Issue.record("Expected unreadable digits"); return }
    }

    @Test("A stable destination with an unreadable arrow can be announced, while a lost candidate returns to search")
    func missingMetadata() throws {
        let query = try #require(SignQuery(queryID: UUID(), label: .restaurant))
        var batch = SignBatch(observations: Array(repeating: .init(reading: nil, distance: .near), count: 3))
        guard case .speak(let feedback) = batch.outcome(for: query) else {
            Issue.record("Expected a visible restaurant")
            return
        }
        #expect(feedback.event == .recognized(arrowDirection: nil, distanceToSign: .near))
        batch.observations.removeLast()
        guard case .lostCandidate = batch.outcome(for: query) else { Issue.record("Expected reacquisition"); return }
    }

    private func until(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(4))
        while !condition() && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        try #require(condition())
    }
}

private func box(_ x: Int) -> SignBoundingBox {
    SignBoundingBox(x: x, y: 30, width: 40, height: 40, frameWidth: 100, frameHeight: 100)
}

private struct FramePrediction {
    var boxes: [SignBoundingBox]
    var reading: SignReading?
    var distance: SignDistanceBand = .near
}

private struct StreamScript {
    let delivery: FrameDelivery
    let frames: [FramePrediction]
    var gap: TimeInterval = 1.0 / 30

    static func search(_ boxes: [SignBoundingBox] = [box(30)], gap: TimeInterval = 1.0 / 30) -> Self {
        Self(delivery: .latest, frames: Array(repeating: FramePrediction(boxes: boxes), count: 3), gap: gap)
    }

    static func read(_ readings: [SignReading?] = Array(repeating: nil, count: 5),
                     boxes: [SignBoundingBox] = [box(30)], gap: TimeInterval = 1.0 / 30) -> Self {
        let distances: [SignDistanceBand] = [.near, .near, .medium, .medium, .far]
        return Self(delivery: .fiveConsecutive, frames: readings.enumerated().map { index, reading in
            FramePrediction(boxes: boxes, reading: reading, distance: distances[index])
        }, gap: gap)
    }
}

private final class CameraFixture: FrameCapturing, @unchecked Sendable {
    private let lock = NSLock()
    private var streams: [(delivery: FrameDelivery, frames: [CapturedFrame])] = []
    private var times: [ContinuousClock.Instant] = []
    var batchTimes: [ContinuousClock.Instant] { lock.withLock { times } }

    init(scripts: [StreamScript]) throws {
        let context = try #require(CGContext(data: nil, width: 100, height: 100, bitsPerComponent: 8,
                                             bytesPerRow: 400, space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let image = try #require(context.makeImage())
        var timestamp = 0.0
        for script in scripts {
            timestamp += script.gap
            var frames: [CapturedFrame] = []
            for _ in script.frames {
                frames.append(CapturedFrame(frame: CameraFrame(image: image), timestamp: timestamp))
                timestamp += 1.0 / 30
            }
            streams.append((script.delivery, frames))
        }
    }

    func start() async throws {}
    func stop() {}
    func discardFrames() {}

    func frames(_ delivery: FrameDelivery) -> AsyncThrowingStream<CapturedFrame, Error> {
        let next = lock.withLock { () -> (delivery: FrameDelivery, frames: [CapturedFrame])? in
            if delivery == .fiveConsecutive { times.append(.now) }
            return streams.isEmpty ? nil : streams.removeFirst()
        }
        return AsyncThrowingStream { continuation in
            guard let next else {
                Issue.record("Unexpected camera stream request")
                continuation.finish(throwing: CaptureError.unavailable)
                return
            }
            #expect(delivery == next.delivery)
            for frame in next.frames { continuation.yield(frame) }
            continuation.finish()
        }
    }
}

private actor VisionFixture: VisionModelInvoking {
    private var predictions: [FramePrediction]
    private var current: FramePrediction?
    private var delay: Bool
    private var pending: CheckedContinuation<Void, Never>?
    private(set) var readBoxes: [Int] = []
    private(set) var detectCount = 0

    init(predictions: [FramePrediction], delayReaders: Bool) {
        self.predictions = predictions
        delay = delayReaders
    }

    func releaseReaders() {
        delay = false
        pending?.resume()
        pending = nil
    }

    func detect(frame: CameraFrame, category: SignLabel) throws -> [SignBoundingBox] {
        let prediction = try #require(predictions.first)
        predictions.removeFirst()
        current = prediction
        detectCount += 1
        return prediction.boxes
    }

    func readSign(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) async throws -> SignReading? {
        if delay { await withCheckedContinuation { pending = $0 } }
        readBoxes.append(box.x)
        return try #require(current).reading
    }

    func estimateDistance(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) throws -> SignDistanceBand {
        try #require(current).distance
    }
}

@MainActor
private final class LanguageFixture: SpeechInterfacing {
    let query: SignQuery
    var invalidInput = false
    var heldEvent: SearchFeedbackEvent?
    var isLoadingModel = false
    var events: [SearchFeedbackEvent] { spoken.map(\.event) }
    private var playback: AsyncStream<Void>.Continuation?
    var playbackWasCancelled = false
    var listenCount = 0
    var spoken: [SpeechFeedback] = []
    var spokeAt: ContinuousClock.Instant?
    var delayQuery = false
    var queryIsPending: Bool { pendingQuery != nil }
    private var pendingQuery: CheckedContinuation<Void, Never>?

    func releaseQuery() {
        pendingQuery?.resume()
        pendingQuery = nil
    }

    init(query: SignQuery) { self.query = query }

    func listenForQuery() async throws -> SignQuery? {
        listenCount += 1
        if delayQuery { await withCheckedContinuation { pendingQuery = $0 } }
        if listenCount == 1 { return invalidInput ? nil : query }
        try await Task.sleep(for: .seconds(60))
        return nil
    }

    func releasePlayback() {
        playback?.yield(())
        playback?.finish()
    }

    func speak(_ feedback: SpeechFeedback) async throws {
        spoken.append(feedback)
        spokeAt = .now
        if feedback.event == heldEvent {
            defer { playbackWasCancelled = Task.isCancelled; playback = nil }
            let stream = AsyncStream<Void> { playback = $0 }
            for await _ in stream { break }
            try Task.checkCancellation()
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
    let camera: CameraFixture
    let pulse = Duration.milliseconds(50)

    init(label: SignLabel, requestedGate: Int? = nil, delayReaders: Bool = false,
         scripts: [StreamScript]? = nil) throws {
        query = try #require(SignQuery(queryID: UUID(), label: label, requestedGateID: requestedGate))
        language = LanguageFixture(query: query)
        let guidance = StreamScript(delivery: .latest, frames: [
            FramePrediction(boxes: [box(30)]), FramePrediction(boxes: [box(30)]), FramePrediction(boxes: []),
            FramePrediction(boxes: [box(30)]), FramePrediction(boxes: [box(30)]), FramePrediction(boxes: [box(30)])
        ])
        let readings: [SignReading?] = label == .gate ? Array(repeating: .gate(1), count: 5)
            : [.arrow(.left), .arrow(.left), .arrow(.left), nil, nil]
        let distances: [SignDistanceBand] = [.near, .near, .medium, .medium, .far]
        let read = StreamScript(delivery: .fiveConsecutive, frames: readings.enumerated().map { index, reading in
            FramePrediction(boxes: [box(0), box(32 + index)], reading: reading, distance: distances[index])
        })
        let streams = scripts ?? [guidance, read]
        camera = try CameraFixture(scripts: streams)
        vision = VisionFixture(predictions: streams.flatMap(\.frames), delayReaders: delayReaders)
        coordinator = SearchCoordinator(camera: camera, speech: language, vision: vision, haptics: haptics, pulseDuration: pulse)
    }

    func start() { coordinator.start() }
}
