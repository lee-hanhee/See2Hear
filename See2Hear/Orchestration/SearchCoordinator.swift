import Foundation
import Observation
import os

@MainActor
@Observable
final class SearchCoordinator {
    enum Phase: Equatable {
        case stopped, starting, listening, searching, reading, speaking, stopping
        case suspended(String)
    }

    private(set) var phase: Phase = .stopped
    private(set) var isEnabled = false
    private var suspensionReason: String?
    private let camera: any FrameCapturing
    private let speech: any SpeechInterfacing
    private let vision: any VisionModelInvoking
    private let haptics: any HapticControlling
    private let logger = Logger(subsystem: "See2Hear", category: "Orchestration")
    private let pulseDuration: Duration
    private var sessionTask: Task<Void, Never>?
    private var roundTask: Task<Void, Error>?

    init(camera: any FrameCapturing, speech: any SpeechInterfacing, vision: any VisionModelInvoking,
         haptics: any HapticControlling, pulseDuration: Duration = .seconds(2)) {
        self.camera = camera
        self.speech = speech
        self.vision = vision
        self.haptics = haptics
        self.pulseDuration = pulseDuration
    }

    func start() {
        switch phase {
        case .stopped, .suspended: break
        case .starting, .listening, .searching, .reading, .speaking, .stopping: return
        }
        isEnabled = true
        suspensionReason = nil
        phase = .starting
        sessionTask = Task {
            defer {
                camera.stop()
                resetOutputs()
                roundTask = nil
                sessionTask = nil
                phase = suspensionReason.map(Phase.suspended) ?? .stopped
            }
            do {
                try await runSession()
            } catch {
                if !Task.isCancelled { stop(reason: error.localizedDescription) }
            }
        }
    }

    func resumeIfEnabled() {
        if isEnabled { start() }
    }

    private func runSession() async throws {
        try await camera.start()
        while !Task.isCancelled {
            phase = .listening
            let round = Task { try await self.runRound() }
            roundTask = round
            do {
                try await round.value
            } catch is CancellationError {
                // Cancelling one round returns to listening while the session remains enabled.
            } catch {
                guard !Task.isCancelled, !round.isCancelled else { continue }
                if phase == .listening || Self.resourceUnavailable(error) { throw error }
                logger.error("Search round ended: \(error.localizedDescription, privacy: .public)")
            }
            roundTask = nil
            resetOutputs()
        }
    }

    private static func resourceUnavailable(_ error: Error) -> Bool {
        if let capture = error as? CaptureError {
            switch capture {
            case .permission, .unavailable: return true
            case .frame: return false
            }
        }
        if let model = error as? VisionModelError {
            switch model {
            case .missingModel, .unexpectedOutput: return true
            case .invalidBox, .invalidImage: return false
            }
        }
        return false
    }

    func cancelRound() {
        roundTask?.cancel()
        resetOutputs()
    }

    func stop(reason: String? = nil) {
        if reason == nil { isEnabled = false }
        suspensionReason = reason
        phase = .stopping
        sessionTask?.cancel()
        cancelRound()
        camera.stop()
        if sessionTask == nil { phase = reason.map(Phase.suspended) ?? .stopped }
    }

    private func resetOutputs() {
        camera.discardFrames()
        haptics.apply(HapticFeedback(strength: .off))
    }

    private func runRound() async throws {
        try Task.checkCancellation()
        guard let query = try await speech.listenForQuery() else { return }
        try Task.checkCancellation()
        while true {
            try Task.checkCancellation()
            phase = .searching
            let target = try await findCenteredSign(for: query)
            try Task.checkCancellation()
            phase = .reading
            let batch = try await readWithPulse(for: query, target: target)
            try Task.checkCancellation()
            switch batch.outcome(for: query) {
            case .abort: return
            case .searchAgain: continue
            case .speak(let feedback):
                phase = .speaking
                try await speech.speak(feedback)
                return
            }
        }
    }

    private func findCenteredSign(for query: SignQuery) async throws -> TrackedSign {
        var streak = CenteringStreak()
        defer { camera.discardFrames() }
        for try await sample in camera.frames(.latest) {
            try Task.checkCancellation()
            let box = try await vision.detect(frame: sample.frame, category: query.label).first
            try Task.checkCancellation()
            if let target = streak.observe(box, at: sample.timestamp) { return target }
            haptics.apply(HapticFeedback(strength: box == nil ? .off : .low))
        }
        try Task.checkCancellation()
        throw CaptureError.unavailable
    }

    private func readWithPulse(for query: SignQuery, target: TrackedSign) async throws -> SignBatch {
        haptics.apply(HapticFeedback(strength: .high))
        let deadline = ContinuousClock.now.advanced(by: pulseDuration)
        async let pulse: Void = finishPulse(at: deadline)
        var track = target
        var batch = SignBatch()
        var received = 0
        defer { camera.discardFrames() }
        for try await sample in camera.frames(.fiveConsecutive) {
            try Task.checkCancellation()
            received += 1
            let boxes = try await vision.detect(frame: sample.frame, category: query.label)
            try Task.checkCancellation()
            if let matched = track.matching(boxes, at: sample.timestamp) {
                track = matched
                let reading = try await vision.readSign(frame: sample.frame, category: query.label, box: matched.box)
                try Task.checkCancellation()
                let distance = try await vision.estimateDistance(frame: sample.frame, category: query.label, box: matched.box)
                batch.observations.append(.init(reading: reading, distance: distance))
            }
            if received == SignBatch.size { break }
        }
        try Task.checkCancellation()
        guard received == SignBatch.size else { throw CaptureError.unavailable }
        try await pulse
        return batch
    }

    private func finishPulse(at deadline: ContinuousClock.Instant) async throws {
        defer { haptics.apply(HapticFeedback(strength: .off)) }
        try await ContinuousClock().sleep(until: deadline)
    }
}
