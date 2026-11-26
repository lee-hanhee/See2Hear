import CoreHaptics
import Foundation
import os
import UIKit

/// Plays a repeating pulse whose intensity and sharpness follow the requested strength.
/// All engine state is confined to `queue`.
final class HapticController: HapticControlling, @unchecked Sendable {
    private enum Pulse {
        static let leadIn: TimeInterval = 0.05
        static let duration: TimeInterval = 0.12
        static let period: TimeInterval = 0.4
        static let baseSharpness: Float = 0.3
    }

    private struct Level {
        let intensity: Float
        /// Added to `Pulse.baseSharpness` by Core Haptics.
        let sharpnessOffset: Float

        static let low = Level(intensity: 0.4, sharpnessOffset: 0)
        static let high = Level(intensity: 1, sharpnessOffset: 0.4)
    }

    private let queue = DispatchQueue(label: "See2Hear.HapticController")
    private let logger = Logger(subsystem: "See2Hear", category: "Haptic")
    private let supportsHaptics = CHHapticEngine.capabilitiesForHardware().supportsHaptics

    private var engine: CHHapticEngine?
    private var player: CHHapticAdvancedPatternPlayer?
    private var requested: HapticFeedback.Strength = .off
    private var playing: HapticFeedback.Strength = .off
    private var foregroundObserver: NSObjectProtocol?

    @MainActor
    init() {
        foregroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didBecomeActiveNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            self?.retryPendingStrength()
        }
    }

    deinit {
        if let foregroundObserver {
            NotificationCenter.default.removeObserver(foregroundObserver)
        }
    }

    func apply(_ feedback: HapticFeedback) {
        let strength = feedback.strength
        queue.async { [self] in
            requested = strength
            reconcile()
        }
    }

    // MARK: - Queue-confined

    private func reconcile() {
        guard supportsHaptics, requested != playing else { return }

        do {
            switch requested {
            case .off:
                try player?.stop(atTime: CHHapticTimeImmediate)
                player = nil
            case .low:
                try play(.low)
            case .high:
                try play(.high)
            }
            playing = requested
        } catch {
            // Stop any partial playback. The next apply or foreground event retries.
            try? player?.stop(atTime: CHHapticTimeImmediate)
            player = nil
            playing = .off
            logger.error("Could not apply haptic strength: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func play(_ level: Level) throws {
        if let player {
            try send(level, to: player)
            return
        }

        let engine = try preparedEngine()
        try engine.start()

        let newPlayer = try engine.makeAdvancedPlayer(with: makePulsePattern())
        newPlayer.loopEnabled = true
        newPlayer.loopEnd = Pulse.period
        try newPlayer.start(atTime: CHHapticTimeImmediate)
        player = newPlayer
        // The silent lead-in gives this time to land before the first pulse.
        try send(level, to: newPlayer)
    }

    private func send(_ level: Level, to player: CHHapticAdvancedPatternPlayer) throws {
        try player.sendParameters(
            [
                CHHapticDynamicParameter(
                    parameterID: .hapticIntensityControl,
                    value: level.intensity,
                    relativeTime: 0
                ),
                CHHapticDynamicParameter(
                    parameterID: .hapticSharpnessControl,
                    value: level.sharpnessOffset,
                    relativeTime: 0
                ),
            ],
            atTime: CHHapticTimeImmediate
        )
    }

    private func preparedEngine() throws -> CHHapticEngine {
        if let engine { return engine }

        let newEngine = try CHHapticEngine()
        newEngine.playsHapticsOnly = true
        newEngine.isAutoShutdownEnabled = true
        newEngine.stoppedHandler = { [weak self] reason in
            self?.logger.info("Haptic engine stopped: \(reason.rawValue, privacy: .public)")
            // Auto-shutdown only idles an unused engine, and `play` starts it again.
            guard reason != .idleTimeout else { return }
            self?.restartAfterEngineLoss()
        }
        newEngine.resetHandler = { [weak self] in
            self?.restartAfterEngineLoss()
        }
        engine = newEngine
        return newEngine
    }

    private func makePulsePattern() throws -> CHHapticPattern {
        let pulse = CHHapticEvent(
            eventType: .hapticContinuous,
            parameters: [
                CHHapticEventParameter(parameterID: .hapticIntensity, value: 1),
                CHHapticEventParameter(parameterID: .hapticSharpness, value: Pulse.baseSharpness),
            ],
            relativeTime: Pulse.leadIn,
            duration: Pulse.duration
        )
        // Silent tail so the pattern lasts exactly one loop period.
        let silence = CHHapticEvent(
            eventType: .hapticContinuous,
            parameters: [CHHapticEventParameter(parameterID: .hapticIntensity, value: 0)],
            relativeTime: Pulse.leadIn + Pulse.duration,
            duration: Pulse.period - Pulse.leadIn - Pulse.duration
        )
        return try CHHapticPattern(events: [pulse, silence], parameters: [])
    }

    // MARK: - Interruptions

    /// The engine's players are invalid after a stop or reset, so rebuild playback.
    /// Stopping first ensures a late callback can never leave two players running.
    private func restartAfterEngineLoss() {
        queue.async { [self] in
            try? player?.stop(atTime: CHHapticTimeImmediate)
            player = nil
            playing = .off
            reconcile()
        }
    }

    /// Retries a strength that could not start, such as while the app was in the background.
    private func retryPendingStrength() {
        queue.async { [self] in
            reconcile()
        }
    }
}
