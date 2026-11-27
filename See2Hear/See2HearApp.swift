import AVFoundation
import Combine
import SwiftUI

@main
struct See2HearApp: App {
    private let camera: CameraCapture
    private let coordinator: SearchCoordinator

    init() {
        let camera = CameraCapture()
        self.camera = camera
        coordinator = SearchCoordinator(camera: camera, speech: SpeechInterface(listeningHaptics: false),
                                        vision: CoreMLSignRecognizer(), haptics: HapticController())
    }

    var body: some Scene {
        WindowGroup { SearchView(coordinator: coordinator, session: camera.session) }
    }
}

private struct SearchView: View {
    @Environment(\.scenePhase) private var scenePhase
    let coordinator: SearchCoordinator
    let session: AVCaptureSession
    @State private var cameraInterrupted = false
    @State private var audioInterrupted = false

    @ScaledMetric(relativeTo: .largeTitle) private var actionSize: CGFloat = 72

    private var canStart: Bool {
        switch coordinator.phase {
        case .stopped, .suspended: true
        case .starting, .listening, .searching, .reading, .speaking, .stopping: false
        }
    }

    private var actionLabel: String {
        if coordinator.phase == .stopping { return "Stopping…" }
        return canStart ? "Start" : "Stop"
    }

    private var status: String {
        switch coordinator.phase {
        case .stopped: "Ready for a new request"
        case .starting: "Starting"
        case .listening: coordinator.isLoadingModel ? "Loading model…" : "Listening for your request"
        case .searching: "Searching for the sign"
        case .reading: "Reading the sign"
        case .speaking: "Speaking the result"
        case .stopping: "Finishing cleanup"
        case .suspended(let reason): reason
        }
    }

    var body: some View {
        Button {
            if canStart {
                if scenePhase == .active && !cameraInterrupted && !audioInterrupted { coordinator.start() }
            } else {
                coordinator.stop()
            }
        } label: {
            VStack(spacing: 24) {
                Text(actionLabel).font(.system(size: actionSize, weight: .bold, design: .rounded))
                Text(status).font(.title2)
            }
            .multilineTextAlignment(.center)
            .padding(32)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .foregroundStyle(.white)
            .background(canStart ? Color.blue : Color(red: 0.65, green: 0.05, blue: 0.05))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .ignoresSafeArea()
        .disabled(coordinator.phase == .stopping || (canStart && (cameraInterrupted || audioInterrupted)))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(actionLabel)
        .accessibilityValue(status)
        .accessibilityHint(canStart ? "Starts a new sign request" : "Ends the current session")
        // TODO(UI): review VoiceOver labels, status changes, and full-screen button activation.
        // TODO(UI): refine contrast and check large-text layout across screen sizes.
        .onChange(of: scenePhase, initial: true) { _, phase in
            if phase == .active { resumeIfAvailable() }
            if phase == .background { coordinator.stop(reason: "App in background.") }
        }
        .onReceive(notifications(AVCaptureSession.wasInterruptedNotification, from: session)) { _ in
            cameraInterrupted = true
            coordinator.stop(reason: "Camera interrupted.")
        }
        .onReceive(notifications(AVCaptureSession.interruptionEndedNotification, from: session)) { _ in
            cameraInterrupted = false
            resumeIfAvailable()
        }
        .onReceive(notifications(AVCaptureSession.runtimeErrorNotification, from: session)) { _ in
            coordinator.stop(reason: "Camera unavailable.")
        }
        .onReceive(notifications(AVAudioSession.interruptionNotification, from: AVAudioSession.sharedInstance())) { notification in
            guard let type = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt else { return }
            audioInterrupted = type == AVAudioSession.InterruptionType.began.rawValue
            if audioInterrupted {
                coordinator.stop(reason: "Audio interrupted.")
            } else {
                resumeIfAvailable()
            }
        }
        .onDisappear { coordinator.stop() }
    }

    private func notifications(_ name: Notification.Name, from object: AnyObject) -> some Publisher<Notification, Never> {
        NotificationCenter.default.publisher(for: name, object: object).receive(on: DispatchQueue.main)
    }

    private func resumeIfAvailable() {
        // TODO(Orchestration): defer recovery received during stopping until cleanup finishes, if still enabled.
        if scenePhase == .active && !cameraInterrupted && !audioInterrupted { coordinator.resumeIfEnabled() }
    }
}
