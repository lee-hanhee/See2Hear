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

    var body: some View {
        VStack(spacing: 20) {
            switch coordinator.phase {
            case .stopped, .starting: Text("Starting camera…")
            case .listening: Text("Listening for a sign request")
            case .searching: Text("Searching for the sign")
            case .reading: Text("Reading the sign")
            case .speaking: Text("Speaking the result")
            case .suspended(let reason):
                Text(reason)
                if !cameraInterrupted && !audioInterrupted {
                    Button("Try again") { resumeIfAvailable() }
                }
            }
            switch coordinator.phase {
            case .searching, .reading, .speaking:
                Button("Cancel search") { coordinator.cancelRound() }
            default: EmptyView()
            }
        }
        .padding()
        .onChange(of: scenePhase, initial: true) { _, phase in
            if phase == .active { resumeIfAvailable() }
            if phase == .background { coordinator.stop() }
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
        if scenePhase == .active && !cameraInterrupted && !audioInterrupted { coordinator.start() }
    }
}
