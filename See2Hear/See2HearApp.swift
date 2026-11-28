import Accessibility
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

    // Fixed colours keep white text above 7:1 contrast in light and dark mode.
    private static let startColor = Color(red: 0.0, green: 0.25, blue: 0.70)
    private static let stopColor = Color(red: 0.65, green: 0.05, blue: 0.05)

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
        case .listening: coordinator.listeningStatus
        case .searching: "Searching for the sign"
        case .reading: "Reading the sign"
        case .speaking: "Speaking the result"
        case .stopping: "Finishing cleanup"
        case .suspended(let reason): reason
        }
    }

    private var isDisabled: Bool {
        coordinator.phase == .stopping || (canStart && (cameraInterrupted || audioInterrupted))
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            actionButton
            // Debug aid: shows exactly what speech recognition heard. It stays inside the
            // safe area above the home indicator, and taps pass through to the button.
            if !coordinator.heardText.isEmpty {
                Text("Heard: \u{201C}\(coordinator.heardText)\u{201D}")
                    .font(.body)
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                    .truncationMode(.head)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 12)
                    .allowsHitTesting(false)
            }
        }
        .overlay(alignment: .topTrailing) {
            // Stays attached to the session and is only hidden, because attaching or detaching a
            // preview layer reconfigures the capture session on the main thread and stalls taps.
            ScanPreviewView(session: session, preview: coordinator.scanPreview)
                .padding(12)
                .opacity(coordinator.phase == .searching || coordinator.phase == .reading ? 1 : 0)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
    }

    private var actionButton: some View {
        Button(action: toggle) {
            VStack(spacing: 24) {
                Text(actionLabel)
                    .font(.system(size: actionSize, weight: .bold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.4)
                Text(status)
                    .font(.title2)
                    .minimumScaleFactor(0.6)
            }
            .multilineTextAlignment(.center)
            .padding(32)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .foregroundStyle(.white)
            // Stays red while stopping and turns blue once cleanup finishes.
            .background(canStart ? Self.startColor : Self.stopColor)
            .transaction { $0.animation = nil }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .ignoresSafeArea()
        .disabled(isDisabled)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(actionLabel)
        .accessibilityValue(status)
        .accessibilityHint(canStart ? "Starts listening for a sign request" : "Ends the current session")
        // VoiceOver stays quiet after Start so it doesn't talk over the microphone or the app's speech.
        .accessibilityAddTraits(canStart ? .startsMediaSession : [])
        // The two-finger double-tap toggles the session from anywhere on screen.
        .accessibilityAction(.magicTap) { if !isDisabled { toggle() } }
        // The app speaks its own search feedback; VoiceOver only announces why a session ended unexpectedly.
        .onChange(of: coordinator.phase) { _, phase in
            if case .suspended(let reason) = phase {
                AccessibilityNotification.Announcement(reason).post()
            }
        }
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

    private func toggle() {
        if canStart {
            if scenePhase == .active && !cameraInterrupted && !audioInterrupted { coordinator.start() }
        } else {
            coordinator.stop()
        }
    }

    private func resumeIfAvailable() {
        if scenePhase == .active && !cameraInterrupted && !audioInterrupted { coordinator.resumeIfEnabled() }
    }
}

/// Debug aid: the live camera feed, with the latest detections outlined and the target in yellow.
/// Detections update at the detector's pace, so they can trail the live video slightly.
private struct ScanPreviewView: View {
    let session: AVCaptureSession
    let preview: SearchCoordinator.ScanPreview?

    var body: some View {
        // Capture frames are 1080 × 1920 when upright, so the preview uses the same 9:16 shape.
        CameraPreview(session: session)
            .aspectRatio(9 / 16, contentMode: .fit)
            .frame(width: 120)
            .background(.black)
            .overlay {
                GeometryReader { geometry in
                    if let preview {
                        ForEach(Array(preview.detections.enumerated()), id: \.offset) { _, box in
                            outline(box, in: geometry.size, color: .white, width: 1)
                        }
                        if let target = preview.target {
                            outline(target, in: geometry.size, color: .yellow, width: 2)
                        }
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.white, lineWidth: 2))
    }

    private func outline(_ box: SignBoundingBox, in size: CGSize, color: Color, width: CGFloat) -> some View {
        let scaleX = size.width / CGFloat(box.frameWidth)
        let scaleY = size.height / CGFloat(box.frameHeight)
        return Rectangle()
            .stroke(color, lineWidth: width)
            .frame(width: CGFloat(box.width) * scaleX, height: CGFloat(box.height) * scaleY)
            .position(x: (CGFloat(box.x) + CGFloat(box.width) / 2) * scaleX,
                      y: (CGFloat(box.y) + CGFloat(box.height) / 2) * scaleY)
    }
}

/// Shows the capture session directly, independent of frame processing.
private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> CameraPreviewView { CameraPreviewView(session: session) }
    func updateUIView(_ uiView: CameraPreviewView, context: Context) { uiView.trackRotationIfNeeded() }
}

private final class CameraPreviewView: UIView {
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    init(session: AVCaptureSession) {
        super.init(frame: .zero)
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspect
        layer.addSublayer(previewLayer)
        trackRotationIfNeeded()
    }

    /// Keeps the preview upright the same way capture keeps model frames upright.
    /// The camera input is added on the first Start, so this retries until it exists.
    func trackRotationIfNeeded() {
        guard rotation == nil,
              let device = previewLayer.session?.inputs.compactMap({ ($0 as? AVCaptureDeviceInput)?.device }).first
        else { return }
        let rotation = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        self.rotation = rotation
        rotationObservation = rotation.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]) {
            @Sendable [weak self] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelPreview
            Task { @MainActor [weak self] in self?.applyRotation(angle) }
        }
    }

    required init?(coder: NSCoder) { nil }

    private func applyRotation(_ angle: CGFloat) {
        guard let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(angle) else { return }
        connection.videoRotationAngle = angle
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
}
