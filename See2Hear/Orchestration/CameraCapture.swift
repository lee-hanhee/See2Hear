import AVFoundation
import CoreImage
import ImageIO

enum FrameDelivery {
    case latest
    case fiveConsecutive
}

protocol FrameCapturing: Sendable {
    func start() async throws
    func frames(_ delivery: FrameDelivery) -> AsyncThrowingStream<CapturedFrame, Error>
    func discardFrames()
    func stop()
}

enum CaptureError: LocalizedError {
    case permission, unavailable, frame

    var errorDescription: String? {
        switch self {
        case .permission: "Allow camera access in Settings to search for signs."
        case .unavailable: "The camera is unavailable."
        case .frame: "The camera could not deliver a frame."
        }
    }
}

/// Capture, configuration and stream changes run on the same serial queue.
/// Without a consumer the camera stays active but no image is converted or retained.
final class CameraCapture: NSObject, FrameCapturing, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "See2Hear.CameraCapture")
    private let context = CIContext()
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var consumer: AsyncThrowingStream<CapturedFrame, Error>.Continuation?
    private var remaining: Int?

    func start() async throws {
        guard let deviceID = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)?.uniqueID
        else { throw CaptureError.unavailable }
        guard await AVCaptureDevice.requestAccess(for: .video) else { throw CaptureError.permission }
        try Task.checkCancellation()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    if self.session.inputs.isEmpty { try self.configure(deviceID: deviceID) }
                    self.session.startRunning()
                    guard self.session.isRunning else { throw CaptureError.unavailable }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func configure(deviceID: String) throws {
        guard let device = AVCaptureDevice(uniqueID: deviceID) else { throw CaptureError.unavailable }
        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
        guard session.canAddInput(input) else { throw CaptureError.unavailable }
        session.addInput(input)
        guard session.canAddOutput(output) else {
            session.removeInput(input)
            throw CaptureError.unavailable
        }
        session.addOutput(output)
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 0
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
        rotation = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
        output.setSampleBufferDelegate(self, queue: queue)
    }

    func frames(_ delivery: FrameDelivery) -> AsyncThrowingStream<CapturedFrame, Error> {
        let batch = delivery == .fiveConsecutive
        return AsyncThrowingStream(bufferingPolicy: batch ? .bufferingOldest(SignBatch.size) : .bufferingNewest(1)) { sink in
            queue.async {
                self.consumer?.finish()
                guard self.session.isRunning else {
                    sink.finish(throwing: CaptureError.unavailable)
                    return
                }
                self.consumer = sink
                self.remaining = batch ? SignBatch.size : nil
            }
        }
    }

    func discardFrames() {
        queue.async { self.finishStream() }
    }

    func stop() {
        queue.async {
            self.finishStream()
            self.session.stopRunning()
        }
    }

    private func finishStream(throwing error: Error? = nil) {
        consumer?.finish(throwing: error)
        consumer = nil
        remaining = nil
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let consumer else { return }
        let timestamp = sampleBuffer.presentationTimeStamp.seconds
        guard timestamp.isFinite, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            finishStream(throwing: CaptureError.frame)
            return
        }
        let quarterTurns = Int(((rotation?.videoRotationAngleForHorizonLevelCapture ?? 0) / 90).rounded())
        let orientations: [CGImagePropertyOrientation] = [.up, .right, .down, .left]
        let image = CIImage(cvPixelBuffer: buffer).oriented(orientations[(quarterTurns % 4 + 4) % 4])
        guard let rgb = context.createCGImage(image, from: image.extent) else {
            finishStream(throwing: CaptureError.frame)
            return
        }
        consumer.yield(CapturedFrame(frame: CameraFrame(image: rgb), timestamp: timestamp))
        if let remaining {
            self.remaining = remaining - 1
            if remaining == 1 { finishStream() }
        }
    }
}
