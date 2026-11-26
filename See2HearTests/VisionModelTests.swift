import Foundation
import ImageIO
import Testing
@testable import See2Hear

struct VisionModelTests {
    private final class TestResources {}

    @Test("The bundled models guide to a gate and only confirm the requested number")
    func numberedGate() async throws {
        let bundle = Bundle(for: TestResources.self)
        let url = try #require(bundle.url(forResource: "frame_000591", withExtension: "png", subdirectory: "frames"))
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let frame = CameraFrame(image: image)
        let models = CoreMLSignRecognizer()
        let gate5 = try #require(SignQuery(queryID: UUID(), label: .gate, requestedGateID: 5))
        let guidance = try await models.locate(frame: frame, for: gate5)
        let candidate = try #require(guidance.boundingBox)
        #expect(guidance.observedGateID == nil)
        #expect(guidance.distanceToSign == nil)
        let verified = try await models.readDetails(frame: frame, for: gate5, candidate: candidate)
        #expect(verified.observedGateID == "5")
        #expect(verified.distanceToSign == .far)
        let gate3 = try #require(SignQuery(queryID: UUID(), label: .gate, requestedGateID: 3))
        let rejected = try await models.readDetails(frame: frame, for: gate3, candidate: candidate)
        #expect(rejected.queryID == gate3.queryID)
        #expect(rejected.boundingBox == nil)
        #expect(rejected.distanceToSign == nil)
    }

    @Test("Measure the nine Core ML models on held-out airport frames")
    func heldOutBenchmark() async throws {
        let bundle = Bundle(for: TestResources.self)
        let labels = try #require(bundle.url(forResource: "labels", withExtension: "json"))
        let scores = try await VisionBenchmark.run(models: CoreMLSignRecognizer(), data: labels.deletingLastPathComponent())
        #expect(scores.values.reduce(0) { $0 + $1.total } == 284)
        #expect(scores.values.reduce(0) { $0 + $1.positives } == 106)
        // The ONNX benchmark provides the reference counts. On frame 1822 the
        // simulator abstains just below 0.55 on the annotated washroom crop;
        // native M5 Core ML and ONNX report left. See the Apple benchmark notes.
        for (category, boxes, signs, bands, joint, falsePositives) in [
            (SignLabel.gate, 15, 13, 16, 11, 2),
            (.restaurant, 24, 22, 18, 13, 1),
            (.washroom, 56, 56, 51, 38, 3),
        ] {
            let result = try #require(scores[category])
            #expect(result.localized >= boxes)
            #expect(result.correctSign >= signs)
            #expect(result.correctDistance >= bands)
            #expect(result.joint >= joint)
            #expect(result.falsePositives <= falsePositives)
        }
    }
}
