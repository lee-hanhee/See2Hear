import CoreML
import Foundation

/// Lazily loads the compiled models. The actor serializes access to each cached MLModel.
actor CoreMLSignRecognizer: VisionModelInvoking, SignRecognizing {
    private let directory: URL
    private var models: [String: MLModel] = [:]

    init(modelDirectory: URL = Bundle.main.resourceURL ?? Bundle.main.bundleURL) {
        directory = modelDirectory
    }

    private func predict(_ name: String, inputs: [String: MLFeatureValue], output: String) throws -> MLMultiArray {
        try Task.checkCancellation()
        let model: MLModel
        if let cached = models[name] {
            model = cached
        } else {
            let url = directory.appendingPathComponent(name + ".mlmodelc", isDirectory: true)
            guard FileManager.default.fileExists(atPath: url.path) else { throw VisionModelError.missingModel(name) }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: url, configuration: config)
            models[name] = model
        }
        let features = try MLDictionaryFeatureProvider(dictionary: inputs)
        let prediction = try model.prediction(from: features)
        guard let values = prediction.featureValue(for: output)?.multiArrayValue
        else { throw VisionModelError.unexpectedOutput(name) }
        return values
    }

    func detect(frame: CameraFrame, category: SignLabel) throws -> [SignBoundingBox] {
        let canvas = DetectorCanvas(frame: frame)
        let image = try canvas.input(frame.image)
        let output = try predict(category.rawValue, inputs: ["image": MLFeatureValue(pixelBuffer: image)], output: "detections")
        return try canvas.boxes(output)
    }

    func readSign(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) throws -> SignReading? {
        var rect = try VisionPixels.crop(frame.image, box: box)
        if category == .gate { rect.size.height = ceil(rect.height * 0.67) }
        guard let crop = frame.image.cropping(to: rect) else { throw VisionModelError.invalidBox }
        let input = try VisionPixels.tensor(crop, width: 192, height: 96)
        let name = category.rawValue + "_sign"
        let logits = try predict(name, inputs: ["crop": MLFeatureValue(multiArray: input)], output: "logits")
        guard logits.count == (category == .gate ? 7 : 4) else { throw VisionModelError.unexpectedOutput(name) }
        let scores = (0..<logits.count).map { logits[$0].doubleValue }
        guard let best = scores.indices.max(by: { scores[$0] < scores[$1] }), best != 0 else { return nil }
        let confidence = 1 / scores.reduce(0) { $0 + exp($1 - scores[best]) }
        switch category {
        case .gate:
            return confidence >= 0.70 ? .gate(best) : nil
        case .restaurant, .washroom:
            let arrows: [ArrowDirection] = [.left, .right, .forward]
            return confidence >= 0.55 ? .arrow(arrows[best - 1]) : nil
        }
    }

    func estimateDistance(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) throws -> SignDistanceBand {
        let rect = try VisionPixels.crop(frame.image, box: box)
        guard let crop = frame.image.cropping(to: rect) else { throw VisionModelError.invalidBox }
        let inputs = try [
            "scene": MLFeatureValue(multiArray: VisionPixels.tensor(frame.image, width: 48, height: 32)),
            "detail": MLFeatureValue(multiArray: VisionPixels.tensor(crop, width: 48, height: 24)),
            "geometry": MLFeatureValue(multiArray: VisionPixels.geometry(rect, in: frame.image)),
        ]
        let name = category.rawValue + "_distance"
        let logits = try predict(name, inputs: inputs, output: "logits")
        let bands: [SignDistanceBand] = [.near, .medium, .far]
        guard logits.count == bands.count, let best = bands.indices.max(by: { logits[$0].floatValue < logits[$1].floatValue })
        else { throw VisionModelError.unexpectedOutput(name) }
        return bands[best]
    }

    func locate(frame: CameraFrame, for query: SignQuery) throws -> VisionResult {
        let box = try detect(frame: frame, category: query.label).first
        return VisionResult(queryID: query.queryID, boundingBox: box, observedGateID: nil, arrowDirection: nil, distanceToSign: nil)
    }

    func readDetails(frame: CameraFrame, for query: SignQuery, candidate: SignBoundingBox) throws -> VisionResult {
        let reading = try readSign(frame: frame, category: query.label, box: candidate)
        if let requested = query.requestedGateID, reading != .gate(requested) {
            return VisionResult(queryID: query.queryID, boundingBox: nil, observedGateID: nil, arrowDirection: nil, distanceToSign: nil)
        }
        var number: String?
        var arrow: ArrowDirection?
        switch reading {
        case .gate(let id): number = String(id)
        case .arrow(let direction): arrow = direction
        case nil: break
        }
        let distance = try estimateDistance(frame: frame, category: query.label, box: candidate)
        return VisionResult(queryID: query.queryID, boundingBox: candidate, observedGateID: number, arrowDirection: arrow, distanceToSign: distance)
    }
}
