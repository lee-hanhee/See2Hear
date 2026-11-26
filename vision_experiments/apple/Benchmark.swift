import Foundation
import ImageIO
#if os(iOS)
@testable import See2Hear
#endif

private struct Dataset: Decodable {
    struct Frame: Decodable {
        let image: String
        let labels: [String: Label]
    }

    struct Label: Decodable {
        let objectPresent: Bool
        let bbox: [Int]?
        let distance: String?
        let gateNumber: String?
        let arrowDirection: String?

        var expectedReading: SignReading? {
            if let number = gateNumber.flatMap(Int.init) { return .gate(number) }
            let name = arrowDirection == "front" ? "forward" : arrowDirection
            return name.flatMap(ArrowDirection.init(rawValue:)).map(SignReading.arrow)
        }
    }

    let frames: [String: Frame]
}

private struct Timings {
    var samples: [Double] = []

    mutating func record<T>(_ operation: () async throws -> T) async rethrows -> T {
        let start = DispatchTime.now().uptimeNanoseconds
        let result = try await operation()
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        return result
    }

    var summary: String {
        let warm = samples.dropFirst().sorted()
        guard !warm.isEmpty else { return "no warm samples" }
        func percentile(_ p: Double) -> Double {
            let index = Double(warm.count - 1) * p
            let low = Int(index), high = Int(ceil(index))
            return warm[low] + (warm[high] - warm[low]) * (index - Double(low))
        }
        return String(format: "n=%d p50=%.2f ms p95=%.2f ms", warm.count, percentile(0.5), percentile(0.95))
    }
}

struct VisionBenchmarkScores {
    fileprivate var detector = Timings(), reader = Timings(), distance = Timings()
    var total = 0, positives = 0, localized = 0, falsePositives = 0, joint = 0
    var correctSign = 0, correctDistance = 0

    fileprivate mutating func evaluate(_ label: Dataset.Label, category: SignLabel, frame: CameraFrame,
                                       models: CoreMLSignRecognizer) async throws {
        total += 1
        let box = try await detector.record { try await models.detect(frame: frame, category: category).first }
        guard label.objectPresent else {
            if box != nil { falsePositives += 1 }
            return
        }
        guard let xyxy = label.bbox, xyxy.count == 4, let distanceLabel = label.distance,
              let expectedDistance = SignDistanceBand(rawValue: distanceLabel == "mid" ? "medium" : distanceLabel)
        else { throw BenchmarkError.badLabel }
        let truth = SignBoundingBox(x: xyxy[0], y: xyxy[1], width: xyxy[2] - xyxy[0], height: xyxy[3] - xyxy[1],
                                    frameWidth: frame.widthPixels, frameHeight: frame.heightPixels)
        positives += 1
        // Time independent readers on annotated boxes, including signs missed by the detector.
        let sign = try await reader.record { try await models.readSign(frame: frame, category: category, box: truth) }
        let band = try await distance.record { try await models.estimateDistance(frame: frame, category: category, box: truth) }
        if sign == label.expectedReading { correctSign += 1 }
        if band == expectedDistance { correctDistance += 1 }
        guard let box else { return }
        let a = CGRect(x: box.x, y: box.y, width: box.width, height: box.height)
        let b = CGRect(x: truth.x, y: truth.y, width: truth.width, height: truth.height)
        let shared = a.intersection(b)
        let area = shared.width * shared.height
        guard area / (a.width * a.height + b.width * b.height - area) >= 0.5 else {
            falsePositives += 1
            return
        }
        localized += 1
        let predictedSign = try await models.readSign(frame: frame, category: category, box: box)
        let predictedDistance = try await models.estimateDistance(frame: frame, category: category, box: box)
        if predictedSign == label.expectedReading && predictedDistance == expectedDistance { joint += 1 }
    }
}

private enum BenchmarkError: Error { case invalidArguments, badImage(String), badLabel }

#if os(macOS)
@main
#endif
enum VisionBenchmark {
    #if os(macOS)
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else { throw BenchmarkError.invalidArguments }
        _ = try await run(models: CoreMLSignRecognizer(modelDirectory: URL(fileURLWithPath: CommandLine.arguments[1])),
                          data: URL(fileURLWithPath: CommandLine.arguments[2]))
    }
    #endif

    static func run(models: CoreMLSignRecognizer, data: URL) async throws -> [SignLabel: VisionBenchmarkScores] {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let dataset = try decoder.decode(Dataset.self, from: Data(contentsOf: data.appendingPathComponent("labels.json")))
        let categories: [SignLabel] = [.gate, .restaurant, .washroom]
        var scores: [SignLabel: VisionBenchmarkScores] = [:]
        for entry in dataset.frames.values.sorted(by: { $0.image < $1.image }) {
            let url = data.appendingPathComponent(entry.image)
            let options = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, options) else { throw BenchmarkError.badImage(entry.image) }
            for category in categories {
                guard let label = entry.labels[category.rawValue] else { continue }
                try await scores[category, default: VisionBenchmarkScores()].evaluate(label, category: category, frame: CameraFrame(image: image), models: models)
            }
        }
        print("\(dataset.frames.count) frames. Reader accuracy and timings use annotated boxes. Joint accuracy uses detected boxes.")
        for category in categories {
            guard let s = scores[category] else { throw BenchmarkError.badLabel }
            print("\(category.rawValue): records=\(s.total) positives=\(s.positives) localized=\(s.localized) falsePositives=\(s.falsePositives) sign=\(s.correctSign) distance=\(s.correctDistance) joint=\(s.joint)")
            print("  detector \(s.detector.summary)\n  sign     \(s.reader.summary)\n  distance \(s.distance.summary)")
        }
        return scores
    }
}
