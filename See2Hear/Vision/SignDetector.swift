import CoreGraphics
import CoreML
import CoreVideo

/// Maps the detector's square canvas to and from upright, top-left source coordinates.
struct DetectorCanvas {
    static let size = 640
    let frameWidth: Int
    let frameHeight: Int
    let width: Int
    let height: Int
    var left: Int { (Self.size - width) / 2 }
    var top: Int { (Self.size - height) / 2 }

    init(frame: CameraFrame) {
        frameWidth = frame.widthPixels
        frameHeight = frame.heightPixels
        let scale = Double(Self.size) / Double(max(frameWidth, frameHeight))
        width = max(1, Int((Double(frameWidth) * scale).rounded(.toNearestOrEven)))
        height = max(1, Int((Double(frameHeight) * scale).rounded(.toNearestOrEven)))
    }

    func input(_ image: CGImage) throws -> CVPixelBuffer {
        let pixels = try VisionPixels.resize(image, width: width, height: height)
        var output: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, Self.size, Self.size, kCVPixelFormatType_32BGRA,
                                         [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &output)
        guard status == kCVReturnSuccess, let output,
              CVPixelBufferLockBaseAddress(output, []) == kCVReturnSuccess else { throw VisionModelError.invalidImage }
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard let base = CVPixelBufferGetBaseAddress(output) else { throw VisionModelError.invalidImage }
        let rowBytes = CVPixelBufferGetBytesPerRow(output)
        var padding = VisionPixels.Pixel(114, 114, 114, 255)
        memset_pattern4(base, &padding, rowBytes * Self.size)
        pixels.withUnsafeBytes { bytes in
            guard let source = bytes.baseAddress else { return }
            for row in 0..<height {
                memcpy(base + (row + top) * rowBytes + left * 4, source + row * width * 4, width * 4)
            }
        }
        return output
    }

    private struct Candidate {
        let rect: CGRect
        let score: Double
        var area: Double { rect.width * rect.height }
    }

    func boxes(_ output: MLMultiArray) throws -> [SignBoundingBox] {
        guard output.shape == [1, 5, 8400], output.dataType == .float32
        else { throw VisionModelError.unexpectedOutput("detections") }
        let candidates = output.withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: Float.self)
            let plane = output.strides[1].intValue, step = output.strides[2].intValue
            var candidates: [Candidate] = []
            for i in 0..<8400 {
                let start = i * step
                let score = Double(values[4 * plane + start])
                guard score >= 0.25 else { continue }
                let cx = Double(values[start]), cy = Double(values[start + plane])
                let halfWidth = Double(values[start + 2 * plane]) / 2, halfHeight = Double(values[start + 3 * plane]) / 2
                func pixel(_ value: Double, _ offset: Int, _ scaled: Int, _ original: Int) -> Double {
                    min(Double(original), max(0, ((value - Double(offset)) * Double(original) / Double(scaled)).rounded()))
                }
                let x = pixel(cx - halfWidth, left, width, frameWidth), y = pixel(cy - halfHeight, top, height, frameHeight)
                let right = pixel(cx + halfWidth, left, width, frameWidth), bottom = pixel(cy + halfHeight, top, height, frameHeight)
                if right > x && bottom > y {
                    candidates.append(Candidate(rect: CGRect(x: x, y: y, width: right - x, height: bottom - y), score: score))
                }
            }
            return candidates.sorted { $0.score > $1.score }
        }
        var kept: [Candidate] = []
        for candidate in candidates {
            let duplicate = kept.contains { previous in
                let shared = previous.rect.intersection(candidate.rect)
                let intersection = shared.width * shared.height
                return intersection > 0.45 * (previous.area + candidate.area - intersection)
            }
            if !duplicate { kept.append(candidate) }
        }
        guard let largest = kept.map(\.area).max() else { return [] }
        func rank(_ candidate: Candidate) -> Double {
            let dx = candidate.rect.midX / Double(frameWidth) - 0.5
            let dy = candidate.rect.midY / Double(frameHeight) - 0.5
            let centrality = 1 - min(1, hypot(dx, dy) * sqrt(2))
            return candidate.score * (1 + 0.30 * sqrt(candidate.area / largest)) * (1 + 0.15 * centrality)
        }
        return kept.sorted { rank($0) > rank($1) }.map {
            SignBoundingBox(x: Int($0.rect.minX), y: Int($0.rect.minY), width: Int($0.rect.width), height: Int($0.rect.height),
                            frameWidth: frameWidth, frameHeight: frameHeight)
        }
    }
}
