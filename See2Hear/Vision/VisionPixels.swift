import CoreGraphics
import CoreML

enum VisionModelError: Error {
    case missingModel(String)
    case invalidImage
    case invalidBox
    case unexpectedOutput(String)
}

/// RGB image preparation shared by all nine models. Pixel storage is BGRA for Core ML.
enum VisionPixels {
    typealias Pixel = SIMD4<UInt8>

    // Pillow's bilinear resize widens its triangle filter when downsampling and
    // rounds to 8 bits after each axis. This matters on small, distant signs.
    private struct Taps {
        let start: Int
        let weights: [Int32]

        init(source: Int, target: Int, index: Int) {
            let scale = Double(source) / Double(target)
            let radius = max(1, scale)
            let center = (Double(index) + 0.5) * scale
            start = max(0, Int(center - radius + 0.5))
            let end = min(source, Int(center + radius + 0.5))
            let taps = (start..<end).map { max(0, 1 - abs((Double($0) + 0.5 - center) / radius)) }
            let sum = taps.reduce(0, +)
            weights = taps.map { Int32(($0 / sum * Double(1 << 22)).rounded()) }
        }

        func sample(_ buffer: UnsafeBufferPointer<Pixel>, base: Int, stride: Int) -> Pixel {
            var result = SIMD4<Int32>(repeating: 1 << 21)
            var offset = base + start * stride
            for weight in weights {
                result &+= SIMD4<Int32>(truncatingIfNeeded: buffer[offset]) &* weight
                offset += stride
            }
            return Pixel(clamping: result &>> 22)
        }
    }

    static func resize(_ image: CGImage, width: Int, height: Int) throws -> [Pixel] {
        var source = [Pixel](repeating: .zero, count: image.width * image.height)
        try source.withUnsafeMutableBytes { bytes in
            let info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: bytes.baseAddress, width: image.width, height: image.height,
                                          bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                          space: space, bitmapInfo: info) else { throw VisionModelError.invalidImage }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        let horizontal = (0..<width).map { Taps(source: image.width, target: width, index: $0) }
        let intermediate = source.withUnsafeBufferPointer { input in
            var rows = [Pixel]()
            rows.reserveCapacity(width * image.height)
            for row in 0..<image.height {
                for filter in horizontal {
                    rows.append(filter.sample(input, base: row * image.width, stride: 1))
                }
            }
            return rows
        }
        return intermediate.withUnsafeBufferPointer { input in
            var result = [Pixel]()
            result.reserveCapacity(width * height)
            for y in 0..<height {
                let vertical = Taps(source: image.height, target: height, index: y)
                for x in 0..<width {
                    result.append(vertical.sample(input, base: x, stride: width))
                }
            }
            return result
        }
    }

    static func tensor(_ image: CGImage, width: Int, height: Int) throws -> MLMultiArray {
        let pixels = try resize(image, width: width, height: height)
        let result = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        result.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
            for y in 0..<height {
                for x in 0..<width {
                    let pixel = pixels[y * width + x]
                    let offset = y * strides[2] + x * strides[3]
                    buffer[offset] = Float(pixel.z) / 255
                    buffer[offset + strides[1]] = Float(pixel.y) / 255
                    buffer[offset + 2 * strides[1]] = Float(pixel.x) / 255
                }
            }
        }
        return result
    }

    static func crop(_ image: CGImage, box: SignBoundingBox) throws -> CGRect {
        let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        guard box.frameWidth == image.width, box.frameHeight == image.height, box.width > 0, box.height > 0
        else { throw VisionModelError.invalidBox }
        let rect = bounds.intersection(CGRect(x: box.x, y: box.y, width: box.width, height: box.height))
        guard !rect.isEmpty else { throw VisionModelError.invalidBox }
        return rect
    }

    static func geometry(_ rect: CGRect, in image: CGImage) throws -> MLMultiArray {
        let width = Double(image.width), height = Double(image.height), longest = max(width, height)
        let values: [Double] = [
            log(rect.width / longest), log(rect.height / longest), log(Double(rect.width / rect.height)),
            rect.midX / width, rect.midY / height, rect.width / width, rect.height / height,
            width / longest, height / longest, rect.minX <= 2 ? 1 : 0, rect.minY <= 2 ? 1 : 0,
            rect.maxX >= width - 2 ? 1 : 0, rect.maxY >= height - 2 ? 1 : 0,
        ]
        let result = try MLMultiArray(shape: [1, 13], dataType: .float32)
        for (index, value) in values.enumerated() { result[index] = NSNumber(value: Float(value)) }
        return result
    }
}
