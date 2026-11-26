protocol SignRecognizing {
    /// Fast track. Find a sign of the requested category and return a guidance box.
    /// A gate query locates a screen regardless of the requested number.
    /// Carry the query ID and leave all sign metadata nil.
    func locate(frame: CameraFrame, for query: SignQuery) async throws -> VisionResult

    /// Slow track. Read this specific candidate in the supplied frame after centering.
    /// For a numbered gate, publish a match only when the visible number agrees.
    /// A mismatch, unreadable requested number, or miss returns a nil box and metadata.
    func readDetails(frame: CameraFrame, for query: SignQuery,
                     candidate: SignBoundingBox) async throws -> VisionResult
}

extension SignRecognizing {
    /// Single-frame convenience for callers explicitly requesting a detailed result.
    func analyze(frame: CameraFrame, for query: SignQuery) async throws -> VisionResult {
        let guidance = try await locate(frame: frame, for: query)
        guard let box = guidance.boundingBox else { return guidance }
        return try await readDetails(frame: frame, for: query, candidate: box)
    }
}

/// Independent per-frame model calls. Boxes use the upright source frame's pixels.
protocol VisionModelInvoking: Sendable {
    /// Candidates sorted by the reference detector's confidence, size and center ranking.
    func detect(frame: CameraFrame, category: SignLabel) async throws -> [SignBoundingBox]
    func readSign(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) async throws -> SignReading?
    func estimateDistance(frame: CameraFrame, category: SignLabel, box: SignBoundingBox) async throws -> SignDistanceBand
}

enum SignReading: Equatable {
    case gate(Int)
    case arrow(ArrowDirection)
}
