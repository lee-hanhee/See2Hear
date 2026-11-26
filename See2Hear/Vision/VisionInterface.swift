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
