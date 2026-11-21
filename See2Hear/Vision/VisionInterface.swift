protocol SignRecognizing {
    /// Analyzes one frame and returns a result carrying the supplied query ID.
    /// A numbered gate request matches only that gate. On a miss, return a result
    /// with a nil bounding box and nil metadata. Throw if analysis fails.
    /// On a match, report only metadata attributable to the requested sign.
    func analyze(frame: CameraFrame, for query: SignQuery) async throws -> VisionResult
}
