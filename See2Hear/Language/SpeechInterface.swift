protocol SpeechInterfacing {
    /// Listens for one spoken request. Return nil for an unsupported request.
    /// Throw if speech capture or recognition fails. Release the microphone
    /// when the call ends, including on cancellation.
    func listenForQuery() async throws -> SignQuery?

    /// Speaks the supplied feedback. Return once playback finishes. 
    /// Throw on failure and stop playback on cancellation.
    func speak(_ feedback: SpeechFeedback) async throws
}
