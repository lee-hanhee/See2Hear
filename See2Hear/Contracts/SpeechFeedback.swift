enum RecognitionRetry: Int {
    case first = 1
    case second = 2
}

enum SearchFeedbackEvent: Equatable {
    /// Confirms the requested destination before searching.
    case searchStarted

    /// The scan continues without a candidate in view.
    case stillSearching

    /// The previously detected candidate needs to be reacquired.
    case lostSightOfSign

    /// The observed number differs from the gate requested in the query.
    /// NOTE: This applies to gates only
    case wrongGate(observedGateID: Int)

    /// The candidate is in view and another read will follow.
    case retryingRecognition(retryNumber: RecognitionRetry)

    /// The final read failed and this search is ending.
    case recognitionFailed

    /// Announces the recognized target with any available distance and direction.
    case recognized(arrowDirection: ArrowDirection?, distanceToSign: SignDistanceBand?)
}

/// Orchestration sends this value when it wants the language interface to speak.
struct SpeechFeedback {
    let query: SignQuery
    let event: SearchFeedbackEvent
}
