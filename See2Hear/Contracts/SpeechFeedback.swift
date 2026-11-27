enum RecognitionRetry: Int {
    case first = 1
    case second = 2
}

enum SearchFeedbackEvent: Equatable {
    /// “Confirmed. Start looking for {Gate 5, washroom, restaurant}.”
    case searchStarted

    /// The scan continues without a candidate in view.
    /// “Still looking for Gate 5. Continue to move your phone slowly.”
    case stillSearching

    /// The previously detected candidate needs to be reacquired.
    /// “I lost sight of the gate sign. Pan back slowly.”
    case lostSightOfSign

    /// The observed number differs from the gate requested in the query.
    /// NOTE: This applies to gates only
    /// “This sign says Gate 3. Keep panning slowly for Gate 5.”
    case wrongGate(observedGateID: Int)

    /// The candidate is in view and another read will follow.
    /// “Hold your phone still while I try to read the sign again.”
    case retryingRecognition(retryNumber: RecognitionRetry)

    /// The final read failed and this search is ending.
    /// “I couldn’t read the sign. Please start a new conversation round.”
    case recognitionFailed

    /// “Washroom sign found. The arrow points left.”
    case recognized(arrowDirection: ArrowDirection?, distanceToSign: SignDistanceBand?)
}

/// Orchestration sends this value when it wants the language interface to speak.
struct SpeechFeedback {
    let query: SignQuery
    let event: SearchFeedbackEvent
}
