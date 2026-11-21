enum SearchOutcome {
    case targetVisible
    case targetNotVisible
}

/// Orchestration sends this value when it wants the language interface to speak.
struct SpeechFeedback {
    let query: SignQuery
    let outcome: SearchOutcome
    let arrowDirection: ArrowDirection?
    let distanceToSign: SignDistanceBand?
}
