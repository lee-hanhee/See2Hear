import Foundation
import Testing
@testable import See2Hear

struct LanguageTests {

    // MARK: - Direct Keyword Parsing (parseQuery)

    @Test("Matches washroom synonyms", arguments: [
        "where is the washroom",
        "I need a restroom please",
        "find the men's bathroom",
        "nearest toilet"
    ])
    func washroomQueries(transcript: String) {
        guard case .matched(let query) = parseQuery(transcript) else {
            Issue.record("Expected .matched for '\(transcript)'")
            return
        }
        #expect(query.label == .washroom)
        #expect(query.requestedGateID == nil)
    }

    @Test("Matches restaurant synonyms", arguments: [
        "find a restaurant",
        "where can I get coffee or food",
        "places to eat lunch"
    ])
    func restaurantQueries(transcript: String) {
        guard case .matched(let query) = parseQuery(transcript) else {
            Issue.record("Expected .matched for '\(transcript)'")
            return
        }
        #expect(query.label == .restaurant)
        #expect(query.requestedGateID == nil)
    }

    @Test("Recognizes a gate request that still needs a number")
    func genericGateQuery() {
        guard case .missingGateNumber = parseQuery("I have 2 minutes to find a gate") else {
            Issue.record("Expected a missing gate number")
            return
        }
    }

    @Test("Matches numbered gate queries 1 through 6", arguments: [
        ("where is gate 1", 1),
        ("gate to", 2),
        ("take me to gate number three", 3),
        ("where is gate 4", 4),
        ("gate for", 4),
        ("gate five please", 5),
        ("looking for gate six", 6),
        ("gate 6", 6)
    ])
    func numberedGateQueries(transcript: String, expectedGate: Int) {
        guard case .matched(let query) = parseQuery(transcript) else {
            Issue.record("Expected .matched for '\(transcript)'")
            return
        }
        #expect(query.label == .gate)
        #expect(query.requestedGateID == expectedGate)
    }

    @Test("Handles capitalization and everyday punctuation in queries")
    func queryCapitalizationAndPunctuation() {
        if case .matched(let washroom) = parseQuery("Where is the Washroom?") {
            #expect(washroom.label == .washroom)
            #expect(washroom.requestedGateID == nil)
        } else {
            Issue.record("Expected .matched washroom query")
        }

        if case .matched(let restaurant) = parseQuery("Find a Restaurant, please!") {
            #expect(restaurant.label == .restaurant)
            #expect(restaurant.requestedGateID == nil)
        } else {
            Issue.record("Expected .matched restaurant query")
        }

        if case .matched(let gate) = parseQuery("GATE 3.") {
            #expect(gate.label == .gate)
            #expect(gate.requestedGateID == 3)
        } else {
            Issue.record("Expected .matched gate query")
        }
    }

    @Test("Requires a number for gates and preserves the query ID")
    func signQueryConstruction() throws {
        #expect(SignQuery(queryID: UUID(), label: .gate) == nil)
        #expect(SignQuery(queryID: UUID(), label: .washroom, requestedGateID: 2) == nil)

        let numberedID = UUID()
        let numberedGate = try #require(SignQuery(queryID: numberedID, label: .gate, requestedGateID: 6))
        #expect(numberedGate.queryID == numberedID)
        #expect(numberedGate.label == .gate)
        #expect(numberedGate.requestedGateID == 6)
    }

    @Test("Rejects out-of-range gate numbers as .unsupported", arguments: [
        "gate zero",
        "where is gate 7",
        "gate number 9"
    ])
    func unsupportedGateNumbers(transcript: String) {
        if case .unsupported = parseQuery(transcript) {
            // Expected
        } else {
            Issue.record("Expected .unsupported for '\(transcript)'")
        }
    }

    @Test("Routes indirect or multi-category queries to .unmatched for LLM fallback", arguments: [
        "I am starving",
        "I need to freshen up",
        "my flight is boarding soon",
        "where is the washroom near gate 2",
        ""
    ])
    func unmatchedQueries(transcript: String) {
        if case .unmatched = parseQuery(transcript) {
            // Expected
        } else {
            Issue.record("Expected .unmatched for '\(transcript)'")
        }
    }

    // MARK: - Strict LLM Output Parsing (parseLLMReply)

    @Test("Parses valid constrained LLM labels")
    func validLLMReplies() {
        #expect(parseLLMReply("washroom")?.label == .washroom)
        #expect(parseLLMReply("restaurant")?.label == .restaurant)

        let gate1 = parseLLMReply("gate 1")
        #expect(gate1?.label == .gate)
        #expect(gate1?.requestedGateID == 1)

        let gate4 = parseLLMReply("Gate 4.")
        #expect(gate4?.label == .gate)
        #expect(gate4?.requestedGateID == 4)

        let gate6 = parseLLMReply("gate 6")
        #expect(gate6?.label == .gate)
        #expect(gate6?.requestedGateID == 6)
    }

    @Test("Rejects 'none' or verbose LLM replies without false positives", arguments: [
        "none",
        "gate",
        "gate 9",
        "The label is none, not washroom"
    ])
    func rejectedLLMReplies(reply: String) {
        #expect(parseLLMReply(reply) == nil)
    }

    // MARK: - Adaptive Silence Timing (silenceDelay)

    @Test("Uses 0.8s silence delay for complete queries and 2.0s for partial/unmatched")
    func adaptiveSilenceTiming() {
        #expect(silenceDelay(for: "where is the washroom") == 0.8)
        #expect(silenceDelay(for: "find a restaurant") == 0.8)
        #expect(silenceDelay(for: "gate 3") == 0.8)
        #expect(silenceDelay(for: "gate 9") == 0.8)
        #expect(silenceDelay(for: "where is gate") == 2.0)
        #expect(silenceDelay(for: "I am starving") == 2.0)
    }

    // MARK: - Feedback Formatting (formatFeedback)

    @Test("Uses natural destination wording for search announcements", arguments: [
        (SignLabel.gate, Optional(5), "Gate 5"),
        (SignLabel.washroom, nil, "the washroom"),
        (SignLabel.restaurant, nil, "a restaurant")
    ])
    func searchAnnouncements(label: SignLabel, gateID: Int?, destination: String) throws {
        let query = try #require(SignQuery(queryID: UUID(), label: label, requestedGateID: gateID))
        #expect(formatFeedback(SpeechFeedback(query: query, event: .searchStarted))
            == "Confirmed. Let’s look for \(destination).")
        #expect(formatFeedback(SpeechFeedback(query: query, event: .stillSearching))
            == "Still looking for \(destination). Continue to move your phone slowly.")
    }

    @Test("Announces lost signs, wrong gates, retries, and recognition failure", arguments: [
        (SearchFeedbackEvent.lostSightOfSign, "I lost sight of the sign. Pan back slowly."),
        (.wrongGate(observedGateID: 3), "This sign says Gate 3. Keep panning slowly to look for Gate 5."),
        (.retryingRecognition(retryNumber: .first),
         "I couldn’t read the sign. Hold your phone still while I try again."),
        (.retryingRecognition(retryNumber: .second),
         "I still couldn’t read the sign. Hold your phone still while I try once more."),
        (.recognitionFailed, "I couldn’t read the sign. Please tell me what you’d like me to look for.")
    ])
    func searchEventAnnouncements(event: SearchFeedbackEvent, expected: String) throws {
        let query = try #require(SignQuery(queryID: UUID(), label: .gate, requestedGateID: 5))
        #expect(formatFeedback(SpeechFeedback(query: query, event: event)) == expected)
    }

    @Test("Formats recognized signs with available distance and direction")
    func feedbackFormatting() throws {
        let gate4 = try #require(SignQuery(queryID: UUID(), label: .gate, requestedGateID: 4))
        let washroom = try #require(SignQuery(queryID: UUID(), label: .washroom))
        let restaurant = try #require(SignQuery(queryID: UUID(), label: .restaurant))

        let visibleGate = SpeechFeedback(
            query: gate4,
            event: .recognized(arrowDirection: .left, distanceToSign: .medium)
        )
        #expect(formatFeedback(visibleGate) == "Gate 4 sign is 5 to 15 meters away. Follow the arrow to the left.")

        let visibleWashroom = SpeechFeedback(
            query: washroom,
            event: .recognized(arrowDirection: .forward, distanceToSign: .near)
        )
        #expect(formatFeedback(visibleWashroom) == "Washroom sign is less than 5 meters away. Head straight ahead.")

        let visibleNoMetadata = SpeechFeedback(
            query: restaurant,
            event: .recognized(arrowDirection: nil, distanceToSign: nil)
        )
        #expect(formatFeedback(visibleNoMetadata) == "Restaurant sign is visible.")

        let visibleFarRight = SpeechFeedback(
            query: gate4,
            event: .recognized(arrowDirection: .right, distanceToSign: .far)
        )
        #expect(formatFeedback(visibleFarRight) == "Gate 4 sign is more than 15 meters away. Follow the arrow to the right.")

        let visibleFarNoArrow = SpeechFeedback(
            query: washroom,
            event: .recognized(arrowDirection: nil, distanceToSign: .far)
        )
        #expect(formatFeedback(visibleFarNoArrow) == "Washroom sign is more than 15 meters away.")

        let visibleRightNoDistance = SpeechFeedback(
            query: restaurant,
            event: .recognized(arrowDirection: .right, distanceToSign: nil)
        )
        #expect(formatFeedback(visibleRightNoDistance) == "Restaurant sign is visible. Follow the arrow to the right.")
    }
}
