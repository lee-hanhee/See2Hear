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

    @Test("Matches generic gate query without a number")
    func genericGateQuery() {
        guard case .matched(let query) = parseQuery("I have 2 minutes to find a gate") else {
            Issue.record("Expected .matched generic gate")
            return
        }
        #expect(query.label == .gate)
        #expect(query.requestedGateID == nil)
    }

    @Test("Matches numbered gate queries 1 through 6", arguments: [
        ("where is gate 1", 1),
        ("gate to", 2),
        ("take me to gate number three", 3),
        ("where is gate 4", 4),
        ("gate for", 4),
        ("gate five please", 5),
        ("looking for gate six", 6)
    ])
    func numberedGateQueries(transcript: String, expectedGate: Int) {
        guard case .matched(let query) = parseQuery(transcript) else {
            Issue.record("Expected .matched for '\(transcript)'")
            return
        }
        #expect(query.label == .gate)
        #expect(query.requestedGateID == expectedGate)
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
        #expect(parseLLMReply("gate")?.label == .gate)
        #expect(parseLLMReply("gate")?.requestedGateID == nil)

        let gate4 = parseLLMReply("Gate 4.")
        #expect(gate4?.label == .gate)
        #expect(gate4?.requestedGateID == 4)
    }

    @Test("Rejects 'none' or verbose LLM replies without false positives", arguments: [
        "none",
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

    @Test("Formats visible and non-visible SpeechFeedback sentences")
    func feedbackFormatting() throws {
        let gate4 = try #require(SignQuery(queryID: UUID(), label: .gate, requestedGateID: 4))
        let washroom = try #require(SignQuery(queryID: UUID(), label: .washroom))
        let restaurant = try #require(SignQuery(queryID: UUID(), label: .restaurant))

        let visibleGate = SpeechFeedback(
            query: gate4,
            outcome: .targetVisible,
            arrowDirection: .left,
            distanceToSign: .medium
        )
        #expect(formatFeedback(visibleGate) == "Gate 4 sign is 5 to 15 meters away. Follow the arrow to the left.")

        let visibleWashroom = SpeechFeedback(
            query: washroom,
            outcome: .targetVisible,
            arrowDirection: .forward,
            distanceToSign: .near
        )
        #expect(formatFeedback(visibleWashroom) == "Washroom sign is less than 5 meters away. Head straight ahead.")

        let visibleNoMetadata = SpeechFeedback(
            query: restaurant,
            outcome: .targetVisible,
            arrowDirection: nil,
            distanceToSign: nil
        )
        #expect(formatFeedback(visibleNoMetadata) == "Restaurant sign is visible.")

        let notVisible = SpeechFeedback(
            query: gate4,
            outcome: .targetNotVisible,
            arrowDirection: .right,
            distanceToSign: .far
        )
        #expect(formatFeedback(notVisible) == "Gate 4 sign is not visible.")
    }
}
