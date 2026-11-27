import Foundation
import Testing
@testable import See2Hear

struct LanguageTests {

    // MARK: - Query Contract

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

    // MARK: - Conversation Replies

    @Test("Decodes supported completed destinations", arguments: [
        (#"{"kind":"ready","label":"washroom"}"#, SignLabel.washroom, Optional<Int>.none),
        (#"{"kind":"ready","label":"restaurant"}"#, .restaurant, nil),
        (#"{"kind":"ready","label":"gate","gateID":1}"#, .gate, 1),
        (#"{"kind":"ready","label":"gate","gateID":6}"#, .gate, 6)
    ])
    func completedReplies(reply: String, label: SignLabel, gateID: Int?) throws {
        guard case .ready(let query) = try decodeConversationReply(reply) else {
            Issue.record("Expected a complete query")
            return
        }
        #expect(query.label == label)
        #expect(query.requestedGateID == gateID)
    }

    @Test("Decodes a clarification for playback")
    func clarificationReply() throws {
        let reply = #"{"kind":"clarify","message":" Which gate number? "}"#
        guard case .clarify(let message) = try decodeConversationReply(reply) else {
            Issue.record("Expected a clarification")
            return
        }
        #expect(message == "Which gate number?")
    }

    @Test("Invalid model output throws one response error", arguments: [
        "{", "{}",
        #"{"kind":"unknown"}"#,
        #"{"kind":"clarify"}"#,
        #"{"kind":"clarify","message":"   "}"#,
        #"{"kind":"ready","label":"luggage"}"#,
        #"{"kind":"ready","label":"gate"}"#,
        #"{"kind":"ready","label":"gate","gateID":0}"#,
        #"{"kind":"ready","label":"gate","gateID":7}"#,
        #"{"kind":"ready","label":"gate","gateID":"five"}"#,
        #"{"kind":"ready","label":"washroom","gateID":2}"#
    ])
    func invalidReplies(reply: String) {
        #expect(throws: ConversationError.self) { try decodeConversationReply(reply) }
    }

    @Test("Scripted conversations order capture, model replies, and playback", arguments: [
        (["find a washroom"], [#"{"kind":"ready","label":"washroom"}"#], SignLabel.washroom, Optional<Int>.none),
        (["I need help", "A gate", "Five"], [
            #"{"kind":"clarify","message":"Which destination?"}"#,
            #"{"kind":"clarify","message":"Which gate number?"}"#,
            #"{"kind":"ready","label":"gate","gateID":5}"#
        ], .gate, 5)
    ])
    @MainActor
    func scriptedConversation(utterances: [String], replies: [String], label: SignLabel, gateID: Int?) async throws {
        var captured = 0
        var responded = 0
        var spoken = 0
        let query = try await resolveQueryConversation(
            capture: {
                #expect(responded == captured)
                #expect(spoken == captured)
                let utterance = try #require(utterances.dropFirst(captured).first)
                captured += 1
                return utterance
            },
            respond: { utterance in
                #expect(utterance == utterances[responded])
                let reply = try #require(replies.dropFirst(responded).first)
                responded += 1
                return reply
            },
            speak: { message in
                guard case .clarify(let expected) = try decodeConversationReply(replies[responded - 1]) else {
                    Issue.record("Only clarifications should be spoken")
                    return
                }
                #expect(message == expected)
                spoken += 1
            }
        )
        #expect(captured == utterances.count)
        #expect(responded == replies.count)
        #expect(spoken == replies.count - 1)
        #expect(query.label == label)
        #expect(query.requestedGateID == gateID)
    }

    @Test("Extended conversations continue until a complete query arrives")
    func extendedConversation() async throws {
        var turns = 0
        var clarifications = 0
        let query = try await resolveQueryConversation(
            capture: { "I am still deciding" },
            respond: { _ in
                turns += 1
                return turns <= 100
                    ? #"{"kind":"clarify","message":"Which destination would you like?"}"#
                    : #"{"kind":"ready","label":"gate","gateID":5}"#
            },
            speak: { _ in clarifications += 1 }
        )
        #expect(turns == 101)
        #expect(clarifications == 100)
        #expect(query.label == .gate)
        #expect(query.requestedGateID == 5)
    }

    @Test("Silence skips the model and does not repeat a clarification")
    @MainActor
    func conversationSilence() async throws {
        var utterances = [" ", "I need a gate", "", " \n ", " Five "]
        var heard: [String] = []
        var questions: [String] = []
        let query = try await resolveQueryConversation(
            capture: { try #require(utterances.isEmpty ? nil : utterances.removeFirst()) },
            respond: { utterance in
                heard.append(utterance)
                return heard.count == 1
                    ? #"{"kind":"clarify","message":"Which gate number?"}"#
                    : #"{"kind":"ready","label":"gate","gateID":5}"#
            },
            speak: { questions.append($0) }
        )
        #expect(heard == ["I need a gate", "Five"])
        #expect(questions == ["Which gate number?"])
        #expect(query.requestedGateID == 5)
    }

    @Test("Invalid replies end the conversation without another capture or playback")
    @MainActor
    func invalidConversationReply() async {
        var captures = 0
        await #expect(throws: ConversationError.self) {
            try await resolveQueryConversation(
                capture: { captures += 1; return "Find a gate" },
                respond: { _ in "gate" },
                speak: { _ in Issue.record("Invalid output must not be spoken") }
            )
        }
        #expect(captures == 1)
    }

    @Test("The next utterance waits for clarification playback to finish")
    @MainActor
    func clarificationPlaybackOrdering() async throws {
        let (started, signal) = AsyncStream<Void>.makeStream()
        let (playback, completion) = AsyncStream<Void>.makeStream()
        var captures = 0
        let task = Task {
            try await resolveQueryConversation(
                capture: { captures += 1; return captures == 1 ? "A gate" : "Five" },
                respond: { _ in
                    captures == 1
                        ? #"{"kind":"clarify","message":"Which gate number?"}"#
                        : #"{"kind":"ready","label":"gate","gateID":5}"#
                },
                speak: { _ in
                    signal.yield(())
                    for await _ in playback {}
                }
            )
        }
        defer { task.cancel(); signal.finish(); completion.finish() }
        for await _ in started { break }
        #expect(captures == 1)
        completion.finish()
        let query = try await task.value
        #expect(captures == 2)
        #expect(query.requestedGateID == 5)
    }

    @Test("Cancellation stops each awaited conversation operation", arguments: ["capture", "respond", "speak"])
    @MainActor
    func conversationCancellation(stage: String) async throws {
        let (started, continuation) = AsyncStream<Void>.makeStream()
        var operations: [String] = []
        let task = Task {
            try await resolveQueryConversation(
                capture: {
                    operations.append("capture")
                    if stage == "capture" {
                        continuation.yield(())
                        try await Task.sleep(for: .seconds(60))
                    }
                    return "A gate"
                },
                respond: { _ in
                    operations.append("respond")
                    if stage == "respond" {
                        continuation.yield(())
                        try await Task.sleep(for: .seconds(60))
                    }
                    return #"{"kind":"clarify","message":"Which gate number?"}"#
                },
                speak: { _ in
                    operations.append("speak")
                    continuation.yield(())
                    try await Task.sleep(for: .seconds(60))
                }
            )
        }
        defer { task.cancel(); continuation.finish() }
        for await _ in started { break }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        let expected = ["capture", "respond", "speak"]
        #expect(operations == Array(expected.prefix(try #require(expected.firstIndex(of: stage)) + 1)))
    }

    @Test("Cancellation discards a model reply that arrives after cancellation")
    @MainActor
    func cancelledLateReply() async throws {
        let (started, continuation) = AsyncStream<Void>.makeStream()
        var response: CheckedContinuation<String, Never>?
        let task = Task {
            try await resolveQueryConversation(
                capture: { "Gate five" },
                respond: { _ in
                    await withCheckedContinuation {
                        response = $0
                        continuation.yield(())
                    }
                },
                speak: { _ in Issue.record("Cancelled output must not be spoken") }
            )
        }
        defer { task.cancel(); continuation.finish() }
        for await _ in started { break }
        task.cancel()
        try #require(response).resume(returning: #"{"kind":"ready","label":"gate","gateID":5}"#)
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    // MARK: - Adaptive Silence Timing (silenceDelay)

    @Test("Endpoint timing uses short pauses for recognized phrases and longer pauses for incomplete utterances", arguments: [
        ("Where is the Washroom?", 0.8),
        ("I need a restroom", 0.8),
        ("Find a Restaurant, please!", 0.8),
        ("gate three", 0.8),
        ("gate to", 0.8),
        ("gate 9", 0.8),
        ("where is gate", 2.0),
        ("I have 2 minutes to find a gate", 2.0),
        ("I am starving", 2.0),
        ("where is the washroom near gate 2", 2.0),
        ("", 2.0)
    ])
    func adaptiveSilenceTiming(transcript: String, expected: TimeInterval) {
        #expect(silenceDelay(for: transcript) == expected)
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
