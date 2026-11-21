import Foundation

enum SignLabel {
    case washroom
    case restaurant
    case gate
}

struct SignQuery {
    /// Reuse this identifier across frames until the passenger changes the request.
    let queryID: UUID
    let label: SignLabel
    /// A specific gate number from 1 through 6, or nil when none was requested.
    let requestedGateID: Int?

    init?(queryID: UUID, label: SignLabel, requestedGateID: Int? = nil) {
        if let requestedGateID {
            guard label == .gate, (1...6).contains(requestedGateID) else {
                return nil
            }
        }
        self.queryID = queryID
        self.label = label
        self.requestedGateID = requestedGateID
    }
}
