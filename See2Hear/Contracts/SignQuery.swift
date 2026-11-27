import Foundation

enum SignLabel: String {
    case washroom
    case restaurant
    case gate
}

struct SignQuery {
    /// Reuse this identifier across frames until the passenger changes the request.
    let queryID: UUID
    let label: SignLabel
    /// Required for gates (1 through 6). Nil for washrooms and restaurants.
    let requestedGateID: Int?

    init?(queryID: UUID, label: SignLabel, requestedGateID: Int? = nil) {
        switch label {
        case .gate:
            guard let requestedGateID, (1...6).contains(requestedGateID) else { return nil }
        case .washroom, .restaurant:
            guard requestedGateID == nil else { return nil }
        }
        self.queryID = queryID
        self.label = label
        self.requestedGateID = requestedGateID
    }
}
