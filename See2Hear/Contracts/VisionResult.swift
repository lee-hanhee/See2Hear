import Foundation

/// Pixel coordinates in the upright input frame, measured from its top-left corner.
struct SignBoundingBox {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
    let frameWidth: Int
    let frameHeight: Int
}

enum ArrowDirection: String {
    case left
    case right
    case forward
}

/// Estimated distance from the phone to the sign.
enum SignDistanceBand: String {
    case near      // In the immediate surroundings
    case medium    // Farther within the same area
    case far       // In the background beyond the immediate area
}

struct VisionResult {
    let queryID: UUID
    /// Fast-track boxes guide centering on a category candidate.
    /// A detailed result for a numbered gate includes a box only after the number matches.
    /// On a miss, the box and all sign metadata are nil.
    let boundingBox: SignBoundingBox?
    let observedGateID: String?
    /// Use only the arrow associated with the matched destination.
    let arrowDirection: ArrowDirection?
    let distanceToSign: SignDistanceBand?
}
