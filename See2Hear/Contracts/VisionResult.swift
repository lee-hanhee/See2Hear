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

enum ArrowDirection {
    case left
    case right
    case forward
}

/// Estimated distance from the phone to the sign.
enum SignDistanceBand {
    case near      // Below 5 m
    case medium    // 5 m to below 15 m
    case far       // 15 m or more
}

struct VisionResult {
    let queryID: UUID
    /// Present only when the requested target matches. On a miss, all metadata is nil.
    let boundingBox: SignBoundingBox?
    let observedGateID: String?
    /// Use only the arrow associated with the matched destination.
    let arrowDirection: ArrowDirection?
    let distanceToSign: SignDistanceBand?
}
