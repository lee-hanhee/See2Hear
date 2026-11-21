struct HapticFeedback {
    enum Strength {
        case off
        case low
        case high
    }

    let strength: Strength
}
