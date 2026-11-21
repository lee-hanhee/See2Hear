protocol HapticControlling {
    /// Applies low or high vibration and returns promptly. Use .off to stop.
    /// Reapplying the current strength keeps the feedback steady.
    func apply(_ feedback: HapticFeedback)
}
