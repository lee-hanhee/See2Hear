# capstone

## Subsystems

- **Vision** analyzes each camera frame for the requested sign and reports its location in the image, along with any gate number, arrow direction, and estimated phone-to-sign distance it can determine.
- **Language** turns a spoken request into a sign query and speaks the feedback supplied by Orchestration.
- **Haptic** controls the phone’s vibration by applying the requested strength (`low` or `high`) or stopping it (`off`).
- **Orchestration** manages the active query, sends frames to Vision, and uses the results to decide when to speak and which haptic strength to request.

Shared data structures are defined in `Contracts/`.

Each subsystem will implement the protocols declared in `*Interface.swift`.

## Linting

Install [SwiftLint](https://github.com/realm/SwiftLint) with Homebrew:

```sh
brew install swiftlint
```

Run the shared local/CI lint check from the repository root:

```sh
./scripts/lint.sh
```