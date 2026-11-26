# Agent Guidance

Keep changes focused, clear, and aligned with See2Hear's subsystem boundaries:

- **Minimal clear change**: Implement the requested business behavior with the smallest clear change so diffs stay easy to review and maintain.
- **Boundary validation**: Trust established internal contracts in `See2Hear/Contracts/`; validate external input (audio, speech transcripts, LLM replies, camera frames) at its boundary to avoid redundant defensive code.
- **Preserve runtime safety**: Keep necessary permission handling, cancellation, resource cleanup, and framework error handling intact when modifying subsystem code.
- **Justify extra complexity**: Add retries, fallbacks, extra validation, or abstractions only for a stated requirement or demonstrated failure, and explain why they are needed.
- **No misleading lint fixes**: Never silence a lint finding by substituting a misleading default value (for example, hiding an error or invalid state to avoid `force_try` or `force_cast`).
- **Exhaustive switches**: Prefer exhaustive `switch` statements for project-owned enums so new cases trigger compiler errors; retain meaningful defaults for open external inputs.
- **Outcome-focused tests**: Test observable business outcomes using existing Swift Testing (`Testing`) conventions in `See2HearTests/`. Avoid tests that duplicate implementation details or simulate impossible internal states.
- **Verify and report**: Run `./scripts/lint.sh` and relevant tests before finishing, and explicitly report any checks that could not run in the current environment.
