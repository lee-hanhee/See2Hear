#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v swiftlint >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: swiftlint is not installed.
Install it with Homebrew:
  brew install swiftlint
EOF
  exit 1
fi

swiftlint lint --strict
