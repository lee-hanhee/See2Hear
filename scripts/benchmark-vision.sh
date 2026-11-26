#!/usr/bin/env bash
# Compile and exercise the app's actual Swift/Core ML code on macOS.
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d "${TMPDIR:-/tmp}/see2hear-vision.XXXXXX")
trap 'rm -rf "$work"' EXIT
for model in vision_experiments/{detection,readers}/models/*.mlpackage; do
    xcrun coremlcompiler compile "$model" "$work" > /dev/null
done
xcrun swiftc -O -whole-module-optimization -swift-version 6 \
    See2Hear/Contracts/*.swift See2Hear/Vision/*.swift vision_experiments/apple/Benchmark.swift \
    -o "$work/benchmark"
"$work/benchmark" "$work" test_data/vision
