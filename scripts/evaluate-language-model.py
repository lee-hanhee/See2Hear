#!/usr/bin/env python3
"""Replay language regressions with the app's model and prompt on an Apple Silicon Mac.

Install mlx-lm in a Python virtual environment, then run this script.
The first run downloads the model. Each turn uses explicit chat history, as in the app.
The app's Swift decoder validates each reply. Swift tests also cover speech endpoints and cancellation.
"""

import json
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "See2Hear/Language/SpeechInterface.swift"

# Each expected value is either a clarification or a completed destination and gate number.
CASES = [
    [("Could you take me to the washroom please", ("washroom", None))],
    [("I want to look for a restaurant", ("restaurant", None))],
    [("To look for a restaurant", ("restaurant", None))],
    [("To the washroom", ("washroom", None))],
    [("Where is the bathroom?", ("washroom", None))],
    [("I want somewhere to eat", ("restaurant", None))],
    [("Gate two", ("gate", 2))],
    [("Find gate seven", "clarify"), ("Four", ("gate", 4))],
    [("I need a gate", "clarify"), ("Five", ("gate", 5))],
    [("I need help", "clarify"), ("A gate", "clarify"), ("Three", ("gate", 3))],
    [("I want to find my flight", "clarify")],
    [("I need flight five", "clarify")],
    [("A restaurant or a washroom", "clarify"), ("Actually gate three", ("gate", 3))],
    [("I do not want a washroom, I want food", ("restaurant", None))],
    [("I need a gate", "clarify"), ("Actually take me to the washroom", ("washroom", None))],
]


def matches(reply, expected):
    if expected == "clarify":
        return reply.get("kind") == "clarify" and bool(reply.get("message", "").strip())
    label, gate = expected
    return (
        reply.get("kind") == "ready"
        and reply.get("label") == label
        and reply.get("gateID") == gate
    )


def build_decoder(directory):
    source = Path(directory) / "main.swift"
    source.write_text('''import Foundation
let input = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
var output: [String: Any]
do {
    switch try decodeConversationReply(input) {
    case .clarify(let message):
        output = ["kind": "clarify", "message": message]
    case .ready(let query):
        output = ["kind": "ready", "label": query.label.rawValue]
        if let gate = query.requestedGateID { output["gateID"] = gate }
    }
} catch {
    output = ["error": error.localizedDescription]
}
print(String(decoding: try JSONSerialization.data(withJSONObject: output), as: UTF8.self))
''')
    executable = Path(directory) / "decode-reply"
    subprocess.run([
        "xcrun", "swiftc", "-swift-version", "6",
        str(ROOT / "See2Hear/Contracts/SignQuery.swift"),
        str(ROOT / "See2Hear/Language/utils/ConversationHelpers.swift"),
        str(source), "-o", str(executable),
    ], check=True)
    return executable


def main():
    source = SOURCE.read_text()
    model_id = re.search(r'defaultModelID = "([^"]+)"', source).group(1)
    prompt = re.search(r'let llmInstructions = """\n(.*?)\n    """', source, re.S).group(1)
    instructions = textwrap.dedent(prompt)
    model, tokenizer = load(model_id)
    with tempfile.TemporaryDirectory(prefix="see2hear-language-") as directory:
        decoder = build_decoder(directory)
        failures = 0
        turns = 0
        for conversation in CASES:
            history = [{"role": "system", "content": instructions}]
            for utterance, expected in conversation:
                history.append({"role": "user", "content": utterance})
                prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                answer = generate(
                    model, tokenizer, prompt=prompt, max_tokens=256,
                    sampler=make_sampler(temp=0), verbose=False,
                )
                history.append({"role": "assistant", "content": answer})
                try:
                    decoded = subprocess.run(
                        [str(decoder)], input=answer, text=True, capture_output=True, check=True,
                    )
                    passed = matches(json.loads(decoded.stdout), expected)
                except (ValueError, AttributeError, TypeError):
                    passed = False
                turns += 1
                failures += not passed
                print(json.dumps({"utterance": utterance, "reply": answer, "passed": passed}), flush=True)
        print(f"{turns - failures}/{turns} turns passed across {len(CASES)} conversations using {model_id}.")
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
