"""
Record a real agent run to a tape file.

Run this once with a live API key to capture the tape:

    ANTHROPIC_API_KEY=sk-... python examples/summarize_agent/record_tape.py

Then commit the tape and never pay for this test run again:

    git add tapes/summarize_document.tape.yaml
    git commit -m "chore: update summarize_agent tape"

After that, CI replays it for free via test_agent.py.
"""
from pathlib import Path

import agent_tape
from agent import run_agent

TAPE_PATH = Path(__file__).parent.parent.parent / "tapes" / "summarize_document.tape.yaml"
HERE = Path(__file__).parent

print(f"Recording to: {TAPE_PATH}")

with agent_tape.record(TAPE_PATH):
    result = run_agent("Please summarize the file notes.txt", workdir=HERE)

print(f"\nAgent result:\n{result}")
print(f"\nTape saved. Run tests with:\n  pytest examples/summarize_agent/test_agent.py")
