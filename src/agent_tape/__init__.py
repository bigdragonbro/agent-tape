"""
agent-tape — framework-agnostic record/replay test harness for LLM agents.

Quick start::

    import agent_tape

    # Record
    with agent_tape.record("tapes/my_run.tape.yaml"):
        await my_agent.run("summarize document.txt")

    # Replay + assert
    with agent_tape.replay("tapes/my_run.tape.yaml") as tape:
        await my_agent.run("summarize document.txt")
        tape.assert_tool_called("read_file")
        tape.assert_steps_under(5)
        tape.assert_task_completed()
"""

from ._assertions import TapeAssertions
from ._patch import record, replay
from ._tape import Tape, TapeInteraction, TapeRequest, TapeResponse

__version__ = "0.1.0"

__all__ = [
    "record",
    "replay",
    "Tape",
    "TapeAssertions",
    "TapeInteraction",
    "TapeRequest",
    "TapeResponse",
]
