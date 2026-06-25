"""
A minimal file-summarizer agent built with the Anthropic SDK.

The agent can read files and write summaries using two tools.
This is the code whose behavior agent-tape records and tests.

Run directly:
    ANTHROPIC_API_KEY=... python examples/summarize_agent/agent.py

Record to a tape (one-time, requires API key):
    ANTHROPIC_API_KEY=... python examples/summarize_agent/record_tape.py

Run tests from the tape (no API key needed, works in CI):
    pytest examples/summarize_agent/test_agent.py
"""
from __future__ import annotations

import json
from pathlib import Path

import anthropic

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a local text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_summary",
        "description": "Write the final summary to a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The summary text"},
                "path": {"type": "string", "description": "Output file path"},
            },
            "required": ["content", "path"],
        },
    },
]


def _handle_tool(name: str, tool_input: dict) -> str:
    if name == "read_file":
        p = Path(tool_input["path"])
        if not p.exists():
            return f"Error: file '{p}' not found"
        return p.read_text(encoding="utf-8")

    if name == "write_summary":
        p = Path(tool_input["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tool_input["content"], encoding="utf-8")
        return f"Summary written to {p}"

    return f"Unknown tool: {name}"


def run_agent(task: str, *, workdir: Path | None = None) -> str:
    """
    Run the summarizer agent on `task`.

    `workdir` is prepended to all file paths so the agent can be tested
    from any working directory. Defaults to the current directory.
    """
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": task}]
    base = workdir or Path(".")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=(
                "You are a helpful assistant. When asked to summarize a file, "
                "use read_file to get its contents, then write_summary to save "
                "a concise summary. Always finish with end_turn once done."
            ),
            messages=messages,
            tools=TOOLS,
        )

        # Collect the assistant turn
        assistant_content: list[dict] = []
        tool_results: list[dict] = []
        final_text: str | None = None

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                final_text = block.text
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                # Resolve file paths relative to workdir
                tool_input = dict(block.input)
                for key in ("path", "output"):
                    if key in tool_input:
                        tool_input[key] = str(base / tool_input[key])

                result = _handle_tool(block.name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            return final_text or ""

        # Feed tool results back for the next turn
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    here = Path(__file__).parent
    print(run_agent("Please summarize the file notes.txt", workdir=here))
