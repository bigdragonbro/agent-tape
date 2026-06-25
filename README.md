# agent-tape

**Framework-agnostic record/replay test harness for LLM agents.**

Record any agent run to a portable `.tape.yaml` file. Replay it in CI — no API keys, no flakiness, no cost. Assert on behavioral properties: which tools were called, in what order, how many steps, how many tokens.

Works with any framework that uses `httpx` under the hood: Anthropic SDK, OpenAI SDK, LangChain, LangGraph, CrewAI, raw `httpx` calls.

---

## Install

```bash
pip install agent-tape
```

## Quick start

### 1. Record a run

Works with any Anthropic SDK agent out of the box — no changes to agent code:

```python
import anthropic
import agent_tape

client = anthropic.Anthropic()

def run_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    tools = [
        {"name": "read_file",  "description": "Read a local file",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "write_summary", "description": "Write summary to disk",
         "input_schema": {"type": "object", "properties": {"content": {"type": "string"}, "path": {"type": "string"}}, "required": ["content", "path"]}},
    ]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096,
            messages=messages, tools=tools,
        )
        if response.stop_reason == "end_turn":
            return response.content[0].text
        # ... handle tool calls ...

# Record the full run to a tape file
with agent_tape.record("tapes/summarize_document.tape.yaml"):
    result = run_agent("Please summarize the file notes.txt")
```

This saves every LLM call (request + response) to a YAML file — see [`tapes/summarize_document.tape.yaml`](tapes/summarize_document.tape.yaml) for a real example.

### 2. Replay in CI — zero API calls, zero cost

```python
with agent_tape.replay("tapes/summarize_document.tape.yaml") as tape:
    result = run_agent("Please summarize the file notes.txt")

    tape.assert_tool_called("read_file")
    tape.assert_tool_called("read_file", before="write_summary")
    tape.assert_steps_under(5)
    tape.assert_task_completed()
    tape.assert_no_hallucinated_tools(["read_file", "write_summary"])
    tape.assert_total_tokens_under(2000)
```

### 3. Streaming agents — also supported

```python
# Record a streaming run (SSE chunks are captured and stored)
with agent_tape.record("tapes/streaming_run.tape.yaml"):
    with client.messages.stream(model="claude-sonnet-4-6", ...) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

# Replay streams back exactly as they were recorded
with agent_tape.replay("tapes/streaming_run.tape.yaml") as tape:
    with client.messages.stream(...) as stream:
        full_text = stream.get_final_text()
    tape.assert_task_completed()
```

### 4. Use pytest markers

```python
import pytest

@pytest.mark.tape("tapes/summarize_document.tape.yaml")
def test_summarizer_behavior(tape_replay):
    result = run_agent("Please summarize the file notes.txt")
    tape_replay.assert_tool_called("read_file")
    tape_replay.assert_tool_called("read_file", before="write_summary")
    tape_replay.assert_task_completed()
    tape_replay.assert_steps_under(5)
```

---

## Tape format

Tapes are plain YAML — human-readable, git-diffable, easy to review in PRs:

```yaml
version: '1'
metadata:
  recorded_at: '2026-06-24'
interactions:
  - id: call_0
    duration_ms: 234.5
    request:
      method: POST
      url: https://api.anthropic.com/v1/messages
      headers:
        x-api-key: '***'
        content-type: application/json
      body:
        model: claude-sonnet-4-6
        max_tokens: 1024
        messages:
          - role: user
            content: summarize document.txt
    response:
      status: 200
      body:
        stop_reason: tool_use
        content:
          - type: tool_use
            name: read_file
            input:
              path: document.txt
        usage:
          input_tokens: 312
          output_tokens: 48
```

Sensitive headers (`x-api-key`, `authorization`, `cookie`) are scrubbed to `***` automatically.

---

## All assertions

| Method | What it checks |
|---|---|
| `assert_tool_called(name)` | Tool was called at least once |
| `assert_tool_called(name, before=other)` | Tool was called before another |
| `assert_tool_called(name, after=other)` | Tool was called after another |
| `assert_tool_not_called(name)` | Tool was never called |
| `assert_tool_input(name, **fields)` | Tool was called with specific inputs |
| `assert_no_hallucinated_tools(allowed)` | Agent only used tools from the list |
| `assert_steps_under(n)` | Agent finished in ≤ n LLM calls |
| `assert_total_tokens_under(n)` | Total tokens (input+output) ≤ n |
| `assert_task_completed()` | Final stop reason is `end_turn` |
| `assert_stop_reason(reason)` | Final stop reason matches exactly |
| `diff(other_tape)` | Returns list of behavioral differences |

---

## How it works

`agent-tape` patches `httpx.Client` and `httpx.AsyncClient` at the transport layer during the `record()`/`replay()` block. Any framework that builds on `httpx` — which includes the Anthropic SDK, OpenAI SDK, and most others — is intercepted transparently.

**Recording:** wraps the real transport, captures every request/response, writes to YAML.

**Replay:** substitutes a mock transport that returns recorded responses in sequence. The agent code runs unchanged — it just gets pre-recorded answers instead of hitting the live API.

---

## Limitations (v0.1)

- **Streaming not yet supported.** Non-streaming calls only. If your agent uses `stream=True`, switch to `stream=False` for test runs, or open an issue.
- **Sequential replay.** Tapes replay interactions in the order they were recorded. Parallel tool calls that happen in a different order will break replay.
- **Frameworks that pre-create clients.** If a framework creates its `httpx.Client` before the `record()`/`replay()` block opens, it won't be intercepted. Pass a custom transport explicitly in that case.

---

## Contributing

```bash
git clone https://github.com/yourusername/agent-tape
cd agent-tape
pip install -e ".[dev]"
pytest
```

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the roadmap.

---

## License

MIT
