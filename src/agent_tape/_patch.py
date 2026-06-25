"""
Context managers that transparently patch httpx so any code using
httpx.Client or httpx.AsyncClient is intercepted — no framework-specific
hooks required.

We wrap whatever transport the caller passes rather than replacing it,
so SDKs that supply their own transport (e.g. Anthropic's retry transport)
are still intercepted correctly.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx

from ._assertions import TapeAssertions
from ._tape import Tape
from ._transport import (
    RecordingAsyncTransport,
    RecordingTransport,
    ReplayingAsyncTransport,
    ReplayingTransport,
)


def _wrap_sync(tape: Tape, existing: httpx.BaseTransport | None) -> RecordingTransport:
    return RecordingTransport(tape, wrapped=existing or httpx.HTTPTransport())


def _wrap_async(tape: Tape, existing: httpx.AsyncBaseTransport | None) -> RecordingAsyncTransport:
    return RecordingAsyncTransport(tape, wrapped=existing or httpx.AsyncHTTPTransport())


@contextmanager
def record(path: str | Path):
    """
    Record all httpx calls made within this block to a tape file.

    Usage::

        with agent_tape.record("tapes/summarize.tape.yaml"):
            result = await my_agent.run("summarize document.txt")
    """
    tape = Tape()

    orig_client = httpx.Client.__init__
    orig_async = httpx.AsyncClient.__init__

    def _patched_client(self: httpx.Client, *args: object, **kwargs: object) -> None:
        # Wrap whatever transport the SDK provides (may be its own retry transport)
        kwargs["transport"] = _wrap_sync(tape, kwargs.get("transport"))  # type: ignore[arg-type]
        orig_client(self, *args, **kwargs)

    def _patched_async(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = _wrap_async(tape, kwargs.get("transport"))  # type: ignore[arg-type]
        orig_async(self, *args, **kwargs)

    with (
        patch.object(httpx.Client, "__init__", _patched_client),
        patch.object(httpx.AsyncClient, "__init__", _patched_async),
    ):
        yield tape

    if not tape.interactions:
        existing = Path(path).exists()
        print(
            f"[agent-tape] Recording captured 0 interactions — "
            f"{'existing tape left unchanged' if existing else 'nothing saved'}."
        )
        return

    tape.save(path)
    print(f"[agent-tape] Saved {path} — {tape.summary()}")


@contextmanager
def replay(path: str | Path):
    """
    Replay from a tape file, returning a TapeAssertions object.

    Usage::

        with agent_tape.replay("tapes/summarize.tape.yaml") as tape:
            result = await my_agent.run("summarize document.txt")
            tape.assert_tool_called("read_file")
            tape.assert_task_completed()
    """
    tape = Tape.load(path)
    sync_tr = ReplayingTransport(tape)
    async_tr = ReplayingAsyncTransport(tape)

    orig_client = httpx.Client.__init__
    orig_async = httpx.AsyncClient.__init__

    def _patched_client(self: httpx.Client, *args: object, **kwargs: object) -> None:
        # Replace any existing transport — we must serve from tape, not the network
        kwargs["transport"] = sync_tr
        orig_client(self, *args, **kwargs)

    def _patched_async(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = async_tr
        orig_async(self, *args, **kwargs)

    with (
        patch.object(httpx.Client, "__init__", _patched_client),
        patch.object(httpx.AsyncClient, "__init__", _patched_async),
    ):
        yield TapeAssertions(tape)
