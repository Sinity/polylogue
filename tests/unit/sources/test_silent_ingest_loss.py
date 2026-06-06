"""Regression tests for surfaced parse-time record loss (#1745).

Each test asserts that a loss path is *surfaced* — raised as a typed error,
counted accurately, or escalated to a durable status — rather than silently
dropping records while the parse reports success.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import pytest

from polylogue.sources.decoder_json import (
    JsonValue,
    PartialJsonStreamError,
    iter_json_stream,
)
from polylogue.sources.parsers.antigravity import (
    AntigravityBinaryUnavailableError,
    AntigravityExportError,
    AntigravityLanguageServerClient,
    AntigravityPartialExportError,
    AntigravitySessionSummary,
    iter_language_server_exports,
)

# ---------------------------------------------------------------------------
# AC #1 — mid-stream JSON corruption raises a typed partial-decode error
# ---------------------------------------------------------------------------


def test_mid_stream_corruption_raises_partial_decode_error() -> None:
    """A top-level array valid for the first N items then truncated must raise.

    The previous behaviour returned the records accumulated before the
    corruption with ``found_any=True`` and logged nothing, silently truncating
    the session set.
    """
    # Valid first two objects, then the array is cut off mid-token.
    truncated = b'[{"id": 1}, {"id": 2}, {"id": 3'
    handle = io.BytesIO(truncated)

    with pytest.raises(PartialJsonStreamError) as excinfo:
        list(iter_json_stream(handle, "sessions.json"))

    err = excinfo.value
    assert err.recovered >= 2
    assert "sessions.json" in str(err)


def test_clean_array_does_not_raise() -> None:
    """A well-formed array must still decode all records without raising."""
    handle = io.BytesIO(b'[{"id": 1}, {"id": 2}, {"id": 3}]')
    records = list(iter_json_stream(handle, "sessions.json"))
    ids = [cast(dict[str, JsonValue], r)["id"] for r in records]
    assert ids == [1, 2, 3]


def test_wrong_prefix_with_zero_items_falls_through_not_raises() -> None:
    """A single top-level object (no array) must not raise PartialJsonStreamError.

    Strategy 1 ("item") finds zero items and a JSONError there is a normal
    "try the next strategy" signal — it must be swallowed, not surfaced.
    """
    handle = io.BytesIO(b'{"sessions": [{"id": 1}]}')
    records = list(iter_json_stream(handle, "single.json"))
    # The object is yielded as a single record (dict payload, no unpack match).
    assert records


# ---------------------------------------------------------------------------
# AC #4 — Antigravity distinguishes missing-binary from mid-export failure
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal stand-in for AntigravityLanguageServerClient."""

    def __init__(self, summaries: list[AntigravitySessionSummary], fail_at: int | None) -> None:
        self._summaries = summaries
        self._fail_at = fail_at
        self.closed = False

    def start(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:
        self.closed = True

    def search_sessions(self) -> list[AntigravitySessionSummary]:
        return self._summaries

    def export_markdown(self, cascade_id: str) -> str:
        index = [s.cascade_id for s in self._summaries].index(cascade_id)
        if self._fail_at is not None and index >= self._fail_at:
            raise AntigravityExportError(f"export failed for {cascade_id}")
        return f"### User Input\nhello {cascade_id}\n"


def _as_client(fake: _FakeClient) -> AntigravityLanguageServerClient:
    return cast(AntigravityLanguageServerClient, fake)


def test_antigravity_export_error_taxonomy_is_distinguishable() -> None:
    """Missing-binary and mid-export are distinct typed subtypes of the export error."""
    summary = AntigravitySessionSummary(cascade_id="c1")
    client = _FakeClient([summary], fail_at=None)
    # Sanity: with a working client all sessions are obtained.
    convos = list(iter_language_server_exports(Path("/tmp/x"), client=_as_client(client)))
    assert len(convos) == 1

    # Both are AntigravityExportError subtypes so the broad fallback still
    # catches them, but callers can distinguish benign vs. lossy.
    assert issubclass(AntigravityBinaryUnavailableError, AntigravityExportError)
    assert issubclass(AntigravityPartialExportError, AntigravityExportError)


def test_antigravity_mid_export_failure_reports_obtained_vs_expected() -> None:
    """A mid-iteration failure raises AntigravityPartialExportError with counts.

    Previously the generator yielded the sessions seen before the failure
    then aborted into the fallback, indistinguishable from "not installed".
    """
    summaries = [AntigravitySessionSummary(cascade_id=f"c{i}") for i in range(5)]
    client = _FakeClient(summaries, fail_at=3)

    obtained: list[object] = []
    with pytest.raises(AntigravityPartialExportError) as excinfo:
        for convo in iter_language_server_exports(Path("/tmp/x"), client=_as_client(client)):
            obtained.append(convo)

    err = excinfo.value
    assert err.expected == 5
    assert err.obtained == 3
    assert len(obtained) == 3
    # The lost remainder is visible in the message.
    assert "3 of 5" in str(err)
