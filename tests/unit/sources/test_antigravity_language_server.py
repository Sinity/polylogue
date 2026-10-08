"""Unit tests for the Antigravity language-server export adapter.

The adapter spawns the local Antigravity language server binary and talks JSON
over a private loopback HTTP port. Tests here mock either the binary
discovery or the HTTP transport so they run without requiring the real
language server to be installed.
"""

from __future__ import annotations

import asyncio
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument
from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.antigravity import (
    AntigravityExportError,
    AntigravityLanguageServerClient,
    AntigravitySessionSummary,
    discover_language_server,
    iter_language_server_exports,
)


class _FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buffer.read()

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture
def fake_client(tmp_path: Path) -> AntigravityLanguageServerClient:
    """A client whose start() is a no-op so tests can drive ._post directly."""

    client = AntigravityLanguageServerClient(tmp_path)

    # Make start()/close() inert and lock the port so _post emits a stable URL.
    def _noop_start() -> None:
        return None

    def _noop_close() -> None:
        return None

    client.start = _noop_start  # type: ignore[method-assign]
    client.close = _noop_close  # type: ignore[method-assign]
    client.port = 49152
    return client


def test_summary_from_payload_requires_cascade_id() -> None:
    assert AntigravitySessionSummary.from_payload({}) is None
    assert AntigravitySessionSummary.from_payload({"cascadeId": ""}) is None
    summary = AntigravitySessionSummary.from_payload(
        {
            "cascadeId": "cascade-1",
            "title": "Title",
            "workspaceName": "ws",
            "snippet": "snip",
            "lastModifiedTime": "2026-03-05T04:21:34Z",
        }
    )
    assert summary is not None
    assert summary.cascade_id == "cascade-1"
    assert summary.workspace_name == "ws"


def test_markdown_export_payload_round_trips() -> None:
    summary = AntigravitySessionSummary(
        cascade_id="c1",
        title="t",
        workspace_name="w",
        snippet="s",
        last_modified_time="2026-03-05T04:21:34Z",
    )
    payload = antigravity.markdown_export_payload(summary, "### User Input\n\nhi\n")
    assert payload["source"] == "antigravity_language_server"
    assert payload["cascadeId"] == "c1"
    assert payload["title"] == "t"
    assert payload["workspaceName"] == "w"
    assert payload["snippet"] == "s"
    assert payload["lastModifiedTime"] == "2026-03-05T04:21:34Z"
    assert antigravity.looks_like_markdown_export(payload) is True


def test_looks_like_markdown_export_rejects_foreign_payloads() -> None:
    assert antigravity.looks_like_markdown_export({"cascadeId": "x", "markdown": "y"}) is False
    assert antigravity.looks_like_markdown_export({"source": "antigravity_language_server", "cascadeId": "x"}) is False


def test_search_sessions_returns_summaries(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(endpoint: str, payload: Any) -> dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {
            "results": [
                {
                    "cascadeId": "cascade-1",
                    "title": "First",
                    "workspaceName": "ws",
                    "snippet": "snip",
                    "lastModifiedTime": "2026-03-05T04:21:34Z",
                },
                {"title": "missing-id"},  # rejected, no cascadeId
                "not-an-object",  # rejected, not a dict
                {"cascadeId": "cascade-2"},
            ]
        }

    monkeypatch.setattr(fake_client, "_post", fake_post)

    summaries = fake_client.search_sessions(limit=5, query="anything")

    assert captured["endpoint"].endswith("/SearchConversations")
    assert captured["payload"] == {"query": "anything", "limit": 5}
    assert [s.cascade_id for s in summaries] == ["cascade-1", "cascade-2"]
    assert summaries[0].workspace_name == "ws"


def test_search_sessions_handles_non_list_results(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_k: {"results": "boom"})
    assert fake_client.search_sessions() == []


def test_export_markdown_returns_string(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fake_client,
        "_post",
        lambda endpoint, payload, **_kwargs: {"markdown": "### User Input\n\nhello\n"},
    )
    assert "User Input" in fake_client.export_markdown("cascade-1")


def test_export_markdown_rejects_missing_or_empty_markdown(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_kwargs: {})
    with pytest.raises(AntigravityExportError):
        fake_client.export_markdown("cascade-1")

    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_kwargs: {"markdown": ""})
    with pytest.raises(AntigravityExportError):
        fake_client.export_markdown("cascade-1")

    monkeypatch.setattr(fake_client, "_post", lambda *_a, **_kwargs: {"markdown": 42})
    with pytest.raises(AntigravityExportError):
        fake_client.export_markdown("cascade-1")


def test_post_wraps_url_errors(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_url_error(*_a: object, **_k: object) -> _FakeHTTPResponse:
        raise URLError("connection refused")

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", raise_url_error)

    with pytest.raises(AntigravityExportError) as exc_info:
        fake_client._post("/endpoint", {"q": "x"})
    assert "connection refused" in str(exc_info.value)


def test_post_wraps_transport_timeouts(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*_a: object, **_k: object) -> _FakeHTTPResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", raise_timeout)

    with pytest.raises(AntigravityExportError, match="timed out"):
        fake_client._post("/endpoint", {})


def test_conversion_outlives_the_probe_budget(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conversion slower than a readiness probe still yields its conversation.

    Real trajectories have taken longer than the probe budget on a loaded
    host, and an expired conversion request is a conversation missing from the
    archive. Anti-vacuity: spend the probe budget on the conversion instead and
    this raises.
    """
    vendor_conversion_s = antigravity._REQUEST_TIMEOUT_S + 5.0

    def fake_urlopen(_request: object, *, timeout: float) -> _FakeHTTPResponse:
        if timeout < vendor_conversion_s:
            raise TimeoutError("timed out")
        return _FakeHTTPResponse(b'{"markdown": "### User Input\\n\\nhello"}')

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", fake_urlopen)

    assert fake_client.export_markdown("cascade").startswith("### User Input")


def test_probe_and_search_keep_the_short_budget(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only conversion gets the long budget; a dead server must still fail fast.

    Anti-vacuity: widen the probe budget to the conversion budget and the
    equality below fails.
    """
    budgets: list[float] = []

    def fake_urlopen(_request: object, *, timeout: float) -> _FakeHTTPResponse:
        budgets.append(timeout)
        return _FakeHTTPResponse(b'{"results": []}')

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", fake_urlopen)

    fake_client.search_sessions()

    assert budgets == [antigravity._REQUEST_TIMEOUT_S]
    assert antigravity._CONVERSION_TIMEOUT_S > antigravity._REQUEST_TIMEOUT_S


def test_post_rejects_non_object_responses(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(*_a: object, **_k: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(b"[1, 2, 3]")

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", fake_urlopen)

    with pytest.raises(AntigravityExportError) as exc_info:
        fake_client._post("/endpoint", {})
    assert "non-object JSON" in str(exc_info.value)


def test_post_returns_decoded_object(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(*_a: object, **_k: object) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(b'{"ok": true, "n": 1}')

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", fake_urlopen)

    result = fake_client._post("/endpoint", {"q": "x"})
    assert result == {"ok": True, "n": 1}


def test_discover_language_server_prefers_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    system = tmp_path / "system" / "language_server_linux_x64"
    system.parent.mkdir()
    system.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._SYSTEM_LANGUAGE_SERVER", system)
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.shutil.which",
        lambda name: "/usr/local/bin/language_server_linux_x64" if name == "language_server_linux_x64" else None,
    )
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._NIX_STORE", Path("/nonexistent-nix-store"))
    found = discover_language_server()
    assert found == Path("/usr/local/bin/language_server_linux_x64")


def test_discover_language_server_finds_the_system_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: dropping the system-install probe returns ``None`` here."""
    system = tmp_path / "system" / "language_server_linux_x64"
    system.parent.mkdir()
    system.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._SYSTEM_LANGUAGE_SERVER", system)
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.shutil.which", lambda _name: None)
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._NIX_STORE", Path("/nonexistent-nix-store"))
    assert discover_language_server() == system


def test_discover_language_server_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._SYSTEM_LANGUAGE_SERVER", tmp_path / "absent")
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.shutil.which", lambda _name: None)
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._NIX_STORE", Path("/nonexistent-nix-store"))
    assert discover_language_server() is None


def test_client_start_raises_when_binary_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.discover_language_server",
        lambda: None,
    )
    client = AntigravityLanguageServerClient(tmp_path)
    with pytest.raises(AntigravityExportError) as exc_info:
        client.start()
    assert (
        "not be found" in str(exc_info.value)
        or "not be found" in str(exc_info.value).lower()
        or "was not found" in str(exc_info.value)
    )


class _FakeClientForExports:
    """Drop-in fake of AntigravityLanguageServerClient for driver tests."""

    def __init__(
        self,
        summaries: list[AntigravitySessionSummary],
        markdown: dict[str, str],
    ) -> None:
        self._summaries = summaries
        self._markdown = markdown
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        return list(self._summaries)

    def export_markdown(self, cascade_id: str) -> str:
        return self._markdown[cascade_id]


def _touch_conversation_pb(root: Path, *cascade_ids: str) -> None:
    """Create empty ``conversations/<cascade_id>.pb`` files.

    Cascade discovery is now ground-truthed off this directory listing
    (polylogue-eo81) rather than the language server's own search/list RPCs,
    which only surface a small recently-tracked subset -- so driver tests
    against a fake client must still provide the disk-truth files.
    """
    conversations = root / "conversations"
    conversations.mkdir(parents=True, exist_ok=True)
    for cascade_id in cascade_ids:
        (conversations / f"{cascade_id}.pb").write_bytes(b"")


def test_iter_language_server_exports_yields_parsed_sessions(
    tmp_path: Path,
) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1", "cascade-2")
    summaries = [
        AntigravitySessionSummary(cascade_id="cascade-1", title="One", workspace_name="ws"),
        AntigravitySessionSummary(cascade_id="cascade-2", title="Two"),
    ]
    markdown = {
        "cascade-1": "### User Input\n\nhello\n\n### Planner Response\n\nhi\n",
        "cascade-2": "### User Input\n\nfoo\n\n### Planner Response\n\nbar\n",
    }
    fake = _FakeClientForExports(summaries, markdown)

    sessions = list(iter_language_server_exports(tmp_path, client=fake))

    assert [c.provider_session_id for c in sessions] == [
        "cascade-1",
        "cascade-2",
    ]
    assert all(c.source_name is Provider.ANTIGRAVITY for c in sessions)
    assert sessions[0].messages[0].text == "hello"
    assert sessions[0].messages[1].text == "hi"
    # Externally supplied client must not be started or closed by the driver.
    assert fake.started is False
    assert fake.closed is False


def test_iter_language_server_exports_only_cascade_ids_filters_corpus(
    tmp_path: Path,
) -> None:
    """polylogue-3m3de: the daemon's periodic reconciler restricts export to
    not-yet-acquired cascades so it never re-converts the whole corpus."""
    _touch_conversation_pb(tmp_path, "cascade-1", "cascade-2", "cascade-3")
    summaries = [
        AntigravitySessionSummary(cascade_id="cascade-1", title="One"),
        AntigravitySessionSummary(cascade_id="cascade-2", title="Two"),
        AntigravitySessionSummary(cascade_id="cascade-3", title="Three"),
    ]
    markdown = {
        "cascade-1": "### User Input\n\na\n",
        "cascade-2": "### User Input\n\nb\n",
        "cascade-3": "### User Input\n\nc\n",
    }
    fake = _FakeClientForExports(summaries, markdown)

    sessions = list(
        iter_language_server_exports(
            tmp_path,
            client=fake,
            only_cascade_ids=frozenset({"cascade-2"}),
        )
    )

    assert [c.provider_session_id for c in sessions] == ["cascade-2"]


def test_iter_language_server_exports_only_cascade_ids_empty_set_is_noop(
    tmp_path: Path,
) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")
    fake = _FakeClientForExports(
        [AntigravitySessionSummary(cascade_id="cascade-1")],
        {"cascade-1": "### User Input\n\na\n"},
    )

    sessions = list(
        iter_language_server_exports(
            tmp_path,
            client=fake,
            only_cascade_ids=frozenset(),
        )
    )

    assert sessions == []


def test_iter_language_server_exports_admits_declared_files_and_excludes_nested_copies(
    tmp_path: Path,
) -> None:
    _touch_conversation_pb(tmp_path, "top-level")
    nested = tmp_path / "conversations" / "workspace" / "nested.pb"
    nested.parent.mkdir()
    nested.write_bytes(b"")
    fake = _FakeClientForExports(
        [
            AntigravitySessionSummary(cascade_id="top-level"),
            AntigravitySessionSummary(cascade_id="nested"),
        ],
        {
            "top-level": "### User Input\n\ntop\n",
            "nested": "### User Input\n\nnested\n",
        },
    )

    assert [path.relative_to(tmp_path).as_posix() for path in antigravity._conversation_pb_paths(tmp_path)] == [
        "conversations/top-level.pb"
    ]
    sessions = list(antigravity.iter_language_server_exports(tmp_path, client=fake))

    assert [session.provider_session_id for session in sessions] == ["top-level"]


def test_export_results_surfaces_duplicate_identity_without_suppressing_other_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_conversation_pb(tmp_path, "duplicate", "healthy")
    duplicate = tmp_path / "conversations" / "duplicate.pb"
    fake = _FakeClientForExports(
        [AntigravitySessionSummary(cascade_id="duplicate"), AntigravitySessionSummary(cascade_id="healthy")],
        {
            "duplicate": "### User Input\n\nduplicate\n",
            "healthy": "### User Input\n\nhealthy\n",
        },
    )

    # Duplicate path identities cannot arise from the declared source layout,
    # so inject the duplicated manifest at the language-server export seam.
    # This preserves coverage of its typed duplicate outcome without treating
    # an out-of-layout nested copy as an admitted conversation.
    monkeypatch.setattr(
        antigravity, "_conversation_pb_paths", lambda _root: [duplicate, duplicate, duplicate.parent / "healthy.pb"]
    )
    outcomes = list(antigravity.iter_language_server_export_results(tmp_path, client=fake))

    assert [outcome.cascade_id for outcome in outcomes] == ["duplicate", "duplicate", "healthy"]
    duplicate_failures = [outcome for outcome in outcomes if outcome.error == "duplicate conversation identity"]
    assert len(duplicate_failures) == 1
    assert [outcome.cascade_id for outcome in outcomes if outcome.obtained] == ["duplicate", "healthy"]


def test_iter_language_server_exports_manages_owned_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")
    summaries = [
        AntigravitySessionSummary(cascade_id="cascade-1", title="One"),
    ]
    markdown = {"cascade-1": "### User Input\n\nhello\n\n### Planner Response\n\nhi\n"}
    fake = _FakeClientForExports(summaries, markdown)

    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        lambda root: fake,
    )

    sessions = list(iter_language_server_exports(tmp_path))

    assert [c.provider_session_id for c in sessions] == ["cascade-1"]
    assert fake.started is True
    assert fake.closed is True


def test_iter_language_server_exports_closes_client_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")

    class _ExplodingClient(_FakeClientForExports):
        def export_markdown(self, cascade_id: str) -> str:
            raise AntigravityExportError("boom")

    fake = _ExplodingClient(
        [AntigravitySessionSummary(cascade_id="cascade-1", title="One")],
        {},
    )
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        lambda root: fake,
    )

    def _consume() -> Iterator[Any]:
        yield from iter_language_server_exports(tmp_path)

    with pytest.raises(AntigravityExportError):
        list(_consume())

    assert fake.started is True
    assert fake.closed is True


@pytest.mark.parametrize("control_flow", [asyncio.CancelledError, KeyboardInterrupt])
def test_owned_export_client_closes_on_cancellation_and_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_flow: type[BaseException],
) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")

    class _InterruptedClient(_FakeClientForExports):
        def export_markdown(self, cascade_id: str) -> str:
            raise control_flow("conversion interrupted")

    fake = _InterruptedClient([AntigravitySessionSummary(cascade_id="cascade-1")], {})
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        lambda root: fake,
    )

    with pytest.raises(control_flow):
        list(antigravity.iter_language_server_export_results(tmp_path))

    assert fake.started is True
    assert fake.closed is True


def test_export_results_reject_source_mutation_during_conversion(tmp_path: Path) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")

    class _MutatingClient:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

        def search_sessions(self, **_kwargs: object) -> list[AntigravitySessionSummary]:
            return []

        def export_markdown(self, cascade_id: str) -> str:
            (tmp_path / "conversations" / f"{cascade_id}.pb").write_bytes(b"changed")
            return "### User Input\n\nhello"

    outcomes = list(antigravity.iter_language_server_export_results(tmp_path, client=_MutatingClient()))

    assert len(outcomes) == 1
    assert outcomes[0].obtained is False
    assert outcomes[0].error == "conversation protobuf changed during conversion"


def test_export_results_surfaces_search_handshake_failure(tmp_path: Path) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")

    class _BrokenSearchClient(_FakeClientForExports):
        def search_sessions(self, **_kwargs: object) -> list[AntigravitySessionSummary]:
            raise AntigravityExportError("server capability mismatch")

    client = _BrokenSearchClient([], {})
    with pytest.raises(AntigravityExportError, match="SearchConversations handshake failed"):
        list(antigravity.iter_language_server_export_results(tmp_path, client=client))


@pytest.mark.parametrize("markdown", ["", "### User Input\n\n", "plain text without a section"])
def test_export_results_rejects_partial_or_untyped_markdown(tmp_path: Path, markdown: str) -> None:
    _touch_conversation_pb(tmp_path, "cascade-1")
    client = _FakeClientForExports(
        [AntigravitySessionSummary(cascade_id="cascade-1")],
        {"cascade-1": markdown},
    )

    outcomes = list(antigravity.iter_language_server_export_results(tmp_path, client=client))

    assert len(outcomes) == 1
    assert outcomes[0].obtained is False
    assert outcomes[0].error


def test_language_server_version_handshake_accepts_declared_vendor_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER_VERSION", "2.1.1")

    assert antigravity._discover_language_server_version(tmp_path / "language_server_linux_x64") == "2.1.1"


def test_language_server_version_handshake_rejects_incompatible_vendor_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER_VERSION", "0.9.0")

    with pytest.raises(AntigravityExportError, match="incompatible"):
        antigravity._discover_language_server_version(tmp_path / "language_server_linux_x64")


# One transcript in the shape the language server actually emits: tool activity
# is rendered as one-line italic markers, and the markers that follow a user
# turn sit inside that turn's own ``### User Input`` section.
_ACTIVITY_TRANSCRIPT = """# Chat Conversation

Note: _This is purely the output of the chat conversation._

### User Input

add a health endpoint

*Listed directory [service](file:///w/service) *

*Viewed [server.py](file:///w/service/server.py) *

*Edited relevant file*

*User accepted the command `pytest -q`*

*Checked command status*

### Planner Response

The endpoint is in place.

*Grep searched codebase*

Tests pass.
"""


def test_tool_activity_markers_become_typed_tool_use_blocks() -> None:
    """Rendered tool activity is lifted out of prose into typed blocks.

    Anti-vacuity: leaving the markers in their section's text makes this red
    twice over -- ``tool_use`` blocks disappear, and the operator's word count
    reabsorbs the five agent actions rendered under ``### User Input``.
    """
    session = antigravity.parse_markdown_export(_ACTIVITY_TRANSCRIPT, AntigravitySessionSummary(cascade_id="cascade-1"))

    assert [(m.role.value, m.message_type.value) for m in session.messages] == [
        ("user", "message"),
        ("assistant", "tool_use"),
        ("assistant", "message"),
        ("assistant", "tool_use"),
        ("assistant", "message"),
    ]
    assert session.messages[0].text == "add a health endpoint"
    assert session.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED

    activity = session.messages[1]
    assert activity.text is None
    assert activity.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert [(b.tool_name, b.tool_input) for b in activity.blocks] == [
        ("listed_directory", {"path": "file:///w/service"}),
        ("viewed_file", {"path": "file:///w/service/server.py"}),
        ("edited_file", None),
        ("accepted_command", {"command": "pytest -q"}),
        ("checked_command_status", None),
    ]
    assert all(b.type is BlockType.TOOL_USE for b in activity.blocks)
    # Every tool_use block carries its own id so the actions view can key on it.
    assert len({b.tool_id for b in activity.blocks}) == len(activity.blocks)

    assert session.messages[2].text == "The endpoint is in place."
    assert [b.tool_name for b in session.messages[3].blocks] == ["grep_searched_codebase"]
    assert session.messages[4].text == "Tests pass."

    # No agent action is left counted as authored prose.
    human_text = " ".join(m.text or "" for m in session.messages if m.material_origin is MaterialOrigin.HUMAN_AUTHORED)
    assert "Edited relevant file" not in human_text
    assert "pytest -q" not in human_text


def test_assistant_italic_prose_is_not_read_as_tool_activity() -> None:
    """The vocabulary is closed, so italicised prose stays one text block.

    Anti-vacuity: matching any single-line italic as a marker turns both
    sentences below into ``tool_use`` blocks with an invented tool name.
    """
    transcript = (
        "### Planner Response\n\n"
        "*What does the retry path actually guarantee?*\n\n"
        "*(Note: the second table is derived, not measured.)*\n"
    )

    session = antigravity.parse_markdown_export(transcript, AntigravitySessionSummary(cascade_id="cascade-2"))

    assert len(session.messages) == 1
    assert [b.type for b in session.messages[0].blocks] == [BlockType.TEXT]
    assert "retry path" in (session.messages[0].text or "")


def test_readiness_retries_past_a_probe_that_spends_the_whole_request_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single slow probe must not end readiness.

    One probe can consume the entire ``_REQUEST_TIMEOUT_S`` socket budget, so
    readiness is bounded by an attempt floor as well as a deadline. Anti-vacuity:
    with the floor removed, ``startup_timeout_s=0.0`` runs no probe at all and
    the first (here: zeroth) failure is terminal.
    """
    monkeypatch.setattr(antigravity, "_READY_RETRY_SLEEP_S", 0.0)
    client = AntigravityLanguageServerClient(tmp_path, startup_timeout_s=0.0)
    attempts: list[int] = []

    def flaky_post(endpoint: str, payload: JSONDocument, *, timeout: float | None = None) -> JSONDocument:
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise AntigravityExportError("timed out")
        return {"results": []}

    client._post = flaky_post  # type: ignore[method-assign]
    client._wait_until_ready()

    assert len(attempts) == 3


def test_readiness_failure_reports_how_many_probes_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(antigravity, "_READY_RETRY_SLEEP_S", 0.0)
    client = AntigravityLanguageServerClient(tmp_path, startup_timeout_s=0.0)

    def always_failing(endpoint: str, payload: JSONDocument, *, timeout: float | None = None) -> JSONDocument:
        raise AntigravityExportError("connection refused")

    client._post = always_failing  # type: ignore[method-assign]
    with pytest.raises(AntigravityExportError, match=r"did not become ready after 3 probes"):
        client._wait_until_ready()


def test_readiness_budget_admits_more_than_one_request_timeout() -> None:
    """The declared deadline must be able to outlast a single slow probe."""
    assert antigravity._STARTUP_TIMEOUT_S > antigravity._REQUEST_TIMEOUT_S
    assert antigravity._MIN_READY_ATTEMPTS >= 2


def test_repeated_identical_activity_runs_keep_distinct_identities() -> None:
    """Two runs rendering the same markers are distinct events, not one.

    Anti-vacuity: seeding activity identity from the rendered markers alone
    gives both edits the same ``provider_message_id``, and the writer then
    drops both to positional identity as an ambiguous native id.
    """
    transcript = (
        "### Planner Response\n\n*Edited relevant file*\n\nFirst pass.\n\n*Edited relevant file*\n\nSecond pass.\n"
    )

    session = antigravity.parse_markdown_export(transcript, AntigravitySessionSummary(cascade_id="cascade-3"))

    activity = [m for m in session.messages if m.message_type.value == "tool_use"]
    assert len(activity) == 2
    assert [b.tool_name for m in activity for b in m.blocks] == ["edited_file", "edited_file"]
    assert activity[0].provider_message_id != activity[1].provider_message_id


def test_a_multi_line_accepted_command_is_one_marker_not_prose() -> None:
    """A heredoc command is rendered verbatim, so a marker spans many lines.

    Anti-vacuity: matching markers line by line leaves this command as text in
    the ``### User Input`` section, where it is counted as operator prose --
    which is what the whole-section scan exists to prevent.
    """
    transcript = (
        "### User Input\n\n"
        "regenerate the fixtures\n\n"
        "*User accepted the command `python3 << 'PYEOF'\n"
        "for row in rows:\n"
        "    print(row)\n"
        "PYEOF`*\n\n"
        "*Checked command status*\n"
    )

    session = antigravity.parse_markdown_export(transcript, AntigravitySessionSummary(cascade_id="cascade-4"))

    assert [(m.role.value, m.message_type.value) for m in session.messages] == [
        ("user", "message"),
        ("assistant", "tool_use"),
    ]
    assert session.messages[0].text == "regenerate the fixtures"
    command = session.messages[1].blocks[0]
    assert command.tool_name == "accepted_command"
    assert command.tool_input is not None
    assert str(command.tool_input["command"]).splitlines()[-1] == "PYEOF"
    assert [b.tool_name for b in session.messages[1].blocks] == ["accepted_command", "checked_command_status"]


def test_packaged_binary_is_found_under_either_ide_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: drop the ``lib/antigravity-ide`` layout and the current package is never found."""
    binary = tmp_path / "abc-antigravity-ide-2.1.1" / antigravity._PACKAGED_LANGUAGE_SERVER_PATHS[1]
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    (tmp_path / "unrelated-package").mkdir()
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._NIX_STORE", tmp_path)
    assert antigravity._nix_store_language_server() == binary


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


def test_client_reads_only_its_own_discovery_file_and_sends_csrf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The vendor port comes from our child's discovery file; every request carries the run token.

    Anti-vacuity: accept any discovery file and the operator's running IDE
    (pid 111) would be used; drop the header and the server answers 401.
    """
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    (daemon_dir / "ls_aaaa.json").write_text('{"pid": 111, "httpPort": 40001}')
    (daemon_dir / "ls_bbbb.json").write_text('{"pid": 222, "httpPort": 40002}')
    client = AntigravityLanguageServerClient(tmp_path, startup_timeout_s=2.0)
    client._process = _FakeProcess(222)  # type: ignore[assignment]
    assert client._await_discovered_port(before_launch={}) == 40002
    client.port = 40002

    seen: list[Any] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeHTTPResponse:
        del timeout
        seen.append(request)
        return _FakeHTTPResponse(b'{"results": []}')

    monkeypatch.setattr(antigravity._LOOPBACK_OPENER, "open", fake_urlopen)
    client.search_sessions()
    request = seen[0]
    assert request.full_url.startswith("http://127.0.0.1:40002/")
    assert request.get_header("X-codeium-csrf-token") == client._csrf_token
    assert len(client._csrf_token) >= 32


def test_client_launches_vendor_on_a_random_port_with_a_run_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: reserve a port ahead of the child and the TOCTOU returns."""
    launched: list[list[str]] = []

    def fake_popen(cmd: list[str], **_kwargs: object) -> _FakeProcess:
        launched.append(cmd)
        return _FakeProcess(333)

    binary = tmp_path / "language_server_linux_x64"
    binary.write_bytes(b"")
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.subprocess.Popen", fake_popen)
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._discover_language_server_version", lambda _b: "1.11.0")
    monkeypatch.setattr(AntigravityLanguageServerClient, "_await_discovered_port", lambda self, *, before_launch: 40003)
    monkeypatch.setattr(AntigravityLanguageServerClient, "_wait_until_ready", lambda self: None)
    client = AntigravityLanguageServerClient(tmp_path / "antigravity", language_server_path=binary)
    client.start()
    cmd = launched[0]
    assert "-http_server_port=0" in cmd
    assert f"-csrf_token={client._csrf_token}" in cmd
    assert client.port == 40003


def test_a_discovery_file_older_than_the_launch_is_not_accepted_for_a_reused_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: accept on pid alone and the crashed server's stale port 40009 wins."""
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    stale = daemon_dir / "ls_stale.json"
    stale.write_text('{"pid": 444, "httpPort": 40009}')
    os.utime(stale, ns=(1_000_000_000, 1_000_000_000))
    client = AntigravityLanguageServerClient(tmp_path)
    before_launch = client._discovery_snapshot()
    client._process = _FakeProcess(444)  # type: ignore[assignment]
    waits: list[float] = []

    def child_rewrites(seconds: float) -> None:
        waits.append(seconds)
        stale.write_text('{"pid": 444, "httpPort": 40010}')
        os.utime(stale, ns=(3_000_000_000, 3_000_000_000))

    monkeypatch.setattr("polylogue.sources.parsers.antigravity.time.sleep", child_rewrites)

    assert client._await_discovered_port(before_launch=before_launch) == 40010
    assert waits


def test_a_slow_child_is_still_waited_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A live child that publishes after the startup budget still starts.

    Anti-vacuity (Codex P2, #5704): stop waiting at ``startup_timeout_s`` and
    a slow but healthy server becomes an export failure.
    """
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    client = AntigravityLanguageServerClient(tmp_path, startup_timeout_s=0.0)
    client._process = _FakeProcess(555)  # type: ignore[assignment]
    waits: list[float] = []

    def publish_late(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 50:
            (daemon_dir / "ls_late.json").write_text('{"pid": 555, "httpPort": 40011}')

    monkeypatch.setattr("polylogue.sources.parsers.antigravity.time.sleep", publish_late)

    assert client._await_discovered_port(before_launch={}) == 40011
    assert len(waits) == 50


def test_a_start_that_fails_stops_its_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A refused start leaves no language server running.

    Anti-vacuity (Codex P2, #5704): raise from ``start`` without cleanup and
    the child stays alive, so retries accumulate servers.
    """
    terminated: list[int] = []

    class Child(_FakeProcess):
        def terminate(self) -> None:
            terminated.append(self.pid)
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode if self.returncode is not None else 0

    binary = tmp_path / "language_server_linux_x64"
    binary.write_bytes(b"")
    monkeypatch.setattr("polylogue.sources.parsers.antigravity.subprocess.Popen", lambda *_a, **_k: Child(666))
    monkeypatch.setattr("polylogue.sources.parsers.antigravity._discover_language_server_version", lambda _b: "1.11.0")
    monkeypatch.setattr(AntigravityLanguageServerClient, "_await_discovered_port", lambda self, *, before_launch: 40012)

    def refuse(self: AntigravityLanguageServerClient) -> None:
        raise AntigravityExportError("not ready")

    monkeypatch.setattr(AntigravityLanguageServerClient, "_wait_until_ready", refuse)
    client = AntigravityLanguageServerClient(tmp_path / "antigravity", language_server_path=binary)

    with pytest.raises(AntigravityExportError, match="not ready"):
        client.start()
    assert terminated == [666]
    assert client._process is None


def test_language_server_rpcs_never_go_through_an_environment_proxy(
    fake_client: AntigravityLanguageServerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CSRF-bearing RPC reaches the loopback server even with ``HTTP_PROXY`` set.

    Anti-vacuity (Codex P1, #5704): open with the default ``urlopen`` and the
    request goes to the dead proxy, so the call raises instead of reaching
    the server.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.headers.get("x-codeium-csrf-token"))
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = b'{"markdown": "### User Input\\n\\nhello"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # A port nothing listens on: a proxied request fails to connect.
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        fake_client.port = server.server_address[1]

        assert fake_client.export_markdown("cascade").startswith("### User Input")
    finally:
        server.shutdown()
        server.server_close()
    assert len(seen) == 1


def test_a_discovery_file_with_a_coarse_timestamp_is_accepted(tmp_path: Path) -> None:
    """A file the child wrote after launch is accepted even if its mtime reads earlier.

    Anti-vacuity (Codex P1, #5704): compare mtime with the nanosecond launch
    instant and a filesystem that rounds timestamps down rejects the child's
    own port forever.
    """
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir()
    client = AntigravityLanguageServerClient(tmp_path)
    before_launch = client._discovery_snapshot()
    written = daemon_dir / "ls_new.json"
    written.write_text('{"pid": 777, "httpPort": 40013}')
    # Rounded down to a whole second, before any plausible launch instant.
    os.utime(written, ns=(1_000_000_000, 1_000_000_000))
    client._process = _FakeProcess(777)  # type: ignore[assignment]

    assert client._await_discovered_port(before_launch=before_launch) == 40013
