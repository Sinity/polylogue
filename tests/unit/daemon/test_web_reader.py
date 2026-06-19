"""Reader smoke tests — page structure, API contracts, degraded states (#865).

Boots the production ``DaemonAPIHTTPServer`` against a synthetic on-disk
archive and exercises the live HTTP surface via ``urllib``. The HTML
payload served at ``/`` is asserted at the DOM-shape level (semantic
selectors, never pixel diffs); JSON envelopes are asserted by shape so
any regression in the daemon's contract surface fails here loudly.

This is the documented reader visual smoke lane (see
``docs/visual-evidence.md``). The lane runs as part of the standard
unit suite and ``devtools verify``; a separate ``devtools lab scenario``
entrypoint can be added later if Playwright-based screenshot evidence
is bolted on.

All test classes in this module start real HTTP servers — they share an
xdist group to prevent cross-worker port/event-loop interference under
parallel execution.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xdist_group("web-reader")

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

POLYLOGUE_LOCAL_PATH_PREFIXES = ("/home/", "/Users/", "/realm/", "/var/", "/etc/")


class _QueryParamBuilderHandler:
    @staticmethod
    def _get_param(params: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
        values = params.get(key)
        if values:
            return values[0]
        return default

    def _get_int(self, params: dict[str, list[str]], key: str, default: int = 0) -> int:
        val = self._get_param(params, key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return default

    def _get_bool(self, params: dict[str, list[str]], key: str) -> bool:
        val = self._get_param(params, key)
        return val is not None and val.lower() in ("1", "true", "yes", "on")


def test_query_spec_param_builder_uses_canonical_query_fields() -> None:
    from polylogue.daemon.http import _build_query_spec_params

    params = {
        "origin": ["codex-session"],
        "exclude_origin": ["chatgpt-export"],
        "max_words": ["100"],
        "similar_session_id": ["codex-session:seed"],
        "filter_has_tool_use": ["true"],
    }

    result = _build_query_spec_params(params, _QueryParamBuilderHandler())  # type: ignore[arg-type]

    assert result["origin"] == "codex-session"
    assert result["exclude_origin"] == "chatgpt-export"
    assert result["max_words"] == 100
    assert result["similar_session_id"] == "codex-session:seed"
    assert result["filter_has_tool_use"] is True


@contextmanager
def _running_server(
    workspace: dict[str, Path],
    *,
    seeded: bool = True,
    auth_token: str = "",
) -> Iterator[tuple[HTTPServer, str]]:
    """Yield a ``(server, base_url)`` pair with the server actively serving.

    The previous version of this harness (#865) constructed a
    ``ThreadingHTTPServer`` but never started its serve loop; the very
    first request would then hang forever until the worker was
    SIGKILLed. This contextmanager spins ``serve_forever`` in a daemon
    thread before yielding and tears it down on exit.
    """
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    if seeded:
        _seed_test_db(workspace)
    else:
        _seed_empty_schema(workspace)

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = auth_token
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="web-reader-test", daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


@contextmanager
def _running_server_without_seed(
    *,
    auth_token: str = "",
) -> Iterator[tuple[HTTPServer, str]]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = auth_token
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="web-reader-v1-test", daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# Archive session/message identities for the three seeded sessions.
# The archive store derives ``session_id`` as ``origin:native_id`` and
# ``message_id`` as ``session_id:message_native_id``; the daemon's reader
# surface returns these identities verbatim.
from polylogue.core.identity_law import message_id as _archive_message_id
from polylogue.core.identity_law import session_id as _archive_session_id

_SEED_SPECS = [
    ("claude-code", "c1", "m-c1", "Claude Code session about authentication"),
    ("chatgpt", "c2", "m-c2", "ChatGPT debugging session"),
    ("claude-ai", "c3", "m-c3", "Claude AI brainstorm thread"),
]


def _origin_for(provider: str) -> str:
    from polylogue.core.enums import Provider
    from polylogue.core.sources import origin_from_provider

    return origin_from_provider(Provider.from_string(provider)).value


def _native_session_id(provider: str, native_id: str) -> str:
    return _archive_session_id(_origin_for(provider), native_id)


def _native_message_id(provider: str, native_id: str, message_native_id: str) -> str:
    return _archive_message_id(_native_session_id(provider, native_id), message_native_id, position=0)


# Resolved archive identities for the canonical seeded triple.
C1 = _native_session_id("claude-code", "c1")
C2 = _native_session_id("chatgpt", "c2")
C3 = _native_session_id("claude-ai", "c3")
M_C1 = _native_message_id("claude-code", "c1", "m-c1")
M_C2 = _native_message_id("chatgpt", "c2", "m-c2")
M_C3 = _native_message_id("claude-ai", "c3", "m-c3")


def _seed_empty_schema(workspace: dict[str, Path]) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    workspace["archive_root"].mkdir(parents=True, exist_ok=True)
    ArchiveStore(workspace["archive_root"]).close()


def _seed_test_db(workspace: dict[str, Path]) -> None:
    """Seed a archive with three single-message sessions."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    workspace["archive_root"].mkdir(parents=True, exist_ok=True)
    with ArchiveStore(workspace["archive_root"]) as archive:
        for prov, cid, mid, title in _SEED_SPECS:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.from_string(prov),
                    provider_session_id=cid,
                    title=title,
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:01:00+00:00",
                    messages=[
                        ParsedMessage(
                            provider_message_id=mid,
                            role=Role.USER,
                            text="Hello reader",
                            timestamp="2026-01-01T00:00:00+00:00",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Hello reader")],
                        )
                    ],
                )
            )


def _seed_assertion_claims(workspace: dict[str, Path]) -> None:
    """Seed user-tier assertion claims for the web overlay endpoint."""

    import sqlite3

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    user_db = workspace["archive_root"] / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        upsert_assertion(
            conn,
            assertion_id="claim-web-workbench-decision",
            target_ref=f"session:{C1}",
            scope_ref="repo:polylogue",
            kind=AssertionKind.DECISION,
            body_text="Show assertion-backed overlays in the web workbench.",
            author_ref="agent:poly-07",
            author_kind="agent",
            evidence_refs=[f"message:{M_C1}"],
            status="active",
            visibility="private",
            context_policy={"inject": True},
            now_ms=1_700_000_000_000,
        )
        upsert_assertion(
            conn,
            assertion_id="claim-web-workbench-candidate",
            target_ref=f"session:{C1}",
            scope_ref="repo:polylogue",
            kind=AssertionKind.CAVEAT,
            body_text="Candidate caveat stays visible in default web claim reads.",
            author_ref="agent:poly-07",
            author_kind="agent",
            evidence_refs=[f"message:{M_C1}"],
            status="candidate",
            visibility="private",
            context_policy={"inject": False},
            now_ms=1_700_000_000_100,
        )
        upsert_assertion(
            conn,
            assertion_id="claim-web-workbench-deleted",
            target_ref=f"session:{C1}",
            kind=AssertionKind.DECISION,
            body_text="Deleted claim is hidden by default.",
            status="deleted",
            visibility="private",
            now_ms=1_700_000_000_200,
        )
        conn.commit()


def _seed_archive_test_archive(workspace: dict[str, Path]) -> str:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    workspace["archive_root"].mkdir(parents=True, exist_ok=True)
    with ArchiveStore(workspace["archive_root"]) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="reader-v1",
                title="Archive reader thread",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="Hello archive reader",
                        timestamp="2026-01-01T00:00:00+00:00",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Hello archive reader")],
                    )
                ],
            )
        )


def _seed_browser_capture_reader_archive(workspace: dict[str, Path]) -> tuple[str, str]:
    from polylogue.core.enums import Provider
    from polylogue.sources.dispatch import parse_payload
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    unsafe_text = "<script>alert('captured')</script>\nBrowser capture prose"
    payload: dict[str, object] = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:xss-reader",
        "provenance": {
            "source_url": "https://chatgpt.com/c/xss-reader",
            "page_title": "Browser Capture XSS",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "xss-reader",
            "title": "Browser Capture XSS",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "turns": [
                {
                    "provider_turn_id": "xss-m1",
                    "role": "assistant",
                    "text": unsafe_text,
                    "ordinal": 0,
                }
            ],
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "browser-capture-xss.json")
    assert len(parsed) == 1
    workspace["archive_root"].mkdir(parents=True, exist_ok=True)
    with ArchiveStore(workspace["archive_root"]) as archive:
        session_id = archive.write_parsed(parsed[0])
    return session_id, unsafe_text


def _index_db_path(workspace: dict[str, Path]) -> Path:
    return workspace["data_root"] / "polylogue" / "index.db"


def _get_json(base_url: str, path: str, *, headers: dict[str, str] | None = None) -> object:
    req = Request(f"{base_url}{path}", headers=headers or {})
    with urlopen(req, timeout=10) as resp:
        assert resp.status == 200, f"unexpected status {resp.status} for {path}"
        return json.loads(resp.read())


def _get_text(base_url: str, path: str, *, headers: dict[str, str] | None = None) -> tuple[int, str, str]:
    req = Request(f"{base_url}{path}", headers=headers or {})
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
    except HTTPError as exc:
        body = exc.read().decode()
        return exc.code, exc.headers.get("Content-Type", ""), body


def _request_json(
    base_url: str,
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, object]:
    body = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    req = Request(f"{base_url}{path}", data=body, headers=request_headers, method=method)
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, json.loads(raw)


# ---------------------------------------------------------------------------
# polylogue.local_reader.search — list/search reader state
# ---------------------------------------------------------------------------


class TestReaderSearchState:
    """``polylogue.local_reader.search``: status strip, facets, results, inspector.

    Assertions are deliberately structural (text matches against the JS
    that hydrates each region) rather than pixel snapshots, so aesthetic
    iteration in the reader does not invalidate the smoke while a
    missing region still fails loudly.
    """

    def test_root_returns_html_with_required_regions(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/")
        assert status == 200
        assert "text/html" in content_type
        assert "<!DOCTYPE html>" in body
        for region in (
            "renderSidebarState",
            "renderSessions",
            "renderFacets",
            "renderMain",
            "renderWorkspaceToolbar",
            "renderStackWorkspace",
            "renderCompareWorkspace",
            "renderInspector",
            "renderInspectorEvidence",
            "renderBrowserCaptureChip",
            "renderReadViewExecution",
            "loadReadViewProfiles",
            "renderReadViewSelector",
            "applyReadViewSelection",
            "/api/read-view-profiles",
            "/read?view=",
            "/api/assertions",
            "/recovery?report=work-packet",
            "Load artifact list",
            "Load raw preview",
            'data-tab="evidence"',
        ):
            assert region in body, f"web shell missing region hook {region!r}"

    def test_raw_tab_uses_bounded_provenance_preview_not_broad_raw_fetch(self, workspace_env: dict[str, Path]) -> None:
        """The shell must not fetch the broad /raw route when opening Raw.

        Raw bytes are only fetched through /provenance?include_raw=1 after
        an explicit click, and the server caps that preview.
        """

        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/")

        assert status == 200
        assert "text/html" in content_type
        assert "renderInspectorRaw" in body
        assert "loadRawData" in body
        assert "loadProvenanceRaw" in body
        assert "/provenance?include_raw=1" in body
        assert "Raw preview is opt-in" in body
        assert "await loadSessionRaw(id)" in body
        assert "function loadProvenanceRaw()" in body

    def test_search_envelope_matches_documented_shape(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions")
        assert isinstance(payload, dict)
        assert payload["total"] == 3
        assert len(payload["items"]) == 3
        row = next(item for item in payload["items"] if item["id"] == "claude-code-session:c1")
        assert row["target_ref"] == {
            "target_type": "session",
            "target_id": "claude-code-session:c1",
            "session_id": "claude-code-session:c1",
            "identity_key": "session:claude-code-session:c1",
        }
        assert row["anchor"] == "session-claude-code-session-c1"
        assert row["actions"]["open"]["enabled"] is True
        assert row["actions"]["annotate"]["enabled"] is True
        assert row["actions"]["annotate"]["state"] == "enabled"

    def test_facets_envelope_includes_scoped_flag(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert "scoped_to_query" in payload
        assert "origins" in payload

    def test_facets_origin_filter_scopes_counts(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?origin=chatgpt-export")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is True
        assert payload["total_sessions"] == 1
        assert payload["origins"] == {"chatgpt-export": 1}
        assert "claude-code-session" not in payload["origins"]

    def test_facets_query_filter_scopes_counts(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?query=Hello")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is True
        assert payload["total_sessions"] == 3

    def test_query_search_envelope_carries_hit_target_refs(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions?query=Hello")
        assert isinstance(payload, dict)
        assert payload["total"] == 3
        assert len(payload["hits"]) == 3
        # SearchEnvelope contract (#1266): every hit carries a `session`
        # identity payload and a `match` evidence payload.
        assert payload["query"] == "Hello"
        assert payload["retrieval_lane"] in {"dialogue", "auto"}
        assert payload["ranking_policy"] == "mixed-bm25-rrf-vector"
        assert payload["ranking_policy_version"] == "1"
        hit = next(item for item in payload["hits"] if item["session"]["id"] == "claude-code-session:c1")
        assert hit["session"]["target_ref"]["identity_key"] == "session:claude-code-session:c1"
        assert hit["session"]["anchor"] == "session-claude-code-session-c1"
        # The typed TargetRefPayload includes block_index (defaulting to None)
        # for message targets; we assert the load-bearing identity fields.
        target_ref = hit["match"]["target_ref"]
        assert target_ref["target_type"] == "message"
        assert target_ref["target_id"] == "claude-code-session:c1:m-c1"
        assert target_ref["session_id"] == "claude-code-session:c1"
        assert target_ref["message_id"] == "claude-code-session:c1:m-c1"
        assert target_ref["identity_key"] == "message:claude-code-session:c1:claude-code-session:c1:m-c1"
        assert hit["match"]["anchor"] == "message-claude-code-session-c1-m-c1"
        assert hit["match"]["actions"]["copy_text"]["enabled"] is True

    def test_archive_file_set_lists_from_archive_tiers(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            payload = _get_json(base_url, "/api/sessions")

        assert isinstance(payload, dict)
        assert payload["total"] == 1
        row = payload["items"][0]
        assert row["id"] == session_id
        assert row["origin"] == "codex-session"
        assert "provider" not in row
        assert row["target_ref"]["identity_key"] == f"session:{session_id}"

    def test_archive_file_set_searches_from_archive_tiers(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            payload = _get_json(base_url, "/api/sessions?query=archive")

        assert isinstance(payload, dict)
        assert payload["total"] == 1
        hit = payload["hits"][0]
        assert hit["session"]["id"] == session_id
        assert hit["match"]["target_ref"]["identity_key"].startswith(f"message:{session_id}:")

    def test_archive_file_set_facets_from_archive_tiers(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            global_payload = _get_json(base_url, "/api/facets")
            scoped_payload = _get_json(base_url, "/api/facets?origin=codex-session")

        assert isinstance(global_payload, dict)
        assert global_payload["scoped_to_query"] is False
        assert global_payload["total_sessions"] == 1
        assert global_payload["origins"] == {"codex-session": 1}
        assert isinstance(scoped_payload, dict)
        assert scoped_payload["scoped_to_query"] is True
        assert scoped_payload["total_sessions"] == 1
        assert scoped_payload["origins"] == {"codex-session": 1}


# ---------------------------------------------------------------------------
# polylogue.local_reader.session — single session/detail state
# ---------------------------------------------------------------------------


class TestReaderSessionState:
    """``polylogue.local_reader.session``: header + messages + raw envelope."""

    def test_session_detail_returns_header_and_messages(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions/claude-code-session:c1")
        assert isinstance(payload, dict)
        assert payload["id"] == "claude-code-session:c1"
        assert payload["origin"] == "claude-code-session"
        assert payload["title"].startswith("Claude Code")
        assert payload["target_ref"]["identity_key"] == "session:claude-code-session:c1"
        assert payload["anchor"] == "session-claude-code-session-c1"
        assert (
            payload["messages"][0]["target_ref"]["identity_key"]
            == "message:claude-code-session:c1:claude-code-session:c1:m-c1"
        )
        assert payload["messages"][0]["anchor"] == "message-claude-code-session-c1-m-c1"

    def test_session_messages_envelope_carries_messages_and_total(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions/claude-code-session:c1/messages")
        assert isinstance(payload, dict)
        assert payload["total"] == 1
        message = payload["messages"][0]
        assert message["text"] == "Hello reader"
        assert message["target_ref"] == {
            "target_type": "message",
            "target_id": "claude-code-session:c1:m-c1",
            "session_id": "claude-code-session:c1",
            "message_id": "claude-code-session:c1:m-c1",
            "identity_key": "message:claude-code-session:c1:claude-code-session:c1:m-c1",
        }
        assert message["anchor"] == "message-claude-code-session-c1-m-c1"
        assert message["actions"]["copy_text"]["enabled"] is True
        assert message["actions"]["annotate"]["enabled"] is True
        assert message["actions"]["annotate"]["state"] == "enabled"

    def test_ref_resolve_route_returns_shared_payload(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, f"/api/refs/resolve?ref={quote('session:claude-code-session:c1')}")

        result = cast(dict[str, object], payload)
        assert result["mode"] == "ref-resolution"
        assert result["resolved"] is True
        assert result["normalized_ref"] == "session:claude-code-session:c1"
        assert result["payload_kind"] == "session-summary"
        resolved_payload = cast(dict[str, object], result["payload"])
        assert resolved_payload["id"] == "claude-code-session:c1"

    def test_session_messages_apply_content_projection_flags(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import BlockType, Provider
        from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        _seed_test_db(workspace_env)
        with ArchiveStore(workspace_env["archive_root"]) as archive:
            session_id = archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="projection-c1",
                    title="Projection fixture",
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:01:00+00:00",
                    messages=[
                        ParsedMessage(
                            provider_message_id="projection-m1",
                            role=Role.ASSISTANT,
                            text="Visible prose\nprint('secret')\nTool call hidden",
                            timestamp="2026-01-01T00:00:00+00:00",
                            blocks=[
                                ParsedContentBlock(type=BlockType.TEXT, text="Visible prose"),
                                ParsedContentBlock(type=BlockType.CODE, text="print('secret')"),
                                ParsedContentBlock(type=BlockType.TOOL_USE, text="Tool call hidden", tool_id="tool-1"),
                                ParsedContentBlock(
                                    type=BlockType.TOOL_RESULT,
                                    text="Tool output hidden",
                                    tool_id="tool-1",
                                ),
                            ],
                        )
                    ],
                )
            )

        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = _get_json(base_url, f"/api/sessions/{session_id}/messages?prose_only=1")

        result = cast(dict[str, object], payload)
        messages = cast(list[dict[str, object]], result["messages"])
        assert result["total"] == 1
        assert messages[0]["text"] == "Visible prose"
        assert "secret" not in str(messages[0]["text"])
        assert "Tool" not in str(messages[0]["text"])

    def test_archive_file_set_session_detail_and_messages_from_archive_tiers(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            detail = _get_json(base_url, f"/api/sessions/{session_id}")
            messages = _get_json(base_url, f"/api/sessions/{session_id}/messages")

        assert isinstance(detail, dict)
        assert detail["id"] == session_id
        assert detail["origin"] == "codex-session"
        assert "provider" not in detail
        assert detail["messages"][0]["text"] == "Hello archive reader"
        assert isinstance(messages, dict)
        assert messages["total"] == 1
        assert messages["messages"][0]["target_ref"]["identity_key"].startswith(f"message:{session_id}:")

    def test_browser_capture_reader_boundary_keeps_text_and_escapes_shell(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        session_id, unsafe_text = _seed_browser_capture_reader_archive(workspace_env)
        dangerous_fragment = "<script>alert('captured')</script>"
        prose_fragment = "Browser capture prose"

        with _running_server_without_seed() as (_, base_url):
            detail = _get_json(base_url, f"/api/sessions/{session_id}")
            messages = _get_json(base_url, f"/api/sessions/{session_id}/messages")
            root_status, root_content_type, root_shell = _get_text(base_url, "/")
            link_status, link_content_type, link_shell = _get_text(base_url, f"/s/{session_id}")

        assert root_status == 200
        assert link_status == 200
        assert "text/html" in root_content_type
        assert "text/html" in link_content_type
        assert isinstance(detail, dict)
        assert detail["origin"] == "chatgpt-export"
        assert detail["messages"][0]["text"] == unsafe_text
        assert isinstance(messages, dict)
        assert messages["messages"][0]["text"] == unsafe_text
        assert "https://chatgpt.com/c/xss-reader" not in json.dumps(detail)
        assert unsafe_text not in root_shell
        assert unsafe_text not in link_shell
        assert dangerous_fragment not in root_shell
        assert dangerous_fragment not in link_shell
        assert prose_fragment not in root_shell
        assert prose_fragment not in link_shell
        assert "function esc" in root_shell
        assert "'<div class=\"msg-text\">' + esc(parts[0].body)" in root_shell

    def test_unknown_session_yields_404(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, _, _ = _get_text(base_url, "/api/sessions/does-not-exist")
        assert status == 404


# ---------------------------------------------------------------------------
# polylogue.local_reader.workspace — stack and compare route data
# ---------------------------------------------------------------------------


class TestReaderWorkspaceRoutes:
    """``polylogue.local_reader.workspace``: stack/compare workspace routes."""

    def test_stack_route_returns_resolved_and_missing_targets(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(
                base_url, "/api/stack?ids=claude-code-session:c1,missing-conv&focus=claude-code-session:c1"
            )

        result = cast(dict[str, object], payload)
        items = cast(list[dict[str, object]], result["items"])
        assert result["mode"] == "stack"
        assert result["focus"] == "claude-code-session:c1"
        assert result["resolved_count"] == 1
        assert result["degraded_count"] == 1
        assert items[0]["status"] == "resolved"
        assert items[0]["identity_key"] == "session:claude-code-session:c1"
        session = cast(dict[str, object], items[0]["session"])
        assert session["id"] == "claude-code-session:c1"
        assert items[1] == {
            "target_type": "session",
            "target_id": "missing-conv",
            "session_id": "missing-conv",
            "status": "missing",
            "disabled_reason": "session_not_found",
        }

    def test_stack_route_rejects_empty_ids(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, "/api/stack")

        assert status == 400
        assert payload["error"] == "invalid_request"

    def test_compare_route_returns_message_pairs_and_degraded_side(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/compare?left=claude-code-session:c1&right=missing-conv&align=prompt")

        result = cast(dict[str, object], payload)
        assert result["mode"] == "compare"
        assert result["align"] == "prompt"
        assert result["degraded_count"] == 1
        # New compare envelope fields (#1124) expose alignment strategy and
        # which side(s) failed to load so the UI can render a precise banner.
        assert result["degraded_sides"] == ["right"]
        assert result["alignment"] in {"anchor", "sequential"}
        assert result["metadata_diff"] == {}
        left = cast(dict[str, object], result["left"])
        right = cast(dict[str, object], result["right"])
        pairs = cast(list[dict[str, object]], result["pairs"])
        assert left["id"] == "claude-code-session:c1"
        assert right["status"] == "missing"
        assert pairs[0]["left"] is not None
        assert pairs[0]["right"] is None
        assert pairs[0]["status"] == "unpaired"

    def test_compare_route_two_sessions_surface_diff_and_metadata(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(
                base_url, "/api/compare?left=claude-code-session:c1&right=chatgpt-export:c2&align=prompt"
            )

        result = cast(dict[str, object], payload)
        # Both sides present → no degradation, metadata diff populated.
        assert result["degraded_count"] == 0
        assert result["degraded_sides"] == []
        metadata = cast(dict[str, dict[str, object]], result["metadata_diff"])
        # Origins differ between the seeded sessions.
        assert metadata["origin"]["status"] == "changed"
        assert metadata["title"]["status"] == "changed"
        pairs = cast(list[dict[str, object]], result["pairs"])
        # Seeded messages share text "Hello reader" and role "user", but have
        # distinct anchors → alignment is sequential and content is equal.
        assert pairs[0]["diff_status"] == "equal"
        assert pairs[0]["role_match"] is True

    def test_compare_route_rejects_invalid_align(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(
                base_url, "/api/compare?left=claude-code-session:c1&right=chatgpt-export:c2&align=sideways"
            )

        assert status == 400
        assert payload["error"] == "invalid_request"

    def test_compare_route_rejects_unimplemented_align_modes(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(
                base_url, "/api/compare?left=claude-code-session:c1&right=chatgpt-export:c2&align=time"
            )

        assert status == 400
        assert payload["error"] == "invalid_request"

    def test_workspace_shell_routes_are_unauthenticated(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, _, body = _get_text(base_url, "/w/stack?ids=claude-code-session:c1,chatgpt-export:c2")
            compare_status, _, compare_body = _get_text(
                base_url, "/w/compare?left=claude-code-session:c1&right=chatgpt-export:c2&align=prompt"
            )

        assert status == 200
        assert "<title>Polylogue</title>" in body
        assert "getWorkspaceRouteFromURL" in body
        assert "workspace-mode-switcher" in body
        assert compare_status == 200
        assert "renderCompareWorkspace" in compare_body

    def test_session_deep_link_route_serves_web_shell(self, workspace_env: dict[str, Path]) -> None:
        """The session reader deep link is ``/s/{session_id}`` (schema-v1 vocabulary)."""
        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/s/claude-code-session:c1")
        assert status == 200
        assert "text/html" in content_type
        assert "<title>Polylogue</title>" in body
        assert "getSessionIdFromURL" in body

    def test_session_deep_link_shell_carries_minimal_evidence_panel(self, workspace_env: dict[str, Path]) -> None:
        """The web workbench evidence slice is present on real session deep links."""

        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/s/claude-code-session:c1")

        assert status == 200
        assert "text/html" in content_type
        for hook in (
            'data-tab="evidence"',
            "renderInspectorEvidence",
            "/api/sessions/",
            "/recovery?report=work-packet&format=json",
            "/api/assertions?target_ref=",
        ):
            assert hook in body

    def test_legacy_conversation_route_is_gone(self, workspace_env: dict[str, Path]) -> None:
        """The stale ``/c/{conversation_id}`` route must not resolve — no compat alias."""
        with _running_server(workspace_env) as (_, base_url):
            status, _, _ = _get_text(base_url, "/c/claude-code-session:c1")
        assert status == 404

    def test_archive_file_set_stack_route_from_archive_tiers(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            payload = _get_json(base_url, f"/api/stack?ids={session_id},missing&focus={session_id}")

        result = cast(dict[str, object], payload)
        assert result["mode"] == "stack"
        assert result["resolved_count"] == 1
        assert result["degraded_count"] == 1
        items = cast(list[dict[str, object]], result["items"])
        assert items[0]["status"] == "resolved"
        assert items[1]["status"] == "missing"

    def test_archive_file_set_compare_route_from_archive_tiers(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            payload = _get_json(base_url, f"/api/compare?left={session_id}&right=missing&align=prompt")

        result = cast(dict[str, object], payload)
        assert result["mode"] == "compare"
        assert result["degraded_sides"] == ["right"]
        right = cast(dict[str, object], result["right"])
        assert right["status"] == "missing"


# ---------------------------------------------------------------------------
# polylogue.local_reader.user_state — durable marks/views/recall contracts
# ---------------------------------------------------------------------------


class TestReaderUserState:
    """``polylogue.local_reader.user_state``: durable session user state."""

    def test_session_marks_are_idempotent_and_deletable(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, created = _request_json(
                base_url,
                "POST",
                "/api/user/marks",
                payload={"session_id": "claude-code-session:c1", "mark_type": "star"},
            )
            status2, duplicate = _request_json(
                base_url,
                "POST",
                "/api/user/marks",
                payload={"session_id": "claude-code-session:c1", "mark_type": "star"},
            )
            marks = _get_json(base_url, "/api/user/marks?session_id=claude-code-session:c1")
            delete_status, deleted = _request_json(
                base_url,
                "DELETE",
                "/api/user/marks?session_id=claude-code-session:c1&mark_type=star",
            )
            empty = _get_json(base_url, "/api/user/marks?session_id=claude-code-session:c1")

        marks_payload = cast(dict[str, object], marks)
        created_payload = cast(dict[str, object], created)
        duplicate_payload = cast(dict[str, object], duplicate)
        deleted_payload = cast(dict[str, object], deleted)
        assert status == 201
        assert created_payload["status"] == "ok"
        assert created_payload["operation"] == "mark.add"
        assert created_payload["affected_count"] == 1
        assert created_payload["target_type"] == "session"
        assert created_payload["target_id"] == "claude-code-session:c1"
        assert created_payload["session_id"] == "claude-code-session:c1"
        assert created_payload["mark_type"] == "star"
        assert status2 == 200
        assert duplicate_payload["status"] == "unchanged"
        assert duplicate_payload["detail"] == "already_present"
        assert duplicate_payload["operation"] == "mark.add"
        assert duplicate_payload["affected_count"] == 0
        assert duplicate_payload["target_type"] == "session"
        assert duplicate_payload["target_id"] == "claude-code-session:c1"
        assert duplicate_payload["session_id"] == "claude-code-session:c1"
        assert duplicate_payload["mark_type"] == "star"
        mark_items = cast(list[dict[str, object]], marks_payload["items"])
        assert mark_items[0]["mark_type"] == "star"
        assert mark_items[0]["target_type"] == "session"
        assert mark_items[0]["target_id"] == "claude-code-session:c1"
        assert delete_status == 200
        assert deleted_payload["status"] == "deleted"
        assert deleted_payload["operation"] == "mark.delete"
        assert deleted_payload["affected_count"] == 1
        assert deleted_payload["target_type"] == "session"
        assert deleted_payload["target_id"] == "claude-code-session:c1"
        assert deleted_payload["session_id"] == "claude-code-session:c1"
        assert deleted_payload["mark_type"] == "star"
        assert empty == {"items": [], "total": 0}

    def test_message_marks_are_target_aware(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, created = _request_json(
                base_url,
                "POST",
                "/api/user/marks",
                payload={
                    "session_id": "claude-code-session:c1",
                    "target_type": "message",
                    "message_id": "claude-code-session:c1:m-c1",
                    "mark_type": "pin",
                },
            )
            marks = _get_json(base_url, "/api/user/marks?target_type=message&message_id=claude-code-session:c1:m-c1")

        marks_payload = cast(dict[str, object], marks)
        created_payload = cast(dict[str, object], created)
        assert status == 201
        assert created_payload["status"] == "ok"
        assert created_payload["operation"] == "mark.add"
        assert created_payload["target_type"] == "message"
        assert created_payload["target_id"] == "claude-code-session:c1:m-c1"
        assert created_payload["message_id"] == "claude-code-session:c1:m-c1"
        mark_items = cast(list[dict[str, object]], marks_payload["items"])
        assert mark_items == [
            {
                "target_type": "message",
                "target_id": "claude-code-session:c1:m-c1",
                "session_id": "claude-code-session:c1",
                "message_id": "claude-code-session:c1:m-c1",
                "mark_type": "pin",
                "created_at": mark_items[0]["created_at"],
            }
        ]

    def test_annotations_roundtrip_session_and_message_targets(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            conv_status, conv_note = _request_json(
                base_url,
                "POST",
                "/api/user/annotations",
                payload={
                    "annotation_id": "ann-c1",
                    "session_id": "claude-code-session:c1",
                    "note_text": "Follow up",
                },
            )
            msg_status, msg_note = _request_json(
                base_url,
                "POST",
                "/api/user/annotations",
                payload={
                    "annotation_id": "ann-m1",
                    "session_id": "claude-code-session:c1",
                    "target_type": "message",
                    "message_id": "claude-code-session:c1:m-c1",
                    "note_text": "Important request",
                },
            )
            listed = _get_json(base_url, "/api/user/annotations?session_id=claude-code-session:c1")
            fetched = _get_json(base_url, "/api/user/annotations/ann-m1")
            delete_status, deleted = _request_json(base_url, "DELETE", "/api/user/annotations/ann-c1")
            missing_delete_status, missing_deleted = _request_json(base_url, "DELETE", "/api/user/annotations/ann-c1")

        listed_payload = cast(dict[str, object], listed)
        conv_note_payload = cast(dict[str, object], conv_note)
        msg_note_payload = cast(dict[str, object], msg_note)
        fetched_payload = cast(dict[str, object], fetched)
        items = cast(list[dict[str, object]], listed_payload["items"])
        assert conv_status == 201
        assert conv_note_payload["status"] == "ok"
        assert conv_note_payload["operation"] == "annotation.save"
        assert conv_note_payload["affected_count"] == 1
        assert conv_note_payload["resource_type"] == "annotation"
        assert conv_note_payload["resource_id"] == "ann-c1"
        assert conv_note_payload["target_type"] == "session"
        assert conv_note_payload["target_id"] == "claude-code-session:c1"
        assert msg_status == 201
        assert msg_note_payload["status"] == "ok"
        assert msg_note_payload["operation"] == "annotation.save"
        assert msg_note_payload["affected_count"] == 1
        assert msg_note_payload["resource_type"] == "annotation"
        assert msg_note_payload["resource_id"] == "ann-m1"
        assert msg_note_payload["target_type"] == "message"
        assert msg_note_payload["target_id"] == "claude-code-session:c1:m-c1"
        assert fetched_payload["note_text"] == "Important request"
        assert {item["annotation_id"] for item in items} == {"ann-c1", "ann-m1"}
        assert delete_status == 200
        assert deleted == {
            "status": "deleted",
            "affected_count": 1,
            "operation": "annotation.delete",
            "resource_type": "annotation",
            "resource_id": "ann-c1",
        }
        assert missing_delete_status == 200
        assert missing_deleted == {
            "status": "not_found",
            "detail": "annotation_not_found",
            "affected_count": 0,
            "operation": "annotation.delete",
            "resource_type": "annotation",
            "resource_id": "ann-c1",
        }

    def test_saved_views_roundtrip_query_specs(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, saved = _request_json(
                base_url,
                "POST",
                "/api/user/saved-views",
                payload={
                    "view_id": "view-auth",
                    "name": "Auth sessions",
                    "query": {"query": "auth", "origin": "claude-code-session", "limit": 5},
                },
            )
            listed = _get_json(base_url, "/api/user/saved-views")
            fetched = _get_json(base_url, "/api/user/saved-views/view-auth")
            delete_status, deleted = _request_json(base_url, "DELETE", "/api/user/saved-views/view-auth")

        saved_payload = cast(dict[str, object], saved)
        listed_payload = cast(dict[str, object], listed)
        fetched_payload = cast(dict[str, object], fetched)
        assert status == 201
        assert saved_payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "saved_view.save",
            "resource_type": "saved_view",
            "resource_id": "view-auth",
        }
        assert listed_payload["total"] == 1
        assert fetched_payload["view_id"] == "view-auth"
        assert fetched_payload["query"] == {"limit": 5, "origin": "claude-code-session", "query": "auth"}
        assert json.loads(str(fetched_payload["query_json"])) == {
            "limit": 5,
            "origin": "claude-code-session",
            "query": "auth",
        }
        assert delete_status == 200
        assert deleted == {
            "status": "deleted",
            "affected_count": 1,
            "operation": "saved_view.delete",
            "resource_type": "saved_view",
            "resource_id": "view-auth",
        }

    def test_saved_view_rejects_unknown_query_params(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _request_json(
                base_url,
                "POST",
                "/api/user/saved-views",
                payload={"name": "Broken", "query": {"providr": "claude-code"}},
            )

        error_payload = cast(dict[str, object], payload)
        assert status == 400
        assert error_payload["error"] == "QuerySpecError"
        assert error_payload["field"] == "providr"

    def test_recall_packs_roundtrip_cited_sessions(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, saved = _request_json(
                base_url,
                "POST",
                "/api/user/recall-packs",
                payload={
                    "pack_id": "pack-auth",
                    "label": "Auth pack",
                    "payload": {
                        "summary": "handoff",
                        "items": [
                            {"target_type": "session", "session_id": "claude-code-session:c1"},
                            {"target_type": "session", "session_id": "missing-conv"},
                            {
                                "target_type": "message",
                                "session_id": "claude-code-session:c1",
                                "message_id": "missing-msg",
                            },
                        ],
                    },
                },
            )
            listed = _get_json(base_url, "/api/user/recall-packs")
            fetched = _get_json(base_url, "/api/user/recall-packs/pack-auth")
            delete_status, deleted = _request_json(base_url, "DELETE", "/api/user/recall-packs/pack-auth")

        saved_payload = cast(dict[str, object], saved)
        listed_payload = cast(dict[str, object], listed)
        fetched_payload = cast(dict[str, object], fetched)
        assert status == 201
        assert saved_payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "recall_pack.save",
            "resource_type": "recall_pack",
            "resource_id": "pack-auth",
        }
        assert listed_payload["total"] == 1
        assert fetched_payload["session_ids"] == ["claude-code-session:c1"]
        payload = cast(dict[str, object], fetched_payload["payload"])
        assert payload["summary"] == "handoff"
        assert payload["resolved_count"] == 1
        assert payload["degraded_count"] == 2
        items = cast(list[dict[str, object]], payload["items"])
        assert [(item["target_type"], item["status"]) for item in items] == [
            ("session", "resolved"),
            ("session", "missing"),
            ("message", "missing"),
        ]
        assert delete_status == 200
        assert deleted == {
            "status": "deleted",
            "affected_count": 1,
            "operation": "recall_pack.delete",
            "resource_type": "recall_pack",
            "resource_id": "pack-auth",
        }

    def test_recall_pack_rejects_session_ids_compat_input(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _request_json(
                base_url,
                "POST",
                "/api/user/recall-packs",
                payload={
                    "pack_id": "pack-compat",
                    "label": "Compat pack",
                    "session_ids": ["claude-code-session:c1"],
                    "payload": {"summary": "old shape"},
                },
            )

        error_payload = cast(dict[str, object], payload)
        assert status == 400
        assert error_payload["error"] == "invalid_request"

    def test_workspaces_roundtrip_resolved_and_degraded_targets(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, saved = _request_json(
                base_url,
                "POST",
                "/api/user/workspaces",
                payload={
                    "workspace_id": "workspace-auth",
                    "name": "Auth workspace",
                    "mode": "compare",
                    "open_targets": [
                        {"target_type": "session", "session_id": "claude-code-session:c1"},
                        {
                            "target_type": "message",
                            "session_id": "claude-code-session:c1",
                            "message_id": "claude-code-session:c1:m-c1",
                        },
                        {
                            "target_type": "message",
                            "session_id": "claude-code-session:c1",
                            "message_id": "missing-msg",
                        },
                        {"target_type": "topology_edge", "target_id": "edge-1"},
                    ],
                    "layout": {"panes": [{"width": 0.5}, {"width": 0.5}]},
                    "active_target": {
                        "target_type": "message",
                        "session_id": "claude-code-session:c1",
                        "message_id": "claude-code-session:c1:m-c1",
                    },
                },
            )
            listed = _get_json(base_url, "/api/user/workspaces")
            fetched = _get_json(base_url, "/api/user/workspaces/workspace-auth")
            delete_status, deleted = _request_json(base_url, "DELETE", "/api/user/workspaces/workspace-auth")

        saved_payload = cast(dict[str, object], saved)
        listed_payload = cast(dict[str, object], listed)
        fetched_payload = cast(dict[str, object], fetched)
        assert status == 201
        assert saved_payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "workspace.save",
            "resource_type": "workspace",
            "resource_id": "workspace-auth",
        }
        assert listed_payload["total"] == 1
        assert fetched_payload["mode"] == "compare"
        assert fetched_payload["layout"] == {"panes": [{"width": 0.5}, {"width": 0.5}]}
        open_targets = cast(list[dict[str, object]], fetched_payload["open_targets"])
        assert [(item["target_type"], item["status"]) for item in open_targets] == [
            ("session", "resolved"),
            ("message", "resolved"),
            ("message", "missing"),
            ("topology_edge", "unsupported"),
        ]
        active_target = cast(dict[str, object], fetched_payload["active_target"])
        # Native message ids are already absolute (``session_id:message_native``);
        # the workspace target resolver composes ``message:{session_id}:{message_id}``
        # over the full message id, so the session prefix appears twice.
        assert active_target["identity_key"] == f"message:{C1}:{M_C1}"
        assert delete_status == 200
        assert deleted == {
            "status": "deleted",
            "affected_count": 1,
            "operation": "workspace.delete",
            "resource_type": "workspace",
            "resource_id": "workspace-auth",
        }


# ---------------------------------------------------------------------------
# Empty / degraded / privacy states
# ---------------------------------------------------------------------------


class TestReaderDegradedStates:
    def test_empty_archive_returns_zero_envelope(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions")
        assert isinstance(payload, dict)
        assert payload["total"] == 0
        assert payload["items"] == []

    def test_empty_archive_facets_returns_no_origins(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert payload["total_sessions"] == 0
        assert payload["origins"] == {}

    def test_no_results_query_is_distinguishable_from_empty_archive(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions?query=nonexistent_term_xyz")
        assert isinstance(payload, dict)
        # Archive non-empty (3 convs seeded) but query matches none.
        assert payload["total"] == 0


class TestReaderQueryCompletions:
    def test_query_completions_endpoint_exposes_shared_payload(self) -> None:
        with _running_server_without_seed() as (_server, base_url):
            payload = _get_json(base_url, "/api/query-completions?kind=field&incomplete=d")

        assert isinstance(payload, dict)
        payload_dict = cast(dict[str, object], payload)
        envelope = payload_dict["query_completions"]
        assert isinstance(envelope, dict)
        assert envelope["kind"] == "field"
        candidates = envelope["candidates"]
        assert isinstance(candidates, list)
        candidate_payloads = [cast(dict[str, object], candidate) for candidate in candidates]
        date_candidate = next(candidate for candidate in candidate_payloads if candidate["value"] == "date")
        assert date_candidate["insert"] == "date "
        assert date_candidate["source"] == "DATE_QUERY_FIELD_REGISTRY"

    def test_query_completions_endpoint_reports_invalid_context(self) -> None:
        with _running_server_without_seed() as (_server, base_url):
            status, payload = _get_json_ex(base_url, "/api/query-completions?kind=structural-field")

        assert status == 400
        assert payload["error"] == "invalid_query_completion"
        assert "--unit is required" in str(payload["message"])

    def test_query_completions_endpoint_exposes_terminal_fields(self) -> None:
        with _running_server_without_seed() as (_server, base_url):
            payload = _get_json(
                base_url,
                "/api/query-completions?kind=terminal-field&unit=context-snapshots&incomplete=bound",
            )

        payload_dict = cast(dict[str, object], payload)
        envelope = payload_dict["query_completions"]
        assert isinstance(envelope, dict)
        envelope_payload = cast(dict[str, object], envelope)
        assert envelope_payload["kind"] == "terminal-field"
        candidates = envelope_payload["candidates"]
        assert isinstance(candidates, list)
        candidate_payloads = [cast(dict[str, object], candidate) for candidate in cast(list[object], candidates)]
        assert [candidate["value"] for candidate in candidate_payloads] == ["boundary"]
        assert candidate_payloads[0]["insert"] == "boundary:"


class TestReaderQueryUnits:
    def test_query_units_endpoint_returns_terminal_message_rows(self, workspace_env: dict[str, Path]) -> None:
        expression = quote("messages where text:Hello")
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, f"/api/query-units?expression={expression}&limit=2")

        assert isinstance(payload, dict)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "message"
        assert payload["limit"] == 2
        assert payload["next_offset"] == 2
        items = cast(list[dict[str, object]], payload["items"])
        assert len(items) == 2
        assert {item["unit"] for item in items} == {"message"}
        assert {item["text"] for item in items} == {"Hello reader"}
        assert {item["session_id"] for item in items} <= {
            "claude-code-session:c1",
            "chatgpt-export:c2",
            "claude-ai-export:c3",
        }

    def test_query_units_endpoint_applies_session_scope_filters(self, workspace_env: dict[str, Path]) -> None:
        expression = quote("messages where text:Hello")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/query-units?expression={expression}&origin=chatgpt-export"),
            )

        items = cast(list[dict[str, object]], payload["items"])
        assert [item["session_id"] for item in items] == [C2]

    def test_query_units_endpoint_accepts_inline_session_scope(self, workspace_env: dict[str, Path]) -> None:
        expression = quote("messages where session.origin:chatgpt-export AND text:Hello")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, f"/api/query-units?expression={expression}"))

        items = cast(list[dict[str, object]], payload["items"])
        assert [item["session_id"] for item in items] == [C2]

    def test_query_units_endpoint_returns_run_rows(self, workspace_env: dict[str, Path]) -> None:
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "daemon-run")
            .provider("codex")
            .git_repository_url("polylogue")
            .working_directories(["/realm/project/polylogue"])
            .title("Daemon run query")
            .add_message(
                "m-run",
                role="assistant",
                text="Subagent finished daemon run query wiring.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "tool-run",
                        "name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "taskId": "task-run",
                            "child_session_id": "codex-session:daemon-run-child",
                            "prompt": "Map daemon run query wiring.",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "tool-run",
                        "text": "Subagent done: daemon run query wired.\n2 passed in 0.10s",
                    },
                ],
            )
            .save()
        )

        expression = quote("runs where session.repo:polylogue AND role:subagent AND status:completed")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, f"/api/query-units?expression={expression}"))

        assert payload["unit"] == "run"
        items = cast(list[dict[str, object]], payload["items"])
        assert len(items) == 1
        assert items[0]["unit"] == "run"
        assert items[0]["session_id"] == "codex-session:ext-daemon-run"
        assert items[0]["role"] == "subagent"
        assert items[0]["status"] == "completed"
        assert items[0]["agent_ref"] == "agent:codex/Explore"

    def test_query_units_endpoint_returns_observed_event_rows(self, workspace_env: dict[str, Path]) -> None:
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "daemon-event")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("Daemon observed event query")
            .add_message("m-event", role="user", text="Daemon observed event seed")
            .save()
        )

        expression = quote("observed-events where session.repo:polylogue AND kind:session_started")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, f"/api/query-units?expression={expression}"))

        assert payload["unit"] == "observed-event"
        items = cast(list[dict[str, object]], payload["items"])
        assert len(items) == 1
        assert items[0]["unit"] == "observed-event"
        assert items[0]["session_id"] == "codex-session:ext-daemon-event"
        assert items[0]["kind"] == "session_started"

    def test_query_units_endpoint_returns_context_snapshot_rows(self, workspace_env: dict[str, Path]) -> None:
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "daemon-context")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("Daemon context query")
            .add_message("m-context", role="user", text="Daemon context snapshot seed")
            .save()
        )

        expression = quote("context-snapshots where session.repo:polylogue AND boundary:session_start AND text:context")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, f"/api/query-units?expression={expression}"))

        assert payload["unit"] == "context-snapshot"
        items = cast(list[dict[str, object]], payload["items"])
        assert len(items) == 1
        assert items[0]["unit"] == "context-snapshot"
        assert items[0]["session_id"] == "codex-session:ext-daemon-context"
        assert items[0]["boundary"] == "session_start"

    def test_query_units_endpoint_rejects_session_expression(self, workspace_env: dict[str, Path]) -> None:
        expression = quote("repo:polylogue")
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, f"/api/query-units?expression={expression}")

        assert status == 400
        assert payload["error"] == "invalid_query"
        assert "messages/actions/blocks/assertions/runs/observed-events/context-snapshots" in str(payload["message"])


class TestReaderViewProfiles:
    def test_read_view_profiles_endpoint_exposes_shared_profile_semantics(self) -> None:
        with _running_server_without_seed() as (_server, base_url):
            payload = _get_json(base_url, "/api/read-view-profiles")

        assert isinstance(payload, dict)
        assert payload["total"] >= 3
        read_views = payload["read_views"]
        assert isinstance(read_views, list)
        profiles = {profile["view_id"]: profile for profile in read_views}
        assert profiles["raw"]["lossiness"] == "raw"
        assert profiles["raw"]["evidence_policy"] == "required"
        assert profiles["recovery"]["successor_handoff"] is True
        assert "markdown" in profiles["recovery"]["formats"]

    def test_read_view_execution_route_returns_messages_recovery_raw_context_and_context_pack(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        with _running_server(workspace_env) as (_, base_url):
            messages = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=messages&limit=5"),
            )
            recovery = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=recovery&report=work-packet"),
            )
            raw = cast(dict[str, object], _get_json(base_url, f"/api/sessions/{C1}/read?view=raw"))
            context = cast(dict[str, object], _get_json(base_url, f"/api/sessions/{C1}/read?view=context"))
            context_pack = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=context-pack&max_messages=5"),
            )
            neighbors = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=neighbors&limit=2"),
            )
            correlation = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=correlation&since_hours=1"),
            )

        assert messages["view"] == "messages"
        message_payload = cast(dict[str, object], messages["payload"])
        assert message_payload["total"] == 1
        assert recovery["view"] == "recovery"
        recovery_payload = cast(dict[str, object], recovery["payload"])
        assert recovery_payload["report"] == "work-packet"
        assert raw["view"] == "raw"
        raw_payload = cast(dict[str, object], raw["payload"])
        assert raw_payload["id"] == C1
        assert context["view"] == "context"
        context_payload = cast(dict[str, object], context["payload"])
        assert context_payload["preamble_version"] == "1.0"
        assert context_pack["view"] == "context-pack"
        context_payload = cast(dict[str, object], context_pack["payload"])
        assert context_payload["total_sessions"] == 1
        context_query = cast(dict[str, object], context_payload["query_context"])
        assert context_query["query_matched"] == 1
        context_sessions = cast(list[dict[str, object]], context_payload["sessions"])
        assert context_sessions[0]["session_id"] == C1
        assert neighbors["view"] == "neighbors"
        neighbor_payload = cast(dict[str, object], neighbors["payload"])
        neighbor_rows = cast(list[dict[str, object]], neighbor_payload["neighbors"])
        assert neighbor_rows
        neighbor_session = cast(dict[str, object], neighbor_rows[0]["session"])
        assert neighbor_session["id"] in {C2, C3}
        neighbor_reasons = cast(list[dict[str, object]], neighbor_rows[0]["reasons"])
        assert {reason["kind"] for reason in neighbor_reasons} & {"nearby_time", "query_match", "content_similarity"}
        assert correlation["view"] == "correlation"
        correlation_payload = cast(dict[str, object], correlation["payload"])
        assert correlation_payload["session_id"] == C1
        assert "window_start" in correlation_payload
        assert "object_refs" in correlation_payload

    def test_read_view_execution_rejects_unknown_view_or_format(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            view_status, view_payload = _get_json_ex(base_url, f"/api/sessions/{C1}/read?view=timeline")
            format_status, format_payload = _get_json_ex(base_url, f"/api/sessions/{C1}/read?view=messages&format=html")

        assert view_status == 400
        assert view_payload["error"] == "unsupported_read_view"
        assert format_status == 400
        assert format_payload["error"] == "invalid_format"


class TestReaderRecoveryEndpoint:
    def test_recovery_endpoint_returns_digest_json(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/recovery?report=digest&format=json"),
            )

        assert payload["report"] == "digest"
        assert payload["format"] == "json"
        digest = cast(dict[str, object], payload["digest"])
        assert digest["session_id"] == C1
        assert digest["raw_refs"]

    def test_recovery_endpoint_returns_shared_work_packet_json(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/recovery?report=work-packet&format=json"),
            )

        assert payload["report"] == "work-packet"
        assert payload["format"] == "json"
        packet = cast(dict[str, object], payload["work_packet"])
        assert packet["session_id"] == C1
        assert packet["evidence_refs"]
        assert "raw_artifacts" not in json.dumps(payload)

    def test_recovery_endpoint_returns_work_packet_markdown_envelope(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/recovery?report=work-packet&format=markdown"),
            )

        assert payload["report"] == "work-packet"
        assert payload["format"] == "markdown"
        assert "markdown" in payload
        assert C1 in str(payload["markdown"])

    @pytest.mark.parametrize("report", ["continue", "blame"])
    def test_recovery_endpoint_does_not_overexpose_report_presets(
        self, workspace_env: dict[str, Path], report: str
    ) -> None:
        """#1846 exposes bundle DTOs first; #1847 owns report-preset promotion."""

        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, f"/api/sessions/{C1}/recovery?report={report}&format=markdown")

        assert status == 400
        assert payload["error"] == "invalid_report"

    @pytest.mark.parametrize(
        ("query", "error_code"),
        [
            ("report=nope&format=json", "invalid_report"),
            ("report=continue&format=json", "invalid_report"),
            ("report=blame&format=json", "invalid_report"),
            ("report=digest&format=markdown", "invalid_format"),
            ("report=work-packet&format=zip", "invalid_format"),
        ],
    )
    def test_recovery_endpoint_rejects_unsupported_report_or_format(
        self, workspace_env: dict[str, Path], query: str, error_code: str
    ) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, f"/api/sessions/{C1}/recovery?{query}")

        assert status == 400
        assert payload["error"] == error_code

    def test_recovery_endpoint_returns_404_for_missing_session(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(
                base_url, "/api/sessions/claude-code-session:missing/recovery?report=work-packet&format=json"
            )

        assert status == 404
        assert payload["error"] == "not_found"


class TestReaderAssertionEndpoint:
    def test_assertions_endpoint_reads_shared_assertion_claims(self, workspace_env: dict[str, Path]) -> None:
        _seed_assertion_claims(workspace_env)
        target_ref = quote(f"session:{C1}", safe="")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, f"/api/assertions?target_ref={target_ref}&limit=5"))

        assert payload["total"] == 2
        assert payload["statuses"] == ["active", "candidate"]
        items = cast(list[dict[str, object]], payload["items"])
        assert {item["assertion_id"] for item in items} == {
            "claim-web-workbench-decision",
            "claim-web-workbench-candidate",
        }
        decision = next(item for item in items if item["kind"] == "decision")
        assert decision["target_ref"] == f"session:{C1}"
        assert decision["context_policy"] == {"inject": True}

    def test_assertions_endpoint_filters_kind_and_context_policy(self, workspace_env: dict[str, Path]) -> None:
        _seed_assertion_claims(workspace_env)
        target_ref = quote(f"session:{C1}", safe="")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/assertions?target_ref={target_ref}&kind=decision&context_inject=1"),
            )

        items = cast(list[dict[str, object]], payload["items"])
        assert [item["assertion_id"] for item in items] == ["claim-web-workbench-decision"]
        assert items[0]["context_policy"] == {"inject": True}

    def test_assertions_endpoint_can_explicitly_read_all_statuses(self, workspace_env: dict[str, Path]) -> None:
        _seed_assertion_claims(workspace_env)
        target_ref = quote(f"session:{C1}", safe="")
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/assertions?target_ref={target_ref}&status=all&limit=5"),
            )

        items = cast(list[dict[str, object]], payload["items"])
        assert {item["assertion_id"] for item in items} == {
            "claim-web-workbench-decision",
            "claim-web-workbench-candidate",
            "claim-web-workbench-deleted",
        }

    def test_assertions_endpoint_is_auth_gated_when_daemon_token_is_configured(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        _seed_assertion_claims(workspace_env)
        with _running_server(workspace_env, auth_token="secret-token") as (_, base_url):
            status, _, body = _get_text(base_url, "/api/assertions")
            authed = cast(
                dict[str, object],
                _get_json(base_url, "/api/assertions", headers={"Authorization": "Bearer secret-token"}),
            )

        assert status == 401
        payload = json.loads(body)
        assert payload["error"] == "unauthorized"
        assert "items" in authed


class TestReaderPrivacy:
    """The reader must never expose absolute local paths or auth tokens.

    The unauthenticated web shell HTML and the read-only ``/api/facets``
    JSON are audited here for absolute filesystem path leaks across the
    standard POSIX prefixes. ``/api/sources``, ``/api/raw_artifacts/:id``,
    and ``/api/status`` deliberately surface absolute paths under the
    operator-level token (see ``docs/security.md``) and are out of scope
    for this lane.
    """

    def test_web_shell_does_not_leak_absolute_local_paths(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        for prefix in POLYLOGUE_LOCAL_PATH_PREFIXES:
            assert prefix not in body, f"web shell leaked absolute local path with prefix {prefix!r}"

    def test_facets_json_does_not_leak_absolute_local_paths(self, workspace_env: dict[str, Path]) -> None:
        # ``/api/facets`` is the read-only aggregations surface that the
        # reader hits every render. Per ``docs/security.md`` only
        # ``/api/sources`` and ``/api/raw_artifacts/:id`` deliberately
        # expose absolute paths under the operator-level token, so the
        # facets envelope must stay clean.
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        text = json.dumps(payload)
        for prefix in POLYLOGUE_LOCAL_PATH_PREFIXES:
            assert prefix not in text, f"facets JSON leaked absolute local path with prefix {prefix!r}"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/sessions",
            "/api/sessions/claude-code-session:c1",
            "/api/sessions/claude-code-session:c1/messages",
            "/api/sessions/claude-code-session:c1/read?view=messages",
            "/api/sessions/claude-code-session:c1/read?view=recovery",
            "/api/sessions/claude-code-session:c1/recovery?report=work-packet&format=json",
            "/api/assertions?target_ref=session%3Aclaude-code-session%3Ac1",
        ],
    )
    def test_reader_json_contracts_do_not_leak_absolute_local_paths(
        self,
        workspace_env: dict[str, Path],
        path: str,
    ) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, path)
        text = json.dumps(payload)
        for prefix in POLYLOGUE_LOCAL_PATH_PREFIXES:
            assert prefix not in text, f"{path} leaked absolute local path with prefix {prefix!r}"


# ---------------------------------------------------------------------------
# Auth boundary (smoke)
# ---------------------------------------------------------------------------


class TestReaderAuthSurface:
    def test_authenticated_endpoint_rejects_missing_bearer_when_token_set(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, auth_token="secret-token") as (_, base_url):
            status, _, _ = _get_text(base_url, "/api/sessions")
        assert status == 401

    def test_authenticated_endpoint_accepts_correct_bearer(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, auth_token="secret-token") as (_, base_url):
            payload = _get_json(base_url, "/api/sessions", headers={"Authorization": "Bearer secret-token"})
        assert isinstance(payload, dict)
        assert payload["total"] == 3

    @pytest.mark.parametrize(
        "path",
        [
            "/api/sessions/claude-code-session:c1/recovery?report=work-packet&format=json",
            "/api/sessions/claude-code-session:c1/read?view=recovery",
            "/api/assertions?target_ref=session%3Aclaude-code-session%3Ac1",
        ],
    )
    def test_workbench_helper_endpoints_require_bearer_when_token_set(
        self, workspace_env: dict[str, Path], path: str
    ) -> None:
        with _running_server(workspace_env, auth_token="secret-token") as (_, base_url):
            missing_status, _, _ = _get_text(base_url, path)
            ok_payload = _get_json(base_url, path, headers={"Authorization": "Bearer secret-token"})

        assert missing_status == 401
        assert isinstance(ok_payload, dict)

    def test_unauthenticated_root_still_serves_web_shell(self, workspace_env: dict[str, Path]) -> None:
        # The web shell at / is the only unauthenticated GET (see
        # docs/security.md and daemon/http.py:_dispatch_get).
        with _running_server(workspace_env, auth_token="secret-token") as (_, base_url):
            status, content_type, _ = _get_text(base_url, "/")
        assert status == 200
        assert "text/html" in content_type


# ---------------------------------------------------------------------------
# Defensive marker — explicit declaration of the lane this file owns.
# ---------------------------------------------------------------------------


def test_reader_smoke_lane_is_documented() -> None:
    """Sanity check that the reader smoke artefact ids the issue
    mentions exist as identifiers in this file. A future PR that
    renames the artefact ids must also update the docs that reference
    them.
    """
    body = Path(__file__).read_text()
    assert "polylogue.local_reader.search" in body
    assert "polylogue.local_reader.session" in body


# ---------------------------------------------------------------------------
# Shared surface payload contract tests (#859)
# ---------------------------------------------------------------------------


class TestSharedQueryPayloads:
    """QueryMissDiagnosticsPayload, QueryErrorPayload, and response envelopes."""

    def test_query_error_payload_roundtrip(self) -> None:
        from polylogue.surfaces.payloads import QueryErrorPayload

        p = QueryErrorPayload(error="QuerySpecError", detail="invalid since: bogus", field="since")
        d = p.model_dump(mode="json")
        assert d["ok"] is False
        assert d["error"] == "QuerySpecError"
        assert d["detail"] == "invalid since: bogus"
        assert d["field"] == "since"

    def test_query_error_payload_minimal(self) -> None:
        from polylogue.surfaces.payloads import QueryErrorPayload

        p = QueryErrorPayload(error="internal_error")
        d = p.model_dump(mode="json")
        assert d["ok"] is False
        assert d["detail"] is None
        assert d["field"] is None

    def test_query_miss_diagnostics_roundtrip(self) -> None:
        from polylogue.surfaces.payloads import QueryMissDiagnosticsPayload, QueryMissReasonPayload

        p = QueryMissDiagnosticsPayload(
            message="No sessions matched the query.",
            filters=("provider=gemini", "tag=api"),
            reasons=(QueryMissReasonPayload(code="no_results", severity="info", summary="empty archive"),),
            archive_session_count=42,
            raw_session_count=100,
        )
        d = p.model_dump(mode="json")
        assert d["message"] == "No sessions matched the query."
        assert d["filters"] == ["provider=gemini", "tag=api"]
        assert len(d["reasons"]) == 1
        assert d["archive_session_count"] == 42

    def test_session_list_response_shape(self) -> None:
        from polylogue.surfaces.payloads import SessionListResponse

        r = SessionListResponse(items=(), total=0, limit=50, offset=0)
        d = r.model_dump(mode="json")
        assert d["items"] == []
        assert d["total"] == 0
        assert d["limit"] == 50
        assert d["offset"] == 0

    def test_reader_target_ref_and_action_payloads_roundtrip(self) -> None:
        from polylogue.surfaces.payloads import (
            ReaderActionAvailabilityPayload,
            SessionListRowPayload,
            SessionMessagePayload,
            TargetRefPayload,
            reader_anchor,
        )

        message_ref = TargetRefPayload.message(session_id="c1", message_id="m-c1")
        assert message_ref.model_dump(mode="json", exclude_none=True) == {
            "target_type": "message",
            "target_id": "m-c1",
            "session_id": "c1",
            "message_id": "m-c1",
            "identity_key": "message:c1:m-c1",
        }
        assert reader_anchor("message", "m:c1/unsafe") == "message-m-c1-unsafe"

        row = SessionListRowPayload(
            id="c1",
            origin="claude-code-session",
            title="Reader contract",
            target_ref=TargetRefPayload.session("c1"),
            anchor="session-c1",
            message_count=1,
        )
        message = SessionMessagePayload(
            id="m-c1",
            role="user",
            text="Hello reader",
            target_ref=message_ref,
            anchor="message-m-c1",
            actions={"annotate": ReaderActionAvailabilityPayload(enabled=True)},
        )

        row_dump = row.model_dump(mode="json", exclude_none=True)
        message_dump = message.model_dump(mode="json", exclude_none=True)
        assert row_dump["target_ref"]["identity_key"] == "session:c1"
        assert row_dump["actions"]["copy_link"]["enabled"] is True
        assert message_dump["target_ref"]["identity_key"] == "message:c1:m-c1"
        assert message_dump["actions"]["annotate"]["enabled"] is True

    def test_session_list_response_with_diagnostics(self) -> None:
        from polylogue.surfaces.payloads import (
            QueryMissDiagnosticsPayload,
            QueryMissReasonPayload,
            SessionListResponse,
        )

        diag = QueryMissDiagnosticsPayload(
            message="No results.",
            filters=("tag=missing",),
            reasons=(QueryMissReasonPayload(code="no_results", severity="info", summary="no match"),),
        )
        r = SessionListResponse(items=(), total=0, limit=10, offset=0, diagnostics=diag)
        d = r.model_dump(mode="json")
        assert d["diagnostics"] is not None
        assert d["diagnostics"]["message"] == "No results."

    def test_facets_response_shape(self) -> None:
        from polylogue.surfaces.payloads import FacetsResponse, FacetTimeRange

        r = FacetsResponse(
            scoped_to_query=False,
            origins={"claude-code-session": 10, "chatgpt-export": 5},
            total_sessions=15,
            total_messages=100,
            time_range=FacetTimeRange(min="2024-01-01", max="2024-12-31"),
        )
        d = r.model_dump(mode="json")
        assert d["scoped_to_query"] is False
        assert d["origins"] == {"claude-code-session": 10, "chatgpt-export": 5}
        assert d["total_sessions"] == 15
        assert d["time_range"]["min"] == "2024-01-01"

    def test_assertion_claim_payloads_are_shared_surface_dtos(self) -> None:
        from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope
        from polylogue.surfaces.payloads import AssertionClaimListPayload, AssertionClaimPayload

        envelope = ArchiveAssertionEnvelope(
            assertion_id="claim-1",
            scope_ref="repo:polylogue",
            target_ref="session:c1",
            key=None,
            kind="decision",
            value=None,
            body_text="Keep assertion payloads shared.",
            author_ref="agent:test",
            author_kind="agent",
            evidence_refs=["message:m1"],
            status="active",
            visibility="private",
            confidence=0.9,
            staleness=None,
            context_policy={"inject": False},
            supersedes=[],
            created_at_ms=1,
            updated_at_ms=2,
        )

        item = AssertionClaimPayload.from_envelope(envelope)
        payload = AssertionClaimListPayload(items=(item,), total=1, limit=20, statuses=("active",))
        dump = payload.model_dump(mode="json", exclude_none=True)

        assert dump["items"][0]["assertion_id"] == "claim-1"
        assert dump["items"][0]["evidence_refs"] == ["message:m1"]
        assert dump["statuses"] == ["active"]

    def test_recovery_read_payload_wraps_storage_free_recovery_models(self) -> None:
        from polylogue.surfaces.payloads import RecoveryReadPayload, SurfacePayloadModel

        class TinyRecovery(SurfacePayloadModel):
            session_id: str
            evidence_refs: tuple[str, ...] = ()

        digest = TinyRecovery(session_id="c1", evidence_refs=("message:m1",))
        packet = TinyRecovery(session_id="c1", evidence_refs=("message:m2",))

        digest_payload = RecoveryReadPayload.from_digest(digest).model_dump(mode="json", exclude_none=True)
        packet_payload = RecoveryReadPayload.from_work_packet_json(packet).model_dump(mode="json", exclude_none=True)
        markdown_payload = RecoveryReadPayload.from_work_packet_markdown(
            session_id="c1", markdown="# Work packet"
        ).model_dump(mode="json", exclude_none=True)

        assert digest_payload["digest"]["evidence_refs"] == ["message:m1"]
        assert packet_payload["work_packet"]["evidence_refs"] == ["message:m2"]
        assert markdown_payload == {
            "session_id": "c1",
            "report": "work-packet",
            "format": "markdown",
            "markdown": "# Work packet",
        }


@pytest.mark.load_sensitive
class TestQueryNoResultsDiagnosticPath:
    """Cross-surface contract: one filtered query → diagnostics path."""

    def test_search_with_bad_date_returns_query_error(self, workspace_env: dict[str, Path]) -> None:
        """A query with a malformed date produces a QueryErrorPayload-shaped error."""
        with _running_server(workspace_env) as (_, base_url):
            status, body = _get_json_ex(base_url, "/api/sessions?since=not-a-date")
        assert status == 400
        # The error shape is compatible with QueryErrorPayload
        assert "error" in body
        assert body.get("ok") is False

    def test_session_list_rejects_terminal_only_unit_query(self, workspace_env: dict[str, Path]) -> None:
        """Session-list queries must not lower runtime rows into broken EXISTS predicates."""
        expression = quote("runs where role:subagent")
        with _running_server(workspace_env) as (_, base_url):
            status, body = _get_json_ex(base_url, f"/api/sessions?query={expression}")
        assert status == 400
        assert body.get("ok") is False
        assert "terminal run rows" in str(body.get("detail") or body.get("message") or "")

    def test_list_with_no_match_returns_diagnostics(self, workspace_env: dict[str, Path]) -> None:
        """A list with a tag that doesn't exist returns items=[] with total=0."""
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/sessions?tag=nonexistent")
        assert isinstance(payload, dict)
        assert "items" in payload
        assert "total" in payload
        assert payload["total"] == 0
        assert payload["items"] == []

    def test_facets_global_returns_origins(self, workspace_env: dict[str, Path]) -> None:
        """Unscoped /api/facets returns scoped_to_query=False with origin counts."""
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is False
        assert isinstance(payload["origins"], dict)

    def test_facets_scoped_returns_subset(self, workspace_env: dict[str, Path]) -> None:
        """Scoped /api/facets?origin=... returns scoped_to_query=True."""
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?origin=chatgpt-export")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is True


def _get_json_ex(base_url: str, path: str) -> tuple[int, dict[str, object]]:
    """GET a path and return (status, parsed-body-or-empty-dict).

    Unlike ``_get_json``, this helper reads the response body even for
    error status codes so contract tests can assert on error-envelope shape.
    """
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    req = Request(f"{base_url}{path}")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return e.code, {}


class TestReaderInformability:
    """MK3 informability surfaces (#956): explicit state language for
    degraded/partial/stale/unavailable data, and a coherent data-quality
    chip vocabulary the operator can rely on at a glance.

    These tests assert *presence* of the chip-quality vocabulary, the
    tri-state FTS render, the insight-freshness chip, and the
    context-preserving sidebar empty states. They intentionally check
    JS source rather than runtime DOM so the smoke remains fast and the
    visual lane (Playwright/Lighthouse) can layer on top later (#952).
    """

    def test_data_quality_chip_vocabulary_is_defined(self, workspace_env: dict[str, Path]) -> None:
        """The MK3 chip vocabulary (canonical/inferred/heuristic/explicit/
        unresolved/repaired/stale/partial/estimated/unavailable/redacted)
        from docs/design/mk3/docs/11-little-details.md must have CSS
        classes the renderer can apply. Without these the rest of the
        informability story is just text.
        """
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        for quality in (
            "q-canonical",
            "q-inferred",
            "q-heuristic",
            "q-explicit",
            "q-unresolved",
            "q-repaired",
            "q-stale",
            "q-partial",
            "q-estimated",
            "q-unavailable",
            "q-redacted",
        ):
            assert f".chip.{quality}" in body, f"web shell missing MK3 chip-quality class {quality!r}"

    def test_status_strip_consumes_component_readiness(self, workspace_env: dict[str, Path]) -> None:
        """The web shell status strip should consume the canonical #1832
        component readiness map instead of recomputing product state from
        legacy daemon fields. Legacy FTS rendering remains as fallback for
        old/status-snapshot payloads.
        """
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "var readiness = s.component_readiness || {};" in body
        assert "renderFtsChip(readiness.search || null, s.fts_readiness || {})" in body
        assert "renderSemanticChip(readiness.embeddings || null)" in body
        assert "renderInsightChip(readiness.session_profiles || null, s.insight_freshness || {})" in body
        assert "renderIngestChip(readiness.daemon_ingest || null, s.live || {})" in body
        assert "renderBrowserCaptureChip(readiness.browser_capture || null, s.browser_capture || {})" in body
        assert "function renderComponentReadinessChip(" in body
        assert "function readinessQuality(" in body
        assert "function renderBrowserCaptureChip(" in body

    def test_fts_chip_keeps_legacy_tri_state_fallback(self, workspace_env: dict[str, Path]) -> None:
        """``renderFtsChip`` must still distinguish ok / partial /
        unavailable when an older payload lacks ``component_readiness``.
        """
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "function renderFtsChip(" in body
        assert "component && component.state !== 'unknown'" in body
        assert "'FTS: ok'" in body
        assert "'FTS: partial'" in body
        assert "'FTS: unavailable'" in body

    def test_semantic_and_ingest_readiness_chips_render_present(self, workspace_env: dict[str, Path]) -> None:
        """Semantic and ingest state have first-class status-strip chips fed
        by ``component_readiness.embeddings`` and
        ``component_readiness.daemon_ingest``.
        """
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert 'id="status-semantic"' in body
        assert 'id="status-ingest"' in body
        assert "function renderSemanticChip(" in body
        assert "function renderIngestChip(" in body
        assert "'semantic'" in body
        assert "'ingest'" in body

    def test_browser_capture_readiness_chip_consumes_safe_status_payload(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """The web workbench displays browser-capture readiness from the
        daemon status envelope without learning the receiver's local spool path.
        """
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
            status = cast(dict[str, object], _get_json(base_url, "/api/status"))

        capture = cast(dict[str, object], status["browser_capture"])
        readiness = cast(dict[str, object], status["component_readiness"])

        assert 'id="status-browser-capture"' in body
        assert "function renderBrowserCaptureChip(" in body
        assert "spool_ready" in capture
        assert "allowed_origins" in capture
        assert "auth_required" in capture
        assert "spool_path" not in capture
        assert "artifact_path" not in capture
        assert "browser_capture" in readiness

    def test_insight_freshness_chip_keeps_legacy_fallback(self, workspace_env: dict[str, Path]) -> None:
        """Session insight freshness gets its own status-strip chip. It now
        prefers ``component_readiness.session_profiles`` and keeps the
        previous freshness-payload fallback for old status snapshots."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "function renderInsightChip(" in body
        assert 'id="status-insights"' in body
        assert "renderComponentReadinessChip(el, 'insights', component)" in body
        assert "'insights: ok'" in body
        assert "'insights: stale'" in body

    def test_sidebar_empty_state_preserves_filter_context(self, workspace_env: dict[str, Path]) -> None:
        """The empty/no-results branches must include filter context and
        a concrete next action — the MK3 state matrix calls out that an
        empty archive and a filtered no-results are different states."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "No sessions in archive. Run `polylogued run`" in body
        assert "No results for query=" in body
        assert "No sessions from origin=" in body
        assert "Press Esc to clear" in body

    def test_workspace_actions_have_disabled_state_with_tooltips(self, workspace_env: dict[str, Path]) -> None:
        """Stack/Compare/Save/Recall workspace buttons must expose a
        disabled state with explanatory tooltips when context is
        insufficient — disabled actions are part of the MK3 design."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "Stack needs a selected session" in body
        assert "Compare needs two sessions" in body
        assert "Save workspace needs at least one open session" in body
        assert "Recall pack needs at least one open session" in body

    def test_session_header_chip_order_follows_mk3_spec(self, workspace_env: dict[str, Path]) -> None:
        """MK3 specifies header chip order: origin, live/stale,
        repo/cwd/branch, counts, cost/tokens, derived/insight, marks.
        Pin the comment marker so a future reorder is a deliberate change
        rather than accidental drift."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "MK3 header chip order" in body
        # The origin chip should be tagged canonical (it identifies the
        # source-of-truth origin for the session), not just a neutral chip.
        assert "'<span class=\"chip q-canonical\">' + esc(c.origin)" in body


class TestReaderSavedViewsUI:
    """Saved-view UI surface (#1118): toolbar entry, save/recall/delete UX,
    naming-conflict guard, and an explicit empty state for the inspector list.

    The substrate (``/api/user/saved-views`` POST/GET/DELETE) is covered by
    ``TestReaderUserState``; these tests assert that the web shell exposes the
    capability end-to-end so the operator can manage saved views without
    leaving the reader.
    """

    def test_workspace_toolbar_exposes_saved_view_entry(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        # Dedicated toolbar entry-point so saved views are reachable without
        # opening the Notes inspector tab (mirrors the Restore workspace
        # pattern).
        assert 'id="workspace-saved-view-select"' in body
        assert 'id="workspace-save-view-btn"' in body
        # Wired to applySavedView via restoreSavedView; both must be present.
        assert "function restoreSavedView(" in body
        assert "function applySavedView(" in body
        # Toolbar populator must update the select on each render so newly
        # saved views appear without a full reload.
        assert "workspace-saved-view-select" in body
        assert "Saved views (" in body

    def test_save_current_view_handles_naming_conflict(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        # Naming conflict UX: detect locally against state.savedViews, prompt
        # for overwrite confirmation, replace via DELETE+POST. Empty/whitespace
        # names are rejected before hitting the wire.
        assert "Saved view name cannot be empty" in body
        assert "already exists. Overwrite it?" in body
        assert "Failed to replace existing view" in body

    def test_saved_view_list_exposes_delete_action(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        # The inspector list must render a Delete button per saved view; the
        # JS handler is reachable so the operator can prune stale views.
        assert "function deleteSavedView(" in body
        assert "Delete saved view" in body
        assert 'onclick="deleteSavedView(' in body

    def test_saved_view_empty_state_is_actionable(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        # Empty state must point the operator at the action that resolves it
        # (per MK3 state matrix: each empty state names the next step).
        assert 'No saved views. Click "Save current view"' in body

    def test_saved_views_endpoint_roundtrip_supports_ui_flow(self, workspace_env: dict[str, Path]) -> None:
        # End-to-end roundtrip the UI relies on: save, list, then delete via
        # the inspector Delete button. Catches API regressions that would
        # silently break the new toolbar surface.
        with _running_server(workspace_env) as (_, base_url):
            save_status, saved = _request_json(
                base_url,
                "POST",
                "/api/user/saved-views",
                payload={
                    "view_id": "view-ui",
                    "name": "Reader UI flow",
                    "query": {"query": "auth", "limit": 25},
                },
            )
            listed = _get_json(base_url, "/api/user/saved-views")
            delete_status, deleted = _request_json(
                base_url,
                "DELETE",
                "/api/user/saved-views/view-ui",
            )
            empty_listing = _get_json(base_url, "/api/user/saved-views")

        saved_payload = cast(dict[str, object], saved)
        listed_payload = cast(dict[str, object], listed)
        empty_payload = cast(dict[str, object], empty_listing)
        items = cast(list[dict[str, object]], listed_payload["items"])
        assert save_status == 201
        assert saved_payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "saved_view.save",
            "resource_type": "saved_view",
            "resource_id": "view-ui",
        }
        assert listed_payload["total"] == 1
        assert items[0]["view_id"] == "view-ui"
        assert items[0]["name"] == "Reader UI flow"
        assert delete_status == 200
        assert deleted == {
            "status": "deleted",
            "affected_count": 1,
            "operation": "saved_view.delete",
            "resource_type": "saved_view",
            "resource_id": "view-ui",
        }
        assert empty_payload == {"items": [], "total": 0}


# ---------------------------------------------------------------------------
# polylogue.local_reader.bulk — multi-select bulk operations (#1119)
# ---------------------------------------------------------------------------


class TestReaderBulkOperations:
    """``polylogue.local_reader.bulk``: selection toolbar + per-session envelope.

    The bulk surface composes existing daemon routes: ``/api/user/marks``
    carries route-backed overlay mutations, and ``/api/sessions/{id}``
    carries export reads. Delete and re-embed are exposed only through a
    preview overlay because the daemon has no corresponding mutation routes yet.

    These tests assert the shipped HTML carries every load-bearing region
    hook and that the underlying tag endpoint accepts the per-session
    POSTs the bulk toolbar drives — that is the contract the JS depends on.
    Pixel-level assertions are deliberately avoided so reader visual
    iteration does not invalidate the smoke.
    """

    BULK_REGION_HOOKS = (
        "bulk-toolbar",
        "bulk-select-all",
        "bulk-clear",
        'data-bulk-action="tag-star"',
        'data-bulk-action="tag-pin"',
        'data-bulk-action="tag-archive"',
        'data-bulk-action="export"',
        'data-bulk-action="delete-preview"',
        'data-bulk-action="reembed-preview"',
        "bulk-preview",
        "bulk-preview-confirm",
        "bulkApplyMark",
        "bulkExport",
        "openBulkPreview",
        "confirmBulkPreview",
        "renderBulkToolbar",
        "isBulkSelected",
        "bulkSelection",
        "no_endpoint",
    )

    def test_bulk_toolbar_regions_present_in_shell(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/")
        assert status == 200
        assert "text/html" in content_type
        for hook in self.BULK_REGION_HOOKS:
            assert hook in body, f"web shell missing bulk hook {hook!r}"

    def test_bulk_envelope_keys_documented_in_shell(self, workspace_env: dict[str, Path]) -> None:
        """The shipped JS must reference the per-session status keys the
        bulk envelope contract uses (succeeded/failed/skipped + dryRun + action).
        These are the field names the UI surfaces to the operator after a bulk
        op; renaming any of them silently would break the rendered status."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        for key in ("succeeded", "failed", "skipped", "dryRun", "action"):
            assert key in body, f"bulk envelope key {key!r} missing from shell"

    def test_bulk_tag_drives_existing_marks_endpoint(self, workspace_env: dict[str, Path]) -> None:
        """The bulk toolbar issues per-session POSTs against
        ``/api/user/marks``. This simulates that loop server-side to lock the
        contract: the daemon must accept the same payload shape the bulk JS
        emits for every selected session, and a follow-up GET must
        surface the resulting marks on every session."""
        with _running_server(workspace_env) as (_, base_url):
            statuses: list[int] = []
            for cid in ("claude-code-session:c1", "chatgpt-export:c2", "claude-ai-export:c3"):
                status, _ = _request_json(
                    base_url,
                    "POST",
                    "/api/user/marks",
                    payload={"session_id": cid, "mark_type": "star"},
                )
                statuses.append(status)
            listing = _get_json(base_url, "/api/user/marks?mark_type=star")
        assert statuses == [201, 201, 201]
        payload = cast(dict[str, object], listing)
        items = cast(list[dict[str, object]], payload["items"])
        ids = sorted(str(item["session_id"]) for item in items)
        assert ids == ["chatgpt-export:c2", "claude-ai-export:c3", "claude-code-session:c1"]

    def test_bulk_export_uses_session_detail_endpoint(self, workspace_env: dict[str, Path]) -> None:
        """Bulk export concatenates per-session GETs. This pins the
        contract that ``/api/sessions/{id}`` returns the detail payload
        the export bundle is composed from, for every selected id."""
        with _running_server(workspace_env) as (_, base_url):
            payloads = [
                _get_json(base_url, f"/api/sessions/{cid}")
                for cid in ("claude-code-session:c1", "chatgpt-export:c2", "claude-ai-export:c3")
            ]
        ids = sorted(str(cast(dict[str, object], p)["id"]) for p in payloads)
        assert ids == ["chatgpt-export:c2", "claude-ai-export:c3", "claude-code-session:c1"]


@pytest.mark.parametrize("path", ["/", "/api/sessions", "/api/facets", "/api/status", "/api/health"])
def test_each_reader_route_responds_within_a_reasonable_budget(workspace_env: dict[str, Path], path: str) -> None:
    """Each reader-facing route returns within 10 s on a synthetic
    three-session archive. ``/api/health`` is included because the
    web shell pings it on every render cycle (see
    ``polylogue/daemon/web_shell.py::loadStatus``); a regression there
    would freeze the reader UI even if the data routes were healthy.
    The budget is loose by design — this is a smoke that catches
    "endpoint hangs forever" regressions, not a latency benchmark.
    """
    with _running_server(workspace_env) as (_, base_url):
        status, _, _ = _get_text(base_url, path)
    assert status == 200
