"""Reader smoke tests — page structure, API contracts, degraded states (#865).

Boots the production ``DaemonAPIHTTPServer`` against a synthetic on-disk
archive and exercises the live HTTP surface via ``urllib``. The HTML
payload served at ``/`` is asserted at the DOM-shape level (semantic
selectors, never pixel diffs); JSON envelopes are asserted by shape so
any regression in the daemon's contract surface fails here loudly.

This is the documented reader visual smoke lane (see
``docs/visual-evidence.md``). The lane runs as part of the standard
unit suite and ``devtools verify``; a separate ``devtools lab smoke``
entrypoint can be added later if Playwright-based screenshot evidence
is bolted on.

All test classes in this module start real HTTP servers — they share an
xdist group to prevent cross-worker port/event-loop interference under
parallel execution.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.xdist_group("web-reader")

import json
import socket
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
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


def test_web_reader_archive_root_rejects_schema_mismatch(tmp_path: Path) -> None:
    from polylogue.daemon.http import _web_reader_archive_root
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")

    with patch("polylogue.paths.archive_root", return_value=tmp_path):
        assert _web_reader_archive_root() is None


def _materialize_run_projection(index_db: Path) -> None:
    """Rebuild session insights for richer digest-derived run-projection rows."""
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(index_db) as conn:
        rebuild_session_insights_sync(conn)


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


def _degrade_message_fts(workspace: dict[str, Path]) -> None:
    """Drop the message search index so /api/sessions?query=... can report degraded route readiness."""

    with sqlite3.connect(workspace["archive_root"] / "index.db") as conn:
        for trigger in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.commit()


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
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        AssertionKind,
        AssertionStatus,
        AssertionVisibility,
        upsert_assertion,
    )

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
            # author_kind="user": an active, injectable decision is always
            # user-authored in production (37t.15) -- non-user writes land
            # as candidates regardless of the caller-supplied status/policy.
            author_kind="user",
            evidence_refs=[f"message:{M_C1}"],
            status=AssertionStatus.ACTIVE,
            visibility=AssertionVisibility.PRIVATE,
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
            status=AssertionStatus.CANDIDATE,
            visibility=AssertionVisibility.PRIVATE,
            context_policy={"inject": False},
            now_ms=1_700_000_000_100,
        )
        upsert_assertion(
            conn,
            assertion_id="claim-web-workbench-deleted",
            target_ref=f"session:{C1}",
            kind=AssertionKind.DECISION,
            body_text="Deleted claim is hidden by default.",
            status=AssertionStatus.DELETED,
            visibility=AssertionVisibility.PRIVATE,
            now_ms=1_700_000_000_200,
        )
        conn.commit()


def _seed_many_assertion_candidates(workspace: dict[str, Path], count: int) -> None:
    """Seed enough candidate assertions to make archive-debt pagination observable."""

    import sqlite3

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        AssertionKind,
        AssertionVisibility,
        upsert_assertion,
    )

    user_db = workspace["archive_root"] / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        for idx in range(count):
            upsert_assertion(
                conn,
                assertion_id=f"claim-archive-debt-candidate-{idx:03d}",
                target_ref=f"session:{C1}",
                scope_ref="repo:polylogue",
                kind=AssertionKind.CAVEAT,
                body_text=f"Candidate debt row {idx}",
                author_ref="agent:debt-test",
                author_kind="agent",
                evidence_refs=[f"message:{M_C1}"],
                status="candidate",
                visibility=AssertionVisibility.PRIVATE,
                context_policy={"inject": False},
                now_ms=1_700_000_010_000 + idx,
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


def _facets_from_facade(
    workspace: dict[str, Path],
    params: dict[str, list[str]],
    *,
    include_deferred: bool,
) -> dict[str, object]:
    from polylogue.api import Polylogue
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.daemon.http import _build_query_spec_params

    query_params = _build_query_spec_params(params, _QueryParamBuilderHandler())  # type: ignore[arg-type]
    spec = SessionQuerySpec.from_params(query_params) if query_params else None
    archive = Polylogue(archive_root=workspace["archive_root"], db_path=workspace["archive_root"] / "index.db")

    async def _run() -> dict[str, object]:
        response = await archive.facets(spec, include_deferred=include_deferred)
        return cast(dict[str, object], response.model_dump(mode="json", by_alias=True))

    return asyncio.run(_run())


def _assert_facets_match_facade(
    workspace: dict[str, Path],
    route_payload: dict[str, object],
    params: dict[str, list[str]],
    *,
    include_deferred: bool,
) -> None:
    expected = _facets_from_facade(workspace, params, include_deferred=include_deferred)
    # The two calls are intentionally independent; all route content should
    # match the facade except the wall-clock timestamp each call assigns.
    route_without_timestamp = {key: value for key, value in route_payload.items() if key != "generated_at"}
    expected_without_timestamp = {key: value for key, value in expected.items() if key != "generated_at"}
    assert route_without_timestamp == expected_without_timestamp


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


def _seed_import_explain_archive(workspace: dict[str, Path]) -> tuple[str, str]:
    import sqlite3

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    raw_id = "raw-import-route"
    source_path = str(Path.home() / ".codex" / "sessions" / "route.jsonl")
    digest = b"r" * 32
    archive_root = workspace["archive_root"]
    archive_root.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(archive_root / "source.db")
    index_conn = sqlite3.connect(archive_root / "index.db")
    try:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
        initialize_archive_tier(index_conn, ArchiveTier.INDEX)
        source_conn.execute(
            """
            INSERT OR REPLACE INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, validated_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                "route-native",
                source_path,
                0,
                digest,
                128,
                1_700_000_000_000,
                1_700_000_000_100,
                1_700_000_000_200,
                "passed",
            ),
        )
        source_conn.execute(
            """
            INSERT OR REPLACE INTO raw_artifacts (
                artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                support_status, classification_reason, parse_as_session, schema_eligible,
                first_observed_at_ms, last_observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "route-artifact",
                raw_id,
                "codex-session",
                source_path,
                0,
                "session_record_stream",
                "supported_parseable",
                "codex route fixture",
                1,
                1,
                1_700_000_000_000,
                1_700_000_000_000,
            ),
        )
        index_conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                native_id, origin, raw_id, title, content_hash, message_count, tool_use_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("route-native", "codex-session", raw_id, "Route import explain", digest, 1, 1),
        )
        index_conn.execute(
            """
            INSERT OR REPLACE INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:route-native", "m1", 0, "assistant", "message", digest),
        )
        index_conn.execute(
            """
            INSERT OR REPLACE INTO blocks (
                message_id, session_id, position, block_type, text, tool_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:route-native:m1", "codex-session:route-native", 0, "tool_use", "pytest", "tool-1"),
        )
        source_conn.commit()
        index_conn.commit()
    finally:
        source_conn.close()
        index_conn.close()
    return raw_id, source_path


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


def test_socket_peer_disconnected_detects_closed_loopback_peer() -> None:
    from polylogue.daemon.http import _socket_peer_disconnected

    server_sock, client_sock = socket.socketpair()
    try:
        assert _socket_peer_disconnected(server_sock) is False
        client_sock.close()
        assert _socket_peer_disconnected(server_sock) is True
    finally:
        server_sock.close()


def test_archive_bounded_query_progress_handler_interrupts_after_client_abort() -> None:
    from polylogue.daemon.http import DaemonAPIHandler

    class _Archive:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

    conn = sqlite3.connect(":memory:")
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    checks = {"count": 0}

    def disconnected_after_start() -> bool:
        checks["count"] += 1
        return checks["count"] > 2

    handler._client_disconnected = disconnected_after_start  # type: ignore[method-assign]
    try:
        with pytest.raises(ConnectionAbortedError):
            handler._run_archive_bounded_query(
                _Archive(conn),  # type: ignore[arg-type]
                deadline_s=None,
                compute=lambda: conn.execute(
                    """
                    WITH RECURSIVE counter(value) AS (
                      VALUES(0)
                      UNION ALL
                      SELECT value + 1 FROM counter WHERE value < 1000000
                    )
                    SELECT SUM(value) FROM counter
                    """
                ).fetchone(),
            )
        assert checks["count"] > 2
    finally:
        conn.close()


def test_archive_session_list_route_uses_bounded_sql_helper() -> None:
    import inspect

    from polylogue.daemon.http import DaemonAPIHandler

    source = inspect.getsource(DaemonAPIHandler._do_archive_list_sessions)

    assert "self._run_archive_bounded_query(" in source
    assert "archive.search_summaries(" in source
    assert "archive.list_summaries(" in source
    assert "archive.count_sessions(" in source


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
            "sessionsFromListPayload",
            "renderFacets",
            "renderMain",
            "renderWorkspaceToolbar",
            "renderStackWorkspace",
            "renderCompareWorkspace",
            "renderInspector",
            "renderInspectorMission",
            "Subagent Exchanges",
            "subagent_exchanges",
            "returned_final_message",
            "renderInspectorEvidence",
            "renderBrowserCaptureChip",
            "renderReadViewExecution",
            "loadReadViewProfiles",
            "renderReadViewSelector",
            "renderRouteStateNotice",
            "renderInlineRouteFailure",
            "sessionList",
            "data-route-state-name",
            "data-route-state",
            "data-sidebar-state",
            "AbortController",
            "timeoutMs",
            "request_timeout_after_",
            "budget_exceeded",
            "applyReadViewSelection",
            "/api/read-view-profiles",
            "/api/agents/coordination",
            "/read?view=",
            "/api/assertions",
            "Load artifact list",
            "Load raw preview",
            'data-tab="evidence"',
            'data-tab="mission"',
        ):
            assert region in body, f"web shell missing region hook {region!r}"

    def test_agent_coordination_endpoint_uses_shared_payload(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, Any], _get_json(base_url, "/api/agents/coordination?view=status&limit=3"))

        assert payload["view"] == "status"
        repo = cast(dict[str, Any], payload["repo"])
        work_item = cast(dict[str, Any], payload["work_item"])
        assert repo["root"]
        assert work_item["source"] in {"beads", "git", "inferred", "none"}
        assert "peers" in payload
        assert "resource_episodes" in payload

    def test_agent_coordination_cache_is_ttl_bounded_and_fresh_bypassable(self, workspace_env: dict[str, Path]) -> None:
        """The live HTTP route caches only a bounded fresh response.

        The test uses the production ``DaemonAPIHandler`` and an actual
        loopback server; the envelope producer is only counted so the cache
        contract is observable.  Removing the cache read/write or the
        ``fresh=1`` bypass causes the call-count and response-header
        assertions to fail.
        """
        calls: list[tuple[str, int]] = []

        class FakePayload:
            def model_dump(self, **_kwargs: object) -> dict[str, object]:
                return {"view": "status", "build": len(calls)}

        def fake_build(*, view: str, limit: int) -> FakePayload:
            calls.append((view, limit))
            return FakePayload()

        def get_response(url: str) -> tuple[dict[str, object], str, str]:
            with urlopen(url) as response:
                body = cast(dict[str, object], json.loads(response.read()))
                return (
                    body,
                    response.headers["X-Polylogue-Coordination-Cache"],
                    response.headers["X-Polylogue-Coordination-Freshness"],
                )

        with patch("polylogue.coordination.build_coordination_envelope", side_effect=fake_build):
            with _running_server(workspace_env) as (server, base_url):
                first, first_state, first_freshness = get_response(
                    f"{base_url}/api/agents/coordination?view=status&limit=3"
                )
                second, second_state, second_freshness = get_response(
                    f"{base_url}/api/agents/coordination?view=status&limit=3"
                )
                bypassed, bypassed_state, _ = get_response(
                    f"{base_url}/api/agents/coordination?view=status&limit=3&fresh=1"
                )
                daemon_server = cast(Any, server)
                with daemon_server.coordination_cache_lock:
                    cached = daemon_server.coordination_cache[("status", 3)]
                    daemon_server.coordination_cache[("status", 3)] = type(cached)(
                        payload=cached.payload,
                        expires_at=0.0,
                    )
                expired, expired_state, _ = get_response(f"{base_url}/api/agents/coordination?view=status&limit=3")

        assert first == second == {"view": "status", "build": 1}
        assert bypassed == {"view": "status", "build": 2}
        assert expired == {"view": "status", "build": 3}
        assert calls == [("status", 3), ("status", 3), ("status", 3)]
        assert (first_state, second_state, bypassed_state, expired_state) == ("miss", "hit", "bypass", "miss")
        assert first_freshness == second_freshness == "ttl=2s; fresh=1 bypasses"

    def test_agent_coordination_cache_coalesces_concurrent_builds(self, workspace_env: dict[str, Path]) -> None:
        """Concurrent cold reads share one envelope build instead of stampeding it.

        Two real loopback clients request the same missing cache key.  The
        producer blocks long enough for the second request to enter the daemon;
        removing the in-flight condition allows a second producer call and
        fails the assertion below.
        """
        calls: list[tuple[str, int]] = []
        first_build_started = threading.Event()
        second_handler_entered = threading.Event()
        second_build_started = threading.Event()
        release_first_build = threading.Event()

        class FakePayload:
            def model_dump(self, **_kwargs: object) -> dict[str, object]:
                return {"view": "status", "build": len(calls)}

        def fake_build(*, view: str, limit: int) -> FakePayload:
            calls.append((view, limit))
            if len(calls) == 1:
                first_build_started.set()
                assert release_first_build.wait(timeout=2.0)
            else:
                second_build_started.set()
            return FakePayload()

        def get_response(url: str) -> dict[str, object]:
            with urlopen(url) as response:
                return cast(dict[str, object], json.loads(response.read()))

        from polylogue.daemon.http import DaemonAPIHandler

        original_handler = DaemonAPIHandler._handle_agent_coordination
        handler_calls = 0

        def observe_handler(handler: object, params: dict[str, list[str]]) -> None:
            nonlocal handler_calls
            handler_calls += 1
            if handler_calls == 2:
                second_handler_entered.set()
            original_handler(cast(Any, handler), params)

        with patch("polylogue.coordination.build_coordination_envelope", side_effect=fake_build):
            with patch.object(DaemonAPIHandler, "_handle_agent_coordination", observe_handler):
                with _running_server(workspace_env) as (_, base_url):
                    url = f"{base_url}/api/agents/coordination?view=status&limit=3"
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        first = executor.submit(get_response, url)
                        assert first_build_started.wait(timeout=2.0)
                        second = executor.submit(get_response, url)
                        assert second_handler_entered.wait(timeout=2.0)
                        assert not second_build_started.wait(timeout=0.2)
                        release_first_build.set()
                        assert first.result(timeout=2.0) == {"view": "status", "build": 1}
                        assert second.result(timeout=2.0) == {"view": "status", "build": 1}

        assert calls == [("status", 3)]

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
        assert payload["route_state"]["state"] == "ready"
        assert payload["route_state"]["route"] == "/api/sessions"

    def test_facets_envelope_includes_scoped_flag(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert "scoped_to_query" in payload
        assert "origins" in payload
        assert isinstance(payload["generated_at"], str)
        assert payload["stale"] is False
        assert payload["budget_exceeded"] is False
        assert set(payload["complete_families"]) >= {"total_counts", "origins", "tags"}
        assert payload["deferred_families"] == {
            "has_flags": "deferred_by_default",
            "material_origins": "deferred_by_default",
            "message_types": "deferred_by_default",
            "repos": "deferred_by_default",
            "role_counts": "deferred_by_default",
            "action_types": "deferred_by_default",
        }
        assert payload["family_status"]["repos"]["state"] == "deferred"
        assert "repos" in payload and "action_types" in payload
        _assert_facets_match_facade(workspace_env, payload, {}, include_deferred=False)

    def test_facets_request_materializes_deferred_families(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?include_deferred=1&budget_ms=0")
        assert isinstance(payload, dict)
        assert payload["budget_exceeded"] is False
        assert payload["deferred_families"] == {}
        repo_status = payload["family_status"]["repos"]
        assert repo_status["state"] == "complete"
        assert repo_status["reason"] is None
        assert repo_status["error"] is None
        assert repo_status["stale"] is False
        assert repo_status["stale_age_s"] is None
        assert repo_status["label"] == "Canonical repositories"
        assert repo_status["source"] == "session_repos + repos"
        assert repo_status["canonicalization"]
        assert repo_status["expensive"] is True
        assert "repos" in payload["complete_families"]
        assert "has_flags" in payload
        assert "idf" in payload
        _assert_facets_match_facade(workspace_env, payload, {}, include_deferred=True)

    def test_session_list_stays_visible_while_facets_include_expensive(self, workspace_env: dict[str, Path]) -> None:
        """A populated session list coexists with materialized expensive facets.

        The workbench must not let expensive facet families empty or block the
        base session list. The route now delegates to the facade; requesting
        deferred families materializes the shared buckets while ``/api/sessions``
        still returns rows.
        """
        with _running_server(workspace_env) as (_, base_url):
            sessions = cast(dict[str, object], _get_json(base_url, "/api/sessions"))
            facets = cast(
                dict[str, object],
                _get_json(base_url, "/api/facets?include_deferred=1&budget_ms=0"),
            )

        # The list itself is independent of the expensive facet path: it stays
        # populated even when the facet budget is exhausted.
        assert sessions["total"] == 3
        assert len(cast(list[object], sessions["items"])) == 3
        assert cast(dict[str, object], sessions["route_state"])["state"] == "ready"

        assert facets["budget_exceeded"] is False
        assert facets["deferred_families"] == {}
        assert {"total_counts", "origins"} <= set(cast(list[str], facets["complete_families"]))
        assert cast(list[object], facets["origins"])
        _assert_facets_match_facade(workspace_env, facets, {}, include_deferred=True)

    @pytest.mark.parametrize(
        "path",
        [
            "/api/sessions?has_paste_evidence=1",
            "/api/sessions?query=Hello&has_paste_evidence=1",
        ],
    )
    def test_paste_filter_uses_storage_keyword(self, workspace_env: dict[str, Path], path: str) -> None:
        """The public paste-evidence HTTP filter must compile to storage's has_paste keyword."""

        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, path)

        route_state = cast(dict[str, object], payload["route_state"])
        assert status == 200
        assert payload["total"] == 0
        assert route_state["state"] == "no_results"

    @pytest.mark.parametrize("path", ["/api/sessions", "/api/facets"])
    def test_archive_busy_routes_return_typed_503(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        path: str,
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        def _raise_locked(*args: object, **kwargs: object) -> ArchiveStore:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(ArchiveStore, "open_existing", _raise_locked)

        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, path)

        assert status == 503
        assert payload["error"] == "archive_busy"
        assert "temporarily unavailable" in cast(str, payload["detail"])

    def test_facets_can_materialize_deferred_families_when_requested(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?families=repos,action_types&budget_ms=5000")
        assert isinstance(payload, dict)
        assert payload["deferred_families"] == {}
        assert {"repos", "action_types"}.issubset(set(payload["complete_families"]))
        assert payload["family_status"]["repos"]["state"] == "complete"
        assert payload["family_status"]["action_types"]["state"] == "complete"

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
        assert [action["id"] for action in payload["action_affordances"]] == [
            "read",
            "continue",
            "select",
            "mark",
            "analyze",
            "delete",
        ]
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
        assert [action["id"] for action in payload["action_affordances"]] == [
            "read",
            "continue",
            "select",
            "mark",
            "analyze",
            "delete",
        ]
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
        assert global_payload["budget_exceeded"] is False
        assert global_payload["deferred_families"] == {
            "has_flags": "deferred_by_default",
            "material_origins": "deferred_by_default",
            "message_types": "deferred_by_default",
            "action_types": "deferred_by_default",
            "repos": "deferred_by_default",
            "role_counts": "deferred_by_default",
        }
        assert global_payload["family_status"]["repos"]["state"] == "deferred"
        assert global_payload["family_status"]["repos"]["reason"] == "deferred_by_default"
        assert global_payload["repos"] == {}
        assert global_payload["action_types"] == {}
        _assert_facets_match_facade(workspace_env, global_payload, {}, include_deferred=False)
        assert isinstance(scoped_payload, dict)
        assert scoped_payload["scoped_to_query"] is True
        assert scoped_payload["total_sessions"] == 1
        assert scoped_payload["origins"] == {"codex-session": 1}
        _assert_facets_match_facade(
            workspace_env,
            scoped_payload,
            {"origin": ["codex-session"]},
            include_deferred=False,
        )

    def test_archive_file_set_facets_can_include_expensive_families(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            payload = _get_json(base_url, "/api/facets?include_expensive=1")

        assert isinstance(payload, dict)
        assert payload["budget_exceeded"] is False
        assert payload["deferred_families"] == {}
        assert "repos" in payload["complete_families"]
        assert "action_types" in payload["complete_families"]
        assert payload["family_status"]["repos"]["state"] == "complete"
        assert isinstance(payload["repos"], dict)
        assert isinstance(payload["action_types"], dict)

    def test_archive_file_set_facets_requested_expensive_metadata(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_archive_test_archive(workspace_env)
        with _running_server_without_seed() as (_, base_url):
            payload = _get_json(base_url, "/api/facets?include_expensive=1&budget_ms=0")

        assert isinstance(payload, dict)
        assert payload["budget_exceeded"] is False
        assert payload["deferred_families"] == {}
        assert payload["family_status"]["repos"]["state"] == "complete"
        assert payload["family_status"]["repos"]["reason"] is None
        _assert_facets_match_facade(workspace_env, payload, {}, include_deferred=True)


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

    def test_session_routes_accept_list_emitted_encoded_ids(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            list_payload = cast(dict[str, object], _get_json(base_url, "/api/sessions"))
            items = cast(list[dict[str, object]], list_payload["items"])
            list_item = next(item for item in items if item["id"] == C1)
            emitted_id = cast(str, list_item["id"])
            identity_key = cast(str, cast(dict[str, object], list_item["target_ref"])["identity_key"])

            for route_id in (emitted_id, identity_key):
                encoded = quote(route_id, safe="")
                routes = (
                    f"/api/sessions/{encoded}",
                    f"/api/sessions/{encoded}/messages",
                    f"/api/sessions/{encoded}/read?view=messages",
                    f"/api/sessions/{encoded}/raw",
                    f"/api/sessions/{encoded}/cost",
                    f"/api/sessions/{encoded}/provenance",
                    f"/api/sessions/{encoded}/topology",
                    f"/api/sessions/{encoded}/similar",
                    f"/api/sessions/{encoded}/attachments",
                    f"/api/insights/sessions/{encoded}",
                )
                for route in routes:
                    status, body = _get_json_ex(base_url, route)
                    assert status == 200, (route, body)

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

    def test_import_explain_route_reads_archived_raw_evidence(self, workspace_env: dict[str, Path]) -> None:
        raw_id, source_path = _seed_import_explain_archive(workspace_env)
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, f"/api/import/explain?raw_ref={quote(f'raw:{raw_id}')}"),
            )

        assert payload["mode"] == "import-explain"
        assert payload["source_path"] == f"archive:raw:{raw_id}"
        produced = cast(dict[str, object], payload["produced"])
        assert produced["sessions"] == 1
        assert produced["messages"] == 1
        assert produced["blocks"] == 1
        assert produced["actions"] == 1
        entries = cast(list[dict[str, object]], payload["entries"])
        assert entries[0]["raw_ref"] == f"raw:{raw_id}"
        assert entries[0]["source_path"] == "~/.codex/sessions/route.jsonl"
        assert entries[0]["artifact_kind"] == "session_record_stream"
        assert entries[0]["parser_mode"] == "archived_raw_session"
        assert source_path not in json.dumps(payload)

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
            "/read?view=context-image",
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
        assert payload["route_state"]["state"] == "empty"
        assert payload["route_state"]["reason"] == "Archive contains no sessions."

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
        assert payload["route_state"]["state"] == "no_results"
        assert payload["route_state"]["reason"] == "No sessions matched the active query or filters."
        assert payload["diagnostics"]["message"] == "No sessions matched 'nonexistent_term_xyz'."

    def test_degraded_search_index_returns_route_state_not_zero_results(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        with _running_server(workspace_env) as (_, base_url):
            _degrade_message_fts(workspace_env)
            status, payload = _get_json_ex(base_url, "/api/sessions?query=Hello")
        assert status == 200
        assert payload["total"] is None
        assert payload["hits"] == []
        route_state = cast(dict[str, object], payload["route_state"])
        diagnostics = cast(dict[str, object], payload["diagnostics"])
        reasons = cast(list[dict[str, object]], diagnostics["reasons"])
        assert route_state["state"] == "degraded"
        assert route_state["component"] == "message_fts"
        assert "Search index" in str(route_state["reason"])
        assert reasons[0]["code"] == "search_index_degraded"


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

    def test_query_completions_endpoint_exposes_pipeline_stages(self) -> None:
        with _running_server_without_seed() as (_server, base_url):
            payload = _get_json(
                base_url,
                "/api/query-completions?kind=pipeline-stage&unit=messages&incomplete=sort",
            )

        payload_dict = cast(dict[str, object], payload)
        envelope = payload_dict["query_completions"]
        assert isinstance(envelope, dict)
        envelope_payload = cast(dict[str, object], envelope)
        assert envelope_payload["kind"] == "pipeline-stage"
        candidates = envelope_payload["candidates"]
        assert isinstance(candidates, list)
        candidate_payloads = [cast(dict[str, object], candidate) for candidate in cast(list[object], candidates)]
        assert {"sort by time", "sort by count", "sort by key"}.issubset(
            {candidate["value"] for candidate in candidate_payloads}
        )


class TestReaderActionAffordances:
    def test_web_shell_renders_shared_action_affordance_rail(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/")

        assert status == 200
        assert "text/html" in content_type
        assert "function renderActionAffordanceRail" in body
        assert "Query action affordances" in body
        assert "state.actionAffordances = data.action_affordances || []" in body
        assert "action.input && action.input.unit" in body
        assert "action.execution && action.execution.cardinality_state" in body
        assert "action.output && action.output.format_support" in body
        assert "action.safety && action.safety.safety_level" in body
        assert "action.input_unit" not in body
        assert "action.safety_level" not in body

    def test_action_affordances_endpoint_matches_cli_contract_payload(self) -> None:
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli
        from polylogue.operations.action_contracts import action_affordance_list_payload

        with _running_server_without_seed() as (_server, base_url):
            payload = _get_json(base_url, "/api/action-affordances")

        cli_result = CliRunner().invoke(cli, ["config", "action-affordances"])
        assert cli_result.exit_code == 0, cli_result.output
        cli_payload = json.loads(cli_result.output)
        contract_payload = action_affordance_list_payload().model_dump(mode="json")

        assert payload == cli_payload == contract_payload
        assert isinstance(payload, dict)
        payload_dict = cast(dict[str, object], payload)
        actions = payload_dict["actions"]
        assert isinstance(actions, list)
        action_payloads = [cast(dict[str, object], action) for action in cast(list[object], actions)]
        action_by_id = {str(action["id"]): action for action in action_payloads}

        read = action_by_id["read"]
        assert read["target"] == "selection"
        assert "input_unit" not in read
        assert cast(dict[str, object], read["input"])["unit"] == "query_result_set"
        assert cast(dict[str, object], read["execution"])["cardinality_state"] == "explicit_multi"
        assert cast(dict[str, object], read["safety"])["safety_level"] == "safe"
        assert cast(dict[str, object], read["safety"])["selection_command"] == "polylogue find QUERY then select"
        assert "terminal" in cast(list[object], cast(dict[str, object], read["output"])["destination_support"])

        delete = action_by_id["delete"]
        assert delete["target"] == "selection"
        assert "safety_level" not in delete
        assert cast(dict[str, object], delete["safety"])["safety_level"] == "destructive"
        assert (
            cast(dict[str, object], delete["safety"])["confirmation_command"]
            == "polylogue find QUERY then delete --dry-run"
        )
        assert cast(dict[str, object], delete["availability"])["next_actions"] == ["find"]


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

        _materialize_run_projection(index_db)

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
        from polylogue.archive.query.metadata import terminal_query_source_list

        expression = quote("repo:polylogue")
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, f"/api/query-units?expression={expression}")

        assert status == 400
        assert payload["error"] == "invalid_query"
        assert terminal_query_source_list() in str(payload["detail"])

    def test_query_units_malformed_expression_returns_canonical_error(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.surfaces.payloads import QueryErrorPayload

        expression = quote("messages where (")
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, f"/api/query-units?expression={expression}")

        assert status == 400
        error = QueryErrorPayload.model_validate(payload)
        assert error.ok is False
        assert error.error == "invalid_query"
        assert error.detail
        assert error.field is None
        assert set(payload) == {"ok", "error", "detail", "field"}


class TestReaderViewProfiles:
    def test_read_view_profiles_endpoint_exposes_shared_profile_semantics(self) -> None:
        from polylogue.archive.viewport import read_view_http_capability_payloads

        with _running_server_without_seed() as (_server, base_url):
            payload = _get_json(base_url, "/api/read-view-profiles")

        assert isinstance(payload, dict)
        assert payload["total"] >= 3
        read_views = payload["read_views"]
        assert isinstance(read_views, list)
        profiles = {profile["view_id"]: profile for profile in read_views}
        http_capabilities = read_view_http_capability_payloads()
        endpoint_http_capabilities = {
            view_id: profile.get("http") for view_id, profile in profiles.items() if profile.get("http")
        }
        assert endpoint_http_capabilities == http_capabilities
        assert profiles["raw"]["lossiness"] == "raw"
        assert profiles["raw"]["evidence_policy"] == "required"
        raw_http = cast(dict[str, object], profiles["raw"]["http"])
        assert raw_http["supported"] is True
        assert raw_http["route"] == "/api/sessions/{session_id}/read"
        assert raw_http["formats"] == ["json"]
        assert "recovery" not in profiles
        context_image_http = cast(dict[str, object], profiles["context-image"]["http"])
        assert "max_tokens" in cast(list[str], context_image_http["query_params"])

    def test_read_view_execution_route_returns_messages_raw_context_and_context_image(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        with _running_server(workspace_env) as (_, base_url):
            messages = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=messages&limit=5"),
            )
            raw = cast(dict[str, object], _get_json(base_url, f"/api/sessions/{C1}/read?view=raw"))
            context = cast(dict[str, object], _get_json(base_url, f"/api/sessions/{C1}/read?view=context"))
            context_image = cast(
                dict[str, object],
                _get_json(base_url, f"/api/sessions/{C1}/read?view=context-image"),
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
        assert messages["target_refs"] == [f"session:{C1}"]
        assert messages["lossiness"] == "filtered"
        assert messages["evidence_policy"] == "optional"
        assert "projection flags" in cast(str, messages["privacy_policy"])
        message_actions = cast(dict[str, dict[str, object]], messages["actions"])
        assert message_actions["open"]["enabled"] is True
        message_payload = cast(dict[str, object], messages["payload"])
        assert message_payload["total"] == 1
        assert raw["view"] == "raw"
        assert raw["target_refs"] == [f"session:{C1}"]
        assert raw["lossiness"] == "raw"
        assert "raw provider data" in cast(str, raw["privacy_policy"])
        raw_payload = cast(dict[str, object], raw["payload"])
        assert raw_payload["id"] == C1
        assert context["view"] == "context"
        context_payload = cast(dict[str, object], context["payload"])
        assert context_payload["preamble_version"] == "1.0"
        assert context_image["view"] == "context-image"
        context_payload = cast(dict[str, object], context_image["payload"])
        # Context pack now returns the shared ContextImage payload compiled from
        # the seed session through compile_context.
        context_spec = cast(dict[str, object], context_payload["spec"])
        assert context_spec["seed_refs"] == [f"session:{C1}"]
        assert "messages" in cast(list[str], context_spec["read_views"])
        projection_spec = cast(dict[str, object], context_payload["projection_spec"])
        projection_selection = cast(dict[str, object], projection_spec["selection"])
        projection = cast(dict[str, object], projection_spec["projection"])
        render = cast(dict[str, object], projection_spec["render"])
        assert projection_selection["refs"] == [f"session:{C1}"]
        assert projection["families"] == ["context", "messages"]
        assert projection["body_policy"] == "authored-dialogue"
        assert {"tool_use", "tool_result", "function_call", "function_call_output"} <= set(
            cast(list[str], projection["exclude_block_kinds"])
        )
        assert render["layout"] == "context-image"
        segments = cast(list[dict[str, object]], context_payload["segments"])
        assert any(segment.get("payload_kind") == "messages" for segment in segments)
        assert isinstance(context_payload["token_estimate"], int)
        assert context_payload["token_estimate"] > 0
        assert isinstance(context_payload["omitted"], list)
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


class TestReaderAssertionEndpoint:
    def test_archive_debt_endpoint_returns_shared_payload(self, workspace_env: dict[str, Path]) -> None:
        _seed_assertion_claims(workspace_env)
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, "/api/archive-debt?kind=assertion-candidate&only_actionable=1&limit=5"),
            )

        assert payload["mode"] == "archive-debt-list"
        totals = cast(dict[str, object], payload["totals"])
        assert totals["total"] == 1
        rows = cast(list[dict[str, object]], payload["rows"])
        assert rows[0]["kind"] == "assertion-candidate"
        assert rows[0]["status"] == "actionable"

    def test_archive_debt_endpoint_defaults_to_bounded_page(self, workspace_env: dict[str, Path]) -> None:
        _seed_many_assertion_candidates(workspace_env, 55)
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, "/api/archive-debt?kind=assertion-candidate&only_actionable=1"),
            )

        totals = cast(dict[str, object], payload["totals"])
        rows = cast(list[dict[str, object]], payload["rows"])
        assert totals["total"] == 50
        assert len(rows) == 50

    def test_archive_debt_endpoint_reports_missing_index_tier(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            (workspace_env["archive_root"] / "index.db").unlink()
            payload = cast(dict[str, object], _get_json(base_url, "/api/archive-debt?kind=archive-tier&limit=10"))

        rows = cast(list[dict[str, object]], payload["rows"])
        assert any(row["kind"] == "archive-tier" and row["subject_ref"] == "archive-tier:index" for row in rows)

    def test_operational_web_payloads_redact_configured_archive_paths(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The HTTP projection must not inherit CLI path diagnostics."""
        archive_root = workspace_env["archive_root"]
        symlink_root = archive_root.parent / "web-visible-archive"
        symlink_root.symlink_to(archive_root, target_is_directory=True)
        monkeypatch.setattr("polylogue.paths.archive_root", lambda: symlink_root)
        monkeypatch.setattr(
            "polylogue.daemon.http.get_status_snapshot_payload",
            lambda: {
                "component_readiness": {},
                "status_snapshot": {
                    "state": "stale",
                    "captured_at": None,
                    "age_s": 91,
                    "refresh_error": f"could not refresh {symlink_root} (resolved {archive_root})",
                },
            },
        )
        with _running_server(workspace_env, seeded=True) as (_, base_url):
            overview = _get_json(base_url, "/api/overview")
            (archive_root / "index.db").unlink()
            provider = _get_json(base_url, "/api/provider-usage")
            debt = _get_json(base_url, "/api/archive-debt?kind=archive-tier")

        text = json.dumps({"provider": provider, "debt": debt, "overview": overview})
        assert "archive_root" not in text
        assert str(symlink_root) not in text
        assert str(archive_root) not in text
        assert "[archive]" in text

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
        with _running_server_without_seed() as (_, base_url):
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
            "/api/overview",
            "/api/sessions",
            "/api/sessions/claude-code-session:c1",
            "/api/sessions/claude-code-session:c1/messages",
            "/api/sessions/claude-code-session:c1/read?view=messages",
            "/api/sessions/claude-code-session:c1/read?view=context-image&include_messages=0",
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


class TestCockpitAggregateRoutes:
    def test_overview_is_bounded_and_reuses_archive_summary_projection(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, "/api/overview"))

        assert payload["mode"] == "cockpit-overview"
        assert cast(dict[str, object], payload["totals"])["sessions"] == 3
        assert len(cast(list[object], payload["recent"])) <= 6
        assert cast(dict[str, object], payload["readiness"])

    @pytest.mark.parametrize(
        ("is_error", "exit_code", "expected_outcomes"),
        [
            (0, 2, {"ok": 0, "failed": 1, "unknown": 0}),
            (1, 0, {"ok": 1, "failed": 0, "unknown": 0}),
        ],
    )
    def test_evidence_summary_matches_structural_tool_relations(
        self,
        workspace_env: dict[str, Path],
        is_error: int,
        exit_code: int,
        expected_outcomes: dict[str, int],
    ) -> None:
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import BlockType, Provider
        from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore(workspace_env["archive_root"]) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="evidence-summary",
                    title="Structural evidence",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m-evidence",
                            role=Role.ASSISTANT,
                            text="ran test",
                            blocks=[
                                ParsedContentBlock(type=BlockType.TOOL_USE, text="pytest", tool_id="tool-evidence"),
                                ParsedContentBlock(
                                    type=BlockType.TOOL_RESULT,
                                    text="failed",
                                    tool_id="tool-evidence",
                                ),
                            ],
                        )
                    ],
                )
            )

        with sqlite3.connect(workspace_env["archive_root"] / "index.db") as conn:
            conn.execute(
                "UPDATE blocks SET tool_result_is_error = ?, tool_result_exit_code = ? WHERE session_id = ? AND block_type = 'tool_result'",
                (is_error, exit_code, "codex-session:evidence-summary"),
            )
            conn.commit()

        session_id = "codex-session:evidence-summary"
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = cast(dict[str, object], _get_json(base_url, f"/api/sessions/{session_id}/evidence-summary"))

        assert payload["tool_calls"] == 1
        outcomes = cast(dict[str, object], payload["outcomes"])
        assert outcomes == expected_outcomes
        assert cast(dict[str, object], payload["cost"])["total_usd"] == 0.0

    def test_evidence_summary_composes_prefix_sharing_tool_evidence(self, workspace_env: dict[str, Path]) -> None:
        """The evidence strip and transcript must describe the same composed
        prefix-sharing session, including inherited tool outcomes."""
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import BlockType, BranchType, Provider
        from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        replayed_tool_blocks = [
            ParsedContentBlock(type=BlockType.TOOL_USE, text="pytest", tool_id="tool-parent"),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                text="ok",
                tool_id="tool-parent",
            ),
        ]
        with ArchiveStore(workspace_env["archive_root"]) as archive:
            parent_id = archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="evidence-parent",
                    title="Parent evidence",
                    messages=[
                        ParsedMessage(provider_message_id="p0", role=Role.USER, text="start"),
                        ParsedMessage(
                            provider_message_id="p1",
                            role=Role.ASSISTANT,
                            text="ran pytest",
                            blocks=replayed_tool_blocks,
                        ),
                    ],
                )
            )
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="evidence-child",
                    title="Child evidence",
                    parent_session_provider_id="evidence-parent",
                    branch_type=BranchType.FORK,
                    messages=[
                        ParsedMessage(provider_message_id="c0", role=Role.USER, text="start"),
                        ParsedMessage(
                            provider_message_id="c1",
                            role=Role.ASSISTANT,
                            text="ran pytest",
                            blocks=replayed_tool_blocks,
                        ),
                        ParsedMessage(provider_message_id="c2", role=Role.USER, text="child tail"),
                    ],
                )
            )

        with sqlite3.connect(workspace_env["archive_root"] / "index.db") as conn:
            conn.execute(
                "UPDATE blocks SET tool_result_is_error = 0, tool_result_exit_code = 0 "
                "WHERE session_id = ? AND block_type = 'tool_result'",
                (parent_id,),
            )
            conn.commit()

        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = cast(
                dict[str, object], _get_json(base_url, "/api/sessions/codex-session:evidence-child/evidence-summary")
            )

        assert payload["tool_calls"] == 1
        assert payload["outcomes"] == {"ok": 1, "failed": 0, "unknown": 0}

    def test_message_endpoint_clamps_oversized_pages(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.query.spec import MAX_QUERY_LIMIT

        with _running_server(workspace_env) as (_, base_url):
            payload = cast(
                dict[str, object], _get_json(base_url, f"/api/sessions/{C1}/messages?limit=999999&offset=-10")
            )

        assert payload["limit"] == MAX_QUERY_LIMIT
        assert payload["offset"] == 0


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
            "/api/sessions/claude-code-session:c1/read?view=context-image&include_messages=0",
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
        from polylogue.surfaces.payloads import RouteReadinessPayload, SessionListResponse

        route_state = RouteReadinessPayload(
            state="empty", route="/api/sessions", reason="Archive contains no sessions."
        )
        r = SessionListResponse(items=(), total=0, limit=50, offset=0, route_state=route_state)
        d = r.model_dump(mode="json")
        assert d["items"] == []
        assert d["total"] == 0
        assert d["limit"] == 50
        assert d["offset"] == 0
        assert d["route_state"]["state"] == "empty"

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
        assert d["route_state"] is None

    def test_facets_response_shape(self) -> None:
        from polylogue.surfaces.payloads import FacetFamilyStatusPayload, FacetsResponse, FacetTimeRange

        r = FacetsResponse(
            scoped_to_query=False,
            generated_at="2026-06-22T00:00:00Z",
            budget_exceeded=True,
            complete_families=("total_counts", "origins", "tags"),
            deferred_families={"repos": "deferred_by_default"},
            family_status={"repos": FacetFamilyStatusPayload(state="deferred", reason="deferred_by_default")},
            origins={"claude-code-session": 10, "chatgpt-export": 5},
            role_counts={"user": 12, "assistant": 8},
            material_origins={"human_authored": 7, "runtime_protocol": 5},
            total_sessions=15,
            total_messages=100,
            time_range=FacetTimeRange(min="2024-01-01", max="2024-12-31"),
        )
        d = r.model_dump(mode="json")
        assert d["scoped_to_query"] is False
        assert d["origins"] == {"claude-code-session": 10, "chatgpt-export": 5}
        assert d["role_counts"] == {"user": 12, "assistant": 8}
        assert d["material_origins"] == {"human_authored": 7, "runtime_protocol": 5}
        assert d["total_sessions"] == 15
        assert d["time_range"]["min"] == "2024-01-01"
        assert d["generated_at"] == "2026-06-22T00:00:00Z"
        assert d["budget_exceeded"] is True
        assert d["deferred_families"] == {"repos": "deferred_by_default"}
        assert d["family_status"]["repos"]["state"] == "deferred"

    def test_assertion_claim_payloads_are_shared_surface_dtos(self) -> None:
        from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
        from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope
        from polylogue.surfaces.payloads import AssertionClaimListPayload, AssertionClaimPayload

        envelope = ArchiveAssertionEnvelope(
            assertion_id="claim-1",
            scope_ref="repo:polylogue",
            target_ref="session:c1",
            key=None,
            kind=AssertionKind.DECISION,
            value=None,
            body_text="Keep assertion payloads shared.",
            author_ref="agent:test",
            author_kind="agent",
            evidence_refs=["message:m1"],
            status=AssertionStatus.ACTIVE,
            visibility=AssertionVisibility.PRIVATE,
            confidence=0.9,
            staleness=None,
            context_policy={"inject": False},
            supersedes=[],
            created_at_ms=1,
            updated_at_ms=2,
        )

        item = AssertionClaimPayload.from_envelope(envelope)
        payload = AssertionClaimListPayload(items=(item,), total=1, limit=20, statuses=(AssertionStatus.ACTIVE,))
        dump = payload.model_dump(mode="json", exclude_none=True)

        assert dump["items"][0]["assertion_id"] == "claim-1"
        assert dump["items"][0]["evidence_refs"] == ["message:m1"]
        assert dump["statuses"] == ["active"]


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

    def test_session_list_accepts_exists_supported_unit_query(self, workspace_env: dict[str, Path]) -> None:
        """run/observed-event/context-snapshot now lower to exists session selectors."""
        expression = quote("runs where role:subagent")
        with _running_server(workspace_env) as (_, base_url):
            status, body = _get_json_ex(base_url, f"/api/sessions?query={expression}")
        assert status == 200
        assert body.get("ok") is not False

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
        assert isinstance(payload["role_counts"], dict)
        assert isinstance(payload["message_types"], dict)
        assert isinstance(payload["material_origins"], dict)

    def test_archive_facets_expose_provider_role_and_material_origin_split(
        self, workspace_env: dict[str, Path]
    ) -> None:
        """Facets make authoredness visible instead of hiding behind role=user."""
        from polylogue.core.enums import MaterialOrigin
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        db_path = archive_root / "index.db"
        (
            SessionBuilder(db_path, "facet-authored-split")
            .provider("claude-code")
            .title("Facet authored split")
            .add_message(
                "m-runtime",
                role="user",
                text="<bash-stdout>runtime output</bash-stdout>",
                material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
            )
            .add_message(
                "m-human",
                role="user",
                text="Run the probe",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
            )
            .add_message(
                "m-assistant",
                role="assistant",
                text="Probe complete",
                material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
            )
            .save()
        )

        with _running_server_without_seed() as (_, base_url):
            payload = cast(
                dict[str, object],
                _get_json(base_url, "/api/facets?origin=claude-code-session&include_deferred=1"),
            )

        role_counts = cast(dict[str, int], payload["role_counts"])
        material_origins = cast(dict[str, int], payload["material_origins"])
        message_types = cast(dict[str, int], payload["message_types"])
        assert role_counts["user"] == 2
        assert role_counts["assistant"] == 1
        assert material_origins["human_authored"] == 1
        assert material_origins["runtime_protocol"] == 1
        assert material_origins["assistant_authored"] == 1
        assert message_types["message"] == 3
        _assert_facets_match_facade(
            workspace_env,
            payload,
            {"origin": ["claude-code-session"]},
            include_deferred=True,
        )

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
        with _running_server_without_seed() as (_, base_url):
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
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "var readiness = s.component_readiness || {};" in body
        assert "index DB:" in body
        assert "renderFtsChip(readiness.search || null, s.fts_readiness || {})" in body
        assert (
            "renderMaterializationChip(readiness.raw_materialization || null, s.raw_materialization_readiness || {})"
            in body
        )
        assert "renderSemanticChip(readiness.embeddings || null)" in body
        assert "renderInsightChip(readiness.session_profiles || null, s.insight_freshness || {})" in body
        assert "renderIngestChip(readiness.daemon_ingest || null, s.live || {})" in body
        assert "renderBrowserCaptureChip(readiness.browser_capture || null, s.browser_capture || {})" in body
        assert "function renderComponentReadinessChip(" in body
        assert "function renderMaterializationChip(" in body
        assert "function readinessQuality(" in body
        assert "function renderBrowserCaptureChip(" in body

    def test_status_strip_starts_with_unknown_counts_not_zero(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert "convs checking" in body
        assert "msgs checking" in body
        assert "0 convs" not in body
        assert "0 msgs" not in body
        assert "updateStatusCountsUnknown" in body
        assert "s.total_sessions != null" in body
        assert "s.total_messages != null" in body

    def test_web_shell_models_failed_route_panels_and_retries(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert "routeStates" in body
        assert "function routeErrorDetails" in body
        assert "function fallbackCommand" in body
        assert "function renderRouteStateNotice" in body
        assert "function renderInlineRouteFailure" in body
        assert "setRouteState('sessionList'" in body
        assert "renderRouteStateNotice('sessionList'" in body
        assert "Retry" in body
        assert "stale_available" in body
        assert "/api/status" in body
        assert "/api/facets" in body
        assert "/api/read-view-profiles" in body
        assert "/api/sessions/" in body

    def test_facets_loader_cancels_previous_request(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert "AbortController" in body
        assert "state.inFlight.facets.controller.abort()" in body
        assert "controller.signal" in body
        assert "state.inFlight.facets.token !== token" in body
        assert "timeoutMs: opts.timeoutMs || 5000" in body
        assert "e.name === 'AbortError' && !e.timed_out" in body
        assert "include_deferred" in body
        assert "budget_ms" in body

    def test_missing_read_view_profiles_route_has_retry_fallback(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert "setRouteState('readViewProfiles'" in body
        assert "Read profiles" in body
        assert "loadReadViewProfiles()" in body
        assert "curl -fsS http://127.0.0.1:8766" in body

    def test_session_detail_failure_does_not_leave_bare_loading(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert "selectedLoadError" in body
        assert "Session detail unavailable" in body
        assert "Loading session detail" in body
        assert "loadSessionFromError" in body
        assert "split('?')[0]" in body
        assert "function loadMoreSessionMessages()" in body
        assert "Load more messages" in body
        assert "/api/sessions/" in body
        assert "Overview readiness is" in body
        assert "data-overview-snapshot-state" in body
        assert "Evidence summary unavailable" in body
        assert "function retryEvidenceSummary(id)" in body

    def test_fts_chip_keeps_legacy_tri_state_fallback(self, workspace_env: dict[str, Path]) -> None:
        """``renderFtsChip`` must still distinguish ok / partial /
        unavailable when an older payload lacks ``component_readiness``.
        """
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "function renderFtsChip(" in body
        assert "component && component.state !== 'unknown'" in body
        assert "'FTS: ok'" in body
        assert "'FTS: partial'" in body
        assert "'FTS: unavailable'" in body

    def test_materialization_semantic_and_ingest_readiness_chips_render_present(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """Materialization, semantic, and ingest state have first-class
        status-strip chips fed by ``component_readiness``.
        """
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert 'id="status-materialization"' in body
        assert 'id="status-semantic"' in body
        assert 'id="status-ingest"' in body
        assert "function renderMaterializationChip(" in body
        assert "counts.raw_artifact_count" in body
        assert "counts.materialized_raw_artifact_count" in body
        assert "counts.join_gap_count" in body
        assert "function renderSemanticChip(" in body
        assert "function renderIngestChip(" in body
        assert "'materialization'" in body
        assert "'semantic'" in body
        assert "'ingest'" in body

    def test_browser_capture_readiness_chip_consumes_safe_status_payload(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """The web workbench displays browser-capture readiness from the
        daemon status envelope without learning the receiver's local spool path.
        """
        with _running_server_without_seed() as (_, base_url):
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

    def test_dev_loop_chip_consumes_branch_local_metadata(self, workspace_env: dict[str, Path]) -> None:
        """The web shell can surface branch-local run metadata when present."""

        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert 'id="status-dev-loop"' in body
        assert "function renderDevLoopChip(" in body
        assert "/api/dev-loop" in body

    def test_web_shell_surfaces_latest_api_request_diagnostics(self, workspace_env: dict[str, Path]) -> None:
        """The web shell has local request diagnostics for UI/API failures."""

        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")

        assert 'id="status-api-debug"' in body
        assert "function nextApiRequestId(" in body
        assert "'X-Request-ID': requestId" in body
        assert "function renderApiDebugChip(" in body
        assert "response_summary" in body
        assert "invalid_json" in body

    def test_insight_freshness_chip_keeps_legacy_fallback(self, workspace_env: dict[str, Path]) -> None:
        """Session insight freshness gets its own status-strip chip. It now
        prefers ``component_readiness.session_profiles`` and keeps the
        previous freshness-payload fallback for old status snapshots."""
        with _running_server_without_seed() as (_, base_url):
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
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "No sessions in archive. Run `polylogued run`" in body
        assert "No results for query=" in body
        assert "No sessions from origin=" in body
        assert "Press Esc to clear" in body

    def test_workspace_actions_have_disabled_state_with_tooltips(self, workspace_env: dict[str, Path]) -> None:
        """Stack/Compare/Save/Recall workspace buttons must expose a
        disabled state with explanatory tooltips when context is
        insufficient — disabled actions are part of the MK3 design."""
        with _running_server_without_seed() as (_, base_url):
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
        with _running_server_without_seed() as (_, base_url):
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
        with _running_server_without_seed() as (_, base_url):
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
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        # Naming conflict UX: detect locally against state.savedViews, prompt
        # for overwrite confirmation, replace via DELETE+POST. Empty/whitespace
        # names are rejected before hitting the wire.
        assert "Saved view name cannot be empty" in body
        assert "already exists. Overwrite it?" in body
        assert "Failed to replace existing view" in body

    def test_saved_view_list_exposes_delete_action(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        # The inspector list must render a Delete button per saved view; the
        # JS handler is reachable so the operator can prune stale views.
        assert "function deleteSavedView(" in body
        assert "Delete saved view" in body
        assert 'onclick="deleteSavedView(' in body

    def test_saved_view_empty_state_is_actionable(self, workspace_env: dict[str, Path]) -> None:
        with _running_server_without_seed() as (_, base_url):
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
# polylogue.local_reader.selection — multi-select selection operations (#1119)
# ---------------------------------------------------------------------------


class TestReaderSelectionOperations:
    """``polylogue.local_reader.selection``: selection toolbar + per-session envelope.

    The selection surface composes existing daemon routes: ``/api/user/marks``
    carries route-backed overlay mutations, and ``/api/sessions/{id}``
    carries export reads. Delete and re-embed are exposed only through a
    preview overlay because the daemon has no corresponding mutation routes yet.

    These tests assert the shipped HTML carries every load-bearing region
    hook and that the underlying tag endpoint accepts the per-session
    POSTs the selection toolbar drives — that is the contract the JS depends on.
    Pixel-level assertions are deliberately avoided so reader visual
    iteration does not invalidate the smoke.
    """

    SELECTION_REGION_HOOKS = (
        "selection-toolbar",
        "selection-select-all",
        "selection-clear",
        'data-selection-action="tag-star"',
        'data-selection-action="tag-pin"',
        'data-selection-action="tag-archive"',
        'data-selection-action="export"',
        'data-selection-action="delete-preview"',
        'data-selection-action="reembed-preview"',
        "selection-preview",
        "selection-preview-confirm",
        "selectionApplyMark",
        "selectionExport",
        "openSelectionPreview",
        "confirmSelectionPreview",
        "renderSelectionToolbar",
        "isSelectionSelected",
        "selectionSet",
        "no_endpoint",
    )

    def test_selection_toolbar_regions_present_in_shell(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, content_type, body = _get_text(base_url, "/")
        assert status == 200
        assert "text/html" in content_type
        for hook in self.SELECTION_REGION_HOOKS:
            assert hook in body, f"web shell missing selection hook {hook!r}"

    def test_selection_envelope_keys_documented_in_shell(self, workspace_env: dict[str, Path]) -> None:
        """The shipped JS must reference the per-session status keys the
        selection envelope contract uses (succeeded/failed/skipped + dryRun + action).
        These are the field names the UI surfaces to the operator after a selection
        op; renaming any of them silently would break the rendered status."""
        with _running_server_without_seed() as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        for key in ("succeeded", "failed", "skipped", "dryRun", "action"):
            assert key in body, f"selection envelope key {key!r} missing from shell"

    def test_selection_tag_drives_existing_marks_endpoint(self, workspace_env: dict[str, Path]) -> None:
        """The selection toolbar issues per-session POSTs against
        ``/api/user/marks``. This simulates that loop server-side to lock the
        contract: the daemon must accept the same payload shape the selection JS
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

    def test_query_set_export_uses_session_detail_endpoint(self, workspace_env: dict[str, Path]) -> None:
        """Query-set export concatenates per-session GETs. This pins the
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
