"""Reader smoke tests — page structure, API contracts, degraded states (#865).

Boots the production ``DaemonAPIHTTPServer`` against a synthetic on-disk
archive and exercises the live HTTP surface via ``urllib``. The HTML
payload served at ``/`` is asserted at the DOM-shape level (semantic
selectors, never pixel diffs); JSON envelopes are asserted by shape so
any regression in the daemon's contract surface fails here loudly.

This is the documented reader visual smoke lane (see
``docs/visual-evidence.md``). The lane runs as part of the standard
unit suite and ``devtools verify``; a separate ``devtools lab-scenario``
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
from urllib.request import Request, urlopen

import pytest

POLYLOGUE_LOCAL_PATH_PREFIXES = ("/home/", "/Users/", "/realm/", "/var/", "/etc/")


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
    from polylogue.core.sources import origin_from_provider
    from polylogue.types import Provider

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
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.types import BlockType, Provider

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
                            content_blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Hello reader")],
                        )
                    ],
                )
            )


def _seed_archive_test_archive(workspace: dict[str, Path]) -> str:
    from polylogue.archive.message.roles import Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.types import BlockType, Provider

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
                        content_blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Hello archive reader")],
                    )
                ],
            )
        )


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
        ):
            assert region in body, f"web shell missing region hook {region!r}"

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
        assert "provider" not in payload
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
        assert status == 201
        assert created == {
            "target_type": "session",
            "target_id": "claude-code-session:c1",
            "session_id": "claude-code-session:c1",
            "message_id": None,
            "mark_type": "star",
            "created": True,
        }
        assert status2 == 200
        assert duplicate == {
            "target_type": "session",
            "target_id": "claude-code-session:c1",
            "session_id": "claude-code-session:c1",
            "message_id": None,
            "mark_type": "star",
            "created": False,
        }
        mark_items = cast(list[dict[str, object]], marks_payload["items"])
        assert mark_items[0]["mark_type"] == "star"
        assert mark_items[0]["target_type"] == "session"
        assert mark_items[0]["target_id"] == "claude-code-session:c1"
        assert delete_status == 200
        assert deleted == {
            "target_type": "session",
            "target_id": "claude-code-session:c1",
            "session_id": "claude-code-session:c1",
            "message_id": None,
            "mark_type": "star",
            "deleted": True,
        }
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

        listed_payload = cast(dict[str, object], listed)
        conv_note_payload = cast(dict[str, object], conv_note)
        msg_note_payload = cast(dict[str, object], msg_note)
        fetched_payload = cast(dict[str, object], fetched)
        items = cast(list[dict[str, object]], listed_payload["items"])
        assert conv_status == 201
        assert conv_note_payload["target_type"] == "session"
        assert conv_note_payload["target_id"] == "claude-code-session:c1"
        assert msg_status == 201
        assert msg_note_payload["target_type"] == "message"
        assert msg_note_payload["target_id"] == "claude-code-session:c1:m-c1"
        assert fetched_payload["note_text"] == "Important request"
        assert {item["annotation_id"] for item in items} == {"ann-c1", "ann-m1"}
        assert delete_status == 200
        assert deleted == {"annotation_id": "ann-c1", "deleted": True}

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
        assert saved_payload["created"] is True
        assert saved_payload["query"] == {"limit": 5, "origin": "claude-code-session", "query": "auth"}
        assert listed_payload["total"] == 1
        assert fetched_payload["view_id"] == "view-auth"
        assert fetched_payload["query_json"] == '{"limit": 5, "origin": "claude-code-session", "query": "auth"}'
        assert delete_status == 200
        assert deleted == {"view_id": "view-auth", "deleted": True}

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
        assert saved_payload["created"] is True
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
        assert deleted == {"pack_id": "pack-auth", "deleted": True}

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
        assert saved_payload["created"] is True
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
        assert deleted == {"workspace_id": "workspace-auth", "deleted": True}


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
            messages=1,
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

    def test_fts_chip_renders_tri_state_not_binary(self, workspace_env: dict[str, Path]) -> None:
        """``renderFtsChip`` must distinguish ok / partial / unavailable
        rather than collapsing partial readiness into the ok bucket. The
        prior code rendered ``messages_ready ? 'ok' : '--'`` which hides
        partial message FTS readiness entirely.
        """
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "function renderFtsChip(" in body
        assert "'FTS: ok'" in body
        assert "'FTS: partial'" in body
        assert "'FTS: unavailable'" in body

    def test_insight_freshness_chip_render_present(self, workspace_env: dict[str, Path]) -> None:
        """Session insight freshness gets its own status-strip chip so
        operators can tell whether session profiles are computed, partial,
        or stale without opening the inspector."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "function renderInsightChip(" in body
        assert 'id="status-insights"' in body
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
        assert saved_payload["name"] == "Reader UI flow"
        assert listed_payload["total"] == 1
        assert items[0]["view_id"] == "view-ui"
        assert delete_status == 200
        assert deleted == {"view_id": "view-ui", "deleted": True}
        assert empty_payload == {"items": [], "total": 0}


# ---------------------------------------------------------------------------
# polylogue.local_reader.bulk — multi-select bulk operations (#1119)
# ---------------------------------------------------------------------------


class TestReaderBulkOperations:
    """``polylogue.local_reader.bulk``: selection toolbar + per-session envelope.

    The bulk surface is composed client-side over existing daemon
    endpoints (``/api/user/marks`` for tag mutations, ``/api/sessions/{id}``
    for export). Delete and re-embed are exposed only through a preview
    overlay because the daemon has no corresponding mutation routes yet.

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
