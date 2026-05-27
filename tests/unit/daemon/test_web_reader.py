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


def _seed_empty_schema(workspace: dict[str, Path]) -> None:
    import sqlite3

    from polylogue.storage.sqlite.schema_ddl_archive import (
        ARCHIVE_STORAGE_DDL,
        MESSAGE_FTS_DDL,
        RECALL_PACKS_DDL,
        SAVED_VIEWS_DDL,
        USER_ANNOTATIONS_DDL,
        USER_MARKS_DDL,
    )

    db = _archive_db_path(workspace)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(ARCHIVE_STORAGE_DDL)
    conn.executescript(MESSAGE_FTS_DDL)
    conn.executescript(USER_MARKS_DDL)
    conn.executescript(USER_ANNOTATIONS_DDL)
    conn.executescript(SAVED_VIEWS_DDL)
    conn.executescript(RECALL_PACKS_DDL)
    conn.commit()
    conn.close()


def _seed_test_db(workspace: dict[str, Path]) -> None:
    """Seed a synthetic archive with three single-message conversations."""
    import sqlite3

    from polylogue.storage.sqlite.schema_ddl_archive import (
        ARCHIVE_STORAGE_DDL,
        MESSAGE_FTS_DDL,
        RECALL_PACKS_DDL,
        SAVED_VIEWS_DDL,
        USER_ANNOTATIONS_DDL,
        USER_MARKS_DDL,
    )

    db = _archive_db_path(workspace)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(ARCHIVE_STORAGE_DDL)
    conn.executescript(MESSAGE_FTS_DDL)
    conn.executescript(USER_MARKS_DDL)
    conn.executescript(USER_ANNOTATIONS_DDL)
    conn.executescript(SAVED_VIEWS_DDL)
    conn.executescript(RECALL_PACKS_DDL)
    for cid, prov, title in [
        ("c1", "claude-code", "Claude Code session about authentication"),
        ("c2", "chatgpt", "ChatGPT debugging conversation"),
        ("c3", "claude-ai", "Claude AI brainstorm thread"),
    ]:
        conn.execute(
            "INSERT INTO conversations(conversation_id, source_name, provider_conversation_id, title, content_hash, version) VALUES(?,?,?,?,?,?)",
            (cid, prov, f"p-{cid}", title, f"hash-{cid}", 1),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, source_name, content_hash, version) VALUES(?,?,?,?,?,?,?)",
            (f"m-{cid}", cid, "user", "Hello reader", prov, f"mhash-{cid}", 1),
        )
    conn.commit()
    conn.close()


def _archive_db_path(workspace: dict[str, Path]) -> Path:
    return workspace["data_root"] / "polylogue" / "polylogue.db"


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
            "renderConversations",
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
            payload = _get_json(base_url, "/api/conversations")
        assert isinstance(payload, dict)
        assert payload["total"] == 3
        assert len(payload["items"]) == 3
        row = next(item for item in payload["items"] if item["id"] == "c1")
        assert row["target_ref"] == {
            "target_type": "conversation",
            "target_id": "c1",
            "conversation_id": "c1",
            "identity_key": "conversation:c1",
        }
        assert row["anchor"] == "conversation-c1"
        assert row["actions"]["open"]["enabled"] is True
        assert row["actions"]["annotate"]["enabled"] is True
        assert row["actions"]["annotate"]["state"] == "enabled"

    def test_facets_envelope_includes_scoped_flag(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert "scoped_to_query" in payload
        assert "providers" in payload

    def test_facets_provider_filter_scopes_counts(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?provider=chatgpt")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is True
        assert payload["total_conversations"] == 1
        assert payload["providers"] == {"chatgpt": 1}
        assert "claude-code" not in payload["providers"]

    def test_facets_query_filter_scopes_counts(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?query=Hello")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is True
        assert payload["total_conversations"] == 3

    def test_query_search_envelope_carries_hit_target_refs(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/conversations?query=Hello")
        assert isinstance(payload, dict)
        assert payload["total"] == 3
        assert len(payload["hits"]) == 3
        # SearchEnvelope contract (#1266): every hit carries a `conversation`
        # identity payload and a `match` evidence payload.
        assert payload["query"] == "Hello"
        assert payload["retrieval_lane"] in {"dialogue", "auto"}
        assert payload["ranking_policy"] == "mixed-bm25-rrf-vector"
        assert payload["ranking_policy_version"] == "1"
        hit = next(item for item in payload["hits"] if item["conversation"]["id"] == "c1")
        assert hit["conversation"]["target_ref"]["identity_key"] == "conversation:c1"
        assert hit["conversation"]["anchor"] == "conversation-c1"
        # The typed TargetRefPayload includes block_index (defaulting to None)
        # for message targets; we assert the load-bearing identity fields.
        target_ref = hit["match"]["target_ref"]
        assert target_ref["target_type"] == "message"
        assert target_ref["target_id"] == "m-c1"
        assert target_ref["conversation_id"] == "c1"
        assert target_ref["message_id"] == "m-c1"
        assert target_ref["identity_key"] == "message:c1:m-c1"
        assert hit["match"]["anchor"] == "message-m-c1"
        assert hit["match"]["actions"]["copy_text"]["enabled"] is True


# ---------------------------------------------------------------------------
# polylogue.local_reader.conversation — single conversation/detail state
# ---------------------------------------------------------------------------


class TestReaderConversationState:
    """``polylogue.local_reader.conversation``: header + messages + raw envelope."""

    def test_conversation_detail_returns_header_and_messages(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/conversations/c1")
        assert isinstance(payload, dict)
        assert payload["id"] == "c1"
        assert payload["provider"] == "claude-code"
        assert payload["title"].startswith("Claude Code")
        assert payload["target_ref"]["identity_key"] == "conversation:c1"
        assert payload["anchor"] == "conversation-c1"
        assert payload["messages"][0]["target_ref"]["identity_key"] == "message:c1:m-c1"
        assert payload["messages"][0]["anchor"] == "message-m-c1"

    def test_conversation_messages_envelope_carries_messages_and_total(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/conversations/c1/messages")
        assert isinstance(payload, dict)
        assert payload["total"] == 1
        message = payload["messages"][0]
        assert message["text"] == "Hello reader"
        assert message["target_ref"] == {
            "target_type": "message",
            "target_id": "m-c1",
            "conversation_id": "c1",
            "message_id": "m-c1",
            "identity_key": "message:c1:m-c1",
        }
        assert message["anchor"] == "message-m-c1"
        assert message["actions"]["copy_text"]["enabled"] is True
        assert message["actions"]["annotate"]["enabled"] is True
        assert message["actions"]["annotate"]["state"] == "enabled"

    def test_unknown_conversation_yields_404(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, _, _ = _get_text(base_url, "/api/conversations/does-not-exist")
        assert status == 404


# ---------------------------------------------------------------------------
# polylogue.local_reader.workspace — stack and compare route data
# ---------------------------------------------------------------------------


class TestReaderWorkspaceRoutes:
    """``polylogue.local_reader.workspace``: stack/compare workspace routes."""

    def test_stack_route_returns_resolved_and_missing_targets(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/stack?ids=c1,missing-conv&focus=c1")

        result = cast(dict[str, object], payload)
        items = cast(list[dict[str, object]], result["items"])
        assert result["mode"] == "stack"
        assert result["focus"] == "c1"
        assert result["resolved_count"] == 1
        assert result["degraded_count"] == 1
        assert items[0]["status"] == "resolved"
        assert items[0]["identity_key"] == "conversation:c1"
        conversation = cast(dict[str, object], items[0]["conversation"])
        assert conversation["id"] == "c1"
        assert items[1] == {
            "target_type": "conversation",
            "target_id": "missing-conv",
            "conversation_id": "missing-conv",
            "status": "missing",
            "disabled_reason": "conversation_not_found",
        }

    def test_stack_route_rejects_empty_ids(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, "/api/stack")

        assert status == 400
        assert payload["error"] == "invalid_request"

    def test_compare_route_returns_message_pairs_and_degraded_side(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/compare?left=c1&right=missing-conv&align=prompt")

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
        assert left["id"] == "c1"
        assert right["status"] == "missing"
        assert pairs[0]["left"] is not None
        assert pairs[0]["right"] is None
        assert pairs[0]["status"] == "unpaired"

    def test_compare_route_two_conversations_surface_diff_and_metadata(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/compare?left=c1&right=c2&align=prompt")

        result = cast(dict[str, object], payload)
        # Both sides present → no degradation, metadata diff populated.
        assert result["degraded_count"] == 0
        assert result["degraded_sides"] == []
        metadata = cast(dict[str, dict[str, object]], result["metadata_diff"])
        # Providers differ between the seeded conversations.
        assert metadata["provider"]["status"] == "changed"
        assert metadata["title"]["status"] == "changed"
        pairs = cast(list[dict[str, object]], result["pairs"])
        # Seeded messages share text "Hello reader" and role "user", but have
        # distinct anchors → alignment is sequential and content is equal.
        assert pairs[0]["diff_status"] == "equal"
        assert pairs[0]["role_match"] is True

    def test_compare_route_rejects_invalid_align(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, "/api/compare?left=c1&right=c2&align=sideways")

        assert status == 400
        assert payload["error"] == "invalid_request"

    def test_compare_route_rejects_unimplemented_align_modes(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _get_json_ex(base_url, "/api/compare?left=c1&right=c2&align=time")

        assert status == 400
        assert payload["error"] == "invalid_request"

    def test_workspace_shell_routes_are_unauthenticated(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, _, body = _get_text(base_url, "/w/stack?ids=c1,c2")
            compare_status, _, compare_body = _get_text(base_url, "/w/compare?left=c1&right=c2&align=prompt")

        assert status == 200
        assert "<title>Polylogue</title>" in body
        assert "getWorkspaceRouteFromURL" in body
        assert "workspace-mode-switcher" in body
        assert compare_status == 200
        assert "renderCompareWorkspace" in compare_body


# ---------------------------------------------------------------------------
# polylogue.local_reader.user_state — durable marks/views/recall contracts
# ---------------------------------------------------------------------------


class TestReaderUserState:
    """``polylogue.local_reader.user_state``: durable conversation user state."""

    def test_conversation_marks_are_idempotent_and_deletable(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, created = _request_json(
                base_url,
                "POST",
                "/api/user/marks",
                payload={"conversation_id": "c1", "mark_type": "star"},
            )
            status2, duplicate = _request_json(
                base_url,
                "POST",
                "/api/user/marks",
                payload={"conversation_id": "c1", "mark_type": "star"},
            )
            marks = _get_json(base_url, "/api/user/marks?conversation_id=c1")
            delete_status, deleted = _request_json(
                base_url,
                "DELETE",
                "/api/user/marks?conversation_id=c1&mark_type=star",
            )
            empty = _get_json(base_url, "/api/user/marks?conversation_id=c1")

        marks_payload = cast(dict[str, object], marks)
        assert status == 201
        assert created == {
            "target_type": "conversation",
            "target_id": "c1",
            "conversation_id": "c1",
            "message_id": None,
            "mark_type": "star",
            "created": True,
        }
        assert status2 == 200
        assert duplicate == {
            "target_type": "conversation",
            "target_id": "c1",
            "conversation_id": "c1",
            "message_id": None,
            "mark_type": "star",
            "created": False,
        }
        mark_items = cast(list[dict[str, object]], marks_payload["items"])
        assert mark_items[0]["mark_type"] == "star"
        assert mark_items[0]["target_type"] == "conversation"
        assert mark_items[0]["target_id"] == "c1"
        assert delete_status == 200
        assert deleted == {
            "target_type": "conversation",
            "target_id": "c1",
            "conversation_id": "c1",
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
                    "conversation_id": "c1",
                    "target_type": "message",
                    "message_id": "m-c1",
                    "mark_type": "pin",
                },
            )
            marks = _get_json(base_url, "/api/user/marks?target_type=message&message_id=m-c1")

        marks_payload = cast(dict[str, object], marks)
        created_payload = cast(dict[str, object], created)
        assert status == 201
        assert created_payload["target_type"] == "message"
        assert created_payload["target_id"] == "m-c1"
        assert created_payload["message_id"] == "m-c1"
        mark_items = cast(list[dict[str, object]], marks_payload["items"])
        assert mark_items == [
            {
                "target_type": "message",
                "target_id": "m-c1",
                "conversation_id": "c1",
                "message_id": "m-c1",
                "mark_type": "pin",
                "created_at": mark_items[0]["created_at"],
            }
        ]

    def test_annotations_roundtrip_conversation_and_message_targets(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            conv_status, conv_note = _request_json(
                base_url,
                "POST",
                "/api/user/annotations",
                payload={"annotation_id": "ann-c1", "conversation_id": "c1", "note_text": "Follow up"},
            )
            msg_status, msg_note = _request_json(
                base_url,
                "POST",
                "/api/user/annotations",
                payload={
                    "annotation_id": "ann-m1",
                    "conversation_id": "c1",
                    "target_type": "message",
                    "message_id": "m-c1",
                    "note_text": "Important request",
                },
            )
            listed = _get_json(base_url, "/api/user/annotations?conversation_id=c1")
            fetched = _get_json(base_url, "/api/user/annotations/ann-m1")
            delete_status, deleted = _request_json(base_url, "DELETE", "/api/user/annotations/ann-c1")

        listed_payload = cast(dict[str, object], listed)
        conv_note_payload = cast(dict[str, object], conv_note)
        msg_note_payload = cast(dict[str, object], msg_note)
        fetched_payload = cast(dict[str, object], fetched)
        items = cast(list[dict[str, object]], listed_payload["items"])
        assert conv_status == 201
        assert conv_note_payload["target_type"] == "conversation"
        assert conv_note_payload["target_id"] == "c1"
        assert msg_status == 201
        assert msg_note_payload["target_type"] == "message"
        assert msg_note_payload["target_id"] == "m-c1"
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
                    "query": {"query": "auth", "provider": "claude-code", "limit": 5},
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
        assert saved_payload["query"] == {"limit": 5, "provider": "claude-code", "query": "auth"}
        assert listed_payload["total"] == 1
        assert fetched_payload["view_id"] == "view-auth"
        assert fetched_payload["query_json"] == '{"limit":5,"provider":"claude-code","query":"auth"}'
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

    def test_recall_packs_roundtrip_cited_conversations(self, workspace_env: dict[str, Path]) -> None:
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
                            {"target_type": "conversation", "conversation_id": "c1"},
                            {"target_type": "conversation", "conversation_id": "missing-conv"},
                            {
                                "target_type": "message",
                                "conversation_id": "c1",
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
        assert fetched_payload["conversation_ids"] == ["c1"]
        payload = cast(dict[str, object], fetched_payload["payload"])
        assert payload["summary"] == "handoff"
        assert payload["resolved_count"] == 1
        assert payload["degraded_count"] == 2
        items = cast(list[dict[str, object]], payload["items"])
        assert [(item["target_type"], item["status"]) for item in items] == [
            ("conversation", "resolved"),
            ("conversation", "missing"),
            ("message", "missing"),
        ]
        assert delete_status == 200
        assert deleted == {"pack_id": "pack-auth", "deleted": True}

    def test_recall_pack_rejects_conversation_ids_compat_input(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, payload = _request_json(
                base_url,
                "POST",
                "/api/user/recall-packs",
                payload={
                    "pack_id": "pack-compat",
                    "label": "Compat pack",
                    "conversation_ids": ["c1"],
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
                        {"target_type": "conversation", "conversation_id": "c1"},
                        {"target_type": "message", "conversation_id": "c1", "message_id": "m-c1"},
                        {"target_type": "message", "conversation_id": "c1", "message_id": "missing-msg"},
                        {"target_type": "topology_edge", "target_id": "edge-1"},
                    ],
                    "layout": {"panes": [{"width": 0.5}, {"width": 0.5}]},
                    "active_target": {"target_type": "message", "conversation_id": "c1", "message_id": "m-c1"},
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
            ("conversation", "resolved"),
            ("message", "resolved"),
            ("message", "missing"),
            ("topology_edge", "unsupported"),
        ]
        active_target = cast(dict[str, object], fetched_payload["active_target"])
        assert active_target["identity_key"] == "message:c1:m-c1"
        assert delete_status == 200
        assert deleted == {"workspace_id": "workspace-auth", "deleted": True}


# ---------------------------------------------------------------------------
# Empty / degraded / privacy states
# ---------------------------------------------------------------------------


class TestReaderDegradedStates:
    def test_empty_archive_returns_zero_envelope(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = _get_json(base_url, "/api/conversations")
        assert isinstance(payload, dict)
        assert payload["total"] == 0
        assert payload["items"] == []

    def test_empty_archive_facets_returns_no_providers(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert payload["total_conversations"] == 0
        assert payload["providers"] == {}

    def test_no_results_query_is_distinguishable_from_empty_archive(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/conversations?query=nonexistent_term_xyz")
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
            "/api/conversations",
            "/api/conversations/c1",
            "/api/conversations/c1/messages",
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
            status, _, _ = _get_text(base_url, "/api/conversations")
        assert status == 401

    def test_authenticated_endpoint_accepts_correct_bearer(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env, auth_token="secret-token") as (_, base_url):
            payload = _get_json(base_url, "/api/conversations", headers={"Authorization": "Bearer secret-token"})
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
    assert "polylogue.local_reader.conversation" in body


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
            message="No conversations matched the query.",
            filters=("provider=gemini", "tag=api"),
            reasons=(QueryMissReasonPayload(code="no_results", severity="info", summary="empty archive"),),
            archive_conversation_count=42,
            raw_conversation_count=100,
        )
        d = p.model_dump(mode="json")
        assert d["message"] == "No conversations matched the query."
        assert d["filters"] == ["provider=gemini", "tag=api"]
        assert len(d["reasons"]) == 1
        assert d["archive_conversation_count"] == 42

    def test_conversation_list_response_shape(self) -> None:
        from polylogue.surfaces.payloads import ConversationListResponse

        r = ConversationListResponse(items=(), total=0, limit=50, offset=0)
        d = r.model_dump(mode="json")
        assert d["items"] == []
        assert d["total"] == 0
        assert d["limit"] == 50
        assert d["offset"] == 0

    def test_reader_target_ref_and_action_payloads_roundtrip(self) -> None:
        from polylogue.surfaces.payloads import (
            ConversationListRowPayload,
            ConversationMessagePayload,
            ReaderActionAvailabilityPayload,
            TargetRefPayload,
            reader_anchor,
        )

        message_ref = TargetRefPayload.message(conversation_id="c1", message_id="m-c1")
        assert message_ref.model_dump(mode="json", exclude_none=True) == {
            "target_type": "message",
            "target_id": "m-c1",
            "conversation_id": "c1",
            "message_id": "m-c1",
            "identity_key": "message:c1:m-c1",
        }
        assert reader_anchor("message", "m:c1/unsafe") == "message-m-c1-unsafe"

        row = ConversationListRowPayload(
            id="c1",
            provider="claude-code",
            title="Reader contract",
            target_ref=TargetRefPayload.conversation("c1"),
            anchor="conversation-c1",
            messages=1,
        )
        message = ConversationMessagePayload(
            id="m-c1",
            role="user",
            text="Hello reader",
            target_ref=message_ref,
            anchor="message-m-c1",
            actions={"annotate": ReaderActionAvailabilityPayload(enabled=True)},
        )

        row_dump = row.model_dump(mode="json", exclude_none=True)
        message_dump = message.model_dump(mode="json", exclude_none=True)
        assert row_dump["target_ref"]["identity_key"] == "conversation:c1"
        assert row_dump["actions"]["copy_link"]["enabled"] is True
        assert message_dump["target_ref"]["identity_key"] == "message:c1:m-c1"
        assert message_dump["actions"]["annotate"]["enabled"] is True

    def test_conversation_list_response_with_diagnostics(self) -> None:
        from polylogue.surfaces.payloads import (
            ConversationListResponse,
            QueryMissDiagnosticsPayload,
            QueryMissReasonPayload,
        )

        diag = QueryMissDiagnosticsPayload(
            message="No results.",
            filters=("tag=missing",),
            reasons=(QueryMissReasonPayload(code="no_results", severity="info", summary="no match"),),
        )
        r = ConversationListResponse(items=(), total=0, limit=10, offset=0, diagnostics=diag)
        d = r.model_dump(mode="json")
        assert d["diagnostics"] is not None
        assert d["diagnostics"]["message"] == "No results."

    def test_facets_response_shape(self) -> None:
        from polylogue.surfaces.payloads import FacetsResponse, FacetTimeRange

        r = FacetsResponse(
            scoped_to_query=False,
            providers={"claude-code": 10, "chatgpt": 5},
            total_conversations=15,
            total_messages=100,
            time_range=FacetTimeRange(min="2024-01-01", max="2024-12-31"),
        )
        d = r.model_dump(mode="json")
        assert d["scoped_to_query"] is False
        assert d["providers"] == {"claude-code": 10, "chatgpt": 5}
        assert d["total_conversations"] == 15
        assert d["time_range"]["min"] == "2024-01-01"


class TestQueryNoResultsDiagnosticPath:
    """Cross-surface contract: one filtered query → diagnostics path."""

    def test_search_with_bad_date_returns_query_error(self, workspace_env: dict[str, Path]) -> None:
        """A query with a malformed date produces a QueryErrorPayload-shaped error."""
        with _running_server(workspace_env) as (_, base_url):
            status, body = _get_json_ex(base_url, "/api/conversations?since=not-a-date")
        assert status == 400
        # The error shape is compatible with QueryErrorPayload
        assert "error" in body
        assert body.get("ok") is False

    def test_list_with_no_match_returns_diagnostics(self, workspace_env: dict[str, Path]) -> None:
        """A list with a tag that doesn't exist returns items=[] with total=0."""
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/conversations?tag=nonexistent")
        assert isinstance(payload, dict)
        assert "items" in payload
        assert "total" in payload
        assert payload["total"] == 0
        assert payload["items"] == []

    def test_facets_global_returns_providers(self, workspace_env: dict[str, Path]) -> None:
        """Unscoped /api/facets returns scoped_to_query=False with provider counts."""
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets")
        assert isinstance(payload, dict)
        assert payload["scoped_to_query"] is False
        assert isinstance(payload["providers"], dict)

    def test_facets_scoped_returns_subset(self, workspace_env: dict[str, Path]) -> None:
        """Scoped /api/facets?provider=... returns scoped_to_query=True."""
        with _running_server(workspace_env) as (_, base_url):
            payload = _get_json(base_url, "/api/facets?provider=chatgpt")
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
        action-event FTS drift entirely.
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
        assert "No conversations in archive. Run `polylogued run`" in body
        assert "No results for query=" in body
        assert "No conversations from provider=" in body
        assert "Press Esc to clear" in body

    def test_workspace_actions_have_disabled_state_with_tooltips(self, workspace_env: dict[str, Path]) -> None:
        """Stack/Compare/Save/Recall workspace buttons must expose a
        disabled state with explanatory tooltips when context is
        insufficient — disabled actions are part of the MK3 design."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "Stack needs a selected conversation" in body
        assert "Compare needs two conversations" in body
        assert "Save workspace needs at least one open conversation" in body
        assert "Recall pack needs at least one open conversation" in body

    def test_conversation_header_chip_order_follows_mk3_spec(self, workspace_env: dict[str, Path]) -> None:
        """MK3 specifies header chip order: provider, live/stale,
        repo/cwd/branch, counts, cost/tokens, derived/insight, marks.
        Pin the comment marker so a future reorder is a deliberate change
        rather than accidental drift."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "MK3 header chip order" in body
        # The provider chip should be tagged canonical (it identifies the
        # source-of-truth provider for the conversation), not just a neutral chip.
        assert "'<span class=\"chip q-canonical\">' + esc(c.provider)" in body


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
    """``polylogue.local_reader.bulk``: selection toolbar + per-conversation envelope.

    The bulk surface is composed client-side over existing daemon
    endpoints (``/api/user/marks`` for tag mutations, ``/api/conversations/{id}``
    for export). Delete and re-embed are exposed only through a preview
    overlay because the daemon has no corresponding mutation routes yet.

    These tests assert the shipped HTML carries every load-bearing region
    hook and that the underlying tag endpoint accepts the per-conversation
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
        """The shipped JS must reference the per-conversation status keys the
        bulk envelope contract uses (succeeded/failed/skipped + dryRun + action).
        These are the field names the UI surfaces to the operator after a bulk
        op; renaming any of them silently would break the rendered status."""
        with _running_server(workspace_env) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        for key in ("succeeded", "failed", "skipped", "dryRun", "action"):
            assert key in body, f"bulk envelope key {key!r} missing from shell"

    def test_bulk_tag_drives_existing_marks_endpoint(self, workspace_env: dict[str, Path]) -> None:
        """The bulk toolbar issues per-conversation POSTs against
        ``/api/user/marks``. This simulates that loop server-side to lock the
        contract: the daemon must accept the same payload shape the bulk JS
        emits for every selected conversation, and a follow-up GET must
        surface the resulting marks on every conversation."""
        with _running_server(workspace_env) as (_, base_url):
            statuses: list[int] = []
            for cid in ("c1", "c2", "c3"):
                status, _ = _request_json(
                    base_url,
                    "POST",
                    "/api/user/marks",
                    payload={"conversation_id": cid, "mark_type": "star"},
                )
                statuses.append(status)
            listing = _get_json(base_url, "/api/user/marks?mark_type=star")
        assert statuses == [201, 201, 201]
        payload = cast(dict[str, object], listing)
        items = cast(list[dict[str, object]], payload["items"])
        ids = sorted(str(item["conversation_id"]) for item in items)
        assert ids == ["c1", "c2", "c3"]

    def test_bulk_export_uses_conversation_detail_endpoint(self, workspace_env: dict[str, Path]) -> None:
        """Bulk export concatenates per-conversation GETs. This pins the
        contract that ``/api/conversations/{id}`` returns the detail payload
        the export bundle is composed from, for every selected id."""
        with _running_server(workspace_env) as (_, base_url):
            payloads = [_get_json(base_url, f"/api/conversations/{cid}") for cid in ("c1", "c2", "c3")]
        ids = sorted(str(cast(dict[str, object], p)["id"]) for p in payloads)
        assert ids == ["c1", "c2", "c3"]


@pytest.mark.parametrize("path", ["/", "/api/conversations", "/api/facets", "/api/status", "/api/health"])
def test_each_reader_route_responds_within_a_reasonable_budget(workspace_env: dict[str, Path], path: str) -> None:
    """Each reader-facing route returns within 10 s on a synthetic
    three-conversation archive. ``/api/health`` is included because the
    web shell pings it on every render cycle (see
    ``polylogue/daemon/web_shell.py::loadStatus``); a regression there
    would freeze the reader UI even if the data routes were healthy.
    The budget is loose by design — this is a smoke that catches
    "endpoint hangs forever" regressions, not a latency benchmark.
    """
    with _running_server(workspace_env) as (_, base_url):
        status, _, _ = _get_text(base_url, path)
    assert status == 200
