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
            "INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id, title, content_hash, version) VALUES(?,?,?,?,?,?)",
            (cid, prov, f"p-{cid}", title, f"hash-{cid}", 1),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, provider_name, content_hash, version) VALUES(?,?,?,?,?,?,?)",
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
        body = exc.read().decode() if exc.fp else ""
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
        raw = exc.read().decode() if exc.fp else "{}"
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
        for region in ("renderSidebarState", "renderConversations", "renderFacets", "renderMain", "renderInspector"):
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
        assert row["actions"]["annotate"] == {"enabled": True}

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
        hit = next(item for item in payload["hits"] if item["id"] == "c1")
        assert hit["target_ref"]["identity_key"] == "conversation:c1"
        assert hit["anchor"] == "conversation-c1"
        assert hit["match"]["target_ref"] == {
            "target_type": "message",
            "target_id": "m-c1",
            "conversation_id": "c1",
            "message_id": "m-c1",
            "identity_key": "message:c1:m-c1",
        }
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
        assert message["actions"]["annotate"] == {"enabled": True}

    def test_unknown_conversation_yields_404(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as (_, base_url):
            status, _, _ = _get_text(base_url, "/api/conversations/does-not-exist")
        assert status == 404


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
                    "conversation_ids": ["c1", "c2"],
                    "payload": {"reason": "handoff"},
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
        assert fetched_payload["conversation_ids"] == ["c1", "c2"]
        assert fetched_payload["payload"] == {"reason": "handoff"}
        assert delete_status == 200
        assert deleted == {"pack_id": "pack-auth", "deleted": True}


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
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return e.code, {}


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
