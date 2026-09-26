"""The user-overlay family is installed as executable daemon operations."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.daemon.route_contracts import DAEMON_ROUTE_DECLARATIONS
from polylogue.daemon.route_families import user_overlay

_EXPECTED = {
    ("GET", "/api/assertions"),
    *(
        ("GET", "/api/user/" + path)
        for path in (
            "marks",
            "annotations",
            "annotations/:id",
            "saved-views",
            "saved-views/:id",
            "recall-packs",
            "recall-packs/:id",
            "workspaces",
            "workspaces/:id",
        )
    ),
    *(
        ("POST", "/api/user/" + path)
        for path in (
            "marks",
            "annotations",
            "saved-views",
            "recall-packs",
            "workspaces",
        )
    ),
    *(
        ("DELETE", "/api/user/" + path)
        for path in (
            "marks",
            "annotations/:id",
            "saved-views/:id",
            "recall-packs/:id",
            "workspaces/:id",
        )
    ),
}


def test_all_twenty_user_overlay_routes_bind_canonical_operations() -> None:
    routes = user_overlay.ROUTES
    assert len(routes) == len(_EXPECTED) == 20
    assert {(route.method, route.path) for route in routes} == _EXPECTED
    assert {(route.method, route.path) for route in DAEMON_ROUTE_DECLARATIONS} >= _EXPECTED
    for route in routes:
        assert route.domain_operation is not None
        assert len(route.kernel.handlers) == 1
        binding = route.kernel.handlers[0]
        assert binding.surface == "daemon-http"
        assert binding.binding_key == f"{route.method} {route.path}"
        assert route.auth_policy == (
            "credential_if_configured" if route.method == "GET" else "credential_and_same_origin"
        )


class _Handler:
    def __init__(self, body: dict[str, object] | None = None) -> None:
        raw = json.dumps(body or {}).encode()
        self.headers = {"Content-Length": str(len(raw))}
        self.rfile = BytesIO(raw)
        self.server = type("Server", (), {"archive_root": "/archive"})()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.responses: list[tuple[int, object]] = []

    def _execute_daemon_operation(self, request: Any) -> dict[str, object]:
        self.calls.append((request.operation, request.payload))
        return {
            "outcome": "completed",
            "result": {
                "operation": request.operation,
                "outcome": "completed",
                "effect": "committed",
                "affected_count": 1,
                "result": {"status": "ok", "operation": request.operation, "affected_count": 1},
            },
        }

    def _send_json(self, status: int, payload: object) -> None:
        self.responses.append((status, payload))

    def _send_error(self, status: int, code: str) -> None:
        self.responses.append((status, code))


def test_post_annotation_keeps_deterministic_id_and_created_status() -> None:
    handler = _Handler({"session_id": "session:1", "note_text": "note"})
    user_overlay.handle_post(handler, ["api", "user", "annotations"], {})
    name, payload = handler.calls[0]
    assert name == "user.annotation.save"
    assert payload["annotation_id"] == user_overlay._default_id("annotation", "session", "session:1", "note")
    assert handler.responses[0][0] == 201
    assert handler.responses[0][1] == {"status": "ok", "operation": "user.annotation.save", "affected_count": 1}


def test_get_absent_overlay_maps_typed_absence_to_404(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _Handler()
    monkeypatch.setattr(
        handler,
        "_execute_daemon_operation",
        lambda request: {"outcome": "completed", "result": {"found": False, "item": None}},
    )
    user_overlay.handle_get(handler, ["api", "user", "workspaces", "missing"], {})
    assert handler.responses == [(404, "not_found")]


def test_delete_mark_retains_query_filters() -> None:
    handler = _Handler()
    user_overlay.handle_delete(handler, ["api", "user", "marks"], {"session_id": ["s"], "mark_type": ["star"]})
    assert handler.calls == [("user.mark.remove", {"session_id": "s", "mark_type": "star"})]


def test_canonical_overlay_reads_use_durable_rows(workspace_env: dict[str, Path]) -> None:
    from polylogue.operations.user_overlay_reads import execute_user_overlay_read
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.storage_records import SessionBuilder, db_setup

    builder = SessionBuilder(db_setup(workspace_env), "overlay-read")
    builder.provider("claude-code").title("Overlay").add_message(message_id="m1", role="user", text="hello")
    builder.save()
    session_id = builder.native_session_id()
    with ArchiveStore.open_existing(workspace_env["archive_root"], read_only=False) as archive:
        archive.add_mark("session", session_id, "star", owner_session_id=session_id)
        archive.save_annotation("a1", "session", session_id, "note", owner_session_id=session_id)
    with ArchiveStore.open_existing(workspace_env["archive_root"], read_only=True) as archive:
        marks = execute_user_overlay_read("user.marks.list", {"session_id": session_id}, archive=archive)
        annotation = execute_user_overlay_read("user.annotations.get", {"id": "a1"}, archive=archive)
        missing = execute_user_overlay_read("user.annotations.get", {"id": "missing"}, archive=archive)
    assert marks["total"] == 1
    assert cast(list[dict[str, object]], marks["items"])[0]["mark_type"] == "star"
    assert annotation["found"] is True
    assert cast(dict[str, object], annotation["item"])["note_text"] == "note"
    assert missing == {"found": False, "item": None}


@pytest.mark.uses_real_clock("runs the daemon operation listener and writer coordination")
def test_cookie_authorized_http_mark_round_trip_uses_daemon_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib
    import hashlib

    from polylogue.daemon.web_auth import WebCredentialRegistry
    from polylogue.storage.sqlite.connection import _clear_connection_cache
    from tests.infra.daemon_http_harness import MockDaemonServer, capture_responses, make_daemon_handler
    from tests.infra.daemon_operations import running_daemon_operations
    from tests.infra.storage_records import SessionBuilder

    session_ids: list[str] = []

    def seed(root: Path) -> None:
        builder = (
            SessionBuilder(root / "index.db", "overlay-roundtrip")
            .provider("codex")
            .title("Overlay round trip")
            .add_message(text="synthetic overlay session")
        )
        builder.save()
        session_ids.append(builder.native_session_id())
        _clear_connection_cache()

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        registry = WebCredentialRegistry()
        origin = "http://127.0.0.1:8766"
        issued = registry.issue(origin)
        server = MockDaemonServer(auth_token="secret", web_credentials=registry)
        monkeypatch.setattr(server, "archive_root", stack.archive_root, raising=False)
        principals: list[Any] = []

        class _RuntimeProxy:
            def call(self, request: Any, principal: Any, *, client_disconnect: Any) -> dict[str, object]:
                principals.append(principal)
                return stack.runtime.call(request, principal, client_disconnect=client_disconnect)

        monkeypatch.setattr(server, "operation_runtime", _RuntimeProxy(), raising=False)
        monkeypatch.setattr(
            "polylogue.daemon.operation_disconnect.observe_peer_disconnect",
            lambda _connection: contextlib.nullcontext(None),
        )

        def http_handler(method: str, path: str, *, body: bytes = b"") -> Any:
            handler = make_daemon_handler(
                method,
                path,
                body=body,
                server=server,
                cookie=f"polylogue_web_credential={issued.token}",
                origin=origin,
                host="127.0.0.1:8766",
                web_client=True,
            )
            handler.connection = object()
            return handler

        post = http_handler(
            "POST",
            "/api/user/marks",
            body=json.dumps({"session_id": session_ids[0], "mark_type": "star"}).encode(),
        )
        post_error, post_json = capture_responses(post)
        post.do_POST()
        post_error.assert_not_called()
        assert post_json.call_args.args[0] == 201
        assert post_json.call_args.args[1]["status"] == "ok"
        listed = stack.client.operation(
            "user.marks.list",
            {"session_id": session_ids[0]},
            archive_root=str(stack.archive_root),
        )
        delete = http_handler("DELETE", f"/api/user/marks?session_id={session_ids[0]}&mark_type=star")
        delete_error, delete_json = capture_responses(delete)
        delete.do_DELETE()
        delete_error.assert_not_called()
        assert delete_json.call_args.args[0] == 200
        assert delete_json.call_args.args[1]["status"] == "deleted"
        after = stack.client.operation(
            "user.marks.list",
            {"session_id": session_ids[0]},
            archive_root=str(stack.archive_root),
        )
    assert len(principals) == 2
    assert {principal.actor_ref for principal in principals} == {
        "daemon:web:" + hashlib.sha256(issued.token.encode()).hexdigest()
    }
    assert all(
        principal.surface == "api" and principal.role_label == "daemon-web-credential" for principal in principals
    )
    assert listed is not None and listed["result"]["total"] == 1
    assert listed["result"]["items"][0]["mark_type"] == "star"
    assert after is not None and after["result"]["total"] == 0
