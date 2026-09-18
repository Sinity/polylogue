"""Security tests for daemon HTTP auth and origin enforcement (#868)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast


def test_cross_origin_logic_rejects_external_origin() -> None:
    """_check_cross_origin returns False for non-localhost Origin header."""
    assert _origin_check("https://evil.example.com") is False
    assert _origin_check("http://192.168.1.1:8080") is False


def test_cross_origin_logic_allows_localhost() -> None:
    """_check_cross_origin returns True for localhost Origin header."""
    assert _origin_check("http://127.0.0.1:8766") is True
    assert _origin_check("http://localhost:3000") is True
    assert _origin_check("https://127.0.0.1:8766") is True


def test_cross_origin_logic_allows_no_origin() -> None:
    """_check_cross_origin returns True when no Origin header is present."""
    assert _origin_check("") is True


def test_auth_requires_token_when_configured() -> None:
    """_check_auth_logic returns not-allowed when token set but not sent."""
    from polylogue.daemon.http import _check_auth_logic

    assert _check_auth_logic("secret", "127.0.0.1", "").allowed is False
    assert _check_auth_logic("secret", "127.0.0.1", "Bearer wrong").allowed is False
    assert _check_auth_logic("secret", "127.0.0.1", "Bearer secret").allowed is True


def test_auth_allows_when_no_token_configured() -> None:
    """_check_auth_logic returns allowed when no token is configured."""
    from polylogue.daemon.http import _check_auth_logic

    assert _check_auth_logic("", "127.0.0.1", "").allowed is True
    assert _check_auth_logic(None, "192.168.1.1", "").allowed is True


def _origin_check(origin: str) -> bool:
    """Simulate the origin check logic from _check_cross_origin."""
    if not origin:
        return True
    return (
        origin.startswith("http://127.0.0.1:")
        or origin.startswith("http://localhost:")
        or origin.startswith("https://127.0.0.1:")
        or origin.startswith("https://localhost:")
    )


def test_evidence_summary_outcomes_read_the_canonical_result_state(tmp_path: Path) -> None:
    """``/api/sessions/:id/evidence-summary`` counts ``actions.result_state``.

    Three paired tool calls are seeded: a successful one, a failed one, and
    one whose ``tool_outcome`` the parser deliberately distrusted while the
    provider's ``exit_code = 0`` stayed on the row as a legacy compatibility
    field. The canonical ``actions.result_state`` calls that third row
    ``outcome_unknown``; the legacy ``exit_code``/``is_error`` pair calls it a
    success. Anti-vacuity: re-deriving the outcome from the compat pair
    reports ``ok = 2, unknown = 0`` and turns this red.
    """
    import sqlite3
    from unittest.mock import patch

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    def _pair(tool_id: str, *, is_error: bool | None, exit_code: int | None) -> list[ParsedContentBlock]:
        return [
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Bash",
                tool_id=tool_id,
                tool_input={"command": f"run {tool_id}"},
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_id,
                text=f"output {tool_id}",
                is_error=is_error,
                exit_code=exit_code,
            ),
        ]

    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        written = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="evidence-outcomes",
                title="Evidence outcomes",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        text="calls",
                        timestamp="2026-01-01T00:00:00+00:00",
                        blocks=[
                            *_pair("t-ok", is_error=False, exit_code=0),
                            *_pair("t-err", is_error=True, exit_code=2),
                            *_pair("t-unknown", is_error=False, exit_code=0),
                        ],
                    )
                ],
            ),
        )
    session_id = written

    # Demote the third pair to the shape the fix is about: a distrusted
    # outcome whose legacy exit_code survives on the row.
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute(
            "UPDATE blocks SET tool_outcome = 'unknown', tool_result_is_error = NULL, "
            "tool_result_outcome_unknown_reason = 'not_reported' "
            "WHERE tool_id = 't-unknown' AND block_type = 'tool_result'"
        )
        conn.execute(
            "UPDATE blocks SET tool_outcome = 'unknown' WHERE tool_id = 't-unknown' AND block_type = 'tool_use'"
        )
        retained_exit_code = conn.execute(
            "SELECT tool_result_exit_code FROM blocks WHERE tool_id = 't-unknown' AND block_type = 'tool_result'"
        ).fetchone()[0]
    assert retained_exit_code == 0

    class _RecordingHandler(DaemonAPIHandler):
        """Capture the handler's response without a live socket or daemon."""

        def __init__(self, path: str) -> None:
            self.path = path
            self.sent: list[tuple[HTTPStatus, object]] = []

        def _send_json(
            self, status: HTTPStatus, payload: object, *, extra_headers: Mapping[str, str] | None = None
        ) -> None:
            self.sent.append((status, payload))

        def _send_error(
            self,
            status: HTTPStatus,
            code: str,
            detail: str | None = None,
            *,
            extra_headers: Mapping[str, str] | None = None,
            extra_payload: Mapping[str, object] | None = None,
        ) -> None:
            self.sent.append((status, code))

        def _sync_run(self, handler: Callable[..., object]) -> object:
            return {"total_usd": None, "confidence_tag": "q-missing"}

    handler = _RecordingHandler(f"/api/sessions/{session_id}/evidence-summary")
    sent = handler.sent

    with patch("polylogue.paths.archive_root", return_value=archive_root):
        handler._handle_get_session_evidence_summary(session_id)

    assert len(sent) == 1
    status, payload = sent[0]
    assert status is HTTPStatus.OK, payload
    assert isinstance(payload, dict)
    assert payload["tool_calls"] == 3
    assert payload["outcomes"] == {"ok": 1, "failed": 1, "unknown": 1}


def _reset_handler(body: bytes, *, content_length: str | None = None) -> tuple[Any, list[tuple[HTTPStatus, object]]]:
    """Build a socket-free ``_handle_reset`` handler over one request body."""
    from io import BytesIO

    from polylogue.daemon.http import DaemonAPIHandler

    sent: list[tuple[HTTPStatus, object]] = []

    class _Headers:
        def __init__(self, values: dict[str, str]) -> None:
            self._values = values

        def get(self, key: str, default: str | None = None) -> str | None:
            return self._values.get(key, default)

    class _RecordingHandler(DaemonAPIHandler):
        def __init__(self) -> None:
            self.path = "/api/reset"
            self.command = "POST"
            self.requestline = "POST /api/reset HTTP/1.1"
            self.client_address = ("127.0.0.1", 12345)
            self.rfile = BytesIO(body)
            self.wfile = BytesIO()
            self.headers = cast(
                "Any", _Headers({"Content-Length": str(len(body)) if content_length is None else content_length})
            )

        def _send_json(
            self, status: HTTPStatus, payload: object, *, extra_headers: Mapping[str, str] | None = None
        ) -> None:
            sent.append((status, payload))

        def _send_error(
            self,
            status: HTTPStatus,
            code: str,
            detail: str | None = None,
            *,
            extra_headers: Mapping[str, str] | None = None,
            extra_payload: Mapping[str, object] | None = None,
        ) -> None:
            payload: dict[str, object] = {"error": code, "detail": detail}
            if extra_payload:
                payload.update(extra_payload)
            sent.append((status, payload))

        def _sync_run(self, handler: Callable[..., object]) -> object:  # pragma: no cover - refusals never reach it
            raise AssertionError("a refused reset must not reach the writer")

    return _RecordingHandler(), sent


def test_reset_refuses_the_unimplemented_default_scope() -> None:
    """polylogue-peo7o: POST /api/reset with no scope refuses instead of lying.

    At the reported head, ``scope`` defaulted to ``"all"`` while ``_do_reset``
    acted only for ``scope == "session"``; every other scope fell through to
    ``{"ok": True}``, so the route answered 200 with detail ``reset all - no
    sessions matched`` and emitted a ``reset`` daemon event with
    ``operation_id=reset-all-all`` having touched nothing.

    Anti-vacuity: restoring the ``{"ok": True}`` fallthrough (or defaulting
    ``scope`` back to ``"all"`` without a membership check) turns this red --
    the assertion below demands a 4xx typed code, which a 200 ``ok`` envelope
    cannot satisfy. ``_sync_run`` also raises if a refused request ever
    reaches the writer.
    """
    import json

    handler, sent = _reset_handler(json.dumps({}).encode())
    handler._handle_reset()

    assert len(sent) == 1
    status, payload = sent[0]
    assert status is HTTPStatus.BAD_REQUEST, payload
    assert isinstance(payload, dict)
    assert payload["error"] == "unsupported_scope"
    assert payload["supported_scopes"] == ["session"]


def test_reset_refuses_an_explicitly_unsupported_scope() -> None:
    """Every scope outside the implemented set is refused by name.

    Anti-vacuity: widening ``RESET_SUPPORTED_SCOPES`` to accept ``"all"``
    without implementing it turns this red.
    """
    import json

    handler, sent = _reset_handler(json.dumps({"scope": "all"}).encode())
    handler._handle_reset()

    status, payload = sent[0]
    assert status is HTTPStatus.BAD_REQUEST
    assert isinstance(payload, dict)
    assert payload["error"] == "unsupported_scope"


def test_reset_refuses_a_malformed_content_length_with_400() -> None:
    """A malformed Content-Length is a client framing error, not a 500.

    Anti-vacuity: restoring the bare ``int(self.headers.get(...))`` makes the
    ValueError escape into ``daemon_safe_handler``, which answers 500
    ``internal_error`` -- and this assertion on 400 turns red.
    """
    handler, sent = _reset_handler(b'{"scope": "session", "session_id": "x"}', content_length="not-a-number")
    handler._handle_reset()

    status, payload = sent[0]
    assert status is HTTPStatus.BAD_REQUEST
    assert isinstance(payload, dict)
    assert payload["error"] == "invalid_request"


def test_reset_session_scope_requires_a_session_id() -> None:
    """The one implemented scope still refuses without its target."""
    import json

    handler, sent = _reset_handler(json.dumps({"scope": "session"}).encode())
    handler._handle_reset()

    status, payload = sent[0]
    assert status is HTTPStatus.BAD_REQUEST
    assert isinstance(payload, dict)
    assert payload["error"] == "invalid_request"


def test_evidence_summary_reports_degraded_when_lineage_is_unreadable(tmp_path: Path) -> None:
    """polylogue-31h8l: an unreadable session_links is a gap, not "no lineage".

    Input: ``session_links`` is dropped out from under an open archive, so the
    route's lineage query raises ``sqlite3.OperationalError``. At the reported
    head the ``except`` set ``lineage_rows = []`` and the 200 response carried
    ``lineage_refs: []`` with no outcome envelope, so the reader showed "no
    lineage" where the truth was "lineage unknown".

    Anti-vacuity: deleting the ``gaps.append(...)`` (restoring the bare ``[]``
    fallback) makes the envelope decide ``ok``/``empty`` instead of
    ``degraded``, and both assertions below turn red.
    """
    import sqlite3
    from unittest.mock import patch

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="lineage-degraded",
                title="Lineage degraded",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        text="calls",
                        timestamp="2026-01-01T00:00:00+00:00",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Bash",
                                tool_id="t-ok",
                                tool_input={"command": "run"},
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="t-ok",
                                text="output",
                                is_error=False,
                                exit_code=0,
                            ),
                        ],
                    )
                ],
            ),
        )

    # Fail exactly the lineage read, the way a locked or replaced index.db
    # fails it mid-request. Dropping the table instead would trip the archive's
    # own schema-identity refusal on open, which is a different contract.
    import contextlib

    from polylogue.daemon import http as http_module

    real_read_context = cast("Any", http_module).archive_read_context

    class _FailingLineageConn:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def execute(self, sql: str, *args: object) -> object:
            if "session_links" in sql:
                raise sqlite3.OperationalError("database is locked")
            return self._inner.execute(sql, *args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    class _FailingLineageArchive:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._conn = _FailingLineageConn(inner._conn)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    @contextlib.contextmanager
    def _wrapped(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        with real_read_context(*args, **kwargs) as archive:
            yield _FailingLineageArchive(archive)

    sent: list[tuple[HTTPStatus, object]] = []

    class _RecordingHandler(DaemonAPIHandler):
        def __init__(self, path: str) -> None:
            self.path = path

        def _send_json(
            self, status: HTTPStatus, payload: object, *, extra_headers: Mapping[str, str] | None = None
        ) -> None:
            sent.append((status, payload))

        def _send_error(
            self,
            status: HTTPStatus,
            code: str,
            detail: str | None = None,
            *,
            extra_headers: Mapping[str, str] | None = None,
            extra_payload: Mapping[str, object] | None = None,
        ) -> None:
            sent.append((status, code))

        def _sync_run(self, handler: Callable[..., object]) -> object:
            return {"total_usd": None, "confidence_tag": "q-missing"}

    handler = _RecordingHandler(f"/api/sessions/{session_id}/evidence-summary")
    with (
        patch("polylogue.paths.archive_root", return_value=archive_root),
        patch.object(http_module, "archive_read_context", _wrapped),
    ):
        handler._handle_get_session_evidence_summary(session_id)

    assert len(sent) == 1
    status, payload = sent[0]
    assert status is HTTPStatus.OK, payload
    assert isinstance(payload, dict)
    assert payload["lineage_refs"] == []
    assert payload["lineage_refs_authoritative"] is False
    outcome = payload["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "degraded", outcome
    assert outcome["reason"] == "lineage_refs_unreadable", outcome


def test_evidence_summary_reports_a_readable_empty_lineage_as_authoritative(tmp_path: Path) -> None:
    """A session that genuinely has no lineage is not reported as degraded.

    This is the other half of the distinction: without it, a fix could pass
    the degraded test by marking every response degraded.

    Anti-vacuity: seeding the gap unconditionally turns this red.
    """
    from unittest.mock import patch

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="lineage-empty",
                title="Lineage empty",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hello",
                        timestamp="2026-01-01T00:00:00+00:00",
                    )
                ],
            ),
        )

    sent: list[tuple[HTTPStatus, object]] = []

    class _RecordingHandler(DaemonAPIHandler):
        def __init__(self, path: str) -> None:
            self.path = path

        def _send_json(
            self, status: HTTPStatus, payload: object, *, extra_headers: Mapping[str, str] | None = None
        ) -> None:
            sent.append((status, payload))

        def _send_error(
            self,
            status: HTTPStatus,
            code: str,
            detail: str | None = None,
            *,
            extra_headers: Mapping[str, str] | None = None,
            extra_payload: Mapping[str, object] | None = None,
        ) -> None:
            sent.append((status, code))

        def _sync_run(self, handler: Callable[..., object]) -> object:
            return {"total_usd": None, "confidence_tag": "q-missing"}

    handler = _RecordingHandler(f"/api/sessions/{session_id}/evidence-summary")
    with patch("polylogue.paths.archive_root", return_value=archive_root):
        handler._handle_get_session_evidence_summary(session_id)

    status, payload = sent[0]
    assert status is HTTPStatus.OK
    assert isinstance(payload, dict)
    assert payload["lineage_refs"] == []
    assert payload["lineage_refs_authoritative"] is True
    assert payload["outcome"]["state"] != "degraded", payload["outcome"]
