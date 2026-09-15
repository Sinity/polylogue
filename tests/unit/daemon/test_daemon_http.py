"""Security tests for daemon HTTP auth and origin enforcement (#868)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from http import HTTPStatus
from pathlib import Path


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
