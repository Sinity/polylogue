"""Real saved-query and MCP readers avoid unused session metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.query.evaluator import QueryEvaluationRequest
from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.enums import Provider
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.query_objects import put_query
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session


def _seed(root: Path) -> str:
    bootstrap_archive_root(root)
    selected = ""
    with ArchiveStore(root) as writer:
        for name in ("selected", "other"):
            session_id = write_index_session(
                writer,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=name,
                    title=f"neutral {name}",
                    working_directories=[f"/synthetic/{name}/dir-{index}" for index in range(101)],
                    messages=[ParsedMessage(provider_message_id="one", role=Role.USER, text="needle prose")],
                ),
            )
            writer.add_user_tags((session_id,), tuple(f"tag-{index}" for index in range(101)))
            if name == "selected":
                selected = session_id
                writer.add_user_tags((session_id,), ("selected",))
    return selected


def test_saved_query_selects_exact_identities_without_summary_metadata(tmp_path: Path) -> None:
    selected = _seed(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as writer:
        query = put_query(
            writer,
            {"kind": "field", "field": "tag", "op": "=", "values": ["selected"]},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
    with open_operation_read(tmp_path) as pinned:
        connection = pinned.archive.index_connection
        assert connection is not None
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        result = ArchiveCanonicalPlanEvaluator(pinned.archive).evaluate(
            QueryEvaluationRequest(query=query, purpose="reference")
        )
        assert result.member_refs == (f"session:{selected}",)
        assert result.exactness == "exact"
        assert not any("json_group_array" in sql.lower() for sql in statements)


@pytest.mark.parametrize("lexical", [False, True])
def test_mcp_prompt_keeps_selected_header_and_page_without_metadata(tmp_path: Path, lexical: bool) -> None:
    from polylogue.mcp.server_prompts import _archive_prompt_session_by_id, _archive_prompt_sessions

    selected = _seed(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        connection = pinned.archive.index_connection
        assert connection is not None
        expected_title = pinned.archive.read_summary(selected).display_label
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        result = _archive_prompt_sessions(
            pinned.archive, SessionQuerySpec(tags=("selected",), query_terms=("needle",) if lexical else (), limit=1)
        )
        direct = _archive_prompt_session_by_id(pinned.archive, selected)
        assert len(result) == 1 and direct == result[0]
        assert result[0].id == selected
        assert result[0].display_title == expected_title
        assert result[0].message_count == 1
        assert not result[0].truncated
        assert result[0].messages[0].text == "needle prose"
        assert not any("json_group_array" in sql.lower() for sql in statements)


@pytest.mark.uses_real_clock("exercises the actual query cancellation checkpoint")
def test_scalar_evaluation_cancellation_closes_selected_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Iterator
    from typing import Any

    from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionIdentity

    _seed(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as writer:
        query = put_query(
            writer,
            {"kind": "field", "field": "tag", "op": "=", "values": ["selected"]},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
    context = QueryExecutionContext.create(query_text="synthetic scalar cancellation", timeout_s=None)
    original = ArchiveStore.iter_session_identities
    closed: list[bool] = []

    def selected(archive: ArchiveStore, **kwargs: Any) -> Iterator[ArchiveSessionIdentity]:
        from contextlib import closing

        try:
            with closing(original(archive, **kwargs)) as rows:
                for row in rows:
                    context.cancel()
                    yield row
        finally:
            closed.append(True)

    monkeypatch.setattr(ArchiveStore, "iter_session_identities", selected)
    with pytest.raises(QueryCancelledError):
        with open_operation_read(tmp_path, execution_context=context) as pinned:
            ArchiveCanonicalPlanEvaluator(pinned.archive).evaluate(
                QueryEvaluationRequest(query=query, purpose="reference")
            )
    assert closed == [True]
    assert context.receipt.cleanup_complete
