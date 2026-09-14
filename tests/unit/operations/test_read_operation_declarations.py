"""Behaviour of the read operations declared for the CLI's remaining reads.

Every test here runs the real handler over a pinned reader on a seeded
synthetic archive, so a handler that answered from its own query logic instead
of the shared executors would have to reproduce these numbers by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.operations.daemon_protocol import (
    OperationResultContractError,
    daemon_operation_spec,
    validate_operation_result,
)
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path, *, count: int = 3, messages: int = 4) -> tuple[str, ...]:
    bootstrap_archive_root(root)
    session_ids: list[str] = []
    for number in range(count):
        builder = SessionBuilder(root / "index.db", f"declared-read-{number}").provider("codex").title(f"Read {number}")
        for position in range(messages):
            builder.add_message(text=f"session {number} message {position} needle")
        builder.save()
        session_ids.append(builder.native_session_id())
    return tuple(session_ids)


def _run(root: Path, name: str, payload: dict[str, object]) -> dict[str, object]:
    with open_operation_read(root) as pinned:
        result = execute_read_operation(name, payload, archive=pinned.archive, serving_identity="daemon")
    validate_operation_result(name, result)
    return result


class TestDeclaration:
    def test_every_new_read_permits_direct_execution(self) -> None:
        """Mutation: declare one of these with DaemonFallback.NEVER and a CLI read
        stops working whenever no daemon is resident."""

        for name in ("query.aggregate", "session.read", "session.reference"):
            spec = daemon_operation_spec(name)
            assert spec is not None, name
            assert spec.direct_allowed, name
            assert spec.capability == "read", name


class TestQueryAggregate:
    def test_count_matches_the_archive_count_executor(self, tmp_path: Path) -> None:
        """Mutation: count from the returned page length instead of the count
        executor and a count larger than one page reads as the page size."""

        _seed(tmp_path, count=3)
        result = _run(tmp_path, "query.aggregate", {"mode": "count", "params": {}})
        assert result["count"] == 3
        assert result["mode"] == "count"

    def test_stats_by_groups_through_the_shared_aggregate(self, tmp_path: Path) -> None:
        """Mutation: drop the stats_filter_kwargs adapter and the aggregate raises
        on a list-only filter keyword instead of grouping."""

        _seed(tmp_path, count=2)
        result = _run(tmp_path, "query.aggregate", {"mode": "stats_by", "group_by": "origin", "params": {}})
        assert result["groups"] == {"codex-session": 2}

    def test_text_selection_scopes_the_aggregate_to_matched_sessions(self, tmp_path: Path) -> None:
        """Mutation: ignore the matched-session scope and a text query reports the
        whole archive's totals as if the query had selected nothing."""

        _seed(tmp_path, count=2)
        result = _run(tmp_path, "query.aggregate", {"mode": "stats", "params": {"query": ("absent-term",)}})
        stats = cast("dict[str, Any]", result["stats"])
        assert stats["total_sessions"] == 0

    def test_unknown_group_by_is_refused(self, tmp_path: Path) -> None:
        """Mutation: swallow the executor's ValueError and an unknown grouping
        silently returns an empty result instead of a typed refusal."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError):
            _run(tmp_path, "query.aggregate", {"mode": "stats_by", "group_by": "not-a-field", "params": {}})

    def test_semantic_selection_is_refused_rather_than_silently_lexical(self, tmp_path: Path) -> None:
        """Mutation: accept similar_text here and an aggregate over a semantic
        selection is computed from the lexical filters alone."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError):
            _run(tmp_path, "query.aggregate", {"mode": "count", "params": {"similar_text": "needle"}})


class TestSessionRead:
    def test_window_is_bounded_and_continues(self, tmp_path: Path) -> None:
        """Mutation: return the whole transcript and ``complete`` is true on the
        first window, so a reader never asks for the rest."""

        sessions = _seed(tmp_path, count=1, messages=6)
        first = _run(tmp_path, "session.read", {"ref": sessions[0], "limit": 2})
        assert first["total"] == 6
        assert first["next_offset"] == 2
        assert first["complete"] is False
        assert len(cast("dict[str, Any]", first["session"])["messages"]) == 2

        second = _run(tmp_path, "session.read", {"ref": sessions[0], "continuation": first["continuation"]})
        assert second["offset"] == 2
        assert second["limit"] == 2
        assert len(cast("dict[str, Any]", second["session"])["messages"]) == 2

    def test_final_window_carries_no_continuation(self, tmp_path: Path) -> None:
        """Mutation: always mint a continuation and a reader loops forever on an
        exhausted transcript."""

        sessions = _seed(tmp_path, count=1, messages=3)
        result = _run(tmp_path, "session.read", {"ref": sessions[0], "limit": 50})
        assert result["complete"] is True
        assert result["continuation"] is None
        assert result["next_offset"] is None

    def test_continuation_from_another_session_is_refused(self, tmp_path: Path) -> None:
        """Mutation: skip the continuation identity check and a token minted for
        one session pages through a different one."""

        sessions = _seed(tmp_path, count=2, messages=4)
        first = _run(tmp_path, "session.read", {"ref": sessions[0], "limit": 2})
        with pytest.raises(Exception, match="continuation"):
            _run(tmp_path, "session.read", {"ref": sessions[1], "continuation": first["continuation"]})

    def test_missing_session_is_a_typed_refusal(self, tmp_path: Path) -> None:
        """Mutation: let the KeyError escape and the operation reports an internal
        failure rather than a missing reference."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError, match="session not found"):
            _run(tmp_path, "session.read", {"ref": "codex-session:absent"})

    def test_a_window_above_the_result_bound_is_refused_not_truncated(self) -> None:
        """Mutation: drop the bound check and an oversized window is either
        rejected by the transport with no guidance or silently truncated."""

        from polylogue.operations.daemon_reads import _require_deliverable_window

        with pytest.raises(ValueError, match="operation result bound"):
            _require_deliverable_window({"filler": "x" * (9 * 1024 * 1024)}, limit=2000)


class TestSessionReference:
    def test_a_non_reference_expression_is_refused(self, tmp_path: Path) -> None:
        """Mutation: fall through to a normal query and a mistyped reference root
        quietly returns session rows instead of naming the error."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError, match="not a reference operand"):
            _run(tmp_path, "session.reference", {"expression": "origin:codex-session"})

    def test_an_unknown_reference_is_named(self, tmp_path: Path) -> None:
        """Mutation: return an empty member list and an unknown reference is
        indistinguishable from an empty one."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError, match="reference not found"):
            _run(tmp_path, "session.reference", {"expression": f"from query:{'a' * 64}"})


class TestResultContracts:
    def test_an_aggregate_body_from_another_mode_is_rejected(self) -> None:
        """Mutation: relax the result model and a count result can carry a stats
        body that no renderer would ever show."""

        with pytest.raises(OperationResultContractError):
            validate_operation_result(
                "query.aggregate",
                {"outcome": {"state": "ok"}, "mode": "count", "count": 1, "groups": {"a": 1}},
            )

    def test_a_window_cannot_claim_completeness_and_a_next_offset(self) -> None:
        """Mutation: relax the result model and a truncated read can report
        itself complete."""

        with pytest.raises(OperationResultContractError):
            validate_operation_result(
                "session.read",
                {
                    "outcome": {"state": "ok"},
                    "session": {},
                    "session_id": "codex-session:x",
                    "total": 10,
                    "limit": 2,
                    "offset": 0,
                    "next_offset": 2,
                    "continuation": "q2.token",
                    "complete": True,
                },
            )
