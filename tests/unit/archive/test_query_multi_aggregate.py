from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
from tests.infra.storage_records import SessionBuilder


def test_multi_field_group_count_reports_proportions_and_denominator(workspace_env: dict[str, Path]) -> None:
    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "claude")
        .provider("claude-code")
        .add_message("user", role="user", text="aggregate")
        .add_message("assistant", role="assistant", text="aggregate")
        .save()
    )
    (
        SessionBuilder(index_db, "codex")
        .provider("codex")
        .add_message("assistant", role="assistant", text="aggregate")
        .save()
    )
    source = parse_unit_source_expression(
        "messages where text:aggregate | group by role, session.origin | count | limit 1"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="multi-aggregate", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert envelope.pipeline_stages == (
        {"kind": "group", "fields": ["role", "session.origin"]},
        {"kind": "count", "metric": "count"},
        {"kind": "limit", "value": 1},
        {"kind": "terminal", "action": "count"},
    )
    assert envelope.pipeline is not None
    result = cast(dict[str, object], envelope.pipeline["result"])
    assert {key: result[key] for key in result if key != "groups"} == {
        "group_by": ["role", "session.origin"],
        "aggregate": ["count", "proportion"],
        "denominator": {"kind": "all_matching_rows", "n": 3},
        "n": 3,
        "missing_counts": {"role": 0, "session.origin": 0},
        "unknown_counts": {"role": 0, "session.origin": 0},
        "limit": 1,
    }
    groups = cast(list[dict[str, object]], result["groups"])
    assert len(groups) == 1
    assert {
        (
            cast(dict[str, str], item["group"])["role"],
            cast(dict[str, str], item["group"])["session.origin"],
            item["count"],
            item["proportion"],
        )
        for item in groups
    } <= {
        ("assistant", "claude-code-session", 1, 1 / 3),
        ("assistant", "codex-session", 1, 1 / 3),
        ("user", "claude-code-session", 1, 1 / 3),
    }
    assert {
        (json.loads(row.group_key)["role"], json.loads(row.group_key)["session.origin"], row.count)
        for row in envelope.items
        if row.group_key is not None
    } <= {
        ("assistant", "claude-code-session", 1),
        ("assistant", "codex-session", 1),
        ("user", "claude-code-session", 1),
    }


def test_multi_field_group_rejects_each_unsupported_field() -> None:
    with pytest.raises(ExpressionCompileError, match=r"group by nope.*action rows"):
        parse_unit_source_expression("actions where tool:bash | group by tool, nope | count")


def _aggregate_page_rows(envelope: QueryUnitAggregateEnvelope) -> list[tuple[tuple[str, str], int]]:
    return [
        (
            (
                cast(dict[str, str], json.loads(row.group_key))["role"],
                cast(dict[str, str], json.loads(row.group_key))["session.repo"],
            ),
            row.count,
        )
        for row in envelope.items
        if row.group_key is not None
    ]


def test_multi_field_group_pages_concatenate_to_exact_stable_result(
    workspace_env: dict[str, Path],
) -> None:
    """Production dependency: ArchiveStore.query_unit_multi_counts.

    Replacing the SQL page with the former all-row Python materializer loses the
    out-of-range page's exact denominator/quality facts and restores unbounded
    selected-row retention.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "repo-a")
        .provider("claude-code")
        .git_repository_url("repo-a")
        .add_message("a-user-1", role="user", text="aggregate")
        .add_message("a-user-2", role="user", text="aggregate")
        .add_message("a-assistant", role="assistant", text="aggregate")
        .save()
    )
    (
        SessionBuilder(index_db, "repo-b")
        .provider("codex")
        .git_repository_url("repo-b")
        .add_message("b-assistant-1", role="assistant", text="aggregate")
        .add_message("b-assistant-2", role="assistant", text="aggregate")
        .add_message("b-system", role="system", text="aggregate")
        .save()
    )
    (
        SessionBuilder(index_db, "missing-repo")
        .provider("chatgpt")
        .git_repository_url(None)
        .add_message("missing-user", role="user", text="aggregate")
        .save()
    )
    (
        SessionBuilder(index_db, "unknown-repo")
        .provider("claude-code")
        .git_repository_url("unknown")
        .add_message("unknown-assistant", role="assistant", text="aggregate")
        .save()
    )
    source = parse_unit_source_expression(
        "messages where text:aggregate | group by role, session.repo | count | sort by count desc"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        full = query_unit_rows(archive, source, query="multi-aggregate-full", limit=100)
        assert isinstance(full, QueryUnitAggregateEnvelope)
        expected = _aggregate_page_rows(full)

        concatenated: list[tuple[tuple[str, str], int]] = []
        offset = 0
        while True:
            page = query_unit_rows(
                archive,
                source,
                query="multi-aggregate-page",
                limit=2,
                offset=offset,
            )
            assert isinstance(page, QueryUnitAggregateEnvelope)
            assert page.pipeline is not None
            result = cast(dict[str, object], page.pipeline["result"])
            assert result["denominator"] == {"kind": "all_matching_rows", "n": 8}
            assert result["missing_counts"] == {"role": 0, "session.repo": 1}
            assert result["unknown_counts"] == {"role": 0, "session.repo": 1}
            concatenated.extend(_aggregate_page_rows(page))
            if page.next_offset is None:
                break
            offset = page.next_offset

        beyond = query_unit_rows(
            archive,
            source,
            query="multi-aggregate-beyond",
            limit=2,
            offset=100,
        )

    assert concatenated == expected
    assert sum(count for _, count in concatenated) == 8
    assert isinstance(beyond, QueryUnitAggregateEnvelope)
    assert beyond.items == ()
    assert beyond.next_offset is None
    assert beyond.pipeline is not None
    beyond_result = cast(dict[str, object], beyond.pipeline["result"])
    assert beyond_result["denominator"] == {"kind": "all_matching_rows", "n": 8}
    assert beyond_result["missing_counts"] == {"role": 0, "session.repo": 1}
    assert beyond_result["unknown_counts"] == {"role": 0, "session.repo": 1}


def test_multi_field_group_does_not_call_terminal_row_query(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring ``_all_aggregate_rows`` makes this fail on ``query_messages``."""

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "sql-only")
        .provider("claude-code")
        .git_repository_url("repo")
        .add_message("sql-only-message", role="assistant", text="aggregate")
        .save()
    )
    source = parse_unit_source_expression("messages where text:aggregate | group by role, session.repo | count")
    assert source is not None

    def _row_materialization_mutant(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("multi-field aggregation called the terminal row-page method")

    monkeypatch.setattr(ArchiveStore, "query_messages", _row_materialization_mutant)
    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="multi-aggregate-sql-only", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert [(json.loads(row.group_key or "{}"), row.count) for row in envelope.items] == [
        ({"role": "assistant", "session.repo": "repo"}, 1)
    ]


def test_file_multi_field_group_uses_lossless_file_relation(workspace_env: dict[str, Path]) -> None:
    """The file unit keeps its de-duplicated session/path grain inside SQLite."""

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "file-multi")
        .provider("claude-code")
        .git_repository_url("polylogue")
        .add_message(
            "file-edit-1",
            role="assistant",
            text="aggregate file",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Edit",
                    "tool_id": "edit-1",
                    "input": {"file_path": "polylogue/archive/query/unit_results.py"},
                    "semantic_type": "file_edit",
                }
            ],
        )
        .add_message(
            "file-edit-2",
            role="assistant",
            text="aggregate file again",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Edit",
                    "tool_id": "edit-2",
                    "input": {"file_path": "polylogue/archive/query/unit_results.py"},
                    "semantic_type": "file_edit",
                }
            ],
        )
        .save()
    )
    source = parse_unit_source_expression(
        "files where action:file_edit | group by path, session.repo | count | sort by key asc"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="file-multi", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert [(json.loads(row.group_key or "{}"), row.count) for row in envelope.items] == [
        (
            {
                "path": "polylogue/archive/query/unit_results.py",
                "session.repo": "polylogue",
            },
            1,
        )
    ]
    assert envelope.pipeline is not None
    result = cast(dict[str, object], envelope.pipeline["result"])
    assert result["denominator"] == {"kind": "all_matching_rows", "n": 1}


def test_multi_field_group_distinguishes_missing_empty_and_explicit_unknown(
    workspace_env: dict[str, Path],
) -> None:
    """Lossless group keys must not collapse three different data states.

    Replacing the closed field expressions with ``COALESCE(NULLIF(...),
    'unknown')`` makes the three rows merge and falsifies both quality counters.
    """

    import sqlite3

    index_db = workspace_env["archive_root"] / "index.db"
    for native_id, repository_url in (
        ("missing", None),
        ("empty", "temporary"),
        ("explicit-unknown", "unknown"),
    ):
        (
            SessionBuilder(index_db, native_id)
            .provider("claude-code")
            .git_repository_url(repository_url)
            .add_message(f"{native_id}-message", role="assistant", text="quality-state")
            .save()
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "UPDATE sessions SET git_repository_url = '' WHERE native_id = ? AND origin = ?",
            ("ext-empty", "claude-code-session"),
        )

    source = parse_unit_source_expression(
        "messages where text:quality-state | group by role, session.repo | count | sort by key asc"
    )
    assert source is not None
    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="quality-states", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert [(json.loads(row.group_key or "{}"), row.count) for row in envelope.items] == [
        ({"role": "assistant", "session.repo": ""}, 1),
        ({"role": "assistant", "session.repo": "[missing]"}, 1),
        ({"role": "assistant", "session.repo": "unknown"}, 1),
    ]
    assert envelope.pipeline is not None
    result = cast(dict[str, object], envelope.pipeline["result"])
    assert result["denominator"] == {"kind": "all_matching_rows", "n": 3}
    assert result["missing_counts"] == {"role": 0, "session.repo": 1}
    assert result["unknown_counts"] == {"role": 0, "session.repo": 1}


def test_multi_field_sql_lowerer_covers_action_block_event_and_delegation_relations(
    workspace_env: dict[str, Path],
) -> None:
    """Each non-file derived relation stays inside the shared SQL lowerer.

    Removing the action-bound CTE, observed-event source CTE composition, or
    delegation-view field mapping makes the corresponding real query fail.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "derived-relations")
        .provider("claude-code")
        .git_repository_url("polylogue")
        .add_message(
            "derived-actions",
            role="assistant",
            text="derived relation evidence",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Bash",
                    "tool_id": "bash-1",
                    "input": {"command": "pytest -q"},
                    "semantic_type": "shell",
                },
                {
                    "type": "tool_result",
                    "tool_id": "bash-1",
                    "text": "passed",
                    "is_error": False,
                    "exit_code": 0,
                },
                {
                    "type": "tool_use",
                    "tool_name": "Task",
                    "tool_id": "task-1",
                    "input": {"prompt": "review", "model": "sonnet"},
                    "semantic_type": "subagent",
                },
                {
                    "type": "tool_result",
                    "tool_id": "task-1",
                    "text": "done",
                    "is_error": False,
                    "exit_code": 0,
                },
            ],
        )
        .save()
    )
    session_id = "claude-code-session:ext-derived-relations"
    cases = (
        (
            f"actions where session.id:{session_id} AND tool:Bash "
            "| group by tool, session.repo | count | sort by key asc",
            [({"tool": "Bash", "session.repo": "polylogue"}, 1)],
        ),
        (
            f"blocks where session.id:{session_id} AND type:tool_use "
            "| group by type, session.repo | count | sort by key asc",
            [({"type": "tool_use", "session.repo": "polylogue"}, 2)],
        ),
        (
            f"observed-events where session.id:{session_id} AND kind:tool_finished "
            "| group by tool, session.repo | count | sort by key asc",
            [
                ({"tool": "Bash", "session.repo": "polylogue"}, 1),
                ({"tool": "Task", "session.repo": "polylogue"}, 1),
            ],
        ),
        (
            "delegations where mapping_state:unresolved | group by basis, session.repo | count | sort by key asc",
            [({"basis": "action", "session.repo": "polylogue"}, 1)],
        ),
    )

    with ArchiveStore.open_existing(index_db.parent) as archive:
        for expression, expected in cases:
            source = parse_unit_source_expression(expression)
            assert source is not None
            envelope = query_unit_rows(archive, source, query=expression, limit=10)
            assert isinstance(envelope, QueryUnitAggregateEnvelope)
            assert [(json.loads(row.group_key or "{}"), row.count) for row in envelope.items] == expected
            assert envelope.pipeline is not None
            result = cast(dict[str, object], envelope.pipeline["result"])
            assert result["denominator"] == {
                "kind": "all_matching_rows",
                "n": sum(count for _, count in expected),
            }


def test_multi_field_sql_lowerer_covers_assertion_defaults_and_session_join(
    workspace_env: dict[str, Path],
) -> None:
    """Assertion defaults and owning-session metadata are grouped in SQLite."""

    import sqlite3

    from polylogue.core.enums import AssertionKind
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "assertion-multi")
        .provider("claude-code")
        .git_repository_url("polylogue")
        .add_message("assertion-source", role="user", text="record caveat")
        .save()
    )
    session_id = "claude-code-session:ext-assertion-multi"
    with sqlite3.connect(index_db.parent / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="assertion-multi-1",
            target_ref=f"session:{session_id}",
            kind=AssertionKind.CAVEAT,
            body_text="review this caveat",
            author_ref="user:test",
            author_kind="user",
            evidence_refs=[session_id],
            now_ms=1_700_000_000_000,
        )

    source = parse_unit_source_expression(
        "assertions where kind:caveat | group by status, session.repo | count | sort by key asc"
    )
    assert source is not None
    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="assertion-multi", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert [(json.loads(row.group_key or "{}"), row.count) for row in envelope.items] == [
        ({"status": "active", "session.repo": "polylogue"}, 1)
    ]
    assert envelope.pipeline is not None
    result = cast(dict[str, object], envelope.pipeline["result"])
    assert result["missing_counts"] == {"status": 0, "session.repo": 0}
    assert result["unknown_counts"] == {"status": 0, "session.repo": 0}
