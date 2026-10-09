from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.archive.query.predicate import QueryPredicate
from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.archive_query_reads import ArchiveAggMetricSpec
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from tests.infra.storage_records import SessionBuilder


def _predicate(expression: str) -> QueryPredicate:
    source = parse_unit_source_expression(expression)
    assert source is not None
    return source.predicate


@pytest.mark.parametrize("negated", [False, True])
def test_terminal_owning_session_alternatives_match_explicit_or(workspace_env: dict[str, Path], negated: bool) -> None:
    """Dropping an alternative in either the SQL predicate or physical action
    bound changes the page or aggregate compared with the explicit OR control.
    """
    root = workspace_env["archive_root"]
    ids = []
    for name in ("a", "b", "c"):
        builder = (
            SessionBuilder(root / "index.db", name)
            .provider("codex")
            .add_message(
                "action",
                role="assistant",
                text="neutral",
                blocks=[
                    {"type": "tool_use", "tool_name": "shell", "tool_id": "call", "input": {"command": "true"}},
                    {"type": "tool_result", "tool_id": "call", "text": "failed", "is_error": True, "exit_code": 1},
                ],
            )
        )
        builder.save()
        ids.append(builder.native_session_id())
    compact = f"session.id:({ids[0]}|{ids[1]})"
    expanded = f"(session.id:{ids[0]} or session.id:{ids[1]})"
    if negated:
        compact, expanded = f"not {compact}", f"not {expanded}"
    expected = {ids[2]} if negated else set(ids[:2])
    with ArchiveStore.open_existing(root) as archive:
        predicates = [_predicate(f"messages where {text}") for text in (compact, expanded)]
        for predicate in predicates:
            rows = archive.query_message_projection(predicate, fields=["session_id"])
            assert {row["session_id"] for row in rows} == expected
        action_predicates = [_predicate(f"actions where {text}") for text in (compact, expanded)]
        for predicate in action_predicates:
            assert {row.session_id for row in archive.query_actions(predicate)} == expected
            assert archive.query_unit_counts("action", predicate)[0].count == len(expected)
            assert archive.query_unit_multi_counts("action", predicate, group_by=("tool", "type")).denominator == len(
                expected
            )
            metrics = archive.query_unit_agg_metrics(
                "action", predicate, metrics=(ArchiveAggMetricSpec(fn="count", field=None, label="count"),)
            )
            assert metrics.rows[0].count == len(expected)
        # This predicate exercises the bounded action relation on the row path.
        for text in (compact, expanded):
            predicate = _predicate(f"actions where {text} and followup_class:ambiguous")
            assert {row.session_id for row in archive.query_actions(predicate)} == expected
        contradiction = _predicate(f"actions where session.id:{ids[0]} and session.id:{ids[1]}")
        assert archive.query_unit_counts("action", contradiction) == []


@pytest.mark.parametrize("field, values", [("author", ("alice", "bob")), ("context", ("alpha", "beta"))])
@pytest.mark.parametrize("negated", [False, True])
def test_assertion_alternatives_preserve_default_expression_bindings(
    workspace_env: dict[str, Path], field: str, values: tuple[str, str], negated: bool
) -> None:
    """Each alternative used to repeat a default placeholder without its bind.
    Row and aggregate routes must agree with separately lowered OR leaves.
    """
    root = workspace_env["archive_root"]
    with sqlite3.connect(root / "user.db") as conn:
        for index, (author, context) in enumerate((("alice", "alpha"), ("bob", "beta"), ("carol", "gamma"))):
            upsert_assertion(
                conn,
                assertion_id=f"alternative-{index}",
                target_ref="session:neutral",
                kind=AssertionKind.CAVEAT,
                author_ref=f"user:{author}",
                context_policy={"fixture": context},
                now_ms=1_700_000_000_000,
            )
    compact = f"{field}:({values[0]}|{values[1]})"
    expanded = f"({field}:{values[0]} or {field}:{values[1]})"
    if negated:
        compact, expanded = f"not {compact}", f"not {expanded}"
    expected = {"alternative-2"} if negated else {"alternative-0", "alternative-1"}
    with ArchiveStore.open_existing(root) as archive:
        for text in (compact, expanded):
            predicate = _predicate(f"assertions where {text}")
            assert {row.assertion_id for row in archive.query_assertions(predicate)} == expected
            assert archive.query_unit_counts("assertion", predicate)[0].count == len(expected)
            assert archive.query_unit_multi_counts(
                "assertion", predicate, group_by=("kind", "status")
            ).denominator == len(expected)


def test_assertion_alternatives_keep_null_defaults(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    with sqlite3.connect(root / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="defaulted",
            target_ref="session:neutral",
            kind=AssertionKind.CAVEAT,
            now_ms=1_700_000_000_000,
        )
        conn.execute(
            "UPDATE assertions SET author_ref = NULL, context_policy_json = NULL WHERE assertion_id = 'defaulted'"
        )
    with ArchiveStore.open_existing(root) as archive:
        for text in ("author:(user:local|missing)", "context:(inject|missing)"):
            predicate = _predicate(f"assertions where {text}")
            assert [row.assertion_id for row in archive.query_assertions(predicate)] == ["defaulted"]
