"""Real (non-fake) production evaluator: canonical plan -> live archive rows.

These tests exercise the actual dependency the earlier substrate PRs (#2813,
#2826) left unwired: ``ArchiveCanonicalPlanEvaluator`` reconstructs a typed
predicate from a durable ``query:<hash>`` definition and runs it through the
same ``SessionFilter`` execution path every real surface uses -- no test
double stands in for the planner. Removing the predicate reconstruction (or
the ``SessionFilter`` call) makes ``test_evaluate_matches_real_archive_rows``
fail, since the assertion depends on the archive actually filtering by
origin. Removing the ``record_query_run`` call makes
``test_evaluate_records_a_production_query_run`` fail, since it reads the
row back from ``ops.db``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.query.evaluator import QueryEvaluationRequest
from polylogue.archive.query.production_evaluator import (
    ArchiveCanonicalPlanEvaluator,
    LegacyQueryDefinitionNotExecutableError,
    UnsupportedEvaluationGrainError,
)
from polylogue.core.enums import BlockType, Provider
from polylogue.core.query_identity import LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION, JsonValue
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import QueryObject, put_query


def _seed_archive(archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        for provider, native_id, title in (
            (Provider.CODEX, "codex-1", "codex session"),
            (Provider.CLAUDE_CODE, "claude-1", "claude session"),
        ):
            archive.write_parsed(
                ParsedSession(
                    source_name=provider,
                    provider_session_id=native_id,
                    title=title,
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:01:00+00:00",
                    messages=[
                        ParsedMessage(
                            provider_message_id=f"{native_id}-m1",
                            role=Role.USER,
                            text="hello",
                            timestamp="2026-01-01T00:00:00+00:00",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello")],
                        )
                    ],
                )
            )
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    initialize_archive_database(archive_root / "ops.db", ArchiveTier.OPS)


def _origin_query(conn: sqlite3.Connection, *, origin: str) -> QueryObject:
    ast: dict[str, JsonValue] = {
        "kind": "field",
        "field": "origin",
        "op": "=",
        "values": [origin],
    }
    return put_query(conn, ast, grain="session", lane="dialogue", rank_policy="mixed", created_at_ms=1)


def test_evaluate_matches_real_archive_rows(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()

    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    evaluation = evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))

    assert evaluation.grain == "session"
    assert evaluation.exactness == "exact"
    assert len(evaluation.member_refs) == 1
    assert evaluation.member_refs[0].startswith("session:codex-session:")
    assert evaluation.receipt.runtime_build_ref.startswith("polylogue:")


def test_evaluate_excludes_origin_prefix(tmp_path: Path) -> None:
    """The self-trigger firewall drops members whose origin matches an excluded prefix."""
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()

    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    evaluation = evaluator.evaluate(
        QueryEvaluationRequest(query=query, purpose="standing-watch", excluded_origin_prefixes=("codex-",))
    )

    assert evaluation.member_refs == ()


def test_evaluate_excludes_scope_refs(tmp_path: Path) -> None:
    """The self-trigger firewall also drops explicitly excluded member refs."""
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()

    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    baseline = evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))
    assert len(baseline.member_refs) == 1

    evaluation = evaluator.evaluate(
        QueryEvaluationRequest(query=query, purpose="standing-watch", excluded_scope_refs=(baseline.member_refs[0],))
    )
    assert evaluation.member_refs == ()


def test_evaluate_records_a_production_query_run(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="claude-code-session")
        conn.commit()

    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))

    with sqlite3.connect(archive_root / "ops.db") as conn:
        rows = conn.execute("SELECT query_hash, surface, member_count, exactness FROM query_runs").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == query.query_hash
    assert rows[0][1] == "daemon-internal"
    assert rows[0][2] == 1
    assert rows[0][3] == "exact"


def test_evaluate_rejects_legacy_protocol_v0() -> None:
    legacy = QueryObject(
        query_hash="b" * 64,
        canonical_plan={"field": "origin", "value": "codex-session"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        definition_protocol_version=LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION,
    )
    evaluator = ArchiveCanonicalPlanEvaluator(Path("/nonexistent/index.db"))
    with pytest.raises(LegacyQueryDefinitionNotExecutableError):
        evaluator.evaluate(QueryEvaluationRequest(query=legacy, purpose="reference"))


def test_evaluate_rejects_unsupported_grain(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        ast: dict[str, JsonValue] = {"kind": "field", "field": "tool_name", "op": "=", "values": ["Bash"]}
        query = put_query(conn, ast, grain="action", lane="dialogue", rank_policy="mixed", created_at_ms=1)
        conn.commit()

    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    with pytest.raises(UnsupportedEvaluationGrainError):
        evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))


def test_resolve_cohort_is_not_yet_implemented(tmp_path: Path) -> None:
    from polylogue.archive.query.expression import RefOperand
    from polylogue.core.refs import ObjectRef

    evaluator = ArchiveCanonicalPlanEvaluator(tmp_path / "index.db")
    with pytest.raises(NotImplementedError):
        evaluator.resolve_cohort(RefOperand(ObjectRef(kind="cohort", object_id="team")))
