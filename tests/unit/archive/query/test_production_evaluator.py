"""Real (non-fake) production evaluator: canonical plan -> live archive rows.

These tests exercise the actual dependency the earlier substrate PRs (#2813,
#2826) left unwired: ``ArchiveCanonicalPlanEvaluator`` reconstructs a typed
predicate from a durable ``query:<hash>`` definition and runs it through the
same ``SessionFilter`` execution path every real surface uses -- no test
double stands in for the planner. Removing the predicate reconstruction (or
the ``SessionFilter`` call) makes ``test_evaluate_matches_real_archive_rows``
fail, since the assertion depends on the archive actually filtering by
origin.
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
from tests.infra.live_ingest import write_index_session


def _seed_archive(archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        for provider, native_id, title in (
            (Provider.CODEX, "codex-1", "codex session"),
            (Provider.CLAUDE_CODE, "claude-1", "claude session"),
        ):
            write_index_session(
                archive,
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
                ),
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


def test_evaluate_does_not_truncate_at_the_default_page_limit(tmp_path: Path) -> None:
    """A watched query matching >50 sessions must enumerate every member.

    ``SessionFilter.list_summaries`` caps at a default page limit of 50 rows.
    If the evaluator used that page-capped read path instead of
    ``list_all_summaries``, this test's 60th+ matching session would be
    silently dropped from ``member_refs`` while ``exactness`` still claimed
    ``"exact"`` -- corrupting the merkle-root-based standing-query drift
    detection that trusts ``member_refs`` as a complete enumeration.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    session_count = 60
    with ArchiveStore(archive_root) as archive:
        for i in range(session_count):
            native_id = f"codex-{i:03d}"
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    title=f"codex session {i}",
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
                ),
            )
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)

    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()

    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    evaluation = evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="standing-watch"))

    assert evaluation.exactness == "exact"
    assert len(evaluation.member_refs) == session_count


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


def test_saved_exact_id_query_preserves_opaque_unicode_through_selection(tmp_path: Path) -> None:
    """Mutation: canonicalization NFC-folds the ID and selects a different session."""
    from typing import cast

    from polylogue.archive.query.predicate import QueryFieldPredicate
    from polylogue.storage.sqlite.query_objects import get_query

    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    native_ids = ("cafe\u0301", "café")
    with ArchiveStore(archive_root) as archive:
        for native_id in native_ids:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    title="neutral session",
                    messages=[
                        ParsedMessage(provider_message_id=f"{native_id}-m", role=Role.USER, text="neutral content")
                    ],
                ),
            )
    evaluator = ArchiveCanonicalPlanEvaluator(archive_root / "index.db")
    hashes = []
    for native_id in native_ids:
        session_id = f"codex-session:{native_id}"
        predicate = QueryFieldPredicate(field="id", values=(session_id,))
        with sqlite3.connect(archive_root / "user.db") as conn:
            saved = put_query(
                conn,
                cast(dict[str, JsonValue], predicate.to_payload()),
                grain="session",
                lane="dialogue",
                rank_policy="mixed",
                created_at_ms=1,
            )
            conn.commit()
            loaded = get_query(conn, saved.query_hash)
        assert loaded is not None
        evaluation = evaluator.evaluate(QueryEvaluationRequest(query=loaded, purpose="reference"))
        assert evaluation.member_refs == (f"session:{session_id}",)
        assert loaded.canonical_plan["ast"] == predicate.to_payload()
        hashes.append(loaded.query_hash)
    assert len(set(hashes)) == 2


@pytest.mark.parametrize("operator", ["and", "or"])
def test_saved_boolean_query_uses_real_wire_kind_for_commutative_identity(operator: str) -> None:
    """Mutation: sorting looks for operator/op instead of the actual typed kind."""
    from typing import cast

    from polylogue.archive.query.predicate import QueryBoolOp, QueryBoolPredicate, QueryFieldPredicate
    from polylogue.storage.sqlite.query_objects import list_watched_queries, put_query_name

    conn = sqlite3.connect(":memory:")
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    initialize_archive_tier(conn, ArchiveTier.USER)
    children = (QueryFieldPredicate("origin", ("codex-session",)), QueryFieldPredicate("tag", ("neutral",)))
    saved = []
    for ordered in (children, tuple(reversed(children))):
        predicate = QueryBoolPredicate(cast(QueryBoolOp, operator), ordered)
        saved.append(
            put_query(
                conn,
                cast(dict[str, JsonValue], predicate.to_payload()),
                grain="session",
                lane="dialogue",
                rank_policy="mixed",
                created_at_ms=1,
            )
        )
    assert saved[0].query_hash == saved[1].query_hash
    assert conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1
    for index, query in enumerate(saved):
        put_query_name(conn, name=f"neutral-{index}", query_hash=query.query_hash, updated_at_ms=2, watch=True)
    assert tuple(query.query_hash for query in list_watched_queries(conn)) == (saved[0].query_hash,)
    conn.close()
