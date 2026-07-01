"""Tests for the ``with <units>`` query-projection clause (#2492).

Covers the parser split (quote-safety, validation), the spec field, the
descriptor-driven fetch helper, and the spec-path attach onto domain models.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.query.archive_execution import list_summaries_archive
from polylogue.archive.query.attached_units import fetch_attached_units
from polylogue.archive.query.expression import (
    ExpressionCompileError,
    compile_expression,
    compile_expression_into,
    parse_expression_ast,
    split_with_clause,
)
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.enums import AssertionKind, BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

# ---------------------------------------------------------------------------
# Parser: split + validation
# ---------------------------------------------------------------------------


class TestWithClauseParsing:
    def test_compact_clause_sets_units_and_keeps_head_filters(self) -> None:
        spec = compile_expression("repo:polylogue with assertions")
        assert spec.repo_names == ("polylogue",)
        assert spec.with_units == ("assertion",)

    def test_boolean_clause_sets_units(self) -> None:
        spec = compile_expression("sessions where repo:polylogue with assertions")
        assert spec.with_units == ("assertion",)
        assert spec.boolean_predicate is not None

    def test_quoted_with_is_not_a_clause(self) -> None:
        spec = compile_expression('"deploy with caveats"')
        assert spec.with_units == ()
        assert spec.query_terms == ("deploy with caveats",)

    def test_parenthesized_with_is_not_split(self) -> None:
        # ``with`` inside an alternation/group must not register a boundary.
        head, units = split_with_clause("(origin:codex-session) with assertions")
        assert units == ("assertion",)
        assert head == "(origin:codex-session)"

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown query unit"):
            compile_expression("repo:polylogue with bogus")

    def test_unsupported_unit_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not yet supported"):
            compile_expression("repo:polylogue with actions")

    def test_duplicate_units_dedup(self) -> None:
        _head, units = split_with_clause("repo:polylogue with assertions, assertions")
        assert units == ("assertion",)

    def test_empty_unit_token_raises(self) -> None:
        with pytest.raises(ExpressionCompileError):
            compile_expression("repo:polylogue with assertions,")

    def test_clause_without_head_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="requires a selection expression"):
            compile_expression("with assertions")

    def test_trailing_bare_with_is_fts_text(self) -> None:
        # A bare trailing ``with`` is not a projection clause.
        head, units = split_with_clause("timeout with")
        assert units == ()
        assert head == "timeout with"

    def test_parse_expression_ast_strips_clause(self) -> None:
        # The AST describes the selection only; stripping must not crash.
        ast = parse_expression_ast("repo:polylogue with assertions")
        assert ast is not None

    def test_compile_expression_into_merges_units(self) -> None:
        base = SessionQuerySpec(with_units=())
        merged = compile_expression_into("repo:polylogue with assertions", base)
        assert merged.with_units == ("assertion",)
        assert merged.repo_names == ("polylogue",)


# ---------------------------------------------------------------------------
# Fetch + attach behaviour (seeded archive)
# ---------------------------------------------------------------------------


def _seed_session_with_assertion(root: Path) -> str:
    with ArchiveStore(root) as archive_db:
        archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_AI,
                provider_session_id="conv-alpha",
                title="Alpha",
                messages=[
                    ParsedMessage(
                        provider_message_id="alpha-m1",
                        role=Role.USER,
                        text="alpha body",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="alpha body")],
                    ),
                ],
            )
        )
    session_id = "claude-ai-export:conv-alpha"
    conn = sqlite3.connect(root / "user.db")
    try:
        initialize_archive_tier(conn, ArchiveTier.USER)
        upsert_assertion(
            conn,
            assertion_id="caveat-alpha",
            target_ref=f"session:{session_id}",
            kind=AssertionKind.CAVEAT,
            body_text="Review findings not read yet.",
            author_ref="user:test",
            author_kind="user",
            evidence_refs=[session_id],
            status="active",
            visibility="private",
            now_ms=1_700_000_000_000,
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


class TestAttachBehaviour:
    def test_fetch_attached_units_buckets_by_session(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, [session_id], ["assertion"])
        assert set(attached) == {"assertion"}
        rows = attached["assertion"][session_id]
        assert len(rows) == 1
        assert rows[0]["body_text"] == "Review findings not read yet."
        assert rows[0]["unit"] == "assertion"

    def test_fetch_attached_units_empty_for_session_without_rows(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, [session_id, "missing:session"], ["assertion"])
        # The session without assertions is simply absent from the bucket.
        assert "missing:session" not in attached["assertion"]
        assert session_id in attached["assertion"]

    async def test_spec_path_attaches_units_to_summaries(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        summaries = await list_summaries_archive(
            SessionQueryPlan(),
            archive_root=tmp_path,
            config=None,
            with_units=("assertion",),
        )
        target = next(summary for summary in summaries if summary.id == session_id)
        assert "assertion" in target.attached_units
        assert len(target.attached_units["assertion"]) == 1
        assert target.attached_units["assertion"][0]["body_text"] == "Review findings not read yet."

    async def test_spec_path_no_units_leaves_attached_empty(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        summaries = await list_summaries_archive(
            SessionQueryPlan(),
            archive_root=tmp_path,
            config=None,
        )
        target = next(summary for summary in summaries if summary.id == session_id)
        assert target.attached_units == {}
