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
    WithUnitWindow,
    compile_expression,
    compile_expression_into,
    parse_expression_ast,
    split_with_clause,
    split_with_projection_clause,
)
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.enums import AssertionKind, BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from tests.infra.identity import archive_message_id
from tests.infra.live_ingest import write_index_session

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

    def test_evidence_units_are_supported(self) -> None:
        spec = compile_expression("repo:polylogue with messages, actions, files")
        assert spec.with_units == ("message", "action", "file")

    def test_unit_field_selection_is_parsed(self) -> None:
        spec = compile_expression("repo:polylogue with messages(message_id, role, text), actions(tool_name,is_error)")
        assert spec.with_units == ("message", "action")
        assert spec.with_unit_fields == {
            "message": ("message_id", "role", "text"),
            "action": ("tool_name", "is_error"),
        }

    def test_split_with_projection_clause_returns_field_map(self) -> None:
        head, units, fields, windows = split_with_projection_clause(
            "repo:polylogue with files(path,action_count), assertions(assertion_id,body_text)"
        )
        assert head == "repo:polylogue"
        assert units == ("file", "assertion")
        assert fields == {
            "file": ("path", "action_count"),
            "assertion": ("assertion_id", "body_text"),
        }
        assert windows == {}

    def test_empty_unit_field_selection_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="field selection"):
            compile_expression("repo:polylogue with messages(message_id,)")

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

    def test_compile_expression_into_merges_unit_fields(self) -> None:
        base = SessionQuerySpec(
            with_units=("message",),
            with_unit_fields={"message": ("message_id",)},
        )
        merged = compile_expression_into("repo:polylogue with messages(role,text), actions(tool_name)", base)
        assert merged.with_units == ("message", "action")
        assert merged.with_unit_fields == {
            "message": ("role", "text"),
            "action": ("tool_name",),
        }


# ---------------------------------------------------------------------------
# Fetch + attach behaviour (seeded archive)
# ---------------------------------------------------------------------------


def _seed_session_with_assertion(root: Path) -> str:
    with ArchiveStore(root) as archive_db:
        write_index_session(
            archive_db,
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
            ),
        )
    session_id = "claude-ai-export:conv-alpha"
    conn = sqlite3.connect(root / "user.db")
    try:
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
        assert set(attached.rows) == {"assertion"}
        rows = attached.rows["assertion"][session_id]
        assert len(rows) == 1
        assert rows[0]["body_text"] == "Review findings not read yet."
        assert rows[0]["unit"] == "assertion"

    def test_fetch_attached_units_empty_for_session_without_rows(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, [session_id, "missing:session"], ["assertion"])
        # The session without assertions is simply absent from the bucket.
        assert "missing:session" not in attached.rows["assertion"]
        assert session_id in attached.rows["assertion"]

    def test_fetch_attached_evidence_units(self, tmp_path: Path) -> None:
        from tests.infra.storage_records import SessionBuilder

        index_db = tmp_path / "index.db"
        (
            SessionBuilder(index_db, "evidence")
            .provider("claude-code")
            .title("Evidence attachment")
            .add_message("m-user", role="user", text="please edit the query projection")
            .add_message(
                "m-assistant",
                role="assistant",
                text="editing query projection",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-edit",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "tool-edit",
                        "text": "result " * 500,
                    },
                ],
            )
            .save()
        )
        session_id = "claude-code-session:ext-evidence"

        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, [session_id], ["message", "action", "file"])

        assert set(attached.rows) == {"message", "action", "file"}
        assert {row["message_id"] for row in attached.rows["message"][session_id]} == {
            archive_message_id("claude-code-session:ext-evidence", "m-user"),
            archive_message_id("claude-code-session:ext-evidence", "m-assistant"),
        }
        action_payload = attached.rows["action"][session_id][0]
        output_text = action_payload["output_text"]
        truncated_chars = action_payload["output_text_truncated_chars"]
        assert action_payload["tool_name"] == "Edit"
        assert action_payload["semantic_type"] == "file_edit"
        assert isinstance(output_text, str)
        assert isinstance(truncated_chars, int)
        assert len(output_text) == 2000
        assert truncated_chars > 0
        assert attached.rows["file"][session_id][0]["path"] == "polylogue/archive/query/expression.py"

    def test_fetch_attached_units_applies_payload_field_selection(self, tmp_path: Path) -> None:
        from tests.infra.storage_records import SessionBuilder

        index_db = tmp_path / "index.db"
        (
            SessionBuilder(index_db, "field-select")
            .provider("claude-code")
            .title("Field selection")
            .add_message("m-user", role="user", text="please edit field projection")
            .save()
        )
        session_id = "claude-code-session:ext-field-select"

        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(
                archive,
                [session_id],
                ["message"],
                unit_fields={"message": ("message_id", "role")},
            )

        assert attached.rows["message"][session_id] == (
            {
                "message_id": archive_message_id("claude-code-session:ext-field-select", "m-user"),
                "role": "user",
            },
        )

    def test_fetch_attached_units_rejects_unknown_payload_field(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            with pytest.raises(ValueError, match="unsupported field"):
                fetch_attached_units(
                    archive,
                    [session_id],
                    ["assertion"],
                    unit_fields={"assertion": ("not_a_payload_field",)},
                )

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

    async def test_spec_path_applies_unit_field_selection(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        summaries = await list_summaries_archive(
            SessionQueryPlan(),
            archive_root=tmp_path,
            config=None,
            with_units=("assertion",),
            with_unit_fields={"assertion": ("assertion_id", "body_text")},
        )
        target = next(summary for summary in summaries if summary.id == session_id)
        assert target.attached_units["assertion"] == (
            {
                "assertion_id": "caveat-alpha",
                "body_text": "Review findings not read yet.",
            },
        )

    async def test_spec_path_no_units_leaves_attached_empty(self, tmp_path: Path) -> None:
        session_id = _seed_session_with_assertion(tmp_path)
        summaries = await list_summaries_archive(
            SessionQueryPlan(),
            archive_root=tmp_path,
            config=None,
        )
        target = next(summary for summary in summaries if summary.id == session_id)
        assert target.attached_units == {}


# ---------------------------------------------------------------------------
# Bracket predicate/window clause: ``with unit[field:value, last:N]`` (polylogue-fnm.2)
# ---------------------------------------------------------------------------


class TestWithUnitBracketParsing:
    def test_bracket_predicate_and_window_are_parsed(self) -> None:
        spec = compile_expression("repo:polylogue with messages[role:user, last:20]")
        assert spec.with_units == ("message",)
        assert spec.with_unit_windows == {
            "message": WithUnitWindow(predicates=(("role", "user"),), window=("last", 20)),
        }

    def test_bracket_combines_with_field_selection(self) -> None:
        spec = compile_expression("repo:polylogue with messages(message_id,role,text)[role:user, first:5]")
        assert spec.with_unit_fields == {"message": ("message_id", "role", "text")}
        assert spec.with_unit_windows == {
            "message": WithUnitWindow(predicates=(("role", "user"),), window=("first", 5)),
        }

    def test_bracket_predicate_only_no_window(self) -> None:
        spec = compile_expression("repo:polylogue with actions[tool:Bash]")
        assert spec.with_unit_windows == {
            "action": WithUnitWindow(predicates=(("tool", "Bash"),), window=None),
        }

    def test_bracket_window_only_no_predicate(self) -> None:
        spec = compile_expression("repo:polylogue with messages[last:10]")
        assert spec.with_unit_windows == {"message": WithUnitWindow(predicates=(), window=("last", 10))}

    def test_bracket_comma_inside_does_not_break_item_splitting(self) -> None:
        # A literal comma inside ``[...]`` must not be mistaken for the
        # top-level ``with`` clause's unit-separating comma.
        spec = compile_expression("repo:polylogue with messages[role:user, last:5], actions[tool:Bash]")
        assert spec.with_units == ("message", "action")
        assert spec.with_unit_windows == {
            "message": WithUnitWindow(predicates=(("role", "user"),), window=("last", 5)),
            "action": WithUnitWindow(predicates=(("tool", "Bash"),), window=None),
        }

    def test_unsupported_bracket_field_names_unit_field_and_supported_set(self) -> None:
        with pytest.raises(
            ExpressionCompileError,
            match=r"messages\[nope:\.\.\.\] is not a supported bracket field for messages rows; "
            r"supported fields: role, type",
        ):
            compile_expression("repo:polylogue with messages[nope:1]")

    def test_unsupported_window_key_value_rejected(self) -> None:
        with pytest.raises(ExpressionCompileError, match="window size must be a positive integer"):
            compile_expression("repo:polylogue with messages[last:0]")
        with pytest.raises(ExpressionCompileError, match="window size must be a positive integer"):
            compile_expression("repo:polylogue with messages[last:abc]")

    def test_two_windows_on_same_unit_rejected(self) -> None:
        with pytest.raises(ExpressionCompileError, match="can only carry one first:N or last:N window"):
            compile_expression("repo:polylogue with messages[first:5, last:5]")

    def test_malformed_bracket_clause_rejected(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not a valid bracket clause"):
            compile_expression("repo:polylogue with messages[justaword]")


class TestWithUnitBracketExecution:
    def _seed_role_sequence(self, tmp_path: Path) -> str:
        from tests.infra.storage_records import SessionBuilder

        index_db = tmp_path / "index.db"
        (
            SessionBuilder(index_db, "bracket-window")
            .provider("claude-code")
            .add_message("m1", role="user", text="one")
            .add_message("m2", role="assistant", text="two three")
            .add_message("m3", role="user", text="four five six")
            .add_message("m4", role="assistant", text="seven")
            .add_message("m5", role="user", text="eight nine ten eleven")
            .save()
        )
        return "claude-code-session:ext-bracket-window"

    def test_bracket_predicate_filters_attached_rows(self, tmp_path: Path) -> None:
        session_id = self._seed_role_sequence(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(
                archive,
                [session_id],
                ["message"],
                unit_windows={"message": WithUnitWindow(predicates=(("role", "user"),))},
            )
        rows = attached.rows["message"][session_id]
        assert [row["role"] for row in rows] == ["user", "user", "user"]
        assert [row["text"] for row in rows] == ["one", "four five six", "eight nine ten eleven"]

    def test_last_window_returns_true_tail_not_head(self, tmp_path: Path) -> None:
        """``last:N`` must return the session's actual tail, not the head of a capped fetch.

        Anti-vacuity: this seeds more than one message so a naive fetch that
        only ever reads ascending-from-start and then slices ``[-n:]`` would
        coincidentally look right for a 5-message session; the assertion pins
        the exact tail-role predicate-filtered rows so a regression to
        ascending-only fetching (which silently mis-selects the tail once a
        session exceeds the per-session row cap, verified against the live
        archive at PR time) would fail this test.
        """
        session_id = self._seed_role_sequence(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(
                archive,
                [session_id],
                ["message"],
                unit_windows={"message": WithUnitWindow(predicates=(("role", "user"),), window=("last", 2))},
            )
        rows = attached.rows["message"][session_id]
        assert [row["text"] for row in rows] == ["four five six", "eight nine ten eleven"]

    def test_first_window_returns_head(self, tmp_path: Path) -> None:
        session_id = self._seed_role_sequence(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(
                archive,
                [session_id],
                ["message"],
                unit_windows={"message": WithUnitWindow(predicates=(("role", "user"),), window=("first", 2))},
            )
        rows = attached.rows["message"][session_id]
        assert [row["text"] for row in rows] == ["one", "four five six"]

    async def test_bracket_predicate_and_window_compose_end_to_end_via_dsl(self, tmp_path: Path) -> None:
        session_id = self._seed_role_sequence(tmp_path)
        spec = compile_expression(f"id:{session_id} with messages[role:user, last:2]")
        summaries = await list_summaries_archive(
            spec.to_plan(),
            archive_root=tmp_path,
            config=None,
            with_units=spec.with_units,
            with_unit_fields=spec.with_unit_fields,
            with_unit_windows=spec.with_unit_windows,
        )
        target = next(summary for summary in summaries if summary.id == session_id)
        assert [row["text"] for row in target.attached_units["message"]] == [
            "four five six",
            "eight nine ten eleven",
        ]


class TestAttachedRowCeilingIsReported:
    """A projection cut by the attached-unit row ceiling names its gap.

    ``_MAX_ROWS_PER_SESSION``/``_MAX_ROWS_PER_PAGE`` are real work protection
    and stay. What changed is that reaching one is reported: the fetch used to
    return a short row list with nothing anywhere saying it had been cut, so a
    250-message session answered ``with messages`` with 200 rows inside an
    ``ok`` envelope -- a complete-looking projection over a truncated row set.
    """

    @staticmethod
    def _seed_messages(tmp_path: Path, count: int, name: str) -> str:
        from tests.infra.storage_records import SessionBuilder

        builder = SessionBuilder(tmp_path / "index.db", name).provider("claude-code").title(name)
        for index in range(count):
            builder = builder.add_message(
                f"m-{index:04d}",
                role="user" if index % 2 == 0 else "assistant",
                text=f"body {index}",
            )
        builder.save()
        return f"claude-code-session:ext-{name}"

    def test_population_above_the_ceiling_names_the_gap(self, tmp_path: Path) -> None:
        """250 rows -- strictly more than the 200-row ceiling -- degrade by name.

        Anti-vacuity: fetch exactly ``fetch_limit`` rows again instead of
        ``fetch_limit + 1`` (or drop the ``ceiling_reached`` branch) and the
        result is 200 rows with ``gaps == ()``, which is the defect: a caller
        cannot tell 200-of-250 from a session that holds exactly 200.
        """
        session_id = self._seed_messages(tmp_path, 250, "over-ceiling")
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, [session_id], ["message"])
        assert len(attached.rows["message"][session_id]) == 200
        assert attached.gaps == ("attached_unit_truncated:message:200",)

    def test_population_exactly_at_the_ceiling_is_not_a_gap(self, tmp_path: Path) -> None:
        """A population equal to the bound is complete, and must not degrade.

        Anti-vacuity: compare ``len(rows) >= fetch_limit`` instead of ``>`` and
        this session -- which holds every row it was asked for -- is reported
        as truncated, turning a correct answer into a false degradation.
        """
        session_id = self._seed_messages(tmp_path, 200, "at-ceiling")
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, [session_id], ["message"])
        assert len(attached.rows["message"][session_id]) == 200
        assert attached.gaps == ()

    def test_satisfied_window_over_a_bounded_fetch_is_complete(self, tmp_path: Path) -> None:
        """``last:N`` the fetch can satisfy is exact, so it names no gap.

        The descending fetch direction means the newest 200 rows are the ones
        read, so the last 5 of a 250-message session are present and correct.

        Anti-vacuity: drop the ``provably_complete`` suppression and this
        exact answer is reported ``degraded``; break the descending fetch
        direction and the returned ids become ``m-0195..m-0199`` instead of
        ``m-0245..m-0249``.
        """
        session_id = self._seed_messages(tmp_path, 250, "tail-window")
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(
                archive,
                [session_id],
                ["message"],
                unit_windows={"message": WithUnitWindow(predicates=(), window=("last", 5))},
            )
        rows = attached.rows["message"][session_id]
        assert [str(row["message_id"]).rsplit(":", 1)[-1] for row in rows] == [
            "m-0245",
            "m-0246",
            "m-0247",
            "m-0248",
            "m-0249",
        ]
        assert attached.gaps == ()


class TestPageBudgetIsSharedEqually:
    """Every selected session gets the same allowance on one page.

    The page ceiling used to be spent in the page's own row order, so a long
    session could take the allowance a short one never received. A *tail-first*
    read is where that bit hardest: under ``last:N`` the descending fetch spent
    the whole budget on the newest rows, which all belonged to the long
    sessions, and a short session answered ``with messages`` with **zero**
    rows. Nothing in the result distinguished "this session has no messages"
    from "this session's allowance was eaten" (polylogue-fvsjn).

    Uniform sessions cannot show this: a flat page limit spread over equal-size
    sessions already lands on roughly equal counts. The defect needs the skew
    below -- one short session whose rows sort behind a crowd of long ones.
    """

    @staticmethod
    def _seed_skewed(tmp_path: Path) -> tuple[str, list[str]]:
        """One short session, then enough long ones to overrun the page budget."""
        from tests.infra.storage_records import SessionBuilder

        ids: list[str] = []
        for name, count in [("short-000", 5)] + [(f"long-{i:03d}", 400) for i in range(13)]:
            builder = SessionBuilder(tmp_path / "index.db", name).provider("claude-code").title(name)
            for index in range(count):
                builder = builder.add_message(
                    f"m-{index:04d}",
                    role="user" if index % 2 == 0 else "assistant",
                    text=f"body {index}",
                )
            builder.save()
            ids.append(f"claude-code-session:ext-{name}")
        return ids[0], ids

    def test_a_tail_window_does_not_starve_the_short_session(self, tmp_path: Path) -> None:
        """The short session gets its full ``last:5``, like every other session.

        Anti-vacuity (executed, not asserted): make ``_per_session_allowance``
        ignore ``session_count`` and cap the fetch at ``_MAX_ROWS_PER_PAGE``
        -- the pre-fvsjn shared budget -- and this short session comes back
        with 0 rows while all 13 long sessions hold 5. Measured red before
        this test was written.
        """
        short_id, session_ids = self._seed_skewed(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(
                archive,
                session_ids,
                ["message"],
                unit_windows={"message": WithUnitWindow(predicates=(), window=("last", 5))},
            )
        counts = {sid: len(attached.rows["message"].get(sid, ())) for sid in session_ids}
        assert counts[short_id] == 5, f"short session starved: {counts[short_id]} rows"
        assert set(counts.values()) == {5}, f"unequal windows: {sorted(set(counts.values()))}"
        assert attached.gaps == ()

    def test_every_session_on_a_wide_page_gets_the_same_allowance(self, tmp_path: Path) -> None:
        """A page wide enough to truncate still truncates every session equally.

        Anti-vacuity: the allowance is ``_MAX_ROWS_PER_PAGE // session_count``
        once that is below ``_MAX_ROWS_PER_SESSION``; return the per-session
        cap unconditionally and the counts stop depending on how many sessions
        share the page, which is the order-dependence this pins against.
        """
        _short_id, session_ids = self._seed_skewed(tmp_path)
        with ArchiveStore.open_existing(tmp_path) as archive:
            attached = fetch_attached_units(archive, session_ids, ["message"])
        counts = {sid: len(attached.rows["message"].get(sid, ())) for sid in session_ids}
        long_counts = {n for sid, n in counts.items() if sid != _short_id}
        assert len(long_counts) == 1, f"unequal allowances: {sorted(long_counts)}"
        allowance = next(iter(long_counts))
        assert counts[_short_id] == 5, "a session below the allowance keeps all its rows"
        assert attached.gaps == (f"attached_unit_truncated:message:{allowance}",)


class TestPageWiderThanTheBudgetIsRefused:
    """A page must leave room for each session's truncation probe.

    Under the 2026-09-21 ruling a computed answer is bounded by a typed
    refusal or not at all. Each session needs one result row and one probe row
    inside the ceiling, otherwise truncation could be hidden or the SQL fetch
    would exceed its declared page budget.
    """

    def test_a_page_with_no_row_per_session_raises(self) -> None:
        """Anti-vacuity: allow one result but no probe, and this stops raising
        even though it cannot prove whether every session was truncated."""
        from polylogue.archive.query.attached_units import (
            _MAX_ROWS_PER_PAGE,
            AttachedUnitPageTooWideError,
            _per_session_allowance,
        )

        with pytest.raises(AttachedUnitPageTooWideError) as refusal:
            _per_session_allowance(_MAX_ROWS_PER_PAGE // 2 + 1)
        assert refusal.value.session_count == _MAX_ROWS_PER_PAGE // 2 + 1
        assert refusal.value.code == "attached_unit_page_exceeds_row_budget"

    def test_probe_rows_fit_inside_the_page_ceiling(self) -> None:
        """A maximum-size request accounts for both rows and probes.

        Anti-vacuity: remove the reserved probe from the allowance arithmetic
        and the computed fetch exceeds the declared page ceiling.
        """
        from polylogue.archive.query.attached_units import (
            _MAX_ROWS_PER_PAGE,
            _per_session_allowance,
        )

        session_count = 1000
        allowance = _per_session_allowance(session_count)
        assert session_count * (allowance + 1) <= _MAX_ROWS_PER_PAGE
        assert allowance == 4

    def test_widest_servable_page_gets_one_row_and_one_probe(self) -> None:
        from polylogue.archive.query.attached_units import _MAX_ROWS_PER_PAGE, _per_session_allowance

        session_count = _MAX_ROWS_PER_PAGE // 2
        assert _per_session_allowance(session_count) == 1
        assert session_count * (_per_session_allowance(session_count) + 1) == _MAX_ROWS_PER_PAGE
