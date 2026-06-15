"""Unit tests for the query expression compiler (#1812).

Covers:
- Lexer AST output for key token forms
- Compiler field mapping (field → spec attribute)
- Flag ↔ expression equivalence (same spec from --origin x and origin:x)
- Rejected unknown fields (loud error)
- Relative + absolute date pass-through
- Exact id query
- Quoted command-looking phrases ("delete" is a phrase, not an action)
- Cross-field OR rejection
- Count comparisons (messages:>=N, words:>=N)
- in-field alternation origin:(a|b)
- has:paste / has:tools / has:thinking → boolean flags
- CLI bare-query path compiles expressions (RootModeRequest.query_spec)
- SessionQuerySpec.from_expression (Python facade entry point)
- Direct JSON spec input
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.query.expression import (
    EXPRESSION_FIELD_REGISTRY,
    ExpressionCompileError,
    _FieldToken,
    _lex,
    _TextToken,
    compile_expression,
    compile_expression_into,
)
from polylogue.archive.query.spec import SessionQuerySpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(**kwargs: Any) -> SessionQuerySpec:
    """Build a SessionQuerySpec with only the given fields set, rest default."""
    return SessionQuerySpec(**kwargs)


# ---------------------------------------------------------------------------
# Lexer tests
# ---------------------------------------------------------------------------


class TestLexer:
    def test_bare_words(self) -> None:
        tokens = _lex("json envelope")
        assert tokens == [
            _TextToken(text="json", quoted=False, negated=False),
            _TextToken(text="envelope", quoted=False, negated=False),
        ]

    def test_quoted_phrase(self) -> None:
        tokens = _lex('"json envelope"')
        assert tokens == [_TextToken(text="json envelope", quoted=True, negated=False)]

    def test_negated_quoted_phrase(self) -> None:
        tokens = _lex('-"bad phrase"')
        assert tokens == [_TextToken(text="bad phrase", quoted=True, negated=True)]

    def test_field_bare(self) -> None:
        tokens = _lex("repo:polylogue")
        assert tokens == [_FieldToken(field="repo", raw_value="polylogue", negated=False)]

    def test_field_negated(self) -> None:
        tokens = _lex("-origin:chatgpt-export")
        assert tokens == [_FieldToken(field="origin", raw_value="chatgpt-export", negated=True)]

    def test_field_quoted_value(self) -> None:
        tokens = _lex('near:"semantic search"')
        assert tokens == [_FieldToken(field="near", raw_value="semantic search", negated=False)]

    def test_field_paren_alternation(self) -> None:
        tokens = _lex("origin:(claude-code-session|codex-session)")
        assert tokens == [_FieldToken(field="origin", raw_value="claude-code-session|codex-session", negated=False)]

    def test_unclosed_quote_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unclosed quoted"):
            _lex('"not closed')

    def test_cross_field_or_paren_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="cross-field OR"):
            _lex("(origin:x OR origin:y)")

    def test_empty_expression(self) -> None:
        assert _lex("") == []

    def test_whitespace_only(self) -> None:
        assert _lex("   ") == []


# ---------------------------------------------------------------------------
# Compiler field-mapping tests
# ---------------------------------------------------------------------------


class TestCompilerFieldMapping:
    def test_repo(self) -> None:
        spec = compile_expression("repo:polylogue")
        assert spec.repo_names == ("polylogue",)

    def test_origin(self) -> None:
        spec = compile_expression("origin:claude-code-session")
        assert spec.origins == ("claude-code-session",)

    def test_origin_negated(self) -> None:
        spec = compile_expression("-origin:chatgpt-export")
        assert spec.excluded_origins == ("chatgpt-export",)

    def test_origin_alternation(self) -> None:
        spec = compile_expression("origin:(claude-code-session|codex-session)")
        assert set(spec.origins) == {"claude-code-session", "codex-session"}

    def test_tag(self) -> None:
        spec = compile_expression("tag:review")
        assert spec.tags == ("review",)

    def test_tag_negated(self) -> None:
        spec = compile_expression("-tag:wip")
        assert spec.excluded_tags == ("wip",)

    def test_path(self) -> None:
        spec = compile_expression("path:polylogue/cli")
        assert spec.referenced_path == ("polylogue/cli",)

    def test_cwd(self) -> None:
        spec = compile_expression("cwd:/realm/project")
        assert spec.cwd_prefix == "/realm/project"

    def test_tool(self) -> None:
        spec = compile_expression("tool:bash")
        assert spec.tool_terms == ("bash",)

    def test_tool_negated(self) -> None:
        spec = compile_expression("-tool:bash")
        assert spec.excluded_tool_terms == ("bash",)

    def test_action_file_edit(self) -> None:
        spec = compile_expression("action:file_edit")
        assert spec.action_terms == ("file_edit",)

    def test_action_negated(self) -> None:
        spec = compile_expression("-action:shell")
        assert spec.excluded_action_terms == ("shell",)

    def test_action_unknown_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown action"):
            compile_expression("action:unknown_action_type")

    def test_has_paste(self) -> None:
        spec = compile_expression("has:paste")
        assert spec.filter_has_paste is True
        assert spec.filter_has_tool_use is False

    def test_has_tools(self) -> None:
        spec = compile_expression("has:tools")
        assert spec.filter_has_tool_use is True

    def test_has_thinking(self) -> None:
        spec = compile_expression("has:thinking")
        assert spec.filter_has_thinking is True

    def test_has_custom_type(self) -> None:
        spec = compile_expression("has:summary")
        assert "summary" in spec.has_types

    def test_id(self) -> None:
        session_id = "abc123def456"
        spec = compile_expression(f"id:{session_id}")
        assert spec.session_id == session_id

    def test_title(self) -> None:
        spec = compile_expression("title:refactor")
        assert spec.title == "refactor"

    def test_near_quoted(self) -> None:
        spec = compile_expression('near:"semantic search test"')
        assert spec.similar_text == "semantic search test"
        # Free text must NOT bleed into the session-seed leg.
        assert spec.similar_session_id is None

    def test_near_bare_word_is_text_not_session(self) -> None:
        # Conservative rule: a bare word stays free-text similarity, never a
        # session reference.
        spec = compile_expression("near:refactor")
        assert spec.similar_text == "refactor"
        assert spec.similar_session_id is None

    def test_near_id_sets_similar_session_id(self) -> None:
        spec = compile_expression("near:id:abc123")
        assert spec.similar_session_id == "abc123"
        # The session-seed leg is distinct from free text.
        assert spec.similar_text is None

    def test_near_id_threads_into_plan(self) -> None:
        plan = compile_expression("near:id:abc123").to_plan()
        assert plan.similar_session_id == "abc123"
        assert plan.similar_text is None

    def test_near_id_empty_ref_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="requires a session reference"):
            compile_expression("near:id:")

    def test_near_id_merge_into_base(self) -> None:
        base = SessionQuerySpec()
        merged = compile_expression_into("near:id:abc123", base)
        assert merged.similar_session_id == "abc123"

    def test_contains(self) -> None:
        spec = compile_expression("contains:foo")
        assert spec.contains_terms == ("foo",)

    def test_lane(self) -> None:
        spec = compile_expression("lane:dialogue")
        assert spec.retrieval_lane == "dialogue"

    def test_lane_invalid_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown retrieval lane"):
            compile_expression("lane:nosuchlane")

    def test_bare_words_to_query_terms(self) -> None:
        spec = compile_expression("json envelope")
        assert spec.query_terms == ("json", "envelope")

    def test_negated_word_to_exclude_text(self) -> None:
        spec = compile_expression("-error")
        assert spec.exclude_text_terms == ("error",)

    def test_messages_gte(self) -> None:
        spec = compile_expression("messages:>=10")
        assert spec.min_messages == 10
        assert spec.max_messages is None

    def test_messages_lte(self) -> None:
        spec = compile_expression("messages:<=50")
        assert spec.max_messages == 50
        assert spec.min_messages is None

    def test_words_gte(self) -> None:
        spec = compile_expression("words:>=200")
        assert spec.min_words == 200

    def test_since_relative(self) -> None:
        spec = compile_expression("since:7d")
        assert spec.since is not None
        assert "days" in spec.since

    def test_until_relative(self) -> None:
        spec = compile_expression("until:2w")
        assert spec.until is not None
        assert "weeks" in spec.until

    def test_since_absolute(self) -> None:
        spec = compile_expression("since:2024-01-15")
        # Absolute dates pass through as-is (dateparser handles them later)
        assert spec.since == "2024-01-15"

    def test_empty_expression(self) -> None:
        spec = compile_expression("")
        assert spec == SessionQuerySpec()

    def test_combined_clauses(self) -> None:
        spec = compile_expression("repo:polylogue since:7d has:paste")
        assert spec.repo_names == ("polylogue",)
        assert spec.since is not None
        assert spec.filter_has_paste is True

    def test_complex_expression(self) -> None:
        spec = compile_expression("origin:(claude-code-session|codex-session) has:paste path:polylogue/cli tool:bash")
        assert set(spec.origins) == {"claude-code-session", "codex-session"}
        assert spec.filter_has_paste is True
        assert spec.referenced_path == ("polylogue/cli",)
        assert spec.tool_terms == ("bash",)


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------


class TestRejection:
    def test_unknown_field_raises_loudly(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown query field") as exc_info:
            compile_expression("nosuchfield:value")
        assert exc_info.value.field == "nosuchfield"

    def test_unknown_field_message_lists_recognized(self) -> None:
        with pytest.raises(ExpressionCompileError, match="recognized fields"):
            compile_expression("xyz:value")

    def test_cross_field_or_raises(self) -> None:
        # Top-level OR keyword triggers the cross-field OR error
        with pytest.raises(ExpressionCompileError, match="cross-field OR"):
            compile_expression("repo:polylogue OR repo:sinex")

    def test_cross_field_or_mentions_follow_up_issue(self) -> None:
        with pytest.raises(ExpressionCompileError, match="1812"):
            compile_expression("repo:polylogue OR origin:claude-code-session")

    def test_unknown_origin_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown origin"):
            compile_expression("origin:no-such-origin")

    def test_messages_without_op_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="comparison operator"):
            compile_expression("messages:10")


# ---------------------------------------------------------------------------
# Quoted command-looking phrase AC
# ---------------------------------------------------------------------------


class TestQuotedCommandPhrases:
    """Quoted command-looking terms must be FTS phrases, not action clauses."""

    def test_quoted_delete_is_phrase_not_action(self) -> None:
        spec = compile_expression('"delete"')
        # Must land in query_terms (FTS), NOT action_terms
        assert "delete" in spec.query_terms
        assert spec.action_terms == ()

    def test_quoted_file_edit_is_phrase(self) -> None:
        spec = compile_expression('"file_edit"')
        assert "file_edit" in spec.query_terms
        assert spec.action_terms == ()

    def test_bare_action_goes_to_action_terms(self) -> None:
        spec = compile_expression("action:file_edit")
        assert spec.action_terms == ("file_edit",)


# ---------------------------------------------------------------------------
# Flag ↔ expression equivalence
# ---------------------------------------------------------------------------


class TestFlagExpressionEquivalence:
    """Same spec from flags and expression clauses (AC from #1812)."""

    def test_origin_flag_vs_expression(self) -> None:
        flag_spec = SessionQuerySpec.from_params({"origin": "claude-code-session"})
        expr_spec = compile_expression("origin:claude-code-session")
        assert expr_spec.origins == flag_spec.origins

    def test_repo_flag_vs_expression(self) -> None:
        flag_spec = SessionQuerySpec.from_params({"repo": "polylogue"})
        expr_spec = compile_expression("repo:polylogue")
        assert expr_spec.repo_names == flag_spec.repo_names

    def test_tag_flag_vs_expression(self) -> None:
        flag_spec = SessionQuerySpec.from_params({"tag": "review"})
        expr_spec = compile_expression("tag:review")
        assert expr_spec.tags == flag_spec.tags

    def test_has_paste_flag_vs_expression(self) -> None:
        flag_spec = SessionQuerySpec.from_params({"filter_has_paste": True})
        expr_spec = compile_expression("has:paste")
        assert expr_spec.filter_has_paste == flag_spec.filter_has_paste

    def test_min_messages_flag_vs_expression(self) -> None:
        flag_spec = SessionQuerySpec.from_params({"min_messages": 10})
        expr_spec = compile_expression("messages:>=10")
        assert expr_spec.min_messages == flag_spec.min_messages


# ---------------------------------------------------------------------------
# compile_expression_into (merge)
# ---------------------------------------------------------------------------


class TestCompileExpressionInto:
    def test_merges_flag_origin_with_expression_repo(self) -> None:
        base = SessionQuerySpec.from_params({"origin": "claude-code-session"})
        merged = compile_expression_into("repo:polylogue", base)
        assert merged.origins == ("claude-code-session",)
        assert merged.repo_names == ("polylogue",)

    def test_expression_overrides_scalar_since(self) -> None:
        base = SessionQuerySpec(since="30d")
        merged = compile_expression_into("since:7d", base)
        # Expression's since takes precedence (both end up merged; last value wins)
        assert merged.since is not None
        assert "7" in (merged.since or "")

    def test_empty_expression_returns_base(self) -> None:
        base = SessionQuerySpec(repo_names=("polylogue",))
        merged = compile_expression_into("", base)
        assert merged.repo_names == ("polylogue",)


# ---------------------------------------------------------------------------
# Direct JSON spec input
# ---------------------------------------------------------------------------


class TestJsonSpecInput:
    def test_json_spec_roundtrip(self) -> None:
        raw = '{"repo": "polylogue", "retrieval_lane": "dialogue"}'
        spec = compile_expression(raw)
        assert spec.repo_names == ("polylogue",)
        assert spec.retrieval_lane == "dialogue"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="invalid JSON"):
            compile_expression("{bad json}")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="JSON object"):
            compile_expression("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Python facade: SessionQuerySpec.from_expression
# ---------------------------------------------------------------------------


class TestSessionQuerySpecFromExpression:
    def test_basic_expression(self) -> None:
        spec = SessionQuerySpec.from_expression("repo:polylogue has:paste")
        assert spec.repo_names == ("polylogue",)
        assert spec.filter_has_paste is True

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ExpressionCompileError):
            SessionQuerySpec.from_expression("notafield:value")

    def test_empty_returns_default_spec(self) -> None:
        assert SessionQuerySpec.from_expression("") == SessionQuerySpec()


# ---------------------------------------------------------------------------
# CLI wiring: RootModeRequest.query_spec
# ---------------------------------------------------------------------------


class TestCLIRootRequestWiring:
    """Test that RootModeRequest.query_spec compiles DSL expressions."""

    def test_bare_words_preserved_as_fts(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("json", "envelope"))
        spec = request.query_spec()
        assert spec.query_terms == ("json", "envelope")

    def test_dsl_field_clause_compiled(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("repo:polylogue",))
        spec = request.query_spec()
        assert spec.repo_names == ("polylogue",)
        # Must NOT go to query_terms (FTS)
        assert "repo:polylogue" not in spec.query_terms

    def test_dsl_mixed_clauses(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(
            params={},
            query_terms=("repo:polylogue", "since:7d", "json"),
        )
        spec = request.query_spec()
        assert spec.repo_names == ("polylogue",)
        assert spec.since is not None
        assert "json" in spec.query_terms

    def test_flags_and_expression_merged(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        # Simulate: polylogue --tag mywork repo:polylogue
        request = RootModeRequest(params={"tag": "mywork"}, query_terms=("repo:polylogue",))
        spec = request.query_spec()
        assert spec.tags == ("mywork",)
        assert spec.repo_names == ("polylogue",)

    def test_term_with_space_quoted_in_expression(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        # "json envelope" was shell-quoted → arrives as one term with a space
        request = RootModeRequest(params={}, query_terms=("json envelope",))
        spec = request.query_spec()
        # Should end up in query_terms as the phrase "json envelope"
        assert any("json envelope" in t for t in spec.query_terms)

    def test_unknown_field_raises(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("nosuchfield:value",))
        with pytest.raises(ExpressionCompileError, match="unknown query field"):
            request.query_spec()


# ---------------------------------------------------------------------------
# Field registry completeness
# ---------------------------------------------------------------------------


class TestFieldRegistry:
    def test_registry_has_required_fields(self) -> None:
        required = {"repo", "origin", "tag", "path", "cwd", "tool", "action", "has", "id", "since", "until", "near"}
        assert required.issubset(EXPRESSION_FIELD_REGISTRY.keys())

    def test_all_registry_entries_have_description(self) -> None:
        for field, info in EXPRESSION_FIELD_REGISTRY.items():
            assert "description" in info, f"{field} missing description"
            assert info["description"], f"{field} has empty description"

    def test_origin_description_has_no_provider_source_wording(self) -> None:
        """Regression: origin field description must not use 'provider source' (#1861)."""
        desc = EXPRESSION_FIELD_REGISTRY["origin"]["description"]
        assert "provider source" not in desc, "origin description must not use 'provider source' wording; got: " + repr(
            desc
        )

    def test_cross_field_or_error_cites_1812(self) -> None:
        """Regression: cross-field OR error must cite issue #1812, not #1858 (#1861)."""
        with pytest.raises(ExpressionCompileError, match="1812"):
            compile_expression("repo:x OR origin:y")


# ---------------------------------------------------------------------------
# MCP surface wiring (#1860)
# ---------------------------------------------------------------------------


class TestMCPWiring:
    """build_query_spec routes free-text query through the shared compiler."""

    def test_bare_words_preserved_as_fts(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        spec = build_query_spec(query="json envelope")
        # Bare words must end up in query_terms (FTS), not in structured fields.
        assert spec.query_terms == ("json", "envelope")

    def test_dsl_origin_clause_compiled(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        spec = build_query_spec(query="origin:claude-code-session")
        assert spec.origins == ("claude-code-session",)
        # Must NOT go to FTS query_terms.
        assert not any("origin" in t for t in spec.query_terms)

    def test_dsl_has_paste_compiled(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        spec = build_query_spec(query="has:paste")
        assert spec.filter_has_paste is True
        assert spec.query_terms == ()

    def test_dsl_flags_merged_with_base_params(self) -> None:
        """DSL expression merges additively with flag-derived base spec."""
        from polylogue.mcp.query_contracts import build_query_spec

        # Simulate: MCP caller passes both a structured 'tag' param and a DSL
        # expression in 'query'.
        spec = build_query_spec(query="origin:codex-session", tag="review")
        assert spec.origins == ("codex-session",)
        assert spec.tags == ("review",)

    def test_unknown_field_raises(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        with pytest.raises(ExpressionCompileError, match="unknown query field"):
            build_query_spec(query="nosuchfield:value")

    def test_cross_field_or_raises(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        with pytest.raises(ExpressionCompileError, match="cross-field OR"):
            build_query_spec(query="repo:x OR origin:y")

    def test_none_query_returns_base_spec(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        spec = build_query_spec(query=None, tag="review")
        # With no query expression, must still apply the flag-derived filters.
        assert spec.tags == ("review",)
        assert spec.query_terms == ()


# ---------------------------------------------------------------------------
# Cross-surface parity (#1860 / #1812)
# ---------------------------------------------------------------------------


class TestCrossSurfaceParity:
    """Same DSL expression → same SessionQuerySpec on all three surfaces.

    CLI path:   RootModeRequest.query_spec()
    MCP path:   build_query_spec(query=...)
    Daemon path: compile_expression_into(query_str, base_spec)  (same function)

    Because all three surfaces ultimately call compile_expression_into, these
    tests verify that the compiler is wired symmetrically and that no surface
    silently re-parses the expression through its own ad-hoc logic.
    """

    _EXPRESSION = "origin:claude-code-session has:paste repo:polylogue"

    def _cli_spec(self) -> SessionQuerySpec:
        from polylogue.cli.root_request import RootModeRequest

        # CLI receives the expression as individual shell-split tokens.
        request = RootModeRequest(
            params={},
            query_terms=tuple(self._EXPRESSION.split()),
        )
        return request.query_spec()

    def _mcp_spec(self) -> SessionQuerySpec:
        from polylogue.mcp.query_contracts import build_query_spec

        return build_query_spec(query=self._EXPRESSION)

    def _daemon_spec(self) -> SessionQuerySpec:
        # The daemon path: compile_expression_into(query_str, base_spec).
        # Reproduce the same logic as _do_list with no extra flag-based params.
        base = SessionQuerySpec()
        return compile_expression_into(self._EXPRESSION, base)

    def test_cli_and_mcp_origins_match(self) -> None:
        assert self._cli_spec().origins == self._mcp_spec().origins

    def test_cli_and_daemon_origins_match(self) -> None:
        assert self._cli_spec().origins == self._daemon_spec().origins

    def test_cli_and_mcp_has_paste_match(self) -> None:
        assert self._cli_spec().filter_has_paste == self._mcp_spec().filter_has_paste

    def test_cli_and_daemon_has_paste_match(self) -> None:
        assert self._cli_spec().filter_has_paste == self._daemon_spec().filter_has_paste

    def test_cli_and_mcp_repo_names_match(self) -> None:
        assert self._cli_spec().repo_names == self._mcp_spec().repo_names

    def test_cli_and_daemon_repo_names_match(self) -> None:
        assert self._cli_spec().repo_names == self._daemon_spec().repo_names

    def test_all_three_query_terms_are_empty(self) -> None:
        """Pure DSL expression (no bare words) → query_terms empty on all surfaces."""
        cli_spec = self._cli_spec()
        mcp_spec = self._mcp_spec()
        daemon_spec = self._daemon_spec()
        assert cli_spec.query_terms == ()
        assert mcp_spec.query_terms == ()
        assert daemon_spec.query_terms == ()


# ---------------------------------------------------------------------------
# Bug 3 regression: words:<=N / words:=N (#1873)
# ---------------------------------------------------------------------------


class TestWordsCountRegressions:
    """words:<=N and words:=N must set max_words, not silently map to min_words."""

    def test_words_lte_sets_max_words(self) -> None:
        """Bug 3a: words:<=N must set max_words (was incorrectly setting min_words)."""
        spec = compile_expression("words:<=500")
        assert spec.max_words == 500
        assert spec.min_words is None

    def test_words_eq_sets_both_bounds(self) -> None:
        """Bug 3b: words:=N must set both min_words and max_words."""
        spec = compile_expression("words:=100")
        assert spec.min_words == 100
        assert spec.max_words == 100

    def test_words_gte_still_sets_min_words_only(self) -> None:
        """words:>=N must still set min_words only (existing behavior preserved)."""
        spec = compile_expression("words:>=200")
        assert spec.min_words == 200
        assert spec.max_words is None

    def test_words_lte_and_gte_together(self) -> None:
        """words:>=N words:<=M → both bounds set independently."""
        spec = compile_expression("words:>=100 words:<=500")
        assert spec.min_words == 100
        assert spec.max_words == 500

    def test_words_lte_did_not_set_wrong_field(self) -> None:
        """Regression: words:<=500 must NOT set min_words (the pre-fix bug)."""
        spec = compile_expression("words:<=500")
        # Before the fix this would erroneously set min_words=500 and max_words=None.
        assert spec.min_words is None, "words:<=500 set min_words instead of max_words — the Bug 3 regression is back"


# ---------------------------------------------------------------------------
# Bug 4 regression: negation on unsupported fields (#1873)
# ---------------------------------------------------------------------------


class TestNegationRejections:
    """Fields without an exclude axis must raise ExpressionCompileError when negated."""

    @pytest.mark.parametrize(
        "expr",
        [
            "-repo:polylogue",
            "-path:some/file",
            "-cwd:/realm/project",
            "-has:paste",
            "-id:abc123",
            "-title:refactor",
            "-since:7d",
            "-until:2024-01-15",
            '-near:"semantic search"',
            "-contains:foo",
            "-lane:dialogue",
        ],
    )
    def test_negated_unsupported_field_raises(self, expr: str) -> None:
        """Bug 4: negating a field without an exclude axis must raise loudly."""
        with pytest.raises(ExpressionCompileError, match="negation is not supported"):
            compile_expression(expr)

    def test_negated_origin_still_works(self) -> None:
        """origin: supports negation via excluded_origins — must continue to work."""
        spec = compile_expression("-origin:claude-code-session")
        assert "claude-code-session" in spec.excluded_origins
        assert "claude-code-session" not in spec.origins

    def test_negated_tag_still_works(self) -> None:
        """tag: supports negation — must continue to work."""
        spec = compile_expression("-tag:review")
        assert "review" in spec.excluded_tags
        assert "review" not in spec.tags

    def test_negated_tool_still_works(self) -> None:
        """tool: supports negation — must continue to work."""
        spec = compile_expression("-tool:bash")
        assert "bash" in spec.excluded_tool_terms
        assert "bash" not in spec.tool_terms

    def test_negated_action_still_works(self) -> None:
        """action: supports negation — must continue to work."""
        spec = compile_expression("-action:file_edit")
        assert "file_edit" in spec.excluded_action_terms
        assert "file_edit" not in spec.action_terms


# ---------------------------------------------------------------------------
# Bug 5 regression: escaped quotes in phrase lexer (#1873)
# ---------------------------------------------------------------------------


class TestEscapedQuotePhrases:
    """Quoted phrases containing \" escapes must be parsed correctly."""

    def test_escaped_quote_inside_phrase(self) -> None:
        """Bug 5: \\\" inside a quoted phrase must produce a literal quote in the term."""
        # The expression: "say \"hello\""
        # root_request.py wraps terms with spaces using \" escapes, so the phrase
        # lexer must consume them without treating the \" as a close-quote.
        spec = compile_expression(r'"say \"hello\""')
        assert spec.query_terms == ('say "hello"',)

    def test_escaped_quote_at_start_of_phrase(self) -> None:
        """A phrase starting with \\\" must parse correctly."""
        spec = compile_expression(r'"\"quoted word\""')
        assert spec.query_terms == ('"quoted word"',)

    def test_roundtrip_via_root_request(self) -> None:
        """Terms with spaces are re-quoted by RootModeRequest.query_spec — must round-trip."""
        from polylogue.cli.root_request import RootModeRequest

        # A term with an embedded space is passed as one element (shell-quoted).
        # RootModeRequest wraps it in "..." and escapes internal quotes.
        request = RootModeRequest(params={}, query_terms=('say "hello"',))
        spec = request.query_spec()
        # The round-tripped phrase must be in query_terms (FTS), intact.
        assert any('"hello"' in t for t in spec.query_terms), (
            "Escaped-quote round-trip failed: term not found in spec.query_terms"
        )


# ---------------------------------------------------------------------------
# Bug 6 regression: JSON spec strict mode (#1873)
# ---------------------------------------------------------------------------


class TestJsonSpecStrictMode:
    """JSON spec input must reject unknown keys (strict=True)."""

    def test_unknown_key_in_json_spec_raises(self) -> None:
        """Bug 6: unknown keys in a JSON spec must raise ExpressionCompileError."""
        with pytest.raises(ExpressionCompileError, match="invalid spec fields"):
            compile_expression('{"repo": "polylogue", "unknown_key": "value"}')

    def test_known_keys_in_json_spec_succeed(self) -> None:
        """Valid JSON spec must still compile correctly."""
        spec = compile_expression('{"repo": "polylogue"}')
        assert spec.repo_names == ("polylogue",)

    def test_json_spec_strict_rejects_typo_key(self) -> None:
        """A common typo like 'repos' (instead of 'repo') must be rejected."""
        with pytest.raises(ExpressionCompileError, match="invalid spec fields"):
            compile_expression('{"repos": ["polylogue"]}')


# ---------------------------------------------------------------------------
# Bug 7 regression: ?contains= not routed through DSL compiler (#1873)
# ---------------------------------------------------------------------------


class TestDaemonContainsParamNotCompiled:
    """_do_archive_list_sessions must not route ?contains= through compile_expression."""

    def test_contains_param_not_compiled_as_dsl(self, workspace_env: dict[str, Path]) -> None:
        """Bug 7: a ?contains= value that the DSL compiler would reject must still
        be treated as a literal FTS filter, not compiled.

        ``action:badaction`` raises ExpressionCompileError("unknown action") if
        routed through compile_expression; as a literal content filter it is
        normalized to a MATCH-safe FTS query and simply returns no matches. The
        old code did ``query or contains`` and compiled the contains value.
        """
        from polylogue.daemon.http import DaemonAPIHandler
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "s1")
            .provider("claude-code")
            .title("s1")
            .add_message("m1", role="user", text="ordinary content")
            .save()
        )
        handler = DaemonAPIHandler.__new__(DaemonAPIHandler)

        # Would raise ExpressionCompileError if compiled; must not here.
        payload = handler._do_archive_list_sessions(
            workspace_env["archive_root"], {"contains": ["action:badaction"]}, 50, 0
        )
        assert isinstance(payload, dict)
        assert payload["total"] == 0


# ---------------------------------------------------------------------------
# Bug 8 regression: id:xyz filter applied in /api/archive/sessions (#1873)
# ---------------------------------------------------------------------------


class TestDaemonSessionIdFilter:
    """Behavioral coverage of /api/sessions id: scoping and contains filtering.

    Replaces the earlier source-grep regression (#1873 Bug 7/8) with real calls
    into ``_do_archive_list_sessions`` over a seeded archive.
    """

    def _handler(self) -> Any:
        from polylogue.daemon.http import DaemonAPIHandler

        return DaemonAPIHandler.__new__(DaemonAPIHandler)

    def _seed(self, index_db: Path, specs: list[tuple[str, str]]) -> list[str]:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        for i, (sid, text) in enumerate(specs):
            (
                SessionBuilder(index_db, sid)
                .provider("claude-code")
                .title(sid)
                .add_message(f"m{i}", role="user", text=text)
                .save()
            )
        with ArchiveStore.open_existing(index_db.parent) as archive:
            return [str(s.session_id) for s in archive.list_summaries(limit=1000)]

    def test_id_clause_produces_session_id_in_spec(self) -> None:
        """compile_expression('id:abc') must produce spec.session_id."""
        spec = compile_expression("id:abc")
        assert spec.session_id == "abc"

    def test_id_query_scopes_results_and_total(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        ids = self._seed(index_db, [("alpha", "alpha body"), ("beta", "beta body")])
        assert len(ids) == 2

        payload = self._handler()._do_archive_list_sessions(
            workspace_env["archive_root"], {"query": [f"id:{ids[0]}"]}, 50, 0
        )
        assert isinstance(payload, dict)
        # total must be scoped to the id match, NOT the archive-wide count of 2.
        assert payload["total"] == 1
        items = payload["items"]
        assert isinstance(items, list) and len(items) == 1

    def test_id_miss_returns_typed_empty_not_500(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db, [("alpha", "alpha body")])

        payload = self._handler()._do_archive_list_sessions(
            workspace_env["archive_root"], {"query": ["id:nonexistentnope"]}, 50, 0
        )
        # A missing id is a typed-empty page, not a 500 propagated from resolve.
        assert payload == {"items": [], "total": 0, "limit": 50, "offset": 0}

    def test_id_ambiguous_prefix_raises_query_spec_error(self, workspace_env: dict[str, Path]) -> None:
        import os

        from polylogue.archive.query.spec import QuerySpecError

        index_db = workspace_env["archive_root"] / "index.db"
        ids = self._seed(index_db, [("aaa", "x body"), ("aab", "y body")])
        prefix = os.path.commonprefix(ids)
        assert prefix and prefix not in ids, "need a shared, non-exact prefix"

        with pytest.raises(QuerySpecError):
            self._handler()._do_archive_list_sessions(workspace_env["archive_root"], {"query": [f"id:{prefix}"]}, 50, 0)

    def test_contains_filters_without_query(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db, [("one", "has findmetoken here"), ("two", "unrelated content")])

        # ?contains=foo with no ?query= must still filter (Bug 7): it routes to the
        # FTS branch as a literal term rather than returning the unfiltered page.
        payload = self._handler()._do_archive_list_sessions(
            workspace_env["archive_root"], {"contains": ["findmetoken"]}, 50, 0
        )
        assert isinstance(payload, dict)
        hits = payload.get("hits")
        assert isinstance(hits, list) and len(hits) == 1
