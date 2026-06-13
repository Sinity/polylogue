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
        with pytest.raises(ExpressionCompileError, match="1858"):
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
