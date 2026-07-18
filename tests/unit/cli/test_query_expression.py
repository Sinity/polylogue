"""Unit tests for the shared Lark query DSL and lowerer (#2006).

Covers:
- Lexer AST output for key token forms
- Lowerer field mapping (field → spec attribute)
- Flag ↔ expression equivalence (same spec from --origin x and origin:x)
- Rejected unknown fields (loud error)
- Relative + absolute date pass-through
- Exact id query
- Quoted command-looking phrases ("delete" is a phrase, not an action)
- Cross-field OR Boolean predicate lowering
- Count comparisons (messages:>=N, words:>=N)
- in-field alternation origin:(a|b)
- has:paste / has:tools / has:thinking → boolean flags
- CLI bare-query path compiles expressions (RootModeRequest.query_spec)
- SessionQuerySpec.from_expression (Python facade entry point)
- Direct JSON spec input
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from polylogue.archive.query.expression import (
    EXPRESSION_FIELD_REGISTRY,
    ExpressionCompileError,
    QueryUnitPipeline,
    QueryUnitSessionScopeStage,
    QueryUnitSource,
    RefOperand,
    RefOperandCycleError,
    ResolvedRefOperand,
    UnsupportedSessionTerminalActionError,
    _CountRangeToken,
    _CountToken,
    _DateComparisonToken,
    _DateRangeToken,
    _FieldToken,
    _TextToken,
    build_session_terminal_pipeline,
    compile_expression,
    compile_expression_into,
    explain_expression,
    parse_expression_ast,
    parse_reference_query_pipeline,
    parse_unit_source_expression,
    resolve_ref_operand,
)
from polylogue.archive.query.metadata import query_unit_descriptors
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryLineagePredicate,
    QueryNotPredicate,
    QuerySemanticPredicate,
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.refs import ObjectRef
from polylogue.storage.runtime import MessageRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _materialize_run_projection(index_db: Path) -> None:
    """Rebuild session insights (profile/work-event/phase rows).

    Run/observed-event/context-snapshot query units are source-derived
    (polylogue-dab) and need no materialization; this only exercises that a
    rebuild alongside a source-derived query doesn't interfere with it.
    """
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(index_db) as conn:
        rebuild_session_insights_sync(conn)


def _assert_run_projection_table_absent(index_db: Path, table_name: str) -> None:
    """Assert a run-projection materialized table was dropped (polylogue-dab).

    Stronger than the old "materialized count stays 0" check this replaces:
    the table no longer exists at all, so there is no write path to assert
    against in the first place.
    """
    with sqlite3.connect(index_db) as conn:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
    assert row is None, f"{table_name} should not exist (polylogue-dab dropped it)"


def _spec(**kwargs: Any) -> SessionQuerySpec:
    """Build a SessionQuerySpec with only the given fields set, rest default."""
    return SessionQuerySpec(**kwargs)


def test_build_session_terminal_pipeline_payload() -> None:
    pipeline = build_session_terminal_pipeline(
        "read",
        args=(("view", "messages"), ("format", "markdown")),
    )

    assert pipeline.to_payload() == {
        "source": {"unit": "sessions"},
        "stages": [
            {
                "kind": "terminal",
                "action": "read",
                "args": {"view": "messages", "format": "markdown"},
            }
        ],
    }


def test_build_session_terminal_pipeline_rejects_unknown_action() -> None:
    with pytest.raises(UnsupportedSessionTerminalActionError):
        build_session_terminal_pipeline("recover")


def _clauses(expression: str) -> list[object]:
    """Return parsed flat-query clauses through the canonical AST entry point."""
    return list(parse_expression_ast(expression).clauses)


def _field_payload(
    field: str,
    *values: str,
    op: str = "=",
    field_ref: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"kind": "field", "field": field, "op": op, "values": list(values)}
    if field_ref is not None:
        payload["field_ref"] = field_ref
    return payload


def _session_field_ref(name: str, *, source_name: str | None = None, unit: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "scope": "session",
        "name": name,
        "source_name": source_name or name,
    }
    if unit is not None:
        payload["unit"] = unit
    return payload


def _unit_field_ref(name: str, unit: str, *, source_name: str | None = None) -> dict[str, object]:
    return {
        "scope": "unit",
        "name": name,
        "source_name": source_name or name,
        "unit": unit,
    }


def _session_scope_stage(field: str, *values: str) -> dict[str, object]:
    return {
        "kind": "session_scope",
        "predicate": _field_payload(field, *values, field_ref=_session_field_ref(field)),
    }


# ---------------------------------------------------------------------------
# Lexer tests
# ---------------------------------------------------------------------------


class TestLexer:
    def test_parse_expression_ast_exposes_clauses(self) -> None:
        ast = parse_expression_ast('repo:polylogue "json envelope" messages:>=10')

        assert ast.clauses == (
            _FieldToken(field="repo", raw_value="polylogue", negated=False),
            _TextToken(text="json envelope", quoted=True, negated=False),
            _CountToken(field="messages", op=">=", number=10),
        )

    def test_parse_expression_ast_exposes_readable_count_clauses(self) -> None:
        ast = parse_expression_ast("messages > 10 words between 100 and 500 tool_use_messages:>=1")

        assert ast.clauses == (
            _CountToken(field="messages", op=">", number=10),
            _CountRangeToken(field="words", min_number=100, max_number=500),
            _CountToken(field="tool_use_messages", op=">=", number=1),
        )

    def test_bare_words(self) -> None:
        tokens = _clauses("json envelope")
        assert tokens == [
            _TextToken(text="json", quoted=False, negated=False),
            _TextToken(text="envelope", quoted=False, negated=False),
        ]

    def test_bare_boolean_words_remain_fts_terms(self) -> None:
        tokens = _clauses("error or timeout not retry")
        assert tokens == [
            _TextToken(text="error", quoted=False, negated=False),
            _TextToken(text="or", quoted=False, negated=False),
            _TextToken(text="timeout", quoted=False, negated=False),
            _TextToken(text="not", quoted=False, negated=False),
            _TextToken(text="retry", quoted=False, negated=False),
        ]

    def test_code_like_bare_words(self) -> None:
        tokens = _clauses("foo(bar) array[0] dict{key}")
        assert tokens == [
            _TextToken(text="foo(bar)", quoted=False, negated=False),
            _TextToken(text="array[0]", quoted=False, negated=False),
            _TextToken(text="dict{key}", quoted=False, negated=False),
        ]

    def test_quoted_phrase(self) -> None:
        tokens = _clauses('"json envelope"')
        assert tokens == [_TextToken(text="json envelope", quoted=True, negated=False)]

    def test_negated_quoted_phrase(self) -> None:
        tokens = _clauses('-"bad phrase"')
        assert tokens == [_TextToken(text="bad phrase", quoted=True, negated=True)]

    def test_field_bare(self) -> None:
        tokens = _clauses("repo:polylogue")
        assert tokens == [_FieldToken(field="repo", raw_value="polylogue", negated=False)]

    def test_field_negated(self) -> None:
        tokens = _clauses("-origin:chatgpt-export")
        assert tokens == [_FieldToken(field="origin", raw_value="chatgpt-export", negated=True)]

    def test_field_quoted_value(self) -> None:
        tokens = _clauses('near:"semantic search"')
        assert tokens == [_FieldToken(field="near", raw_value="semantic search", negated=False)]

    def test_field_paren_alternation(self) -> None:
        tokens = _clauses("origin:(claude-code-session|codex-session)")
        assert tokens == [_FieldToken(field="origin", raw_value="claude-code-session|codex-session", negated=False)]

    def test_unclosed_quote_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unclosed quoted"):
            parse_expression_ast('"not closed')

    def test_cross_field_or_paren_builds_boolean_ast(self) -> None:
        ast = parse_expression_ast("(origin:claude-code-session OR origin:chatgpt-export)")
        assert isinstance(ast.boolean_predicate, QueryBoolPredicate)

    def test_parse_expression_ast_preserves_escaped_quotes(self) -> None:
        ast = parse_expression_ast(r'near:"say \"hello\"" -"bad \"phrase\""')

        assert ast.clauses == (
            _FieldToken(field="near", raw_value='say "hello"', negated=False),
            _TextToken(text='bad "phrase"', quoted=True, negated=True),
        )

    def test_empty_expression(self) -> None:
        assert _clauses("") == []

    def test_whitespace_only(self) -> None:
        assert _clauses("   ") == []


class TestExplainExpression:
    def test_explain_expression_reports_ast_lowerer_and_plan(self) -> None:
        explanation = explain_expression('repo:polylogue has:paste "json envelope"')

        assert explanation.source_text == 'repo:polylogue has:paste "json envelope"'
        assert explanation.lowerer == "lark-query-expression-to-session-query-spec"
        assert explanation.unsupported_nodes == ()
        assert explanation.selected_units == ("session",)
        assert explanation.execution_legs == ("fts", "sql")
        assert explanation.lowered_spec.repo_names == ("polylogue",)
        assert explanation.lowered_spec.filter_has_paste is True
        assert explanation.clauses[0].to_payload() == {
            "kind": "field",
            "field": "repo",
            "value": "polylogue",
        }
        assert explanation.clauses[2].to_payload() == {
            "kind": "text",
            "value": "json envelope",
            "quoted": True,
        }
        assert "repo: polylogue" in explanation.plan_description

    def test_explain_expression_reports_json_spec_mode(self) -> None:
        explanation = explain_expression('{"repo": "polylogue", "limit": 5}')

        assert explanation.lowerer == "json-spec"
        assert explanation.lowered_spec.repo_names == ("polylogue",)
        assert explanation.lowered_spec.limit == 5
        assert explanation.clauses[0].kind == "json"
        assert explanation.selected_units == ("session",)
        assert explanation.execution_legs == ("sql",)
        assert explanation.to_payload()["unsupported_nodes"] == []

    def test_explain_expression_reports_boolean_units_and_legs(self) -> None:
        explanation = explain_expression("sessions where exists block(type:code) AND lineage:id:root")

        assert explanation.selected_units == ("block", "lineage", "session")
        assert explanation.execution_legs == ("exists-block", "lineage-recursive-cte", "sql")
        payload = explanation.to_payload()
        assert payload["selected_units"] == ["block", "lineage", "session"]
        assert payload["execution_legs"] == ["exists-block", "lineage-recursive-cte", "sql"]
        assert payload["ast"] == {
            "entry": "boolean",
            "predicate": {
                "children": [
                    {
                        "child": _field_payload("type", "code", field_ref=_unit_field_ref("type", "block")),
                        "kind": "exists",
                        "unit": "block",
                    },
                    {"kind": "lineage", "seed_session_id": "root", "unit": "session"},
                ],
                "kind": "and",
            },
        }
        assert payload["lowering_plan"] == {
            "lowerer": "lark-query-expression-to-session-query-spec",
            "selected_units": ["block", "lineage", "session"],
            "execution_legs": ["exists-block", "lineage-recursive-cte", "sql"],
            "plan_description": explanation.to_payload()["plan_description"],
        }

    def test_explain_expression_reports_readable_count_range_clause(self) -> None:
        explanation = explain_expression("messages between 5 and 20")

        assert explanation.lowered_spec.min_messages == 5
        assert explanation.lowered_spec.max_messages == 20
        assert explanation.clauses[0].to_payload() == {
            "kind": "count_range",
            "field": "messages",
            "min_number": 5,
            "max_number": 20,
        }

    def test_explain_expression_reports_semantic_and_sequence_legs(self) -> None:
        semantic = explain_expression('sessions where semantic:"query compiler" AND title:hit')
        sequence = explain_expression("sessions where seq(action:file_edit -> action:shell AND output:failed)")

        assert semantic.execution_legs == ("sql", "vector")
        assert sequence.selected_units == ("action", "session")
        assert sequence.execution_legs == ("sequence-action",)
        assert sequence.predicate is not None
        assert sequence.predicate.to_payload() == {
            "kind": "sequence",
            "unit": "action",
            "steps": [
                _field_payload("action", "file_edit", field_ref=_unit_field_ref("action", "action")),
                {
                    "kind": "and",
                    "children": [
                        _field_payload("action", "shell", field_ref=_unit_field_ref("action", "action")),
                        _field_payload("output", "failed", field_ref=_unit_field_ref("output", "action")),
                    ],
                },
            ],
        }

    def test_explain_expression_reports_terminal_unit_source(self) -> None:
        explanation = explain_expression("messages where role:assistant AND text:timeout")

        assert explanation.lowerer == "lark-query-unit-source-to-terminal-unit"
        assert explanation.selected_units == ("message",)
        assert explanation.execution_legs == ("sql", "terminal-message-rows")
        assert explanation.plan_description == (
            "terminal unit source: message",
            "compatibility session selector: exists message(...)",
        )
        payload = explanation.to_payload()
        assert payload["ast"] == {
            "entry": "unit_source",
            "predicate": {
                "children": [
                    _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
                    _field_payload("text", "timeout", field_ref=_unit_field_ref("text", "message")),
                ],
                "kind": "and",
            },
            "unit_source": {
                "unit": "message",
                "predicate": {
                    "children": [
                        _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
                        _field_payload("text", "timeout", field_ref=_unit_field_ref("text", "message")),
                    ],
                    "kind": "and",
                },
                "pipeline": {
                    "source": {
                        "unit": "message",
                        "predicate": {
                            "children": [
                                _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
                                _field_payload("text", "timeout", field_ref=_unit_field_ref("text", "message")),
                            ],
                            "kind": "and",
                        },
                    },
                    "stages": [{"kind": "terminal", "action": "rows"}],
                },
            },
        }
        assert payload["lowering_plan"] == {
            "lowerer": "lark-query-unit-source-to-terminal-unit",
            "selected_units": ["message"],
            "execution_legs": ["sql", "terminal-message-rows"],
            "plan_description": [
                "terminal unit source: message",
                "compatibility session selector: exists message(...)",
            ],
            "compatibility_selector": "exists message(...)",
            "pipeline": cast(dict[str, Any], payload["ast"])["unit_source"]["pipeline"],
        }
        assert explanation.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="role", values=("assistant",)),
                QueryFieldPredicate(field="text", values=("timeout",)),
            ),
        )

    def test_explain_expression_reports_terminal_pipeline_stages(self) -> None:
        explanation = explain_expression("messages where role:assistant | sort by time desc | limit 2 | offset 3")

        assert explanation.selected_units == ("message",)
        payload = explanation.to_payload()
        assert payload["ast"] == {
            "entry": "unit_source",
            "predicate": _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
            "unit_source": {
                "unit": "message",
                "predicate": _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
                "limit": 2,
                "offset": 3,
                "sort": {"field": "time", "direction": "desc"},
                "pipeline_stages": [
                    {"kind": "sort", "sort": {"field": "time", "direction": "desc"}},
                    {"kind": "limit", "value": 2},
                    {"kind": "offset", "value": 3},
                ],
                "pipeline": {
                    "source": {
                        "unit": "message",
                        "predicate": _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
                    },
                    "stages": [
                        {"kind": "sort", "sort": {"field": "time", "direction": "desc"}},
                        {"kind": "limit", "value": 2},
                        {"kind": "offset", "value": 3},
                        {"kind": "terminal", "action": "rows"},
                    ],
                    "result": {
                        "sort": {"field": "time", "direction": "desc"},
                        "limit": 2,
                        "offset": 3,
                    },
                },
            },
        }
        lowering_plan = cast(dict[str, Any], payload["lowering_plan"])
        assert lowering_plan["pipeline"] == cast(dict[str, Any], payload["ast"])["unit_source"]["pipeline"]
        assert lowering_plan["pipeline_stages"] == [
            {"kind": "sort", "sort": {"field": "time", "direction": "desc"}},
            {"kind": "limit", "value": 2},
            {"kind": "offset", "value": 3},
        ]

    def test_explain_expression_reports_session_scoped_pipeline_stage(self) -> None:
        explanation = explain_expression("sessions where repo:polylogue | messages where role:assistant | limit 5")

        payload = explanation.to_payload()
        ast_payload = cast(dict[str, Any], payload["ast"])
        unit_source = cast(dict[str, Any], ast_payload["unit_source"])
        assert unit_source["session_predicate"] == _field_payload(
            "repo", "polylogue", field_ref=_session_field_ref("repo")
        )
        assert unit_source["pipeline_stages"] == [
            _session_scope_stage("repo", "polylogue"),
            {"kind": "limit", "value": 5},
        ]
        assert unit_source["pipeline"] == {
            "source": {
                "unit": "message",
                "predicate": {
                    "children": [
                        _field_payload(
                            "session.repo",
                            "polylogue",
                            field_ref=_session_field_ref("repo", source_name="session.repo", unit="message"),
                        ),
                        _field_payload("role", "assistant", field_ref=_unit_field_ref("role", "message")),
                    ],
                    "kind": "and",
                },
            },
            "session_scope": _field_payload("repo", "polylogue", field_ref=_session_field_ref("repo")),
            "stages": [
                _session_scope_stage("repo", "polylogue"),
                {"kind": "limit", "value": 5},
                {"kind": "terminal", "action": "rows"},
            ],
            "result": {"limit": 5},
        }
        lowering_plan = cast(dict[str, Any], payload["lowering_plan"])
        assert lowering_plan["pipeline"] == unit_source["pipeline"]
        assert lowering_plan["pipeline_stages"] == unit_source["pipeline_stages"]


class TestBooleanQueryExpression:
    def test_reference_pipeline_preserves_result_set_as_typed_operand(self) -> None:
        pipeline = parse_reference_query_pipeline("from result-set:stable-set | group by model | count")

        assert pipeline is not None
        assert pipeline.operand == RefOperand(reference=ObjectRef(kind="result-set", object_id="stable-set"))
        assert pipeline.stages == ("group by model", "count")

    def test_reference_pipeline_bypasses_lark_field_clause_colon_ambiguity(self) -> None:
        ast = parse_expression_ast("from result-set:stable-set | count")

        assert ast.ref_operand == RefOperand(reference=ObjectRef(kind="result-set", object_id="stable-set"))
        assert ast.clauses == ()

    @pytest.mark.parametrize(
        ("expression", "match"),
        [
            ("from query:not-a-hash", "query hash must be 64 lowercase hexadecimal characters"),
            ("from query-run:not-a-run", "query run id must start with 'qr_'"),
        ],
    )
    def test_reference_pipeline_uses_canonical_substrate_identity_validation(self, expression: str, match: str) -> None:
        with pytest.raises(ExpressionCompileError, match=match):
            parse_reference_query_pipeline(expression)

    def test_reference_explain_reports_durable_lineage_without_textual_expansion(self) -> None:
        payload = explain_expression("from result-set:stable-set | group by model | count").to_payload()

        assert payload["ast"] == {
            "entry": "reference_pipeline",
            "reference_pipeline": {
                "source": {
                    "kind": "ref_operand",
                    "reference": "result-set:stable-set",
                    "reference_kind": "result-set",
                    "evaluation_mode": "retained",
                },
                "stages": ["group by model", "count"],
            },
        }
        assert cast(dict[str, Any], payload["lowering_plan"])["reference_lineage"] == ["result-set:stable-set"]

    def test_reference_pipeline_rejects_compatibility_text_lowering(self) -> None:
        with pytest.raises(ExpressionCompileError, match="never expanded into text"):
            compile_expression("from result-set:stable-set")

    def test_reference_resolver_rejects_a_cycle_before_materialization(self) -> None:
        operand = RefOperand(reference=ObjectRef(kind="query", object_id="stable-query"))

        class _CyclicResolver:
            def resolve_ref_operand(self, resolved_operand: RefOperand) -> ResolvedRefOperand:
                repeated_parent = ObjectRef(kind="query", object_id="parent-query")
                return ResolvedRefOperand(
                    operand=resolved_operand,
                    grain="session",
                    lineage=(repeated_parent, repeated_parent),
                )

        with pytest.raises(RefOperandCycleError, match="reference cycle"):
            resolve_ref_operand(operand, _CyclicResolver())

    def test_macro_token_is_not_reinterpreted_as_a_reference_operand(self) -> None:
        assert parse_reference_query_pipeline("@saved") is None
        assert _clauses("@saved") == [_TextToken(text="@saved", quoted=False, negated=False)]

    def test_boolean_ast_exposes_predicate_tree(self) -> None:
        ast = parse_expression_ast("repo:polylogue OR origin:chatgpt-export")

        assert ast.clauses == ()
        assert ast.boolean_predicate == QueryBoolPredicate(
            op="or",
            children=(
                QueryFieldPredicate(field="repo", values=("polylogue",)),
                QueryFieldPredicate(field="origin", values=("chatgpt-export",)),
            ),
        )

    def test_boolean_ast_exposes_readable_count_comparisons(self) -> None:
        ast = parse_expression_ast("messages >= 5 AND tool_use_messages >= 1 AND paste_messages = 0")

        assert ast.clauses == ()
        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="messages", values=("5",), op=">="),
                QueryFieldPredicate(field="tool_use_messages", values=("1",), op=">="),
                QueryFieldPredicate(field="paste_messages", values=("0",), op="="),
            ),
        )

    def test_boolean_ast_exposes_readable_count_range(self) -> None:
        ast = parse_expression_ast("sessions where thinking_messages between 1 and 3")

        assert ast.clauses == ()
        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="thinking_messages", values=("1",), op=">="),
                QueryFieldPredicate(field="thinking_messages", values=("3",), op="<="),
            ),
        )

    def test_boolean_ast_exposes_readable_numeric_comparisons(self) -> None:
        ast = parse_expression_ast("sessions where duration_ms >= 60000 AND duration_ms < 120000")

        assert ast.clauses == ()
        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="duration_ms", values=("60000",), op=">="),
                QueryFieldPredicate(field="duration_ms", values=("120000",), op="<"),
            ),
        )

    def test_parse_expression_ast_exposes_readable_date_clauses(self) -> None:
        ast = parse_expression_ast("date >= 2026-01-01 date between 2026-01-01 and 2026-02-01")

        assert ast.clauses == (
            _DateComparisonToken(op=">=", value="2026-01-01"),
            _DateRangeToken(min_value="2026-01-01", max_value="2026-02-01"),
        )

    def test_boolean_ast_exposes_readable_date_comparisons(self) -> None:
        ast = parse_expression_ast("sessions where date >= 2026-01-01 AND repo:polylogue")

        assert ast.clauses == ()
        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="date", values=("2026-01-01",), op=">="),
                QueryFieldPredicate(field="repo", values=("polylogue",)),
            ),
        )

    def test_boolean_ast_exposes_readable_date_range(self) -> None:
        ast = parse_expression_ast("sessions where date between 2026-01-01 and 2026-02-01")

        assert ast.clauses == ()
        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="date", values=("2026-01-01",), op=">="),
                QueryFieldPredicate(field="date", values=("2026-02-01",), op="<="),
            ),
        )

    def test_boolean_not_wraps_leaf_predicate(self) -> None:
        ast = parse_expression_ast("origin:chatgpt-export AND NOT title:slop")

        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="origin", values=("chatgpt-export",)),
                QueryNotPredicate(QueryFieldPredicate(field="title", values=("slop",))),
            ),
        )

    def test_sessions_where_prefix_is_accepted(self) -> None:
        spec = compile_expression("sessions where origin:chatgpt-export OR title:needle")

        assert isinstance(spec.boolean_predicate, QueryBoolPredicate)
        assert spec.query_terms == ()

    def test_structural_exists_ast_exposes_child_unit_predicate(self) -> None:
        ast = parse_expression_ast("exists message(role:assistant AND text:timeout)")

        assert ast.boolean_predicate == QueryExistsPredicate(
            unit="message",
            child=QueryBoolPredicate(
                op="and",
                children=(
                    QueryFieldPredicate(field="role", values=("assistant",)),
                    QueryFieldPredicate(field="text", values=("timeout",)),
                ),
            ),
        )

    def test_message_source_where_lowers_to_structural_exists(self) -> None:
        ast = parse_expression_ast("messages where role:assistant AND text:timeout")

        assert ast.boolean_predicate == QueryExistsPredicate(
            unit="message",
            child=QueryBoolPredicate(
                op="and",
                children=(
                    QueryFieldPredicate(field="role", values=("assistant",)),
                    QueryFieldPredicate(field="text", values=("timeout",)),
                ),
            ),
        )

    def test_parse_unit_source_expression_preserves_terminal_unit(self) -> None:
        source = parse_unit_source_expression("messages where role:assistant AND text:timeout")

        assert source is not None
        assert source.unit == "message"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="role", values=("assistant",)),
                QueryFieldPredicate(field="text", values=("timeout",)),
            ),
        )

    def test_terminal_source_time_comparison_is_parsed(self) -> None:
        source = parse_unit_source_expression("messages where time > 2026-01-02T00:00:00+00:00")

        assert source is not None
        assert source.unit == "message"
        assert source.predicate == QueryFieldPredicate(
            field="time",
            values=("2026-01-02T00:00:00+00:00",),
            op=">",
        )

    def test_terminal_source_time_range_is_parsed(self) -> None:
        source = parse_unit_source_expression(
            "messages where time between 2026-01-01T00:00:00+00:00 and 2026-01-02T00:00:00+00:00"
        )

        assert source is not None
        assert source.unit == "message"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="time", values=("2026-01-01T00:00:00+00:00",), op=">="),
                QueryFieldPredicate(field="time", values=("2026-01-02T00:00:00+00:00",), op="<="),
            ),
        )

    def test_time_predicate_is_rejected_on_sessions(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not supported for session predicates"):
            parse_expression_ast("sessions where time >= 2026-01-02T00:00:00+00:00")

    def test_session_predicate_carries_validated_field_ref(self) -> None:
        ast = parse_expression_ast("sessions where repo:polylogue")

        predicate = cast(QueryFieldPredicate, ast.boolean_predicate)

        assert predicate.field == "repo"
        assert predicate.field_ref is not None
        assert predicate.field_ref.scope == "session"
        assert predicate.field_ref.name == "repo"
        assert predicate.field_ref.source_name == "repo"
        assert predicate.field_ref.unit is None

    def test_boolean_session_alias_carries_exact_id_field_ref(self) -> None:
        ast = parse_expression_ast("sessions where session:claude-code-session:abc123def456")

        predicate = cast(QueryFieldPredicate, ast.boolean_predicate)

        assert predicate.field == "session"
        assert predicate.values == ("claude-code-session:abc123def456",)
        assert predicate.field_ref is not None
        assert predicate.field_ref.scope == "session"
        assert predicate.field_ref.name == "id"
        assert predicate.field_ref.source_name == "session"
        assert predicate.field_ref.unit is None

        spec = compile_expression("session:claude-code-session:abc123def456 OR repo:polylogue")
        boolean = cast(QueryBoolPredicate, spec.boolean_predicate)
        alias = cast(QueryFieldPredicate, boolean.children[0])
        assert alias.field == "session"
        assert alias.field_ref is not None
        assert alias.field_ref.scope == "session"
        assert alias.field_ref.name == "id"
        assert alias.field_ref.source_name == "session"

    def test_pipeline_session_source_scopes_session_alias_as_exact_id(self) -> None:
        source = parse_unit_source_expression(
            "sessions where session:claude-code-session:abc123def456 | messages where role:user"
        )

        assert source is not None
        session_predicate = cast(QueryFieldPredicate, source.session_predicate)
        assert session_predicate.field_ref is not None
        assert session_predicate.field_ref.name == "id"

        combined = cast(QueryBoolPredicate, source.predicate)
        scoped_alias = cast(QueryFieldPredicate, combined.children[0])
        assert scoped_alias.field == "session.id"
        assert scoped_alias.field_ref is not None
        assert scoped_alias.field_ref.scope == "session"
        assert scoped_alias.field_ref.name == "id"
        assert scoped_alias.field_ref.source_name == "session.id"
        assert scoped_alias.field_ref.unit == "message"

    def test_runtime_terminal_unit_accepts_descriptor_backed_session_scope(self) -> None:
        source = parse_unit_source_expression(
            "context-snapshots where session.repo:polylogue AND boundary:session_start"
        )

        assert source is not None
        assert source.unit == "context-snapshot"
        predicate = cast(QueryBoolPredicate, source.predicate)
        scoped_repo = cast(QueryFieldPredicate, predicate.children[0])
        boundary = cast(QueryFieldPredicate, predicate.children[1])

        assert scoped_repo.field_ref is not None
        assert scoped_repo.field_ref.scope == "session"
        assert scoped_repo.field_ref.name == "repo"
        assert scoped_repo.field_ref.source_name == "session.repo"
        assert scoped_repo.field_ref.unit == "context-snapshot"
        assert boundary.field_ref is not None
        assert boundary.field_ref.scope == "unit"
        assert boundary.field_ref.name == "boundary"
        assert boundary.field_ref.unit == "context-snapshot"
        assert scoped_repo.to_payload()["field_ref"] == {
            "scope": "session",
            "name": "repo",
            "source_name": "session.repo",
            "unit": "context-snapshot",
        }

    def test_sql_terminal_unit_accepts_full_session_scope(self) -> None:
        source = parse_unit_source_expression("context-snapshots where session.action:file_edit")

        assert source is not None
        predicate = cast(QueryFieldPredicate, source.predicate)
        assert predicate.field_ref is not None
        assert predicate.field_ref.scope == "session"
        assert predicate.field_ref.name == "action"
        assert predicate.field_ref.unit == "context-snapshot"

    def test_sql_terminal_unit_accepts_session_path_scope(self) -> None:
        source = parse_unit_source_expression("messages where session.path:polylogue/archive AND role:assistant")

        assert source is not None
        predicate = cast(QueryBoolPredicate, source.predicate)
        scoped_path = cast(QueryFieldPredicate, predicate.children[0])

        assert scoped_path.field_ref is not None
        assert scoped_path.field_ref.scope == "session"
        assert scoped_path.field_ref.name == "path"
        assert scoped_path.field_ref.source_name == "session.path"
        assert scoped_path.field_ref.unit == "message"

    def test_pipeline_session_source_scopes_terminal_unit_query(self) -> None:
        source = parse_unit_source_expression(
            "sessions where repo:polylogue AND origin:claude-code-session | messages where role:assistant"
        )

        assert source is not None
        assert source.unit == "message"
        assert source.session_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="repo", values=("polylogue",)),
                QueryFieldPredicate(field="origin", values=("claude-code-session",)),
            ),
        )
        session_predicate = source.session_predicate
        assert isinstance(session_predicate, QueryBoolPredicate)
        session_repo = cast(QueryFieldPredicate, session_predicate.children[0])
        session_origin = cast(QueryFieldPredicate, session_predicate.children[1])
        assert session_repo.field_ref is not None
        assert session_repo.field_ref.scope == "session"
        assert session_repo.field_ref.name == "repo"
        assert session_repo.field_ref.source_name == "repo"
        assert session_repo.field_ref.unit is None
        assert session_origin.field_ref is not None
        assert session_origin.field_ref.scope == "session"
        assert session_origin.field_ref.name == "origin"
        assert session_origin.field_ref.source_name == "origin"
        assert session_origin.field_ref.unit is None
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryBoolPredicate(
                    op="and",
                    children=(
                        QueryFieldPredicate(field="session.repo", values=("polylogue",)),
                        QueryFieldPredicate(field="session.origin", values=("claude-code-session",)),
                    ),
                ),
                QueryFieldPredicate(field="role", values=("assistant",)),
            ),
        )
        assert isinstance(source.predicate, QueryBoolPredicate)
        scoped_session = source.predicate.children[0]
        role = source.predicate.children[1]
        assert isinstance(scoped_session, QueryBoolPredicate)
        assert isinstance(role, QueryFieldPredicate)
        scoped_repo = scoped_session.children[0]
        scoped_origin = scoped_session.children[1]
        assert isinstance(scoped_repo, QueryFieldPredicate)
        assert isinstance(scoped_origin, QueryFieldPredicate)

        assert scoped_repo.field_ref is not None
        assert scoped_repo.field_ref.scope == "session"
        assert scoped_repo.field_ref.name == "repo"
        assert scoped_repo.field_ref.source_name == "session.repo"
        assert scoped_repo.field_ref.unit == "message"
        assert scoped_origin.field_ref is not None
        assert scoped_origin.field_ref.scope == "session"
        assert scoped_origin.field_ref.name == "origin"
        assert scoped_origin.field_ref.source_name == "session.origin"
        assert scoped_origin.field_ref.unit == "message"
        assert role.field_ref is not None
        assert role.field_ref.scope == "unit"
        assert role.field_ref.name == "role"
        assert role.field_ref.unit == "message"

    def test_pipeline_session_source_accepts_lineage_predicate(self) -> None:
        source = parse_unit_source_expression(
            "sessions where lineage:id:chatgpt-export:ext-child | messages where role:user"
        )

        assert source is not None
        assert source.unit == "message"
        assert source.session_predicate == QueryLineagePredicate(seed_session_id="chatgpt-export:ext-child")
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryLineagePredicate(seed_session_id="chatgpt-export:ext-child"),
                QueryFieldPredicate(field="role", values=("user",)),
            ),
        )
        assert source.pipeline_stages == (
            QueryUnitSessionScopeStage(QueryLineagePredicate(seed_session_id="chatgpt-export:ext-child")),
        )

    def test_pipeline_split_ignores_field_alternation_pipe(self) -> None:
        source = parse_unit_source_expression(
            "sessions where origin:(codex-session|claude-code-session) | actions where action:file_edit"
        )

        assert source is not None
        assert source.unit == "action"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="session.origin", values=("codex-session", "claude-code-session")),
                QueryFieldPredicate(field="action", values=("file_edit",)),
            ),
        )

    def test_pipeline_limit_and_offset_stages_are_parsed(self) -> None:
        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where role:assistant | limit 2 | offset 3"
        )

        assert source is not None
        assert source.limit == 2
        assert source.offset == 3
        assert [stage.to_payload() for stage in source.pipeline_stages] == [
            _session_scope_stage("repo", "polylogue"),
            {"kind": "limit", "value": 2},
            {"kind": "offset", "value": 3},
        ]
        assert source.pipeline == QueryUnitPipeline(
            source_unit="message",
            predicate=source.predicate,
            session_predicate=source.session_predicate,
            stages=source.pipeline_stages,
            limit=2,
            offset=3,
        )
        assert source.pipeline.to_payload()["result"] == {"limit": 2, "offset": 3}

    def test_terminal_source_pipeline_limit_offset_and_sort_are_parsed(self) -> None:
        source = parse_unit_source_expression("messages where role:assistant | sort by time desc | limit 2 | offset 3")

        assert source is not None
        assert source.unit == "message"
        assert source.session_predicate is None
        assert source.limit == 2
        assert source.offset == 3
        assert source.sort is not None
        assert source.sort.field == "time"
        assert source.sort.direction == "desc"
        assert source.predicate == QueryFieldPredicate(field="role", values=("assistant",))
        assert [stage.to_payload() for stage in source.pipeline_stages] == [
            {"kind": "sort", "sort": {"field": "time", "direction": "desc"}},
            {"kind": "limit", "value": 2},
            {"kind": "offset", "value": 3},
        ]

    def test_pipeline_sort_stage_is_parsed(self) -> None:
        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where role:assistant | sort by time desc"
        )

        assert source is not None
        assert source.sort is not None
        assert source.sort.field == "time"
        assert source.sort.direction == "desc"

    def test_pipeline_group_by_count_stages_are_parsed(self) -> None:
        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where role:assistant | group by role | count | sort by count asc"
        )

        assert source is not None
        assert source.group_by == "role"
        assert source.aggregate == "count"
        assert source.sort is not None
        assert source.sort.field == "count"
        assert source.sort.direction == "asc"
        assert [stage.to_payload() for stage in source.pipeline_stages] == [
            _session_scope_stage("repo", "polylogue"),
            {"kind": "group", "field": "role"},
            {"kind": "count", "metric": "count"},
            {"kind": "sort", "sort": {"field": "count", "direction": "asc"}},
        ]

    def test_pipeline_rejects_unknown_terminal_stage(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unsupported pipeline stage"):
            parse_unit_source_expression("sessions where repo:polylogue | messages where role:assistant | sample 3")

    def test_pipeline_sessions_count_names_the_unsupported_shape(self) -> None:
        """`sessions where ... | count` (polylogue-9srm): sessions has no terminal
        row/aggregate lowerer of its own, so this must fail with an error that
        names `count` specifically and points at the working alternative
        (`find ... then analyze --count`), not the generic
        "must be an executable `<unit>s where ...` query" message.
        """

        with pytest.raises(ExpressionCompileError) as exc_info:
            parse_unit_source_expression("sessions where repo:polylogue | count")

        message = str(exc_info.value)
        assert "`count` cannot follow `sessions where ...` directly" in message
        assert "analyze --count" in message
        assert "actions/blocks" in message  # supported terminal units are named

    def test_pipeline_sessions_group_by_names_the_unsupported_shape(self) -> None:
        """Same as above for `group by` directly on `sessions where ...`
        (polylogue-9srm): previously raised the identical generic message even
        though `group by` (not the missing terminal stage) is the actual
        problem.
        """

        with pytest.raises(ExpressionCompileError) as exc_info:
            parse_unit_source_expression("sessions where origin:codex-session | group by origin | count")

        message = str(exc_info.value)
        assert "`group by` cannot follow `sessions where ...` directly" in message
        assert "analyze --count" in message

    @pytest.mark.parametrize("stage", ["sort by time", "limit 10", "offset 5"])
    def test_pipeline_sessions_other_terminal_keywords_name_the_unsupported_shape(self, stage: str) -> None:
        with pytest.raises(ExpressionCompileError) as exc_info:
            parse_unit_source_expression(f"sessions where repo:polylogue | {stage}")

        message = str(exc_info.value)
        assert "cannot follow `sessions where ...` directly" in message

    def test_pipeline_sessions_unrecognized_second_stage_keeps_generic_message(self) -> None:
        """A second stage that isn't a `<unit>s where ...` query AND doesn't
        look like a known terminal pipeline keyword still falls back to the
        original generic message (now naming supported terminal units too).
        """

        with pytest.raises(ExpressionCompileError) as exc_info:
            parse_unit_source_expression("sessions where repo:polylogue | not a real stage")

        message = str(exc_info.value)
        assert "must be an executable `<unit>s where ...` query" in message
        assert "cannot follow `sessions where ...` directly" not in message

    def test_pipeline_accepts_sort_by_time_on_sql_run_rows(self) -> None:
        source = parse_unit_source_expression(
            "sessions where repo:polylogue | runs where status:completed | sort by time desc"
        )
        assert source is not None
        assert source.unit == "run"
        assert source.sort is not None
        assert source.sort.field == "time"
        assert source.sort.direction == "desc"

    def test_pipeline_rejects_aggregate_on_unaggregated_sql_units(self) -> None:
        with pytest.raises(ExpressionCompileError, match="has no aggregate lowerer"):
            parse_unit_source_expression(
                "sessions where repo:polylogue | context-snapshots where boundary:session_start | count"
            )

    @pytest.mark.parametrize(
        "source",
        [
            descriptor.plural_source
            for descriptor in query_unit_descriptors(lowerer_kind="sql")
            if descriptor.aggregate_group_fields
        ],
    )
    def test_descriptor_sql_units_accept_advertised_aggregate_pipeline(self, source: str) -> None:
        descriptor = next(
            descriptor
            for descriptor in query_unit_descriptors(lowerer_kind="sql")
            if descriptor.plural_source == source
        )
        group_field = descriptor.aggregate_group_fields[0]

        parsed = parse_unit_source_expression(
            f"{source} where {descriptor.fields[0].example} | group by {group_field} | count | sort by count desc"
        )

        assert parsed is not None
        assert parsed.unit == descriptor.unit
        assert parsed.group_by == group_field
        assert parsed.aggregate == "count"
        assert parsed.sort is not None
        assert parsed.sort.field == "count"

    @pytest.mark.parametrize(
        "source",
        [descriptor.plural_source for descriptor in query_unit_descriptors(lowerer_kind="runtime_transform")],
    )
    def test_descriptor_runtime_transform_units_reject_aggregate_pipeline(self, source: str) -> None:
        descriptor = next(
            descriptor
            for descriptor in query_unit_descriptors(lowerer_kind="runtime_transform")
            if descriptor.plural_source == source
        )

        with pytest.raises(ExpressionCompileError, match="runtime-transform units need an aggregate lowerer"):
            parse_unit_source_expression(f"{source} where {descriptor.fields[0].example} | count")

    def test_pipeline_rejects_aggregate_sort_before_count(self) -> None:
        with pytest.raises(ExpressionCompileError, match="require an aggregate `count` stage"):
            parse_unit_source_expression("messages where role:assistant | group by role | sort by count desc")

    def test_pipeline_rejects_aggregate_sort_after_limit(self) -> None:
        with pytest.raises(ExpressionCompileError, match="must appear before `limit` and `offset`"):
            parse_unit_source_expression(
                "messages where role:assistant | group by role | count | limit 5 | sort by count"
            )

    def test_pipeline_rejects_row_sort_before_aggregate(self) -> None:
        with pytest.raises(ExpressionCompileError, match="cannot feed aggregate stages"):
            parse_unit_source_expression("messages where role:assistant | sort by time asc | group by role | count")

    def test_pipeline_rejects_ranked_semantic_session_stage(self) -> None:
        with pytest.raises(ExpressionCompileError, match="semantic stages need a ranked terminal lowerer"):
            parse_unit_source_expression('sessions where semantic:"timeout" | messages where role:assistant')

    def test_compile_expression_rejects_terminal_pipelines_as_session_selector(self) -> None:
        with pytest.raises(ExpressionCompileError, match="pipeline unit queries return terminal rows"):
            compile_expression("messages where role:assistant | limit 10")

        with pytest.raises(ExpressionCompileError, match="pipeline unit queries return terminal rows"):
            compile_expression("messages where role:assistant | group by role | count")

        with pytest.raises(ExpressionCompileError, match="pipeline unit queries return terminal rows"):
            parse_expression_ast("messages where role:assistant | group by role | count")

        with pytest.raises(ExpressionCompileError, match="pipeline unit queries return terminal rows"):
            compile_expression("sessions where repo:polylogue | messages where role:assistant")

    def test_assertion_source_where_lowers_to_structural_exists(self) -> None:
        ast = parse_expression_ast("assertions where kind:decision AND text:review")

        assert ast.boolean_predicate == QueryExistsPredicate(
            unit="assertion",
            child=QueryBoolPredicate(
                op="and",
                children=(
                    QueryFieldPredicate(field="kind", values=("decision",)),
                    QueryFieldPredicate(field="text", values=("review",)),
                ),
            ),
        )

    def test_parse_assertion_source_expression_preserves_terminal_unit(self) -> None:
        source = parse_unit_source_expression("assertions where kind:decision AND status:active")

        assert source is not None
        assert source.unit == "assertion"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="kind", values=("decision",)),
                QueryFieldPredicate(field="status", values=("active",)),
            ),
        )

    def test_assertion_value_path_equality_predicate(self) -> None:
        source = parse_unit_source_expression("assertions where value.status:approved")

        assert source is not None
        assert source.unit == "assertion"
        assert source.predicate == QueryFieldPredicate(field="value.status", values=("approved",), op="=")

    def test_assertion_value_path_quoted_string_preserves_literal_type(self) -> None:
        source = parse_unit_source_expression('assertions where value.status:"true"')

        assert source is not None
        assert source.predicate == QueryFieldPredicate(field="value.status", values=('"true"',), op="=")

    def test_assertion_value_path_equality_preserves_alternation(self) -> None:
        source = parse_unit_source_expression('assertions where value.status:("4"|"true"|null)')

        assert source is not None
        assert source.predicate == QueryFieldPredicate(
            field="value.status",
            values=('"4"', '"true"', "null"),
            op="=",
        )

    def test_assertion_value_path_quoted_pipe_is_not_an_alternation(self) -> None:
        source = parse_unit_source_expression('assertions where value.status:"a|b"')

        assert source is not None
        assert source.predicate == QueryFieldPredicate(field="value.status", values=('"a|b"',), op="=")

    def test_assertion_value_path_comparison_predicate(self) -> None:
        source = parse_unit_source_expression("assertions where value.score:>=4")

        assert source is not None
        assert source.predicate == QueryFieldPredicate(field="value.score", values=("4",), op=">=")

    def test_assertion_value_path_nested_predicate(self) -> None:
        source = parse_unit_source_expression("assertions where value.rubric.confidence:<0.5")

        assert source is not None
        assert source.predicate == QueryFieldPredicate(field="value.rubric.confidence", values=("0.5",), op="<")

    def test_assertion_value_path_negation(self) -> None:
        source = parse_unit_source_expression("assertions where NOT value.score:>=4")

        assert source is not None
        assert source.predicate == QueryNotPredicate(QueryFieldPredicate(field="value.score", values=("4",), op=">="))

    def test_assertion_value_path_boolean_combination(self) -> None:
        source = parse_unit_source_expression("assertions where kind:annotation AND value.score:>=4")

        assert source is not None
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="kind", values=("annotation",)),
                QueryFieldPredicate(field="value.score", values=("4",), op=">="),
            ),
        )

    def test_exists_assertion_value_path_predicate(self) -> None:
        ast = parse_expression_ast("exists assertion(value.score:>=4 AND kind:annotation)")

        assert ast.boolean_predicate == QueryExistsPredicate(
            unit="assertion",
            child=QueryBoolPredicate(
                op="and",
                children=(
                    QueryFieldPredicate(field="value.score", values=("4",), op=">="),
                    QueryFieldPredicate(field="kind", values=("annotation",)),
                ),
            ),
        )

    def test_assertion_value_path_non_numeric_comparator_rejected(self) -> None:
        with pytest.raises(ExpressionCompileError, match="numeric value"):
            parse_unit_source_expression("assertions where value.score:>=not-a-number")

    def test_assertion_value_path_non_finite_comparator_rejected(self) -> None:
        with pytest.raises(ExpressionCompileError, match="finite numeric value"):
            parse_unit_source_expression("assertions where value.score:>=NaN")

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1" + "0" * 400])
    def test_assertion_value_path_non_finite_equality_rejected(self, literal: str) -> None:
        with pytest.raises(ExpressionCompileError, match="finite JSON scalar"):
            parse_unit_source_expression(f"assertions where value.score:{literal}")

    @pytest.mark.parametrize("literal", ["[]", "{}"])
    def test_assertion_value_path_structured_equality_rejected(self, literal: str) -> None:
        # These shapes are rejected at the grammar boundary before the
        # value-path scalar validator runs; either way they cannot become an
        # equality predicate or reach SQL lowering.
        with pytest.raises(ExpressionCompileError):
            parse_unit_source_expression(f"assertions where value.score:{literal}")

    def test_assertion_value_path_rejected_for_non_assertion_unit(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not supported"):
            parse_unit_source_expression("messages where value.score:>=4")

    def test_assertion_value_path_rejected_for_empty_path(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown query field"):
            parse_unit_source_expression("assertions where value.:4")

    def test_assertion_value_path_requires_a_value(self) -> None:
        # An empty value never matches FIELD_CLAUSE's non-empty value
        # requirement, so this is a grammar-level parse failure (same as any
        # other empty-valued field clause) rather than a value-path-specific
        # message -- documented here so the empty-path shape has a pinned
        # regression rather than relying on incidental grammar behavior.
        with pytest.raises(ExpressionCompileError):
            compile_expression("exists assertion(value.score:)")

    def test_parse_observed_event_source_expression_preserves_terminal_unit(self) -> None:
        source = parse_unit_source_expression(
            "observed-events where session.repo:polylogue AND delivery_state:acted_on AND text:#2100"
        )

        assert source is not None
        assert source.unit == "observed-event"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="session.repo", values=("polylogue",)),
                QueryFieldPredicate(field="delivery_state", values=("acted_on",)),
                QueryFieldPredicate(field="text", values=("#2100",)),
            ),
        )
        explanation = explain_expression(
            "observed-events where session.repo:polylogue AND delivery_state:acted_on AND text:#2100"
        )
        assert explanation.selected_units == ("observed-event",)
        assert explanation.lowering_plan is not None
        assert "compatibility_selector" in explanation.lowering_plan

    def test_parse_run_source_expression_preserves_terminal_unit(self) -> None:
        source = parse_unit_source_expression("runs where session.repo:polylogue AND role:subagent AND agent:Explore")

        assert source is not None
        assert source.unit == "run"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="session.repo", values=("polylogue",)),
                QueryFieldPredicate(field="role", values=("subagent",)),
                QueryFieldPredicate(field="agent", values=("Explore",)),
            ),
        )
        explanation = explain_expression("runs where session.repo:polylogue AND role:subagent AND agent:Explore")
        assert explanation.lowerer == "lark-query-unit-source-to-terminal-unit"
        assert explanation.selected_units == ("run",)
        assert explanation.execution_legs == ("sql", "terminal-run-rows")
        assert explanation.plan_description == (
            "terminal unit source: run",
            "compatibility session selector: exists run(...)",
        )
        assert explanation.lowering_plan is not None
        assert "compatibility_selector" in explanation.lowering_plan

    @pytest.mark.parametrize(
        "expression",
        (
            "runs where role:subagent",
            "observed-events where kind:session_started",
            "context-snapshots where boundary:session_start",
        ),
    )
    def test_exists_supported_unit_sources_compile_to_session_selectors(
        self,
        expression: str,
    ) -> None:
        # These units are now SQL + exists-supported, so a bare terminal-source
        # expression lowers to an ``exists <unit>(...)`` session selector instead
        # of being rejected as terminal-only.
        spec = compile_expression(expression)
        assert spec is not None

    def test_parse_context_snapshot_source_expression_preserves_terminal_unit(self) -> None:
        source = parse_unit_source_expression(
            "context-snapshots where session.repo:polylogue AND boundary:session_start AND text:run"
        )

        assert source is not None
        assert source.unit == "context-snapshot"
        assert source.predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="session.repo", values=("polylogue",)),
                QueryFieldPredicate(field="boundary", values=("session_start",)),
                QueryFieldPredicate(field="text", values=("run",)),
            ),
        )
        explanation = explain_expression(
            "context-snapshots where session.repo:polylogue AND boundary:session_start AND text:run"
        )
        assert explanation.selected_units == ("context-snapshot",)
        assert explanation.lowering_plan is not None
        assert "compatibility_selector" in explanation.lowering_plan

    def test_action_source_where_lowers_to_structural_exists(self) -> None:
        ast = parse_expression_ast("actions where action:file_edit AND path:archive/query")

        assert ast.boolean_predicate == QueryExistsPredicate(
            unit="action",
            child=QueryBoolPredicate(
                op="and",
                children=(
                    QueryFieldPredicate(field="action", values=("file_edit",)),
                    QueryFieldPredicate(field="path", values=("archive/query",)),
                ),
            ),
        )

    @pytest.mark.parametrize(
        "expression", ["messages where", "actions where   ", "blocks where\t", "assertions where "]
    )
    def test_source_where_without_predicate_is_rejected(self, expression: str) -> None:
        with pytest.raises(ExpressionCompileError, match="where requires a predicate"):
            compile_expression(expression)

    def test_block_exists_ast_exposes_child_unit_predicate(self) -> None:
        ast = parse_expression_ast("exists block(type:code AND text:timeout)")

        assert ast.boolean_predicate == QueryExistsPredicate(
            unit="block",
            child=QueryBoolPredicate(
                op="and",
                children=(
                    QueryFieldPredicate(field="type", values=("code",)),
                    QueryFieldPredicate(field="text", values=("timeout",)),
                ),
            ),
        )

    def test_compile_expression_detects_single_block_exists_predicate(self) -> None:
        spec = compile_expression("exists block(type:code)")

        assert spec.boolean_predicate == QueryExistsPredicate(
            unit="block",
            child=QueryFieldPredicate(field="type", values=("code",)),
        )

    def test_sequence_ast_exposes_ordered_action_terms(self) -> None:
        ast = parse_expression_ast("seq(action:file_edit -> action:shell -> action:file_edit)")

        assert ast.boolean_predicate == QuerySequencePredicate(action_terms=("file_edit", "shell", "file_edit"))

    def test_sequence_ast_exposes_next_and_within_constraints(self) -> None:
        ast = parse_expression_ast("seq(action:file_edit ->[next] action:shell ->[within:5m] action:file_edit)")

        assert isinstance(ast.boolean_predicate, QuerySequencePredicate)
        assert ast.boolean_predicate.constraints == (
            QuerySequenceConstraint(kind="next"),
            QuerySequenceConstraint(kind="within", within_ms=300_000),
        )
        assert ast.boolean_predicate.to_payload()["constraints"] == [
            {"kind": "next"},
            {"kind": "within", "within_ms": 300_000},
        ]

    @pytest.mark.parametrize("modifier", ["within:0s", "within:5weeks", "adjacent"])
    def test_sequence_rejects_invalid_constraints(self, modifier: str) -> None:
        with pytest.raises(ExpressionCompileError):
            compile_expression(f"seq(action:file_edit ->[{modifier}] action:shell)")

    def test_sequence_predicate_derives_action_terms_from_steps_when_both_are_supplied(self) -> None:
        predicate = QuerySequencePredicate(
            steps=(
                QueryFieldPredicate(field="action", values=("file_edit",)),
                QueryFieldPredicate(field="action", values=("shell",)),
            ),
            action_terms=("conflicting",),
        )

        assert predicate.action_terms == ("file_edit", "shell")
        assert predicate.to_payload()["actions"] == ["file_edit", "shell"]

    def test_sequence_ast_exposes_step_predicates(self) -> None:
        ast = parse_expression_ast("seq(action:file_edit -> action:shell AND output:failed -> action:file_edit)")

        assert ast.boolean_predicate == QuerySequencePredicate(
            steps=(
                QueryFieldPredicate(field="action", values=("file_edit",)),
                QueryBoolPredicate(
                    op="and",
                    children=(
                        QueryFieldPredicate(field="action", values=("shell",)),
                        QueryFieldPredicate(field="output", values=("failed",)),
                    ),
                ),
                QueryFieldPredicate(field="action", values=("file_edit",)),
            ),
        )
        assert isinstance(ast.boolean_predicate, QuerySequencePredicate)
        sequence = ast.boolean_predicate
        first_step = sequence.steps[0]
        second_step = sequence.steps[1]
        third_step = sequence.steps[2]
        assert isinstance(first_step, QueryFieldPredicate)
        assert isinstance(second_step, QueryBoolPredicate)
        assert isinstance(third_step, QueryFieldPredicate)
        second_action = second_step.children[0]
        second_output = second_step.children[1]
        assert isinstance(second_action, QueryFieldPredicate)
        assert isinstance(second_output, QueryFieldPredicate)

        assert first_step.field_ref is not None
        assert first_step.field_ref.scope == "unit"
        assert first_step.field_ref.unit == "action"
        assert second_action.field_ref is not None
        assert second_action.field_ref.name == "action"
        assert second_output.field_ref is not None
        assert second_output.field_ref.name == "output"
        assert third_step.field_ref is not None
        assert third_step.field_ref.scope == "unit"
        assert third_step.field_ref.unit == "action"
        assert third_step.field_ref.name == "action"

    def test_sequence_step_rejects_session_fields(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not supported for action predicates"):
            compile_expression("seq(action:file_edit -> repo:polylogue)")

    def test_fts_ast_exposes_text_predicate(self) -> None:
        ast = parse_expression_ast('repo:polylogue AND ~"timeout failure"')

        assert ast.boolean_predicate == QueryBoolPredicate(
            op="and",
            children=(
                QueryFieldPredicate(field="repo", values=("polylogue",)),
                QueryTextPredicate(text="timeout failure"),
            ),
        )

    def test_bare_fts_ast_exposes_text_predicate(self) -> None:
        ast = parse_expression_ast("~timeout")

        assert ast.boolean_predicate == QueryTextPredicate(text="timeout")

    def test_structural_field_at_session_scope_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not supported for session predicates"):
            compile_expression("role:assistant OR title:needle")

    def test_session_field_inside_structural_predicate_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="not supported for message predicates"):
            compile_expression("exists message(origin:chatgpt-export)")

    def test_structural_words_requires_numeric_value(self) -> None:
        with pytest.raises(ExpressionCompileError, match="requires a numeric value"):
            compile_expression("exists message(words:many)")

    def test_boolean_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "claude-hit")
            .provider("claude-code")
            .title("ordinary")
            .add_message("m1", role="user", text="alpha")
            .save()
        )
        (
            SessionBuilder(index_db, "title-hit")
            .provider("chatgpt")
            .title("needle")
            .add_message("m2", role="user", text="beta")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("other")
            .add_message("m3", role="user", text="gamma")
            .save()
        )

        spec = compile_expression("origin:claude-code-session OR title:needle")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)
            count = archive.count_sessions(boolean_predicate=spec.boolean_predicate)

        assert {row.session_id for row in rows} == {
            "claude-code-session:ext-claude-hit",
            "chatgpt-export:ext-title-hit",
        }
        assert count == 2

    def test_boolean_and_not_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        for session_id, title in [("keep", "needle"), ("drop", "other")]:
            (
                SessionBuilder(index_db, session_id)
                .provider("chatgpt")
                .title(title)
                .add_message(f"m-{session_id}", role="user", text=title)
                .save()
            )

        spec = compile_expression("origin:chatgpt-export AND NOT title:other")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-keep"]

    def test_boolean_date_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "old")
            .provider("chatgpt")
            .created_at("2026-01-01T00:00:00+00:00")
            .updated_at("2026-01-01T00:00:00+00:00")
            .title("old")
            .add_message("m-old", role="user", text="old")
            .save()
        )
        (
            SessionBuilder(index_db, "new")
            .provider("chatgpt")
            .created_at("2026-02-01T00:00:00+00:00")
            .updated_at("2026-02-01T00:00:00+00:00")
            .title("new")
            .add_message("m-new", role="user", text="new")
            .save()
        )

        spec = compile_expression("sessions where date >= 2026-02-01 AND origin:chatgpt-export")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-new"]

    def test_exists_message_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("structural hit")
            .add_message("m-hit", role="assistant", text="the timeout happened")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-role")
            .provider("chatgpt")
            .title("wrong role")
            .add_message("m-wrong-role", role="user", text="the timeout happened")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-text")
            .provider("chatgpt")
            .title("wrong text")
            .add_message("m-wrong-text", role="assistant", text="ordinary response")
            .save()
        )

        spec = compile_expression("exists message(role:assistant AND text:timeout)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_structural_text_alternation_uses_or_semantics(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        for session_id, text in [
            ("timeout", "request timeout"),
            ("panic", "panic in parser"),
            ("miss", "ordinary response"),
        ]:
            (
                SessionBuilder(index_db, session_id)
                .provider("chatgpt")
                .title(session_id)
                .add_message(f"m-{session_id}", role="assistant", text=text)
                .save()
            )

        spec = compile_expression("exists message(text:(timeout|panic))")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert {row.session_id for row in rows} == {
            "chatgpt-export:ext-timeout",
            "chatgpt-export:ext-panic",
        }

    def test_message_source_where_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("message source hit")
            .add_message("m-hit", role="assistant", text="the timeout happened")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-role")
            .provider("chatgpt")
            .title("wrong role")
            .add_message("m-wrong-role", role="user", text="the timeout happened")
            .save()
        )

        spec = compile_expression("messages where role:assistant AND text:timeout")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_terminal_message_source_returns_message_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("message hit")
            .add_message("m-hit", role="assistant", text="the timeout happened")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("message miss")
            .add_message("m-miss", role="user", text="the timeout happened")
            .save()
        )

        source = parse_unit_source_expression("messages where role:assistant AND text:timeout")
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.role, row.text) for row in rows] == [
            (
                "chatgpt-export:ext-hit",
                "chatgpt-export:ext-hit:m-hit",
                "assistant",
                "the timeout happened",
            )
        ]

    def test_terminal_message_source_filters_by_row_time(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("message time hit")
            .add_message("m-old", role="assistant", text="old", timestamp="2026-01-01T00:00:00+00:00")
            .add_message("m-new", role="assistant", text="new", timestamp="2026-01-02T00:00:00+00:00")
            .save()
        )

        source = parse_unit_source_expression("messages where time >= 2026-01-02T00:00:00+00:00")
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.message_id, row.text) for row in rows] == [("chatgpt-export:ext-hit:m-new", "new")]
        assert rows[0].occurred_at_ms is not None

    def test_exists_message_predicate_filters_by_row_time(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("message time hit")
            .add_message("m-old", role="assistant", text="old", timestamp="2026-01-01T00:00:00+00:00")
            .add_message("m-new", role="assistant", text="new", timestamp="2026-01-02T00:00:00+00:00")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("message time miss")
            .add_message("m-old", role="assistant", text="old", timestamp="2026-01-01T00:00:00+00:00")
            .save()
        )

        spec = compile_expression("exists message(time >= 2026-01-02T00:00:00+00:00)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_boolean_session_alias_executes_as_exact_id(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("exact alias hit")
            .add_message("m-hit", role="user", text="target body")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("exact alias miss")
            .add_message("m-miss", role="user", text="target body")
            .save()
        )

        spec = compile_expression("sessions where session:chatgpt-export:ext-hit")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_session_aggregate_comparisons_execute_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("aggregate count hit")
            .add_message(
                "m-tool",
                role="assistant",
                text="used tool",
                blocks=[{"type": "tool_use", "tool_name": "bash", "text": "pytest"}],
            )
            .add_message(
                "m-thinking",
                role="assistant",
                text="reasoning",
                blocks=[{"type": "thinking", "text": "plan"}],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss-paste")
            .provider("chatgpt")
            .title("aggregate count miss paste")
            .add_message(
                "m-tool",
                role="assistant",
                text="used tool",
                blocks=[{"type": "tool_use", "tool_name": "bash", "text": "pytest"}],
            )
            .add_message("m-paste", role="user", text="pasted", has_paste=1)
            .save()
        )

        spec = compile_expression(
            "sessions where tool_use_messages >= 1 AND thinking_messages >= 1 AND paste_messages = 0"
        )
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_session_duration_comparison_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        import sqlite3

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "duration-hit")
            .provider("chatgpt")
            .title("duration hit")
            .add_message("m-hit", role="assistant", text="long enough")
            .save()
        )
        (
            SessionBuilder(index_db, "duration-miss")
            .provider("chatgpt")
            .title("duration miss")
            .add_message("m-miss", role="assistant", text="too short")
            .save()
        )
        (
            SessionBuilder(index_db, "duration-boundary")
            .provider("chatgpt")
            .title("duration boundary")
            .add_message("m-boundary", role="assistant", text="exact threshold")
            .save()
        )
        (
            SessionBuilder(index_db, "duration-null")
            .provider("chatgpt")
            .title("duration null")
            .add_message("m-null", role="assistant", text="unknown duration")
            .save()
        )
        with sqlite3.connect(index_db) as conn:
            conn.execute(
                "UPDATE sessions SET reported_duration_ms = ? WHERE session_id = ?",
                (90_000, "chatgpt-export:ext-duration-hit"),
            )
            conn.execute(
                "UPDATE sessions SET reported_duration_ms = ? WHERE session_id = ?",
                (5_000, "chatgpt-export:ext-duration-miss"),
            )
            conn.execute(
                "UPDATE sessions SET reported_duration_ms = ? WHERE session_id = ?",
                (60_000, "chatgpt-export:ext-duration-boundary"),
            )
            conn.commit()

        with ArchiveStore.open_existing(index_db.parent) as archive:
            spec = compile_expression("sessions where duration_ms >= 60000")
            assert spec.boolean_predicate is not None
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == [
            "chatgpt-export:ext-duration-boundary",
            "chatgpt-export:ext-duration-hit",
        ]

        with ArchiveStore.open_existing(index_db.parent) as archive:
            spec = compile_expression("sessions where NOT duration_ms >= 60000")
            assert spec.boolean_predicate is not None
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-duration-miss"]

        with ArchiveStore.open_existing(index_db.parent) as archive:
            spec = compile_expression("sessions where duration_ms > 60000")
            assert spec.boolean_predicate is not None
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-duration-hit"]

    def test_message_numeric_comparisons_execute_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "message-numeric")
            .provider("chatgpt")
            .title("message numeric")
            .add_message(
                "m-hit",
                role="assistant",
                text="token heavy",
                input_tokens=1200,
                output_tokens=400,
                cache_read_tokens=300,
                cache_write_tokens=20,
                duration_ms=2500,
            )
            .add_message(
                "m-miss",
                role="assistant",
                text="token light",
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=0,
                cache_write_tokens=0,
                duration_ms=100,
            )
            .add_message(
                "m-lower-boundary",
                role="assistant",
                text="token lower boundary",
                input_tokens=1200,
                output_tokens=400,
                cache_read_tokens=300,
                cache_write_tokens=20,
                duration_ms=2000,
            )
            .add_message(
                "m-upper-boundary",
                role="assistant",
                text="token upper boundary",
                input_tokens=1200,
                output_tokens=400,
                cache_read_tokens=300,
                cache_write_tokens=20,
                duration_ms=3000,
            )
            .save()
        )

        source = parse_unit_source_expression(
            "messages where input_tokens >= 1000 AND output_tokens >= 300 "
            "AND cache_read_tokens >= 100 AND duration_ms > 2000 AND duration_ms < 3000"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="message numeric", limit=20)

        from polylogue.surfaces.payloads import MessageQueryRowPayload

        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, MessageQueryRowPayload)
        assert row.message_id == "chatgpt-export:ext-message-numeric:m-hit"

    def test_session_scoped_message_predicate_executes_against_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .git_repository_url("polylogue")
            .title("session scoped hit")
            .add_message("m-hit", role="assistant", text="the timeout happened")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .git_repository_url("sinex")
            .title("session scoped miss")
            .add_message("m-miss", role="assistant", text="the timeout happened")
            .save()
        )

        spec = compile_expression("exists message(session.repo:polylogue AND role:assistant AND text:timeout)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_terminal_message_source_accepts_session_scoped_predicate(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("message source hit")
            .add_message("m-hit", role="assistant", text="the timeout happened")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("message source miss")
            .add_message("m-miss", role="assistant", text="the timeout happened")
            .save()
        )

        source = parse_unit_source_expression("messages where session.origin:claude-code-session AND role:assistant")
        assert source is not None
        field_predicate = source.predicate.children[0] if isinstance(source.predicate, QueryBoolPredicate) else None
        assert isinstance(field_predicate, QueryFieldPredicate)
        assert field_predicate.field == "session.origin"
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.session_id, row.message_id) for row in rows] == [
            ("claude-code-session:ext-hit", "claude-code-session:ext-hit:m-hit")
        ]

    def test_session_to_message_pipeline_executes_terminal_rows(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("pipeline hit")
            .add_message("m-hit-user", role="user", text="please inspect")
            .add_message("m-hit-assistant", role="assistant", text="the pipeline answer")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-repo")
            .provider("claude-code")
            .git_repository_url("sinex")
            .title("pipeline wrong repo")
            .add_message("m-wrong-repo", role="assistant", text="the pipeline answer")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-origin")
            .provider("chatgpt")
            .git_repository_url("polylogue")
            .title("pipeline wrong origin")
            .add_message("m-wrong-origin", role="assistant", text="the pipeline answer")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where repo:polylogue AND origin:claude-code-session | messages where role:assistant"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.text) for row in rows] == [
            (
                "claude-code-session:ext-hit",
                "claude-code-session:ext-hit:m-hit-assistant",
                "the pipeline answer",
            )
        ]

    def test_session_to_message_pipeline_fts_stage_executes_against_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("fts pipeline hit")
            .add_message(
                "m-fts",
                role="assistant",
                text="diagnostic",
                blocks=[{"type": "text", "text": "timeout failure in lowerer"}],
            )
            .add_message("m-selected", role="user", text="selected terminal row")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("fts pipeline miss")
            .add_message(
                "m-fts",
                role="assistant",
                text="diagnostic",
                blocks=[{"type": "text", "text": "ordinary lowerer"}],
            )
            .add_message("m-selected", role="user", text="wrong terminal row")
            .save()
        )

        source = parse_unit_source_expression('sessions where ~"timeout failure" | messages where role:user')
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.text) for row in rows] == [
            (
                "chatgpt-export:ext-hit",
                "chatgpt-export:ext-hit:m-selected",
                "selected terminal row",
            )
        ]

    def test_session_to_message_pipeline_exists_stage_executes_against_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("exists pipeline hit")
            .add_message(
                "m-code",
                role="assistant",
                text="diagnostic",
                blocks=[{"type": "code", "text": "pipeline_sentinel()"}],
            )
            .add_message("m-selected", role="user", text="selected terminal row")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("exists pipeline miss")
            .add_message("m-code", role="assistant", text="no code here")
            .add_message("m-selected", role="user", text="wrong terminal row")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where exists block(type:code AND text:pipeline_sentinel) | messages where role:user"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.text) for row in rows] == [
            (
                "chatgpt-export:ext-hit",
                "chatgpt-export:ext-hit:m-selected",
                "selected terminal row",
            )
        ]

    def test_session_to_message_pipeline_sequence_stage_executes_against_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("sequence pipeline hit")
            .add_message(
                "m-edit",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-edit",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-shell",
                role="assistant",
                text="shell",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "tool-shell",
                        "input": {"command": "pytest"},
                        "semantic_type": "shell",
                    }
                ],
            )
            .add_message("m-selected", role="user", text="selected terminal row")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("claude-code")
            .title("sequence pipeline miss")
            .add_message(
                "m-shell",
                role="assistant",
                text="shell",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "tool-shell-miss",
                        "input": {"command": "pytest"},
                        "semantic_type": "shell",
                    }
                ],
            )
            .add_message(
                "m-edit",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-edit-miss",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message("m-selected", role="user", text="wrong terminal row")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where seq(action:file_edit -> action:shell) | messages where role:user"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_messages(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.text) for row in rows] == [
            (
                "claude-code-session:ext-hit",
                "claude-code-session:ext-hit:m-selected",
                "selected terminal row",
            )
        ]

    def test_session_to_message_pipeline_limit_offset_executes_terminal_window(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("pipeline window")
            .add_message("m-1", role="assistant", text="first")
            .add_message("m-2", role="assistant", text="second")
            .add_message("m-3", role="assistant", text="third")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where role:assistant | limit 1 | offset 1"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="pipeline-window", limit=20)

        assert envelope.limit == 1
        assert envelope.offset == 0
        assert envelope.next_offset == 1
        assert envelope.pipeline_stages == (
            _session_scope_stage("repo", "polylogue"),
            {"kind": "limit", "value": 1},
            {"kind": "offset", "value": 1},
            {"kind": "terminal", "action": "rows"},
        )
        assert envelope.pipeline is not None
        assert envelope.pipeline["stages"] == list(envelope.pipeline_stages)
        assert envelope.pipeline["result"] == {"limit": 1, "offset": 1}
        assert len(envelope.items) == 1
        row = envelope.items[0]
        from polylogue.surfaces.payloads import MessageQueryRowPayload

        assert isinstance(row, MessageQueryRowPayload)
        assert row.message_id == "claude-code-session:ext-hit:m-2"
        assert row.text == "second"

        with ArchiveStore.open_existing(index_db.parent) as archive:
            next_page = query_unit_rows(archive, source, query="pipeline-window", limit=20, offset=envelope.next_offset)

        assert next_page.offset == 1
        assert next_page.next_offset is None
        assert len(next_page.items) == 1
        next_row = next_page.items[0]
        assert isinstance(next_row, MessageQueryRowPayload)
        assert next_row.message_id == "claude-code-session:ext-hit:m-3"
        assert next_row.text == "third"

    def test_session_to_message_pipeline_lineage_executes_against_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import MessageQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "root")
            .provider("chatgpt")
            .title("lineage root")
            .add_message("m-root", role="user", text="root message")
            .save()
        )
        (
            SessionBuilder(index_db, "child")
            .provider("chatgpt")
            .title("lineage child")
            .parent_session("ext-root")
            .add_message("m-child", role="user", text="child message")
            .save()
        )
        (
            SessionBuilder(index_db, "other")
            .provider("chatgpt")
            .title("lineage other")
            .add_message("m-other", role="user", text="other message")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where lineage:id:chatgpt-export:ext-child | messages where role:user | sort by time asc"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="pipeline-lineage", limit=20)

        assert envelope.pipeline_stages == (
            {
                "kind": "session_scope",
                "predicate": {
                    "kind": "lineage",
                    "seed_session_id": "chatgpt-export:ext-child",
                    "unit": "session",
                },
            },
            {"kind": "sort", "sort": {"field": "time", "direction": "asc"}},
            {"kind": "terminal", "action": "rows"},
        )
        rows = [row for row in envelope.items if isinstance(row, MessageQueryRowPayload)]
        assert [row.session_id for row in rows] == [
            "chatgpt-export:ext-root",
            "chatgpt-export:ext-child",
        ]
        assert [row.text for row in rows] == ["root message", "child message"]

    def test_cli_json_reports_terminal_pipeline_stages(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("pipeline cli")
            .add_message("m-1", role="assistant", text="first")
            .add_message("m-2", role="assistant", text="second")
            .save()
        )

        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "--format",
                "json",
                "find",
                "sessions where repo:polylogue | messages where role:assistant | limit 1 | offset 1",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["pipeline_stages"] == [
            _session_scope_stage("repo", "polylogue"),
            {"kind": "limit", "value": 1},
            {"kind": "offset", "value": 1},
            {"kind": "terminal", "action": "rows"},
        ]
        assert [item["text"] for item in payload["items"]] == ["second"]

    def test_session_to_message_pipeline_group_by_count_executes_exact_counts(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("pipeline aggregate")
            .add_message("m-user", role="user", text="aggregate needle")
            .add_message("m-assistant-1", role="assistant", text="aggregate needle one")
            .add_message("m-assistant-2", role="assistant", text="aggregate needle two")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-repo")
            .provider("claude-code")
            .git_repository_url("sinex")
            .title("aggregate wrong repo")
            .add_message("m-wrong", role="assistant", text="aggregate needle")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where text:aggregate | group by role | count"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="pipeline-aggregate", limit=20)

        assert isinstance(envelope, QueryUnitAggregateEnvelope)
        assert envelope.mode == "query-unit-aggregate"
        assert envelope.pipeline_stages == (
            _session_scope_stage("repo", "polylogue"),
            {"kind": "group", "field": "role"},
            {"kind": "count", "metric": "count"},
            {"kind": "terminal", "action": "count"},
        )
        assert envelope.pipeline is not None
        assert envelope.pipeline["stages"] == list(envelope.pipeline_stages)
        assert envelope.pipeline["result"] == {"group_by": "role", "aggregate": "count"}
        assert [(row.group_by, row.group_key, row.count) for row in envelope.items] == [
            ("role", "assistant", 2),
            ("role", "user", 1),
        ]

    def test_terminal_aggregate_sort_count_asc_executes_before_limit(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("aggregate sort")
            .add_message("m-user", role="user", text="aggregate sort")
            .add_message("m-system", role="system", text="aggregate sort")
            .add_message("m-assistant-1", role="assistant", text="aggregate sort")
            .add_message("m-assistant-2", role="assistant", text="aggregate sort")
            .save()
        )

        source = parse_unit_source_expression(
            "messages where text:aggregate | group by role | count | sort by count asc | limit 2"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="pipeline-aggregate-sort", limit=20)

        assert isinstance(envelope, QueryUnitAggregateEnvelope)
        assert envelope.pipeline_stages == (
            {"kind": "group", "field": "role"},
            {"kind": "count", "metric": "count"},
            {"kind": "sort", "sort": {"field": "count", "direction": "asc"}},
            {"kind": "limit", "value": 2},
            {"kind": "terminal", "action": "count"},
        )
        assert [(row.group_key, row.count) for row in envelope.items] == [
            ("system", 1),
            ("user", 1),
        ]
        assert envelope.next_offset == 2

    def test_cli_json_reports_terminal_aggregate_counts(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("aggregate cli")
            .add_message("m-user", role="user", text="aggregate cli")
            .add_message("m-assistant", role="assistant", text="aggregate cli")
            .save()
        )

        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "--format",
                "json",
                "find",
                "sessions where repo:polylogue | messages where text:aggregate | group by role | count",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["mode"] == "query-unit-aggregate"
        assert payload["pipeline_stages"] == [
            _session_scope_stage("repo", "polylogue"),
            {"kind": "group", "field": "role"},
            {"kind": "count", "metric": "count"},
            {"kind": "terminal", "action": "count"},
        ]
        assert sorted((item["group_key"], item["count"]) for item in payload["items"]) == [
            ("assistant", 1),
            ("user", 1),
        ]

    def test_observed_event_tool_outcomes_are_terminal_aggregate_query_units(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows, query_unit_session_filters
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
        from tests.infra.storage_records import SessionBuilder

        # Observed events are source-derived (polylogue-dab): a 'tool_finished'
        # event exists only where a real tool_use block joins a tool_result
        # block by tool_id, with handler_kind/status derived from tool_name
        # and tool_result_is_error -- seed real blocks, not a materialized row.
        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "tool-events")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("tool events")
            .add_message("m-user", role="user", text="tool events")
            .add_message(
                "m-assistant",
                role="assistant",
                text="running tools",
                blocks=[
                    {"type": "tool_use", "tool_name": "mcp__serena__find_symbol", "tool_id": "t-serena-ok"},
                    {"type": "tool_use", "tool_name": "mcp__serena__find_symbol", "tool_id": "t-serena-failed"},
                    {"type": "tool_use", "tool_name": "Bash", "tool_id": "t-bash-ok"},
                ],
            )
            .add_message(
                "m-tool-results",
                role="tool",
                text="",
                blocks=[
                    {"type": "tool_result", "tool_id": "t-serena-ok", "tool_result_is_error": 0, "text": "serena ok"},
                    {
                        "type": "tool_result",
                        "tool_id": "t-serena-failed",
                        "tool_result_is_error": 1,
                        "text": "serena failed",
                    },
                    {"type": "tool_result", "tool_id": "t-bash-ok", "tool_result_is_error": 0, "text": "bash ok"},
                ],
            )
            .save()
        )

        source = parse_unit_source_expression(
            "observed-events where kind:tool_finished AND handler:mcp | group by status | count"
        )
        assert source is not None
        statements: list[str] = []
        with ArchiveStore.open_existing(index_db.parent) as archive:
            archive._conn.set_trace_callback(statements.append)
            envelope = query_unit_rows(
                archive,
                source,
                query="observed-event-aggregate",
                limit=20,
                session_filters=query_unit_session_filters(),
            )
            archive._conn.set_trace_callback(None)

        assert isinstance(envelope, QueryUnitAggregateEnvelope)
        assert envelope.pipeline_stages == (
            {"kind": "group", "field": "status"},
            {"kind": "count", "metric": "count"},
            {"kind": "terminal", "action": "count"},
        )
        assert sorted((row.group_by, row.group_key, row.count) for row in envelope.items) == [
            ("status", "failed", 1),
            ("status", "ok", 1),
        ]
        aggregate_statement = next(statement for statement in statements if "GROUP BY group_key" in statement)
        assert "FROM observed_events e" in aggregate_statement
        assert "JOIN sessions" not in aggregate_statement

    def test_unknown_terminal_unit_does_not_fall_through_to_block_rows(self) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        source = QueryUnitSource(
            unit=cast(Any, "bundle"),
            predicate=QueryFieldPredicate(field="text", values=("needle",)),
        )

        with pytest.raises(ValueError, match="Unsupported terminal query unit: bundle"):
            query_unit_rows(cast(ArchiveStore, object()), source, query="bundles where text:needle", limit=10)

    def test_pipeline_carries_typed_rows_terminal_node(self) -> None:
        from polylogue.archive.query.expression import QueryUnitTerminalStage

        source = parse_unit_source_expression("messages where role:assistant | limit 2")
        assert source is not None
        pipeline = source.pipeline
        assert pipeline.terminal == QueryUnitTerminalStage(action="rows")
        stages = cast(list[dict[str, object]], pipeline.to_payload()["stages"])
        assert stages[-1] == {"kind": "terminal", "action": "rows"}

    def test_pipeline_carries_typed_count_terminal_node(self) -> None:
        from polylogue.archive.query.expression import QueryUnitTerminalStage

        source = parse_unit_source_expression("messages where role:assistant | group by role | count")
        assert source is not None
        pipeline = source.pipeline
        assert pipeline.terminal == QueryUnitTerminalStage(action="count")
        stages = cast(list[dict[str, object]], pipeline.to_payload()["stages"])
        assert stages[-1] == {"kind": "terminal", "action": "count"}

    def test_single_executor_runs_select_shape_terminal_chain(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import MessageQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("terminal chain")
            .add_message("m-old", role="assistant", text="old", timestamp="2026-01-01T00:00:00+00:00")
            .add_message("m-new", role="assistant", text="new", timestamp="2026-01-02T00:00:00+00:00")
            .save()
        )

        # Full pipeline: session scope -> unit filter -> shape (sort) -> terminal rows.
        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where role:assistant | sort by time desc | limit 1"
        )
        assert source is not None
        assert source.pipeline.terminal.action == "rows"
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="terminal-chain", limit=20)

        # One executor ran the whole select -> shape -> terminal chain: the page
        # carries the terminal node as the final pipeline stage and the rows are
        # shaped/sorted by it.
        assert envelope.pipeline_stages[-1] == {"kind": "terminal", "action": "rows"}
        assert envelope.limit == 1
        rows = [row for row in envelope.items if isinstance(row, MessageQueryRowPayload)]
        assert [row.text for row in rows] == ["new"]

    def test_unsupported_terminal_action_raises_typed_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.archive.query import unit_results
        from polylogue.archive.query.metadata import query_unit_descriptor
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        source = parse_unit_source_expression("messages where role:assistant")
        assert source is not None
        descriptor = query_unit_descriptor("message")
        assert descriptor is not None

        # Simulate a terminal action with no registered executor (e.g. a future
        # `read`/`analyze` action not yet wired). The dispatcher must fail typed
        # and narrow, never silently broaden to a different terminal.
        monkeypatch.delitem(unit_results.TERMINAL_ACTION_EXECUTORS, "rows")
        with pytest.raises(unit_results.UnsupportedTerminalActionError, match="unsupported terminal action 'rows'"):
            unit_results._build_sql_envelope(
                cast(ArchiveStore, object()),
                source,
                descriptor,
                query="messages where role:assistant",
                limit=10,
                offset=0,
                caller_offset=0,
                fetch_limit=11,
                session_filters=None,
                execution_context=None,
            )

    def test_terminal_source_pipeline_sort_desc_executes_before_limit(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("direct pipeline sorted")
            .add_message("m-old", role="assistant", text="old", timestamp="2026-01-01T00:00:00+00:00")
            .add_message("m-new", role="assistant", text="new", timestamp="2026-01-02T00:00:00+00:00")
            .save()
        )

        source = parse_unit_source_expression("messages where role:assistant | sort by time desc | limit 1")
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="direct-pipeline-sort", limit=20)

        assert envelope.limit == 1
        assert envelope.next_offset == 1
        assert len(envelope.items) == 1
        row = envelope.items[0]
        from polylogue.surfaces.payloads import MessageQueryRowPayload

        assert isinstance(row, MessageQueryRowPayload)
        assert row.message_id == "claude-code-session:ext-hit:m-new"
        assert row.text == "new"

    def test_session_to_message_pipeline_sort_desc_executes_before_limit(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("pipeline sorted")
            .add_message("m-old", role="assistant", text="old", timestamp="2026-01-01T00:00:00+00:00")
            .add_message("m-new", role="assistant", text="new", timestamp="2026-01-02T00:00:00+00:00")
            .save()
        )

        source = parse_unit_source_expression(
            "sessions where repo:polylogue | messages where role:assistant | sort by time desc | limit 1"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="pipeline-sort", limit=20)

        assert envelope.limit == 1
        assert envelope.next_offset == 1
        assert len(envelope.items) == 1
        row = envelope.items[0]
        from polylogue.surfaces.payloads import MessageQueryRowPayload

        assert isinstance(row, MessageQueryRowPayload)
        assert row.message_id == "claude-code-session:ext-hit:m-new"
        assert row.text == "new"

    def test_exists_action_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("action hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("claude-code")
            .title("action miss")
            .add_message(
                "m-miss",
                role="assistant",
                text="read file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_id": "tool-2",
                        "input": {"file_path": "polylogue/README.md"},
                        "semantic_type": "file_read",
                    }
                ],
            )
            .save()
        )

        spec = compile_expression("exists action(action:file_edit AND path:archive/query)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-hit"]

    def test_terminal_action_source_returns_action_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("action hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        source = parse_unit_source_expression("actions where action:file_edit AND path:archive/query")
        assert source is not None
        assert source.unit == "action"
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_actions(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.semantic_type, row.tool_path) for row in rows] == [
            (
                "claude-code-session:ext-hit",
                "claude-code-session:ext-hit:m-hit",
                "file_edit",
                "polylogue/archive/query/expression.py",
            )
        ]

    def test_codex_exec_freeform_arguments_are_queryable_as_commands(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.sources.parsers.codex import parse as parse_codex
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        script = 'const result = await tools.exec_command({"cmd":"polylogue find repo:polylogue"});'
        parsed = parse_codex(
            [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "dogfood-command-query",
                        "cwd": "/realm/project/polylogue",
                        "git": {"repository_url": "https://example.test/polylogue.git"},
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "call-exec",
                        "name": "exec",
                        "arguments": script,
                    },
                },
            ],
            "dogfood-command-query",
        )

        archive_root = workspace_env["archive_root"]
        with ArchiveStore(archive_root) as archive:
            session_id = archive.write_parsed(parsed)

        source = parse_unit_source_expression("actions where command:polylogue")
        assert source is not None
        with ArchiveStore.open_existing(archive_root) as archive:
            rows = archive.query_actions(source.predicate, limit=100)

        assert [(row.session_id, row.semantic_type, row.tool_command) for row in rows] == [
            (session_id, "shell", script)
        ]

    def test_legacy_codex_execution_payloads_are_queryable_without_rewrite(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        fixtures = (
            ("legacy-arguments", "exec", {"arguments": "polylogue find repo:polylogue"}),
            ("legacy-cmd", "exec_command", {"cmd": "polylogue status"}),
            ("unrelated-arguments", "Task", {"arguments": "polylogue should remain ordinary task input"}),
        )
        for session_id, tool_name, tool_input in fixtures:
            (
                SessionBuilder(index_db, session_id)
                .provider("codex")
                .add_message(
                    f"message-{session_id}",
                    role="assistant",
                    text="tool invocation",
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": tool_name,
                            "tool_id": f"tool-{session_id}",
                            "input": tool_input,
                        }
                    ],
                )
                .save()
            )

        action_source = parse_unit_source_expression("actions where command:polylogue")
        block_source = parse_unit_source_expression("blocks where command:polylogue")
        assert action_source is not None
        assert block_source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            action_rows = archive.query_actions(action_source.predicate, limit=100)
            block_rows = archive.query_blocks(block_source.predicate, limit=100)

        assert sorted((row.session_id, row.tool_name, row.tool_command) for row in action_rows) == [
            ("codex-session:ext-legacy-arguments", "exec", "polylogue find repo:polylogue"),
            ("codex-session:ext-legacy-cmd", "exec_command", "polylogue status"),
        ]
        assert sorted((row.session_id, row.tool_name, row.tool_command) for row in block_rows) == [
            ("codex-session:ext-legacy-arguments", "exec", "polylogue find repo:polylogue"),
            ("codex-session:ext-legacy-cmd", "exec_command", "polylogue status"),
        ]

    def test_file_unit_source_parses_terminal_and_exists_forms(self) -> None:
        terminal = parse_unit_source_expression("files where action:file_edit AND path:archive/query")
        assert terminal is not None
        assert terminal.unit == "file"

        spec = compile_expression("exists file(action:file_edit AND path:archive/query)")
        assert spec.boolean_predicate is not None

    def test_terminal_file_source_returns_distinct_path_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("file hit")
            .add_message(
                "m-edit-1",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-edit-2",
                role="assistant",
                text="edited file again",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-2",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-read",
                role="assistant",
                text="read other file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_id": "tool-3",
                        "input": {"file_path": "README.md"},
                        "semantic_type": "file_read",
                    }
                ],
            )
            .save()
        )

        source = parse_unit_source_expression("files where action:file_edit AND path:archive/query")
        assert source is not None
        assert source.unit == "file"
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_files(source.predicate, limit=100)

        assert [(row.session_id, row.path, row.action_count) for row in rows] == [
            ("claude-code-session:ext-hit", "polylogue/archive/query/expression.py", 2)
        ]
        assert rows[0].first_tool_use_block_id == "claude-code-session:ext-hit:m-edit-1:0"

    def test_exists_file_source_filters_sessions(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("file exists hit")
            .add_message(
                "m-edit",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-hit",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("claude-code")
            .title("file exists miss")
            .add_message(
                "m-read",
                role="assistant",
                text="read file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_id": "tool-miss",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_read",
                    }
                ],
            )
            .save()
        )

        spec = compile_expression("exists file(action:file_edit AND path:archive/query)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-hit"]

    def test_file_unit_aggregate_counts_distinct_paths(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("file aggregate")
            .add_message(
                "m-edit-1",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-edit-2",
                role="assistant",
                text="edited same file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-2",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-edit-3",
                role="assistant",
                text="edited other file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-3",
                        "input": {"file_path": "polylogue/cli/archive_query.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        source = parse_unit_source_expression(
            "files where session.repo:polylogue AND action:file_edit | group by path | count | sort by key asc"
        )
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            envelope = query_unit_rows(archive, source, query="file-aggregate", limit=20)

        assert isinstance(envelope, QueryUnitAggregateEnvelope)
        assert [(row.group_key, row.count) for row in envelope.items] == [
            ("polylogue/archive/query/expression.py", 1),
            ("polylogue/cli/archive_query.py", 1),
        ]

    def test_cli_json_reports_file_unit_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("file cli")
            .add_message(
                "m-edit",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-hit",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "--format",
                "json",
                "find",
                "files where action:file_edit AND path:archive/query",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["unit"] == "file"
        assert payload["items"][0]["path"] == "polylogue/archive/query/expression.py"

    def test_cli_json_attaches_evidence_units_to_session_results(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "attached-evidence")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("attached evidence")
            .add_message("m-user", role="user", text="attach evidence units")
            .add_message(
                "m-assistant",
                role="assistant",
                text="using edit tool",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-attach",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "--format",
                "json",
                "find",
                "sessions where title:attached with messages(message_id,role), "
                "actions(tool_name,semantic_type,is_error), files(path)",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        attached = payload["items"][0]["attached_units"]
        assert {tuple(row) for row in attached["message"]} == {("message_id", "role")}
        assert {row["role"] for row in attached["message"]} == {"user", "assistant"}
        assert attached["action"][0] == {"tool_name": "Edit", "semantic_type": "file_edit", "is_error": None}
        assert attached["file"][0] == {"path": "polylogue/archive/query/expression.py"}

    def test_query_action_read_accepts_shell_quoted_terminal_action_source(
        self, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("action read hit")
            .add_message(
                "m-edit",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-hit",
                        "input": {"file_path": "polylogue/cli/query_verbs.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "find",
                "actions where action:file_edit AND path:query_verbs",
                "then",
                "read",
                "--view",
                "messages",
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["session_id"] == "claude-code-session:ext-hit"
        assert payload["messages"][0]["id"] == "claude-code-session:ext-hit:m-edit"

    def test_terminal_action_source_filters_by_row_time(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("action time hit")
            .add_message(
                "m-old",
                role="assistant",
                text="old edit",
                timestamp="2026-01-01T00:00:00+00:00",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-old",
                        "input": {"file_path": "polylogue/archive/old.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-boundary",
                role="assistant",
                text="boundary edit",
                timestamp="2026-01-02T00:00:00+00:00",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-boundary",
                        "input": {"file_path": "polylogue/archive/boundary.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-new",
                role="assistant",
                text="new edit",
                timestamp="2026-01-03T00:00:00+00:00",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-new",
                        "input": {"file_path": "polylogue/archive/new.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        source = parse_unit_source_expression("actions where time > 2026-01-02T00:00:00+00:00")
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_actions(source.predicate, limit=100)

        assert [(row.message_id, row.tool_path) for row in rows] == [
            ("claude-code-session:ext-hit:m-new", "polylogue/archive/new.py")
        ]
        assert rows[0].occurred_at_ms is not None

    def test_terminal_action_source_accepts_session_scoped_predicate(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        for session_id, repo_name in [("hit", "polylogue"), ("miss", "sinex")]:
            (
                SessionBuilder(index_db, session_id)
                .provider("claude-code")
                .git_repository_url(repo_name)
                .title(f"action {session_id}")
                .add_message(
                    f"m-{session_id}",
                    role="assistant",
                    text="edited file",
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Edit",
                            "tool_id": f"tool-{session_id}",
                            "input": {"file_path": "polylogue/archive/query/expression.py"},
                            "semantic_type": "file_edit",
                        }
                    ],
                )
                .save()
            )

        source = parse_unit_source_expression("actions where session.repo:polylogue AND action:file_edit")
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_actions(source.predicate, limit=100)

        assert [(row.session_id, row.semantic_type) for row in rows] == [("claude-code-session:ext-hit", "file_edit")]

    def test_exists_action_path_normalizes_backslashes(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "windows-path")
            .provider("claude-code")
            .title("windows path")
            .add_message(
                "m-path",
                role="assistant",
                text="edited windows path",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-path",
                        "input": {"file_path": r"polylogue\archive\query\expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        spec = compile_expression("exists action(path:archive/query)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-windows-path"]

    def test_exists_block_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("block hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="assistant response",
                blocks=[{"type": "code", "text": "def query_timeout_guard(): pass"}],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("block miss")
            .add_message(
                "m-miss",
                role="assistant",
                text="assistant response",
                blocks=[{"type": "text", "text": "timeout appears outside code"}],
            )
            .save()
        )

        spec = compile_expression("exists block(type:code AND text:timeout)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_terminal_block_source_returns_block_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("block hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="assistant response",
                blocks=[{"type": "code", "text": "def query_timeout_guard(): pass"}],
            )
            .save()
        )

        source = parse_unit_source_expression("blocks where type:code AND text:timeout")
        assert source is not None
        assert source.unit == "block"
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_blocks(source.predicate, limit=100)

        assert [(row.session_id, row.message_id, row.block_type, row.text) for row in rows] == [
            (
                "chatgpt-export:ext-hit",
                "chatgpt-export:ext-hit:m-hit",
                "code",
                "def query_timeout_guard(): pass",
            )
        ]

    def test_terminal_block_source_accepts_session_scoped_predicate(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("block hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="assistant response",
                blocks=[{"type": "code", "text": "def query_timeout_guard(): pass"}],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("claude-code")
            .title("block miss")
            .add_message(
                "m-miss",
                role="assistant",
                text="assistant response",
                blocks=[{"type": "code", "text": "def query_timeout_guard(): pass"}],
            )
            .save()
        )

        source = parse_unit_source_expression("blocks where session.origin:chatgpt-export AND type:code")
        assert source is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.query_blocks(source.predicate, limit=100)

        assert [(row.session_id, row.block_type) for row in rows] == [("chatgpt-export:ext-hit", "code")]

    def test_terminal_action_source_exposes_followup_class_rows_and_aggregates(
        self, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ActionQueryRowPayload, QueryUnitAggregateEnvelope
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "followups")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("failed action followups")
            .add_message(
                "m-ack-tool",
                role="assistant",
                text="Ran the focused check.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "ack-tool",
                        "name": "Bash",
                        "tool_input": {"command": "pytest ack"},
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "ack-tool",
                        "text": "1 failed",
                        "tool_result_is_error": 1,
                        "tool_result_exit_code": 1,
                    },
                ],
            )
            .add_message(
                "m-ack-followup", role="assistant", text="The command failed with exit code 1; I will inspect it."
            )
            .add_message(
                "m-silent-tool",
                role="assistant",
                text="Ran another check.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "silent-tool",
                        "name": "Bash",
                        "tool_input": {"command": "pytest silent"},
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "silent-tool",
                        "text": "1 failed",
                        "tool_result_is_error": 1,
                        "tool_result_exit_code": 1,
                    },
                ],
            )
            .add_message(
                "m-silent-followup",
                role="assistant",
                text="I will inspect the generated fixture rows and continue with the next patch.",
            )
            .add_message(
                "m-wordless-tool",
                role="assistant",
                text="Ran a third check.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "wordless-tool",
                        "name": "Bash",
                        "tool_input": {"command": "pytest wordless"},
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "wordless-tool",
                        "text": "1 failed",
                        "tool_result_is_error": 1,
                        "tool_result_exit_code": 1,
                    },
                ],
            )
            .add_message(
                "m-wordless-followup",
                role="assistant",
                text="",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "read-after-failure",
                        "name": "Read",
                        "tool_input": {"file_path": "polylogue/archive/query/unit_results.py"},
                    }
                ],
            )
            .save()
        )

        query = "actions where followup_class:silent_proceed"
        source = parse_unit_source_expression(query)
        assert source is not None
        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(archive, source, query=query, limit=10)

        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ActionQueryRowPayload)
        assert row.tool_command == "pytest silent"
        assert row.is_error == 1
        assert row.exit_code == 1
        assert row.followup_class == "silent_proceed"
        assert row.followup_message_ref == "message:codex-session:ext-followups:m-silent-followup"

        aggregate_query = "actions where is_error:true | group by followup_class | count"
        aggregate_source = parse_unit_source_expression(aggregate_query)
        assert aggregate_source is not None
        with ArchiveStore.open_existing(archive_root) as archive:
            aggregate = query_unit_rows(archive, aggregate_source, query=aggregate_query, limit=10)

        assert isinstance(aggregate, QueryUnitAggregateEnvelope)
        assert sorted((item.group_key, item.count) for item in aggregate.items) == [
            ("acknowledged", 1),
            ("silent_proceed", 1),
            ("wordless_continuation", 1),
        ]

    def test_exists_assertion_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        import sqlite3

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("codex")
            .title("assertion hit")
            .add_message("m-hit", role="assistant", text="target session")
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("codex")
            .title("assertion miss")
            .add_message("m-miss", role="assistant", text="target session")
            .save()
        )
        user_db = archive_root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        with sqlite3.connect(user_db) as conn:
            upsert_assertion(
                conn,
                assertion_id="decision-hit",
                target_ref="session:codex-session:ext-hit",
                kind=AssertionKind.DECISION,
                body_text="Review the query unit before merge.",
                author_ref="agent:codex",
                # author_kind="user": these seed rows test status/filter
                # logic, not the write-side trust invariant (37t.15) -- a
                # non-user author would land as a candidate regardless of the
                # requested status here.
                author_kind="user",
                status="active",
                now_ms=2000,
            )
            upsert_assertion(
                conn,
                assertion_id="decision-miss",
                target_ref="session:codex-session:ext-miss",
                kind=AssertionKind.DECISION,
                body_text="Review unrelated branch.",
                author_ref="agent:codex",
                author_kind="user",
                status="deleted",
                now_ms=1000,
            )
            conn.commit()

        spec = compile_expression("exists assertion(kind:decision AND status:active AND text:query)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(archive_root) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["codex-session:ext-hit"]

    def test_terminal_assertion_source_returns_assertion_rows(self, workspace_env: dict[str, Path]) -> None:
        import sqlite3

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("assertion hit")
            .add_message("m-hit", role="assistant", text="target session")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-repo")
            .provider("codex")
            .git_repository_url("sinex")
            .title("assertion wrong repo")
            .add_message("m-wrong-repo", role="assistant", text="target session")
            .save()
        )
        user_db = archive_root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        with sqlite3.connect(user_db) as conn:
            upsert_assertion(
                conn,
                assertion_id="decision-hit",
                target_ref="session:codex-session:ext-hit",
                kind=AssertionKind.DECISION,
                key="next-step",
                value={"priority": 1},
                body_text="Review the query unit before merge.",
                author_ref="agent:codex",
                # author_kind="user": these seed rows test status/filter
                # logic, not the write-side trust invariant (37t.15).
                author_kind="user",
                evidence_refs=["session:codex-session:ext-hit#message:m-hit"],
                status="active",
                visibility="public",
                staleness={"policy": "manual"},
                context_policy={"inject": False},
                now_ms=2000,
            )
            upsert_assertion(
                conn,
                assertion_id="decision-wrong-repo",
                target_ref="session:codex-session:ext-wrong-repo",
                kind=AssertionKind.DECISION,
                body_text="Review the other archive.",
                author_ref="agent:codex",
                author_kind="user",
                status="active",
                now_ms=1000,
            )
            conn.commit()

        source = parse_unit_source_expression(
            "assertions where session.repo:polylogue AND kind:decision AND status:active AND text:query"
        )
        assert source is not None
        assert source.unit == "assertion"
        with ArchiveStore.open_existing(archive_root) as archive:
            rows = archive.query_assertions(source.predicate, limit=100)

        assert [(row.assertion_id, row.target_ref, row.kind, row.body_text) for row in rows] == [
            (
                "decision-hit",
                "session:codex-session:ext-hit",
                "decision",
                "Review the query unit before merge.",
            )
        ]
        assert rows[0].value == {"priority": 1}
        assert rows[0].evidence_refs == ("session:codex-session:ext-hit#message:m-hit",)
        assert rows[0].staleness == {"policy": "manual"}
        assert rows[0].context_policy == {"inject": False}

    def test_terminal_observed_event_source_returns_runtime_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ObservedEventQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        _failed_suite_blocks = [
            {"type": "tool_use", "id": "t1", "name": "Bash", "tool_input": {"command": "pytest tests/unit"}},
            {"type": "tool_result", "tool_id": "t1", "text": "1 failed", "tool_result_exit_code": 1},
        ]
        (
            SessionBuilder(index_db, "hit")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("tests failed")
            .add_message("m-hit", role="assistant", text="Ran the suite.", blocks=_failed_suite_blocks)
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-repo")
            .provider("codex")
            .git_repository_url("sinex")
            .title("tests failed elsewhere")
            .add_message("m-wrong-repo", role="assistant", text="Ran the suite.", blocks=_failed_suite_blocks)
            .save()
        )

        # kind:test_failed (a richer digest-derived event kind) has no
        # source-derived equivalent (polylogue-dab): the CTE only synthesizes
        # 'session_started' and 'tool_finished' directly from sessions/blocks.
        # This test's distinguishing concern -- session.repo scoping -- still
        # applies to the tool_finished kind the model does produce.
        query = "observed-events where session.repo:polylogue AND kind:tool_finished AND delivery_state:observed"
        source = parse_unit_source_expression(query)
        assert source is not None
        _materialize_run_projection(index_db)

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(archive, source, query=query, limit=10)

        assert envelope.unit == "observed-event"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ObservedEventQueryRowPayload)
        assert (row.kind, row.delivery_state, row.session_id) == (
            "tool_finished",
            "observed",
            "codex-session:ext-hit",
        )
        assert row.object_refs == ("tool-call:codex-session:ext-hit:t1",)

    def test_terminal_observed_event_tool_finished_reads_blocks_without_materialization(
        self, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ObservedEventQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "tool-finished")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("source-derived tool event")
            .add_message(
                "m-tool",
                role="assistant",
                text="Inspected symbols.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "serena-1",
                        "name": "mcp__serena__find_symbol",
                        "tool_input": {"name_path": "ArchiveStore/query_observed_events"},
                    },
                    {"type": "tool_result", "tool_id": "serena-1", "text": "found"},
                ],
            )
            .save()
        )

        query = "observed-events where kind:tool_finished AND tool:mcp__serena__find_symbol"
        source = parse_unit_source_expression(query)
        assert source is not None

        _assert_run_projection_table_absent(index_db, "session_observed_events")

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(archive, source, query=query, limit=10)

        assert envelope.unit == "observed-event"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ObservedEventQueryRowPayload)
        assert row.kind == "tool_finished"
        assert row.delivery_state == "observed"
        assert row.session_id == "claude-code-session:ext-tool-finished"
        assert row.subject_ref == "message:claude-code-session:ext-tool-finished:m-tool"
        assert row.object_refs == ("tool-call:claude-code-session:ext-tool-finished:serena-1",)
        assert row.evidence_refs == (
            "claude-code-session:ext-tool-finished::claude-code-session:ext-tool-finished:m-tool::0",
            "claude-code-session:ext-tool-finished::claude-code-session:ext-tool-finished:m-tool::1",
        )

    def test_terminal_observed_event_tool_finished_aggregate_reads_blocks_without_materialization(
        self, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "tool-aggregate")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("source-derived aggregate")
            .add_message(
                "m-tool",
                role="assistant",
                text="Ran tools.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "bash-1",
                        "name": "Bash",
                        "tool_input": {"command": "pytest tests/unit"},
                    },
                    {"type": "tool_result", "tool_id": "bash-1", "text": "passed", "tool_result_exit_code": 0},
                    {
                        "type": "tool_use",
                        "id": "bash-2",
                        "name": "Bash",
                        "tool_input": {"command": "pytest tests/integration"},
                    },
                    {"type": "tool_result", "tool_id": "bash-2", "text": "failed", "tool_result_exit_code": 1},
                ],
            )
            .save()
        )

        query = "observed-events where kind:tool_finished AND handler:shell | group by status | count"
        source = parse_unit_source_expression(query)
        assert source is not None

        _assert_run_projection_table_absent(index_db, "session_observed_events")

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(archive, source, query=query, limit=10)

        assert isinstance(envelope, QueryUnitAggregateEnvelope)
        assert sorted((row.group_key, row.count) for row in envelope.items) == [("failed", 1), ("ok", 1)]

    def test_terminal_run_source_returns_runtime_rows(self, workspace_env: dict[str, Path]) -> None:
        """A subagent run row is derived from a real branch_type='subagent' child session.

        Source-derived runs (polylogue-dab) key one row per session, not one
        per Task dispatch, so 'role:subagent' requires a real child session --
        there is no synthesized virtual run from parsing a Task tool_use/
        tool_result pair alone. agent_ref is deliberately coarse
        ('agent:<harness>/subagent'): the old model's per-dispatch
        subagent_type ('Explore') is not reconstructed by the source-derived
        CTE (see run_projection_relations.py's run_relation_sql docstring).
        """
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import RunQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("codex")
            .git_repository_url("polylogue")
            .git_branch("feature/query-runs")
            .working_directories(["/realm/project/polylogue"])
            .title("run projection hit")
            .add_message("m-user", role="user", text="Coordinate the query DSL branch.")
            .add_message(
                "m-subagent",
                role="assistant",
                text="Subagent finished the run-query audit.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "tool-run",
                        "name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "taskId": "task-run",
                            "child_session_id": "codex-session:ext-hit-subagent",
                            "prompt": "Map the remaining run query substrate.",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "tool-run",
                        "text": "Subagent done: run query substrate mapped.\n4 passed in 0.31s",
                    },
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "hit-subagent")
            .provider("codex")
            .parent_session("ext-hit")
            .branch_type("subagent")
            .git_repository_url("polylogue")
            .git_branch("feature/query-runs")
            .working_directories(["/realm/project/polylogue"])
            .title("run-query audit subagent")
            .add_message("m-child", role="assistant", text="Run query substrate mapped.")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-repo-subagent")
            .provider("codex")
            .branch_type("subagent")
            .git_repository_url("sinex")
            .title("subagent in a different repo")
            .add_message("m-wrong", role="assistant", text="Subagent finished elsewhere.")
            .save()
        )

        source = parse_unit_source_expression(
            "runs where session.repo:polylogue AND role:subagent AND status:completed"
        )
        assert source is not None
        _materialize_run_projection(index_db)

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(
                archive,
                source,
                query="runs where session.repo:polylogue AND role:subagent AND status:completed",
                limit=10,
            )

        assert envelope.unit == "run"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, RunQueryRowPayload)
        assert row.session_id == "codex-session:ext-hit-subagent"
        assert row.role == "subagent"
        assert row.status == "completed"
        assert row.harness == "codex"
        assert row.agent_ref == "agent:codex/subagent"
        assert row.parent_run_ref == "run:codex-session:ext-hit"
        assert row.run_ref == "run:codex-session:ext-hit-subagent"
        assert row.git_branch == "feature/query-runs"
        assert row.cwd is None  # run_relation_sql() has no cwd source column post-dab
        assert row.context_snapshot_ref == "context-snapshot:codex-session:ext-hit-subagent:subagent_start"

    def test_terminal_main_run_reads_sessions_without_materialization(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import RunQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "main-run")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .git_branch("feature/main-run-source")
            .title("source main run")
            .add_message("m-run", role="user", text="Inspect the run query source.")
            .save()
        )

        query = "runs where harness:claude-code AND role:main AND branch:feature/main-run-source"
        source = parse_unit_source_expression(query)
        assert source is not None

        _assert_run_projection_table_absent(index_db, "session_runs")

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(archive, source, query=query, limit=10)

        assert envelope.unit == "run"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, RunQueryRowPayload)
        assert row.session_id == "claude-code-session:ext-main-run"
        assert row.run_ref == "run:claude-code-session:ext-main-run"
        assert row.harness == "claude-code"
        assert row.role == "main"
        assert row.status == "completed"
        assert row.confidence == "raw"
        assert row.agent_ref == "agent:claude-code/main"
        assert row.evidence_refs == ("claude-code-session:ext-main-run",)
        assert row.context_snapshot_ref == "context-snapshot:claude-code-session:ext-main-run:session_start"

    def test_terminal_run_sort_by_time_orders_across_sessions(self, workspace_env: dict[str, Path]) -> None:
        """`runs | sort by time` orders by owning-session time, not per-session position.

        Native ids old/mid/new have alphabetical order (mid<new<old) distinct from
        their chronological order, so position-only ordering (the pre-fix bug, which
        interleaved every session's position-0 row and broke ties on run_ref) would
        return them mis-ordered. The fix orders by source_updated_at first.
        """
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import RunQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        for native, stamp in (
            ("old", "2026-01-01T00:00:00+00:00"),
            ("mid", "2026-02-01T00:00:00+00:00"),
            ("new", "2026-03-01T00:00:00+00:00"),
        ):
            (
                SessionBuilder(index_db, native)
                .provider("codex")
                .git_repository_url("polylogue")
                .title(f"run sort {native}")
                .created_at(stamp)
                .updated_at(stamp)
                .add_message(f"m-{native}", role="user", text=f"Session {native} work.")
                .save()
            )
        _materialize_run_projection(index_db)

        def _suffixes(direction: str) -> list[str]:
            source = parse_unit_source_expression(f"runs where session.repo:polylogue | sort by time {direction}")
            assert source is not None
            with ArchiveStore.open_existing(archive_root) as archive:
                envelope = query_unit_rows(
                    archive,
                    source,
                    query=f"runs where session.repo:polylogue | sort by time {direction}",
                    limit=10,
                )
            suffixes: list[str] = []
            for row in envelope.items:
                assert isinstance(row, RunQueryRowPayload)
                suffixes.append(row.session_id.rsplit("-", 1)[-1])
            return suffixes

        assert _suffixes("asc") == ["old", "mid", "new"]
        assert _suffixes("desc") == ["new", "mid", "old"]

    def test_terminal_context_snapshot_source_returns_runtime_rows(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ContextSnapshotQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("context hit")
            .add_message("m-hit", role="user", text="Initial prompt for ext-hit.")
            .add_message("m-subagent", role="assistant", text="Subagent final report")
            .save()
        )
        (
            SessionBuilder(index_db, "wrong-repo")
            .provider("codex")
            .git_repository_url("sinex")
            .title("context elsewhere")
            .add_message("m-wrong-repo", role="user", text="Initial prompt")
            .save()
        )

        source = parse_unit_source_expression(
            "context-snapshots where session.repo:polylogue AND boundary:session_start AND text:ext-hit"
        )
        assert source is not None
        _materialize_run_projection(index_db)

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(
                archive,
                source,
                query=("context-snapshots where session.repo:polylogue AND boundary:session_start AND text:ext-hit"),
                limit=10,
            )

        assert envelope.unit == "context-snapshot"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ContextSnapshotQueryRowPayload)
        assert row.session_id == "codex-session:ext-hit"
        assert row.boundary == "session_start"
        assert row.inheritance_mode == "unknown"
        assert row.snapshot_ref.startswith("context-snapshot:")
        assert row.run_ref.startswith("run:")
        assert row.evidence_refs == ("codex-session:ext-hit",)

        # SQL terminal units now accept the full session-scoped field set.
        assert parse_unit_source_expression("context-snapshots where session.tool:bash AND boundary:session_start")
        assert parse_unit_source_expression("context-snapshots where session.path:polylogue AND boundary:session_start")

    def test_terminal_context_snapshot_reads_sessions_without_materialization(
        self, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ContextSnapshotQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "source-context")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("source context snapshot")
            .add_message("m-context", role="user", text="Initial context prompt")
            .save()
        )

        query = "context-snapshots where session.repo:polylogue AND boundary:session_start AND text:source-context"
        source = parse_unit_source_expression(query)
        assert source is not None

        _assert_run_projection_table_absent(index_db, "session_context_snapshots")

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(archive, source, query=query, limit=10)

        assert envelope.unit == "context-snapshot"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ContextSnapshotQueryRowPayload)
        assert row.session_id == "codex-session:ext-source-context"
        assert row.snapshot_ref == "context-snapshot:codex-session:ext-source-context:session_start"
        assert row.run_ref == "run:codex-session:ext-source-context"
        assert row.boundary == "session_start"
        assert row.inheritance_mode == "unknown"
        assert row.segment_refs == ("session:codex-session:ext-source-context",)
        assert row.evidence_refs == ("codex-session:ext-source-context",)
        assert row.metadata == {"source": "archive-session"}

    def test_sql_terminal_unit_rejects_unbound_session_predicate(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "unbound-context")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("unbound context")
            .add_message("m-unbound", role="user", text="Initial prompt")
            .save()
        )

        source = QueryUnitSource(
            unit="context-snapshot",
            predicate=QueryFieldPredicate(field="session.repo", values=("polylogue",)),
        )
        with ArchiveStore.open_existing(archive_root) as archive:
            with pytest.raises(ValueError, match="unbound session-scoped query field predicate"):
                query_unit_rows(
                    archive,
                    source,
                    query="context-snapshots where session.repo:polylogue",
                    limit=10,
                )

    def test_context_snapshot_session_since_uses_created_at_fallback(self, workspace_env: dict[str, Path]) -> None:
        """Runtime-transform session date filters match normal session timestamp semantics."""
        import sqlite3

        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ContextSnapshotQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "created-only")
            .provider("codex")
            .created_at("2026-01-03T00:00:00+00:00")
            .updated_at("2026-01-03T00:00:00+00:00")
            .title("created only context")
            .add_message("m-created-only", role="user", text="created-only context row")
            .save()
        )
        with sqlite3.connect(index_db) as conn:
            conn.execute(
                "UPDATE sessions SET updated_at_ms = NULL WHERE session_id = ?", ("codex-session:created-only",)
            )

        source = parse_unit_source_expression(
            "context-snapshots where session.since:2026-01-02 AND boundary:session_start AND text:created-only"
        )
        assert source is not None
        _materialize_run_projection(index_db)

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(
                archive,
                source,
                query=(
                    "context-snapshots where session.since:2026-01-02 AND boundary:session_start AND text:created-only"
                ),
                limit=10,
            )

        assert envelope.unit == "context-snapshot"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ContextSnapshotQueryRowPayload)
        assert row.session_id == "codex-session:ext-created-only"

    def test_context_snapshot_session_summary_comparisons(self, workspace_env: dict[str, Path]) -> None:
        """Runtime-transform rows honor scoped session count/date predicates."""
        import sqlite3

        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ContextSnapshotQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "summary-hit")
            .provider("codex")
            .created_at("2026-01-03T00:00:00+00:00")
            .updated_at("2026-01-03T00:00:00+00:00")
            .title("summary comparison hit")
            .add_message("m-hit-1", role="user", text="one two")
            .add_message(
                "m-hit-2",
                role="assistant",
                text="three four",
                blocks=[
                    {"type": "tool_use", "tool_name": "bash", "text": "pytest"},
                    {"type": "thinking", "text": "plan"},
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "summary-control")
            .provider("codex")
            .created_at("2026-01-01T00:00:00+00:00")
            .updated_at("2026-01-01T00:00:00+00:00")
            .title("summary comparison control")
            .add_message("m-control", role="user", text="one two three four five", has_paste=1)
            .save()
        )
        (
            SessionBuilder(index_db, "summary-boundary")
            .provider("codex")
            .created_at("2026-01-02T00:00:00+00:00")
            .updated_at("2026-01-02T00:00:00+00:00")
            .title("summary comparison boundary")
            .add_message("m-boundary-1", role="user", text="one two")
            .add_message(
                "m-boundary-2",
                role="assistant",
                text="three four",
                blocks=[
                    {"type": "tool_use", "tool_name": "bash", "text": "pytest"},
                    {"type": "thinking", "text": "plan"},
                ],
            )
            .save()
        )
        with sqlite3.connect(index_db) as conn:
            conn.execute(
                "UPDATE sessions SET reported_duration_ms = ? WHERE session_id = ?",
                (90_000, "codex-session:ext-summary-hit"),
            )
            conn.execute(
                "UPDATE sessions SET reported_duration_ms = ? WHERE session_id = ?",
                (5_000, "codex-session:ext-summary-control"),
            )
            conn.execute(
                "UPDATE sessions SET reported_duration_ms = ? WHERE session_id = ?",
                (60_000, "codex-session:ext-summary-boundary"),
            )
            conn.commit()

        source = parse_unit_source_expression(
            "context-snapshots where session.messages:>=2 "
            "AND session.words:<=4 "
            "AND session.tool_use_messages:>=1 "
            "AND session.thinking_messages:>=1 "
            "AND session.paste_messages:=0 "
            "AND session.duration_ms:>60000 "
            "AND session.date:>2026-01-02 "
            "AND boundary:session_start"
        )
        assert source is not None
        _materialize_run_projection(index_db)

        with ArchiveStore.open_existing(archive_root) as archive:
            envelope = query_unit_rows(
                archive,
                source,
                query=(
                    "context-snapshots where session.messages:>=2 "
                    "AND session.words:<=4 "
                    "AND session.tool_use_messages:>=1 "
                    "AND session.thinking_messages:>=1 "
                    "AND session.paste_messages:=0 "
                    "AND session.duration_ms:>60000 "
                    "AND session.date:>2026-01-02 "
                    "AND boundary:session_start"
                ),
                limit=10,
            )

        assert envelope.unit == "context-snapshot"
        assert len(envelope.items) == 1
        row = envelope.items[0]
        assert isinstance(row, ContextSnapshotQueryRowPayload)
        assert row.session_id == "codex-session:ext-summary-hit"

    def test_context_snapshot_unit_source_paginates_via_sql_limit_offset(self, workspace_env: dict[str, Path]) -> None:
        """SQL terminal row pagination honours limit/offset over materialized rows."""
        from polylogue.archive.query.unit_results import query_unit_rows
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import ContextSnapshotQueryRowPayload
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        for index in range(3):
            (
                SessionBuilder(index_db, f"page-{index}")
                .provider("codex")
                .title(f"page {index}")
                .add_message(f"m-page-{index}", role="user", text=f"page-marker-{index}")
                .save()
            )
        _materialize_run_projection(index_db)

        expression = "context-snapshots where boundary:session_start"
        source = parse_unit_source_expression(expression)
        assert source is not None

        with ArchiveStore.open_existing(archive_root) as archive:
            first_page = query_unit_rows(archive, source, query=expression, limit=2, offset=0)
            second_page = query_unit_rows(archive, source, query=expression, limit=2, offset=2)

        assert len(first_page.items) == 2
        assert first_page.next_offset == 2
        assert len(second_page.items) == 1
        assert second_page.next_offset is None
        seen = {
            cast(ContextSnapshotQueryRowPayload, item).session_id for item in (*first_page.items, *second_page.items)
        }
        assert seen == {
            "codex-session:ext-page-0",
            "codex-session:ext-page-1",
            "codex-session:ext-page-2",
        }

    def test_exists_run_projection_predicates_select_sessions(self, workspace_env: dict[str, Path]) -> None:
        """`exists run/observed-event/context-snapshot(...)` lower to SQL session selectors.

        role:subagent and boundary:subagent_start are only true of a real
        branch_type='subagent' child session (polylogue-dab): they select the
        CHILD's own session_id, not the dispatching parent's. kind:tool_finished
        (not kind:subagent_started -- unsupported post-dab, see
        run_projection_relations.py) is exercised against the parent's real
        Task tool_use/tool_result pair, so it selects the parent instead --
        giving each exists(...) unit a distinct, model-accurate selection.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "exists-hit")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("exists hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="Subagent dispatched for the audit.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "tool-hit",
                        "name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "taskId": "task-hit",
                            "child_session_id": "codex-session:ext-exists-child",
                            "prompt": "Audit the run projection.",
                        },
                    },
                    {"type": "tool_result", "tool_id": "tool-hit", "text": "Subagent done.\n1 passed"},
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "exists-child")
            .provider("codex")
            .parent_session("ext-exists-hit")
            .branch_type("subagent")
            .git_repository_url("polylogue")
            .title("exists child")
            .add_message("m-child", role="assistant", text="Audit complete.")
            .save()
        )
        (
            SessionBuilder(index_db, "exists-miss")
            .provider("codex")
            .git_repository_url("polylogue")
            .title("exists miss")
            .add_message("m-miss", role="user", text="No subagents here.")
            .save()
        )
        _materialize_run_projection(index_db)

        with ArchiveStore.open_existing(archive_root) as archive:

            def selected(expression: str) -> list[str]:
                spec = compile_expression(expression)
                assert spec.boolean_predicate is not None
                rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)
                return sorted(row.session_id for row in rows)

            # Discriminating exists predicates select only the subagent session.
            assert selected("sessions where exists run(role:subagent)") == ["codex-session:ext-exists-child"]
            assert selected("exists observed-event(kind:tool_finished)") == ["codex-session:ext-exists-hit"]
            assert selected("exists context-snapshot(boundary:subagent_start)") == ["codex-session:ext-exists-child"]

    def test_exists_run_projection_predicates_use_source_relations(self, workspace_env: dict[str, Path]) -> None:
        """`exists` selectors agree with source-derived run projection terminal rows."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "source-exists-hit")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("source exists hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="I will inspect the archive.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "tool-hit",
                        "name": "Read",
                        "tool_input": {"file_path": "polylogue/storage/sqlite/archive_tiers/archive.py"},
                    },
                    {"type": "tool_result", "tool_id": "tool-hit", "text": "ok"},
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "source-exists-miss")
            .provider("claude-code")
            .git_repository_url("polylogue")
            .title("source exists miss")
            .save()
        )
        # Run/observed-event/context-snapshot rows are source-derived
        # (polylogue-dab): there is no materialized table left to clear, so
        # this itself proves the premise -- these predicates run against
        # `blocks`/`sessions` directly with no separate materialization step.

        with ArchiveStore.open_existing(archive_root) as archive:

            def selected(expression: str) -> list[str]:
                spec = compile_expression(expression)
                assert spec.boolean_predicate is not None
                rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)
                return sorted(row.session_id for row in rows)

            assert selected("sessions where exists run(role:main)") == [
                "claude-code-session:ext-source-exists-hit",
                "claude-code-session:ext-source-exists-miss",
            ]
            assert selected("exists context-snapshot(boundary:session_start)") == [
                "claude-code-session:ext-source-exists-hit",
                "claude-code-session:ext-source-exists-miss",
            ]
            assert selected("exists observed-event(tool:Read)") == ["claude-code-session:ext-source-exists-hit"]

    def test_exists_block_tool_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("block tool hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="edited file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-hit",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("claude-code")
            .title("block tool miss")
            .add_message(
                "m-miss",
                role="assistant",
                text="read file",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_id": "tool-miss",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_read",
                    }
                ],
            )
            .save()
        )

        spec = compile_expression("exists block(action:file_edit AND path:archive/query)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-hit"]

    def test_role_count_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "role-count-hit")
            .provider("claude-code")
            .title("role count hit")
            .add_message("m-hit-user-1", role="user", text="please inspect the query")
            .add_message("m-hit-user-2", role="user", text="and add coverage")
            .add_message("m-hit-assistant", role="assistant", text="I added focused aggregate count coverage")
            .save()
        )
        (
            SessionBuilder(index_db, "role-count-control")
            .provider("claude-code")
            .title("role count control")
            .add_message("m-control-user", role="user", text="please inspect")
            .add_message("m-control-assistant", role="assistant", text="short reply")
            .save()
        )

        spec = compile_expression("sessions where user_messages >= 2 AND assistant_words between 5 and 8")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-role-count-hit"]

    def test_authored_user_count_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "authored-count-hit")
            .provider("claude-code")
            .title("authored count hit")
            .add_message(
                "m-hit-prompt-1",
                role="user",
                text="please inspect",
                material_origin="human_authored",
            )
            .add_message(
                "m-hit-prompt-2",
                role="user",
                text="and add coverage",
                material_origin="human_authored",
            )
            .add_message(
                "m-hit-runtime",
                role="user",
                text="<task-notification>ignored runtime row</task-notification>",
                message_type="protocol",
            )
            .save()
        )
        (
            SessionBuilder(index_db, "authored-count-control")
            .provider("claude-code")
            .title("authored count control")
            .add_message(
                "m-control-prompt",
                role="user",
                text="please inspect",
                material_origin="human_authored",
            )
            .add_message(
                "m-control-runtime-1",
                role="user",
                text="<task-notification>runtime one</task-notification>",
                message_type="protocol",
            )
            .add_message(
                "m-control-runtime-2",
                role="user",
                text="<task-notification>runtime two</task-notification>",
                message_type="protocol",
            )
            .save()
        )

        role_spec = compile_expression("sessions where user_messages >= 3")
        authored_spec = compile_expression("sessions where authored_user_messages >= 2 AND authored_user_words >= 4")
        assert role_spec.boolean_predicate is not None
        assert authored_spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            role_rows = archive.list_summaries(limit=100, boolean_predicate=role_spec.boolean_predicate)
            authored_rows = archive.list_summaries(limit=100, boolean_predicate=authored_spec.boolean_predicate)

        assert {row.session_id for row in role_rows} == {
            "claude-code-session:ext-authored-count-hit",
            "claude-code-session:ext-authored-count-control",
        }
        assert [row.session_id for row in authored_rows] == ["claude-code-session:ext-authored-count-hit"]

    def test_sequence_predicate_executes_in_order_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("claude-code")
            .title("sequence hit")
            .add_message(
                "m-hit-1",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-hit-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-hit-2",
                role="assistant",
                text="test",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "tool-hit-2",
                        "input": {"command": "pytest tests/unit/cli/test_query_expression.py"},
                        "semantic_type": "shell",
                    }
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "reversed")
            .provider("claude-code")
            .title("sequence reversed")
            .add_message(
                "m-reversed-1",
                role="assistant",
                text="test",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "tool-reversed-1",
                        "input": {"command": "pytest"},
                        "semantic_type": "shell",
                    }
                ],
            )
            .add_message(
                "m-reversed-2",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-reversed-2",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        spec = compile_expression("seq(action:file_edit -> action:shell)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-hit"]

    def test_constrained_sequence_predicates_execute_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"

        def add_action(builder: SessionBuilder, message_id: str, action: str, timestamp: str | None) -> None:
            tool_name = {"file_edit": "Edit", "search": "Grep", "shell": "Bash"}[action]
            builder.add_message(
                message_id,
                role="assistant",
                text=action,
                timestamp=timestamp,
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": tool_name,
                        "tool_id": f"tool-{message_id}",
                        "input": {"command": action},
                        "semantic_type": action,
                    }
                ],
            )

        adjacent = SessionBuilder(index_db, "constrained-adjacent").provider("claude-code")
        add_action(adjacent, "a1", "file_edit", "2026-01-01T00:00:00+00:00")
        add_action(adjacent, "a2", "shell", "2026-01-01T00:05:00+00:00")
        adjacent.save()

        intervening = SessionBuilder(index_db, "constrained-intervening").provider("claude-code")
        add_action(intervening, "i1", "file_edit", "2026-01-01T00:00:00+00:00")
        add_action(intervening, "i2", "search", "2026-01-01T00:01:00+00:00")
        add_action(intervening, "i3", "shell", "2026-01-01T00:02:00+00:00")
        intervening.save()

        timeless = SessionBuilder(index_db, "constrained-timeless").provider("claude-code")
        add_action(timeless, "t1", "file_edit", None)
        add_action(timeless, "t2", "shell", None)
        timeless.save()

        def selected(expression: str) -> list[str]:
            spec = compile_expression(expression)
            assert spec.boolean_predicate is not None
            with ArchiveStore.open_existing(index_db.parent) as archive:
                rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)
            return sorted(row.session_id for row in rows)

        assert selected("seq(action:file_edit -> action:shell)") == [
            "claude-code-session:ext-constrained-adjacent",
            "claude-code-session:ext-constrained-intervening",
            "claude-code-session:ext-constrained-timeless",
        ]
        assert selected("seq(action:file_edit ->[next] action:shell)") == [
            "claude-code-session:ext-constrained-adjacent",
            "claude-code-session:ext-constrained-timeless",
        ]
        assert selected("seq(action:file_edit ->[within:5m] action:shell)") == [
            "claude-code-session:ext-constrained-adjacent",
            "claude-code-session:ext-constrained-intervening",
        ]

    def test_single_action_sequence_filter_still_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "shell-hit")
            .provider("claude-code")
            .title("shell action")
            .add_message(
                "m-shell",
                role="assistant",
                text="test",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "tool-shell",
                        "input": {"command": "pytest tests/unit/cli/test_query_expression.py"},
                        "semantic_type": "shell",
                    }
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "edit-miss")
            .provider("claude-code")
            .title("edit action")
            .add_message(
                "m-edit",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "tool-edit",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, action_sequence=("shell",))

        assert [row.session_id for row in rows] == ["claude-code-session:ext-shell-hit"]

    def test_sequence_predicate_filters_step_fields_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "failed-cycle")
            .provider("claude-code")
            .title("failed test edit cycle")
            .add_message(
                "m-failed-1",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "edit-hit-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-failed-2",
                role="assistant",
                text="test failed",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "bash-hit-1",
                        "input": {"command": "pytest tests/unit/cli/test_query_expression.py"},
                        "semantic_type": "shell",
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "bash-hit-1",
                        "text": "FAILED tests/unit/cli/test_query_expression.py::test_sequence",
                    },
                ],
            )
            .add_message(
                "m-failed-3",
                role="assistant",
                text="fix",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "edit-hit-2",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "passed-cycle")
            .provider("claude-code")
            .title("passed test edit cycle")
            .add_message(
                "m-passed-1",
                role="assistant",
                text="edit",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "edit-miss-1",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .add_message(
                "m-passed-2",
                role="assistant",
                text="test passed",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": "bash-miss-1",
                        "input": {"command": "pytest tests/unit/cli/test_query_expression.py"},
                        "semantic_type": "shell",
                    },
                    {"type": "tool_result", "tool_id": "bash-miss-1", "text": "1 passed"},
                ],
            )
            .add_message(
                "m-passed-3",
                role="assistant",
                text="fix",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "tool_id": "edit-miss-2",
                        "input": {"file_path": "polylogue/archive/query/expression.py"},
                        "semantic_type": "file_edit",
                    }
                ],
            )
            .save()
        )

        spec = compile_expression("seq(action:file_edit -> action:shell AND output:failed -> action:file_edit)")
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["claude-code-session:ext-failed-cycle"]

    def test_fts_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("fts hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="timeout failure",
                blocks=[{"type": "text", "text": "timeout failure in query pipeline"}],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("fts miss")
            .add_message(
                "m-miss",
                role="assistant",
                text="ordinary response",
                blocks=[{"type": "text", "text": "ordinary response"}],
            )
            .save()
        )

        spec = compile_expression('sessions where origin:chatgpt-export AND ~"timeout failure"')
        assert spec.boolean_predicate is not None
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(limit=100, boolean_predicate=spec.boolean_predicate)

        assert [row.session_id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_semantic_predicate_lowers_to_vector_seed_and_residual_filter(self) -> None:
        spec = compile_expression('sessions where semantic:"query compiler" AND title:hit')

        assert spec.similar_text == "query compiler"
        assert spec.boolean_predicate == QueryFieldPredicate(field="title", values=("hit",))

    def test_near_text_predicate_is_semantic_alias(self) -> None:
        ast = parse_expression_ast('sessions where near:text:"query compiler"')

        assert ast.boolean_predicate == QuerySemanticPredicate(text="query compiler")
        assert compile_expression('sessions where near:text:"query compiler"').similar_text == "query compiler"

    @pytest.mark.parametrize(
        "expression, message",
        [
            ('sessions where semantic:"query compiler" OR title:hit', "under OR"),
            ('sessions where NOT semantic:"query compiler"', "under NOT"),
            ('sessions where semantic:"query compiler" AND semantic:"review loop"', "only one semantic"),
        ],
    )
    def test_semantic_predicate_rejects_ranked_boolean_forms(self, expression: str, message: str) -> None:
        with pytest.raises(ExpressionCompileError, match=message):
            compile_expression(expression)

    async def test_semantic_predicate_executes_via_vector_plan(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.archive.query.archive_execution import list_summaries_archive
        from tests.infra.storage_records import SessionBuilder

        class StubVectorProvider:
            model = "stub"

            def upsert(self, session_id: str, messages: list[MessageRecord]) -> None:
                raise NotImplementedError

            def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
                assert text == "query compiler"
                assert limit >= 6
                return [
                    ("chatgpt-export:ext-hit:m-hit", 0.01),
                    ("chatgpt-export:ext-miss:m-miss", 0.02),
                ]

            def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
                raise NotImplementedError

        archive_root = workspace_env["archive_root"]
        index_db = archive_root / "index.db"
        (
            SessionBuilder(index_db, "hit")
            .provider("chatgpt")
            .title("semantic hit")
            .add_message(
                "m-hit",
                role="assistant",
                text="query compiler plan",
                blocks=[{"type": "text", "text": "query compiler plan"}],
            )
            .save()
        )
        (
            SessionBuilder(index_db, "miss")
            .provider("chatgpt")
            .title("semantic miss")
            .add_message(
                "m-miss",
                role="assistant",
                text="query compiler plan",
                blocks=[{"type": "text", "text": "query compiler plan"}],
            )
            .save()
        )

        spec = compile_expression('sessions where semantic:"query compiler" AND title:hit')
        rows = await list_summaries_archive(
            spec.to_plan(vector_provider=StubVectorProvider()),
            archive_root=archive_root,
            config=None,
        )

        assert [row.id for row in rows] == ["chatgpt-export:ext-hit"]

    def test_lineage_predicate_ast_exposes_seed(self) -> None:
        ast = parse_expression_ast("lineage:id:chatgpt-export:ext-root")

        assert ast.boolean_predicate == QueryLineagePredicate(seed_session_id="chatgpt-export:ext-root")

    def test_lineage_predicate_executes_against_archive(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.storage_records import SessionBuilder

        index_db = workspace_env["archive_root"] / "index.db"
        (SessionBuilder(index_db, "root").provider("chatgpt").title("root").add_message("m-root", text="root").save())
        (
            SessionBuilder(index_db, "child")
            .provider("chatgpt")
            .title("child")
            .parent_session("ext-root")
            .add_message("m-child", text="child")
            .save()
        )
        (
            SessionBuilder(index_db, "other")
            .provider("chatgpt")
            .title("other")
            .add_message("m-other", text="other")
            .save()
        )

        spec = compile_expression("lineage:id:chatgpt-export:ext-child")
        assert spec.boolean_predicate == QueryLineagePredicate(seed_session_id="chatgpt-export:ext-child")
        with ArchiveStore.open_existing(index_db.parent) as archive:
            rows = archive.list_summaries(
                limit=100,
                boolean_predicate=spec.boolean_predicate,
            )

        assert sorted(row.session_id for row in rows) == [
            "chatgpt-export:ext-child",
            "chatgpt-export:ext-root",
        ]


# ---------------------------------------------------------------------------
# Lowerer field-mapping tests
# ---------------------------------------------------------------------------


class TestLowererFieldMapping:
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

    def test_session_alias(self) -> None:
        session_id = "claude-code-session:abc123def456"
        spec = compile_expression(f"session:{session_id}")
        assert spec.session_id == session_id

    def test_session_alias_inside_boolean_predicate(self) -> None:
        session_id = "claude-code-session:abc123def456"
        spec = compile_expression(f"sessions where session:{session_id}")
        assert spec.boolean_predicate == QueryFieldPredicate(field="session", values=(session_id,))

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

    def test_readable_messages_comparison(self) -> None:
        spec = compile_expression("messages > 10")

        assert spec.min_messages == 11
        assert spec.max_messages is None

    def test_readable_words_less_than(self) -> None:
        spec = compile_expression("words < 500")

        assert spec.max_words == 499
        assert spec.min_words is None

    def test_readable_count_range(self) -> None:
        spec = compile_expression("messages between 5 and 20 words between 100 and 500")

        assert spec.min_messages == 5
        assert spec.max_messages == 20
        assert spec.min_words == 100
        assert spec.max_words == 500

    def test_aggregate_count_comparison_requires_boolean_query(self) -> None:
        with pytest.raises(ExpressionCompileError, match="supported only inside `sessions where`"):
            compile_expression("tool_use_messages >= 2")

    def test_boolean_ast_exposes_aggregate_count_comparisons(self) -> None:
        ast = parse_expression_ast(
            "sessions where authored_user_messages >= 2 AND assistant_words between 5 and 20 AND tool_use_messages >= 1"
        )

        assert ast.boolean_predicate == QueryBoolPredicate(
            "and",
            (
                QueryFieldPredicate(field="authored_user_messages", values=("2",), op=">="),
                QueryFieldPredicate(field="assistant_words", values=("5",), op=">="),
                QueryFieldPredicate(field="assistant_words", values=("20",), op="<="),
                QueryFieldPredicate(field="tool_use_messages", values=("1",), op=">="),
            ),
        )

    def test_readable_count_range_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ExpressionCompileError, match="lower bound 20 is greater than upper bound 5"):
            compile_expression("messages between 20 and 5")

    def test_readable_date_range(self) -> None:
        spec = compile_expression("date between 2026-01-01 and 2026-02-01")

        assert spec.since == "2026-01-01"
        assert spec.until == "2026-02-01"

    def test_readable_date_gte(self) -> None:
        spec = compile_expression("date >= 2026-01-01")

        assert spec.since == "2026-01-01"
        assert spec.until is None

    def test_readable_date_lte(self) -> None:
        spec = compile_expression("date <= 2026-02-01")

        assert spec.until == "2026-02-01"
        assert spec.since is None

    def test_readable_date_equality_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="date equality is not supported"):
            compile_expression("date = 2026-01-01")

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

    def test_unknown_field_suggests_close_registry_match(self) -> None:
        with pytest.raises(ExpressionCompileError, match="did you mean: origin") as exc_info:
            compile_expression("origon:chatgpt-export")
        assert exc_info.value.field == "origon"

    def test_unknown_field_message_lists_recognized(self) -> None:
        with pytest.raises(ExpressionCompileError, match="recognized fields"):
            compile_expression("xyz:value")

    def test_boolean_unknown_field_suggests_close_registry_match(self) -> None:
        with pytest.raises(ExpressionCompileError, match="did you mean: origin") as exc_info:
            compile_expression("sessions where origon:chatgpt-export")
        assert exc_info.value.field == "origon"

    def test_cross_field_or_compiles_to_boolean_predicate(self) -> None:
        spec = compile_expression("repo:polylogue OR repo:sinex")
        assert isinstance(spec.boolean_predicate, QueryBoolPredicate)

    def test_cross_field_or_no_longer_becomes_fts_terms(self) -> None:
        spec = compile_expression("repo:polylogue OR origin:claude-code-session")
        assert spec.boolean_predicate is not None
        assert spec.query_terms == ()

    def test_unknown_origin_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="unknown origin"):
            compile_expression("origin:no-such-origin")

    def test_messages_without_op_raises(self) -> None:
        with pytest.raises(ExpressionCompileError, match="comparison operator"):
            compile_expression("messages:10")

    @pytest.mark.parametrize("expr", ["messages:>=10x", "words:<=5m", "tool_use_messages:>=1x"])
    def test_malformed_count_clause_raises_instead_of_broadening(self, expr: str) -> None:
        with pytest.raises(ExpressionCompileError, match="comparison operator") as exc_info:
            compile_expression(expr)
        assert exc_info.value.field in {"messages", "words", "tool_use_messages"}


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

    def test_dsl_multi_field_clause_single_shell_quoted_token(self) -> None:
        """Regression for polylogue-zrdp.

        `polylogue find "repo:polylogue since:7d"` arrives as ONE query_term
        containing a space (the shell already stripped the outer quotes).
        This must be parsed as two ANDed field clauses, not wrapped whole as
        a single literal FTS phrase.
        """
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("repo:polylogue since:7d",))
        spec = request.query_spec()
        assert spec.repo_names == ("polylogue",)
        assert spec.since is not None
        assert spec.query_terms == ()

    def test_dsl_multi_field_clause_matches_split_token_form(self) -> None:
        """The single-quoted-token and pre-split-token forms must compile identically."""
        from polylogue.cli.root_request import RootModeRequest

        quoted = RootModeRequest(params={}, query_terms=("repo:polylogue origin:claude-code-session",)).query_spec()
        split = RootModeRequest(params={}, query_terms=("repo:polylogue", "origin:claude-code-session")).query_spec()
        assert quoted.repo_names == split.repo_names == ("polylogue",)
        assert quoted.origins == split.origins == ("claude-code-session",)

    def test_colon_containing_phrase_stays_literal_text(self) -> None:
        """Regression for the zrdp fix's own review (#2626): a shell-quoted
        phrase containing an unregistered colon-word (a URL, a log label)
        must stay literal FTS text, not get parsed as an unknown DSL field.
        """
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("http://host failed",))
        spec = request.query_spec()
        assert any("http://host failed" in t for t in spec.query_terms)
        assert not spec.repo_names

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

    def test_shell_quoted_terminal_source_compiles_as_structural_query(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("actions where action:file_edit AND path:archive/query",))
        spec = request.query_spec()

        assert isinstance(spec.boolean_predicate, QueryExistsPredicate)
        assert spec.boolean_predicate.unit == "action"
        assert spec.has_filters()
        assert spec.query_terms == ()

    def test_shell_quoted_sessions_where_compiles_as_structural_query(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("sessions where origin:claude-code-session",))
        spec = request.query_spec()

        assert spec.boolean_predicate is not None
        assert spec.has_filters()
        assert spec.query_terms == ()

    @pytest.mark.parametrize(
        ("expression", "unit"),
        [
            ("exists file(path:archive/query)", "file"),
            ("exists assertion(kind:decision)", "assertion"),
        ],
    )
    def test_shell_quoted_extended_exists_units_compile_as_structural_query(self, expression: str, unit: str) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=(expression,))
        spec = request.query_spec()

        assert isinstance(spec.boolean_predicate, QueryExistsPredicate)
        assert spec.boolean_predicate.unit == unit
        assert spec.has_filters()
        assert spec.query_terms == ()

    def test_unknown_field_raises(self) -> None:
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={}, query_terms=("nosuchfield:value",))
        with pytest.raises(ExpressionCompileError, match="unknown query field"):
            request.query_spec()


# ---------------------------------------------------------------------------
# Field registry completeness
# ---------------------------------------------------------------------------


class TestFieldRegistry:
    FIELD_DESCRIPTOR_CASES = {
        "repo": ("repo:polylogue", "repo_names", ("polylogue",)),
        "project": ("project:g-p-6a40343a", "project_refs", ("g-p-6a40343a",)),
        "origin": ("origin:claude-code-session", "origins", ("claude-code-session",)),
        "tag": ("tag:review", "tags", ("review",)),
        "path": ("path:polylogue/cli", "referenced_path", ("polylogue/cli",)),
        "cwd": ("cwd:/realm/project", "cwd_prefix", "/realm/project"),
        "tool": ("tool:bash", "tool_terms", ("bash",)),
        "action": ("action:file_edit", "action_terms", ("file_edit",)),
        "has": ("has:paste", "filter_has_paste", True),
        "id": ("id:abc123", "session_id", "abc123"),
        "session": ("session:claude-code-session:abc123", "session_id", "claude-code-session:abc123"),
        "title": ("title:refactor", "title", "refactor"),
        "since": ("since:7d", "since", "7 days ago"),
        "until": ("until:2024-01-15", "until", "2024-01-15"),
        "near": ('near:"semantic search"', "similar_text", "semantic search"),
        "contains": ("contains:foo", "contains_terms", ("foo",)),
        "messages": ("messages:>=10", "min_messages", 10),
        "words": ("words:>=200", "min_words", 200),
        "user_messages": (
            "sessions where user_messages >= 2",
            "boolean_predicate",
            QueryFieldPredicate(field="user_messages", values=("2",), op=">="),
        ),
        "authored_user_messages": (
            "sessions where authored_user_messages >= 2",
            "boolean_predicate",
            QueryFieldPredicate(field="authored_user_messages", values=("2",), op=">="),
        ),
        "assistant_messages": (
            "sessions where assistant_messages >= 2",
            "boolean_predicate",
            QueryFieldPredicate(field="assistant_messages", values=("2",), op=">="),
        ),
        "system_messages": (
            "sessions where system_messages = 0",
            "boolean_predicate",
            QueryFieldPredicate(field="system_messages", values=("0",), op="="),
        ),
        "tool_messages": (
            "sessions where tool_messages = 0",
            "boolean_predicate",
            QueryFieldPredicate(field="tool_messages", values=("0",), op="="),
        ),
        "tool_use_messages": (
            "sessions where tool_use_messages >= 1",
            "boolean_predicate",
            QueryFieldPredicate(field="tool_use_messages", values=("1",), op=">="),
        ),
        "thinking_messages": (
            "sessions where thinking_messages >= 1",
            "boolean_predicate",
            QueryFieldPredicate(field="thinking_messages", values=("1",), op=">="),
        ),
        "paste_messages": (
            "sessions where paste_messages = 0",
            "boolean_predicate",
            QueryFieldPredicate(field="paste_messages", values=("0",), op="="),
        ),
        "duration_ms": (
            "sessions where duration_ms >= 60000",
            "boolean_predicate",
            QueryFieldPredicate(field="duration_ms", values=("60000",), op=">="),
        ),
        "user_words": (
            "sessions where user_words >= 100",
            "boolean_predicate",
            QueryFieldPredicate(field="user_words", values=("100",), op=">="),
        ),
        "authored_user_words": (
            "sessions where authored_user_words >= 100",
            "boolean_predicate",
            QueryFieldPredicate(field="authored_user_words", values=("100",), op=">="),
        ),
        "assistant_words": (
            "sessions where assistant_words >= 500",
            "boolean_predicate",
            QueryFieldPredicate(field="assistant_words", values=("500",), op=">="),
        ),
        "lane": ("lane:dialogue", "retrieval_lane", "dialogue"),
        "lineage": (
            "lineage:id:chatgpt-export:ext-root",
            "boolean_predicate",
            QueryLineagePredicate(seed_session_id="chatgpt-export:ext-root"),
        ),
    }

    def test_registry_has_required_fields(self) -> None:
        required = {"repo", "origin", "tag", "path", "cwd", "tool", "action", "has", "id", "since", "until", "near"}
        assert required.issubset(EXPRESSION_FIELD_REGISTRY.keys())

    def test_descriptor_cases_cover_every_registry_field(self) -> None:
        assert set(self.FIELD_DESCRIPTOR_CASES) == set(EXPRESSION_FIELD_REGISTRY)

    def test_all_registry_entries_have_required_descriptor_keys(self) -> None:
        for field, info in EXPRESSION_FIELD_REGISTRY.items():
            assert "description" in info, f"{field} missing description"
            assert info["description"], f"{field} has empty description"
            assert info.get("spec_field"), f"{field} missing spec_field"
            assert info.get("example"), f"{field} missing example"
            assert info.get("negatable") in {"yes", "no"}, f"{field} has invalid negatable descriptor"

    @pytest.mark.parametrize("field", sorted(EXPRESSION_FIELD_REGISTRY))
    def test_registry_example_reaches_declared_spec_field(self, field: str) -> None:
        expression, spec_field, expected = self.FIELD_DESCRIPTOR_CASES[field]
        assert EXPRESSION_FIELD_REGISTRY[field]["example"].split(" | ")[0] == expression
        advertised_fields = EXPRESSION_FIELD_REGISTRY[field]["spec_field"].split("/")
        assert spec_field in advertised_fields

        spec = compile_expression(expression)
        assert getattr(spec, spec_field) == expected

    @pytest.mark.parametrize("field", sorted(EXPRESSION_FIELD_REGISTRY))
    def test_registry_negatable_descriptor_matches_behavior(self, field: str) -> None:
        expression, spec_field, expected = self.FIELD_DESCRIPTOR_CASES[field]
        negated = "-" + expression

        if EXPRESSION_FIELD_REGISTRY[field]["negatable"] == "no":
            with pytest.raises(ExpressionCompileError):
                compile_expression(negated)
            return

        spec = compile_expression(negated)
        if field == "origin":
            assert spec.excluded_origins == expected
        elif field == "tag":
            assert spec.excluded_tags == expected
        elif field == "tool":
            assert spec.excluded_tool_terms == expected
        elif field == "action":
            assert spec.excluded_action_terms == expected
        assert getattr(spec, spec_field) in {(), None, False}

    def test_cross_field_or_reaches_boolean_predicate(self) -> None:
        spec = compile_expression("repo:x OR origin:chatgpt-export")
        assert isinstance(spec.boolean_predicate, QueryBoolPredicate)

    def test_bare_or_text_reports_explicit_boolean_syntax(self) -> None:
        with pytest.raises(ExpressionCompileError, match="bare OR is ambiguous"):
            compile_expression("alpha OR beta")


# ---------------------------------------------------------------------------
# MCP surface wiring (#1860)
# ---------------------------------------------------------------------------


class TestMCPWiring:
    """build_query_spec routes free-text query through the shared DSL lowerer."""

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

    def test_cross_field_or_compiles(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        spec = build_query_spec(query="repo:x OR origin:chatgpt-export")
        assert isinstance(spec.boolean_predicate, QueryBoolPredicate)
        assert spec.query_terms == ()

    def test_none_query_returns_base_spec(self) -> None:
        from polylogue.mcp.query_contracts import build_query_spec

        spec = build_query_spec(query=None, tag="review")
        # With no query expression, must still apply the flag-derived filters.
        assert spec.tags == ("review",)
        assert spec.query_terms == ()


# ---------------------------------------------------------------------------
# Cross-surface parity (#1860 / #2006)
# ---------------------------------------------------------------------------


class TestCrossSurfaceParity:
    """Same DSL expression → same SessionQuerySpec on all three surfaces.

    CLI path:   RootModeRequest.query_spec()
    MCP path:   build_query_spec(query=...)
    Daemon path: compile_expression_into(query_str, base_spec)  (same function)

    Because all three surfaces ultimately call compile_expression_into, these
    tests verify that the Lark-backed lowerer is wired symmetrically and that
    no surface silently re-parses the expression through its own ad-hoc logic.
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
# Bug 7 regression: ?contains= not routed through DSL parsing (#1873)
# ---------------------------------------------------------------------------


class TestDaemonContainsParamNotCompiled:
    """The archive session-list route must not route ?contains= through compile_expression."""

    def test_contains_param_not_compiled_as_dsl(self, workspace_env: dict[str, Path]) -> None:
        """Bug 7: a ?contains= value rejected as DSL must still be treated as
        a literal FTS filter, not parsed/lowered as query syntax.

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
        payload = handler._do_archive_session_list(
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
    into the archive session-list route over a seeded archive.
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

    def test_session_clause_produces_session_id_in_spec(self) -> None:
        """compile_expression('session:abc') must be an exact-ref alias."""
        spec = compile_expression("session:abc")
        assert spec.session_id == "abc"

    def test_id_query_scopes_results_and_total(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        ids = self._seed(index_db, [("alpha", "alpha body"), ("beta", "beta body")])
        assert len(ids) == 2

        payload = self._handler()._do_archive_session_list(
            workspace_env["archive_root"], {"query": [f"id:{ids[0]}"]}, 50, 0
        )
        assert isinstance(payload, dict)
        # total must be scoped to the id match, NOT the archive-wide count of 2.
        assert payload["total"] == 1
        items = payload["items"]
        assert isinstance(items, list) and len(items) == 1

    def test_session_query_scopes_results_and_total(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        ids = self._seed(index_db, [("alpha", "alpha body"), ("beta", "beta body")])
        assert len(ids) == 2

        payload = self._handler()._do_archive_session_list(
            workspace_env["archive_root"], {"query": [f"session:{ids[0]}"]}, 50, 0
        )
        assert isinstance(payload, dict)
        assert payload["total"] == 1
        items = payload["items"]
        assert isinstance(items, list) and len(items) == 1

    def test_id_miss_returns_typed_empty_not_500(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db, [("alpha", "alpha body")])

        payload = self._handler()._do_archive_session_list(
            workspace_env["archive_root"], {"query": ["id:nonexistentnope"]}, 50, 0
        )
        # A missing id is a typed-empty page, not a 500 propagated from resolve.
        assert payload["items"] == []
        assert payload["total"] == 0
        assert payload["limit"] == 50
        assert payload["offset"] == 0
        assert payload["route_state"]["state"] == "no_results"

    def test_session_miss_returns_typed_empty_not_500(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db, [("alpha", "alpha body")])

        payload = self._handler()._do_archive_session_list(
            workspace_env["archive_root"], {"query": ["session:nonexistentnope"]}, 50, 0
        )
        assert payload["items"] == []
        assert payload["total"] == 0
        assert payload["limit"] == 50
        assert payload["offset"] == 0
        assert payload["route_state"]["state"] == "no_results"

    def test_id_ambiguous_prefix_raises_query_spec_error(self, workspace_env: dict[str, Path]) -> None:
        import os

        from polylogue.archive.query.spec import QuerySpecError

        index_db = workspace_env["archive_root"] / "index.db"
        ids = self._seed(index_db, [("aaa", "x body"), ("aab", "y body")])
        prefix = os.path.commonprefix(ids)
        assert prefix and prefix not in ids, "need a shared, non-exact prefix"

        with pytest.raises(QuerySpecError):
            self._handler()._do_archive_session_list(workspace_env["archive_root"], {"query": [f"id:{prefix}"]}, 50, 0)

    def test_contains_filters_without_query(self, workspace_env: dict[str, Path]) -> None:
        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db, [("one", "has findmetoken here"), ("two", "unrelated content")])

        # ?contains=foo with no ?query= must still filter (Bug 7): it routes to the
        # FTS branch as a literal term rather than returning the unfiltered page.
        payload = self._handler()._do_archive_session_list(
            workspace_env["archive_root"], {"contains": ["findmetoken"]}, 50, 0
        )
        assert isinstance(payload, dict)
        hits = payload.get("hits")
        assert isinstance(hits, list) and len(hits) == 1
