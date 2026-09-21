"""The published query-AST schema is derived from the producing nodes.

``polylogue.archive.query.query_ast_schema`` builds its Pydantic models from
the payload declarations the AST nodes in
:mod:`polylogue.archive.query.predicate` and
:mod:`polylogue.archive.query.expression` serialize themselves from. There is
no hand-written mirror of the wire shape, so these tests exercise the claims
that matter once the mirror is gone:

1. A payload key declared on a node reaches both the serializer and the
   published model from that one declaration, and every declared key is
   present in the rendered OpenAPI document.
2. The published tagged unions cover every declared member, so a new
   predicate leaf or pipeline stage cannot be published as unrepresentable.
3. An independently written expected AST document -- hand-authored from the
   declarations, not captured from the generator -- still matches. A generator
   that is uniformly wrong fails this even though every generated artifact
   agrees with every other.
4. The predicate tree round-trips losslessly, and ``explain_expression``'s
   real output validates for every DSL surface, including the ``| agg ...``
   pipeline the previous hand-written mirror rejected.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from polylogue.archive.query.expression import (
    PIPELINE_STAGE_PAYLOAD_SCHEMAS,
    QUERY_AST_PAYLOAD_SCHEMAS,
    QueryUnitLimitStage,
    explain_expression,
    parse_unit_source_expression,
)
from polylogue.archive.query.payload_schema import (
    MISSING,
    PayloadField,
    PayloadSchema,
    payload_model,
    render_payload,
)
from polylogue.archive.query.predicate import (
    PREDICATE_PAYLOAD_SCHEMAS,
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryFieldRef,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.query.query_ast_schema import (
    QUERY_AST_SCHEMA_VERSION,
    QueryExpressionExplanationAst,
    QueryPredicateAst,
    ast_to_predicate,
    explanation_payload_to_ast,
    predicate_to_ast,
    published_ast_models,
)

ROOT = Path(__file__).resolve().parents[4]
OPENAPI_DOCUMENT = ROOT / "docs" / "openapi" / "search.yaml"


def _openapi_schemas() -> dict[str, Any]:
    document = yaml.safe_load(OPENAPI_DOCUMENT.read_text())
    return dict(document["components"]["schemas"])


_ALL_DECLARATIONS = (
    *PREDICATE_PAYLOAD_SCHEMAS,
    *QUERY_AST_PAYLOAD_SCHEMAS,
    QueryFieldRef.PAYLOAD,
    QuerySequenceConstraint.PAYLOAD,
)


def _declared_tags(declarations: tuple[PayloadSchema, ...]) -> set[str]:
    """Return every discriminator tag the declarations can emit."""

    tags: set[str] = set()
    for declaration in declarations:
        field = next(field for field in declaration.fields if field.key == "kind")
        if field.const is not MISSING:
            tags.add(str(field.const))
            continue
        tags.update(str(tag) for tag in get_args(field.type()))
    return tags


def test_every_declared_payload_key_is_published_in_the_rendered_openapi() -> None:
    """A key declared on a node reaches the rendered document with no schema edit.

    Anti-vacuity: add a key to any node's payload declaration and re-render;
    it appears here with no second file edited. Publish a model that drops a
    declared key -- the failure mode a hand-written mirror had -- and the key
    sets differ.
    """

    schemas = _openapi_schemas()
    mismatches = {}
    for declaration in _ALL_DECLARATIONS:
        published = schemas.get(declaration.name)
        assert published is not None, f"{declaration.name} is declared but not published"
        published_keys = set(published.get("properties", {}))
        if published_keys != set(declaration.keys()):
            mismatches[declaration.name] = (
                sorted(set(declaration.keys()) - published_keys),
                sorted(published_keys - set(declaration.keys())),
            )
    assert mismatches == {}


def test_published_unions_cover_every_declared_member() -> None:
    """A declared predicate leaf or pipeline stage is reachable from its union.

    Anti-vacuity: declare a new stage without adding it to the published union
    and its tag is missing from the discriminator map here. The previous
    hand-written union omitted the ``agg`` stage exactly this way.
    """

    schemas = _openapi_schemas()
    stage_mapping = schemas["QueryUnitPipelineAst"]["properties"]["stages"]["items"]["discriminator"]["mapping"]
    assert set(stage_mapping) == _declared_tags(PIPELINE_STAGE_PAYLOAD_SCHEMAS)
    assert len(stage_mapping) == len(PIPELINE_STAGE_PAYLOAD_SCHEMAS)

    predicate_mapping = schemas["QueryUnitPipelineSourceAst"]["properties"]["predicate"]["discriminator"]["mapping"]
    assert set(predicate_mapping) == _declared_tags(PREDICATE_PAYLOAD_SCHEMAS)
    # ``and``/``or`` and ``lineage``/``logical`` each come from one declaration
    # whose tag is the node's own value, so there are more tags than members.
    assert len(predicate_mapping) == len(PREDICATE_PAYLOAD_SCHEMAS) + 2


def test_one_declaration_drives_both_the_serializer_and_the_published_model() -> None:
    """Adding a declared key changes the payload and the model together.

    Anti-vacuity: this is the property a hand-written mirror cannot have. Build
    the model from anything other than the declaration the serializer uses and
    one of the two assertions below fails.
    """

    declaration = QueryUnitLimitStage.PAYLOAD
    extended = replace(
        declaration,
        fields=(*declaration.fields, PayloadField("note", lambda: str | None, omit="none", default=None)),
    )

    model = payload_model(extended, suffix="Extended")
    assert "note" in model.model_json_schema()["properties"]

    class _AnnotatedStage(QueryUnitLimitStage):
        note = "declared once"

    rendered = render_payload(_AnnotatedStage(value=5), extended)
    assert rendered == {"kind": "limit", "value": 5, "note": "declared once"}
    # The unextended declaration still renders the unextended payload.
    assert render_payload(QueryUnitLimitStage(value=5), declaration) == {"kind": "limit", "value": 5}


#: Hand-authored from the payload declarations, not captured from the
#: generator. A generator that is uniformly wrong still agrees with every
#: artifact it produces; it does not agree with this.
_EXPECTED_AGG_PIPELINE: dict[str, object] = {
    "source": {
        "unit": "message",
        "predicate": {
            "kind": "field",
            "field": "role",
            "op": "=",
            "values": ["assistant"],
            "field_ref": {"scope": "unit", "name": "role", "source_name": "role", "unit": "message"},
        },
    },
    "stages": [
        {"kind": "group", "field": "role"},
        {
            "kind": "agg",
            "metrics": [
                {"fn": "count", "label": "count"},
                {"fn": "avg", "label": "avg_word_count", "field": "word_count"},
            ],
        },
        {"kind": "limit", "value": 2},
        {"kind": "terminal", "action": "agg"},
    ],
    "result": {
        "group_by": "role",
        "agg_metrics": [
            {"fn": "count", "label": "count"},
            {"fn": "avg", "label": "avg_word_count", "field": "word_count"},
        ],
        "limit": 2,
    },
}


def test_pipeline_payload_matches_an_independently_written_document() -> None:
    """The emitted AST equals a hand-authored expected document, key for key."""

    source = parse_unit_source_expression(
        "messages where role:assistant | group by role | agg count, avg:word_count | limit 2"
    )
    assert source is not None
    assert source.pipeline.to_payload() == _EXPECTED_AGG_PIPELINE


_PREDICATE_ROUNDTRIP_CASES: tuple[QueryPredicate, ...] = (
    QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
    QueryFieldPredicate(field="origin", values=("codex-session",), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="origin", source_name="origin")
    ),
    QueryFieldPredicate(field="count", values=("3",), op=">=").with_field_ref(
        QueryFieldRef(scope="unit", name="count", source_name="count", unit="message")
    ),
    QueryNotPredicate(QueryFieldPredicate(field="origin", values=("codex-session",), op="=")),
    QueryBoolPredicate(
        "and",
        (
            QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
            QueryFieldPredicate(field="repo", values=("polylogue",), op="="),
        ),
    ),
    QueryBoolPredicate(
        "or",
        (
            QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
            QueryNotPredicate(QueryFieldPredicate(field="repo", values=("polylogue",), op="=")),
        ),
    ),
    QueryExistsPredicate(unit="block", child=QueryFieldPredicate(field="tool_name", values=("Bash",), op="=")),
    QuerySequencePredicate(action_terms=("plan", "edit", "test")),
    QuerySequencePredicate(
        steps=(
            QueryFieldPredicate(field="action", values=("plan",), op="="),
            QueryFieldPredicate(field="action", values=("edit",), op="="),
        ),
        constraints=(QuerySequenceConstraint(kind="within", within_ms=60_000),),
    ),
    QueryTextPredicate(text="deploy with caveats"),
    QuerySemanticPredicate(text="deploy with caveats"),
    QueryLineagePredicate(seed_session_id="codex-session:abc123"),
    QueryLineagePredicate(seed_session_id="codex-session:abc123", logical=True),
)


@pytest.mark.parametrize("predicate", _PREDICATE_ROUNDTRIP_CASES, ids=lambda p: type(p).__name__)
def test_predicate_ast_roundtrip_is_lossless(predicate: QueryPredicate) -> None:
    ast = predicate_to_ast(predicate)
    reconstructed = ast_to_predicate(ast)
    assert reconstructed == predicate
    # And the reconstructed predicate must re-validate to an equal AST node,
    # not merely compare equal as a dataclass.
    assert predicate_to_ast(reconstructed) == ast


_EXPRESSION_CASES: tuple[str, ...] = (
    'repo:polylogue since:7d "json envelope"',
    "sessions where exists block(type:code) AND lineage:id:root",
    "sessions where logical:codex-session:root AND title:hit",
    'sessions where semantic:"query compiler" AND title:hit',
    'sessions where ~"deploy caveat" AND title:hit',
    "sessions where seq(action:file_edit -> action:shell AND output:failed)",
    "sessions where seq(action:file_edit ->[within:60s] action:shell)",
    "messages where role:assistant AND text:timeout",
    "messages where role:assistant | sort by time desc | limit 2 | offset 3",
    "messages where role:assistant | select role,text",
    # The named-metric aggregate: the hand-written mirror this schema replaced
    # rejected the producer's own output here, because it declared no `agg`
    # stage and no `agg_metrics` result key.
    "messages where role:assistant | agg count, avg:word_count, p90:word_count",
    "messages where role:assistant | group by role | agg count, sum:word_count",
    "actions where tool:Bash | group by tool | agg count, max:exit_code | limit 3",
    "delegations where mapping_state:resolved | group by basis | count",
    "sessions where repo:polylogue | messages where role:assistant | limit 5",
    "from result-set:stable-set | group by model | count",
    '{"repo": "polylogue", "limit": 5}',
    "messages between 5 and 20",
    "date between 2024-01-01 and 2024-06-01",
    "context-snapshots where session.repo:polylogue AND boundary:session_start",
    "actions where action:file_edit AND path:polylogue",
    "deploy failed today",
)


@pytest.mark.parametrize("expression", _EXPRESSION_CASES)
def test_explanation_payload_validates_against_canonical_ast(expression: str) -> None:
    explanation = explain_expression(expression)
    payload = explanation.to_payload()

    ast = explanation_payload_to_ast(payload)

    assert ast.schema_version == QUERY_AST_SCHEMA_VERSION
    assert payload["schema_version"] == QUERY_AST_SCHEMA_VERSION
    assert ast.source_text == expression
    assert ast.lowerer == explanation.lowerer

    if explanation.predicate is not None:
        assert ast.predicate is not None
        # The typed predicate sub-tree round-trips back to the exact
        # dataclass the parser produced, not just a structurally similar one.
        assert ast_to_predicate(ast.predicate) == explanation.predicate


def test_canonical_ast_schema_is_json_schema_serializable() -> None:
    """The schema powering ``devtools render openapi`` must build cleanly."""
    schema = QueryExpressionExplanationAst.model_json_schema(mode="serialization")
    assert schema["title"] == "QueryExpressionExplanationAst"
    # A representative nested predicate variant must be reachable from $defs
    # so OpenAPI consumers can resolve the full recursive predicate tree.
    defs = schema.get("$defs", {})
    assert "QueryBoolPredicateAst" in defs
    assert "QueryFieldPredicateAst" in defs
    assert "QueryUnitAggStageAst" in defs


def test_every_declaration_has_exactly_one_published_model() -> None:
    """The published model set is the declaration set -- no extras, no gaps."""

    assert set(published_ast_models()) == {declaration.name for declaration in _ALL_DECLARATIONS}


def test_canonical_ast_rejects_unknown_top_level_key() -> None:
    """Schema drift (an added/removed key) must fail loudly, not silently pass."""
    explanation = explain_expression("repo:polylogue")
    payload = dict(explanation.to_payload())
    payload["unexpected_new_field"] = "surprise"
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        explanation_payload_to_ast(payload)


def test_canonical_predicate_ast_rejects_unknown_kind() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(QueryPredicateAst)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "made-up"})
