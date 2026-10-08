"""The query DSL's lowering agrees with a reference evaluator on random expressions.

The grammar lowers to SQL. Individual lowering tests pin the cases someone
thought of; this compares the lowered result set against an independent Python
evaluator over the same seeded rows, for a seeded sample of random boolean
expressions. It is deliberately small in surface (origin and title leaves,
AND/OR/NOT, parentheses) and exact in oracle.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.expression import ExpressionCompileError, compile_expression
from polylogue.archive.query.filter_kwargs import plan_filter_kwargs
from polylogue.archive.query.predicate import predicate_from_payload
from polylogue.core.query_identity import JsonValue, canonical_query_plan
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder

#: (builder id, provider token, public origin token, title)
_CORPUS: tuple[tuple[str, str, str, str], ...] = (
    ("alpha", "claude-code", "claude-code-session", "needle"),
    ("bravo", "claude-code", "claude-code-session", "haystack"),
    ("charlie", "chatgpt", "chatgpt-export", "needle"),
    ("delta", "chatgpt", "chatgpt-export", "thread"),
    ("echo", "codex", "codex-session", "haystack"),
    ("foxtrot", "codex", "codex-session", "thread"),
)

_ORIGINS = tuple(sorted({origin for _id, _provider, origin, _title in _CORPUS}))
_TITLES = tuple(sorted({title for _id, _provider, _origin, title in _CORPUS}))

#: Depth of the generated expression tree. Three levels already mixes both
#: operators with negation under parentheses, which is where precedence and
#: NULL handling go wrong.
_MAX_DEPTH = 3
_SAMPLE_SIZE = 40
_SEED = 20260916


def _leaf(rng: random.Random) -> tuple[str, object]:
    if rng.random() < 0.5:
        value = rng.choice(_ORIGINS)
        return f"origin:{value}", lambda origin, _title, value=value: origin == value
    value = rng.choice(_TITLES)
    return f"title:{value}", lambda _origin, title, value=value: title == value


def _expression(rng: random.Random, depth: int) -> tuple[str, object]:
    if depth <= 0:
        return _leaf(rng)
    choice = rng.random()
    if choice < 0.35:
        return _leaf(rng)
    if choice < 0.5:
        text, predicate = _expression(rng, depth - 1)
        return f"NOT ({text})", lambda origin, title, predicate=predicate: not predicate(origin, title)
    left_text, left = _expression(rng, depth - 1)
    right_text, right = _expression(rng, depth - 1)
    if choice < 0.75:
        return (
            f"({left_text}) AND ({right_text})",
            lambda origin, title, left=left, right=right: left(origin, title) and right(origin, title),
        )
    return (
        f"({left_text}) OR ({right_text})",
        lambda origin, title, left=left, right=right: left(origin, title) or right(origin, title),
    )


@pytest.fixture
def seeded_query_archive(workspace_env: dict[str, Path]) -> Path:
    index_db = workspace_env["archive_root"] / "index.db"
    for native_id, provider, _origin, title in _CORPUS:
        (
            SessionBuilder(index_db, native_id)
            .provider(provider)
            .title(title)
            .add_message(f"m-{native_id}", role="user", text="body")
            .save()
        )
    return index_db.parent


def _expected(predicate: object) -> set[str]:
    return {
        f"{origin}:ext-{native_id}"
        for native_id, _provider, origin, title in _CORPUS
        if predicate(origin, title)  # type: ignore[operator]
    }


def test_lowered_expressions_match_a_reference_evaluator(seeded_query_archive: Path) -> None:
    """Anti-vacuity: swap AND/OR precedence in the lowering, or invert one leaf's
    comparison, and a generated expression disagrees with the Python oracle.
    Seeded, so a disagreement reproduces from the printed expression."""
    rng = random.Random(_SEED)
    checked = 0
    with ArchiveStore.open_existing(seeded_query_archive) as archive:
        for _index in range(_SAMPLE_SIZE):
            text, predicate = _expression(rng, _MAX_DEPTH)
            spec = compile_expression(text)
            kwargs = plan_filter_kwargs(spec.to_plan())
            rows = archive.list_summaries(limit=100, **kwargs)
            count = archive.count_sessions(**kwargs)

            observed = {row.session_id for row in rows}
            assert observed == _expected(predicate), f"lowering disagreed for {text!r}"
            assert count == len(observed), f"count disagreed with the page for {text!r}"
            checked += 1

    assert checked == _SAMPLE_SIZE


def test_the_generator_produces_both_operators_and_negation() -> None:
    """Without this the sample could be all leaves and the property vacuous."""
    rng = random.Random(_SEED)
    texts = [_expression(rng, _MAX_DEPTH)[0] for _index in range(_SAMPLE_SIZE)]

    assert any(" AND " in text for text in texts)
    assert any(" OR " in text for text in texts)
    assert any(text.startswith("NOT ") or " NOT " in text for text in texts)


def test_the_reference_evaluator_separates_the_corpus() -> None:
    """Every generated leaf must be able to change the answer."""
    for origin in _ORIGINS:
        matched = _expected(lambda row_origin, _title, origin=origin: row_origin == origin)
        assert 0 < len(matched) < len(_CORPUS)
    for title in _TITLES:
        matched = _expected(lambda _origin, row_title, title=title: row_title == title)
        assert 0 < len(matched) < len(_CORPUS)


@pytest.mark.parametrize("field", ["id", "session", "title", "origin"])
@pytest.mark.parametrize("negated", [False, True])
@pytest.mark.parametrize("source_prefix", ["", "sessions where "])
def test_compact_field_alternatives_select_the_same_rows_as_explicit_or(
    seeded_query_archive: Path, field: str, negated: bool, source_prefix: str
) -> None:
    if field in {"id", "session"}:
        values = ("claude-code-session:ext-alpha", "claude-code-session:ext-bravo")
        expected = set(values)
    elif field == "title":
        values = ("needle", "haystack")
        expected = _expected(lambda _origin, title: title in values)
    else:
        values = ("claude-code-session", "codex-session")
        expected = _expected(lambda origin, _title: origin in values)
    if negated:
        expected = _expected(lambda _origin, _title: True) - expected
    prefix = "NOT " if negated else ""
    expressions = (
        f"{source_prefix}{prefix}{field}:({'|'.join(values)})",
        f"{source_prefix}{prefix}({field}:{values[0]} OR {field}:{values[1]})",
    )
    with ArchiveStore.open_existing(seeded_query_archive) as archive:
        for expression in expressions:
            spec = compile_expression(expression)
            if spec.boolean_predicate is not None:
                persisted = canonical_query_plan(
                    cast(dict[str, JsonValue], spec.boolean_predicate.to_payload()),
                    grain="session",
                    lane="dialogue",
                    rank_policy="mixed",
                )["ast"]
                spec = replace(spec, boolean_predicate=predicate_from_payload(cast(dict[str, object], persisted)))
            else:
                assert field == "origin" and not negated and not source_prefix
            kwargs = plan_filter_kwargs(spec.to_plan())
            observed = {row.session_id for row in archive.list_summaries(limit=100, **kwargs)}
            assert observed == expected
            assert archive.count_sessions(**kwargs) == len(expected)


def test_compact_scalar_alternatives_retain_other_selection_filters(seeded_query_archive: Path) -> None:
    with ArchiveStore.open_existing(seeded_query_archive) as archive:
        for expression in (
            "body origin:codex-session title:(haystack|thread)",
            "sessions where ~body AND origin:codex-session AND (title:haystack OR title:thread)",
        ):
            kwargs = plan_filter_kwargs(compile_expression(expression).to_plan())
            expected = {"codex-session:ext-echo", "codex-session:ext-foxtrot"}
            assert {row.session_id for row in archive.list_summaries(limit=100, **kwargs)} == expected
            assert archive.count_sessions(**kwargs) == len(expected)


@pytest.mark.parametrize("source_prefix", ["", "sessions where "])
def test_scalar_alternative_lowering_retains_unsupported_field_refusal(source_prefix: str) -> None:
    with pytest.raises(ExpressionCompileError) as caught:
        compile_expression(f"{source_prefix}action_text:(alpha|beta)")
    assert caught.value.field == "action_text"


@pytest.mark.parametrize("field", ["id", "session", "title"])
@pytest.mark.parametrize("source_prefix", ["", "sessions where "])
@pytest.mark.parametrize("negated", [False, True])
def test_quoted_scalar_literals_preserve_pipes_and_whitespace(
    workspace_env: dict[str, Path], field: str, source_prefix: str, negated: bool
) -> None:
    import json

    root = workspace_env["archive_root"]
    ids: list[str] = []
    for native, title in (("opaque|pipe ", " alpha|beta "), ("alpha", "alpha"), ("beta", "beta")):
        builder = SessionBuilder(root / "index.db", native).provider("codex").title(title)
        builder.add_message(text="synthetic literal evidence").save()
        ids.append(builder.native_session_id())
    value = " alpha|beta " if field == "title" else ids[0]
    expression = f"{source_prefix}{'NOT ' if negated else ''}{field}:{json.dumps(value)}"
    kwargs = plan_filter_kwargs(compile_expression(expression).to_plan())
    expected = set(ids[1:]) if negated else {ids[0]}
    with ArchiveStore.open_existing(root) as archive:
        assert {row.session_id for row in archive.list_summaries(limit=100, **kwargs)} == expected
        assert archive.count_sessions(**kwargs) == len(expected)


@pytest.mark.parametrize("value", ["project ", " alpha|beta "])
def test_literal_repository_operands_and_facets(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    import json

    from polylogue.api.archive import _archive_aggregate_facet_families
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.storage.sqlite.connection import open_connection

    root = workspace_env["archive_root"]
    ids = []
    for native, label in (("literal", value), ("neighbor", value.strip().split("|")[0])):
        builder = SessionBuilder(root / "index.db", native).provider("codex").git_repository_url(native)
        builder.add_message(text="synthetic repository evidence").save()
        ids.append(builder.native_session_id())
        with open_connection(root / "index.db") as conn:
            conn.execute("UPDATE repos SET repo_name=? WHERE origin_url=?", (label, native))
            conn.commit()
    with ArchiveStore.open_existing(root) as archive:
        for spec in (
            compile_expression(f"repo:{json.dumps(value)}"),
            compile_expression(f"sessions where repo:{json.dumps(value)}"),
            SessionQuerySpec.from_params({"repo": value}),
        ):
            kwargs = plan_filter_kwargs(spec.to_plan())
            assert {row.session_id for row in archive.list_summaries(limit=100, **kwargs)} == {ids[0]}
            assert archive.count_sessions(**kwargs) == 1
    with open_connection(root / "index.db") as conn:
        assert _archive_aggregate_facet_families(conn, session_ids=None)["repos"] == {
            value: 1,
            value.strip().split("|")[0]: 1,
        }
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli
    from tests.infra.daemon_operations import cli_daemon_archive

    with cli_daemon_archive(root, monkeypatch):
        result = CliRunner().invoke(cli, ["--plain", "--format", "json", "--repo", value, "find"])
    assert result.exit_code == 0, result.output
    assert ids[0] in result.output
    assert ids[1] not in result.output


def test_field_explain_preserves_public_quoted_flag() -> None:
    import json

    from click.testing import CliRunner

    from polylogue.archive.query.expression import explain_expression
    from polylogue.cli.click_app import cli

    for expression, quoted in (('title:"alpha|beta"', True), ("title:alpha", False)):
        payload = explain_expression(expression).clauses[0].to_payload()
        assert payload.get("quoted", False) is quoted
        result = CliRunner().invoke(cli, ["--plain", "--format", "json", "--explain", "find", expression])
        assert result.exit_code == 0, result.output
        decoded = json.loads(result.output)
        assert decoded["ast"]["clauses"][0].get("quoted", False) is quoted
        assert decoded["clauses"][0].get("quoted", False) is quoted


@pytest.mark.parametrize(
    "value, expected",
    [("project ,other", ("project ", "other")), (("project ", "literal,comma"), ("project ", "literal,comma"))],
)
def test_repository_csv_preserves_segments_and_typed_members(value: object, expected: tuple[str, ...]) -> None:
    from polylogue.archive.query.spec import SessionQuerySpec, split_repo_names
    from polylogue.archive.query.unit_results import query_unit_session_filters

    assert split_repo_names(value) == expected
    assert SessionQuerySpec.from_params({"repo": value}).repo_names == expected
    assert query_unit_session_filters(repo=value)["repo_names"] == expected


def test_explicit_repository_alternation_remains_or() -> None:
    assert compile_expression("repo:(project|other)").repo_names == ("project", "other")
    assert compile_expression('repo:"pipe|comma, "').repo_names == ("pipe|comma, ",)
