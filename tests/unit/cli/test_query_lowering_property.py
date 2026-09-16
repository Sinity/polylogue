"""The query DSL's lowering agrees with a reference evaluator on random expressions.

The grammar lowers to SQL. Individual lowering tests pin the cases someone
thought of; this compares the lowered result set against an independent Python
evaluator over the same seeded rows, for a seeded sample of random boolean
expressions. It is deliberately small in surface (origin and title leaves,
AND/OR/NOT, parentheses) and exact in oracle.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from polylogue.archive.query.expression import compile_expression
from polylogue.archive.query.filter_kwargs import plan_filter_kwargs
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
