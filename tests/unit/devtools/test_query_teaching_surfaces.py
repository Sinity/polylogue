"""Parser-gate concrete query examples that remain in generated/manual docs."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from polylogue.archive.query.expression import (
    ExpressionCompileError,
    compile_expression,
    parse_unit_source_expression,
)
from polylogue.archive.query.metadata import terminal_query_sources

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TERMINAL_SOURCES = frozenset(terminal_query_sources())
_NON_QUERY_ROOT_COMMANDS = frozenset(
    {
        "completion",
        "completions",
        "config",
        "export",
        "facets",
        "generate",
        "import",
        "mark",
        "mcp",
        "ops",
        "query-completions",
        "read",
        "serve",
    }
)
_GLOBAL_OPTIONS_WITH_VALUES = frozenset(
    {
        "--format",
        "-f",
        "--origin",
        "-o",
        "--referenced-path",
        "--retrieval-lane",
        "--since",
    }
)


def _shell_words(line: str) -> list[str]:
    return shlex.split(line.strip().removesuffix("\\").rstrip(), comments=False, posix=True)


def _trim_at_option(words: list[str]) -> list[str]:
    for index, word in enumerate(words):
        if word.startswith("--"):
            return words[:index]
    return words


def _search_reference_examples() -> tuple[tuple[int, str, str], ...]:
    rows: list[tuple[int, str, str]] = []
    path = _REPO_ROOT / "docs/search.md"
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip().startswith("polylogue "):
            continue
        words = _shell_words(line)[1:]
        while words and words[0].startswith("-"):
            option = words.pop(0)
            if option in _GLOBAL_OPTIONS_WITH_VALUES and words:
                words.pop(0)
            elif option == "--no-idf":
                continue
            else:
                break
        if not words or words[0] in _NON_QUERY_ROOT_COMMANDS or words[0] == "find":
            continue
        if len(words) >= 2 and words[0] in {*_TERMINAL_SOURCES, "sessions"} and words[1] == "where":
            expression = " ".join(_trim_at_option(words))
        else:
            cut = next((index for index, word in enumerate(words) if word in _NON_QUERY_ROOT_COMMANDS), len(words))
            expression = " ".join(words[:cut])
        if not expression:
            continue
        parser = (
            "unit"
            if (
                (words[0] in _TERMINAL_SOURCES and len(words) > 1 and words[1] == "where")
                or any(f"| {source} where" in expression for source in _TERMINAL_SOURCES)
            )
            else "session"
        )
        rows.append((lineno, parser, expression))
    return tuple(rows)


def _cli_reference_find_examples() -> tuple[tuple[int, str, str], ...]:
    rows: list[tuple[int, str, str]] = []
    path = _REPO_ROOT / "docs/cli-reference.md"
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip().startswith("polylogue find "):
            continue
        words = _shell_words(line)[2:]
        if not words or words[0].startswith("--"):
            continue
        cut = next(
            (index for index, word in enumerate(words) if word == "then" or word.startswith("--")),
            len(words),
        )
        expression = " ".join(words[:cut]).strip()
        if not expression or expression in {"QUERY", "<query>"} or "<query>" in expression:
            continue
        rows.append((lineno, "session", expression))
    return tuple(rows)


@pytest.mark.parametrize(
    ("lineno", "parser", "expression"),
    _search_reference_examples(),
    ids=lambda value: str(value),
)
def test_search_reference_concrete_queries_parse(lineno: int, parser: str, expression: str) -> None:
    """Changing a concrete search-guide query to a rejected form fails at its source line."""

    if parser == "unit":
        assert parse_unit_source_expression(expression) is not None, f"docs/search.md:{lineno}"
    else:
        compile_expression(expression)


@pytest.mark.parametrize(
    ("lineno", "_parser", "expression"),
    _cli_reference_find_examples(),
    ids=lambda value: str(value),
)
def test_cli_reference_find_queries_parse(lineno: int, _parser: str, expression: str) -> None:
    """Changing generated CLI help to teach a rejected query fails at its source line."""

    try:
        compile_expression(expression)
    except ExpressionCompileError as exc:
        raise AssertionError(f"docs/cli-reference.md:{lineno}: {expression}") from exc


def test_docs_audit_is_not_vacuous() -> None:
    assert len(_search_reference_examples()) >= 40
    assert len(_cli_reference_find_examples()) >= 40
