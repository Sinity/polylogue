"""Every query example the help surfaces show is one the grammar still runs.

Syntax cards, completions and the discovery catalog are generated from two
declarations — :data:`EXPRESSION_FIELD_REGISTRY` (one example per field token)
and :data:`QUERY_DISCOVERY_EXAMPLES` (the shipped corpus).  Nothing else binds
those strings to the parser, so a grammar change that renames a token or
tightens a value leaves the help offering an example that no longer runs.

The check is compile *and* execute against a seeded archive: compiling alone
would pass an expression that lowers to a filter no reader accepts.  Row counts
are deliberately not asserted — the contract is that the example resolves to a
real selection, not that this fixture happens to match it.

Anti-vacuity: :func:`test_every_declared_field_token_contributes_an_example`
keeps the corpus bound to the registry, so a token added without an example is
red rather than silently unexercised, and
:func:`test_the_not_executed_table_names_only_live_examples` deletes an
exemption the moment the example it excuses is gone.
"""

from __future__ import annotations

import asyncio

import pytest

from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
from polylogue.archive.query.expression import compile_expression
from polylogue.archive.query.metadata import EXPRESSION_FIELD_REGISTRY
from polylogue.config import Config
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder

#: Declared examples not executed here, and why.
_NOT_EXECUTED: dict[str, str] = {
    "near:id:abc123": "session-seeded similarity needs stored embeddings for the referenced session",
    "near:id:example-origin:session-001": (
        "session-seeded similarity needs stored embeddings for the referenced session"
    ),
}


def _field_token_examples() -> tuple[tuple[str, str], ...]:
    """Return ``(token, expression)`` for every example a syntax card shows.

    A registry example may offer alternative spellings separated by ``|``
    outside any field value; each alternative is its own invocation.
    """
    rows: list[tuple[str, str]] = []
    for token, info in sorted(EXPRESSION_FIELD_REGISTRY.items()):
        for alternative in (part.strip() for part in info["example"].split(" | ")):
            if alternative:
                rows.append((token, alternative))
    return tuple(rows)


def _discovery_examples() -> tuple[tuple[str, str], ...]:
    """Return ``(key, expression)`` for the session-scope discovery corpus.

    Unit-source rows are excluded: they lower through the unit parser, not the
    session lowering this module executes.  Templated rows carry placeholders
    that the parameterization tests substitute and execute separately.
    """
    return tuple(
        (row.key, row.expression) for row in QUERY_DISCOVERY_EXAMPLES if row.parser == "session" and not row.template
    )


_EXECUTABLE: tuple[tuple[str, str], ...] = tuple(
    (label, expression)
    for label, expression in (*_field_token_examples(), *_discovery_examples())
    if expression not in _NOT_EXECUTED
)


@pytest.fixture(scope="module")
def example_archive(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Seed an archive whose rows the declared examples can resolve against."""
    root = tmp_path_factory.mktemp("help-examples")
    initialize_active_archive_root(root)
    index_db = root / "index.db"
    for index in range(3):
        (
            SessionBuilder(index_db, f"conv-{index}")
            .provider("claude-ai" if index % 2 == 0 else "chatgpt")
            .title(f"Session {index}")
            .git_repository_url("https://github.com/example/example-repo")
            .working_directories(["/workspace/example-repo"])
            .add_message(f"m{index}a", role="user", text="alpha timeout error review")
            .add_message(f"m{index}b", role="assistant", text="bravo warning README.md")
            .save()
        )
    return Config(archive_root=root, render_root=root, sources=[], db_path=index_db)


@pytest.mark.parametrize(
    ("label", "expression"),
    _EXECUTABLE,
    ids=[f"{label}:{expression}"[:70] for label, expression in _EXECUTABLE],
)
def test_a_declared_example_compiles_and_executes(example_archive: Config, label: str, expression: str) -> None:
    """The help shows nothing the parser and the readers cannot run today."""
    spec = compile_expression(expression)
    asyncio.run(spec.list_summaries(example_archive))


def test_every_declared_field_token_contributes_an_example() -> None:
    """A field token with no example is a token no card can teach."""
    exercised = {token for token, _ in _field_token_examples()}
    assert sorted(set(EXPRESSION_FIELD_REGISTRY) - exercised) == []


def test_the_corpus_is_not_empty() -> None:
    """An empty corpus would make every example assertion vacuously true."""
    assert len(_EXECUTABLE) >= len(EXPRESSION_FIELD_REGISTRY)


def test_the_not_executed_table_names_only_live_examples() -> None:
    """An exemption for an example that no longer ships is dead text."""
    declared = {expression for _, expression in (*_field_token_examples(), *_discovery_examples())}
    assert sorted(set(_NOT_EXECUTED) - declared) == []
