"""Every documented find/read invocation still resolves and selects the same rows.

The corpus is read out of the generated CLI reference rather than restated
here, so a newly documented invocation joins it automatically and a renamed
flag or dropped option turns the run red instead of quietly changing what the
reference promises.

Two guarantees, with distinct anti-vacuity conditions:

*Execution* — every documented invocation reaches a typed outcome (a result, or
the typed empty-selection envelope).  A dropped or renamed option produces a
Click usage error instead, which :func:`_assert_typed_outcome` rejects
specifically; asserting only "non-zero exit" would pass on that failure,
because the empty-selection envelope also exits 2.

*Selection* — for structural field predicates, the row set the CLI returns
equals the row set the canonical :class:`SessionQuerySpec` lowering returns
through the Python route.  A filter the CLI silently drops on its way to SQL
widens one side; a filter it applies twice or post-query narrows one side.
Free-text terms are deliberately excluded: the CLI searches FTS while the
Python route content-scans, so the two disagree on text for reasons that have
nothing to do with the shared filter set.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from devtools import repo_root
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.click_app import cli
from polylogue.config import Config
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder

_SEEDED_SESSION_ID = "claude-ai-export:ext-conv-0"

#: Documented placeholders rewritten to values the seeded archive resolves.
_PLACEHOLDERS: tuple[tuple[str, str], ...] = (
    ("id:abc", f"id:{_SEEDED_SESSION_ID}"),
    ("session:abc123", f"session:{_SEEDED_SESSION_ID}"),
    ("--id abc123", f"--id {_SEEDED_SESSION_ID}"),
    ("repo:github.com/Sinity/polylogue", "repo:polylogue"),
    ("session.repo:example-repo", "session.repo:polylogue"),
)

#: Documented invocations that are not executed here, and why.
_NOT_EXECUTED: dict[str, str] = {
    "polylogue find QUERY then ACTION": "dispatch template, not an invocation",
    "polylogue 'QUERY' then read": "dispatch template, not an invocation",
    "polylogue find --help": "help screen, covered by the help contract tests",
    "polylogue --latest find 'repo:polylogue' then read --to browser": "opens a browser",
    "polylogue find id:abc then read --to browser": "opens a browser",
    "polylogue find id:abc then continue --exec": "spawns the origin's own CLI",
    "polylogue find id:abc then delete --yes": "unconfirmed destructive write",
    "polylogue find 'repo:polylogue since:7d' then delete --yes --all": "unconfirmed destructive write",
    'polylogue find "urgent" then mark --tag-add review': "write route, covered by the mutation authority tests",
    "polylogue find id:abc then mark --tag-add reviewed": "write route, covered by the mutation authority tests",
    'polylogue find id:abc then mark --star --note "key insight"': (
        "write route, covered by the mutation authority tests"
    ),
    "polylogue find id:abc then mark --unstar --tag-remove reviewed": (
        "write route, covered by the mutation authority tests"
    ),
    "polylogue find id:abc then mark --pin": "write route, covered by the mutation authority tests",
    "polylogue find 'repo:polylogue since:7d' then mark --tag-add sprint --all": (
        "write route, covered by the mutation authority tests"
    ),
}

#: Structural selections whose row set both routes must agree on.
#:
#: A leading-dash clause (``-origin:x``) is absent deliberately: the root
#: parser routes ``find -origin:x`` to the bare-invocation status screen
#: instead of the query, which is a parsing defect upstream of the lowering
#: these cases cover.
_SELECTION_EXPRESSIONS: tuple[str, ...] = (
    "repo:polylogue",
    "origin:claude-ai-export",
    "origin:(claude-ai-export|chatgpt-export)",
    "repo:polylogue since:2020-01-01",
    "messages:>=2",
    "words:>=1",
    "root:true",
    "root:false",
    "cwd:/realm/project",
    "repo:polylogue origin:claude-ai-export",
)


def documented_invocations() -> tuple[str, ...]:
    """Return every documented ``polylogue`` invocation that names find or read."""
    reference = (repo_root() / "docs" / "cli-reference.md").read_text(encoding="utf-8")
    candidates = [span.strip() for span in re.findall(r"`([^`\n]+)`", reference)]
    for block in re.findall(r"```[a-z]*\n(.*?)```", reference, re.S):
        candidates.extend(line.strip().removeprefix("$ ").strip() for line in block.splitlines())

    found: list[str] = []
    for candidate in candidates:
        if not candidate.startswith("polylogue "):
            continue
        stripped = re.sub(r"\s+#.*$", "", candidate).strip()
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            continue
        if "find" not in tokens and "read" not in tokens:
            continue
        if stripped not in found:
            found.append(stripped)
    return tuple(found)


_EXECUTED = tuple(command for command in documented_invocations() if command not in _NOT_EXECUTED)


@pytest.fixture(scope="module")
def documented_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seed an archive whose rows satisfy the reference's example filters."""
    root = tmp_path_factory.mktemp("documented-corpus")
    initialize_active_archive_root(root)
    index_db = root / "index.db"
    for index in range(4):
        (
            SessionBuilder(index_db, f"conv-{index}")
            .provider("claude-ai" if index % 2 == 0 else "chatgpt")
            .title(f"Session {index}")
            .git_repository_url("https://github.com/Sinity/polylogue")
            .working_directories(["/realm/project/polylogue"])
            .add_message(f"m{index}a", role="user", text="hello alpha bravo urgent migration cost tracking")
            .add_message(f"m{index}b", role="assistant", text="charlie delta echo README.md")
            .save()
        )
    return root


def _invoke(archive_root: Path, argv: list[str]) -> Result:
    env = {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_DB_PATH": str(archive_root / "index.db"),
        "POLYLOGUE_NO_DAEMON": "1",
    }
    return CliRunner().invoke(cli, ["--no-daemon", *argv], env=env)


def _resolve_placeholders(command: str) -> list[str]:
    resolved = command
    for placeholder, replacement in _PLACEHOLDERS:
        resolved = resolved.replace(placeholder, replacement)
    return shlex.split(resolved)[1:]


#: Click prints its usage banner for typed domain refusals too ("No sessions
#: matched", "matched multiple sessions. Use --all"), so the banner does not
#: separate a refusal from a parse failure.  These fragments do.
_PARSE_FAILURES: tuple[str, ...] = (
    "No such option",
    "unexpected extra argument",
    "Missing argument",
    "Invalid value for",
    "does not take a value",
)


def _assert_typed_outcome(command: str, result: Result) -> None:
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise AssertionError(f"{command!r} raised {result.exception!r}")
    if result.exit_code == 0:
        return
    output = result.output or ""
    assert result.exit_code == 2, f"{command!r} exited {result.exit_code}: {output}"
    for fragment in _PARSE_FAILURES:
        assert fragment not in output, f"{command!r} no longer parses: {output}"


def test_every_documented_invocation_is_classified() -> None:
    """The reference and the not-executed table name exactly the same commands."""
    documented = set(documented_invocations())
    assert documented, "the generated reference documents no find/read invocation"
    unknown = sorted(set(_NOT_EXECUTED) - documented)
    assert not unknown, f"the not-executed table names commands the reference no longer documents: {unknown}"


@pytest.mark.parametrize("command", _EXECUTED, ids=lambda value: value[:70])
def test_documented_invocation_reaches_a_typed_outcome(documented_archive: Path, command: str) -> None:
    """Each documented invocation still parses and runs to a typed result."""
    _assert_typed_outcome(command, _invoke(documented_archive, _resolve_placeholders(command)))


def _cli_selection(archive_root: Path, expression: str) -> list[str]:
    result = _invoke(archive_root, ["--format", "json", "--limit", "50", "find", expression])
    _assert_typed_outcome(expression, result)
    payload = json.loads(result.output)
    selected = []
    for item in payload.get("items", []):
        identity = item.get("id") or item.get("session_id") or item.get("match", {}).get("session_id")
        if identity:
            selected.append(str(identity))
    return sorted(selected)


def _canonical_selection(archive_root: Path, expression: str) -> list[str]:
    config = Config(
        archive_root=archive_root,
        render_root=archive_root,
        sources=[],
        db_path=archive_root / "index.db",
    )
    spec = SessionQuerySpec.from_expression(expression)
    summaries = asyncio.run(spec.list_summaries(config))
    return sorted(str(summary.id) for summary in summaries)


@pytest.mark.parametrize("expression", _SELECTION_EXPRESSIONS)
def test_cli_selection_matches_the_canonical_lowering(documented_archive: Path, expression: str) -> None:
    """The CLI and the Python route push down the same structural filters."""
    assert _cli_selection(documented_archive, expression) == _canonical_selection(documented_archive, expression)
