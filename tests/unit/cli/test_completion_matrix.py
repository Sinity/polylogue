"""Per-shell coverage matrix for dynamic completers.

Every dynamic completer must work on every supported shell (bash, zsh,
fish). This test parametrizes the matrix and asserts that each
(completer, shell) pair returns at least one completion item without
raising — both against an empty archive (static and graceful-empty
behavior) and against a seeded archive (dynamic completers that read
SQLite).

The script-emit side of ``polylogue config completions --shell <SHELL>``
is covered by ``test_completions_contract.py``; this test covers the
runtime completion-handler protocol the script invokes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.shell_completion import (
    BashComplete,
    FishComplete,
    ShellComplete,
    ZshComplete,
)

from polylogue.cli.click_app import cli
from polylogue.operations.action_contracts import CompletionContext, action_completion_contexts

SUPPORTED_SHELLS: tuple[tuple[str, type[ShellComplete]], ...] = (
    ("bash", BashComplete),
    ("zsh", ZshComplete),
    ("fish", FishComplete),
)

# Each entry is (completer_label, partial-cmdline-without-trailing-incomplete).
# These pin the option/argument surface that wires each completer; if
# the surface changes (e.g. --provider renamed), update both sides.
STATIC_COMPLETERS: tuple[tuple[str, list[str]], ...] = (
    ("provider", ["--provider"]),
    ("retrieval_lane", ["--retrieval-lane"]),
    ("action", ["--action"]),
    ("action_sequence", ["--action-sequence"]),
    ("material_origin", ["--material-origin"]),
    ("message_type", ["messages", "--message-type"]),
    ("read_view", ["read", "--view"]),
    ("read_format", ["read", "--format"]),
)

CONTRACT_COMPLETION_COMMANDS: dict[CompletionContext, list[str]] = {
    "session_id": ["--id"],
}
SHELL_MATRIX_REQUIRED_CONTRACT_CONTEXTS: frozenset[CompletionContext] = frozenset({"session_id"})

DYNAMIC_COMPLETERS: tuple[tuple[str, list[str]], ...] = (
    *(
        (context, cwords)
        for context, cwords in CONTRACT_COMPLETION_COMMANDS.items()
        if context in action_completion_contexts()
    ),
    ("tag", ["--tag"]),
    ("repo", ["--repo"]),
    ("cwd_prefix", ["--cwd-prefix"]),
    ("tool", ["--tool"]),
)


def test_contract_completion_contexts_are_in_shell_matrix() -> None:
    """Action-contract completion contexts with shell completers are matrix-covered."""
    declared = set(action_completion_contexts())
    required = declared.intersection(SHELL_MATRIX_REQUIRED_CONTRACT_CONTEXTS)
    missing = required - set(CONTRACT_COMPLETION_COMMANDS)
    assert not missing, f"contract completion contexts missing shell matrix rows: {sorted(missing)}"


@contextmanager
def _shell_env(shell: str, cwords: list[str]) -> Iterator[None]:
    """Set the env vars the named Click completion class reads."""
    saved = {k: os.environ.get(k) for k in ("COMP_WORDS", "COMP_CWORD")}
    cmdline = " ".join(["polylogue", *cwords, ""])
    os.environ["COMP_WORDS"] = cmdline
    if shell == "fish":
        # Fish: COMP_CWORD is the partial word (empty for a fresh prompt).
        os.environ["COMP_CWORD"] = ""
    else:
        # Bash/Zsh: index of the incomplete word; trailing empty = len(cwords)+1.
        os.environ["COMP_CWORD"] = str(len(cwords) + 1)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_completion(
    shell: str,
    comp_cls: type[ShellComplete],
    cwords: list[str],
) -> list[tuple[str, str | None]]:
    with _shell_env(shell, cwords):
        comp = comp_cls(cli, {}, "polylogue", "_POLYLOGUE_COMPLETE")
        args, incomplete = comp.get_completion_args()
        items = comp.get_completions(args, incomplete)
        # Exercise format_completion so per-shell formatting doesn't crash.
        formatted = [comp.format_completion(item) for item in items]
        assert all(isinstance(line, str) for line in formatted), shell
    return [(it.value, it.help) for it in items]


def _run_completion_for_partial(
    shell: str,
    comp_cls: type[ShellComplete],
    cwords: list[str],
    incomplete: str,
) -> list[tuple[str, str | None]]:
    saved = {k: os.environ.get(k) for k in ("COMP_WORDS", "COMP_CWORD")}
    os.environ["COMP_WORDS"] = " ".join(["polylogue", *cwords, incomplete])
    os.environ["COMP_CWORD"] = incomplete if shell == "fish" else str(len(cwords) + 1)
    try:
        comp = comp_cls(cli, {}, "polylogue", "_POLYLOGUE_COMPLETE")
        args, actual_incomplete = comp.get_completion_args()
        assert actual_incomplete == incomplete
        items = comp.get_completions(args, actual_incomplete)
        formatted = [comp.format_completion(item) for item in items]
        assert all(isinstance(line, str) for line in formatted), shell
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return [(it.value, it.help) for it in items]


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
@pytest.mark.parametrize("label,cwords", STATIC_COMPLETERS, ids=[label for label, _ in STATIC_COMPLETERS])
def test_static_completers_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    label: str,
    cwords: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static completers (provider/lane/action/etc) return items on every shell.

    These do not depend on archive contents — they enumerate known
    enum or registry values — so they must always produce results.
    """
    # Point at an isolated empty archive root so the test doesn't read
    # the developer's real polylogue archive.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, cwords)
    assert items, f"static completer {label} produced no items on {shell}"


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_field_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root query completion suggests DSL fields from the shared grammar registry."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, [], "re")
    values = {value for value, _ in items}
    assert "repo:" in values


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_field_value_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root query completion suggests values through field completion sources."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, [], "origin:cla")
    values = {value for value, _ in items}
    assert "origin:claude-ai-export" in values
    assert "origin:claude-code-session" in values


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_structural_unit_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root query completion suggests structural units after ``exists``."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["exists"], "b")
    values = {value for value, _ in items}
    assert "block(" in values


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_structural_field_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root query completion suggests fields inside structural predicates."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["exists", "block("], "t")
    values = {value for value, _ in items}
    assert "text:" in values
    assert "type:" in values


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_terminal_source_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root query completion suggests terminal row sources."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["find"], "context")
    values = {value for value, _ in items}
    assert "context-snapshots where " in values


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_terminal_field_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``<source> where``, completion suggests terminal-unit fields."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["find", "observed-events", "where"], "d")
    item_map = dict(items)
    assert item_map == {"delivery_state:": item_map["delivery_state:"]}
    assert "Observed-event delivery state" in (item_map["delivery_state:"] or "")


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_terminal_field_completion_after_connector_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal-unit field completion keeps source context after connectors."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(
        shell,
        comp_cls,
        ["find", "context-snapshots", "where", "boundary:session_start", "AND"],
        "sess",
    )
    item_map = dict(items)
    assert "session.repo:" in item_map
    assert "Owning session scope" in (item_map["session.repo:"] or "")


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_root_option_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root option completion still belongs to Click outside query syntax."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, [], "--")
    assert "--help" in {value for value, _ in items}


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_count_operator_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readable count syntax completion suggests grammar-backed operators."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["messages"], "b")
    item_map = dict(items)
    assert item_map == {"between ": item_map["between "]}
    assert "messages between 5 and 20" in (item_map["between "] or "")


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_count_operator_empty_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a count field, completion is operator-only, not root-command mixed."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, ["messages"])
    values = {value for value, _ in items}
    assert {">=", "<=", "=", ">", "<", "between "}.issubset(values)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_date_operator_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readable date syntax completion suggests grammar-backed operators."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["date"], "b")
    item_map = dict(items)
    assert item_map == {"between ": item_map["between "]}
    assert "date between 2026-01-01 and 2026-02-01" in (item_map["between "] or "")


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_date_operator_empty_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``date``, completion is operator-only, not root-command mixed."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, ["date"])
    values = {value for value, _ in items}
    assert {">=", "<=", ">", "<", "between "}.issubset(values)
    assert "=" not in values


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_numeric_operator_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readable numeric syntax completion suggests grammar-backed operators."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["duration_ms"], "b")
    item_map = dict(items)
    assert item_map == {"between ": item_map["between "]}
    assert "duration_ms >= 60000" in (item_map["between "] or "")


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_numeric_operator_empty_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a numeric field, completion is operator-only, not root-command mixed."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, ["duration_ms"])
    values = {value for value, _ in items}
    assert {">=", "<=", "=", ">", "<", "between "}.issubset(values)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_then_connector_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query completion suggests the ``then`` connector after query text."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["find", "id:abc"], "th")
    item_map = dict(items)
    assert "then" in item_map
    assert item_map["then"] == "Connect query results to a verb/action."


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_quoted_query_then_connector_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quoted query expressions keep their token shape when completing ``then``."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["find", "'id:abc'"], "th")
    item_map = dict(items)
    assert item_map == {"then": "Connect query results to a verb/action."}


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_then_action_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``then``, completion is action-contract backed, not root-command mixed."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["find", "id:abc", "then"], "s")
    item_map = dict(items)

    assert "select" in item_map
    assert item_map["select"] is not None and "input=query_result_set" in item_map["select"]
    assert set(item_map) >= {"select"}
    assert all(description is None or "input=query_result_set" in description for description in item_map.values())


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_quoted_query_then_action_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a quoted query and ``then``, action completion stays contract-backed."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, ["find", "'id:abc'", "then"], "s")
    item_map = dict(items)

    assert "select" in item_map
    assert item_map["select"] is not None and "input=query_result_set" in item_map["select"]
    assert all(description is None or "input=query_result_set" in description for description in item_map.values())


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_completion_marks_destructive_actions_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root action completion is backed by contracts and marks destructive actions."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion_for_partial(shell, comp_cls, [], "de")
    item_map = dict(items)
    assert "delete" in item_map
    assert item_map["delete"] is not None and item_map["delete"].startswith("DANGER:")


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_read_view_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``find QUERY then read --view``, completion stays on read-view profiles."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, ["find", "id:abc", "then", "read", "--view"])
    item_map = dict(items)

    assert {"messages", "recovery", "context-pack"}.issubset(item_map)
    assert item_map["messages"] is not None and "Messages:" in item_map["messages"]


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_read_destination_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``find QUERY then read``, --to completion exposes supported destinations."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = dict(_run_completion(shell, comp_cls, ["find", "id:abc", "then", "read", "--to"]))

    assert {"terminal", "stdout", "browser", "clipboard", "file"}.issubset(items)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_read_format_completion_uses_selected_view_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``find QUERY then read``, --format completion still narrows by view."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, ["find", "id:abc", "then", "read", "--view", "raw", "--format"])

    assert dict(items) == {"json": "Supported by read --view raw"}


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_mutating_guard_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating/destructive action completions expose guards after ``then``."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    mark_items = dict(_run_completion_for_partial(shell, comp_cls, ["find", "id:abc", "then", "mark"], "--"))
    delete_items = dict(_run_completion_for_partial(shell, comp_cls, ["find", "id:abc", "then", "delete"], "--"))

    assert {"--tag-add", "--all", "--first"}.issubset(mark_items)
    assert {"--dry-run", "--yes", "--all"}.issubset(delete_items)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_mark_candidates_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mark candidates`` subcommands complete inside the query-action grammar."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = dict(_run_completion(shell, comp_cls, ["find", "id:abc", "then", "mark", "candidates"]))

    assert {"list", "accept", "reject", "defer", "supersede"}.issubset(items)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_query_action_continue_candidates_completion_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``continue --candidates`` options complete after ``find QUERY then`` routing."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = dict(
        _run_completion_for_partial(shell, comp_cls, ["find", "id:abc", "then", "continue", "--candidates"], "--")
    )

    assert {"--repo", "--cwd", "--recent", "--limit", "--format"}.issubset(items)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
def test_read_format_completion_uses_selected_view_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read --format completion narrows to the selected read-view profile."""

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    items = _run_completion(shell, comp_cls, ["read", "--view", "raw", "--format"])
    item_map = dict(items)

    assert item_map == {"json": "Supported by read --view raw"}


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
@pytest.mark.parametrize("label,cwords", DYNAMIC_COMPLETERS, ids=[label for label, _ in DYNAMIC_COMPLETERS])
def test_dynamic_completers_empty_archive_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    label: str,
    cwords: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic completers degrade gracefully (empty list, no traceback) on empty archive."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    # Returns [] when DB is missing — must not raise.
    items = _run_completion(shell, comp_cls, cwords)
    assert isinstance(items, list)


@pytest.mark.parametrize("shell,comp_cls", SUPPORTED_SHELLS, ids=[s for s, _ in SUPPORTED_SHELLS])
@pytest.mark.parametrize("label,cwords", DYNAMIC_COMPLETERS, ids=[label for label, _ in DYNAMIC_COMPLETERS])
def test_dynamic_completers_seeded_archive_per_shell(
    shell: str,
    comp_cls: type[ShellComplete],
    label: str,
    cwords: list[str],
    corpus_seeded_db: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a seeded archive, every dynamic completer returns at least one item on every shell.

    This is the core coverage claim of #1271 — providers/source-families,
    session IDs, tags, repos, actions, tools all complete in all
    three shells. Repos/tags/cwd may legitimately yield zero from the
    default synthetic corpus, so we only require session_id and
    tool (which are always populated) to be non-empty, while still
    asserting the call itself succeeds for all completers.
    """
    # ``corpus_seeded_db`` ingests through the archive pipeline and points
    # ``POLYLOGUE_ARCHIVE_ROOT`` at the archive root that holds the seeded
    # ``index.db`` — exactly the store ``active_index_db_path()`` resolves
    # for the completers, so no extra staging is required.
    corpus_seeded_db(providers=("chatgpt", "claude-ai"), count=3, seed=1271)

    items = _run_completion(shell, comp_cls, cwords)
    # The call must succeed for every completer.
    assert isinstance(items, list)
    # Guarantee: completers whose values come from messages/sessions
    # always have data in a seeded archive.
    if label in {"session_id"}:
        assert items, f"{label} returned no items on seeded archive ({shell})"
