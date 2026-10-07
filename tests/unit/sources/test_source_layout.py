"""Discovery reaches only each source's declared layout, never nested copies.

A full copy of a provider tree inside its own root (an agent git worktree of
``~/.claude/projects`` under ``.claude/worktrees/``) used to be admitted file
for file, because discovery walked every directory and admitted anything an
unanchored artifact-rule regex or a suffix matched. These tests pin each
canonical source's layout and the exclusion of nested copies.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from polylogue.sources.live.discovery import _source_path_steps
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import HOOK_CARRIER_PROVIDERS, WatchSource, daemon_watch_sources
from polylogue.sources.source_layout import (
    ANY_DEPTH,
    declared_source_layout,
    declared_source_layouts,
    layout_declaration_defects,
    source_layout_for,
)

SESSION = "00000000-0000-4000-8000-000000000001"

#: Synthetic canonical roots: artifact-rule patterns are written against the
#: provider's install paths, so the rule check needs roots of that shape.
SYNTHETIC_ROOTS = {
    "claude-code": "/home/user/.claude/projects",
    "claude-code-todos": "/home/user/.claude/todos",
    "claude-code-history": "/home/user/.claude",
    "codex": "/home/user/.codex/sessions",
    "codex-state": "/home/user/.codex",
    "codex-memories": "/home/user/.codex/memories",
    "gemini-cli": "/home/user/.gemini/tmp",
    "hermes": "/home/user/.hermes",
    "antigravity": "/home/user/.gemini/antigravity",
    "antigravity-cli": "/home/user/.gemini/antigravity-cli",
    **{f"{provider}-hooks": f"/state/polylogue/hooks/carriers/{provider}" for provider in HOOK_CARRIER_PROVIDERS},
}


def test_every_daemon_watch_source_declares_a_layout() -> None:
    """Every canonical source walks by its own declared layout, never a stand-in."""

    sources = daemon_watch_sources()
    assert sources
    assert all(source.layout.identity() == declared_source_layout(source.name).identity() for source in sources)
    assert {source.name for source in sources} <= set(declared_source_layouts())


def test_every_route_resolves_a_layout_from_the_source_name() -> None:
    """A canonical name walks by its declaration anywhere; any other name is an export drop.

    Anti-vacuity: resolving an export-only origin's directory by provider
    position, or a canonical name by format, turns the respective line red.
    """

    assert source_layout_for("codex") is declared_source_layout("codex")
    assert WatchSource(name="claude-code", root=Path("/synthetic")).layout is declared_source_layout("claude-code")
    for label in ("chatgpt", "claude-ai", "gemini", "operator-label"):
        layout = source_layout_for(label)
        assert layout.provider is None
        assert layout.artifact_kind(("export", "conversations.json")) == "export_drop"
        assert layout.artifact_kind(("readme.txt",)) is None
        assert layout.artifact_kind((".claude", "worktrees", "conversations.json")) is None


def test_layout_entries_agree_with_their_origin_artifact_rules() -> None:
    """Each entry admits its example, and the owning rule classifies it as the same kind.

    Breaking the agreement in either direction -- an entry naming a kind its
    provider's rule does not assign there, or a layout-only kind a rule
    claims -- is reported as a defect.
    """

    assert set(SYNTHETIC_ROOTS) == {
        name for name, layout in declared_source_layouts().items() if layout.provider is not None
    }
    assert layout_declaration_defects(SYNTHETIC_ROOTS) == ()


CLAUDE_CODE_CASES = (
    (f"-home-user-repo/{SESSION}.jsonl", "coordinator_session_stream"),
    ("-home-user-repo/agent-a1.jsonl", "coordinator_session_stream"),
    (f"-home-user-repo/{SESSION}/subagents/agent-a1.jsonl", "agent_transcript"),
    (f"-home-user-repo/{SESSION}/subagents/agent-a1.meta.json", "agent_sidecar_meta"),
    (f"-home-user-repo/{SESSION}/subagents/workflows/wf_1/agent-a1.jsonl", "agent_transcript"),
    (f"-home-user-repo/{SESSION}/subagents/workflows/wf_1/agent-a1.meta.json", "agent_sidecar_meta"),
    (f"-home-user-repo/{SESSION}/subagents/workflows/wf_1/journal.jsonl", "workflow_journal"),
    (f"-home-user-repo/{SESSION}/workflows/wf_1.json", "workflow_run_snapshot"),
    (f"-home-user-repo/{SESSION}/tool-results/toolu_1.txt", "tool_result_sidecar"),
    ("-home-user-repo/memory/MEMORY.md", "agent_memory_document"),
    ("-home-user-repo/memory/archive/older.md", "agent_memory_document"),
    ("-home-user-repo/sessions-index.json", "session_index"),
    # Outside the layout: nested copies, stray files and undeclared families.
    (f".claude/worktrees/agent-1/-home-user-repo/{SESSION}/subagents/agent-a1.jsonl", None),
    (f".claude/worktrees/agent-1/-home-user-repo/{SESSION}/tool-results/toolu_1.txt", None),
    (f".claude/worktrees/agent-1/-home-user-repo/{SESSION}.jsonl", None),
    (f"-home-user-repo/.claude/worktrees/agent-1/-home-user-repo/{SESSION}.jsonl", None),
    (f".git/objects/{SESSION}.jsonl", None),
    (f"-home-user-repo/{SESSION}/{SESSION}.jsonl", None),
    (f"-home-user-repo/{SESSION}/tool-results/hook-{SESSION}-stdout.txt", None),
    (f"-home-user-repo/{SESSION}/workflows/scripts/run.js", None),
    (f"-home-user-repo/{SESSION}/subagents/nested/agent-a1.jsonl", None),
    ("-home-user-repo/bridge-pointer.json", None),
    ("-home-user-repo/scratch/notes.md", None),
    (f"{SESSION}.jsonl", None),
)

OTHER_ORIGIN_CASES = (
    ("claude-code-todos", f"{SESSION}-agent-{SESSION}.json", "todo_snapshot"),
    ("claude-code-todos", f"{SESSION}.json.tmp.1.2", None),
    ("claude-code-todos", f"nested/{SESSION}.json", None),
    ("claude-code-history", "history.jsonl", "prompt_history_log"),
    ("claude-code-history", "projects/-home-user-repo/history.jsonl", None),
    ("claude-code-history", "settings.json", None),
    ("codex", "2026/01/02/rollout-2026-01-02T00-00-00-1.jsonl", "session_stream"),
    ("codex", "rollout-2026-01-02T00-00-00-1.jsonl", None),
    ("codex", "2026/01/02/copy/rollout-1.jsonl", None),
    ("codex-state", "state_5.sqlite", "database_member"),
    ("codex-state", "sqlite/codex-dev.db", "database_member"),
    ("codex-state", "session_index.jsonl", "session_index"),
    ("codex-state", "history.jsonl", "prompt_history_log"),
    ("codex-state", "codex-dev.db", None),
    ("codex-state", "worktrees/repo/.cache/mypy/3.14/cache.db", None),
    ("codex-state", "sessions/2026/01/02/rollout-1.jsonl", None),
    ("codex-state", "unrelated.sqlite", None),
    ("codex-memories", "MEMORY.md", "agent_memory_document"),
    ("codex-memories", "archive/MEMORY-1.md", "agent_memory_document"),
    ("codex-memories", ".git/description.md", None),
    ("gemini-cli", "project/chats/session-2026-01-01T00-00-1.json", "session_document"),
    ("gemini-cli", f"project/chats/{SESSION}/sub1.json", "subagent_session_document"),
    ("gemini-cli", "project/logs.json", "prompt_history_log"),
    ("gemini-cli", "project/tool-outputs/session-1/run_shell_command_1.txt", "tool_result_sidecar"),
    ("gemini-cli", "project/notes.json", None),
    ("gemini-cli", "logs.json", None),
    ("hermes", "state.db", "database_member"),
    ("hermes", "verification_evidence.db", "database_member"),
    ("hermes", "profiles/work/state.db", "database_member"),
    ("hermes", "sessions/session_1.json", "session_snapshot"),
    ("hermes", "sessions/saved/conversation_1.json", "session_snapshot"),
    ("hermes", "profiles/work/sessions/session_1.json", "session_snapshot"),
    ("hermes", "observability/nemo-relay/atif/trajectory-1.json", "atif_document"),
    ("hermes", "observability/nemo-relay/atof/events.jsonl", "atof_stream"),
    ("hermes", "auth.json", None),
    ("hermes", "backup.db", None),
    ("hermes", "hermes-agent/optional-skills/x/templates/prompt.json", None),
    ("hermes", "skills/creative/workflows/flow.json", None),
    ("antigravity", "conversations/c1.pb", "session_document"),
    ("antigravity", "brain/c1/task.md", "metadata_document"),
    ("antigravity", "brain/c1/task.md.metadata.json", "agent_sidecar_meta"),
    ("antigravity", "brain/c1/task.md.resolved", None),
    ("antigravity", "implicit/c1.pb", None),
    ("antigravity", "user_settings.pb", None),
    ("antigravity", "code_tracker/active/repo/notes.md", None),
    ("antigravity-cli", "conversations/00000000-0000-4000-8000-000000000001.db", "trajectory_store"),
    ("antigravity-cli", "conversation_summaries.db", None),
    ("antigravity-cli", "conversations/nested/c1.db", None),
    ("browser-capture", "chatgpt/session-0123456789ab.json", "browser_capture_envelope"),
    ("browser-capture", "browser-actions/a1/action.json", None),
    ("browser-capture", "not-a-provider/session-0123456789ab.json", None),
    ("inbox", "export.zip", "export_drop"),
    ("inbox", "export/conversations.json", "export_drop"),
    ("inbox", ".staging/export.json", None),
    ("inbox", "export/readme.txt", None),
    *(
        (f"{provider}-hooks", "2026-01-01/carrier-1.ndjson", "hook_event_carrier")
        for provider in HOOK_CARRIER_PROVIDERS
    ),
    *((f"{provider}-hooks", "carrier-1.ndjson", None) for provider in HOOK_CARRIER_PROVIDERS),
)


@pytest.mark.parametrize(
    ("name", "relative", "kind"),
    [("claude-code", relative, kind) for relative, kind in CLAUDE_CODE_CASES] + list(OTHER_ORIGIN_CASES),
)
def test_declared_layout_pins_each_artifact_position(name: str, relative: str, kind: str | None) -> None:
    layout = declared_source_layout(name)
    assert layout.artifact_kind(PurePosixPath(relative).parts) == kind


@pytest.mark.parametrize(
    ("relative", "reached"),
    [
        (".claude", False),
        (".claude/worktrees", False),
        (".git", False),
        ("-home-user-repo", True),
        (f"-home-user-repo/{SESSION}", True),
        (f"-home-user-repo/{SESSION}/subagents/workflows/wf_1", True),
        (f"-home-user-repo/{SESSION}/remote-agents", False),
        (f"-home-user-repo/{SESSION}/subagents/.ruff_cache", False),
        ("-home-user-repo/memory/archive", True),
        ("-home-user-repo/memory/.git", False),
        ("-home-user-repo/scratch/notes", False),
    ],
)
def test_claude_code_layout_reaches_only_declared_directories(relative: str, reached: bool) -> None:
    layout = declared_source_layout("claude-code")
    assert layout.admits_directory(PurePosixPath(relative).parts) is reached


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    return path


def _discover(source: WatchSource) -> tuple[set[Path], list[tuple[Path, str, str]], list[Path]]:
    decisions: list[tuple[Path, str, str]] = []
    scanned: list[Path] = []

    def scandir(directory: Path) -> Any:
        scanned.append(Path(directory))
        return os.scandir(directory)

    def record(path: Path, disposition: str, reason: str) -> None:
        decisions.append((path, disposition, reason))

    accepted = {
        path
        for path in _source_path_steps(source, (source,), after=None, scandir=scandir, on_disposition=record)
        if path is not None
    }
    return accepted, decisions, scanned


def test_nested_worktree_copy_of_claude_code_projects_is_never_walked(tmp_path: Path) -> None:
    """The real tree is admitted; its nested git-worktree copy is one excluded entry.

    Anti-vacuity: the previous recursive suffix-and-rule walk admitted the
    copied ``subagents/`` and ``tool-results/`` files and scanned every
    directory of the copy.
    """

    root = tmp_path / ".claude" / "projects"
    real = {
        _write(root / "-home-user-repo" / f"{SESSION}.jsonl"),
        _write(root / "-home-user-repo" / SESSION / "subagents" / "agent-a1.jsonl"),
        _write(root / "-home-user-repo" / SESSION / "subagents" / "agent-a1.meta.json"),
        _write(root / "-home-user-repo" / SESSION / "tool-results" / "toolu_1.txt"),
        _write(root / "-home-user-repo" / "memory" / "MEMORY.md"),
    }
    copy = root / ".claude" / "worktrees" / "agent-1"
    for path in real:
        _write(copy / path.relative_to(root))
    _write(root / ".git" / "objects" / "pack.jsonl")
    _write(root / "-home-user-repo" / "scratch" / "notes.md")

    source = WatchSource(name="claude-code", root=root, layout=declared_source_layout("claude-code"))
    accepted, decisions, scanned = _discover(source)

    assert accepted == real
    excluded = {(path, reason) for path, disposition, reason in decisions if disposition == "excluded"}
    assert (root / ".claude", "outside_declared_layout") in excluded
    assert (root / ".git", "outside_declared_layout") in excluded
    # A session-level directory is walked (session ids name it), but nothing
    # in it outside ``subagents/``, ``tool-results/`` or ``workflows/`` is.
    assert (root / "-home-user-repo" / "scratch" / "notes.md", "outside_declared_layout") in excluded
    assert not [path for path in scanned if path == root / ".claude" or root / ".claude" in path.parents]
    assert not [path for path, _, _ in decisions if copy in path.parents]
    for path in real:
        copied = copy / path.relative_to(root)
        assert not source.accepts(copied)
        assert deepest_source_for_path(copied, (source,)) is source
        assert not source.admits_directory(copied.parent)


def _layout_names() -> list[str]:
    return sorted(declared_source_layouts())


@pytest.mark.parametrize("name", _layout_names())
def test_every_layout_admits_its_own_positions_and_no_nested_copy(name: str, tmp_path: Path) -> None:
    """Each declared position is discovered; the same files under a hidden copy are not.

    A layout without :data:`ANY_DEPTH` also refuses the copy under a visible
    directory: its depth and position are exact.
    """

    layout = declared_source_layout(name)
    root = tmp_path / "root"
    root.mkdir()
    expected = {_write(root / entry.example) for entry in layout.entries}
    hidden_copy = root / ".worktrees" / "agent-1"
    for entry in layout.entries:
        _write(hidden_copy / entry.example)
    exact = all(ANY_DEPTH not in entry.segments for entry in layout.entries)
    visible_copy = root / "nested-copy" / "agent-1"
    if exact:
        for entry in layout.entries:
            _write(visible_copy / entry.example)

    source = WatchSource(name=name, root=root, layout=layout)
    accepted, decisions, _ = _discover(source)

    assert accepted == expected
    assert (root / ".worktrees", "excluded", "outside_declared_layout") in decisions
    if exact:
        assert not [path for path in accepted if visible_copy in path.parents]


def _sources_by_name(sources: tuple[WatchSource, ...]) -> dict[str, WatchSource]:
    return {source.name: source for source in sources}


def test_overlapping_canonical_roots_do_not_walk_each_others_trees(tmp_path: Path) -> None:
    """``codex-state`` (``~/.codex``) never descends into ``sessions/`` or ``memories/``.

    Anti-vacuity: a recursive suffix walk of ``~/.codex`` visits every rollout
    and reports it as owned by another source.
    """

    codex_home = tmp_path / ".codex"
    rollout = _write(codex_home / "sessions" / "2026" / "01" / "02" / "rollout-1.jsonl")
    memory = _write(codex_home / "memories" / "MEMORY.md")
    state = _write(codex_home / "state_5.sqlite")
    _write(codex_home / "worktrees" / "repo" / ".cache" / "mypy" / "cache.db")
    names: dict[str, Callable[[], Path]] = {
        "codex": lambda: codex_home / "sessions",
        "codex-state": lambda: codex_home,
        "codex-memories": lambda: codex_home / "memories",
    }
    sources = tuple(
        WatchSource(name=name, root=root(), layout=declared_source_layout(name)) for name, root in names.items()
    )
    by_name = _sources_by_name(sources)

    state_accepted, state_decisions, state_scanned = _discover(by_name["codex-state"])
    assert state_accepted == {state}
    assert codex_home / "sessions" not in state_scanned
    assert (codex_home / "sessions", "excluded", "outside_declared_layout") in state_decisions
    assert (codex_home / "worktrees", "excluded", "outside_declared_layout") in state_decisions
    assert _discover(by_name["codex"])[0] == {rollout}
    assert _discover(by_name["codex-memories"])[0] == {memory}
    assert deepest_source_for_path(rollout, sources) is by_name["codex"]
    assert deepest_source_for_path(memory, sources) is by_name["codex-memories"]


def test_one_shot_and_census_routes_walk_the_same_layout(tmp_path: Path) -> None:
    """A one-shot root named for a canonical source never admits its nested copy.

    Anti-vacuity: the previous one-shot walk (every supported suffix at any
    depth) returns the copied transcript too.
    """

    from polylogue.config import Source
    from polylogue.core.enums import Provider
    from polylogue.sources.source_walk import _resolve_source_paths, census_source_root

    root = tmp_path / "claude-code"
    session = _write(root / "-home-user-repo" / f"{SESSION}.jsonl")
    _write(root / ".claude" / "worktrees" / "agent-1" / "-home-user-repo" / f"{SESSION}.jsonl")
    _write(root / "notes" / "copy.jsonl")

    assert _resolve_source_paths(Source(name="claude-code", path=root)) == [session]
    assert census_source_root(root, provider=Provider.CLAUDE_CODE).candidate_count == 1


def test_live_watch_installs_only_on_layout_reachable_directories(tmp_path: Path) -> None:
    """The inotify set is the layout's reach: a nested copy gets no watch.

    Anti-vacuity: a recursive watch of the root (the previous behaviour)
    includes ``.claude/worktrees/...`` and ``.git``.
    """

    from types import SimpleNamespace

    from polylogue.sources.live.watcher import LiveWatcher

    root = tmp_path / ".claude" / "projects"
    _write(root / "-home-user-repo" / SESSION / "subagents" / "agent-a1.jsonl")
    _write(root / "-home-user-repo" / "memory" / "archive" / "old.md")
    _write(root / ".claude" / "worktrees" / "agent-1" / "-home-user-repo" / SESSION / "subagents" / "agent-a1.jsonl")
    _write(root / ".git" / "objects" / "pack")
    _write(root / "-home-user-repo" / "scratch" / "notes.md")
    source = WatchSource(name="claude-code", root=root)
    watcher = SimpleNamespace(_sources=(source,))

    watched = set(LiveWatcher.watched_directories(watcher))  # type: ignore[arg-type]

    assert watched == {
        root,
        root / "-home-user-repo",
        root / "-home-user-repo" / SESSION,
        root / "-home-user-repo" / SESSION / "subagents",
        root / "-home-user-repo" / "memory",
        root / "-home-user-repo" / "memory" / "archive",
        # A visible directory at the session depth may be a session directory.
        root / "-home-user-repo" / "scratch",
    }


@pytest.mark.asyncio
async def test_live_watch_rearms_with_a_new_reachable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory a layout reaches is watched as soon as it appears; others never are.

    Anti-vacuity: without the re-arm, the second arming never happens and the
    new session's ``subagents/`` directory stays unwatched; re-arming on any
    created directory also arms the nested copy.
    """

    import asyncio
    from types import SimpleNamespace

    import watchfiles

    from polylogue.sources.live.watcher import LiveWatcher

    root = tmp_path / "projects"
    project = root / "-home-user-repo"
    project.mkdir(parents=True)
    source = WatchSource(name="claude-code", root=root)
    armed: list[set[Path]] = []
    hints: list[Path] = []
    stop = asyncio.Event()
    new_session = project / SESSION / "subagents"
    copy = root / ".claude"

    async def fake_awatch(*paths: Path, stop_event: Any, **_kwargs: Any) -> Any:
        armed.append({Path(path) for path in paths})
        if len(armed) == 1:
            copy.mkdir()
            yield {(watchfiles.Change.added, str(copy))}
            assert not stop_event.is_set()
            new_session.mkdir(parents=True)
            yield {(watchfiles.Change.added, str(project / SESSION))}
            assert stop_event.is_set()
            return
        stop.set()
        return

    monkeypatch.setattr(watchfiles, "awatch", fake_awatch)
    watcher: Any = SimpleNamespace(
        _sources=(source,),
        _stop=stop,
        _watch_filter=lambda _change, _path: True,
        _note_intake_hint=hints.append,
    )
    watcher._source_for_directory = lambda path: LiveWatcher._source_for_directory(watcher, path)
    watcher.watched_directories = lambda: LiveWatcher.watched_directories(watcher)

    await LiveWatcher._watch_changes(watcher)

    assert armed[0] == {root, project}
    assert armed[1] == {root, project, project / SESSION, new_session}
    assert copy not in armed[1]
    assert project / SESSION in hints
