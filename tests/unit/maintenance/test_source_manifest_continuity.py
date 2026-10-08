from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import polylogue.maintenance.source_manifest_continuity as continuity
from polylogue.maintenance.source_manifest_continuity import (
    FrontierState,
    SourceContinuityError,
    SourceDeclaration,
    SourceRole,
    build_source_frontier,
    canonical_source_declarations,
)


def _source(tmp_path: Path, name: str = "source") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    (root / "two.json").write_text("two", encoding="utf-8")
    return root


def test_canonical_declaration_contains_all_source_roles_once(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("hooks", "legacy", "restored", "queue", "attachments", "exports", "live")]
    for root in roots:
        root.mkdir()
    declarations = canonical_source_declarations(
        hook_primary=roots[0],
        hook_legacy=[roots[1]],
        restored_spools=[roots[2]],
        browser_queue=roots[3],
        attachments=roots[4],
        exports=[roots[5]],
        live_sources=[roots[6]],
    )
    assert {declaration.source_id for declaration in declarations} == {
        "hooks-primary",
        "hooks-legacy-0",
        "restored-spool-0",
        "browser-queue",
        "attachments",
        "export-0",
        "live-source-0",
    }
    assert sum(declaration.role is SourceRole.SPOOL for declaration in declarations) == 3


def test_frontier_uses_declared_source_layout_and_excludes_spool_bookkeeping(tmp_path: Path) -> None:
    root = tmp_path / "browser-capture"
    (root / "chatgpt").mkdir(parents=True)
    (root / "chatgpt" / "session-0123456789ab.json").write_text("{}", encoding="utf-8")
    (root / "browser-actions").mkdir()
    (root / "browser-actions" / "action.json").write_text("{}", encoding="utf-8")

    frontier = build_source_frontier(
        (SourceDeclaration("capture", SourceRole.DIRECTORY, root, True, "browser-capture"),)
    )

    assert frontier.complete
    assert [member.coordinate for member in frontier.members] == ["chatgpt/session-0123456789ab.json"]


def test_configured_frontier_includes_browser_source_and_hook_spools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import config, paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    archive = tmp_path / "archive"
    browser = tmp_path / "browser-capture"
    (browser / "chatgpt").mkdir(parents=True)
    (browser / "chatgpt" / "session-0123456789ab.json").write_text("{}", encoding="utf-8")
    hooks = tmp_path / "hooks"
    for provider in ("claude-code", "codex", "hermes"):
        (hooks / "carriers" / provider).mkdir(parents=True)
    carrier = hooks / "carriers" / "claude-code" / "2026-10-07" / "123.ndjson"
    carrier.parent.mkdir(parents=True, exist_ok=True)
    carrier.write_text("{}\n", encoding="utf-8")
    pending = hooks / "pending" / "2026-10-07" / "event.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        config,
        "resolve_runtime_config",
        lambda: SimpleNamespace(sources=(SimpleNamespace(name="browser-capture", path=browser),)),
    )
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    monkeypatch.setattr(
        "polylogue.sources.hooks.hook_spool_sources",
        lambda: (SimpleNamespace(source_id="primary-hook-spool", root=hooks),),
    )

    frontier = continuity.configured_source_frontier(archive)

    assert frontier.complete
    assert {declaration.source_id for declaration in frontier.declarations} == {
        "configured:browser-capture",
        "primary-hook-spool:carrier:claude-code",
        "primary-hook-spool:carrier:codex",
        "primary-hook-spool:carrier:hermes",
        "primary-hook-spool:pending",
    }
    assert frontier.item_count == 3


def test_configured_frontier_omits_never_created_hook_spool_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    home = tmp_path / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    (codex / "history.jsonl").write_text('{"id":"synthetic"}\n', encoding="utf-8")
    archive = tmp_path / "archive"
    # Production's primary spool root may exist before its carrier and pending
    # children are first written.
    (archive / "hooks").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)

    frontier = continuity.configured_source_frontier(archive)

    assert frontier.complete
    assert "configured:codex-state" in {row.source_id for row in frontier.declarations}
    assert not any(":carrier:" in row.source_id or row.source_id.endswith(":pending") for row in frontier.declarations)


def test_configured_frontier_refuses_an_unavailable_primary_hook_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "history.jsonl").write_text('{"id":"synthetic"}\n', encoding="utf-8")
    archive = tmp_path / "archive"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)

    with pytest.raises(SourceContinuityError, match="hook spool root is unavailable"):
        continuity.configured_source_frontier(archive)


def test_configured_frontier_keeps_broken_hook_child_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "history.jsonl").write_text('{"id":"synthetic"}\n', encoding="utf-8")
    archive = tmp_path / "archive"
    broken = archive / "hooks" / "carriers" / "codex"
    broken.parent.mkdir(parents=True)
    broken.symlink_to(archive / "missing-carrier", target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)

    frontier = continuity.configured_source_frontier(archive)

    assert not frontier.complete
    assert frontier.root_states["primary-hook-spool:carrier:codex"] is FrontierState.UNAVAILABLE


def test_configured_frontier_keeps_unreadable_hook_child_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "history.jsonl").write_text('{"id":"synthetic"}\n', encoding="utf-8")
    archive = tmp_path / "archive"
    unreadable = archive / "hooks" / "carriers" / "codex"
    unreadable.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    original_lstat = Path.lstat

    def deny_child(path: Path) -> os.stat_result:
        if path == unreadable:
            raise PermissionError("synthetic unreadable hook root")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", deny_child)

    frontier = continuity.configured_source_frontier(archive)

    assert not frontier.complete
    assert frontier.root_states["primary-hook-spool:carrier:codex"] is FrontierState.UNAVAILABLE


def test_codex_state_member_is_in_cold_baseline_and_configured_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon's Codex state root also belongs to the continuity denominator."""
    import sqlite3

    from polylogue import paths
    from polylogue.maintenance import source_manifest_continuity as continuity
    from polylogue.sources.live.production_baseline import capture_production_source_baseline
    from polylogue.sources.live.watcher import daemon_watch_sources

    home = tmp_path / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    member = codex / "history.jsonl"
    member.write_text('{"id":"synthetic-codex-state"}\n', encoding="utf-8")
    database = codex / "state_5.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE threads(id TEXT)")
        connection.execute("CREATE TABLE thread_spawn_edges(parent TEXT, child TEXT)")
        connection.execute("INSERT INTO threads VALUES ('synthetic-thread')")
    projection = codex / "thread_history_1.sqlite"
    with sqlite3.connect(projection) as connection:
        connection.execute("CREATE TABLE projection(value TEXT)")
        connection.execute("INSERT INTO projection VALUES ('duplicate rollout evidence')")
    session = codex / "sessions" / "2026" / "01" / "02" / "rollout-2026-01-02T00-00-00-1.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        '{"type":"session_meta","payload":{"id":"synthetic-session","timestamp":"2026-01-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"synthetic-message",'
        '"role":"user","content":[{"type":"input_text","text":"synthetic"}]}}\n',
        encoding="utf-8",
    )
    memory = codex / "memories" / "MEMORY.md"
    memory.parent.mkdir()
    memory.write_text("# Synthetic memory\n", encoding="utf-8")
    credential = codex / "auth.json"
    credential.write_text('{"token":"synthetic"}\n', encoding="utf-8")
    archive = tmp_path / "archive"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.sources.hooks.hook_spool_sources", lambda: ())

    watched = tuple(source for source in daemon_watch_sources() if source.name == "codex-state")
    assert len(watched) == 1
    baseline = capture_production_source_baseline(watched, operation_id="synthetic-codex-frontier")
    assert {decision.path for decision in baseline.accepted} == {str(member), str(database)}

    frontier = continuity.configured_source_frontier(archive)
    declaration = next(row for row in frontier.declarations if row.source_id == "configured:codex-state")
    assert declaration.root == codex
    assert "thread_history_1.sqlite" in declaration.exclude_coordinates
    assert "configured:codex" in {row.source_id for row in frontier.declarations}
    assert "configured:codex-memories" in {row.source_id for row in frontier.declarations}
    owned = [(row.source_id, row.coordinate) for row in frontier.members]
    assert owned.count(("configured:codex-state", "history.jsonl")) == 1
    assert owned.count(("configured:codex-state:sqlite:state_5.sqlite", "state_5.sqlite")) == 1
    assert owned.count(("configured:codex", "2026/01/02/rollout-2026-01-02T00-00-00-1.jsonl")) == 1
    assert owned.count(("configured:codex-memories", "MEMORY.md")) == 1
    assert all("auth.json" not in coordinate for _, coordinate in owned)
    assert all("thread_history_1.sqlite" not in coordinate for _, coordinate in owned)
    assert projection.is_file()
    assert any(row.path == str(projection) and row.disposition == "excluded" for row in baseline.decisions)


def test_configured_frontier_keeps_a_disappeared_source_in_its_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue import config, paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    archive = tmp_path / "archive"
    missing = tmp_path / "configured-source"
    monkeypatch.setattr(
        config,
        "resolve_runtime_config",
        lambda: SimpleNamespace(sources=(SimpleNamespace(name="browser-capture", path=missing),)),
    )
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.sources.hooks.hook_spool_sources", lambda: ())

    frontier = continuity.configured_source_frontier(archive)

    assert not frontier.complete
    assert [item.source_id for item in frontier.declarations] == ["configured:browser-capture"]
    assert frontier.root_states["configured:browser-capture"] is FrontierState.UNAVAILABLE
    assert any(blocker.startswith("unavailable:configured:browser-capture:") for blocker in frontier.blockers)


def test_configured_frontier_retains_unreadable_canonical_provider_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config.exists() may omit a root, but the resolved path declaration survives."""
    from polylogue import paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    home = tmp_path / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    archive = tmp_path / "archive"
    (archive / "hooks").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    monkeypatch.setattr(
        "polylogue.sources.hooks.hook_spool_sources",
        lambda: (SimpleNamespace(source_id="primary-hook-spool", root=archive / "hooks"),),
    )
    original_stat = Path.stat
    original_lstat = Path.lstat

    def denied(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == codex or codex in path.parents:
            raise PermissionError("synthetic unreadable provider root")
        return original_stat(path, follow_symlinks=follow_symlinks)

    def denied_lstat(path: Path) -> os.stat_result:
        if path == codex or codex in path.parents:
            raise PermissionError("synthetic unreadable provider root")
        return original_lstat(path)

    monkeypatch.setattr(Path, "stat", denied)
    monkeypatch.setattr(Path, "lstat", denied_lstat)

    frontier = continuity.configured_source_frontier(archive)

    assert "configured:codex-state" in {row.source_id for row in frontier.declarations}
    assert frontier.root_states["configured:codex-state"] is FrontierState.UNAVAILABLE
    assert not frontier.complete


def test_frontier_members_are_spilled_and_digest_remains_canonical(tmp_path: Path) -> None:
    import json

    root = tmp_path / "many"
    root.mkdir()
    for index in range(700):
        (root / f"{index:04}.json").write_text(f"{index}\n", encoding="utf-8")

    frontier = build_source_frontier((SourceDeclaration("many", SourceRole.DIRECTORY, root, True),))
    try:
        assert frontier.item_count == 700
        assert len(frontier.members) == 700
        assert isinstance(frontier.members, continuity._FrontierMembers)
        db_path = frontier.members._store.connection.execute("PRAGMA database_list").fetchone()[2]
        assert db_path
        assert Path(db_path).is_file()
        payload = {
            "declarations": [
                (d.source_id, d.role.value, str(d.root), d.mutable, d.layout_name, d.exclude_coordinates)
                for d in frontier.declarations
            ],
            "members": [
                (m.source_id, m.coordinate, m.identity, m.content_sha256, m.size, m.logical_sha256)
                for m in frontier.members
            ],
            "root_states": sorted((key, value.value) for key, value in frontier.root_states.items()),
            "blockers": list(frontier.blockers),
        }
        assert (
            frontier.frontier_sha256
            == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        )
        frontier.verify_integrity()
    finally:
        frontier.close()


@pytest.mark.parametrize("payload", [b"", b'{"display":"prompt"}\n'], ids=["empty", "nonempty"])
def test_configured_frontier_observes_optional_claude_history_file_as_append_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    from polylogue import config, paths
    from polylogue.maintenance import source_manifest_continuity as continuity

    archive = tmp_path / "archive"
    history = tmp_path / "home" / ".claude" / "history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_bytes(payload)
    monkeypatch.setattr(
        config,
        "resolve_runtime_config",
        lambda: SimpleNamespace(sources=(SimpleNamespace(name="claude-code-history", path=history),)),
    )
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.sources.hooks.hook_spool_sources", lambda: ())

    frontier = continuity.configured_source_frontier(archive)

    declaration = next(item for item in frontier.declarations if item.source_id == "configured:claude-code-history")
    assert declaration.role is SourceRole.APPEND_JSONL
    assert declaration.layout_name is None
    assert frontier.complete
    assert [(item.coordinate, item.content_sha256, item.size) for item in frontier.members] == [
        ("history.jsonl", hashlib.sha256(payload).hexdigest(), len(payload))
    ]


def test_duplicate_roots_and_symlinks_fail_closed(tmp_path: Path) -> None:
    root = _source(tmp_path)
    with pytest.raises(SourceContinuityError, match="duplicate root"):
        canonical_source_declarations(
            configured=[
                SourceDeclaration("a", SourceRole.DIRECTORY, root, True),
                SourceDeclaration("b", SourceRole.DIRECTORY, root, True),
            ]
        )
    with pytest.raises(SourceContinuityError, match="duplicate roots"):
        build_source_frontier(
            [
                SourceDeclaration("a", SourceRole.DIRECTORY, root, True),
                SourceDeclaration("b", SourceRole.DIRECTORY, root, True),
            ]
        )
    alias = tmp_path / "alias" / ".." / root.name
    with pytest.raises(SourceContinuityError, match="duplicate roots"):
        build_source_frontier(
            [
                SourceDeclaration("a", SourceRole.DIRECTORY, root, True),
                SourceDeclaration("b", SourceRole.DIRECTORY, alias, True),
            ]
        )
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    frontier = build_source_frontier([SourceDeclaration("link", SourceRole.DIRECTORY, link, True)])
    assert frontier.root_states["link"] is FrontierState.UNAVAILABLE
    assert frontier.blockers


def test_frontier_rejects_hardlinked_members_across_declarations(tmp_path: Path) -> None:
    """A hard link cannot be counted once under each source declaration.

    Anti-vacuity: per-declaration identity sets would accept both names and
    double the frontier denominator while pointing at the same inode.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "one.json").write_text("same", encoding="utf-8")
    os.link(first / "one.json", second / "two.json")
    with pytest.raises(SourceContinuityError, match="duplicate physical member identity"):
        build_source_frontier(
            [
                SourceDeclaration("first", SourceRole.DIRECTORY, first, True),
                SourceDeclaration("second", SourceRole.DIRECTORY, second, True),
            ]
        )


def test_corrupt_mutable_sqlite_becomes_unavailable_blocker(tmp_path: Path) -> None:
    """One bad SQLite root must not abort observation of other declarations.

    Anti-vacuity: if ``sqlite3.DatabaseError`` escapes, no frontier records
    the corrupt root as unavailable and later roots are never observed.
    """
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not sqlite")
    healthy = _source(tmp_path, "healthy")
    frontier = build_source_frontier(
        [
            SourceDeclaration("corrupt", SourceRole.MUTABLE_SQLITE, corrupt, True),
            SourceDeclaration("healthy", SourceRole.DIRECTORY, healthy, True),
        ]
    )
    assert frontier.root_states["corrupt"] is FrontierState.UNAVAILABLE
    assert frontier.root_states["healthy"] is FrontierState.PRESENT
    assert any(item.startswith("unavailable:corrupt:") for item in frontier.blockers)


def test_non_regular_members_fail_closed(tmp_path: Path) -> None:
    root = _source(tmp_path)
    fifo = root / "unreadable-pipe"
    os.mkfifo(fifo)
    frontier = build_source_frontier([SourceDeclaration("live", SourceRole.DIRECTORY, root, True)])
    assert frontier.root_states["live"] is FrontierState.UNAVAILABLE
    assert any("not a regular file" in blocker for blocker in frontier.blockers)


def test_file_backed_source_is_manifested(tmp_path: Path) -> None:
    source = tmp_path / "export.json"
    source.write_text('{"export": true}', encoding="utf-8")
    frontier = build_source_frontier([SourceDeclaration("export", SourceRole.IMMUTABLE_EXPORT, source)])
    assert [(member.coordinate, member.size) for member in frontier.members] == [("export.json", 16)]
    assert frontier.blockers == ()


def test_frontier_retains_missing_roots_and_valid_empty_roots(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    frontier = build_source_frontier(
        [
            SourceDeclaration("empty", SourceRole.DIRECTORY, empty, True),
            SourceDeclaration("missing", SourceRole.DIRECTORY, tmp_path / "missing", True),
        ]
    )
    assert frontier.root_states["empty"] is FrontierState.VALID_EMPTY
    assert frontier.root_states["missing"] is FrontierState.UNAVAILABLE
    assert frontier.complete is False
    assert any(item.startswith("unavailable:missing:") for item in frontier.blockers)
    frontier.verify_integrity()


def test_frontier_digest_binds_captured_member_after_path_mutation(tmp_path: Path) -> None:
    root = _source(tmp_path, "captured")
    frontier = build_source_frontier([SourceDeclaration("captured", SourceRole.DIRECTORY, root, True)])
    (root / "one.jsonl").write_text("changed", encoding="utf-8")
    # The captured denominator remains verifiable without rereading mutable
    # source bytes; a changed live root is a new observation, not a refreshed
    # frontier.
    frontier.verify_integrity()
    assert frontier.members[0].content_sha256 != ""


def test_same_byte_replacement_is_a_new_frontier_identity(tmp_path: Path) -> None:
    source = _source(tmp_path, "replacement")
    declaration = SourceDeclaration("replacement", SourceRole.DIRECTORY, source, True)
    first = build_source_frontier([declaration])
    original = (source / "one.jsonl").read_bytes()
    (source / "one.jsonl").unlink()
    (source / "one.jsonl").write_bytes(original)
    second = build_source_frontier([declaration])
    assert first.members[0].content_sha256 == second.members[0].content_sha256
    assert first.members[0].identity != second.members[0].identity
    assert first.frontier_sha256 != second.frontier_sha256


@pytest.mark.parametrize("initial_directory", [False, True])
def test_frontier_refuses_root_kind_replacement_after_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_directory: bool
) -> None:
    """A replacement between root capture and observation cannot misaddress rows."""
    from polylogue.sources import source_snapshot

    root = tmp_path / "root"
    if initial_directory:
        root.mkdir()
        (root / "member.json").write_bytes(b"{}")
    else:
        root.write_bytes(b"{}")
    original = source_snapshot.bind_source_observation

    def replace_after_binding(declaration: SourceDeclaration) -> source_snapshot.SourceCutBinding:
        binding = original(declaration)
        root.rename(tmp_path / "retired")
        if initial_directory:
            root.write_bytes(b"{}")
        else:
            root.mkdir()
            (root / "member.json").write_bytes(b"{}")
        return binding

    monkeypatch.setattr(source_snapshot, "bind_source_observation", replace_after_binding)
    with build_source_frontier([SourceDeclaration("root", SourceRole.DIRECTORY, root, True)]) as frontier:
        assert frontier.root_states["root"] is FrontierState.UNAVAILABLE
        assert not frontier.complete
        assert frontier.item_count == 0
    monkeypatch.setattr(source_snapshot, "bind_source_observation", original)
    with build_source_frontier([SourceDeclaration("root", SourceRole.DIRECTORY, root, True)]) as frontier:
        assert frontier.complete
        assert frontier.item_count == 1
        assert list(frontier.members)[0].coordinate == ("root" if initial_directory else "member.json")


def test_configured_frontier_refuses_sqlite_arrival_then_retries_logically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member arriving after layout discovery cannot become a page-image row."""
    import sqlite3

    from polylogue import paths
    from polylogue.sources import source_walk
    from polylogue.sources.sqlite_snapshot import sqlite_member_revision

    home = tmp_path / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    (codex / "history.jsonl").write_bytes(b"{}\n")
    archive = tmp_path / "archive"
    database = codex / "state_5.sqlite"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setattr(paths, "archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.sources.hooks.hook_spool_sources", lambda: ())
    original = source_walk.layout_source_candidates

    def discover_then_create(name: str, root: Path, **kwargs: object) -> Iterator[Path]:
        yield from original(name, root, **kwargs)
        if name == "codex-state" and not database.exists():
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE threads(id TEXT)")
                connection.execute("CREATE TABLE thread_spawn_edges(parent TEXT, child TEXT)")
                connection.execute("INSERT INTO threads VALUES ('synthetic-thread')")

    monkeypatch.setattr(source_walk, "layout_source_candidates", discover_then_create)
    with continuity.configured_source_frontier(archive) as frontier:
        assert frontier.root_states["configured:codex-state"] is FrontierState.UNAVAILABLE
        assert not frontier.complete
        assert frontier.item_count == 0
    with continuity.configured_source_frontier(archive) as frontier:
        assert frontier.complete
        databases = [row for row in frontier.members if row.logical_sha256 is not None]
        assert len(databases) == 1
        assert databases[0].source_id == "configured:codex-state:sqlite:state_5.sqlite"
        assert databases[0].logical_sha256 == sqlite_member_revision(database)
        assert frontier.item_count == 2


@pytest.mark.parametrize("initial_directory", [False, True])
def test_frontier_addresses_replaced_root_from_the_captured_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_directory: bool
) -> None:
    """A separate pre-observation kind probe addresses the replacement incorrectly."""
    from polylogue.sources import source_snapshot

    root = tmp_path / "root"
    if initial_directory:
        root.mkdir()
        (root / "member.json").write_bytes(b"{}")
    else:
        root.write_bytes(b"{}")
    original = source_snapshot._root_identity
    replaced = False

    def capture_replacement(path: Path) -> source_snapshot.SourceRootIdentity:
        nonlocal replaced
        if path == root and not replaced:
            replaced = True
            root.rename(tmp_path / "retired")
            if initial_directory:
                root.write_bytes(b"{}")
            else:
                root.mkdir()
                (root / "member.json").write_bytes(b"{}")
        return original(path)

    monkeypatch.setattr(source_snapshot, "_root_identity", capture_replacement)
    with build_source_frontier([SourceDeclaration("root", SourceRole.DIRECTORY, root, True)]) as frontier:
        assert frontier.complete
        assert frontier.item_count == 1
        member = list(frontier.members)[0]
        assert member.coordinate == ("root" if initial_directory else "member.json")
        import sqlite3

        expected_path = root if initial_directory else root / "member.json"
        with sqlite3.connect(":memory:") as connection:
            frontier.copy_members_to(connection)
            assert connection.execute("SELECT source_path FROM _polylogue_source_frontier_path").fetchall() == [
                (str(expected_path),)
            ]


def test_arriving_declared_out_of_scope_projection_is_not_hashed_or_owed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: an excluded projection arriving after binding becomes a physical-page obligation."""
    import sqlite3
    from contextlib import contextmanager

    from polylogue.sources import source_snapshot

    root = tmp_path / "codex-state"
    root.mkdir()
    history = root / "history.jsonl"
    history.write_bytes(b'{"prompt":"owed raw-only evidence"}\n')
    projection = root / "thread_history_1.sqlite"
    original_open = source_snapshot._open_source_root
    original_snapshot = source_snapshot._snapshot_regular_file

    @contextmanager
    def arrive(binding: source_snapshot.SourceCutBinding) -> Iterator[tuple[int, os.stat_result, Path]]:
        with original_open(binding) as captured:
            if not projection.exists():
                with sqlite3.connect(projection) as connection:
                    connection.execute("CREATE TABLE projection(value TEXT)")
            yield captured

    def refuse_projection_hash(
        path: Path, expected: os.stat_result, *, anchor: int, coordinate: str
    ) -> tuple[str, int, str]:
        assert path != projection, "out-of-scope source pages must not be read"
        return original_snapshot(path, expected, anchor=anchor, coordinate=coordinate)

    monkeypatch.setattr(source_snapshot, "_open_source_root", arrive)
    monkeypatch.setattr(source_snapshot, "_snapshot_regular_file", refuse_projection_hash)
    frontier = build_source_frontier([SourceDeclaration("state", SourceRole.DIRECTORY, root, True, "codex-state")])
    assert frontier.complete
    assert [(member.coordinate, member.size) for member in frontier.members] == [
        ("history.jsonl", history.stat().st_size)
    ]
    assert projection.is_file()


def test_arriving_metadata_excluded_request_dump_is_not_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: metadata exclusion is applied only to the initial discovery."""
    from contextlib import contextmanager

    from polylogue.core.enums import Provider
    from polylogue.sources import source_snapshot
    from polylogue.sources.origin_specs import pre_acquisition_path_exclusion

    root = tmp_path / "hermes"
    (root / "sessions").mkdir(parents=True)
    session = root / "sessions" / "session_neutral.json"
    session.write_bytes(b'{"neutral":"session input"}')
    excluded = root / "sessions" / "request_dump_neutral.json"
    original_open = source_snapshot._open_source_root
    original_snapshot = source_snapshot._snapshot_regular_file

    @contextmanager
    def arrive(binding: source_snapshot.SourceCutBinding) -> Iterator[tuple[int, os.stat_result, Path]]:
        with original_open(binding) as captured:
            excluded.write_bytes(b'{"neutral":"excluded metadata"}')
            yield captured

    def refuse_excluded_hash(
        path: Path, expected: os.stat_result, *, anchor: int, coordinate: str
    ) -> tuple[str, int, str]:
        assert path != excluded
        return original_snapshot(path, expected, anchor=anchor, coordinate=coordinate)

    assert pre_acquisition_path_exclusion(Provider.CODEX, "history.jsonl") is None
    assert pre_acquisition_path_exclusion(Provider.CODEX, "state_5.sqlite") is None
    monkeypatch.setattr(source_snapshot, "_open_source_root", arrive)
    monkeypatch.setattr(source_snapshot, "_snapshot_regular_file", refuse_excluded_hash)
    with build_source_frontier([SourceDeclaration("hermes", SourceRole.DIRECTORY, root, True, "hermes")]) as frontier:
        assert frontier.complete
        assert [member.coordinate for member in frontier.members] == ["sessions/session_neutral.json"]
    assert excluded.is_file()
