from __future__ import annotations

import os
from pathlib import Path

import pytest

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
