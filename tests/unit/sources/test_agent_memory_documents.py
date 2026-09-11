"""Harness memory documents are retained as source artifacts (polylogue-rovf5).

Claude Code keeps Markdown memory under ``projects/<project>/memory/`` and
Codex keeps it under ``memories/``. Neither family was admitted by any
declared source, walk filter or watch filter, so harness-authored evidence was
visible on disk and acquired by nothing.

Anti-vacuity: delete the ``agent_memory_document`` ``OriginArtifactRule`` from
``_claude_code_spec``/``_codex_spec`` in ``sources/origin_specs.py``. Every
acquisition test below then goes red at discovery -- ``_walk_source_paths``
returns no candidate, so the production ingest it feeds writes no
``raw_sessions`` row, no ``raw_artifacts`` row and retains no bytes. The
classification assertions are secondary; the acquisition ones are the
contract.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.origin_specs import artifact_suffixes_for_provider
from polylogue.sources.source_walk import census_source_root

_CLAUDE_MEMORY = "---\nname: archive-root\n---\n\nResolve the live root first.\n"
_CODEX_MEMORY = "# MEMORY\n\n- Codex keeps its own memory documents here.\n"


def _claude_source(root: Path) -> WatchSource:
    return WatchSource(
        name="claude-code",
        root=root,
        suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
    )


def _codex_source(root: Path) -> WatchSource:
    return WatchSource(name="codex-memories", root=root, suffixes=())


def _discover(root: Path, *, provider: Provider) -> list[Path]:
    """The production source walk, which is what a rule removal disables."""
    from polylogue.sources.source_walk import _walk_source_paths

    return _walk_source_paths(root, provider=provider)


async def _acquire(
    workspace_env: dict[str, Path],
    source: WatchSource,
    paths: list[Path],
) -> None:
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "index.db")
    cursor = CursorStore(workspace_env["data_root"] / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        (source,),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        await processor.ingest_files(paths, emit_event=False)
    finally:
        await archive.close()


def _source_rows(archive_root: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        return [tuple(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def _retained_bytes(archive_root: Path, blob_hash: object) -> bytes:
    from polylogue.storage.blob_store import BlobStore

    assert isinstance(blob_hash, bytes)
    return BlobStore(archive_root / "blob").read_all(blob_hash.hex())


def _blob_count(archive_root: Path) -> int:
    return sum(1 for path in (archive_root / "blob").rglob("*") if path.is_file())


def _claude_memory(root: Path, project: str, name: str, text: str) -> Path:
    path = root / project / "memory" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_claude_memory_document_is_discovered_and_retained(workspace_env: dict[str, Path]) -> None:
    """A project memory document is walked, acquired, and readable from its bytes."""
    root = workspace_env["data_root"] / "projects"
    memory = _claude_memory(root, "-realm-project-x", "MEMORY.md", _CLAUDE_MEMORY)

    discovered = _discover(root, provider=Provider.CLAUDE_CODE)
    assert discovered == [memory]
    assert _claude_source(root).accepts(memory)

    await _acquire(workspace_env, _claude_source(root), discovered)

    raws = _source_rows(
        workspace_env["archive_root"],
        "SELECT origin, blob_hash, parse_error FROM raw_sessions WHERE source_path = ?",
        (str(memory),),
    )
    assert len(raws) == 1, "the memory document must leave exactly one retained raw row"
    origin, blob_hash, parse_error = raws[0]
    assert origin == "claude-code-session"
    assert parse_error is None
    assert _retained_bytes(workspace_env["archive_root"], blob_hash) == _CLAUDE_MEMORY.encode("utf-8")
    assert _source_rows(
        workspace_env["archive_root"],
        "SELECT artifact_kind, parse_as_session FROM raw_artifacts WHERE source_path = ?",
        (str(memory),),
    ) == [(ArtifactKind.AGENT_MEMORY_DOCUMENT.value, 0)]


@pytest.mark.asyncio
async def test_codex_memory_document_is_discovered_and_retained(workspace_env: dict[str, Path]) -> None:
    """The Codex memories root is a declared source of its own."""
    root = workspace_env["data_root"] / "memories"
    nested = root / "rollout_summaries" / "2026-06-21-orientation.md"
    nested.parent.mkdir(parents=True)
    nested.write_text(_CODEX_MEMORY, encoding="utf-8")

    discovered = _discover(root, provider=Provider.CODEX)
    assert discovered == [nested]
    assert _codex_source(root).accepts(nested)

    await _acquire(workspace_env, _codex_source(root), discovered)

    raws = _source_rows(
        workspace_env["archive_root"],
        "SELECT origin, blob_hash, parse_error FROM raw_sessions WHERE source_path = ?",
        (str(nested),),
    )
    assert len(raws) == 1
    origin, blob_hash, parse_error = raws[0]
    assert origin == "codex-session"
    assert parse_error is None
    assert _retained_bytes(workspace_env["archive_root"], blob_hash) == _CODEX_MEMORY.encode("utf-8")
    assert _source_rows(
        workspace_env["archive_root"],
        "SELECT artifact_kind, parse_as_session FROM raw_artifacts WHERE source_path = ?",
        (str(nested),),
    ) == [(ArtifactKind.AGENT_MEMORY_DOCUMENT.value, 0)]


@pytest.mark.asyncio
async def test_memory_documents_create_no_session_and_no_user_assertion(
    workspace_env: dict[str, Path],
) -> None:
    """An observed harness artifact is not a conversation and not a user claim."""
    root = workspace_env["data_root"] / "projects"
    memory = _claude_memory(root, "-realm-project-x", "MEMORY.md", _CLAUDE_MEMORY)

    await _acquire(workspace_env, _claude_source(root), [memory])

    index = sqlite3.connect(f"file:{workspace_env['data_root'] / 'index.db'}?mode=ro", uri=True)
    try:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert index.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    finally:
        index.close()

    user_db = workspace_env["archive_root"] / "user.db"
    user = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
    try:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 0
    finally:
        user.close()


@pytest.mark.asyncio
async def test_removing_the_original_preserves_the_archived_document(
    workspace_env: dict[str, Path],
) -> None:
    """Disappearance is an observation: the retained bytes stay readable."""
    root = workspace_env["data_root"] / "projects"
    memory = _claude_memory(root, "-realm-project-x", "MEMORY.md", _CLAUDE_MEMORY)

    await _acquire(workspace_env, _claude_source(root), _discover(root, provider=Provider.CLAUDE_CODE))
    memory.unlink()
    assert not memory.exists()

    raws = _source_rows(
        workspace_env["archive_root"],
        "SELECT blob_hash FROM raw_sessions WHERE source_path = ?",
        (str(memory),),
    )
    assert len(raws) == 1
    assert _retained_bytes(workspace_env["archive_root"], raws[0][0]) == _CLAUDE_MEMORY.encode("utf-8")


@pytest.mark.asyncio
async def test_unchanged_content_adds_no_blob_and_a_rewrite_retains_both(
    workspace_env: dict[str, Path],
) -> None:
    """Re-observation is content-addressed; a rewrite is a newer observation."""
    root = workspace_env["data_root"] / "projects"
    memory = _claude_memory(root, "-realm-project-x", "MEMORY.md", _CLAUDE_MEMORY)

    await _acquire(workspace_env, _claude_source(root), [memory])
    blobs_after_first = _blob_count(workspace_env["archive_root"])

    # Same bytes observed again: content addressing collapses the duplicate.
    await _acquire(workspace_env, _claude_source(root), [memory])
    assert _blob_count(workspace_env["archive_root"]) == blobs_after_first
    assert (
        len(
            _source_rows(
                workspace_env["archive_root"],
                "SELECT raw_id FROM raw_sessions WHERE source_path = ?",
                (str(memory),),
            )
        )
        == 1
    )

    rewritten = _CLAUDE_MEMORY + "\nAlso check the daemon owner.\n"
    memory.write_text(rewritten, encoding="utf-8")
    await _acquire(workspace_env, _claude_source(root), [memory])

    retained = {
        _retained_bytes(workspace_env["archive_root"], row[0])
        for row in _source_rows(
            workspace_env["archive_root"],
            "SELECT blob_hash FROM raw_sessions WHERE source_path = ?",
            (str(memory),),
        )
    }
    assert rewritten.encode("utf-8") in retained, "the newer observation must be retained"
    assert _CLAUDE_MEMORY.encode("utf-8") in retained, "the superseded observation is not deleted"


@pytest.mark.asyncio
async def test_same_basename_in_two_projects_stays_two_scoped_objects(
    workspace_env: dict[str, Path],
) -> None:
    """Scope is part of identity: one basename under two projects is two objects."""
    root = workspace_env["data_root"] / "projects"
    first = _claude_memory(root, "-realm-project-x", "MEMORY.md", "# x\n")
    second = _claude_memory(root, "-realm-project-y", "MEMORY.md", "# y\n")

    discovered = _discover(root, provider=Provider.CLAUDE_CODE)
    assert discovered == sorted([first, second])

    await _acquire(workspace_env, _claude_source(root), discovered)

    rows: dict[object, object] = {
        row[0]: row[1]
        for row in _source_rows(
            workspace_env["archive_root"],
            "SELECT source_path, blob_hash FROM raw_sessions ORDER BY source_path",
        )
    }
    assert set(rows) == {str(first), str(second)}
    assert _retained_bytes(workspace_env["archive_root"], rows[str(first)]) == b"# x\n"
    assert _retained_bytes(workspace_env["archive_root"], rows[str(second)]) == b"# y\n"


def test_markdown_outside_the_declared_memory_roots_is_not_swept_in(
    workspace_env: dict[str, Path],
) -> None:
    """Only the declared ``memory/``/``memories/`` families are admitted."""
    root = workspace_env["data_root"] / "projects"
    inside = _claude_memory(root, "-realm-project-x", "MEMORY.md", "# in\n")
    outside_names = [
        root / "-realm-project-x" / "notes.md",
        root / "-realm-project-x" / "memory.md",
        root / "-realm-project-x" / "docs" / "design.md",
        root / "README.md",
    ]
    for path in outside_names:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unrelated\n", encoding="utf-8")

    assert _discover(root, provider=Provider.CLAUDE_CODE) == [inside]
    source = _claude_source(root)
    assert source.accepts(inside)
    assert [path for path in outside_names if source.accepts(path)] == []

    census = census_source_root(root, provider=Provider.CLAUDE_CODE)
    assert census.candidate_count == 1
    assert census.disposition_counts["non_session"] == 1
    assert census.is_complete


def test_codex_markdown_outside_the_memories_root_is_not_swept_in(
    workspace_env: dict[str, Path],
) -> None:
    """A Codex install's other Markdown families stay outside the declaration."""
    install = workspace_env["data_root"] / "codex"
    memories = install / "memories"
    inside = memories / "MEMORY.md"
    inside.parent.mkdir(parents=True)
    inside.write_text(_CODEX_MEMORY, encoding="utf-8")
    outside = [
        install / "AGENTS.md",
        install / "vendor_imports" / "skills" / "README.md",
        install / "prompts" / "notes.md",
    ]
    for path in outside:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unrelated\n", encoding="utf-8")

    assert _discover(install, provider=Provider.CODEX) == [inside]
    source = _codex_source(memories)
    assert source.accepts(inside)
    assert [path for path in outside if source.accepts(path)] == []


def test_codex_memories_is_a_declared_runtime_source(workspace_env: dict[str, Path]) -> None:
    """The runtime resolves ``~/.codex/memories`` as its own scoped source."""
    from polylogue.config import resolve_runtime_config
    from polylogue.core.enums import Provider as RuntimeProvider

    home = workspace_env["home_dir"]
    (home / ".codex" / "memories").mkdir(parents=True)
    resolved = resolve_runtime_config(home=home)

    assert resolved.source_paths.codex_memories == home / ".codex" / "memories"
    memories_sources = [source for source in resolved.sources if source.name == "codex-memories"]
    assert [source.path for source in memories_sources] == [home / ".codex" / "memories"]
    assert RuntimeProvider.from_string("codex-memories") is RuntimeProvider.CODEX


def test_default_watch_sources_include_the_codex_memories_root(
    workspace_env: dict[str, Path],
) -> None:
    """The daemon watcher observes the memories root without widening a Codex root."""
    from polylogue.sources.live.watcher import default_sources

    by_name = {source.name: source for source in default_sources()}
    memories = by_name["codex-memories"]
    assert memories.root == workspace_env["home_dir"] / ".codex" / "memories"
    assert memories.suffixes == ()
    assert by_name["codex"].suffixes == (".jsonl",)
    assert by_name["codex-state"].suffixes == (".sqlite", ".db")
