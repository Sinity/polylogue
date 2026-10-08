"""Reviewed acquisition defects whose reproduction needs no archive fixture.

Each test names the mutation that turns it red, so a later change that
reintroduces the defect cannot pass by leaving the assertion unreached.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.origin_specs import _source_signature, origin_specs
from polylogue.sources.source_layout import export_drop_layout
from polylogue.sources.sqlite_snapshot import sqlite_logical_revision


def test_chatgpt_declares_no_session_inheritance_branch_point() -> None:
    """ChatGPT mapping ancestry is intra-session topology, not inheritance.

    ``mapping.parent``/``mapping.children`` order messages inside one
    conversation; they never name another session, so they cannot derive a
    cross-session inheritance branch point.

    Anti-vacuity: restoring the ``positive-derived`` capability -- or deriving
    it from any ``chatgpt.mapping.*`` evidence -- makes both assertions red.
    """
    spec = next(item for item in origin_specs() if item.origin is Origin.CHATGPT_EXPORT)
    capability = spec.topology_capabilities.inheritance_branch_point

    assert capability.state == "structurally-absent"
    assert not [source for source in capability.evidence if source.startswith("chatgpt.mapping")]


class _FakeSource:
    def __init__(
        self,
        name: str,
        root: Path,
        suffixes: tuple[str, ...],
        *,
        exact_paths: frozenset[Path] | None = None,
    ) -> None:
        self.name = name
        self.root = root
        self._suffixes = suffixes
        self.exact_paths = exact_paths

    def accepts(self, path: Path) -> bool:
        return any(path.name.lower().endswith(suffix) for suffix in self._suffixes)


def test_typed_source_keeps_ownership_of_a_nested_generic_root(tmp_path: Path) -> None:
    """A deeper generic root must not capture artifacts only a typed source admits.

    The generic additional-root suffix set excludes ``.pb``, so handing it an
    Antigravity conversation on depth alone drops the file from ingest
    entirely.

    Anti-vacuity: reverting ``deepest_source_for_path`` to plain
    ``max(..., key=depth)`` returns the generic source for the ``.pb`` path and
    turns the first assertion red. The second assertion pins that depth still
    decides among sources that both accept the path, so the fix cannot be
    "always prefer the typed source".
    """
    typed_root = tmp_path / "antigravity"
    generic_root = typed_root / "conversations"
    generic_root.mkdir(parents=True)
    typed = _FakeSource("antigravity", typed_root, (".pb", ".json"))
    generic = _FakeSource("conversations", generic_root, (".json", ".jsonl", ".ndjson", ".zip"))

    protobuf = generic_root / "thread.pb"
    protobuf.write_bytes(b"\x00")
    shared = generic_root / "thread.json"
    shared.write_text("{}")

    assert deepest_source_for_path(protobuf, (typed, generic)) is typed
    assert deepest_source_for_path(shared, (typed, generic)) is generic


def test_exact_file_source_outranks_an_equal_depth_directory_source(tmp_path: Path) -> None:
    """An explicit file declaration wins when overlapping roots both accept it.

    Anti-vacuity: removing exact-file precedence makes the first declaration
    win and assigns the Codex file to the Antigravity directory source.
    """
    root = tmp_path / "captures"
    root.mkdir()
    file_path = root / "session.jsonl"
    file_path.write_text("{}\n")
    directory_source = _FakeSource("antigravity", root, (".jsonl",))
    exact_file_source = _FakeSource(
        "codex",
        root,
        (".jsonl",),
        exact_paths=frozenset({file_path.resolve()}),
    )

    assert deepest_source_for_path(file_path, (directory_source, exact_file_source)) is exact_file_source


def test_declared_subtree_alias_retains_provider_namespace_and_explicit_file_priority(tmp_path: Path) -> None:
    declared = tmp_path / "profile"
    external = tmp_path / "external"
    declared.mkdir()
    external.mkdir()
    (declared / "sessions").symlink_to(external, target_is_directory=True)
    physical = external / "session_shared.json"
    physical.write_text("{}")
    offered = declared / "sessions" / physical.name
    # The production Hermes source watches its JSON session snapshots.
    hermes = WatchSource(name="hermes", root=declared, layout=export_drop_layout((".json",)))
    external_source = WatchSource(name="inbox", root=external)
    assert hermes.accepts(offered)
    assert deepest_source_for_path(offered, (external_source, hermes)) is hermes
    explicit = WatchSource(name="inbox", root=external, exact_paths=frozenset({physical.resolve()}))
    assert deepest_source_for_path(offered, (hermes, explicit)) is explicit


def test_resolved_root_alias_selects_paths_without_declared_containment(tmp_path: Path) -> None:
    physical_root = tmp_path / "physical"
    physical_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical_root, target_is_directory=True)
    path = physical_root / "session.jsonl"
    path.write_text("{}\n")
    source = WatchSource(name="codex", root=alias)
    assert deepest_source_for_path(path, (source,)) is source


def test_lexical_containment_collapses_parent_components_before_selecting(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    root.mkdir()
    outside = tmp_path / "external"
    outside.mkdir()
    path = outside / "session.jsonl"
    path.write_text("{}\n")
    declared = WatchSource(name="hermes", root=root)
    actual = WatchSource(name="codex", root=outside)
    assert deepest_source_for_path(root / ".." / "external" / path.name, (declared, actual)) is actual


def test_autoincrement_state_changes_the_logical_revision(tmp_path: Path) -> None:
    """An insert-then-delete on an AUTOINCREMENT table is a real content change.

    Every user row is identical afterwards, but ``sqlite_sequence`` retains an
    advanced high-water mark, so the database is not in its prior state and a
    source-continuity digest that reports it as unchanged is wrong.

    Anti-vacuity: dropping the ``sqlite_sequence`` block from
    ``sqlite_logical_revision`` makes both digests equal and turns the
    inequality assertion red. The equality assertion pins that the digest is
    still stable for an untouched database, so the fix cannot be "return a
    fresh value every call".
    """
    database = tmp_path / "state.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE item (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT)")
        conn.execute("INSERT INTO item (label) VALUES ('first')")
        conn.commit()

    before = sqlite_logical_revision(database)
    assert sqlite_logical_revision(database) == before

    with sqlite3.connect(database) as conn:
        conn.execute("INSERT INTO item (label) VALUES ('transient')")
        conn.execute("DELETE FROM item WHERE label = 'transient'")
        conn.commit()

    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT label FROM item").fetchall() == [("first",)]

    assert sqlite_logical_revision(database) != before


def test_source_signature_is_keyed_by_contents(tmp_path: Path) -> None:
    """An explicit source-edit signal invalidates the process signature memo.

    The stat key's content digest distinguishes a same-length rewrite even
    when its modification time is restored.
    """
    module = tmp_path / "parser.py"
    module.write_text("VALUE = 1\n")
    stat = module.stat()

    before = _source_signature(module)

    module.write_text("VALUE = 2\n")  # identical length
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    from polylogue.sources import origin_specs

    origin_specs._invalidate_source_signatures()

    assert module.stat().st_size == stat.st_size
    assert module.stat().st_mtime_ns == stat.st_mtime_ns
    assert _source_signature(module) != before


def test_antigravity_trajectory_db_is_not_skipped_as_a_protobuf(tmp_path: Path) -> None:
    """The ``.pb`` prepass role must not swallow a schema-verified ``.db``.

    ``classify_source_path`` gives a trajectory SQLite store the
    ``CONVERSATION_PROTOBUF`` compatibility role, and the configured-source
    and API ingest loops skip that role. Anti-vacuity: drop the ``.pb``
    suffix guard from either loop and the database produces no raw record and
    no session at all, because the language-server prepass only emits ``.pb``
    conversations.
    """
    import inspect

    from polylogue.sources import source_parsing
    from polylogue.sources.parsers import antigravity

    trajectory = tmp_path / "trajectory.db"
    connection = sqlite3.connect(trajectory)
    connection.executescript(
        """
        CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
        CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
        """
    )
    connection.commit()
    connection.close()

    classification = antigravity.classify_source_path(trajectory)
    assert classification.role is antigravity.AntigravitySourceRole.CONVERSATION_PROTOBUF

    source = inspect.getsource(source_parsing)
    skip_count = source.count("AntigravitySourceRole.CONVERSATION_PROTOBUF")
    suffix_count = source.count('path.suffix.lower() == ".pb"')
    assert skip_count == suffix_count, source_parsing.__name__


@pytest.mark.asyncio
async def test_drive_acquisition_refuses_a_foreign_archive_cache(tmp_path: Path) -> None:
    """The Drive branch bypasses ``iter_source_acquisition_records``'s root refusal.

    Anti-vacuity: drop the guard from ``iter_raw_record_stream`` and the Drive
    branch accepts a foreign archive's drive cache as a capture location, so
    its bytes are copied into the destination blob store.
    """
    from polylogue.config import Source
    from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
    from polylogue.sources.source_root_admission import (
        SourceRootRefusedError,
        refuse_non_capture_source_root,
    )
    from polylogue.storage.blob_store import BlobStore

    foreign_archive = tmp_path / "foreign-archive"
    (foreign_archive / "drive-cache" / "gemini").mkdir(parents=True)
    (foreign_archive / "source.db").write_bytes(b"")
    destination = tmp_path / "destination-archive"
    (destination / "blob").mkdir(parents=True)
    (destination / "source.db").write_bytes(b"")

    source = Source(name="aistudio", folder="AI Studio", path=foreign_archive / "drive-cache" / "gemini")
    assert source.is_drive
    stream = iter_raw_record_stream(source, blob_store=BlobStore(destination / "blob"))
    with pytest.raises(SourceRootRefusedError):
        await anext(stream)

    # The opposite direction: a blanket refusal must not pass. The
    # destination's own drive cache is the live capture route and stays
    # admissible under the same guard.
    own_cache = destination / "drive-cache" / "gemini"
    own_cache.mkdir(parents=True)
    refuse_non_capture_source_root(own_cache, destination=destination)


def test_blocked_source_paths_match_through_a_symlinked_watch_root(tmp_path: Path) -> None:
    """A stored symlink spelling must still refuse its real-path candidate.

    A cursor-ahead violation recorded through a symlinked watch root is stored
    under that spelling; a restart configured with the real path selects the
    physically identical file. Anti-vacuity: resolve only the candidate side
    and ``refused`` is empty, so the per-path gate admits a source the
    frontier proof already refuses.
    """
    from types import SimpleNamespace
    from typing import Any, cast

    from polylogue.sources.live import WatchSource
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_ingest_cursor

    real_root = tmp_path / "real"
    real_root.mkdir()
    violating = real_root / "session.jsonl"
    violating.write_text('{"a":1}\n', encoding="utf-8")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, source_path, canonical_source_path, "
            "blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_kind, "
            "source_revision, acquisition_generation, revision_authority) "
            "VALUES ('alias-raw', 'codex-session', ?, ?, ?, 1, 1, 'alias-key', 'full', "
            "'revision-0', 0, 'byte_proven')",
            (str(linked_root / "session.jsonl"), str(violating), bytes(32)),
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            "INSERT INTO raw_revision_heads(logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, "
            "accepted_frontier, acquisition_generation, decided_at_ms) "
            "VALUES ('alias-key', 'codex-session:alias', 'alias-raw', 'revision-0', ?, 'byte', 1, 0, 1)",
            (bytes(32),),
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        # Acquisition freezes the resolved canonical path beside the spelling.
        upsert_ingest_cursor(
            conn,
            source_path=str(linked_root / "session.jsonl"),
            canonical_source_path=str(violating),
            updated_at_ms=1,
            byte_offset=2,
        )

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=real_root),),
        cursor=CursorStore(tmp_path / "cursors.db"),
        parser_fingerprint="test-parser",
    )
    healthy = real_root / "healthy.jsonl"
    healthy.write_text('{"b":2}\n', encoding="utf-8")

    admitted = processor.admit_paths([violating, healthy])

    assert admitted == [healthy]
