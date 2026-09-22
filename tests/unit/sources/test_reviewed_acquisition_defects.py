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
from polylogue.sources.origin_specs import _source_signature, origin_specs
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
    def __init__(self, name: str, root: Path, suffixes: tuple[str, ...]) -> None:
        self.name = name
        self.root = root
        self._suffixes = suffixes

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
    """A same-length rewrite under a restored mtime must not reuse the memo key.

    Checkout, patch application, and archive extraction all reproduce that
    shape, and a stale parser fingerprint then claims semantics the file no
    longer has.

    Anti-vacuity: restoring the ``(path, st_mtime_ns, st_size)`` signature
    makes both signatures identical and turns the inequality assertion red.
    """
    module = tmp_path / "parser.py"
    module.write_text("VALUE = 1\n")
    stat = module.stat()

    before = _source_signature(module)

    module.write_text("VALUE = 2\n")  # identical length
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert module.stat().st_size == stat.st_size
    assert module.stat().st_mtime_ns == stat.st_mtime_ns
    assert _source_signature(module) != before


def test_gemini_cli_parsing_is_not_path_independent() -> None:
    """A path-dependent parser must not share one parse across source paths.

    ``dispatch`` passes ``source_path`` into ``parse_gemini_cli``, which
    resolves its ``tool-outputs/`` sidecar scope from it. Anti-vacuity: put
    ``GEMINI_CLI`` back in the set and ``_parse_retained_raws`` fans one
    representative's recovered output out to every byte-identical row,
    regardless of which sidecar directory each row's path names.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.revision_backfill import _PATH_INDEPENDENT_PARSE_PROVIDERS

    assert Provider.GEMINI_CLI not in _PATH_INDEPENDENT_PARSE_PROVIDERS
    # The opposite direction: emptying the set would also pass the assertion
    # above, so pin a provider that is genuinely path-independent.
    assert Provider.CHATGPT in _PATH_INDEPENDENT_PARSE_PROVIDERS


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

    from polylogue.pipeline.services import archive_ingest
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

    for module in (source_parsing, archive_ingest):
        source = inspect.getsource(module)
        skip_count = source.count("AntigravitySourceRole.CONVERSATION_PROTOBUF")
        suffix_count = source.count('path.suffix.lower() == ".pb"')
        assert skip_count == suffix_count, module.__name__


@pytest.mark.asyncio
async def test_drive_acquisition_refuses_a_foreign_archive_cache(tmp_path: Path) -> None:
    """The Drive branch bypasses ``iter_source_raw_data``'s root refusal.

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
    from polylogue.storage.raw_retention import RawFrontierBlockedPaths

    real_root = tmp_path / "real"
    real_root.mkdir()
    violating = real_root / "session.jsonl"
    violating.write_text('{"a":1}\n', encoding="utf-8")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=real_root),),
        cursor=CursorStore(tmp_path / "cursors.db"),
        parser_fingerprint="test-parser",
    )
    healthy = real_root / "healthy.jsonl"
    healthy.write_text('{"b":2}\n', encoding="utf-8")

    processor.cursor_authority_block_reason = lambda: "cursor ahead of accepted raw material"  # type: ignore[method-assign]
    processor._blocked_source_paths = lambda: RawFrontierBlockedPaths(  # type: ignore[method-assign]
        source_paths=frozenset({str(linked_root / "session.jsonl")}),
        unattributed_reason=None,
    )

    admitted = processor.admit_paths([violating, healthy])

    assert admitted == [healthy]
