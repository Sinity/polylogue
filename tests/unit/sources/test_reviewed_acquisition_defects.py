"""Reviewed acquisition defects whose reproduction needs no archive fixture.

Each test names the mutation that turns it red, so a later change that
reintroduces the defect cannot pass by leaving the assertion unreached.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from polylogue.core.enums import Origin
from polylogue.operations import append_acquisition_replay
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.origin_specs import _source_signature, origin_specs
from polylogue.sources.sqlite_snapshot import sqlite_logical_revision


def test_codex_session_meta_read_is_bounded(tmp_path: Path) -> None:
    """Replay bounds its session_meta read instead of consuming the whole line.

    Replay runs against arbitrary on-disk files, so an unbounded ``readline``
    is only as bounded as the file happens to be. A header larger than the
    bound is truncated, fails to parse, and yields no identity -- which is the
    observable difference from reading it in full.

    Anti-vacuity: dropping the ``_SESSION_META_READ_LIMIT`` argument from
    ``handle.readline(...)`` reads the entire oversized record, parses it, and
    returns ``"oversized"``, turning the final assertion red. The bounded-file
    assertion above it pins that ordinary headers still resolve, so the fix
    cannot be "always return None".
    """
    limit = append_acquisition_replay._SESSION_META_READ_LIMIT

    ordinary = tmp_path / "ordinary.jsonl"
    ordinary.write_bytes(b'{"type":"session_meta","payload":{"id":"ordinary"}}\n')
    assert append_acquisition_replay._codex_session_meta_id(str(ordinary)) == "ordinary"

    padding = "x" * (limit + 4096)
    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(f'{{"type":"session_meta","payload":{{"id":"oversized","pad":"{padding}"}}}}\n'.encode())
    assert oversized.stat().st_size > limit

    assert append_acquisition_replay._codex_session_meta_id(str(oversized)) is None


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
