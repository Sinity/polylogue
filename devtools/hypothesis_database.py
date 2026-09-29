"""The suite's Hypothesis example database, with a declared revision marker.

Focused-run receipt reuse must know whether the example database changed,
since a saved counterexample is replayed by the next run of the same
selection. Walking the database on every run grows with its size, so every
write through this database replaces one small marker file instead, and the
reuse key reads only that marker.
"""

from __future__ import annotations

import fcntl
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hypothesis.database import DirectoryBasedExampleDatabase

#: Written beside the example directory: ``<examples>.revision``.
REVISION_SUFFIX = ".revision"


def revision_marker(examples: Path) -> Path:
    return examples.with_name(examples.name + REVISION_SUFFIX)


@contextmanager
def _revision_lock(examples: Path, *, exclusive: bool) -> Iterator[None]:
    """Serialize a write's bump-mutate-bump against every other write and read.

    A writer holds it exclusively across its whole sequence and a reader
    shares it, so no reader sees a token taken between a write's first bump
    and its mutation, and no second writer's final bump can stand for a
    mutation still in flight. A killed holder's lock is released by the OS.
    """
    lock_path = examples.with_name(examples.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_revision(examples: Path) -> str:
    """The marker's token, or ``absent`` when nothing was ever written."""
    if not examples.parent.is_dir():
        return "absent"
    try:
        with _revision_lock(examples, exclusive=False):
            return revision_marker(examples).read_text(encoding="utf-8").strip() or "empty"
    except FileNotFoundError:
        return "absent"
    except OSError:
        # Unreadable: a fresh token each time, so no receipt matches it.
        return f"unreadable:{uuid.uuid4().hex}"


class RevisionedExampleDatabase(DirectoryBasedExampleDatabase):
    """A directory example database that bumps its revision marker on every write."""

    def __init__(self, path: os.PathLike[str] | str) -> None:
        super().__init__(path)
        self._examples = Path(path)
        #: Hypothesis's own ``save`` and ``move`` call ``save``/``delete``
        #: again (the meta-key entry); only the outermost call of a thread
        #: takes the lock and bumps.
        self._nesting = threading.local()

    @contextmanager
    def _write(self) -> Iterator[None]:
        depth = getattr(self._nesting, "depth", 0)
        if depth:
            self._nesting.depth = depth + 1
            try:
                yield
            finally:
                self._nesting.depth = depth
            return
        with _revision_lock(self._examples, exclusive=True):
            self._nesting.depth = 1
            try:
                self._bump()
                yield
                self._bump()
            finally:
                self._nesting.depth = 0

    def _bump(self) -> None:
        marker = revision_marker(self._examples)
        marker.parent.mkdir(parents=True, exist_ok=True)
        staging = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
        staging.write_text(uuid.uuid4().hex, encoding="utf-8")
        os.replace(staging, marker)

    # The marker moves before each mutation, so a write interrupted or
    # failing midway leaves a token no earlier receipt carries, and again
    # after it, so a receipt taken while the write ran cannot match the data
    # it did not see.

    def save(self, key: bytes, value: bytes) -> None:
        with self._write():
            super().save(key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        with self._write():
            super().delete(key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        with self._write():
            super().move(src, dest, value)


__all__ = ["REVISION_SUFFIX", "RevisionedExampleDatabase", "read_revision", "revision_marker"]
