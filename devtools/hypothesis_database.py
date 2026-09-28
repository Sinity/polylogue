"""The suite's Hypothesis example database, with a declared revision marker.

Focused-run receipt reuse must know whether the example database changed,
since a saved counterexample is replayed by the next run of the same
selection. Walking the database on every run grows with its size, so every
write through this database replaces one small marker file instead, and the
reuse key reads only that marker.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from hypothesis.database import DirectoryBasedExampleDatabase

#: Written beside the example directory: ``<examples>.revision``.
REVISION_SUFFIX = ".revision"


def revision_marker(examples: Path) -> Path:
    return examples.with_name(examples.name + REVISION_SUFFIX)


def read_revision(examples: Path) -> str:
    """The marker's token, or ``absent`` when nothing was ever written."""
    try:
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
        self._bump()
        super().save(key, value)
        self._bump()

    def delete(self, key: bytes, value: bytes) -> None:
        self._bump()
        super().delete(key, value)
        self._bump()

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        self._bump()
        super().move(src, dest, value)
        self._bump()


__all__ = ["REVISION_SUFFIX", "RevisionedExampleDatabase", "read_revision", "revision_marker"]
