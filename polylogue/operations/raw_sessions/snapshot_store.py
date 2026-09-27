"""Bounded metadata snapshots behind raw-session search continuations.

A broad raw search pages through a selected population of provider JSONL
files. The continuation token cannot carry that population (15,000+ files
would exceed any sane cursor), and recomputing it on resume makes one
unrelated live append invalidate an otherwise untouched historical scan.

The population is therefore retained here as a private per-user state file:
relative paths plus the stat identity observed at selection, and never any
session content. It is not an archive tier: raw search reads provider files,
not the archive, and runs from CLI, MCP and daemon processes alike, so
routing it through the archive writer would add a writer to processes that
own none. Files are written atomically and bounded by a TTL that slides
from last use and by one global capacity; they survive process restart
until they expire or are evicted, after which a resume is reported as a
typed degraded outcome rather than silently restarting the scan.

There is no per-principal cap: every production caller reaches this store
through ``raw_operation`` with the same service scope, so a per-principal
cap would be a smaller host-wide cap, not isolation between callers. The
signed token and the stored binding still tie each handle to its scope.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.core.durable_fs import atomic_replace

SNAPSHOT_TTL_MS = 60 * 60 * 1000
MAX_GLOBAL_SNAPSHOTS = 64
_SUFFIX = ".snapshot"
# Handles this young are never eviction victims: a concurrent creator may have
# published one and not yet returned its token. The global bound can then be
# exceeded briefly by the number of simultaneous creations.
IN_FLIGHT_GRACE_MS = 60_000


@dataclass(frozen=True)
class FileObservation:
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> FileObservation:
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def matches(self, info: os.stat_result) -> bool:
        return self == FileObservation.of(info)


@dataclass(frozen=True)
class SearchSnapshot:
    handle: str
    files: tuple[tuple[Path, FileObservation], ...]


@dataclass(frozen=True)
class SnapshotBinding:
    """Everything a resume must match; a mismatch is a stale continuation."""

    principal: str
    provider: str
    query_sha256: str
    reference: str | None
    root: Path

    def as_json(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "provider": self.provider,
            "query_sha256": self.query_sha256,
            "reference": self.reference,
            "root": str(self.root),
        }


class SnapshotUnavailableError(LookupError):
    """The handle expired, was evicted, or never belonged to this scope."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _principal_key(principal: str) -> str:
    return hashlib.sha256(principal.encode()).hexdigest()[:16]


class SnapshotStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def _entries(self) -> list[tuple[int, str, str, Path]]:
        """(last_used_ms, principal_key, handle, path) parsed from file names only."""
        try:
            names = os.listdir(self.directory)
        except FileNotFoundError:
            return []
        entries = []
        for name in names:
            if not name.endswith(_SUFFIX):
                continue
            parts = name[: -len(_SUFFIX)].split("-")
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            entries.append((int(parts[0]), parts[1], parts[2], self.directory / name))
        entries.sort()
        return entries

    @staticmethod
    def _unlink(path: Path) -> None:
        path.unlink(missing_ok=True)

    def _sweep_orphaned_temporaries(self, now_ms: int) -> None:
        """Remove write temporaries a crashed writer left behind for a full TTL."""
        try:
            names = os.listdir(self.directory)
        except FileNotFoundError:
            return
        for name in names:
            if not name.endswith(".tmp"):
                continue
            path = self.directory / name
            try:
                modified_ms = path.stat().st_mtime_ns // 1_000_000
            except FileNotFoundError:
                continue
            if modified_ms + SNAPSHOT_TTL_MS <= now_ms:
                self._unlink(path)

    def _prune(self, now_ms: int, *, reserve: int = 0, keep: str | None = None) -> None:
        """Expire idle handles and keep at most ``MAX_GLOBAL_SNAPSHOTS - reserve``.

        ``keep`` names a handle that is never the eviction victim: names that
        tie on the millisecond sort by random handle, so a prune after a write
        could otherwise evict the snapshot just created. Creation prunes before
        its write (reserving a slot) and again after it (keeping its own
        handle), so concurrent creators converge back to the bound instead of
        each trusting the same pre-write listing.
        """
        self._sweep_orphaned_temporaries(now_ms)
        live = []
        for entry in self._entries():
            if entry[0] + SNAPSHOT_TTL_MS <= now_ms:
                self._unlink(entry[3])
            else:
                live.append(entry)
        # Least recently used first, so the survivors are the handles in use.
        victims = [entry for entry in live if entry[2] != keep and entry[0] + IN_FLIGHT_GRACE_MS <= now_ms]
        for entry in victims[: max(0, len(live) - (MAX_GLOBAL_SNAPSHOTS - reserve))]:
            self._unlink(entry[3])

    def create(
        self,
        binding: SnapshotBinding,
        files: Sequence[tuple[Path, Any]],
    ) -> SearchSnapshot:
        rows = [
            [
                path.relative_to(binding.root).as_posix(),
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ]
            for path, info in files
        ]
        now_ms = _now_ms()
        handle = uuid.uuid4().hex
        principal_key = _principal_key(binding.principal)
        body = {"v": 1, "binding": binding.as_json(), "created_at_ms": now_ms, "files": rows}
        # Rosters under one deep prefix repeat it on every row; compression
        # keeps retained bytes proportional to distinct path content.
        encoded = zlib.compress(json.dumps(body, separators=(",", ":")).encode(), level=6)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # See _prune: reserve a slot before the write, then enforce the bound keeping this handle.
        self._prune(now_ms, reserve=1)
        atomic_replace(self.directory / f"{now_ms:013d}-{principal_key}-{handle}{_SUFFIX}", encoded, mode=0o600)
        self._prune(now_ms, keep=handle)
        return SearchSnapshot(handle, self._decode_rows(binding.root, rows))

    @staticmethod
    def _decode_rows(root: Path, rows: list[list[Any]]) -> tuple[tuple[Path, FileObservation], ...]:
        return tuple(
            (root / relative, FileObservation(int(dev), int(ino), int(size), int(mtime), int(ctime)))
            for relative, dev, ino, size, mtime, ctime in rows
        )

    def load(self, handle: str, binding: SnapshotBinding) -> SearchSnapshot:
        if len(handle) != 32 or not all(c in "0123456789abcdef" for c in handle):
            raise SnapshotUnavailableError("session continuation snapshot is malformed")
        now_ms = _now_ms()
        principal_key = _principal_key(binding.principal)
        # A concurrent resume of the same handle may rename it between our
        # listing and our read; that is a refreshed snapshot, not an eviction,
        # so look it up again under its new name.
        # Each retry follows a rename some other resume completed, so the loop
        # ends when the handle is found, expired, or gone -- not at a count.
        while True:
            entry = next(
                (row for row in self._entries() if row[2] == handle and row[1] == principal_key),
                None,
            )
            if entry is None:
                break
            last_used_ms, key, entry_handle, path = entry
            if last_used_ms + SNAPSHOT_TTL_MS <= now_ms:
                self._unlink(path)
                break
            try:
                body = json.loads(zlib.decompress(path.read_bytes()))
            except FileNotFoundError:
                continue
            except (OSError, ValueError, zlib.error):
                break
            if not isinstance(body, dict) or body.get("v") != 1 or body.get("binding") != binding.as_json():
                raise SnapshotUnavailableError("session continuation does not match its original search scope")
            # The TTL slides from last use: the name carries the timestamp and
            # a rename is atomic. Losing the rename to a concurrent resume
            # means that resume refreshed it; the contents are the same.
            with contextlib.suppress(FileNotFoundError):
                path.rename(self.directory / f"{now_ms:013d}-{key}-{entry_handle}{_SUFFIX}")
            return SearchSnapshot(handle, self._decode_rows(binding.root, body["files"]))
        raise SnapshotUnavailableError("session continuation expired or was evicted; restart the search")
