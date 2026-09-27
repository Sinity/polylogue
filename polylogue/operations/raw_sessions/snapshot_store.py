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
own none. Files are written atomically and bounded by a TTL and by
per-principal and global capacity; they survive process restart until they
expire or are evicted, after which a resume is reported as a typed degraded
outcome rather than silently restarting the scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.core.durable_fs import atomic_replace

SNAPSHOT_TTL_MS = 60 * 60 * 1000
MAX_PRINCIPAL_SNAPSHOTS = 4
MAX_GLOBAL_SNAPSHOTS = 64
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_SUFFIX = ".json"


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
        """(created_ms, principal_key, handle, path) parsed from file names only."""
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

    def _prune(self, now_ms: int, principal_key: str) -> None:
        entries = self._entries()
        live = []
        for entry in entries:
            if entry[0] + SNAPSHOT_TTL_MS <= now_ms:
                self._unlink(entry[3])
            else:
                live.append(entry)
        # Oldest first, so the survivors of each cap are the newest handles.
        mine = [entry for entry in live if entry[1] == principal_key]
        for entry in mine[: max(0, len(mine) - MAX_PRINCIPAL_SNAPSHOTS)]:
            self._unlink(entry[3])
            live.remove(entry)
        for entry in live[: max(0, len(live) - MAX_GLOBAL_SNAPSHOTS)]:
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
        encoded = json.dumps(body, separators=(",", ":")).encode()
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise ValueError("session search selected population exceeds snapshot capacity")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Capacity is reserved before the new handle exists so it counts as the newest.
        self._prune(now_ms, principal_key)
        atomic_replace(self.directory / f"{now_ms:013d}-{principal_key}-{handle}{_SUFFIX}", encoded, mode=0o600)
        self._prune(now_ms, principal_key)
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
        for created_ms, key, entry_handle, path in self._entries():
            if entry_handle != handle or key != principal_key:
                continue
            if created_ms + SNAPSHOT_TTL_MS <= now_ms:
                self._unlink(path)
                break
            try:
                body = json.loads(path.read_bytes())
            except (OSError, ValueError):
                break
            if not isinstance(body, dict) or body.get("v") != 1 or body.get("binding") != binding.as_json():
                raise SnapshotUnavailableError("session continuation does not match its original search scope")
            return SearchSnapshot(handle, self._decode_rows(binding.root, body["files"]))
        raise SnapshotUnavailableError("session continuation expired or was evicted; restart the search")
