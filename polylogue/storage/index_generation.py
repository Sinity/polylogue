"""Owned inactive index generations and atomic active-index promotion."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


@dataclass(frozen=True, slots=True)
class IndexGeneration:
    generation_id: str
    owner_id: str
    archive_root: str
    index_path: str
    state: str
    created_at_ms: int
    source_high_water: tuple[str, ...] = ()
    cursor: int = 0


class RebuildLeaseUnavailableError(RuntimeError):
    """Another process owns the archive-wide rebuild lease."""


class RebuildLease:
    """Process-held exclusive lease for an offline index rebuild."""

    def __init__(self, archive_root: Path) -> None:
        self.path = archive_root / ".index-rebuild.lock"
        self._fd: int | None = None

    def __enter__(self) -> RebuildLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RebuildLeaseUnavailableError(f"index rebuild lease is already held: {self.path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} host={socket.gethostname()}\n".encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class ActiveWriterLease:
    """Shared process-held lease refused while an offline rebuild owns the archive."""

    def __init__(self, archive_root: Path) -> None:
        self.path = archive_root / ".index-rebuild.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RebuildLeaseUnavailableError(f"offline index rebuild owns archive: {self.path}") from exc
        self._fd = fd

    def close(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class IndexGenerationStore:
    """Create, checkpoint, and atomically promote inactive generations."""

    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root
        self.generations_root = archive_root / ".index-generations"

    def create(self, *, owner_id: str | None = None, source_high_water: tuple[str, ...] = ()) -> IndexGeneration:
        generation_id = f"gen-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        owner = owner_id or str(uuid.uuid4())
        root = self.generations_root / generation_id
        root.mkdir(parents=True, exist_ok=False)
        for filename in ("source.db", "user.db", "embeddings.db", "ops.db", "blob"):
            source = self.archive_root / filename
            if source.exists() or source.is_symlink():
                (root / filename).symlink_to(source.resolve(strict=False), target_is_directory=source.is_dir())
        index_path = root / "index.db"
        initialize_archive_database(index_path, ArchiveTier.INDEX)
        generation = IndexGeneration(
            generation_id=generation_id,
            owner_id=owner,
            archive_root=str(self.archive_root.resolve(strict=False)),
            index_path=str(index_path),
            state="inactive",
            created_at_ms=int(time.time() * 1000),
            source_high_water=source_high_water,
        )
        self._write(generation)
        return generation

    def load(self, generation_id: str) -> IndexGeneration:
        payload = json.loads(self._metadata_path(generation_id).read_text(encoding="utf-8"))
        payload["source_high_water"] = tuple(payload.get("source_high_water", ()))
        return IndexGeneration(**payload)

    def checkpoint(self, generation: IndexGeneration, *, cursor: int) -> IndexGeneration:
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            raise RuntimeError("generation ownership changed before checkpoint")
        updated = IndexGeneration(**{**asdict(current), "cursor": cursor})
        self._write(updated)
        return updated

    def promote(self, generation: IndexGeneration) -> IndexGeneration:
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            raise RuntimeError("only the owning inactive generation can be promoted")
        target = Path(current.index_path).resolve(strict=True)
        pointer = self.archive_root / "index.db"
        retired = self.generations_root / f"retired-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        retired.mkdir(parents=True, exist_ok=False)
        if pointer.exists() or pointer.is_symlink():
            os.link(pointer, retired / "index.db", follow_symlinks=False)
        temporary = self.archive_root / f".index.db.promote-{uuid.uuid4().hex}"
        temporary.symlink_to(target)
        os.replace(temporary, pointer)
        promoted = IndexGeneration(**{**asdict(current), "state": "active"})
        self._write(promoted)
        return promoted

    def _metadata_path(self, generation_id: str) -> Path:
        return self.generations_root / generation_id / "generation.json"

    def _write(self, generation: IndexGeneration) -> None:
        path = self._metadata_path(generation.generation_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(generation), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)


__all__ = [
    "ActiveWriterLease",
    "IndexGeneration",
    "IndexGenerationStore",
    "RebuildLease",
    "RebuildLeaseUnavailableError",
]
