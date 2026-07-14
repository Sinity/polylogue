"""Owned inactive index generations and atomic active-index promotion."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import sqlite3
import time
import uuid
from contextlib import closing
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
    source_snapshot: str = ""


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
        configured_index = archive_root / "index.db"
        anchor = archive_root / ".index-active-pointer"
        if anchor.exists():
            anchored = Path(anchor.read_text(encoding="utf-8").strip())
            if not anchored.is_absolute() or anchored.name != "index.db" or ".index-generations" in anchored.parts:
                raise RuntimeError(f"invalid canonical index pointer anchor: {anchored}")
            self.active_pointer = anchored
        else:
            if configured_index.is_symlink():
                target = Path(os.readlink(configured_index))
                self.active_pointer = target if target.is_absolute() else configured_index.parent / target
            else:
                self.active_pointer = configured_index
            temporary = anchor.with_suffix(".tmp")
            temporary.write_text(str(self.active_pointer.absolute()), encoding="utf-8")
            os.replace(temporary, anchor)
            _fsync_directory(anchor.parent)
        self.generations_root = self.active_pointer.parent / ".index-generations"

    def create(self, *, owner_id: str | None = None, source_snapshot: str) -> IndexGeneration:
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
            source_snapshot=source_snapshot,
        )
        self._write(generation)
        return generation

    def load(self, generation_id: str) -> IndexGeneration:
        payload = json.loads(self._metadata_path(generation_id).read_text(encoding="utf-8"))
        return IndexGeneration(**payload)

    def promote(self, generation: IndexGeneration) -> IndexGeneration:
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            raise RuntimeError("only the owning inactive generation can be promoted")
        target = Path(current.index_path).resolve(strict=True)
        _checkpoint_truncate(target, label="new index")
        pointer = self.active_pointer
        retired = self.generations_root / f"retired-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        retired.mkdir(parents=True, exist_ok=False)
        if pointer.exists() or pointer.is_symlink():
            _checkpoint_truncate(pointer, label="active index")
            for suffix in ("-wal", "-shm"):
                sidecar = pointer.with_name(pointer.name + suffix)
                if sidecar.exists():
                    if suffix == "-wal" and sidecar.stat().st_size != 0:
                        raise RuntimeError(f"non-empty active index sidecar blocks promotion: {sidecar}")
                    os.replace(sidecar, retired / sidecar.name)
        if pointer.exists() or pointer.is_symlink():
            os.link(pointer, retired / "index.db", follow_symlinks=False)
            _fsync_directory(retired)
        promoting = IndexGeneration(**{**asdict(current), "state": "promoting"})
        self._write(promoting)
        temporary = pointer.parent / f".index.db.promote-{uuid.uuid4().hex}"
        temporary.symlink_to(target)
        os.replace(temporary, pointer)
        _fsync_directory(pointer.parent)
        promoted = IndexGeneration(**{**asdict(current), "state": "active"})
        self._write(promoted)
        return promoted

    def recover_promotion(self, generation_id: str) -> IndexGeneration:
        """Reconcile a crash after the pointer swap but before active metadata."""
        generation = self.load(generation_id)
        if generation.state != "promoting":
            return generation
        pointer = self.active_pointer
        state = "inactive"
        if pointer.exists() or pointer.is_symlink():
            state = (
                "active"
                if pointer.resolve(strict=True) == Path(generation.index_path).resolve(strict=True)
                else "inactive"
            )
        recovered = IndexGeneration(**{**asdict(generation), "state": state})
        self._write(recovered)
        return recovered

    def discard_if_inactive(self, generation: IndexGeneration) -> bool:
        """Remove a terminal failed candidate without risking an active target."""
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            return False
        shutil.rmtree(self._metadata_path(generation.generation_id).parent)
        _fsync_directory(self.generations_root)
        return True

    def _metadata_path(self, generation_id: str) -> Path:
        return self.generations_root / generation_id / "generation.json"

    def _write(self, generation: IndexGeneration) -> None:
        path = self._metadata_path(generation.generation_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(generation), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        _fsync_directory(path.parent)


def source_revision_snapshot(archive_root: Path) -> str:
    """Stable raw-evidence vector used to reject an unsafe rebuild delta."""
    import hashlib

    digest = hashlib.sha256()
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        for raw_id, acquired_at_ms in conn.execute(
            "SELECT raw_id, acquired_at_ms FROM raw_sessions ORDER BY acquired_at_ms, raw_id"
        ):
            digest.update(str(raw_id).encode())
            digest.update(b"\0")
            digest.update(str(acquired_at_ms).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_truncate(path: Path, *, label: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0:
        raise RuntimeError(f"{label} WAL checkpoint failed: {checkpoint!r}")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "ActiveWriterLease",
    "IndexGeneration",
    "IndexGenerationStore",
    "RebuildLease",
    "RebuildLeaseUnavailableError",
    "source_revision_snapshot",
]
