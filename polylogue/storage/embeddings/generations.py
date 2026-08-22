"""Archive-owned lifecycle for the replaceable embeddings tier.

The lifecycle is intentionally small and conservative.  It owns only the
archive-local pointer, generation metadata, and retention evidence; embedding
materialization remains the owner of SQLite rows.  Every state transition is
serialized by the archive lifecycle lock and all evidence is authenticated
before a pointer or generation is reclaimed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


class EmbeddingGenerationError(RuntimeError):
    """Raised when embedding generation state is unsafe to mutate."""


class EmbeddingGenerationState(StrEnum):
    ACTIVE = "active"
    RETAINED = "retained"
    ELIGIBLE = "eligible"
    RECLAIMED = "reclaimed"
    PROMOTING = "promoting"


@dataclass(frozen=True, slots=True)
class EmbeddingGeneration:
    generation_id: str
    archive_root: str
    database_path: str
    owner_id: str
    state: str
    created_at_ns: int
    promoted_at_ns: int = 0
    predecessor_generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingRetentionRecord:
    generation_id: str
    owner_id: str
    state: str


@dataclass(frozen=True, slots=True)
class EmbeddingPromotionReceipt:
    promoted_generation_id: str
    promoted_at_ns: int
    retention_boundary: int
    automatic: bool
    records: tuple[EmbeddingRetentionRecord, ...]
    eligible_generation_ids: tuple[str, ...] = ()
    reclaimed_generation_ids: tuple[str, ...] = ()


_GENERATIONS = ".embeddings-generations"
_RECEIPTS = "retention-receipts"
_RECLAMATION = "reclamation"
_MAX_RETAINED = 1
_ID = re.compile(r"^gen-[0-9]+-[0-9a-f]{10}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_REQUIRED_TABLES = {
    "message_embeddings",
    "message_embeddings_meta",
    "message_embedding_refs",
    "embedding_status",
    "embedding_derivation_state",
    "embedding_failures",
}


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _under(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


class EmbeddingGenerationStore:
    """Own archive-local embedding pointers and bounded rollback retention."""

    def __init__(self, archive_root: str | Path, *, active_path: str | Path | None = None) -> None:
        root = Path(archive_root).expanduser().absolute()
        if not root.is_absolute():
            raise EmbeddingGenerationError("archive root must be absolute")
        self.archive_root = root
        self.root = root / _GENERATIONS
        self.active_path = (Path(active_path) if active_path is not None else root / "embeddings.db").absolute()
        if not _under(root, self.active_path) or self.active_path.name != "embeddings.db":
            raise EmbeddingGenerationError("embedding active path must be archive-local embeddings.db")
        # These are archive-owned namespaces.  Never follow an operator- or
        # attacker-supplied link while creating lifecycle evidence.
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise EmbeddingGenerationError("embedding generation root is not an owned directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise EmbeddingGenerationError("embedding generation root is not an owned directory")
        self.receipts = self.root / _RECEIPTS
        if self.receipts.exists() and (self.receipts.is_symlink() or not self.receipts.is_dir()):
            raise EmbeddingGenerationError("embedding receipt root is not an owned directory")
        self.receipts.mkdir(parents=True, exist_ok=True)
        if self.receipts.is_symlink() or not self.receipts.is_dir():
            raise EmbeddingGenerationError("embedding receipt root is not an owned directory")
        self.reclamation = self.root / _RECLAMATION
        if self.reclamation.exists() and (self.reclamation.is_symlink() or not self.reclamation.is_dir()):
            raise EmbeddingGenerationError("embedding reclamation root is not an owned directory")
        self.reclamation.mkdir(parents=True, exist_ok=True)
        if self.reclamation.is_symlink() or not self.reclamation.is_dir():
            raise EmbeddingGenerationError("embedding reclamation root is not an owned directory")
        self.lock_path = self.root / ".lifecycle.lock"
        if self.lock_path.exists() and (self.lock_path.is_symlink() or not self.lock_path.is_file()):
            raise EmbeddingGenerationError("embedding lifecycle lock is not an owned file")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _metadata_path(self, generation_id: str) -> Path:
        if not _ID.fullmatch(generation_id):
            raise EmbeddingGenerationError("invalid embedding generation identity")
        return self.root / generation_id / "generation.json"

    def _validate_database(self, path: Path) -> None:
        if not _regular_file(path) or not _under(self.archive_root, path):
            raise EmbeddingGenerationError(f"embedding database is not an archive-owned regular file: {path}")
        if path.with_name(path.name + "-wal").exists() or path.with_name(path.name + "-shm").exists():
            raise EmbeddingGenerationError(f"embedding database has an uncheckpointed WAL: {path}")
        try:
            # The lifecycle lock serializes writers; avoid URI-only connection
            # options so embedding integrations can instrument ordinary
            # sqlite3.connect calls without changing admission semantics.
            with sqlite3.connect(path, timeout=1.0) as conn:
                ok, error = try_load_sqlite_vec(conn)
                if not ok:
                    raise EmbeddingGenerationError("embedding database requires sqlite-vec") from error
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version != EMBEDDINGS_SCHEMA_VERSION:
                    raise EmbeddingGenerationError(
                        f"embedding database has schema v{version}, expected v{EMBEDDINGS_SCHEMA_VERSION}"
                    )
                objects = {
                    str(row[0])
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
                }
                missing = _REQUIRED_TABLES - objects
                if missing:
                    raise EmbeddingGenerationError(f"embedding database is incomplete; missing {sorted(missing)}")
                if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise EmbeddingGenerationError("embedding database failed SQLite quick_check")
        except EmbeddingGenerationError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise EmbeddingGenerationError(f"malformed embedding database: {path}") from exc

    def _read_generation(self, path: Path) -> EmbeddingGeneration:
        try:
            if path.is_symlink() or path.name != "generation.json" or not _regular_file(path):
                raise ValueError("generation metadata must be a regular file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            generation = EmbeddingGeneration(**payload)
            if not _ID.fullmatch(generation.generation_id) or not _OWNER.fullmatch(generation.owner_id):
                raise ValueError("invalid generation or owner identity")
            expected_dir = self.root / generation.generation_id
            expected_db = expected_dir / "embeddings.db"
            if path.parent != expected_dir or generation.archive_root != str(self.archive_root):
                raise ValueError("generation metadata is not archive-bound")
            if Path(generation.database_path) != expected_db:
                raise ValueError("generation database path is not canonical")
            if generation.state not in {s.value for s in EmbeddingGenerationState}:
                raise ValueError("invalid generation state")
            if generation.created_at_ns <= 0 or generation.promoted_at_ns < 0:
                raise ValueError("invalid generation chronology")
            if generation.predecessor_generation_id is not None and not _ID.fullmatch(
                generation.predecessor_generation_id
            ):
                raise ValueError("invalid predecessor identity")
            if not _regular_file(expected_db):
                raise ValueError("generation database is missing")
            self._validate_database(expected_db)
            return generation
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise EmbeddingGenerationError(f"malformed embedding generation metadata: {path}") from exc

    def _generations(self) -> list[EmbeddingGeneration]:
        result: list[EmbeddingGeneration] = []
        for child in self.root.iterdir():
            if child.name in {_RECEIPTS, _RECLAMATION, ".lifecycle.lock"} or child.name.startswith("retired-"):
                # Retired artifacts predate this owner and are deliberately
                # outside the lifecycle namespace.  They must not poison
                # admission or become candidates for reclamation.
                continue
            if child.is_symlink() or not child.is_dir():
                raise EmbeddingGenerationError(f"unexpected embedding generation child: {child}")
            result.append(self._read_generation(child / "generation.json"))
        return result

    def _next_ns(self) -> int:
        latest = max((max(g.created_at_ns, g.promoted_at_ns) for g in self._generations()), default=0)
        return max(time.time_ns(), latest + 1)

    def _write_generation(self, generation: EmbeddingGeneration) -> None:
        _atomic_json(self._metadata_path(generation.generation_id), asdict(generation))

    def _active_generation(self, generations: list[EmbeddingGeneration]) -> EmbeddingGeneration | None:
        if self.active_path.is_symlink():
            target = self.active_path.resolve(strict=False)
            if not _under(self.archive_root, target) or not _regular_file(target):
                raise EmbeddingGenerationError("embedding active pointer is dangling or escapes archive root")
        elif self.active_path.exists():
            if not _regular_file(self.active_path):
                raise EmbeddingGenerationError("embedding active path is not a regular file")
            return None
        else:
            target = self.active_path.resolve(strict=False)
        matches = [g for g in generations if Path(g.database_path) == target]
        if len(matches) > 1:
            raise EmbeddingGenerationError("ambiguous embedding active generation metadata")
        if self.active_path.is_symlink() and not matches:
            raise EmbeddingGenerationError("embedding active pointer has no generation metadata")
        return matches[0] if matches else None

    def _validate_receipt(
        self, path: Path, generations: list[EmbeddingGeneration] | None = None
    ) -> EmbeddingPromotionReceipt:
        try:
            # Validate the caller-supplied path before opening it.  In
            # particular, a receipt symlink must never be read as evidence.
            if (
                path.is_symlink()
                or path.suffix != ".json"
                or not _regular_file(path)
                or path.parent != self.receipts
                or not _under(self.root, path)
            ):
                raise ValueError("receipt must be an owned regular JSON file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected_keys = {
                "promoted_generation_id",
                "promoted_at_ns",
                "retention_boundary",
                "automatic",
                "records",
                "eligible_generation_ids",
                "reclaimed_generation_ids",
            }
            if not isinstance(payload, dict) or set(payload) != expected_keys:
                raise ValueError("receipt schema is incomplete or has unknown fields")
            if not isinstance(payload["automatic"], bool) or not isinstance(payload["records"], list):
                raise ValueError("receipt fields have invalid types")
            if not isinstance(payload["eligible_generation_ids"], list) or not isinstance(
                payload["reclaimed_generation_ids"], list
            ):
                raise ValueError("receipt generation lists have invalid types")
            if payload["automatic"] is not True:
                raise ValueError("embedding lifecycle receipts must be automatic")
            if not isinstance(payload["promoted_generation_id"], str):
                raise ValueError("receipt generation identity has invalid type")
            if not isinstance(payload["promoted_at_ns"], int) or isinstance(payload["promoted_at_ns"], bool):
                raise ValueError("receipt chronology has invalid type")
            if not isinstance(payload["retention_boundary"], int) or isinstance(payload["retention_boundary"], bool):
                raise ValueError("receipt retention boundary has invalid type")
            if any(
                not isinstance(record, dict)
                or set(record) != {"generation_id", "owner_id", "state"}
                or not isinstance(record["generation_id"], str)
                or not isinstance(record["owner_id"], str)
                or not isinstance(record["state"], str)
                for record in payload["records"]
            ):
                raise ValueError("receipt records have invalid types")
            records = tuple(EmbeddingRetentionRecord(**record) for record in payload["records"])
            receipt = EmbeddingPromotionReceipt(
                payload["promoted_generation_id"],
                payload["promoted_at_ns"],
                payload["retention_boundary"],
                payload["automatic"],
                records,
                tuple(payload["eligible_generation_ids"]),
                tuple(payload["reclaimed_generation_ids"]),
            )
            if path.stem != receipt.promoted_generation_id or not _ID.fullmatch(receipt.promoted_generation_id):
                raise ValueError("receipt identity does not match filename")
            if receipt.promoted_at_ns <= 0 or receipt.retention_boundary != _MAX_RETAINED:
                raise ValueError("invalid receipt chronology or retention boundary")
            ids = [r.generation_id for r in receipt.records]
            if len(ids) != len(set(ids)) or any(
                not _ID.fullmatch(r.generation_id)
                or not _OWNER.fullmatch(r.owner_id)
                or r.state not in {"active", "retained", "eligible", "reclaimed"}
                for r in records
            ):
                raise ValueError("invalid receipt generation ownership")
            eligible = receipt.eligible_generation_ids
            reclaimed = receipt.reclaimed_generation_ids
            if any(not _ID.fullmatch(i) for i in (*eligible, *reclaimed)):
                raise ValueError("invalid receipt generation identity")
            if len(set(eligible)) != len(eligible) or len(set(reclaimed)) != len(reclaimed):
                raise ValueError("receipt generation lists contain duplicates")
            if set(eligible) & set(reclaimed):
                raise ValueError("receipt eligible and reclaimed sets overlap")
            if (
                receipt.promoted_generation_id not in ids
                or receipt.promoted_generation_id in eligible
                or receipt.promoted_generation_id in reclaimed
            ):
                raise ValueError("receipt active identity is inconsistent")
            state_by_id = {r.generation_id: r.state for r in records}
            if state_by_id[receipt.promoted_generation_id] != "active":
                raise ValueError("receipt active generation state is inconsistent")
            if any(state_by_id.get(i) != "eligible" for i in eligible):
                raise ValueError("receipt eligible state is inconsistent")
            if any(state_by_id.get(i) != "reclaimed" for i in reclaimed):
                raise ValueError("receipt reclaimed state is inconsistent")
            if generations is not None:
                by_id = {g.generation_id: g for g in generations}
                active = by_id.get(receipt.promoted_generation_id)
                if active is None or active.owner_id != next(
                    r.owner_id for r in records if r.generation_id == active.generation_id
                ):
                    raise ValueError("receipt active generation is stale or unbound")
                if active.promoted_at_ns != receipt.promoted_at_ns:
                    raise ValueError("receipt chronology does not match generation")
                # Entries absent from disk are permitted only after durable
                # reclamation and must be explicitly marked reclaimed.
                for record in records:
                    generation = by_id.get(record.generation_id)
                    if generation is None:
                        if record.state != "reclaimed" or record.generation_id not in reclaimed:
                            raise ValueError("receipt references an unknown generation")
                    elif generation.owner_id != record.owner_id:
                        raise ValueError("receipt generation metadata does not match")
            return receipt
        except (OSError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError) as exc:
            raise EmbeddingGenerationError(f"malformed embedding retention receipt: {path}") from exc

    def _validate_receipts(self, generations: list[EmbeddingGeneration]) -> list[tuple[int, Path]]:
        paths = sorted(self.receipts.iterdir(), key=lambda item: item.name)
        result = []
        decoded: list[tuple[EmbeddingPromotionReceipt, Path]] = []
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise EmbeddingGenerationError(f"unexpected embedding receipt child: {path}")
            decoded.append((self._validate_receipt(path), path))
        # Bind a receipt to the current namespace only while it names the
        # current active generation.  Once that generation is retained, the
        # receipt is immutable historical evidence and its old coverage cannot
        # include generations promoted later.
        active = self._active_generation(generations)
        active_id = active.generation_id if active is not None else None
        for receipt, path in decoded:
            if receipt.promoted_generation_id == active_id:
                receipt = self._validate_receipt(path, generations)
            result.append((receipt.promoted_at_ns, path))
        return result

    def _recover_interrupted_locked(self) -> None:
        generations = self._generations()
        active = self._active_generation(generations)
        for generation in generations:
            if generation.state != EmbeddingGenerationState.PROMOTING.value:
                continue
            candidate = Path(generation.database_path)
            self._validate_database(candidate)
            if active is not None and active.generation_id == generation.generation_id:
                self._write_generation(
                    EmbeddingGeneration(
                        **{
                            **asdict(generation),
                            "state": "active",
                            "promoted_at_ns": generation.promoted_at_ns or self._next_ns(),
                        }
                    )
                )
            elif active is None and generation.predecessor_generation_id is None:
                # Adoption intent was durable before the pointer swap.  Complete
                # it rather than creating a second owner for the legacy file.
                if not self.active_path.exists() or _regular_file(self.active_path):
                    temporary = self.active_path.with_name(f".{self.active_path.name}.{uuid.uuid4().hex}.tmp")
                    temporary.symlink_to(candidate)
                    os.replace(temporary, self.active_path)
                    _fsync_dir(self.active_path.parent)
                    self._write_generation(
                        EmbeddingGeneration(
                            **{
                                **asdict(generation),
                                "state": "active",
                                "promoted_at_ns": generation.promoted_at_ns or self._next_ns(),
                            }
                        )
                    )
                else:
                    raise EmbeddingGenerationError("embedding promotion target became unsafe during recovery")
            else:
                # A failed promotion with a predecessor cannot displace the
                # already-live database.  The candidate is safe to discard.
                shutil.rmtree(candidate.parent)
        _fsync_dir(self.root)

    def recover_interrupted(self) -> None:
        with self._lock():
            self._recover_interrupted_locked()

    @contextmanager
    def writer_lock(self) -> Iterator[Path]:
        """Admit one embedding SQLite writer for its complete write lifetime."""
        with self._lock():
            self._recover_interrupted_locked()
            generations = self._generations()
            self._validate_receipts(generations)
            active = self._active_generation(generations)
            if active is None and _regular_file(self.active_path):
                self._adopt_existing_active_locked()
            elif active is None and self.active_path.is_symlink():
                raise EmbeddingGenerationError("embedding active pointer has no active generation")
            elif active is None:
                raise EmbeddingGenerationError("embedding lifecycle has no active database")
            yield self.active_path

    def ensure_active(self) -> Path:
        with self._lock():
            generations = self._generations()
            self._validate_receipts(generations)
            active = self._active_generation(generations)
            if active is None and _regular_file(self.active_path):
                self._adopt_existing_active_locked()
            elif active is None and self.active_path.is_symlink():
                raise EmbeddingGenerationError("embedding active pointer has no active generation")
            _fsync_dir(self.root)
            return self.active_path

    def _adopt_existing_active_locked(self) -> EmbeddingGeneration | None:
        if self.active_path.is_symlink() or not _regular_file(self.active_path):
            return None
        self._validate_database(self.active_path)
        now = self._next_ns()
        generation_id = f"gen-{now}-{uuid.uuid4().hex[:10]}"
        destination = self.root / generation_id / "embeddings.db"
        destination.parent.mkdir(parents=True, exist_ok=False)
        with self.active_path.open("rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        _fsync_dir(destination.parent)
        generation = EmbeddingGeneration(
            generation_id, str(self.archive_root), str(destination), uuid.uuid4().hex, "promoting", now
        )
        self._write_generation(generation)
        _fsync_dir(self.root)
        temporary = self.active_path.with_name(f".{self.active_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.symlink_to(destination)
        os.replace(temporary, self.active_path)
        _fsync_dir(self.active_path.parent)
        promoted = EmbeddingGeneration(**{**asdict(generation), "state": "active", "promoted_at_ns": self._next_ns()})
        self._write_generation(promoted)
        return promoted

    def replace(self, candidate: str | Path, *, owner_id: str | None = None) -> Path:
        """Promote a validated archive-local candidate under the lifecycle lock."""
        candidate = Path(candidate).absolute()
        if candidate.is_symlink() or not _regular_file(candidate) or not _under(self.archive_root, candidate):
            raise EmbeddingGenerationError("embedding replacement candidate must be an archive-local regular file")
        if candidate == self.active_path or candidate.resolve(strict=False) == self.active_path.resolve(strict=False):
            raise EmbeddingGenerationError("embedding replacement candidate cannot be the active database")
        self._validate_database(candidate)
        with self._lock():
            generations = self._generations()
            self._validate_receipts(generations)
            current = self._active_generation(generations)
            if current is None and _regular_file(self.active_path):
                current = self._adopt_existing_active_locked()
                generations = self._generations()
            now = self._next_ns()
            generation_id = f"gen-{now}-{uuid.uuid4().hex[:10]}"
            destination = self.root / generation_id / "embeddings.db"
            destination.parent.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(candidate, destination)
            _fsync_file(destination)
            _fsync_dir(destination.parent)
            owner = owner_id or uuid.uuid4().hex
            if not _OWNER.fullmatch(owner):
                raise EmbeddingGenerationError("embedding owner identity is invalid")
            generation = EmbeddingGeneration(
                generation_id,
                str(self.archive_root),
                str(destination),
                owner,
                "promoting",
                now,
                predecessor_generation_id=current.generation_id if current else None,
            )
            self._write_generation(generation)
            _fsync_dir(self.root)
            temporary = self.active_path.with_name(f".{self.active_path.name}.{uuid.uuid4().hex}.tmp")
            temporary.symlink_to(destination)
            os.replace(temporary, self.active_path)
            _fsync_dir(self.active_path.parent)
            self._write_generation(
                EmbeddingGeneration(
                    **{
                        **asdict(generation),
                        "state": "active",
                        "promoted_at_ns": generation.promoted_at_ns or self._next_ns(),
                    }
                )
            )
            self._collect_locked()
            return self.active_path

    def collect(self) -> EmbeddingPromotionReceipt | None:
        with self._lock():
            return self._collect_locked()

    def _resume_reclamation_locked(self) -> None:
        """Finish durable deletion intents left by an interrupted collector."""
        for path in sorted(self.reclamation.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or path.suffix != ".json" or not _regular_file(path):
                raise EmbeddingGenerationError(f"unexpected embedding reclamation child: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if set(payload) != {"promoted_generation_id", "targets", "state"}:
                    raise ValueError("invalid reclamation schema")
                promoted = payload["promoted_generation_id"]
                targets = payload["targets"]
                state = payload["state"]
                if not isinstance(promoted, str) or _ID.fullmatch(promoted) is None:
                    raise ValueError("invalid reclamation owner")
                if path.stem != promoted or not isinstance(targets, list) or state not in {"pending", "complete"}:
                    raise ValueError("invalid reclamation identity")
                seen: set[str] = set()
                for target in targets:
                    if not isinstance(target, dict) or set(target) != {"generation_id", "owner_id"}:
                        raise ValueError("invalid reclamation target")
                    generation_id = target["generation_id"]
                    owner_id = target["owner_id"]
                    if (
                        not isinstance(generation_id, str)
                        or _ID.fullmatch(generation_id) is None
                        or generation_id in seen
                        or not isinstance(owner_id, str)
                        or not _OWNER.fullmatch(owner_id)
                    ):
                        raise ValueError("invalid reclamation target identity")
                    seen.add(generation_id)
                    directory = self.root / generation_id
                    if (
                        directory.is_symlink()
                        or (directory.exists() and not directory.is_dir())
                        or not _under(self.root, directory)
                    ):
                        raise EmbeddingGenerationError("embedding reclamation target is unsafe")
                    if directory.exists():
                        generation = self._read_generation(directory / "generation.json")
                        if (
                            generation.owner_id != owner_id
                            or generation.state != EmbeddingGenerationState.ELIGIBLE.value
                        ):
                            raise EmbeddingGenerationError("embedding reclamation target ownership changed")
                        shutil.rmtree(directory)
                if state != "complete":
                    _atomic_json(path, {"promoted_generation_id": promoted, "targets": targets, "state": "complete"})
            except EmbeddingGenerationError:
                raise
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise EmbeddingGenerationError(f"malformed embedding reclamation intent: {path}") from exc
        _fsync_dir(self.root)

    def _write_reclamation_intent(self, receipt: EmbeddingPromotionReceipt) -> Path:
        path = self.reclamation / f"{receipt.promoted_generation_id}.json"
        targets = [
            {"generation_id": record.generation_id, "owner_id": record.owner_id}
            for record in receipt.records
            if record.state == "eligible"
        ]
        _atomic_json(
            path, {"promoted_generation_id": receipt.promoted_generation_id, "targets": targets, "state": "pending"}
        )
        return path

    def _collect_locked(self) -> EmbeddingPromotionReceipt | None:
        self._resume_reclamation_locked()
        generations = self._generations()
        self._validate_receipts(generations)
        active = self._active_generation(generations)
        if active is None:
            return None
        if active.state not in {"active", "promoting"}:
            raise EmbeddingGenerationError("active embedding pointer names non-active metadata")
        predecessors = [
            g
            for g in generations
            if g.generation_id != active.generation_id and g.state in {"active", "retained", "eligible"}
        ]
        predecessors.sort(
            key=lambda g: (g.promoted_at_ns or g.created_at_ns, g.created_at_ns, g.generation_id), reverse=True
        )
        retained, eligible = predecessors[:_MAX_RETAINED], predecessors[_MAX_RETAINED:]
        records = [EmbeddingRetentionRecord(active.generation_id, active.owner_id, "active")]
        for generation in retained:
            self._write_generation(EmbeddingGeneration(**{**asdict(generation), "state": "retained"}))
            records.append(EmbeddingRetentionRecord(generation.generation_id, generation.owner_id, "retained"))
        for generation in eligible:
            self._write_generation(EmbeddingGeneration(**{**asdict(generation), "state": "eligible"}))
            records.append(EmbeddingRetentionRecord(generation.generation_id, generation.owner_id, "eligible"))
        receipt = EmbeddingPromotionReceipt(
            active.generation_id,
            active.promoted_at_ns,
            _MAX_RETAINED,
            True,
            tuple(records),
            tuple(g.generation_id for g in eligible),
        )
        self._write_receipt(receipt)
        # The intent is durable before the first directory removal.  Restart
        # can therefore distinguish an unfinished reclaim from corruption.
        self._write_reclamation_intent(receipt)
        receipt_files = self._validate_receipts(self._generations())
        receipt_files.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        for _, path in receipt_files[2:]:
            path.unlink()
        if len(receipt_files) > 2:
            _fsync_dir(self.receipts)
        self._resume_reclamation_locked()
        reclaimed = [generation.generation_id for generation in eligible]
        completed = EmbeddingPromotionReceipt(
            receipt.promoted_generation_id,
            receipt.promoted_at_ns,
            receipt.retention_boundary,
            True,
            tuple(
                EmbeddingRetentionRecord(
                    r.generation_id, r.owner_id, "reclaimed" if r.generation_id in reclaimed else r.state
                )
                for r in receipt.records
            ),
            (),
            tuple(reclaimed),
        )
        self._write_receipt(completed)
        return completed

    def _write_receipt(self, receipt: EmbeddingPromotionReceipt) -> None:
        _atomic_json(self.receipts / f"{receipt.promoted_generation_id}.json", asdict(receipt))

    def load_receipt(self, generation_id: object) -> EmbeddingPromotionReceipt:
        # Do not interpolate untrusted IDs into a path until the identity is
        # sealed.  This also blocks separators, dot segments, and symlink
        # traversal before any filesystem access occurs.
        if not isinstance(generation_id, str) or _ID.fullmatch(generation_id) is None:
            raise EmbeddingGenerationError("invalid embedding generation identity")
        path = self.receipts / f"{generation_id}.json"
        receipt = self._validate_receipt(path)
        generations = {generation.generation_id: generation for generation in self._generations()}
        for record in receipt.records:
            generation = generations.get(record.generation_id)
            if generation is not None and generation.owner_id != record.owner_id:
                raise EmbeddingGenerationError("receipt generation metadata does not match")
        return receipt


def ensure_embedding_lifecycle(archive_root: str | Path, *, active_path: str | Path | None = None) -> Path:
    """Actual daemon/CLI entrypoint for recovery, legacy adoption, and collection."""
    store = EmbeddingGenerationStore(archive_root, active_path=active_path)
    store.recover_interrupted()
    path = store.ensure_active()
    store.collect()
    return path


__all__ = [
    "EmbeddingGeneration",
    "EmbeddingGenerationError",
    "EmbeddingGenerationState",
    "EmbeddingGenerationStore",
    "EmbeddingPromotionReceipt",
    "EmbeddingRetentionRecord",
    "ensure_embedding_lifecycle",
]
