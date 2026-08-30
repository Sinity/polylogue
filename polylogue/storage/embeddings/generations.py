"""Archive-owned lifecycle for the replaceable embeddings tier.

The lifecycle is intentionally small and conservative.  It owns only the
archive-local pointer, generation metadata, and retention evidence; embedding
materialization remains the owner of SQLite rows.  Every state transition is
serialized by the archive lifecycle lock and all evidence is authenticated
before a pointer or generation is reclaimed.
"""

from __future__ import annotations

import fcntl
import hashlib
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
from typing import Any

from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


class EmbeddingGenerationError(RuntimeError):
    """Raised when embedding generation state is unsafe to mutate."""


class EmbeddingGenerationState(StrEnum):
    IN_PROGRESS = "in_progress"
    ACCEPTED = "accepted"
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
    # These fields are part of the generation identity, rather than advisory
    # receipt data.  A generation whose computation contract or archive
    # inputs cannot be proved is not safe to activate or reclaim.
    recipe_hash: str = ""
    source_generation: str = ""
    index_generation: str = ""
    schema_version: int = EMBEDDINGS_SCHEMA_VERSION
    physical_root: str = ""
    sealed: bool = False
    membership_digest: str = ""
    lease_owner: str | None = None
    reservation_owner: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingGenerationBinding:
    """Authenticated admission token for one archive embedding generation.

    A pathname is only a locator.  The device/inode observations make the
    locator an admission token: replacing the archive root, active pointer,
    or generation file cannot silently redirect a write that is already in
    flight.
    """

    archive_root: str
    archive_root_identity: tuple[int, int]
    generation_id: str
    owner_id: str
    database_path: str
    database_identity: tuple[int, int]
    active_path_identity: tuple[int, int]

    def __fspath__(self) -> str:
        return self.database_path

    def __str__(self) -> str:
        return self.database_path


@dataclass(frozen=True, slots=True)
class EmbeddingRetentionRecord:
    generation_id: str
    owner_id: str
    state: str


@dataclass(frozen=True, slots=True)
class EmbeddingPromotionReceipt:
    archive_root: str
    archive_root_identity: tuple[int, int]
    promoted_generation_id: str
    promoted_at_ns: int
    retention_boundary: int
    automatic: bool
    records: tuple[EmbeddingRetentionRecord, ...]
    eligible_generation_ids: tuple[str, ...] = ()
    reclaimed_generation_ids: tuple[str, ...] = ()


_GENERATIONS = ".embeddings-generations"
_RECEIPTS = "retention-receipts"
_MAX_RETAINED = 1
_ID = re.compile(r"^gen-[0-9]+-[0-9a-f]{10}$")
_RETIRED = re.compile(r"^retired-(gen-[0-9]+-[0-9a-f]{10})-[0-9a-f]{32}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_REQUIRED_TABLES = {
    "message_embeddings",
    "message_embeddings_meta",
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
        if root.is_symlink() or not root.is_dir():
            raise EmbeddingGenerationError("embedding archive root must be an owned directory")
        self.archive_root = root
        self.root = root / _GENERATIONS
        self.active_path = (Path(active_path) if active_path is not None else root / "embeddings.db").absolute()
        if not _under(root, self.active_path) or self.active_path.name != "embeddings.db":
            raise EmbeddingGenerationError("embedding active path must be archive-local embeddings.db")
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
        self.lock_path = self.root / ".lifecycle.lock"

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
        uri = f"file:{path}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
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

    def _database_contract(self, path: Path, *, physical_root: Path | None = None) -> dict[str, Any]:
        """Derive the immutable contract carried by a published database.

        The database is rebuildable, but its publication must still be
        attributable.  In particular, an otherwise valid SQLite file copied
        from another archive must not become an active generation merely
        because its filename looks plausible.
        """
        self._validate_database(path)
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                rows = conn.execute(
                    """
                    SELECT vector_derivation_hash, model, dimension, recipe_hash, output_contract_hash
                    FROM message_embeddings_meta
                    ORDER BY vector_derivation_hash
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise EmbeddingGenerationError("cannot read embedding membership digest") from exc
        digest = hashlib.sha256()
        recipe_values: set[bytes] = set()
        output_values: set[bytes] = set()
        model_values: set[str] = set()
        dimensions: set[int] = set()
        for vector_hash, model, dimension, recipe_hash, output_contract_hash in rows:
            value = bytes(vector_hash)
            recipe_value = bytes(recipe_hash)
            output_value = bytes(output_contract_hash)
            if len(value) != 32:
                raise EmbeddingGenerationError("embedding membership contains malformed vector identity")
            if len(recipe_value) != 32 or len(output_value) != 32:
                raise EmbeddingGenerationError("embedding membership contains malformed recipe identity")
            recipe_values.add(recipe_value)
            output_values.add(output_value)
            model_values.add(str(model))
            dimensions.add(int(dimension))
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        if len(recipe_values) > 1 or len(output_values) > 1 or len(model_values) > 1 or len(dimensions) > 1:
            raise EmbeddingGenerationError("embedding membership contains mixed vector contracts")
        digest.update(len(rows).to_bytes(8, "big"))

        def stable(path_value: Path) -> str:
            try:
                stat = path_value.stat()
            except OSError:
                return f"missing:{path_value.absolute()}"
            return f"dev:{stat.st_dev}:ino:{stat.st_ino}"

        index = self.archive_root / "index.db"
        source = self.archive_root / "source.db"
        return {
            "recipe_hash": next(iter(recipe_values)).hex() if recipe_values else "empty",
            "source_generation": stable(source),
            "index_generation": stable(index),
            "schema_version": EMBEDDINGS_SCHEMA_VERSION,
            "physical_root": str((physical_root or path.parent).absolute()),
            "sealed": True,
            "membership_digest": digest.hexdigest(),
        }

    def prepare_legacy_active_database(self) -> None:
        """Checkpoint a pre-lifecycle active database before copying it.

        The lifecycle never copies bytes accompanied by SQLite sidecars.  The
        daemon owns the archive writer when this route runs, so ask SQLite to
        complete a truncate checkpoint and reject the handoff if any sidecar
        remains.  In particular, do not unlink a WAL or SHM file ourselves:
        SQLite remains the authority for recovery and lock safety.
        """
        if self.active_path.is_symlink() or not self.active_path.exists():
            return
        if not _regular_file(self.active_path):
            raise EmbeddingGenerationError("embedding active path is not a regular file")
        try:
            with sqlite3.connect(self.active_path, timeout=30.0) as conn:
                row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except (OSError, sqlite3.Error) as exc:
            raise EmbeddingGenerationError("could not checkpoint legacy embedding database") from exc
        if row is None or int(row[0] or 0) != 0:
            raise EmbeddingGenerationError("legacy embedding database checkpoint is blocked")
        sidecars = tuple(
            path
            for path in (
                self.active_path.with_name(self.active_path.name + "-wal"),
                self.active_path.with_name(self.active_path.name + "-shm"),
            )
            if path.exists()
        )
        if sidecars:
            raise EmbeddingGenerationError("legacy embedding database retains SQLite sidecars after checkpoint")

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
            if (
                not generation.recipe_hash
                or not generation.source_generation
                or not generation.index_generation
                or generation.schema_version != EMBEDDINGS_SCHEMA_VERSION
                or Path(generation.physical_root) != expected_dir
                or not _under(self.root, Path(generation.physical_root))
                or generation.sealed is not True
                or not re.fullmatch(r"[0-9a-f]{64}", generation.membership_digest)
            ):
                raise ValueError("generation metadata is incomplete or unsealed")
            if generation.predecessor_generation_id is not None and not _ID.fullmatch(
                generation.predecessor_generation_id
            ):
                raise ValueError("invalid predecessor identity")
            for label, owner in (("lease", generation.lease_owner), ("reservation", generation.reservation_owner)):
                if owner is not None and not _OWNER.fullmatch(owner):
                    raise ValueError(f"invalid {label} owner identity")
            if not _regular_file(expected_db):
                raise ValueError("generation database is missing")
            self._validate_database(expected_db)
            actual_contract = self._database_contract(expected_db, physical_root=expected_db.parent)
            for field in (
                "recipe_hash",
                "source_generation",
                "index_generation",
                "schema_version",
                "physical_root",
                "sealed",
                "membership_digest",
            ):
                if getattr(generation, field) != actual_contract[field]:
                    raise ValueError(f"generation metadata contract mismatch: {field}")
            return generation
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise EmbeddingGenerationError(f"malformed embedding generation metadata: {path}") from exc

    def _generations(self) -> list[EmbeddingGeneration]:
        result: list[EmbeddingGeneration] = []
        for child in self.root.iterdir():
            if child.name in {_RECEIPTS, ".lifecycle.lock"} or child.name.startswith("retired-"):
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

    @staticmethod
    def _identity(path: Path, *, label: str) -> tuple[int, int]:
        try:
            info = path.stat()
        except OSError as exc:
            raise EmbeddingGenerationError(f"{label} disappeared during lifecycle admission") from exc
        return int(info.st_dev), int(info.st_ino)

    @staticmethod
    def _link_identity(path: Path, *, label: str) -> tuple[int, int]:
        try:
            info = path.lstat()
        except OSError as exc:
            raise EmbeddingGenerationError(f"{label} disappeared during lifecycle admission") from exc
        return int(info.st_dev), int(info.st_ino)

    def _binding(
        self, generations: list[EmbeddingGeneration], active: EmbeddingGeneration
    ) -> EmbeddingGenerationBinding:
        if active.state not in {"active", "promoting"}:
            raise EmbeddingGenerationError("embedding active pointer names non-active metadata")
        return EmbeddingGenerationBinding(
            archive_root=str(self.archive_root),
            archive_root_identity=self._identity(self.archive_root, label="embedding archive root"),
            generation_id=active.generation_id,
            owner_id=active.owner_id,
            database_path=active.database_path,
            database_identity=self._identity(Path(active.database_path), label="embedding database"),
            active_path_identity=self._link_identity(self.active_path, label="embedding active pointer"),
        )

    def assert_binding(self, binding: EmbeddingGenerationBinding) -> None:
        """Reject a binding whose root, pointer, or generation changed."""
        if str(self.archive_root) != binding.archive_root:
            raise EmbeddingGenerationError("embedding archive root binding mismatch")
        if self._identity(self.archive_root, label="embedding archive root") != binding.archive_root_identity:
            raise EmbeddingGenerationError("embedding archive root was replaced during materialization")
        if self._link_identity(self.active_path, label="embedding active pointer") != binding.active_path_identity:
            raise EmbeddingGenerationError("embedding active pointer was replaced during materialization")
        generations = self._generations()
        self._validate_receipts(generations)
        active = self._active_generation(generations)
        if active is None or active.generation_id != binding.generation_id or active.owner_id != binding.owner_id:
            raise EmbeddingGenerationError("embedding active generation changed during materialization")
        if active.database_path != binding.database_path:
            raise EmbeddingGenerationError("embedding generation database path changed during materialization")
        if self._identity(Path(binding.database_path), label="embedding database") != binding.database_identity:
            raise EmbeddingGenerationError("embedding generation database was replaced during materialization")

    def _validate_receipt(
        self, path: Path, generations: list[EmbeddingGeneration] | None = None
    ) -> EmbeddingPromotionReceipt:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.is_symlink() or path.suffix != ".json" or not _regular_file(path):
                raise ValueError("receipt must be a regular JSON file")
            expected_keys = {
                "archive_root",
                "archive_root_identity",
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
            archive_root_identity_values = tuple(int(x) for x in payload["archive_root_identity"])
            if len(archive_root_identity_values) != 2:
                raise ValueError("receipt archive root identity is stale or unbound")
            archive_root_identity = (archive_root_identity_values[0], archive_root_identity_values[1])
            records = tuple(EmbeddingRetentionRecord(**record) for record in payload["records"])
            receipt = EmbeddingPromotionReceipt(
                str(payload["archive_root"]),
                archive_root_identity,
                str(payload["promoted_generation_id"]),
                int(payload["promoted_at_ns"]),
                int(payload["retention_boundary"]),
                payload["automatic"],
                records,
                tuple(str(x) for x in payload["eligible_generation_ids"]),
                tuple(str(x) for x in payload["reclaimed_generation_ids"]),
            )
            if receipt.archive_root != str(self.archive_root) or len(receipt.archive_root_identity) != 2:
                raise ValueError("receipt archive root identity is stale or unbound")
            if receipt.archive_root_identity != self._identity(self.archive_root, label="embedding archive root"):
                raise ValueError("receipt archive root was replaced")
            if path.stem != receipt.promoted_generation_id or not _ID.fullmatch(receipt.promoted_generation_id):
                raise ValueError("receipt identity does not match filename")
            if receipt.promoted_at_ns <= 0 or receipt.retention_boundary != _MAX_RETAINED:
                raise ValueError("invalid receipt chronology or retention boundary")
            ids = [r.generation_id for r in receipt.records]
            if len(ids) != len(set(ids)) or any(
                not _ID.fullmatch(i)
                or not _OWNER.fullmatch(r.owner_id)
                or r.state not in {"active", "retained", "eligible", "reclaimed"}
                for i, r in zip(ids, records, strict=True)
            ):
                raise ValueError("invalid receipt generation ownership")
            if (
                receipt.promoted_generation_id not in ids
                or receipt.promoted_generation_id in receipt.eligible_generation_ids
            ):
                raise ValueError("receipt active identity is inconsistent")
            if any(not _ID.fullmatch(i) for i in (*receipt.eligible_generation_ids, *receipt.reclaimed_generation_ids)):
                raise ValueError("invalid receipt generation identity")
            if generations is not None:
                by_id = {g.generation_id: g for g in generations}
                active = by_id.get(receipt.promoted_generation_id)
                if active is None or active.owner_id != next(
                    r.owner_id for r in records if r.generation_id == active.generation_id
                ):
                    raise ValueError("receipt active generation is stale or unbound")
                if active.promoted_at_ns != receipt.promoted_at_ns:
                    raise ValueError("receipt chronology does not match generation")
                for record in records:
                    generation = by_id.get(record.generation_id)
                    if generation is not None and generation.owner_id != record.owner_id:
                        raise ValueError("receipt owner does not match generation")
            return receipt
        except (OSError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError) as exc:
            raise EmbeddingGenerationError(f"malformed embedding retention receipt: {path}") from exc

    def _validate_receipts(self, generations: list[EmbeddingGeneration]) -> list[tuple[int, Path]]:
        result = []
        for path in self.receipts.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise EmbeddingGenerationError(f"unexpected embedding receipt child: {path}")
            receipt = self._validate_receipt(path, generations)
            result.append((receipt.promoted_at_ns, path))
        return result

    def recover_interrupted(self) -> None:
        with self._lock():
            generations = self._generations()
            receipts = self._validate_receipts(generations)
            planned_reclaims: set[str] = set()
            for _, path in receipts:
                receipt = self._validate_receipt(path, generations)
                planned_reclaims.update(receipt.eligible_generation_ids)
                planned_reclaims.update(receipt.reclaimed_generation_ids)
            for child in self.root.iterdir():
                match = _RETIRED.fullmatch(child.name)
                if not match:
                    continue
                if child.is_symlink() or not child.is_dir() or match.group(1) not in planned_reclaims:
                    raise EmbeddingGenerationError(f"unbound retired embedding generation: {child}")
                shutil.rmtree(child)
            if planned_reclaims:
                _fsync_dir(self.root)
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
                    if self.active_path.exists() and not _regular_file(self.active_path):
                        raise EmbeddingGenerationError("embedding active path is not a regular file")
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
                    shutil.rmtree(candidate.parent)
            _fsync_dir(self.root)

    @contextmanager
    def writer_lock(self) -> Iterator[EmbeddingGenerationBinding]:
        """Admit one embedding SQLite writer for its complete write lifetime."""
        with self._lock():
            generations = self._generations()
            self._validate_receipts(generations)
            active = self._active_generation(generations)
            if active is None and _regular_file(self.active_path):
                self._adopt_existing_active_locked()
            elif active is None and self.active_path.is_symlink():
                raise EmbeddingGenerationError("embedding active pointer has no active generation")
            elif active is None:
                raise EmbeddingGenerationError("embedding lifecycle has no active database")
            generations = self._generations()
            active = self._active_generation(generations)
            if active is None:
                raise EmbeddingGenerationError("embedding lifecycle has no active database")
            binding = self._binding(generations, active)
            yield binding

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
            generation_id,
            str(self.archive_root),
            str(destination),
            uuid.uuid4().hex,
            "promoting",
            now,
            **self._database_contract(destination, physical_root=destination.parent),
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
                **self._database_contract(destination, physical_root=destination.parent),
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

    def _collect_locked(self) -> EmbeddingPromotionReceipt | None:
        generations = self._generations()
        self._validate_receipts(generations)
        active = self._active_generation(generations)
        if active is None:
            return None
        if active.state not in {"active", "promoting"}:
            raise EmbeddingGenerationError("active embedding pointer names non-active metadata")
        # Only the whole inventory can establish liveness.  In particular,
        # accepted candidates and work carrying a lease or reservation remain
        # protected even when they are not named by the active pointer.
        predecessors = [
            g
            for g in generations
            if g.generation_id != active.generation_id
            and g.state in {"active", "retained", "eligible"}
            and g.lease_owner is None
            and g.reservation_owner is None
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
        # The inventory is the only reclamation authority.  Capture the
        # complete candidate set before publishing the plan; the set is
        # checked again immediately before any directory mutation.  This is
        # deliberately an identity check, not a timestamp/newest-file
        # heuristic: a newly arrived candidate, lease, or promotion makes the
        # old plan stale and therefore harmlessly aborts this pass.
        retained_ids = {item.generation_id for item in retained}
        eligible_ids = {item.generation_id for item in eligible}

        def inventory_state(generation: EmbeddingGeneration) -> str:
            if generation.generation_id == active.generation_id:
                return EmbeddingGenerationState.ACTIVE.value
            if generation.generation_id in retained_ids:
                return EmbeddingGenerationState.RETAINED.value
            if generation.generation_id in eligible_ids:
                return EmbeddingGenerationState.ELIGIBLE.value
            return generation.state

        planned_inventory = tuple(
            sorted(
                (
                    g.generation_id,
                    g.owner_id,
                    inventory_state(g),
                )
                for g in generations
            )
        )
        receipt = EmbeddingPromotionReceipt(
            str(self.archive_root),
            self._identity(self.archive_root, label="embedding archive root"),
            active.generation_id,
            active.promoted_at_ns,
            _MAX_RETAINED,
            True,
            tuple(records),
            tuple(g.generation_id for g in eligible),
        )
        self._write_receipt(receipt)
        current_inventory = tuple(sorted((g.generation_id, g.owner_id, g.state) for g in self._generations()))
        if current_inventory != planned_inventory:
            raise EmbeddingGenerationError("embedding reclamation inventory changed; retry")
        receipt_files = self._validate_receipts(self._generations())
        receipt_files.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        for _, path in receipt_files[2:]:
            path.unlink()
        if len(receipt_files) > 2:
            _fsync_dir(self.receipts)
        reclaimed = []
        for generation in eligible:
            # Revalidate the whole union before each mutation.  A concurrent
            # writer normally cannot pass the lifecycle lock, but this check
            # also protects against an independently restored/leased
            # generation and is the fail-closed boundary for crash recovery.
            current_inventory = tuple(sorted((g.generation_id, g.owner_id, g.state) for g in self._generations()))
            if current_inventory != planned_inventory:
                raise EmbeddingGenerationError("embedding reclamation inventory changed; retry")
            self._reclaim_generation(generation)
            reclaimed.append(generation.generation_id)
        completed = EmbeddingPromotionReceipt(
            receipt.archive_root,
            receipt.archive_root_identity,
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
            receipt.eligible_generation_ids,
            tuple(reclaimed),
        )
        self._write_receipt(completed)
        return completed

    def _write_receipt(self, receipt: EmbeddingPromotionReceipt) -> None:
        _atomic_json(self.receipts / f"{receipt.promoted_generation_id}.json", asdict(receipt))

    def _reclaim_generation(self, generation: EmbeddingGeneration) -> None:
        """Move a planned generation out of the live namespace before unlinking."""
        directory = self._metadata_path(generation.generation_id).parent
        if directory.is_symlink() or not directory.is_dir() or not _under(self.root, directory):
            raise EmbeddingGenerationError("embedding reclaim directory is unsafe")
        trash = self.root / f"retired-{generation.generation_id}-{uuid.uuid4().hex}"
        os.replace(directory, trash)
        _fsync_dir(trash.parent)
        _fsync_dir(self.root)
        shutil.rmtree(trash)
        _fsync_dir(self.root)

    def load_receipt(self, generation_id: str) -> EmbeddingPromotionReceipt:
        if not _ID.fullmatch(generation_id):
            raise EmbeddingGenerationError("invalid embedding generation identity")
        return self._validate_receipt(self.receipts / f"{generation_id}.json")


def ensure_embedding_lifecycle(archive_root: str | Path, *, active_path: str | Path | None = None) -> Path:
    """Actual daemon/CLI entrypoint for recovery, legacy adoption, and collection."""
    store = EmbeddingGenerationStore(archive_root, active_path=active_path)
    store.prepare_legacy_active_database()
    store.recover_interrupted()
    path = store.ensure_active()
    store.collect()
    return path


__all__ = [
    "EmbeddingGeneration",
    "EmbeddingGenerationBinding",
    "EmbeddingGenerationError",
    "EmbeddingGenerationState",
    "EmbeddingGenerationStore",
    "EmbeddingPromotionReceipt",
    "EmbeddingRetentionRecord",
    "ensure_embedding_lifecycle",
]
