"""Build-specific source denominator from the daemon's production discovery route."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.zip_admission import ZIP_JSON_SUFFIXES, ZipAdmission, ZipBombError
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.raw_coordinates import zip_member_source_index
from polylogue.maintenance.receipt_fs import (
    atomic_replace_receipt,
    existing_maintenance_receipt_directory,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.sources.decoder_zip import (
    ZipEntryValidator,
    declared_artifact_provider,
    is_declared_artifact_path,
    provider_detection_path,
)
from polylogue.sources.live.batch_support import classify_pre_acquisition
from polylogue.sources.live.discovery import _source_path_steps
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    replay_zip_entry_acquisition_payloads,
    sniff_zip_provider,
)
from polylogue.sources.sqlite_snapshot import is_sqlite_path, sqlite_member_revision_and_size
from polylogue.sources.walk_faults import WalkRefusedError
from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_PENDING_DIR = "production-source-baseline"
_PENDING_FILE = "pending.json"
MATERIAL_BYTE_DEFINITION = "retained-canonical-payload-v1"
_RETRYABLE_READ_ERRNOS = frozenset(
    {
        errno.EIO,
        errno.EACCES,
        errno.EPERM,
        errno.ESTALE,
        errno.ETIMEDOUT,
        errno.EAGAIN,
        errno.EBUSY,
        errno.ENOSPC,
        errno.EDQUOT,
    }
)


def _retryable_read_fault(exc: Exception) -> bool:
    sqlite_code = getattr(exc, "sqlite_errorcode", None)
    return (isinstance(exc, OSError) and exc.errno in _RETRYABLE_READ_ERRNOS) or (
        isinstance(exc, sqlite3.Error)
        and isinstance(sqlite_code, int)
        and sqlite_code & 0xFF
        in {
            sqlite3.SQLITE_IOERR,
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
            sqlite3.SQLITE_CANTOPEN,
            sqlite3.SQLITE_PERM,
        }
    )


class ProductionBaselineError(RuntimeError):
    """The build cannot prove its discovered source revisions were retained."""


class ProductionBaselineReadUnavailableError(ProductionBaselineError):
    """Every unresolved source fault is a read observation worth retrying."""


class ProductionBaselineObservationCancelledError(Exception):
    """A superseded read-only source observation stopped cooperatively."""


def _check_observation_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ProductionBaselineObservationCancelledError


@dataclass(frozen=True, slots=True)
class SourceDecision:
    source: str
    path: str
    disposition: str
    reason: str
    revision: str | None = None
    source_index: int | None = None
    material_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ProductionSourceBaseline:
    operation_id: str
    source_signature: str
    decisions: tuple[SourceDecision, ...]
    digest: str

    @property
    def accepted(self) -> tuple[SourceDecision, ...]:
        return tuple(row for row in self.decisions if row.disposition == "accepted")

    @property
    def prospective_material_bytes(self) -> int | None:
        sizes = [row.material_bytes for row in self.accepted]
        return None if any(size is None for size in sizes) else sum(size for size in sizes if size is not None)

    def prospective_retained_allocation_bytes(self, block_bytes: int) -> int | None:
        """Conservatively charge each accepted payload as a separate retained blob."""
        if block_bytes <= 0:
            raise ValueError("allocation block size must be positive")
        if self.prospective_material_bytes is None:
            return None
        return sum(
            ((row.material_bytes + block_bytes - 1) // block_bytes) * block_bytes
            for row in self.accepted
            if row.material_bytes is not None
        )

    def prospective_source_db_allocation_bytes(self, block_bytes: int) -> int:
        """Estimate one metadata allocation per accepted raw revision."""
        if block_bytes <= 0:
            raise ValueError("allocation block size must be positive")
        return sum(
            ((max(block_bytes, len(row.path.encode("utf-8")) + 512) + block_bytes - 1) // block_bytes) * block_bytes
            for row in self.accepted
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "polylogue.production-source-baseline.v2",
            "material_byte_definition": MATERIAL_BYTE_DEFINITION,
            "prospective_material_bytes": self.prospective_material_bytes,
            "operation_id": self.operation_id,
            "source_signature": self.source_signature,
            "decisions": [
                {
                    "source": row.source,
                    "path": row.path,
                    "disposition": row.disposition,
                    "reason": row.reason,
                    "revision": row.revision,
                    "source_index": row.source_index,
                    "material_bytes": row.material_bytes,
                }
                for row in self.decisions
            ],
            "digest": self.digest,
        }

    def verify(self, source_db: Path) -> None:
        self.verify_integrity()
        faults = [row for row in self.decisions if row.disposition == "fault"]
        if faults:
            if all(row.reason.startswith("revision_io_unavailable:") for row in faults):
                raise ProductionBaselineReadUnavailableError(
                    f"production source baseline has {len(faults)} unreadable revision(s)"
                )
            raise ProductionBaselineError(f"production source baseline has {len(faults)} unresolved discovery fault(s)")
        missing = unretained_source_decisions(self, source_db)
        if missing:
            raise ProductionBaselineError(
                f"production source baseline has {len(missing)} unretained revision(s): {[row.path for row in missing[:3]]}"
            )

    def verify_integrity(self) -> None:
        payload = self.as_dict()
        digest = payload.pop("digest")
        if hashlib.sha256(_json(payload)).hexdigest() != digest:
            raise ProductionBaselineError("production source baseline integrity failed")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProductionSourceBaseline:
        try:
            if payload["schema"] != "polylogue.production-source-baseline.v2":
                raise ValueError("schema")
            if payload["material_byte_definition"] != MATERIAL_BYTE_DEFINITION:
                raise ValueError("material_byte_definition")
            raw_decisions = payload["decisions"]
            if not isinstance(raw_decisions, list):
                raise ValueError("decisions")
            decisions = tuple(
                SourceDecision(
                    source=str(row["source"]),
                    path=str(row["path"]),
                    disposition=str(row["disposition"]),
                    reason=str(row["reason"]),
                    revision=None if row["revision"] is None else str(row["revision"]),
                    source_index=None if row.get("source_index") is None else int(row["source_index"]),
                    material_bytes=None if row.get("material_bytes") is None else int(row["material_bytes"]),
                )
                for row in raw_decisions
                if isinstance(row, dict)
            )
            if len(decisions) != len(raw_decisions):
                raise ValueError("decisions")
            if any(
                row.material_bytes is not None and (row.disposition != "accepted" or row.material_bytes < 0)
                for row in decisions
            ):
                raise ValueError("material_bytes")
            result = cls(
                operation_id=str(payload["operation_id"]),
                source_signature=str(payload["source_signature"]),
                decisions=decisions,
                digest=str(payload["digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionBaselineError("invalid pending production source baseline") from exc
        result.verify_integrity()
        return result


def unretained_source_decisions(baseline: ProductionSourceBaseline, source_db: Path) -> tuple[SourceDecision, ...]:
    """Accepted coordinates absent from the durable raw-session ledger."""
    conn = open_readonly_connection(
        source_db, tier=ArchiveTier.SOURCE, validate_schema=False, timeout_class="background-read"
    )
    try:
        retained = {
            (
                str(path),
                int(source_index),
                bytes(blob_hash).hex() if isinstance(blob_hash, (bytes, memoryview)) else str(blob_hash),
            )
            for path, source_index, blob_hash in conn.execute(
                "SELECT source_path, source_index, blob_hash FROM raw_sessions"
            )
        }
    finally:
        conn.close()
    return tuple(row for row in baseline.accepted if (row.path, row.source_index or 0, row.revision) not in retained)


def unretained_source_material(
    baseline: ProductionSourceBaseline, source_db: Path, blob_block_bytes: int, source_db_block_bytes: int
) -> tuple[int, int, int]:
    """Headroom for accepted revisions that have not already been retained."""
    if blob_block_bytes <= 0 or source_db_block_bytes <= 0:
        raise ValueError("allocation block size must be positive")
    rows = unretained_source_decisions(baseline, source_db)
    if any(row.material_bytes is None for row in rows):
        raise ProductionBaselineError("accepted production material has unknown retained byte size")
    material = sum(row.material_bytes for row in rows if row.material_bytes is not None)
    blobs = sum(
        ((row.material_bytes + blob_block_bytes - 1) // blob_block_bytes) * blob_block_bytes
        for row in rows
        if row.material_bytes is not None
    )
    source_rows = sum(
        (
            (max(source_db_block_bytes, len(row.path.encode("utf-8")) + 512) + source_db_block_bytes - 1)
            // source_db_block_bytes
        )
        * source_db_block_bytes
        for row in rows
    )
    return material, blobs, source_rows


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _seal(operation_id: str, source_signature: str, decisions: tuple[SourceDecision, ...]) -> ProductionSourceBaseline:
    rows = tuple(
        sorted(
            decisions,
            key=lambda row: (row.source, row.path, row.disposition, row.source_index or 0, row.revision or ""),
        )
    )
    provisional = ProductionSourceBaseline(operation_id, source_signature, rows, "")
    payload = provisional.as_dict()
    payload.pop("digest")
    return ProductionSourceBaseline(operation_id, source_signature, rows, hashlib.sha256(_json(payload)).hexdigest())


def load_pending_production_baseline(archive_root: Path) -> ProductionSourceBaseline | None:
    with existing_maintenance_receipt_directory(archive_root, _PENDING_DIR) as directory_fd:
        if directory_fd is None:
            return None
        raw = read_optional_receipt(directory_fd, _PENDING_FILE)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionBaselineError("pending production source baseline is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ProductionBaselineError("pending production source baseline is not a JSON object")
    return ProductionSourceBaseline.from_dict(payload)


def publish_pending_production_baseline(archive_root: Path, baseline: ProductionSourceBaseline) -> None:
    baseline.verify_integrity()
    (archive_root / MAINTENANCE_STATE_DIRNAME).mkdir(mode=0o700, exist_ok=True)
    with maintenance_receipt_directory(archive_root, _PENDING_DIR) as directory_fd:
        atomic_replace_receipt(directory_fd, _PENDING_FILE, _json(baseline.as_dict()))


def clear_pending_production_baseline(
    archive_root: Path, baseline: ProductionSourceBaseline, *, allow_missing: bool = False
) -> None:
    with existing_maintenance_receipt_directory(archive_root, _PENDING_DIR) as directory_fd:
        if directory_fd is None:
            raise ProductionBaselineError("pending production source baseline disappeared before promotion")
        raw = read_optional_receipt(directory_fd, _PENDING_FILE)
        if raw is None and allow_missing:
            # An earlier unlink may have succeeded before its directory fsync
            # failed. Re-establish that durability boundary on reconciliation.
            os.fsync(directory_fd)
            return
        if raw is None or ProductionSourceBaseline.from_dict(json.loads(raw)).digest != baseline.digest:
            raise ProductionBaselineError("pending production source baseline changed before promotion")
        os.unlink(_PENDING_FILE, dir_fd=directory_fd)
        os.fsync(directory_fd)


def merge_pending_production_baseline(
    current: ProductionSourceBaseline, previous: ProductionSourceBaseline | None
) -> ProductionSourceBaseline:
    if previous is None:
        return current
    previous.verify_integrity()
    current_by_coordinate = {(row.source, row.path): row for row in current.decisions}
    rows = list(current.decisions)
    keys = {(row.source, row.path, row.disposition, row.source_index, row.revision) for row in rows}
    intake_excluded = {
        (row.source, row.path)
        for row in current.decisions
        if row.disposition == "excluded" and row.reason.startswith("intake_excluded:")
    }
    for row in previous.decisions:
        if row.disposition == "accepted":
            if (row.source, row.path) in intake_excluded and _unchanged_revision(row):
                # Intake never retains these exact bytes, so an earlier
                # observation that accepted them is not a revision promotion
                # can demand. A revision the path no longer holds stays
                # demanded: the file may have been rewritten after a valid
                # session was observed.
                continue
            key = (row.source, row.path, row.disposition, row.source_index, row.revision)
            if key not in keys:
                rows.append(row)
                keys.add(key)
        elif row.disposition == "fault":
            replacement = current_by_coordinate.get((row.source, row.path))
            resolved = replacement is not None and replacement.disposition in {"accepted", "alias"}
            if replacement is not None and replacement.reason in {"available_root", "expanded_to_members"}:
                resolved = True
            if not resolved:
                key = (row.source, row.path, row.disposition, row.source_index, row.revision)
                if key not in keys:
                    rows.append(row)
                    keys.add(key)
    return _seal(current.operation_id, current.source_signature, tuple(rows))


BaselineProgress = Callable[..., None]
"""``progress(phase, *, inspected=0, revisions=0, hashed_bytes=0)``: cheap counters, no I/O."""


def _probe_sqlite_readable(path: Path) -> None:
    """Raise the read fault of a database the baseline is about to exclude, if any."""
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    finally:
        conn.close()


def _unchanged_revision(row: SourceDecision) -> bool:
    """Whether an earlier accepted revision is still the path's current content."""
    try:
        return _revision(Path(row.path))[0] == row.revision
    except (OSError, sqlite3.Error, ValueError):
        return False


def _revision(path: Path, *, cancelled: Callable[[], bool] | None = None) -> tuple[str, int]:
    _check_observation_cancelled(cancelled)
    if is_sqlite_path(path):
        return sqlite_member_revision_and_size(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _check_observation_cancelled(cancelled)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _archive_members(
    path: Path,
    source_name: str,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: BaselineProgress | None = None,
) -> tuple[SourceDecision, ...]:
    members: list[SourceDecision] = []
    with zipfile.ZipFile(path) as archive:
        central_directory = archive.infolist()
        ordinals = {id(info): ordinal for ordinal, info in enumerate(central_directory)}
        detection_entries = [
            info
            for info in ZipAdmission(zip_path=path).filter_entries(
                central_directory, allowed_suffixes=ZIP_JSON_SUFFIXES
            )
            if provider_detection_path(info.filename)
        ]
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        if provider is Provider.UNKNOWN:
            provider = sniff_zip_provider(archive, detection_entries) or provider
        allowed_path = is_declared_artifact_path if provider is Provider.UNKNOWN else None

        def excluded(info: zipfile.ZipInfo, reason: str) -> None:
            members.append(SourceDecision(source_name, f"{path}:{info.filename}", "excluded", reason))

        def fault(info: zipfile.ZipInfo, reason: str) -> None:
            members.append(SourceDecision(source_name, f"{path}:{info.filename}", "fault", reason))

        entries = ZipEntryValidator(provider, cursor_state=None, zip_path=path).filter_entries(
            central_directory,
            allowed_path=allowed_path,
            on_rejected=fault,
            on_unselected=excluded,
        )
        for info in entries:
            _check_observation_cancelled(cancelled)
            if info.file_size == 0:
                excluded(info, "empty_member")
                continue
            try:
                entry_provider = provider
                if provider is Provider.UNKNOWN:
                    entry_provider = declared_artifact_provider(info.filename) or provider
                context = ZipEntryReadContext(
                    Source(name=source_name, path=path.parent),
                    path,
                    info,
                    None,
                    entry_provider,
                    None,  # type: ignore[arg-type]
                )
                for payload in replay_zip_entry_acquisition_payloads(archive, context):
                    _check_observation_cancelled(cancelled)
                    split = payload.source_index or 0
                    members.append(
                        SourceDecision(
                            source_name,
                            f"{path}:{info.filename}",
                            "accepted",
                            "archive_member",
                            hashlib.sha256(payload.payload_bytes).hexdigest(),
                            zip_member_source_index(entry_ordinal=ordinals[id(info)], split_index=split),
                            len(payload.payload_bytes),
                        )
                    )
                    if progress is not None:
                        progress("baseline_hash", revisions=1, hashed_bytes=len(payload.payload_bytes))
            except (OSError, UnicodeError, ValueError, ZipBombError, zipfile.BadZipFile) as exc:
                reason = "revision_io_unavailable" if _retryable_read_fault(exc) else "archive_member_unreadable"
                fault(info, f"{reason}:{exc}")
                continue
    return tuple(members)


def capture_production_source_baseline(
    sources: tuple[WatchSource, ...],
    *,
    operation_id: str,
    cancelled: Callable[[], bool] | None = None,
    progress: BaselineProgress | None = None,
) -> ProductionSourceBaseline:
    """Observe the exact typed sources through the same walker as file intake.

    This call runs before any intake cursor filtering. The baseline is immutable;
    files that arrive after this observation are outside this build's denominator.
    ``progress`` receives walk and hashing counts as they happen, so a caller
    can report this pre-intake interval without re-walking anything.
    """

    def inspected() -> None:
        _check_observation_cancelled(cancelled)
        if progress is not None:
            progress("baseline_walk", inspected=1)

    if not sources:
        raise ProductionBaselineError("cold build has no effective watch sources")
    signature = hashlib.sha256(
        _json(
            [
                (
                    source.name,
                    str(source.root),
                    source.suffixes,
                    sorted(source.ignored_dir_names),
                    source.source_id,
                    source.role,
                    source.allow_path_scoped_artifacts,
                    source.required,
                )
                for source in sources
            ]
        )
    ).hexdigest()
    decisions: list[SourceDecision] = []
    observed: list[tuple[str, Path, str, str]] = []
    for source in sources:
        _check_observation_cancelled(cancelled)
        if not source.root.is_dir():
            decisions.append(
                SourceDecision(
                    source.name,
                    str(source.root),
                    "fault" if source.required else "excluded",
                    "absent_root",
                )
            )
            continue
        decisions.append(SourceDecision(source.name, str(source.root), "excluded", "available_root"))

        def record(path: Path, disposition: str, reason: str, source_name: str = source.name) -> None:
            observed.append((source_name, path, disposition, reason))

        try:
            for _ in _source_path_steps(
                source,
                sources,
                after=None,
                on_disposition=record,
                on_inspected=inspected,
            ):
                _check_observation_cancelled(cancelled)
        except WalkRefusedError as exc:
            cause = exc.__cause__
            reason = (
                f"revision_io_unavailable:{exc}"
                if isinstance(cause, Exception) and _retryable_read_fault(cause)
                else str(exc)
            )
            decisions.append(SourceDecision(source.name, str(source.root), "fault", reason))
            continue
    accepted_real = {
        str(path.resolve())
        for _, path, disposition, _ in observed
        if disposition == "accepted" and not path.is_symlink()
    }
    independent_roots = {
        (source.name, source.root.resolve())
        for source in sources
        if source.root.is_dir() and not source.root.is_symlink()
    }
    for source_name, path, disposition, reason in observed:
        _check_observation_cancelled(cancelled)
        if path.is_symlink():
            target = str(path.resolve())
            independently_accepted = (
                target in accepted_real
                or any(candidate.startswith(target + os.sep) for candidate in accepted_real)
                or (
                    path.is_dir()
                    and any(
                        other_name != source_name and Path(target).is_relative_to(root)
                        for other_name, root in independent_roots
                    )
                )
            )
            if disposition in {"accepted", "alias"} or reason in {
                "escaping_symlink",
                "broken_symlink",
                "symlink_cycle",
            }:
                if independently_accepted:
                    disposition, reason = "alias", "independently_accepted_target"
                else:
                    disposition, reason = "fault", "alias_target_not_independently_accepted"
        if disposition == "accepted":
            # Enter the hash phase before reading: one large file or ZIP can
            # take long enough that status must not still say ``baseline_walk``.
            if progress is not None:
                progress("baseline_hash")
            try:
                if path.suffix.lower() == ".zip":
                    members = _archive_members(path, source_name, cancelled=cancelled, progress=progress)
                    decisions.append(SourceDecision(source_name, str(path), "excluded", "expanded_to_members"))
                    decisions.extend(members)
                    continue
                # Intake's own pre-acquisition decision: a file it excludes
                # with a typed reason is never retained, so the baseline
                # records that exclusion instead of requiring a raw row. A
                # cold build writes derived tiers, so the ordinary route (not
                # the source-only acquisition route) is the one it runs.
                admission = classify_pre_acquisition(
                    path,
                    fallback_provider=Provider.from_string(
                        canonical_acquisition_provider(source_name, source_name=source_name)
                    ),
                    source_only=False,
                    size_bytes=path.stat().st_size,
                    checkpoint=lambda: _check_observation_cancelled(cancelled),
                )
                if admission.excluded_reason is not None:
                    if is_sqlite_path(path):
                        # A structural recognizer reads an unreadable database
                        # as "not ours". A read fault stays a retryable fault,
                        # not a terminal exclusion that would drop a valid
                        # database from the promotion demand.
                        _probe_sqlite_readable(path)
                    decisions.append(
                        SourceDecision(
                            source_name, str(path), "excluded", f"intake_excluded:{admission.excluded_reason}"
                        )
                    )
                    continue
                revision, material_bytes = _revision(path, cancelled=cancelled)
                if progress is not None:
                    progress("baseline_hash", revisions=1, hashed_bytes=material_bytes)
            except (OSError, sqlite3.Error, ValueError, zipfile.BadZipFile) as exc:
                reason = "revision_io_unavailable" if _retryable_read_fault(exc) else "revision_unreadable"
                decisions.append(SourceDecision(source_name, str(path), "fault", f"{reason}:{exc}"))
                continue
        else:
            revision = None
            material_bytes = None
        decisions.append(
            SourceDecision(source_name, str(path), disposition, reason, revision, material_bytes=material_bytes)
        )
    return _seal(operation_id, signature, tuple(decisions))
