"""Build-specific source denominator from the daemon's production discovery route."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zipfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from polylogue.config import Source
from polylogue.core.content_identity import ContentIdentityRefusal
from polylogue.core.enums import Provider
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.raw_coordinates import zip_member_source_coordinate, zip_member_source_index
from polylogue.maintenance.receipt_fs import (
    atomic_replace_receipt,
    existing_maintenance_receipt_directory,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.sources.acquisition_boundary import open_bound_container, open_bound_path
from polylogue.sources.decoder_zip import ZipEntryValidator
from polylogue.sources.dispatch import ForeignOriginContentError, bound_location_provider
from polylogue.sources.live.batch_support import (
    RetryableSourceReadError,
    classify_pre_acquisition,
    foreign_origin_exclusion,
    retryable_read_fault,
)
from polylogue.sources.live.discovery import _source_path_steps
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    replay_zip_entry_acquisition_revisions,
    zip_acquisition_fingerprint,
    zip_member_admission,
)
from polylogue.sources.source_staging import SourceInputBinding, bind_source_input
from polylogue.sources.sqlite_snapshot import is_sqlite_path, sqlite_member_revision_and_size
from polylogue.sources.walk_faults import WalkRefusedError
from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_PENDING_DIR = "production-source-baseline"
_PENDING_FILE = "pending.json"
MATERIAL_BYTE_DEFINITION = "retained-canonical-payload-v1"


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
    """Accepted revisions whose exact bytes the durable raw-session ledger does not hold.

    A revision is retained by a raw row with its exact coordinate and hash,
    or, for a file that kept growing after it was baselined, by retained
    bytes that reproduce it as a prefix (see :func:`_retained_as_prefix`).
    """
    conn = open_readonly_connection(
        source_db, tier=ArchiveTier.SOURCE, validate_schema=False, timeout_class="background-read"
    )
    try:
        retained = {
            (str(path), int(source_index), _hex(blob_hash))
            for path, source_index, blob_hash in conn.execute(
                "SELECT source_path, source_index, blob_hash FROM raw_sessions"
            )
        }
        blobs = BlobStore(source_db.parent / "blob")
        return tuple(
            row
            for row in baseline.accepted
            if (row.path, row.source_index or 0, row.revision) not in retained
            and not _retained_as_prefix(conn, blobs, row)
        )
    finally:
        conn.close()


def _hex(blob_hash: object) -> str:
    return bytes(blob_hash).hex() if isinstance(blob_hash, (bytes, memoryview)) else str(blob_hash)


@dataclass(frozen=True, slots=True)
class _RetainedPathRow:
    raw_id: str
    source_index: int
    blob_hash: str
    blob_size: int
    revision_kind: str
    revision_authority: str
    source_revision: str | None
    predecessor_raw_id: str | None
    predecessor_source_revision: str | None
    baseline_raw_id: str | None
    append_start_offset: int | None
    append_end_offset: int | None


def _retained_as_prefix(conn: sqlite3.Connection, blobs: BlobStore, row: SourceDecision) -> bool:
    """Whether the path's retained bytes reproduce ``row``'s revision as a prefix.

    A live file can grow between the baseline hashing it and intake
    retaining it. Intake then holds a larger whole-file capture, or an
    earlier capture followed by byte-proven append tails, and no raw row
    carries the baselined whole-file hash. The revision is still retained
    when those bytes, read in their recorded order, begin with exactly the
    baselined bytes: the first ``material_bytes`` of a whole-file capture
    or of a contiguous append chain rooted at one are streamed and hashed.
    Row metadata alone never proves retention; a missing blob, a short
    read, a gap, an unproven tail or different bytes leaves the revision
    unretained. A database export and a ZIP member are whole units, never
    prefixes.
    """
    if (
        row.revision is None
        or row.material_bytes is None
        or (row.source_index or 0) != 0
        or row.reason == "archive_member"
        or is_sqlite_path(Path(row.path))
    ):
        return False
    size = row.material_bytes
    rows = {
        str(raw_id): _RetainedPathRow(
            raw_id=str(raw_id),
            source_index=int(source_index),
            blob_hash=_hex(blob_hash),
            blob_size=int(blob_size),
            revision_kind=str(revision_kind),
            revision_authority=str(revision_authority),
            source_revision=None if source_revision is None else str(source_revision),
            predecessor_raw_id=None if predecessor_raw_id is None else str(predecessor_raw_id),
            predecessor_source_revision=(
                None if predecessor_source_revision is None else str(predecessor_source_revision)
            ),
            baseline_raw_id=None if baseline_raw_id is None else str(baseline_raw_id),
            append_start_offset=None if append_start_offset is None else int(append_start_offset),
            append_end_offset=None if append_end_offset is None else int(append_end_offset),
        )
        for (
            raw_id,
            source_index,
            blob_hash,
            blob_size,
            revision_kind,
            revision_authority,
            source_revision,
            predecessor_raw_id,
            predecessor_source_revision,
            baseline_raw_id,
            append_start_offset,
            append_end_offset,
        ) in conn.execute(
            """
            SELECT raw_id, source_index, blob_hash, blob_size, revision_kind, revision_authority,
                   source_revision, predecessor_raw_id, predecessor_source_revision, baseline_raw_id,
                   append_start_offset, append_end_offset
            FROM raw_sessions
            WHERE source_path = ?
            ORDER BY raw_id
            """,
            (row.path,),
        )
    }
    # Covers over the same blobs read the same bytes; each is hashed once.
    blob_sequences = {tuple(member.blob_hash for member in cover) for cover in _prefix_covers(rows, size)}
    sizes = {member.blob_hash: member.blob_size for member in rows.values()}
    return any(
        _prefix_digest(blobs, tuple((blob_hash, sizes[blob_hash]) for blob_hash in sequence), size) == row.revision
        for sequence in sorted(blob_sequences, key=len)
    )


def _chain_end(member: _RetainedPathRow) -> int:
    return member.blob_size if member.revision_kind != "append" else int(member.append_end_offset or 0)


def _proven_parent(rows: Mapping[str, _RetainedPathRow], member: _RetainedPathRow) -> _RetainedPathRow | None:
    """The retained predecessor an append tail continues byte for byte, or None."""
    if (
        member.revision_authority != "byte_proven"
        or member.append_start_offset is None
        or member.append_end_offset is None
        or member.blob_size != member.append_end_offset - member.append_start_offset
        or member.predecessor_raw_id is None
    ):
        return None
    parent = rows.get(member.predecessor_raw_id)
    if (
        parent is None
        or parent.source_revision is None
        or parent.source_revision != member.predecessor_source_revision
        or _chain_end(parent) != member.append_start_offset
    ):
        return None
    return parent


def _prefix_covers(rows: Mapping[str, _RetainedPathRow], size: int) -> set[tuple[_RetainedPathRow, ...]]:
    """Every retained member sequence that holds source bytes ``[0, size)`` in order.

    A whole-file capture (``source_index`` 0) starts at byte 0. An append
    tail extends a chain only when :func:`_proven_parent` accepts its link
    and it names the chain's full capture as its baseline. A sequence ends at
    the first member that reaches ``size``, so a longer chain through the
    same members is the same proof. Each member's chain root is resolved
    once per path.
    """
    roots: dict[str, str | None] = {}

    def root_of(head: _RetainedPathRow) -> str | None:
        trail: list[_RetainedPathRow] = []
        on_trail: set[str] = set()
        member = head
        while member.raw_id not in roots:
            if member.raw_id in on_trail:
                for link in trail:
                    roots[link.raw_id] = None
                return None
            if member.revision_kind != "append":
                roots[member.raw_id] = member.raw_id if member.source_index == 0 else None
                break
            parent = _proven_parent(rows, member)
            if parent is None:
                roots[member.raw_id] = None
                break
            trail.append(member)
            on_trail.add(member.raw_id)
            member = parent
        for link in reversed(trail):
            root = roots[str(link.predecessor_raw_id)]
            chained = root is not None and rows[root].revision_kind == "full" and link.baseline_raw_id == root
            roots[link.raw_id] = root if chained else None
        return roots[head.raw_id]

    covers: set[tuple[_RetainedPathRow, ...]] = set()
    for member in rows.values():
        if _chain_end(member) < size or root_of(member) is None:
            continue
        if member.revision_kind == "append" and _chain_end(rows[str(member.predecessor_raw_id)]) >= size:
            continue
        chain = [member]
        while chain[-1].revision_kind == "append":
            chain.append(rows[str(chain[-1].predecessor_raw_id)])
        covers.add(tuple(reversed(chain)))
    return covers


def _prefix_digest(blobs: BlobStore, cover: tuple[tuple[str, int], ...], size: int) -> str | None:
    """Stream the first ``size`` bytes of ``cover``'s ``(blob_hash, blob_size)`` members.

    None when a blob is absent, short or otherwise not the retained bytes. A
    read fault a later read can clear is raised, typed retryable, rather than
    reported as a revision the archive does not hold.
    """
    digest = hashlib.sha256()
    remaining = size
    for blob_hash, blob_size in cover:
        take = min(blob_size, remaining)
        try:
            with blobs.blob_path(blob_hash).open("rb") as stream:
                while take:
                    chunk = stream.read(min(1024 * 1024, take))
                    if not chunk:
                        return None
                    digest.update(chunk)
                    take -= len(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            if retryable_read_fault(exc):
                raise ProductionBaselineReadUnavailableError(f"retained blob {blob_hash} is unreadable: {exc}") from exc
            return None
        except ValueError:
            return None
        if not remaining:
            break
    return digest.hexdigest() if not remaining else None


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


def _unchanged_revision(row: SourceDecision) -> bool:
    """Whether an earlier accepted revision is still the path's current content.

    A database's logical revision is not re-derived here: an earlier accepted
    database revision stays demanded.
    """
    if row.reason == "archive_member":
        return _unchanged_member_revision(row)
    path = Path(row.path)
    if is_sqlite_path(path):
        return False
    try:
        return _revision(path)[0] == row.revision
    except OSError:
        return False


def _unchanged_member_revision(row: SourceDecision) -> bool:
    """Whether an accepted ZIP member revision is still what the member holds.

    The member is replayed unbound, exactly as acquisition splits it, and the
    earlier revision is unchanged when one of the current payloads hashes to
    it. Any read fault keeps the earlier revision demanded.
    """
    cut = row.path.lower().find(".zip:")
    if cut == -1 or row.source_index is None:
        return False
    archive_file, member = Path(row.path[: cut + 4]), row.path[cut + 5 :]
    try:
        entry_ordinal, _split = zip_member_source_coordinate(row.source_index)
        with zipfile.ZipFile(archive_file) as archive:
            entries = archive.infolist()
            if entry_ordinal >= len(entries) or entries[entry_ordinal].filename != member:
                return False
            context = ZipEntryReadContext(
                Source(name=row.source, path=archive_file.parent),
                archive_file,
                entries[entry_ordinal],
                None,
                Provider.from_string(canonical_acquisition_provider(row.source, source_name=row.source)),
                None,  # type: ignore[arg-type]
                bound_provider=None,
            )
            return any(
                unit.revision == row.revision for unit in replay_zip_entry_acquisition_revisions(archive, context)
            )
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile, ContentIdentityRefusal):
        return False


def _revision(
    path: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
    location: Provider | None = None,
    source_binding: SourceInputBinding | None = None,
) -> tuple[str, int]:
    """Hash one file; with ``location``, through the boundary that validates the bytes it hashes.

    Validating a separate read would let a file that changes in between be
    baselined as accepted while live capture refuses its bytes.
    """
    _check_observation_cancelled(cancelled)
    if is_sqlite_path(path):
        return sqlite_member_revision_and_size(path, source_binding=source_binding)
    digest = hashlib.sha256()
    size = 0
    # The bytes hashed are read through the acquisition boundary, as live
    # capture reads them, or the baseline would accept a file whose raw
    # revision live intake refuses.
    with open_bound_path(path, location) as stream:
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
    with (
        TemporaryDirectory(prefix="polylogue-zip-baseline-") as scratch,
        bind_source_input(path) as captured,
        open_bound_container(
            BlobStore(Path(scratch)),
            captured,
            heartbeat=lambda: _check_observation_cancelled(cancelled),
        ) as physical,
        zipfile.ZipFile(physical.stream) as archive,
    ):
        central_directory = archive.infolist()
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        # The location binds, not the sniffed dominant provider: an inbox
        # archive stays unbound so each member classifies, as in live intake.
        location_binding = bound_location_provider(provider)
        admission = zip_member_admission(archive, path, central_directory, provider)

        def excluded(info: zipfile.ZipInfo, reason: str) -> None:
            members.append(SourceDecision(source_name, f"{path}:{info.filename}", "excluded", reason))

        def fault(info: zipfile.ZipInfo, reason: str) -> None:
            members.append(SourceDecision(source_name, f"{path}:{info.filename}", "fault", reason))

        validator = ZipEntryValidator(admission.provider_hint, cursor_state=None, zip_path=path)
        for entry_ordinal, info in enumerate(central_directory):
            if (
                next(
                    iter(
                        validator.filter_entries((info,), allowed_path=admission.allowed_path, on_unselected=excluded)
                    ),
                    None,
                )
                is None
            ):
                continue
            _check_observation_cancelled(cancelled)
            if info.file_size == 0:
                excluded(info, "empty_member")
                continue
            # A member is one admission unit: its accepted decisions join the
            # denominator only once every record validated.
            member_decisions: list[SourceDecision] = []
            try:
                context = ZipEntryReadContext(
                    Source(name=source_name, path=path.parent),
                    path,
                    info,
                    None,
                    admission.entry_provider_hint(archive, info),
                    None,  # type: ignore[arg-type]
                    bound_provider=location_binding,
                    captured_input_identity=captured.captured_identity,
                    container_blob_hash=physical.blob_hash,
                    decoder_fingerprint=zip_acquisition_fingerprint(provider),
                    entry_ordinal=entry_ordinal,
                )
                for unit in replay_zip_entry_acquisition_revisions(
                    archive, context, checkpoint=lambda: _check_observation_cancelled(cancelled)
                ):
                    _check_observation_cancelled(cancelled)
                    split = unit.source_index or 0
                    member_decisions.append(
                        SourceDecision(
                            source_name,
                            f"{path}:{info.filename}",
                            "accepted",
                            "archive_member",
                            unit.revision,
                            zip_member_source_index(entry_ordinal=entry_ordinal, split_index=split),
                            unit.size_bytes,
                        )
                    )
                    if progress is not None:
                        progress("baseline_hash", revisions=1, hashed_bytes=unit.size_bytes)
                members.extend(member_decisions)
            except ForeignOriginContentError as exc:
                # The live acquisition refuses this member with this reason;
                # the baseline must not expect a raw row for it, and a resumed
                # build retires an earlier acceptance of the same bytes.
                excluded(info, f"intake_excluded:{foreign_origin_exclusion(exc)}")
            except ContentIdentityRefusal as exc:
                # Raised once the member was read whole. The refused element is
                # a typed member fault; acquisition still retains its validated
                # sibling splits, so they stay demanded.
                members.extend(member_decisions)
                fault(info, f"content_identity_refused:{exc}")
                continue
            except (OSError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
                reason = "revision_io_unavailable" if retryable_read_fault(exc) else "archive_member_unreadable"
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
                    source.source_id,
                    source.role,
                    source.layout.identity(),
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
                if isinstance(cause, Exception) and retryable_read_fault(cause)
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
        with ExitStack() as stack:
            source_binding = None
            try:
                if is_sqlite_path(path) and not path.is_symlink():
                    source_binding = stack.enter_context(bind_source_input(path))
            except OSError as exc:
                reason = "revision_io_unavailable" if retryable_read_fault(exc) else "revision_unreadable"
                decisions.append(SourceDecision(source_name, str(path), "fault", f"{reason}:{exc}"))
                continue
            _check_observation_cancelled(cancelled)
            retained_path = str(source_binding.source_path) if source_binding is not None else str(path)
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
                        # A retryable read fault of a database is raised by the
                        # decision itself and stays a fault (handled below).
                        decisions.append(
                            SourceDecision(
                                source_name, retained_path, "excluded", f"intake_excluded:{admission.excluded_reason}"
                            )
                        )
                        continue
                    # A file intake retains is captured through the acquisition
                    # boundary; hashing it through the same boundary refuses
                    # exactly the files that capture refuses, with intake's reason.
                    location = bound_location_provider(
                        Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
                    )
                    try:
                        revision, material_bytes = _revision(
                            path, cancelled=cancelled, location=location, source_binding=source_binding
                        )
                    except ForeignOriginContentError as exc:
                        decisions.append(
                            SourceDecision(
                                source_name,
                                retained_path,
                                "excluded",
                                f"intake_excluded:{foreign_origin_exclusion(exc)}",
                            )
                        )
                        continue
                    if progress is not None:
                        progress("baseline_hash", revisions=1, hashed_bytes=material_bytes)
                except RetryableSourceReadError as exc:
                    decisions.append(
                        SourceDecision(source_name, retained_path, "fault", f"revision_io_unavailable:{exc.cause}")
                    )
                    continue
                except (OSError, sqlite3.Error, ValueError, zipfile.BadZipFile) as exc:
                    reason = "revision_io_unavailable" if retryable_read_fault(exc) else "revision_unreadable"
                    decisions.append(SourceDecision(source_name, retained_path, "fault", f"{reason}:{exc}"))
                    continue
            else:
                revision = None
                material_bytes = None
            decisions.append(
                SourceDecision(source_name, retained_path, disposition, reason, revision, material_bytes=material_bytes)
            )
    return _seal(operation_id, signature, tuple(decisions))
