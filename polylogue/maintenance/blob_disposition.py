"""Read-only disposition plan for the physical blob namespace.

The blob store is forensic evidence, never desired-state authority. Every
physical object therefore receives exactly one disposition proven against a
configured source:

``source_present``
    Current source material reproduces the object's content, byte-identically
    or through the owning production route's semantic equality. The object is
    redundant storage and may be removed.
``superseded_prefix``
    The object is the exact prefix of a larger retained carrier of the same
    logical source item (append lineage). It may be removed.
``restore_required``
    The object is the only verified carrier of wanted material and names an
    ordinary spool destination that current acquisition admits. Restoration
    precedes any removal.
``unresolved``
    Nothing above holds. Unresolved blocks: it is never downgraded to
    discard, and it never authorizes restoration.

A plan is acceptable only at zero unresolved members. It is immutable, bound
to the archive identity, blob namespace identity, and exact denominators it
was compiled from, and consumed by :mod:`polylogue.maintenance.
blob_disposition_apply` under a separate authorization.

This is a one-time transition planner. Its deletion trigger is the terminal
disposition receipt: once the physical namespace is accounted for, this
module and its apply sibling go with it, and only the recurring liveness,
publication, GC, and spool-admission laws remain in their owners.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import IO, Protocol

from polylogue.storage.blob_store import BlobNamespaceEntry, BlobNamespaceEntryKind, BlobStore

TOOL_VERSION = "blob-disposition-plan-v1"

_HASH_CHUNK_BYTES = 1 << 20
# An object larger than this is never one of the small JSON envelopes the
# spool provers own; probing it would read hundreds of megabytes to decide a
# question its size already answers.
_MAX_ENVELOPE_PROBE_BYTES = 256 << 20


class BlobDispositionError(RuntimeError):
    """Raised when a disposition plan cannot be compiled or trusted."""


class BlobDisposition(StrEnum):
    """The only terminal dispositions a physical blob may receive."""

    SOURCE_PRESENT = "source_present"
    SUPERSEDED_PREFIX = "superseded_prefix"
    RESTORE_REQUIRED = "restore_required"
    UNRESOLVED = "unresolved"


class SourceProofMode(StrEnum):
    """How a prover established that current source material holds the content."""

    BYTE_IDENTICAL = "byte_identical"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    STRICT_PREFIX = "strict_prefix"


class RestorationDestination(StrEnum):
    """Ordinary spool destinations current acquisition already admits."""

    HOOK_EVENT_SPOOL = "hook_event_spool"
    BROWSER_CAPTURE_SPOOL = "browser_capture_spool"


@dataclass(frozen=True, slots=True)
class SourceProof:
    """One prover's evidence that a configured source holds the content."""

    prover: str
    mode: SourceProofMode
    source_id: str
    source_path: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "prover": self.prover,
            "mode": self.mode.value,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RestorationTarget:
    """Where a sole-copy carrier is restored before its removal is considered."""

    destination: RestorationDestination
    logical_id: str

    def to_dict(self) -> dict[str, str]:
        return {"destination": self.destination.value, "logical_id": self.logical_id}


@dataclass(frozen=True, slots=True)
class BlobDispositionMember:
    """One physical blob and its single proven disposition."""

    blob_hash: str
    size_bytes: int
    referenced: bool
    disposition: BlobDisposition
    reason: str
    proof: SourceProof | None = None
    restoration: RestorationTarget | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "referenced": self.referenced,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "proof": self.proof.to_dict() if self.proof is not None else None,
            "restoration": self.restoration.to_dict() if self.restoration is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BlobDispositionDenominator:
    """The exact population a plan was compiled from."""

    physical_file_count: int
    distinct_hash_count: int
    total_bytes: int
    referenced_hash_count: int
    referenced_present_count: int
    referenced_absent_count: int
    invalid_namespace_entries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_file_count": self.physical_file_count,
            "distinct_hash_count": self.distinct_hash_count,
            "total_bytes": self.total_bytes,
            "referenced_hash_count": self.referenced_hash_count,
            "referenced_present_count": self.referenced_present_count,
            "referenced_absent_count": self.referenced_absent_count,
            "invalid_namespace_entries": list(self.invalid_namespace_entries),
        }


@dataclass(frozen=True, slots=True)
class BlobDispositionPlan:
    """An immutable, identity-bound, zero-unknown disposition plan."""

    tool_version: str
    archive_root: str
    blob_root: str
    denominator: BlobDispositionDenominator
    members: tuple[BlobDispositionMember, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = {disposition.value: 0 for disposition in BlobDisposition}
        for member in self.members:
            counts[member.disposition.value] += 1
        return counts

    @property
    def bytes_by_disposition(self) -> dict[str, int]:
        totals = {disposition.value: 0 for disposition in BlobDisposition}
        for member in self.members:
            totals[member.disposition.value] += member.size_bytes
        return totals

    @property
    def unresolved_count(self) -> int:
        return self.counts[BlobDisposition.UNRESOLVED.value]

    @property
    def accepted(self) -> bool:
        """A plan is acceptable only when nothing is unexplained."""
        return self.unresolved_count == 0 and not self.denominator.invalid_namespace_entries

    def members_for(self, disposition: BlobDisposition) -> tuple[BlobDispositionMember, ...]:
        return tuple(member for member in self.members if member.disposition is disposition)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_version": self.tool_version,
            "archive_root": self.archive_root,
            "blob_root": self.blob_root,
            "denominator": self.denominator.to_dict(),
            "counts": self.counts,
            "bytes_by_disposition": self.bytes_by_disposition,
            "unresolved_count": self.unresolved_count,
            "accepted": self.accepted,
            "read_only": True,
            "members": [member.to_dict() for member in self.members],
        }

    def digest(self) -> str:
        """Bind identity, denominators, and every exact member outcome."""
        payload = {
            "tool_version": self.tool_version,
            "archive_root": self.archive_root,
            "blob_root": self.blob_root,
            "denominator": self.denominator.to_dict(),
            "members": [member.to_dict() for member in self.members],
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BlobDispositionPlan:
        """Reload a persisted plan without re-deriving any judgment."""
        try:
            denominator = payload["denominator"]
            raw_members = payload["members"]
            if not isinstance(denominator, Mapping) or not isinstance(raw_members, list):
                raise BlobDispositionError("plan denominator and members must be structured")
            members = tuple(_member_from_dict(item) for item in raw_members)
            return cls(
                tool_version=str(payload["tool_version"]),
                archive_root=str(payload["archive_root"]),
                blob_root=str(payload["blob_root"]),
                denominator=BlobDispositionDenominator(
                    physical_file_count=int(denominator["physical_file_count"]),
                    distinct_hash_count=int(denominator["distinct_hash_count"]),
                    total_bytes=int(denominator["total_bytes"]),
                    referenced_hash_count=int(denominator["referenced_hash_count"]),
                    referenced_present_count=int(denominator["referenced_present_count"]),
                    referenced_absent_count=int(denominator["referenced_absent_count"]),
                    invalid_namespace_entries=tuple(
                        str(entry) for entry in denominator.get("invalid_namespace_entries", ())
                    ),
                ),
                members=members,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BlobDispositionError(f"unreadable disposition plan: {exc}") from exc


def _member_from_dict(payload: object) -> BlobDispositionMember:
    if not isinstance(payload, Mapping):
        raise BlobDispositionError("plan member must be an object")
    proof_payload = payload.get("proof")
    proof = None
    if isinstance(proof_payload, Mapping):
        proof = SourceProof(
            prover=str(proof_payload["prover"]),
            mode=SourceProofMode(str(proof_payload["mode"])),
            source_id=str(proof_payload["source_id"]),
            source_path=str(proof_payload["source_path"]),
            detail=str(proof_payload.get("detail", "")),
        )
    restoration_payload = payload.get("restoration")
    restoration = None
    if isinstance(restoration_payload, Mapping):
        restoration = RestorationTarget(
            destination=RestorationDestination(str(restoration_payload["destination"])),
            logical_id=str(restoration_payload["logical_id"]),
        )
    return BlobDispositionMember(
        blob_hash=str(payload["blob_hash"]),
        size_bytes=int(payload["size_bytes"]),
        referenced=bool(payload["referenced"]),
        disposition=BlobDisposition(str(payload["disposition"])),
        reason=str(payload["reason"]),
        proof=proof,
        restoration=restoration,
    )


class BlobSourceProver(Protocol):
    """Establishes that configured source material still holds a blob's content."""

    name: str

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None: ...


class BlobRestorationResolver(Protocol):
    """Names the ordinary spool destination a sole-copy carrier belongs to."""

    def restoration_target(self, path: Path) -> RestorationTarget | None: ...


def _read_envelope(path: Path, *, expected_keys: frozenset[str]) -> dict[str, object] | None:
    """Load a small JSON object envelope, refusing anything of another shape."""
    try:
        if path.stat().st_size > _MAX_ENVELOPE_PROBE_BYTES:
            return None
        with path.open("rb") as handle:
            head = handle.read(1)
            if head != b"{":
                return None
            handle.seek(0)
            value = json.load(handle)
    except (OSError, json.JSONDecodeError, RecursionError, ValueError):
        return None
    if not isinstance(value, dict) or not expected_keys.issubset(value):
        return None
    return value


class HookEventSpoolProver:
    """Prove a hook-event envelope against the declared hook spool topology.

    Acquisition stores the *validated* record, whose ``observed_at_ms`` the
    spool file does not carry, and both sides are serialized independently.
    Byte equality is therefore the wrong law here: the proof is equality of
    the production-route record, which is what admission would reproduce.
    """

    name = "hook-event-spool"
    _ENVELOPE_KEYS = frozenset({"event_id", "event_type", "session_id", "timestamp", "provider", "payload"})

    def __init__(self, sources: Sequence[tuple[str, Path]]) -> None:
        self._sources = tuple(sources)
        self._index: dict[str, tuple[str, Path]] | None = None

    def _spool_index(self) -> dict[str, tuple[str, Path]]:
        if self._index is not None:
            return self._index
        index: dict[str, tuple[str, Path]] = {}
        for source_id, root in self._sources:
            for directory, subdirectories, filenames in os.walk(root):
                subdirectories.sort()
                for filename in sorted(filenames):
                    if not filename.endswith(".json"):
                        continue
                    index.setdefault(filename[: -len(".json")], (source_id, Path(directory) / filename))
        self._index = index
        return index

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        envelope = _read_envelope(path, expected_keys=self._ENVELOPE_KEYS)
        if envelope is None:
            return None
        event_id = envelope.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return None
        located = self._spool_index().get(event_id)
        if located is None:
            return None
        source_id, spool_path = located
        from polylogue.sources.hooks import HookSpoolRecordError, read_hook_spool_record

        try:
            record = read_hook_spool_record(spool_path)
        except HookSpoolRecordError:
            return None
        if record != envelope:
            return None
        return SourceProof(
            prover=self.name,
            mode=SourceProofMode.SEMANTIC_EQUIVALENT,
            source_id=source_id,
            source_path=str(spool_path),
            detail=f"hook event {event_id} reproduces through the spool read route",
        )

    def restoration_target(self, path: Path) -> RestorationTarget | None:
        envelope = _read_envelope(path, expected_keys=self._ENVELOPE_KEYS)
        if envelope is None:
            return None
        event_id = envelope.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return None
        return RestorationTarget(destination=RestorationDestination.HOOK_EVENT_SPOOL, logical_id=event_id)


class BrowserCaptureSpoolProver:
    """Prove a browser-capture envelope against the ordinary capture spool."""

    name = "browser-capture-spool"
    _ENVELOPE_KEYS = frozenset({"polylogue_capture_kind", "schema_version", "session", "provenance"})

    def __init__(self, spool_root: Path, *, source_id: str = "browser-capture-spool") -> None:
        self._spool_root = spool_root
        self._source_id = source_id

    def _envelope(self, path: Path) -> object | None:
        payload = _read_envelope(path, expected_keys=self._ENVELOPE_KEYS)
        if payload is None:
            return None
        from pydantic import ValidationError

        from polylogue.browser_capture.models import BrowserCaptureEnvelope

        try:
            return BrowserCaptureEnvelope.model_validate(payload)
        except ValidationError:
            return None

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        envelope = self._envelope(path)
        if envelope is None:
            return None
        from polylogue.browser_capture.models import BrowserCaptureEnvelope
        from polylogue.browser_capture.receiver import capture_artifact_path, capture_dedup_content_hash

        assert isinstance(envelope, BrowserCaptureEnvelope)
        spooled = capture_artifact_path(envelope, self._spool_root)
        if not spooled.is_file():
            return None
        try:
            existing = BrowserCaptureEnvelope.model_validate_json(spooled.read_bytes())
        except (OSError, ValueError):
            return None
        if capture_dedup_content_hash(existing) != capture_dedup_content_hash(envelope):
            return None
        return SourceProof(
            prover=self.name,
            mode=SourceProofMode.SEMANTIC_EQUIVALENT,
            source_id=self._source_id,
            source_path=str(spooled),
            detail="capture spool holds a dedup-equivalent envelope",
        )

    def restoration_target(self, path: Path) -> RestorationTarget | None:
        envelope = self._envelope(path)
        if envelope is None:
            return None
        from polylogue.browser_capture.models import BrowserCaptureEnvelope

        assert isinstance(envelope, BrowserCaptureEnvelope)
        return RestorationTarget(
            destination=RestorationDestination.BROWSER_CAPTURE_SPOOL,
            logical_id=f"{envelope.session.provider}:{envelope.session.provider_session_id}",
        )


def _hash_stream(handle: IO[bytes], *, limit: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    consumed = 0
    while True:
        want = _HASH_CHUNK_BYTES if limit is None else min(_HASH_CHUNK_BYTES, limit - consumed)
        if want <= 0:
            break
        chunk = handle.read(want)
        if not chunk:
            break
        digest.update(chunk)
        consumed += len(chunk)
    return digest.hexdigest(), consumed


@dataclass(frozen=True, slots=True)
class RawSourceCarrier:
    """One acquisition's record of where a payload came from."""

    source_path: str
    append_start_offset: int | None = None


class RawSourceFileProver:
    """Prove a raw payload against the source file it was acquired from.

    Three shapes all reproduce the content and all require a fresh hash:
    the whole file, the file's own prefix (append-structured providers grow
    in place), and the recorded append span for a row that captured only its
    own increment. Path existence proves nothing.
    """

    name = "raw-source-file"

    def __init__(self, carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]]) -> None:
        self._carriers = dict(carriers_by_hash)

    def _attempt(self, source: Path, *, offset: int, size_bytes: int, whole: bool) -> tuple[str, int] | None:
        try:
            with source.open("rb") as handle:
                if offset:
                    handle.seek(offset)
                return _hash_stream(handle, limit=None if whole else size_bytes)
        except OSError:
            return None

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        for carrier in self._carriers.get(blob_hash, ()):
            source = Path(carrier.source_path)
            try:
                if not source.is_file():
                    continue
                source_size = source.stat().st_size
            except OSError:
                continue
            attempts: list[tuple[SourceProofMode, int, bool]] = []
            if source_size == size_bytes:
                attempts.append((SourceProofMode.BYTE_IDENTICAL, 0, True))
            elif source_size > size_bytes:
                attempts.append((SourceProofMode.STRICT_PREFIX, 0, False))
            offset = carrier.append_start_offset
            if offset is not None and offset > 0 and source_size >= offset + size_bytes:
                attempts.append((SourceProofMode.STRICT_PREFIX, offset, False))
            for mode, start, whole in attempts:
                measured = self._attempt(source, offset=start, size_bytes=size_bytes, whole=whole)
                if measured is None:
                    continue
                digest, consumed = measured
                if consumed != size_bytes or digest != blob_hash:
                    continue
                span = "whole file" if whole else f"{size_bytes} bytes at offset {start}"
                return SourceProof(
                    prover=self.name,
                    mode=mode,
                    source_id="configured-source-file",
                    source_path=str(source),
                    detail=f"fresh hash over the {span} of the live source",
                )
        return None


class AppendPrefixProver:
    """Prove a blob is the exact prefix of a retained carrier of the same item.

    Scoped to carriers that share a logical source identity: an unrelated
    object that merely happens to start with the same bytes is not append
    lineage, and treating it as such would discard a distinct carrier.
    """

    name = "append-prefix"

    def __init__(self, successors_by_hash: Mapping[str, tuple[str, ...]], *, blob_store: BlobStore) -> None:
        self._successors = dict(successors_by_hash)
        self._store = blob_store

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        for successor in self._successors.get(blob_hash, ()):
            successor_path = self._store.blob_path(successor)
            try:
                if not successor_path.is_file() or successor_path.stat().st_size <= size_bytes:
                    continue
                with successor_path.open("rb") as handle:
                    digest, consumed = _hash_stream(handle, limit=size_bytes)
            except OSError:
                continue
            if consumed != size_bytes or digest != blob_hash:
                continue
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.STRICT_PREFIX,
                source_id="retained-blob",
                source_path=successor,
                detail="exact prefix of a larger retained carrier of the same logical item",
            )
        return None


@dataclass(frozen=True, slots=True)
class BlobDispositionContext:
    """Everything a compilation needs, resolved once and reused per member."""

    blob_store: BlobStore
    provers: tuple[BlobSourceProver, ...]
    referenced_hashes: frozenset[str]
    restoration_provers: tuple[BlobRestorationResolver, ...] = field(default=())


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def referenced_blob_hashes(source_db: Path) -> frozenset[str]:
    """Union every durable relation that names a physical blob hash.

    A relation that exists but cannot be read is a failure, never an empty
    set: reading zero references from an unreadable tier would license
    deleting the whole namespace.
    """
    relations = (
        ("blob_refs", "blob_hash"),
        ("raw_sessions", "blob_hash"),
        ("raw_hook_events", "blob_hash"),
        ("raw_artifacts", "blob_hash"),
        ("blob_publication_reservations", "blob_hash"),
    )
    hashes: set[str] = set()
    with closing(_open_ro(source_db)) as conn:
        present = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
        }
        for table, column in relations:
            if table not in present:
                continue
            try:
                columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.Error as exc:
                raise BlobDispositionError(f"reference relation {table} is unreadable: {exc}") from exc
            if column not in columns:
                continue
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT lower(hex({column})) FROM {table} WHERE {column} IS NOT NULL"
                ).fetchall()
            except sqlite3.Error as exc:
                raise BlobDispositionError(f"reference relation {table} is unreadable: {exc}") from exc
            hashes.update(str(row[0]) for row in rows)
    return frozenset(hashes)


def raw_source_carriers_by_hash(source_db: Path) -> dict[str, tuple[RawSourceCarrier, ...]]:
    """Map each acquired payload hash to the source carriers that produced it."""
    mapping: dict[str, set[RawSourceCarrier]] = {}
    with closing(_open_ro(source_db)) as conn:
        try:
            rows = conn.execute(
                "SELECT lower(hex(blob_hash)), source_path, append_start_offset FROM raw_sessions "
                "WHERE blob_hash IS NOT NULL AND source_path IS NOT NULL"
            ).fetchall()
        except sqlite3.Error as exc:
            raise BlobDispositionError(f"raw_sessions is unreadable: {exc}") from exc
    for blob_hash, source_path, offset in rows:
        carrier = RawSourceCarrier(str(source_path), int(offset) if offset is not None else None)
        mapping.setdefault(str(blob_hash), set()).add(carrier)
    return {
        key: tuple(sorted(value, key=lambda item: (item.source_path, item.append_start_offset or 0)))
        for key, value in mapping.items()
    }


def append_successors_by_hash(source_db: Path) -> dict[str, tuple[str, ...]]:
    """Map each payload hash to larger carriers of the same logical item."""
    with closing(_open_ro(source_db)) as conn:
        try:
            rows = conn.execute(
                "SELECT origin, native_id, lower(hex(blob_hash)), blob_size FROM raw_sessions "
                "WHERE blob_hash IS NOT NULL AND native_id IS NOT NULL"
            ).fetchall()
        except sqlite3.Error as exc:
            raise BlobDispositionError(f"raw_sessions is unreadable: {exc}") from exc
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for origin, native_id, blob_hash, size in rows:
        if size is None:
            continue
        grouped.setdefault((str(origin), str(native_id)), []).append((int(size), str(blob_hash)))
    successors: dict[str, tuple[str, ...]] = {}
    for carriers in grouped.values():
        carriers.sort()
        for index, (size, blob_hash) in enumerate(carriers):
            larger = tuple(other for other_size, other in carriers[index + 1 :] if other_size > size)
            if larger:
                successors[blob_hash] = larger
    return successors


def _restoration_target(path: Path, provers: Sequence[BlobRestorationResolver]) -> RestorationTarget | None:
    for prover in provers:
        target = prover.restoration_target(path)
        if target is not None:
            return target
    return None


def classify_blob(
    entry: BlobNamespaceEntry,
    *,
    context: BlobDispositionContext,
) -> BlobDispositionMember:
    """Assign exactly one disposition to one physical blob."""
    assert entry.hash_hex is not None
    blob_hash = entry.hash_hex
    try:
        size_bytes = entry.path.stat().st_size
    except OSError as exc:
        return BlobDispositionMember(
            blob_hash=blob_hash,
            size_bytes=0,
            referenced=blob_hash in context.referenced_hashes,
            disposition=BlobDisposition.UNRESOLVED,
            reason=f"physical object is unreadable: {exc}",
        )
    referenced = blob_hash in context.referenced_hashes
    for prover in context.provers:
        proof = prover.prove(blob_hash, entry.path, size_bytes)
        if proof is None:
            continue
        disposition = (
            BlobDisposition.SUPERSEDED_PREFIX
            if proof.prover == AppendPrefixProver.name
            else BlobDisposition.SOURCE_PRESENT
        )
        return BlobDispositionMember(
            blob_hash=blob_hash,
            size_bytes=size_bytes,
            referenced=referenced,
            disposition=disposition,
            reason=f"{proof.prover} proved {proof.mode.value}",
            proof=proof,
        )
    restoration = _restoration_target(entry.path, context.restoration_provers)
    if restoration is not None:
        return BlobDispositionMember(
            blob_hash=blob_hash,
            size_bytes=size_bytes,
            referenced=referenced,
            disposition=BlobDisposition.RESTORE_REQUIRED,
            reason="no configured source holds this content and it names an ordinary spool destination",
            restoration=restoration,
        )
    return BlobDispositionMember(
        blob_hash=blob_hash,
        size_bytes=size_bytes,
        referenced=referenced,
        disposition=BlobDisposition.UNRESOLVED,
        reason="no source proof and no ordinary restoration destination",
    )


def resolve_disposition_roots(archive_root: Path) -> tuple[Path, tuple[tuple[str, Path], ...], Path]:
    """Resolve the primary hook spool, the declared spool topology, and captures.

    The declared topology already includes the legacy read-only roots, so a
    carrier whose event still sits in a superseded spool is proven at a
    configured source rather than restored a second time.
    """
    from polylogue.sources.hooks import hook_spool_sources

    hooks_root = archive_root / "hooks"
    sources = tuple(
        (spec.source_id, spec.root) for spec in hook_spool_sources(primary_root=hooks_root) if spec.root.is_dir()
    )
    return hooks_root, sources, archive_root / "browser-capture"


def build_disposition_context(
    *,
    archive_root: Path,
    blob_root: Path,
    source_db: Path,
    hook_spool_sources: Sequence[tuple[str, Path]],
    browser_capture_spool: Path,
) -> BlobDispositionContext:
    """Resolve the prover set from configured sources, not from history."""
    store = BlobStore(blob_root)
    hook_prover = HookEventSpoolProver(hook_spool_sources)
    capture_prover = BrowserCaptureSpoolProver(browser_capture_spool)
    provers: tuple[BlobSourceProver, ...] = (
        hook_prover,
        capture_prover,
        RawSourceFileProver(raw_source_carriers_by_hash(source_db)),
        AppendPrefixProver(append_successors_by_hash(source_db), blob_store=store),
    )
    return BlobDispositionContext(
        blob_store=store,
        provers=provers,
        referenced_hashes=referenced_blob_hashes(source_db),
        restoration_provers=(hook_prover, capture_prover),
    )


def compile_disposition_plan(
    *,
    archive_root: Path,
    blob_root: Path,
    source_db: Path,
    context: BlobDispositionContext | None = None,
    hook_spool_sources: Sequence[tuple[str, Path]] | None = None,
    browser_capture_spool: Path | None = None,
    progress: object | None = None,
) -> BlobDispositionPlan:
    """Walk the complete physical namespace and compile one immutable plan."""
    if context is None:
        if hook_spool_sources is None or browser_capture_spool is None:
            raise BlobDispositionError("compilation needs either a context or the configured spool roots")
        context = build_disposition_context(
            archive_root=archive_root,
            blob_root=blob_root,
            source_db=source_db,
            hook_spool_sources=hook_spool_sources,
            browser_capture_spool=browser_capture_spool,
        )
    members: list[BlobDispositionMember] = []
    invalid: list[str] = []
    seen: set[str] = set()
    file_count = 0
    for entry in context.blob_store.iter_namespace():
        if entry.kind is not BlobNamespaceEntryKind.BLOB:
            invalid.append(f"{entry.relative_path}: {entry.issue.value if entry.issue else 'unclassified'}")
            continue
        file_count += 1
        assert entry.hash_hex is not None
        if entry.hash_hex in seen:
            continue
        seen.add(entry.hash_hex)
        members.append(classify_blob(entry, context=context))
        if progress is not None and len(members) % 1000 == 0:
            progress(len(members))  # type: ignore[operator]
    present = frozenset(seen)
    denominator = BlobDispositionDenominator(
        physical_file_count=file_count,
        distinct_hash_count=len(members),
        total_bytes=sum(member.size_bytes for member in members),
        referenced_hash_count=len(context.referenced_hashes),
        referenced_present_count=len(context.referenced_hashes & present),
        referenced_absent_count=len(context.referenced_hashes - present),
        invalid_namespace_entries=tuple(sorted(invalid)),
    )
    return BlobDispositionPlan(
        tool_version=TOOL_VERSION,
        archive_root=str(archive_root),
        blob_root=str(blob_root),
        denominator=denominator,
        members=tuple(sorted(members, key=lambda member: member.blob_hash)),
    )


__all__ = [
    "TOOL_VERSION",
    "AppendPrefixProver",
    "BlobDisposition",
    "BlobDispositionContext",
    "BlobDispositionDenominator",
    "BlobDispositionError",
    "BlobDispositionMember",
    "BlobDispositionPlan",
    "BlobRestorationResolver",
    "BlobSourceProver",
    "BrowserCaptureSpoolProver",
    "HookEventSpoolProver",
    "RawSourceCarrier",
    "RawSourceFileProver",
    "RestorationDestination",
    "RestorationTarget",
    "SourceProof",
    "SourceProofMode",
    "append_successors_by_hash",
    "build_disposition_context",
    "classify_blob",
    "compile_disposition_plan",
    "raw_source_carriers_by_hash",
    "resolve_disposition_roots",
    "referenced_blob_hashes",
]
