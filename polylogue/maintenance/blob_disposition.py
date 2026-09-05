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
``positively_excluded``
    The object is not product material — it entered the archive through a
    declared non-product route — so no source is expected to hold it. The
    rule that matched is recorded; a hash allowlist is not such a rule.
``explained_residue``
    The object's provenance is known and a named task owns its remaining
    question, but no configured source holds it and no ordinary spool admits
    it. It is retained, never deleted, and it does not block: what blocks is
    material nobody can explain.
``unresolved``
    Nothing above holds. Unresolved blocks: it is never downgraded to
    discard, and it never authorizes restoration.

A proof must name a location the acquisition route actually reads. A file
that merely exists on disk somewhere under a source root proves storage, not
reacquirability, and the difference decides whether deleting the object loses
it.

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

import base64
import binascii
import hashlib
import json
import mmap
import os
import re
import sqlite3
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
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
    POSITIVELY_EXCLUDED = "positively_excluded"
    EXPLAINED_RESIDUE = "explained_residue"
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
    rule: TerminalRule | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "referenced": self.referenced,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "proof": self.proof.to_dict() if self.proof is not None else None,
            "restoration": self.restoration.to_dict() if self.restoration is not None else None,
            "rule": self.rule.to_dict() if self.rule is not None else None,
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
    rule_payload = payload.get("rule")
    rule = None
    if isinstance(rule_payload, Mapping):
        rule = TerminalRule(
            rule=str(rule_payload["rule"]),
            owner=str(rule_payload["owner"]),
            reason=str(rule_payload.get("reason", "")),
        )
    return BlobDispositionMember(
        blob_hash=str(payload["blob_hash"]),
        size_bytes=int(payload["size_bytes"]),
        referenced=bool(payload["referenced"]),
        disposition=BlobDisposition(str(payload["disposition"])),
        reason=str(payload["reason"]),
        proof=proof,
        restoration=restoration,
        rule=rule,
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

    Only each declared root's ``pending`` directory counts. That is what
    ``drain_hook_event_spool`` and the live watcher read; an acknowledged
    receipt is addressed to the source tier that consumed it, so on a fresh
    archive it is re-ingested by nothing. An envelope surviving only there
    falls through to restoration.
    """

    name = "hook-event-spool"
    _ENVELOPE_KEYS = frozenset({"event_id", "event_type", "session_id", "timestamp", "provider", "payload"})

    def __init__(self, sources: Sequence[tuple[str, Path]]) -> None:
        self._sources = tuple(sources)
        self._index: dict[str, tuple[str, Path]] | None = None

    def _spool_index(self) -> dict[str, tuple[str, Path]]:
        if self._index is not None:
            return self._index
        from polylogue.sources.hooks import pending_hook_spool_dir

        index: dict[str, tuple[str, Path]] = {}
        for source_id, spool_root in self._sources:
            root = pending_hook_spool_dir(spool_root)
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


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# A needle larger than this is not an append span; searching for it would read
# the whole object into memory to answer a question its size already answers.
_MAX_SUBSTRING_NEEDLE_BYTES = 64 << 20
_MAX_SUBSTRING_CARRIER_BYTES = 4 << 30
_MAX_JSON_DOCUMENT_BYTES = 1 << 30
_SQLITE_MAGIC = b"SQLite format 3\x00"


def blob_candidate_sizes(blob_root: Path) -> frozenset[int]:
    """Every physical object size present in the namespace.

    Content indexes are bounded by this set. A string, a base64 decode, or a
    file whose length no physical object has cannot prove anything, and
    hashing it would turn a bounded pass over the sources into an unbounded
    one over every byte the operator owns.
    """
    sizes: set[int] = set()
    for directory, subdirectories, filenames in os.walk(blob_root):
        subdirectories.sort()
        for filename in filenames:
            try:
                sizes.add(os.stat(os.path.join(directory, filename)).st_size)
            except OSError:
                continue
    sizes.discard(0)
    return frozenset(sizes)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _open_sqlite_immutable(path: Path) -> sqlite3.Connection:
    """Open an arbitrary file as a database without touching it.

    ``immutable=1`` is required rather than preferred: opening a blob-store
    object as an ordinary database creates ``-wal``/``-shm`` siblings inside
    the physical namespace, which invalidates the namespace being planned.
    """
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _load_json_document(path: Path, *, limit: int = _MAX_JSON_DOCUMENT_BYTES) -> object | None:
    try:
        if path.stat().st_size > limit:
            return None
        with path.open("rb") as handle:
            document: object = json.load(handle)
        return document
    except (OSError, json.JSONDecodeError, RecursionError, ValueError):
        return None


def _b64_decoded_length(text: str) -> int | None:
    """Exact decoded length of an unwrapped base64 string, else ``None``."""
    length = len(text)
    if length < 4 or length % 4:
        return None
    padding = 2 if text.endswith("==") else 1 if text.endswith("=") else 0
    return length // 4 * 3 - padding


def _iter_json_strings(document: object) -> Iterator[tuple[str, str]]:
    """Yield every string value in a decoded document with its JSON path."""
    stack: list[tuple[object, str]] = [(document, "$")]
    while stack:
        node, where = stack.pop()
        if isinstance(node, str):
            yield node, where
        elif isinstance(node, dict):
            for key, value in node.items():
                stack.append((value, f"{where}.{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                stack.append((value, f"{where}[{index}]"))


class _EmbeddedPayloadIndex:
    """Content hashes of payloads embedded as strings inside JSON carriers.

    Both a string's raw UTF-8 and its base64 decode are candidates: carriers
    hold extracted text the first way and binary attachments the second.
    """

    def __init__(self, candidate_sizes: frozenset[int]) -> None:
        self._sizes = candidate_sizes
        self._index: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._index)

    def get(self, blob_hash: str) -> str | None:
        return self._index.get(blob_hash)

    def absorb(self, document: object, *, location: str) -> None:
        for text, trail in _iter_json_strings(document):
            raw = text.encode("utf-8")
            if len(raw) in self._sizes:
                self._index.setdefault(_sha256(raw), f"{location}!{trail}")
            decoded_length = _b64_decoded_length(text)
            if decoded_length is None or decoded_length not in self._sizes:
                continue
            try:
                decoded = base64.b64decode(text, validate=True)
            except (binascii.Error, ValueError):
                continue
            if len(decoded) != decoded_length:
                continue
            self._index.setdefault(_sha256(decoded), f"{location}!{trail}|base64")


@dataclass(frozen=True, slots=True)
class HookEventCarrier:
    """One acquired hook event and the state it was read out of."""

    source_path: str
    event_type: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class StateRowSpec:
    """How one hook event's payload maps onto a live state-database row."""

    table: str
    key_columns: tuple[str, ...]
    column_for: tuple[tuple[str, str], ...] = ()

    def column(self, payload_key: str) -> str:
        for key, column in self.column_for:
            if key == payload_key:
                return column
        return payload_key


CODEX_STATE_ROW_SPECS: Mapping[str, StateRowSpec] = {
    "codex_thread_title": StateRowSpec(table="threads", key_columns=("thread_id",), column_for=(("thread_id", "id"),)),
    "codex_thread_spawn_edge": StateRowSpec(
        table="thread_spawn_edges", key_columns=("parent_thread_id", "child_thread_id")
    ),
}


def _normalized_sql_value(value: object) -> object:
    """JSON booleans and SQLite integers denote the same stored value."""
    if isinstance(value, bool):
        return int(value)
    return value


def hook_event_carriers_by_hash(source_db: Path) -> dict[str, tuple[HookEventCarrier, ...]]:
    """Map each hook-event payload hash to the state it was acquired from."""
    mapping: dict[str, set[HookEventCarrier]] = {}
    with closing(_open_ro(source_db)) as conn:
        present = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "raw_hook_events" not in present:
            return {}
        try:
            rows = conn.execute(
                "SELECT lower(hex(blob_hash)), source_path, event_type, payload_json FROM raw_hook_events "
                "WHERE blob_hash IS NOT NULL AND source_path IS NOT NULL AND payload_json IS NOT NULL"
            ).fetchall()
        except sqlite3.Error as exc:
            raise BlobDispositionError(f"raw_hook_events is unreadable: {exc}") from exc
    for blob_hash, source_path, event_type, payload_json in rows:
        carrier = HookEventCarrier(str(source_path), str(event_type), str(payload_json))
        mapping.setdefault(str(blob_hash), set()).add(carrier)
    return {key: tuple(sorted(value, key=lambda item: item.source_path)) for key, value in mapping.items()}


class CodexStateRowProver:
    """Prove a hook event's payload against the live state-database row.

    The row is the source, not the file: these databases are rewritten in
    place, so comparing bytes against the carrier decides nothing, while the
    row the acquisition route reads is either present and equal or gone.
    """

    name = "codex-state-row"

    def __init__(
        self,
        carriers_by_hash: Mapping[str, tuple[HookEventCarrier, ...]],
        *,
        specs: Mapping[str, StateRowSpec] = CODEX_STATE_ROW_SPECS,
    ) -> None:
        self._carriers = dict(carriers_by_hash)
        self._specs = dict(specs)

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        for carrier in self._carriers.get(blob_hash, ()):
            spec = self._specs.get(carrier.event_type)
            if spec is None:
                continue
            if _sha256(carrier.payload_json.encode("utf-8")) != blob_hash:
                continue
            try:
                payload = json.loads(carrier.payload_json)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict) or not payload:
                continue
            if not all(key in payload for key in spec.key_columns):
                continue
            source = Path(carrier.source_path)
            if not source.is_file():
                continue
            keys = tuple(payload)
            try:
                with closing(_open_sqlite_immutable(source)) as connection:
                    columns = {
                        str(row[1]) for row in connection.execute(f'PRAGMA table_info("{spec.table}")').fetchall()
                    }
                    if not columns or not columns.issuperset(spec.column(key) for key in keys):
                        continue
                    selected = ", ".join(f'"{spec.column(key)}"' for key in keys)
                    predicate = " AND ".join(f'"{spec.column(key)}" = ?' for key in spec.key_columns)
                    row = connection.execute(
                        f'SELECT {selected} FROM "{spec.table}" WHERE {predicate}',
                        tuple(_normalized_sql_value(payload[key]) for key in spec.key_columns),
                    ).fetchone()
            except sqlite3.Error:
                continue
            if row is None:
                continue
            if any(
                _normalized_sql_value(payload[key]) != _normalized_sql_value(value)
                for key, value in zip(keys, row, strict=True)
            ):
                continue
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.SEMANTIC_EQUIVALENT,
                source_id="codex-state-database",
                source_path=f"{source}::{spec.table}",
                detail=f"{carrier.event_type} reproduces from the live {spec.table} row",
            )
        return None


class ExportBundleMemberProver:
    """Prove an extracted sub-object against the export bundle it came from.

    An export bundle is the acquisition route's own input, so a payload the
    archive extracted out of one is reproduced by re-reading the bundle. The
    bundle roots are declared, never discovered.
    """

    name = "export-bundle-member"

    def __init__(
        self,
        roots: Sequence[Path],
        *,
        candidate_sizes: frozenset[int],
        source_id: str = "declared-export-bundle",
    ) -> None:
        self._roots = tuple(Path(root) for root in roots)
        self._sizes = candidate_sizes
        self._source_id = source_id
        self._index: _EmbeddedPayloadIndex | None = None

    def _payloads(self) -> _EmbeddedPayloadIndex:
        if self._index is not None:
            return self._index
        index = _EmbeddedPayloadIndex(self._sizes)
        for root in self._roots:
            if not root.is_dir():
                continue
            for bundle in sorted(root.rglob("*.zip")):
                try:
                    with zipfile.ZipFile(bundle) as archive:
                        for member in archive.namelist():
                            if not member.endswith(".json"):
                                continue
                            with archive.open(member) as handle:
                                document = json.load(handle)
                            index.absorb(document, location=f"{bundle}::{member}")
                except (OSError, zipfile.BadZipFile, json.JSONDecodeError, RecursionError, ValueError):
                    continue
        self._index = index
        return index

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        located = self._payloads().get(blob_hash)
        if located is None:
            return None
        return SourceProof(
            prover=self.name,
            mode=SourceProofMode.BYTE_IDENTICAL,
            source_id=self._source_id,
            source_path=located,
            detail="byte-identical to a value an export bundle member still carries",
        )


class DataLakeFileProver:
    """Prove an object byte-identical to a file under a declared data root.

    The roots are an explicit list because the claim is "this content is at
    its source" and the plan has to say which source. A walk of the
    filesystem at large would make the claim unauditable.
    """

    name = "data-lake-file"

    def __init__(
        self,
        roots: Sequence[Path],
        *,
        candidate_sizes: frozenset[int],
        source_id: str = "declared-data-root",
    ) -> None:
        self._roots = tuple(Path(root) for root in roots)
        self._sizes = candidate_sizes
        self._source_id = source_id
        self._by_size: dict[int, list[Path]] | None = None
        self._hashes: dict[Path, str | None] = {}

    def _size_index(self) -> dict[int, list[Path]]:
        if self._by_size is not None:
            return self._by_size
        index: dict[int, list[Path]] = {}
        for root in self._roots:
            if not root.is_dir():
                continue
            for directory, subdirectories, filenames in os.walk(root):
                subdirectories.sort()
                for filename in sorted(filenames):
                    candidate = Path(directory) / filename
                    try:
                        size = candidate.stat().st_size
                    except OSError:
                        continue
                    if size in self._sizes:
                        index.setdefault(size, []).append(candidate)
        self._by_size = index
        return index

    def _digest(self, candidate: Path) -> str | None:
        if candidate in self._hashes:
            return self._hashes[candidate]
        digest: str | None
        try:
            with candidate.open("rb") as handle:
                digest, _ = _hash_stream(handle)
        except OSError:
            digest = None
        self._hashes[candidate] = digest
        return digest

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        for candidate in self._size_index().get(size_bytes, ()):
            if self._digest(candidate) != blob_hash:
                continue
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.BYTE_IDENTICAL,
                source_id=self._source_id,
                source_path=str(candidate),
                detail="fresh hash of a byte-identical file under a declared data root",
            )
        return None


def _jsonl_line_digests(path: Path) -> list[str] | None:
    try:
        with path.open("rb") as handle:
            return [_sha256(line.rstrip(b"\n")) for line in handle]
    except OSError:
        return None


def _jsonl_session_id(path: Path, *, records: int = 3) -> str | None:
    """Read the session identity the provider names in its own leading records."""
    try:
        with path.open("rb") as handle:
            for _ in range(records):
                line = handle.readline()
                if not line:
                    return None
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    return None
                if not isinstance(record, dict):
                    return None
                candidates: list[object] = [record.get("sessionId"), record.get("session_id")]
                payload = record.get("payload")
                if isinstance(payload, dict):
                    candidates.extend((payload.get("id"), payload.get("session_id")))
                for value in candidates:
                    if isinstance(value, str) and _UUID.fullmatch(value):
                        return value
    except OSError:
        return None
    return None


class JsonlLineContainmentProver:
    """Prove a session snapshot against the JSONL its provider still appends to.

    A snapshot is a moment in an append-structured file, so byte equality is
    the wrong law and line containment is the right one — with exactly one
    exception the format forces: Codex rewrites its leading ``session_meta``
    record in place, so index 0 may be absent. Nothing beyond index 0 may be,
    because a general subset rule would accept a carrier that dropped the
    snapshot's content.
    """

    name = "jsonl-line-containment"

    def __init__(
        self,
        carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]],
        *,
        session_roots: Sequence[Path],
    ) -> None:
        self._carriers = dict(carriers_by_hash)
        self._roots = tuple(Path(root) for root in session_roots)
        self._by_session: dict[str, list[Path]] | None = None

    def _session_index(self) -> dict[str, list[Path]]:
        if self._by_session is not None:
            return self._by_session
        index: dict[str, list[Path]] = {}
        for root in self._roots:
            if not root.is_dir():
                continue
            for directory, subdirectories, filenames in os.walk(root):
                subdirectories.sort()
                for filename in sorted(filenames):
                    if not filename.endswith(".jsonl"):
                        continue
                    found = _UUID.findall(filename)
                    if found:
                        index.setdefault(found[-1], []).append(Path(directory) / filename)
        self._by_session = index
        return index

    def _candidates(self, blob_hash: str, path: Path) -> list[Path]:
        candidates = [Path(carrier.source_path) for carrier in self._carriers.get(blob_hash, ())]
        session_id = _jsonl_session_id(path)
        if session_id is not None:
            candidates.extend(self._session_index().get(session_id, ()))
        return candidates

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        if _jsonl_session_id(path) is None and blob_hash not in self._carriers:
            return None
        blob_lines = _jsonl_line_digests(path)
        if not blob_lines:
            return None
        for candidate in self._candidates(blob_hash, path):
            if not candidate.is_file():
                continue
            live = _jsonl_line_digests(candidate)
            if live is None:
                continue
            available = Counter(live)
            absent: list[int] = []
            for index, digest in enumerate(blob_lines):
                if available[digest]:
                    available[digest] -= 1
                    continue
                absent.append(index)
                if absent != [0]:
                    break
            if absent and absent != [0]:
                continue
            detail = (
                "every line is present in the live carrier"
                if not absent
                else "every line but the provider-rewritten leading record is present in the live carrier"
            )
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.SEMANTIC_EQUIVALENT,
                source_id="live-session-jsonl",
                source_path=str(candidate),
                detail=detail,
            )
        return None


class TrajectoryStepPrefixProver:
    """Prove a trajectory snapshot is a strict step prefix of the live file.

    Identity fields must be equal and every retained step identical; only the
    step list may be shorter, and only by truncation. ``final_metrics`` is
    excluded because it is computed over the whole run rather than carried
    forward from the prefix.
    """

    name = "trajectory-step-prefix"
    STEP_FIELD = "steps"
    EXCLUDED_FIELDS = frozenset({"steps", "final_metrics"})

    def __init__(self, carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]]) -> None:
        self._carriers = dict(carriers_by_hash)

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        snapshot = _load_json_document(path)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get(self.STEP_FIELD), list):
            return None
        retained = snapshot[self.STEP_FIELD]
        for carrier in self._carriers.get(blob_hash, ()):
            source = Path(carrier.source_path)
            live = _load_json_document(source)
            if not isinstance(live, dict) or not isinstance(live.get(self.STEP_FIELD), list):
                continue
            if {key for key in snapshot if key not in self.EXCLUDED_FIELDS} != {
                key for key in live if key not in self.EXCLUDED_FIELDS
            }:
                continue
            if any(snapshot[key] != live[key] for key in snapshot if key not in self.EXCLUDED_FIELDS):
                continue
            live_steps = live[self.STEP_FIELD]
            if len(retained) >= len(live_steps) or retained != live_steps[: len(retained)]:
                continue
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.STRICT_PREFIX,
                source_id="live-trajectory-file",
                source_path=str(source),
                detail=f"identity fields equal and steps are the live file's first {len(retained)} of {len(live_steps)}",
            )
        return None


class ByteSpanSubstringProver:
    """Prove an append span is still a contiguous byte range of its carrier.

    A row that captured only its own increment may carry no
    ``append_start_offset``, which is the coordinate ``RawSourceFileProver``
    needs. The span itself is unchanged, so it is located by search rather
    than by a stored coordinate; anything short of an exact contiguous match
    refuses.
    """

    name = "byte-span-substring"

    def __init__(self, carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]]) -> None:
        self._carriers = dict(carriers_by_hash)

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        if not 0 < size_bytes <= _MAX_SUBSTRING_NEEDLE_BYTES:
            return None
        needle: bytes | None = None
        for carrier in self._carriers.get(blob_hash, ()):
            source = Path(carrier.source_path)
            try:
                source_size = source.stat().st_size
            except OSError:
                continue
            if not source.is_file() or not size_bytes < source_size <= _MAX_SUBSTRING_CARRIER_BYTES:
                continue
            if needle is None:
                try:
                    needle = path.read_bytes()
                except OSError:
                    return None
            try:
                with source.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
                    offset = view.find(needle)
            except (OSError, ValueError):
                continue
            if offset < 0:
                continue
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.STRICT_PREFIX,
                source_id="configured-source-file",
                source_path=str(source),
                detail=f"contiguous {size_bytes} bytes at offset {offset} of the live source",
            )
        return None


def _without_key(node: object, key: str) -> object:
    if isinstance(node, dict):
        return {name: _without_key(value, key) for name, value in node.items() if name != key}
    if isinstance(node, list):
        return [_without_key(value, key) for value in node]
    return node


class DriveCacheInjectedKeyProver:
    """Prove a Drive cache document against the same document at a declared root.

    Polylogue injects each Drive item's live bytes into its own cache file,
    so a carrier recorded before that injection differs from the current one
    by exactly that key. Stripping it is the whole permitted transformation:
    any other difference, a changed chunk included, refuses.
    """

    name = "drive-cache-injected-key"
    INJECTED_KEY = "_polylogue_drive_live_bytes_b64"

    def __init__(
        self,
        carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]],
        *,
        roots: Sequence[Path],
        injected_key: str = INJECTED_KEY,
    ) -> None:
        self._carriers = dict(carriers_by_hash)
        self._roots = tuple(Path(root) for root in roots)
        self._injected_key = injected_key

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        recorded = _load_json_document(path)
        if not isinstance(recorded, dict):
            return None
        stripped = _without_key(recorded, self._injected_key)
        for carrier in self._carriers.get(blob_hash, ()):
            basename = Path(carrier.source_path).name
            for root in self._roots:
                candidate = root / basename
                if not candidate.is_file():
                    continue
                live = _load_json_document(candidate)
                if not isinstance(live, dict):
                    continue
                if _without_key(live, self._injected_key) != stripped:
                    continue
                return SourceProof(
                    prover=self.name,
                    mode=SourceProofMode.SEMANTIC_EQUIVALENT,
                    source_id="declared-drive-cache-root",
                    source_path=str(candidate),
                    detail=f"equal once the injected {self._injected_key} key is stripped from both sides",
                )
        return None


class CaptureEmbeddedPayloadProver:
    """Prove an extracted payload against the capture document that embeds it.

    Attachment bytes and Drive documents live inside the capture and cache
    JSON the acquisition route reads, as raw text or as base64. The roots are
    declared, so a payload surviving only under a retired path does not prove.
    """

    name = "capture-embedded-payload"

    def __init__(
        self,
        roots: Sequence[Path],
        *,
        candidate_sizes: frozenset[int],
        suffixes: tuple[str, ...] = (".json", ".jsonl"),
        source_id: str = "declared-capture-root",
    ) -> None:
        self._roots = tuple(Path(root) for root in roots)
        self._sizes = candidate_sizes
        self._suffixes = suffixes
        self._source_id = source_id
        self._index: _EmbeddedPayloadIndex | None = None

    def _payloads(self) -> _EmbeddedPayloadIndex:
        if self._index is not None:
            return self._index
        index = _EmbeddedPayloadIndex(self._sizes)
        for root in self._roots:
            if not root.is_dir():
                continue
            for directory, subdirectories, filenames in os.walk(root):
                subdirectories.sort()
                for filename in sorted(filenames):
                    if not filename.endswith(self._suffixes):
                        continue
                    carrier = Path(directory) / filename
                    document = _load_json_document(carrier)
                    if document is None:
                        continue
                    index.absorb(document, location=str(carrier))
        self._index = index
        return index

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        located = self._payloads().get(blob_hash)
        if located is None:
            return None
        return SourceProof(
            prover=self.name,
            mode=SourceProofMode.BYTE_IDENTICAL,
            source_id=self._source_id,
            source_path=located,
            detail="byte-identical to a payload a live capture document embeds",
        )


class SqliteRowContainmentProver:
    """Prove a state-database snapshot's rows still exist in the live database.

    These files are rewritten in place and their schema drifts, so ``SELECT *``
    compares differently shaped tuples on the two sides and can decide
    nothing. The comparison is over the columns both sides declare. Full-text
    shadow tables are excluded because they are rebuilt from the rows they
    index; every other table must be contained, so a snapshot holding rows
    the live database dropped refuses.
    """

    name = "sqlite-row-containment"
    _SHADOW_TABLE = re.compile(r"_fts(_[a-z0-9]+)?$")

    def __init__(self, carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]]) -> None:
        self._carriers = dict(carriers_by_hash)

    @staticmethod
    def _is_sqlite(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                return handle.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
        except OSError:
            return False

    @classmethod
    def _user_tables(cls, connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [str(row[0]) for row in rows if not cls._SHADOW_TABLE.search(str(row[0]))]

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
        return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()]

    def _contained(self, snapshot: sqlite3.Connection, live: sqlite3.Connection) -> str | None:
        tables = self._user_tables(snapshot)
        if not tables:
            return None
        live_tables = set(self._user_tables(live))
        compared = 0
        for table in tables:
            if table not in live_tables:
                return None
            snapshot_columns = self._columns(snapshot, table)
            shared = [column for column in snapshot_columns if column in set(self._columns(live, table))]
            if not shared:
                return None
            projection = ", ".join(f'"{column}"' for column in shared)
            wanted = Counter(snapshot.execute(f'SELECT {projection} FROM "{table}"').fetchall())
            if not wanted:
                continue
            held = Counter(live.execute(f'SELECT {projection} FROM "{table}"').fetchall())
            for row, count in wanted.items():
                if held[row] < count:
                    return None
            compared += 1
        if not compared:
            return None
        return f"{compared} tables contained over their shared columns"

    def prove(self, blob_hash: str, path: Path, size_bytes: int) -> SourceProof | None:
        if not self._is_sqlite(path):
            return None
        for carrier in self._carriers.get(blob_hash, ()):
            source = Path(carrier.source_path)
            if not source.is_file() or not self._is_sqlite(source):
                continue
            try:
                with (
                    closing(_open_sqlite_immutable(path)) as snapshot,
                    closing(_open_sqlite_immutable(source)) as live,
                ):
                    detail = self._contained(snapshot, live)
            except sqlite3.Error:
                continue
            if detail is None:
                continue
            return SourceProof(
                prover=self.name,
                mode=SourceProofMode.SEMANTIC_EQUIVALENT,
                source_id="live-state-database",
                source_path=str(source),
                detail=detail,
            )
        return None


@dataclass(frozen=True, slots=True)
class TerminalRule:
    """A named, owned rule that ends an object's classification without a source."""

    rule: str
    owner: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "owner": self.owner, "reason": self.reason}


class BlobTerminalRuleResolver(Protocol):
    """Decides an object needs no source proof, and says under which rule."""

    disposition: BlobDisposition

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None: ...


class NonProductRouteExcluder:
    """Exclude material that entered the archive through a non-product route.

    The rule is the route, not the object: an acquisition whose declared
    source path lies under a test, fixture, or development-loop directory is
    not product material, so no source is expected to reproduce it. Matching
    by hash would record the answer instead of the reason, and would say
    nothing about the next object the same route admits.
    """

    disposition = BlobDisposition.POSITIVELY_EXCLUDED

    def __init__(
        self,
        carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]],
        *,
        markers: Sequence[str],
        owner: str,
        rule: str = "non-product-acquisition-route",
    ) -> None:
        self._carriers = dict(carriers_by_hash)
        self._markers = tuple(markers)
        self._owner = owner
        self._rule = rule

    def _matched_marker(self, source_path: str) -> str | None:
        parts = Path(source_path).parts
        for marker in self._markers:
            needle = tuple(Path(marker).parts)
            if any(parts[index : index + len(needle)] == needle for index in range(len(parts))):
                return marker
        return None

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None:
        carriers = self._carriers.get(blob_hash, ())
        if not carriers:
            return None
        matched = [self._matched_marker(carrier.source_path) for carrier in carriers]
        if not all(matched):
            return None
        return TerminalRule(
            rule=self._rule,
            owner=self._owner,
            reason=f"every acquisition of this object came from the non-product route {matched[0]!r}",
        )


class ExplainedResidueResolver:
    """Mark residue whose provenance is known and whose question has an owner.

    Explained residue is retained, never deleted, and it does not block: the
    plan blocks on material nobody can account for, which is a different
    state and has to be counted separately from material a named task is
    already carrying.
    """

    disposition = BlobDisposition.EXPLAINED_RESIDUE

    def __init__(self, hashes: Mapping[str, TerminalRule]) -> None:
        self._hashes = dict(hashes)

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None:
        return self._hashes.get(blob_hash)


class BrowserCaptureAttachmentResidue:
    """Mark a capture attachment payload no live capture document embeds.

    The payload is understood — the durable tables name it as one capture's
    attachment — but it is not itself a capture envelope, so the capture
    spool admits nothing for it and no ordinary destination exists. Its
    disposition is owned rather than unknown.
    """

    disposition = BlobDisposition.EXPLAINED_RESIDUE

    def __init__(self, payload_hashes: frozenset[str], *, owner: str, rule: str) -> None:
        self._payloads = payload_hashes
        self._owner = owner
        self._rule = rule

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None:
        if blob_hash not in self._payloads:
            return None
        return TerminalRule(
            rule=self._rule,
            owner=self._owner,
            reason="a durable row names this object as a browser-capture attachment payload, "
            "and no capture document the acquisition route reads still embeds it",
        )


def browser_capture_attachment_payloads(source_db: Path) -> frozenset[str]:
    """Hashes durable rows name as browser-capture attachment payloads.

    Both halves of the predicate carry weight: ``ref_type`` says the object
    is an attachment payload rather than the capture envelope around it, and
    the capture-spool path says which acquisition owns it. Attachments from
    any other route are a different question.
    """
    with closing(_open_ro(source_db)) as conn:
        present = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "blob_refs" not in present:
            return frozenset()
        try:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(blob_refs)").fetchall()}
            if not {"blob_hash", "ref_type", "source_path"}.issubset(columns):
                return frozenset()
            rows = conn.execute(
                "SELECT DISTINCT lower(hex(blob_hash)) FROM blob_refs "
                "WHERE blob_hash IS NOT NULL AND ref_type = 'attachment' "
                "AND source_path LIKE '%browser-capture%'"
            ).fetchall()
        except sqlite3.Error as exc:
            raise BlobDispositionError(f"blob_refs is unreadable: {exc}") from exc
    return frozenset(str(row[0]) for row in rows)


class TestCorpusFixtureExcluder:
    """Exclude synthetic material the tracked test corpus itself composes.

    These objects carry no durable row at all — no acquisition ever recorded
    a source for them — so no route predicate can reach them. What is
    observable is that every identity they name — session, message, and
    record identifiers — is spelled out as a literal in the checkout's own
    tests. Product material does not have that property: a real session's
    identifiers are provider-assigned, not constants in the test suite. The
    rule is content against a declared corpus, so it stays true for the next
    fixture the same tests leak; a list of hashes would only record the
    answer to one run.

    Comparing whole lines would not work: the fixtures are serialized from
    dictionaries the tests build at runtime, so no literal holds the line.
    """

    IDENTITY_KEYS = frozenset({"id", "uuid", "sessionId", "session_id", "parentUuid", "parent_uuid"})

    disposition = BlobDisposition.POSITIVELY_EXCLUDED

    def __init__(
        self,
        corpus_roots: Sequence[Path],
        *,
        referenced_hashes: frozenset[str],
        owner: str,
        rule: str = "tracked-test-corpus-literal",
        max_object_bytes: int = 1 << 16,
    ) -> None:
        self._roots = tuple(Path(root) for root in corpus_roots)
        self._referenced = referenced_hashes
        self._owner = owner
        self._rule = rule
        self._max_object_bytes = max_object_bytes
        self._literals: tuple[str, ...] | None = None

    @staticmethod
    def _module_literals(source: str) -> str:
        """Join a module's string and bytes literals in source order.

        Adjacent literals concatenate at parse time, which is exactly how the
        fixtures spell a multi-line payload, so joining in source order
        reproduces the payload contiguously.
        """
        import ast

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""
        parts: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if isinstance(node.value, str):
                parts.append(node.value)
            elif isinstance(node.value, bytes):
                parts.append(node.value.decode("utf-8", errors="replace"))
        return "".join(parts)

    def _corpus(self) -> tuple[str, ...]:
        if self._literals is not None:
            return self._literals
        modules: list[str] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            for directory, subdirectories, filenames in os.walk(root):
                subdirectories.sort()
                for filename in sorted(filenames):
                    if not filename.endswith(".py"):
                        continue
                    try:
                        source = (Path(directory) / filename).read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    joined = self._module_literals(source)
                    if joined:
                        modules.append(joined)
        self._literals = tuple(modules)
        return self._literals

    @classmethod
    def _identities(cls, node: object, found: set[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in cls.IDENTITY_KEYS and isinstance(value, str) and value:
                    found.add(value)
                cls._identities(value, found)
        elif isinstance(node, list):
            for value in node:
                cls._identities(value, found)

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None:
        if blob_hash in self._referenced or not 0 < size_bytes <= self._max_object_bytes:
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            return None
        identities: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(record, dict):
                return None
            self._identities(record, identities)
        if not identities:
            return None
        for module in self._corpus():
            if all(identity in module for identity in identities):
                return TerminalRule(
                    rule=self._rule,
                    owner=self._owner,
                    reason=(
                        f"unreferenced object whose {len(identities)} identifiers are all declared "
                        "as literals in one tracked test module"
                    ),
                )
        return None


class SidechainTranscriptResidue:
    """Mark a subagent transcript no provider file ever carried on its own.

    Claude Code writes sidechain records into the parent session rather than
    to a file of their own, so a standalone object of them names no carrier
    to reacquire from and no spool that admits one. It is understood
    material without a destination, which is a different fact from material
    of unknown provenance.
    """

    disposition = BlobDisposition.EXPLAINED_RESIDUE

    def __init__(
        self,
        *,
        referenced_hashes: frozenset[str],
        owner: str,
        rule: str = "standalone-sidechain-transcript",
        max_object_bytes: int = 64 << 20,
    ) -> None:
        self._referenced = referenced_hashes
        self._owner = owner
        self._rule = rule
        self._max_object_bytes = max_object_bytes

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None:
        if blob_hash in self._referenced or not 0 < size_bytes <= self._max_object_bytes:
            return None
        records = 0
        try:
            with path.open("rb") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except (json.JSONDecodeError, ValueError):
                        return None
                    if not isinstance(record, dict) or record.get("isSidechain") is not True:
                        return None
                    records += 1
        except (OSError, ValueError):
            return None
        if not records:
            return None
        return TerminalRule(
            rule=self._rule,
            owner=self._owner,
            reason=(
                f"{records} records, every one a sidechain entry, which the provider writes "
                "into its parent session rather than to a file of its own"
            ),
        )


class ForeignStateSnapshotResidue:
    """Mark a foreign application's state snapshot the live database outgrew.

    The object is understood: a declared acquisition names the state database
    it was copied from, and that database is still there but no longer holds
    these rows. Polylogue has no write path into another application's state,
    so there is no destination to restore it to. That is a different fact
    from not knowing what an object is, and it is counted differently.
    """

    disposition = BlobDisposition.EXPLAINED_RESIDUE

    def __init__(
        self,
        carriers_by_hash: Mapping[str, tuple[RawSourceCarrier, ...]],
        *,
        owner: str,
        rule: str = "foreign-state-database-snapshot",
    ) -> None:
        self._carriers = dict(carriers_by_hash)
        self._owner = owner
        self._rule = rule

    def resolve(self, blob_hash: str, path: Path, size_bytes: int) -> TerminalRule | None:
        if not SqliteRowContainmentProver._is_sqlite(path):
            return None
        for carrier in self._carriers.get(blob_hash, ()):
            source = Path(carrier.source_path)
            if source.is_file() and SqliteRowContainmentProver._is_sqlite(source):
                return TerminalRule(
                    rule=self._rule,
                    owner=self._owner,
                    reason=(
                        f"a snapshot of {source}, which is still present but no longer holds these rows; "
                        "no ordinary route writes into a foreign application's state database"
                    ),
                )
        return None


@dataclass(frozen=True, slots=True)
class BlobDispositionContext:
    """Everything a compilation needs, resolved once and reused per member."""

    blob_store: BlobStore
    provers: tuple[BlobSourceProver, ...]
    referenced_hashes: frozenset[str]
    restoration_provers: tuple[BlobRestorationResolver, ...] = field(default=())
    excluders: tuple[BlobTerminalRuleResolver, ...] = field(default=())
    residue_resolvers: tuple[BlobTerminalRuleResolver, ...] = field(default=())


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
    for excluder in context.excluders:
        rule = excluder.resolve(blob_hash, entry.path, size_bytes)
        if rule is not None:
            return BlobDispositionMember(
                blob_hash=blob_hash,
                size_bytes=size_bytes,
                referenced=referenced,
                disposition=excluder.disposition,
                reason=f"excluded by {rule.rule}: no source is expected to hold non-product material",
                rule=rule,
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
    for resolver in context.residue_resolvers:
        rule = resolver.resolve(blob_hash, entry.path, size_bytes)
        if rule is not None:
            return BlobDispositionMember(
                blob_hash=blob_hash,
                size_bytes=size_bytes,
                referenced=referenced,
                disposition=resolver.disposition,
                reason=f"explained residue owned by {rule.owner} under {rule.rule}",
                rule=rule,
            )
    return BlobDispositionMember(
        blob_hash=blob_hash,
        size_bytes=size_bytes,
        referenced=referenced,
        disposition=BlobDisposition.UNRESOLVED,
        reason="no source proof, no ordinary restoration destination, and no named owner",
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


@dataclass(frozen=True, slots=True)
class DeclaredSourceRoots:
    """The explicit locations a proof is allowed to name.

    Every root is declared rather than discovered, because "the content is
    still at its source" is only auditable if the plan says which source. A
    prover that walked the filesystem at large would make the same claim
    without anyone being able to check it.
    """

    data_roots: tuple[Path, ...] = ()
    export_bundle_roots: tuple[Path, ...] = ()
    drive_cache_roots: tuple[Path, ...] = ()
    session_jsonl_roots: tuple[Path, ...] = ()
    capture_document_roots: tuple[Path, ...] = ()
    test_corpus_roots: tuple[Path, ...] = ()

    @classmethod
    def declared(cls, *, home: Path | None = None, data_lake: Path | None = None) -> DeclaredSourceRoots:
        """The operator-declared topology this one-time transition plans against."""
        base = (home or Path.home()).expanduser()
        lake = (data_lake or Path("/realm/data")).expanduser()
        drive_cache = lake / "ai" / "polylogue" / "drive-cache" / "gemini"
        return cls(
            data_roots=(lake,),
            export_bundle_roots=(lake / "ai" / "chatlog" / "raw" / "claude",),
            drive_cache_roots=(drive_cache,),
            session_jsonl_roots=(base / ".claude" / "projects", base / ".codex" / "sessions"),
            capture_document_roots=(drive_cache,),
            test_corpus_roots=(Path(__file__).resolve().parents[2] / "tests",),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "data_roots": [str(root) for root in self.data_roots],
            "export_bundle_roots": [str(root) for root in self.export_bundle_roots],
            "drive_cache_roots": [str(root) for root in self.drive_cache_roots],
            "session_jsonl_roots": [str(root) for root in self.session_jsonl_roots],
            "capture_document_roots": [str(root) for root in self.capture_document_roots],
        }


# Acquisitions whose declared source path lies under one of these is not
# product material: these are test, fixture, and development-loop routes.
NON_PRODUCT_ROUTE_MARKERS: tuple[str, ...] = (
    ".cache/dev-loop",
    "tests/fixtures",
    "tests/data",
    "tests/infra",
)
NON_PRODUCT_ROUTE_OWNER = "polylogue-251y8"
CAPTURE_ATTACHMENT_RESIDUE_OWNER = "polylogue-hcm7h"


def build_disposition_context(
    *,
    archive_root: Path,
    blob_root: Path,
    source_db: Path,
    hook_spool_sources: Sequence[tuple[str, Path]],
    browser_capture_spool: Path,
    declared_roots: DeclaredSourceRoots | None = None,
) -> BlobDispositionContext:
    """Resolve the prover set from configured sources, not from history.

    Order is cost: the provers that answer from a row or a recorded carrier
    run before the ones that must index a source tree, so an object is
    usually decided without any index being built at all.
    """
    roots = declared_roots if declared_roots is not None else DeclaredSourceRoots.declared()
    store = BlobStore(blob_root)
    carriers = raw_source_carriers_by_hash(source_db)
    referenced = referenced_blob_hashes(source_db)
    sizes = blob_candidate_sizes(blob_root)
    hook_prover = HookEventSpoolProver(hook_spool_sources)
    capture_prover = BrowserCaptureSpoolProver(browser_capture_spool)
    provers: tuple[BlobSourceProver, ...] = (
        hook_prover,
        capture_prover,
        RawSourceFileProver(carriers),
        CodexStateRowProver(hook_event_carriers_by_hash(source_db)),
        DriveCacheInjectedKeyProver(carriers, roots=roots.drive_cache_roots),
        TrajectoryStepPrefixProver(carriers),
        SqliteRowContainmentProver(carriers),
        AppendPrefixProver(append_successors_by_hash(source_db), blob_store=store),
        JsonlLineContainmentProver(carriers, session_roots=roots.session_jsonl_roots),
        ByteSpanSubstringProver(carriers),
        CaptureEmbeddedPayloadProver((browser_capture_spool, *roots.capture_document_roots), candidate_sizes=sizes),
        ExportBundleMemberProver(roots.export_bundle_roots, candidate_sizes=sizes),
        DataLakeFileProver(roots.data_roots, candidate_sizes=sizes),
    )
    return BlobDispositionContext(
        blob_store=store,
        provers=provers,
        referenced_hashes=referenced,
        restoration_provers=(hook_prover, capture_prover),
        excluders=(
            NonProductRouteExcluder(carriers, markers=NON_PRODUCT_ROUTE_MARKERS, owner=NON_PRODUCT_ROUTE_OWNER),
            TestCorpusFixtureExcluder(
                roots.test_corpus_roots, referenced_hashes=referenced, owner=NON_PRODUCT_ROUTE_OWNER
            ),
        ),
        residue_resolvers=(
            BrowserCaptureAttachmentResidue(
                browser_capture_attachment_payloads(source_db),
                owner=CAPTURE_ATTACHMENT_RESIDUE_OWNER,
                rule="browser-capture-attachment-payload",
            ),
            ForeignStateSnapshotResidue(carriers, owner=CAPTURE_ATTACHMENT_RESIDUE_OWNER),
            SidechainTranscriptResidue(referenced_hashes=referenced, owner=CAPTURE_ATTACHMENT_RESIDUE_OWNER),
        ),
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
    "AppendPrefixProver",
    "BlobDisposition",
    "BlobDispositionContext",
    "BlobDispositionDenominator",
    "BlobDispositionError",
    "BlobDispositionMember",
    "BlobDispositionPlan",
    "BlobRestorationResolver",
    "BlobSourceProver",
    "BlobTerminalRuleResolver",
    "BrowserCaptureAttachmentResidue",
    "BrowserCaptureSpoolProver",
    "ByteSpanSubstringProver",
    "CAPTURE_ATTACHMENT_RESIDUE_OWNER",
    "CODEX_STATE_ROW_SPECS",
    "CaptureEmbeddedPayloadProver",
    "CodexStateRowProver",
    "DataLakeFileProver",
    "DeclaredSourceRoots",
    "DriveCacheInjectedKeyProver",
    "ExplainedResidueResolver",
    "ExportBundleMemberProver",
    "HookEventCarrier",
    "HookEventSpoolProver",
    "JsonlLineContainmentProver",
    "NON_PRODUCT_ROUTE_MARKERS",
    "NON_PRODUCT_ROUTE_OWNER",
    "NonProductRouteExcluder",
    "RawSourceCarrier",
    "RawSourceFileProver",
    "RestorationDestination",
    "RestorationTarget",
    "SidechainTranscriptResidue",
    "SourceProof",
    "SourceProofMode",
    "SqliteRowContainmentProver",
    "StateRowSpec",
    "TOOL_VERSION",
    "ForeignStateSnapshotResidue",
    "TerminalRule",
    "TestCorpusFixtureExcluder",
    "TrajectoryStepPrefixProver",
    "append_successors_by_hash",
    "blob_candidate_sizes",
    "browser_capture_attachment_payloads",
    "build_disposition_context",
    "classify_blob",
    "compile_disposition_plan",
    "hook_event_carriers_by_hash",
    "raw_source_carriers_by_hash",
    "referenced_blob_hashes",
    "resolve_disposition_roots",
]
