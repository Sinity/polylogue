"""Explicit ordinary-source inventory and reduced-evidence collection.

The archive-backed sampler remains its own route.  This module observes
operator-selected source roots directly, retains only recipe-bound reduced
evidence in its private cache, and reports every terminal disposition.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import BinaryIO, Literal, cast
from uuid import UUID

from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDecodeError, JSONDocument, JSONValue, is_json_value, loads
from polylogue.schemas.generation.evidence import SchemaEvidence
from polylogue.schemas.observation import extract_schema_units_from_payload, resolve_provider_config
from polylogue.schemas.source_cache import CachedContribution, SourceContributionCache
from polylogue.sources.decoder_zip import ZipBombError, ZipEntryValidator, open_bounded_zip_entry
from polylogue.sources.live.watcher import WatchSource, default_sources
from polylogue.sources.origin_specs import _fingerprint_sources, artifact_suffixes_for_provider, recognize_source_class
from polylogue.sources.source_walk import _iter_source_entries

SourceOutcome = Literal[
    "included",
    "intentionally_excluded",
    "unsupported",
    "decode_failed",
    "changed_during_read",
    "partial_trailing_record",
    "too_large",
]

_RECIPE_VERSION = "source-evidence-v2"
_MAX_UNSTREAMABLE_DOCUMENT_BYTES = 32 * 1024 * 1024
_SOURCE_EVIDENCE_CHUNK_RECORD_LIMIT = 32
_SOURCE_EVIDENCE_PENDING_RECORD_LIMIT = 128
_DECLARED_PRODUCER_VERSION = re.compile(r"v?\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z][0-9A-Za-z.-]{0,63})?")


class SourceInferenceError(RuntimeError):
    """Source inference could not form a complete source observation."""


@dataclass(frozen=True, slots=True)
class SchemaSourceInput:
    """One explicitly selected source root for a provider subject."""

    provider: str
    root: Path

    def __post_init__(self) -> None:
        _canonical_schema_provider(self.provider)


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """Private identity and immutable bytes of an inventoried source member."""

    provider: str
    path: Path
    logical_source_id: str
    revision_sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class SourceTerminal:
    """One attributable source-member terminal outcome without source text."""

    outcome: SourceOutcome
    byte_count: int = 0
    record_count: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SourceObservation:
    """A one-pass unit of source records for the evidence collector.

    ``logical_source_id`` is private ownership data.  It is intentionally not
    serialized by the reduced-evidence cache or projected into package output.
    """

    logical_source_id: str
    revision_sha256: str
    subject: str
    element_kind: str
    records: Iterable[JSONValue]
    is_current: bool = True
    outcome: SourceOutcome = "included"


@dataclass(frozen=True, slots=True)
class SourceInferenceResult:
    """Reduced evidence and aggregate, public-safe run provenance."""

    evidence_by_element: Mapping[str, tuple[JSONDocument, ...]]
    terminal_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    input_bytes: int
    record_count: int
    cache_hits: int
    cache_misses: int
    phase_timings_ms: dict[str, float]
    producer_version_counts: dict[str, int]
    producer_version_missing_sources: int
    producer_version_conflicting_sources: int
    producer_version_unrecognized_sources: int
    input_manifest_digest: str

    def provenance(self) -> JSONDocument:
        """Return aggregate-only source provenance safe for package metadata."""
        return {
            "source_input_bytes": self.input_bytes,
            "source_record_count": self.record_count,
            "source_cache_hits": self.cache_hits,
            "source_cache_misses": self.cache_misses,
            "source_terminal_outcomes": dict(sorted(self.terminal_counts.items())),
            "source_terminal_reasons": dict(sorted(self.terminal_reason_counts.items())),
            "source_phase_timings_ms": dict(sorted(self.phase_timings_ms.items())),
            "producer_version_counts": dict(sorted(self.producer_version_counts.items())),
            "producer_version_missing_sources": self.producer_version_missing_sources,
            "producer_version_conflicting_sources": self.producer_version_conflicting_sources,
            "producer_version_unrecognized_sources": self.producer_version_unrecognized_sources,
            "source_input_manifest_digest": self.input_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class _SourceCandidate:
    provider: str
    root: Path
    path: Path
    logical_source_id: str


@dataclass(frozen=True, slots=True)
class _SourceContribution:
    """Reduced evidence for one native source revision within a member."""

    logical_source_id: str
    revision_sha256: str
    evidence_by_element: dict[str, JSONDocument]
    record_count: int
    declared_updated_at: tuple[int, str] | None


@dataclass(frozen=True, slots=True)
class _ContributionDescriptor:
    """Selection metadata retained between the structure and statistics passes."""

    logical_source_id: str
    revision_sha256: str
    record_count: int
    declared_updated_at: tuple[int, str] | None


@dataclass(frozen=True, slots=True)
class _CandidateDescriptor:
    """One stable member plus its private contribution-selection metadata."""

    candidate: _SourceCandidate
    revision_sha256: str
    byte_count: int
    contributions: tuple[_ContributionDescriptor, ...]
    producer_versions: tuple[str, ...]
    producer_version_unrecognized: bool


@dataclass(frozen=True, slots=True)
class _CollectedCandidate:
    candidate: _SourceCandidate
    revision: SourceRevision | None
    terminal: SourceTerminal
    contributions: tuple[_SourceContribution, ...] = ()
    producer_versions: tuple[str, ...] = ()
    producer_version_unrecognized: bool = False


_EXPLICIT_SCHEMA_SUBJECTS = frozenset({"browser-capture"})


def _canonical_schema_provider(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in _EXPLICIT_SCHEMA_SUBJECTS:
        return normalized
    return Provider.from_string(normalized).value


def parse_schema_source_input(value: str) -> SchemaSourceInput:
    """Parse the explicit ``provider=path`` command argument."""
    provider, separator, path_text = value.partition("=")
    if not separator or not provider or not path_text:
        raise ValueError("schema source inputs must use provider=path")
    return SchemaSourceInput(provider=_canonical_schema_provider(provider), root=Path(path_text).expanduser())


def default_schema_source_inputs(*, provider: str) -> tuple[SchemaSourceInput, ...]:
    """Adapt executable watcher declarations without inventing path defaults."""
    subject = _canonical_schema_provider(provider)
    rows: list[SchemaSourceInput] = []
    for source in default_sources():
        if source.name == subject:
            rows.append(SchemaSourceInput(provider=subject, root=source.root))
            continue
        try:
            source_provider = _canonical_schema_provider(source.name)
        except ValueError:
            continue
        if source_provider == subject:
            rows.append(SchemaSourceInput(provider=subject, root=source.root))
    return tuple(rows)


def _root_identity(root: Path) -> str:
    try:
        source_stat = root.stat()
    except OSError as exc:
        raise SourceInferenceError(f"source root is unavailable: {root}") from exc
    return f"{source_stat.st_dev}:{source_stat.st_ino}"


def _canonical_inputs(inputs: Iterable[SchemaSourceInput]) -> tuple[SchemaSourceInput, ...]:
    """Keep one spelling per physical root/provider pair."""
    canonical: dict[tuple[str, str], SchemaSourceInput] = {}
    for source_input in inputs:
        provider = _canonical_schema_provider(source_input.provider)
        key = provider, _root_identity(source_input.root)
        existing = canonical.get(key)
        if existing is None or str(source_input.root) < str(existing.root):
            canonical[key] = SchemaSourceInput(provider=provider, root=source_input.root)
    return tuple(sorted(canonical.values(), key=lambda item: (item.provider, str(item.root))))


def inventory_schema_sources(inputs: Iterable[SchemaSourceInput]) -> tuple[_SourceCandidate, ...]:
    """Enumerate every admitted source candidate through watcher semantics."""
    candidates: list[_SourceCandidate] = []
    for source_input in _canonical_inputs(inputs):
        provider = _canonical_schema_provider(source_input.provider)
        provider_token = Provider.from_string(provider)
        root = source_input.root
        watcher_source = WatchSource(
            name=provider,
            root=root,
            suffixes=artifact_suffixes_for_provider(
                provider_token,
                defaults=(".json", ".jsonl", ".ndjson", ".zip", ".db", ".sqlite", ".sqlite3"),
            ),
        )
        root_identity = _root_identity(root)
        paths = (root,) if root.is_file() else _iter_source_entries(root)
        for path in paths:
            try:
                mode = os.stat(path, follow_symlinks=False).st_mode
            except OSError:
                continue
            if not stat.S_ISREG(mode) or not watcher_source.accepts(path):
                continue
            relative = Path(path.name) if root.is_file() else path.relative_to(root)
            candidates.append(
                _SourceCandidate(
                    provider=provider,
                    root=root,
                    path=path,
                    logical_source_id=f"{provider}:{root_identity}:{relative.as_posix()}",
                )
            )
    return tuple(sorted(candidates, key=lambda item: (item.provider, item.logical_source_id, str(item.path))))


def _candidate_byte_count(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _preflight_terminal(candidate: _SourceCandidate) -> SourceTerminal | None:
    """Refuse source classes that lack a source-evidence adapter before reading bytes."""
    byte_count = _candidate_byte_count(candidate.path)
    if candidate.provider == "browser-capture":
        return SourceTerminal("unsupported", byte_count, reason="browser_capture_adapter_unavailable")
    provider = Provider.from_string(candidate.provider)
    if provider is Provider.ANTIGRAVITY and candidate.path.suffix.lower() == ".pb":
        return SourceTerminal("unsupported", byte_count, reason="antigravity_protobuf_adapter_unavailable")
    if provider is Provider.ANTIGRAVITY and candidate.path.suffix.lower() == ".md":
        return SourceTerminal("intentionally_excluded", byte_count, reason="antigravity_markdown_sidecar")
    recognition = recognize_source_class(provider, candidate.path)
    if recognition is not None and recognition.source_class != "session":
        return SourceTerminal(
            "intentionally_excluded" if recognition.source_class == "non_session" else "unsupported",
            byte_count,
            reason=f"source_class_{recognition.source_class}",
        )
    if candidate.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return SourceTerminal("unsupported", byte_count, reason="sqlite_value_inference_not_supported")
    return None


def _terminal_reason_code(terminal: SourceTerminal) -> str | None:
    """Return a stable aggregate code without retaining parser or path text."""
    if terminal.outcome == "included":
        return None
    if terminal.reason in {
        "antigravity_markdown_sidecar",
        "antigravity_protobuf_adapter_unavailable",
        "browser_capture_adapter_unavailable",
        "sqlite_value_inference_not_supported",
        "source_class_non_session",
        "source_class_unsupported",
        "invalid_zip",
        "no_schema_units",
        "no_schema_zip_members",
        "partial_trailing_record",
        "unreadable_source",
    }:
        return terminal.reason
    if terminal.reason is not None and terminal.reason.startswith("malformed_jsonl_record:"):
        return "malformed_jsonl_record"
    if terminal.reason is not None and terminal.reason.startswith("malformed_json:"):
        return "malformed_json"
    return terminal.outcome


def _stable_file_digest(path: Path) -> tuple[str, int]:
    """Hash a member without buffering its source text in memory."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    after_identity = after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    if before_identity != after_identity:
        raise SourceInferenceError("changed_during_read")
    return digest.hexdigest(), before.st_size


def _is_strict_file_prefix(shorter: Path, longer: Path) -> bool:
    """Compare revisions by bytes without retaining either source in memory."""
    try:
        if shorter.stat().st_size >= longer.stat().st_size:
            return False
        with shorter.open("rb") as left, longer.open("rb") as right:
            while chunk := left.read(1024 * 1024):
                if right.read(len(chunk)) != chunk:
                    return False
        return True
    except OSError:
        return False


def _iter_jsonl_payloads(handle: Iterable[bytes]) -> Iterator[JSONValue]:
    """Decode JSONL record by record and fail closed on incomplete input."""
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        if not line.endswith((b"\n", b"\r")):
            raise SourceInferenceError("partial_trailing_record")
        try:
            value = loads(line)
        except JSONDecodeError as exc:
            raise SourceInferenceError(f"malformed_jsonl_record:{line_number}") from exc
        if not is_json_value(value):
            raise SourceInferenceError("non_json_value")
        yield value


def _iter_document_payloads(
    open_handle: Callable[[], BinaryIO],
    path_name: str,
    *,
    byte_count: int,
) -> Iterator[JSONValue]:
    """Stream top-level export arrays without materializing whole exports."""
    import ijson

    for prefix in ("item", "conversations.item", "sessions.item"):
        found = False
        try:
            with open_handle() as handle:
                for value in ijson.items(handle, prefix, use_float=True):
                    found = True
                    if not is_json_value(value):
                        raise SourceInferenceError("non_json_value")
                    yield value
        except ijson.JSONError as exc:
            raise SourceInferenceError(f"malformed_json:{path_name}") from exc
        if found:
            return
    if byte_count > _MAX_UNSTREAMABLE_DOCUMENT_BYTES:
        raise SourceInferenceError("too_large_unstreamable_document")
    with open_handle() as handle:
        try:
            value = loads(handle.read())
        except JSONDecodeError as exc:
            raise SourceInferenceError(f"malformed_json:{path_name}") from exc
    if not is_json_value(value):
        raise SourceInferenceError("non_json_value")
    yield value


def _iter_file_payloads(path: Path, *, byte_count: int) -> Iterator[JSONValue]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("rb") as handle:
            yield from _iter_jsonl_payloads(handle)
        return
    yield from _iter_document_payloads(lambda: path.open("rb"), str(path), byte_count=byte_count)


def _native_source_id(provider: Provider, payload: JSONValue, fallback: str) -> str:
    """Return the provider-native session identifier when the record declares one.

    The collector hashes this private token before retaining equality evidence.
    A path-derived fallback is needed for source formats without a session key,
    but it must never replace a declared native identity.
    """
    if not isinstance(payload, dict):
        return fallback
    if provider is Provider.CLAUDE_CODE:
        session_id = payload.get("sessionId")
        if isinstance(session_id, str) and session_id:
            return f"claude-code:{session_id}"
    if provider is Provider.CODEX and payload.get("type") == "session_meta":
        session_payload = payload.get("payload")
        if isinstance(session_payload, dict):
            session_id = session_payload.get("id")
            if isinstance(session_id, str) and session_id:
                return f"codex:{session_id}"
    if provider is Provider.CHATGPT:
        session_id = payload.get("conversation_id") or payload.get("id") or payload.get("uuid")
        if isinstance(session_id, str) and session_id:
            return f"chatgpt:{session_id}"
    if provider is Provider.CLAUDE_AI:
        session_id = payload.get("uuid") or payload.get("id")
        if isinstance(session_id, str) and session_id:
            return f"claude-ai:{session_id}"
    return fallback


def _claude_code_native_identity(declared: str, candidate: _SourceCandidate) -> str:
    """Apply Claude Code's declared file-identity rules before contribution grouping."""
    fallback_id = candidate.path.stem
    session_id = declared.removeprefix("claude-code:")
    if fallback_id.startswith("agent-"):
        return f"claude-code:{session_id}:{fallback_id}"
    try:
        UUID(fallback_id)
    except ValueError:
        return declared
    return declared if fallback_id == session_id else f"claude-code:{fallback_id}"


def _declared_update_key(provider: Provider, payload: JSONValue) -> tuple[int, str] | None:
    """Return an ordering key from a provider-declared session update field."""
    if not isinstance(payload, dict):
        return None
    keys = {
        Provider.CHATGPT: ("update_time",),
        Provider.CLAUDE_AI: ("updated_at", "updatedAt"),
        Provider.CLAUDE_CODE: ("timestamp",),
    }.get(provider, ())
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return 1, f"{value:030.9f}"
        if isinstance(value, str) and value:
            return 0, value
    return None


def _collect_payload_evidence(
    candidate: _SourceCandidate,
    revision: SourceRevision,
    payloads: Iterable[JSONValue],
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]],
    include_statistics: bool = True,
    chunk_record_limit: int = _SOURCE_EVIDENCE_CHUNK_RECORD_LIMIT,
) -> tuple[tuple[_SourceContribution, ...], int, tuple[str, ...], bool]:
    """Reduce each native source revision without retaining decoded records."""
    from polylogue.schemas.generation.evidence import SchemaEvidenceAccumulator, collect_source_evidence

    if chunk_record_limit < 1:
        raise ValueError("chunk_record_limit must be positive")
    provider = Provider.from_string(candidate.provider)
    config = resolve_provider_config(provider)
    evidence_rows: dict[str, dict[str, SchemaEvidenceAccumulator]] = {}
    record_counts: Counter[str] = Counter()
    update_keys: dict[str, tuple[int, str] | None] = {}
    producer_versions: set[str] = set()
    producer_version_unrecognized = False
    source_seen: set[tuple[str, str]] = set()
    pending_records: dict[tuple[str, str], list[JSONValue]] = {}
    pending_record_count = 0

    def flush(source_key: tuple[str, str]) -> None:
        nonlocal pending_record_count
        records = pending_records.pop(source_key)
        pending_record_count -= len(records)
        source_id, element_kind = source_key
        contribution = collect_source_evidence(
            SourceObservation(
                logical_source_id=source_id,
                revision_sha256=revision.revision_sha256,
                subject=candidate.provider,
                element_kind=element_kind,
                records=records,
            ),
            dynamic_paths=dynamic_paths_by_element.get(element_kind, ()),
            include_statistics=include_statistics,
        )
        if source_key in source_seen:
            contribution = replace(contribution, current_source_count=0)
        else:
            source_seen.add(source_key)
        evidence_rows.setdefault(source_id, {}).setdefault(element_kind, SchemaEvidenceAccumulator()).add(contribution)

    def append(source_id: str, element_kind: str, records: Iterable[JSONValue]) -> None:
        nonlocal pending_record_count
        source_key = source_id, element_kind
        pending = pending_records.setdefault(source_key, [])
        for record in records:
            pending.append(record)
            pending_record_count += 1
            if len(pending) >= chunk_record_limit:
                flush(source_key)
                pending = pending_records.setdefault(source_key, [])
            while pending_record_count >= _SOURCE_EVIDENCE_PENDING_RECORD_LIMIT:
                oldest = next(iter(pending_records))
                flush(oldest)
                if oldest == source_key:
                    pending = pending_records.setdefault(source_key, [])

    header_source_id = candidate.logical_source_id
    for payload_index, payload in enumerate(payloads):
        versions, unrecognized = _declared_producer_versions(provider, (payload,))
        producer_versions.update(versions)
        producer_version_unrecognized = producer_version_unrecognized or unrecognized
        declared = _native_source_id(provider, payload, "")
        if declared and provider is Provider.CLAUDE_CODE:
            declared = _claude_code_native_identity(declared, candidate)
        if declared:
            header_source_id = declared
        declared_source_id = hash_payload({"source": declared or header_source_id})
        update = _declared_update_key(provider, payload)
        prior_update = update_keys.get(declared_source_id)
        if update is not None and (prior_update is None or update > prior_update):
            update_keys[declared_source_id] = update
        units = extract_schema_units_from_payload(
            [payload] if config.sample_granularity == "record" else payload,
            source_name=provider,
            source_path=candidate.path,
            raw_id=f"{revision.revision_sha256}:{payload_index}",
            config=config,
            full_corpus=True,
            compact_values=False,
        )
        for unit in units:
            record_counts[declared_source_id] += len(unit.schema_samples)
            append(declared_source_id, unit.artifact_kind, unit.schema_samples)
    while pending_records:
        flush(next(iter(pending_records)))
    contributions: list[_SourceContribution] = []
    for source_id, rows in sorted(evidence_rows.items()):
        payload_by_element: dict[str, JSONDocument] = {}
        for element_kind, accumulator in sorted(rows.items()):
            payload = accumulator.finish().to_json()
            if not isinstance(payload, dict):
                raise SourceInferenceError("source evidence must serialize to a JSON object")
            payload_by_element[element_kind] = payload
        contributions.append(
            _SourceContribution(
                logical_source_id=source_id,
                revision_sha256=revision.revision_sha256,
                evidence_by_element=payload_by_element,
                record_count=record_counts[source_id],
                declared_updated_at=update_keys.get(source_id),
            )
        )
    return (
        tuple(contributions),
        sum(record_counts.values()),
        tuple(sorted(producer_versions)),
        producer_version_unrecognized,
    )


def _declared_producer_versions(provider: Provider, payloads: Iterable[JSONValue]) -> tuple[tuple[str, ...], bool]:
    """Extract only provider-declared release evidence, never generic keys."""
    versions: set[str] = set()
    unrecognized = False
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if provider is Provider.CLAUDE_CODE:
            version = payload.get("version")
            if isinstance(version, str) and version:
                if _DECLARED_PRODUCER_VERSION.fullmatch(version):
                    versions.add(version.removeprefix("v"))
                else:
                    unrecognized = True
        elif provider is Provider.CODEX and payload.get("type") == "session_meta":
            session_payload = payload.get("payload")
            if isinstance(session_payload, dict):
                version = session_payload.get("cli_version")
                if isinstance(version, str) and version:
                    if _DECLARED_PRODUCER_VERSION.fullmatch(version):
                        versions.add(version.removeprefix("v"))
                    else:
                        unrecognized = True
    return tuple(sorted(versions)), unrecognized


def _collect_candidate(
    candidate: _SourceCandidate,
    dynamic_paths_by_element: dict[str, tuple[str, ...]] | None = None,
    *,
    include_statistics: bool = True,
) -> _CollectedCandidate:
    """Read one member fully and construct one-pass evidence observations."""
    dynamic_paths_by_element = dynamic_paths_by_element or {}
    preflight = _preflight_terminal(candidate)
    if preflight is not None:
        return _CollectedCandidate(candidate, None, preflight)
    try:
        digest, byte_count = _stable_file_digest(candidate.path)
    except SourceInferenceError:
        return _CollectedCandidate(candidate, None, SourceTerminal("changed_during_read"))
    except OSError:
        return _CollectedCandidate(candidate, None, SourceTerminal("decode_failed", reason="unreadable_source"))
    revision = SourceRevision(
        provider=candidate.provider,
        path=candidate.path,
        logical_source_id=candidate.logical_source_id,
        revision_sha256=digest,
        byte_count=byte_count,
    )
    if candidate.path.suffix.lower() == ".zip":
        return _collect_zip_candidate(
            candidate,
            revision,
            dynamic_paths_by_element=dynamic_paths_by_element,
            include_statistics=include_statistics,
        )
    try:
        contributions, record_count, producer_versions, producer_version_unrecognized = _collect_payload_evidence(
            candidate,
            revision,
            _iter_file_payloads(candidate.path, byte_count=byte_count),
            dynamic_paths_by_element=dynamic_paths_by_element,
            include_statistics=include_statistics,
        )
    except SourceInferenceError as exc:
        reason = str(exc)
        outcome: SourceOutcome = (
            "partial_trailing_record"
            if reason == "partial_trailing_record"
            else "too_large"
            if reason.startswith("too_large")
            else "decode_failed"
        )
        return _CollectedCandidate(candidate, revision, SourceTerminal(outcome, byte_count, reason=reason))
    try:
        after_digest, _after_bytes = _stable_file_digest(candidate.path)
    except (OSError, SourceInferenceError):
        return _CollectedCandidate(candidate, revision, SourceTerminal("changed_during_read", byte_count))
    if after_digest != revision.revision_sha256:
        return _CollectedCandidate(candidate, revision, SourceTerminal("changed_during_read", byte_count))
    if not contributions:
        return _CollectedCandidate(
            candidate, revision, SourceTerminal("unsupported", byte_count, reason="no_schema_units")
        )
    return _CollectedCandidate(
        candidate,
        revision,
        SourceTerminal("included", byte_count, record_count),
        contributions,
        producer_versions,
        producer_version_unrecognized,
    )


def _collect_zip_candidate(
    candidate: _SourceCandidate,
    revision: SourceRevision,
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]],
    include_statistics: bool,
) -> _CollectedCandidate:
    """Reduce ZIP members as independent source revisions."""
    contributions: list[_SourceContribution] = []
    byte_count = 0
    record_count = 0
    producer_versions: set[str] = set()
    producer_version_unrecognized = False
    try:
        with zipfile.ZipFile(candidate.path) as archive:
            validator = ZipEntryValidator(candidate.provider, cursor_state=None, zip_path=candidate.path)
            members = validator.filter_entries(archive.infolist(), allowed_suffixes=(".json", ".jsonl", ".ndjson"))
            for member in sorted(members, key=lambda item: item.filename):
                member_path = Path(member.filename)
                with open_bounded_zip_entry(archive, member) as member_handle:
                    digest_builder = hashlib.sha256()
                    for chunk in iter(lambda: member_handle.read(1024 * 1024), b""):
                        digest_builder.update(chunk)
                member_revision = SourceRevision(
                    provider=revision.provider,
                    path=candidate.path,
                    logical_source_id=f"{candidate.logical_source_id}:zip:{member.filename}",
                    revision_sha256=digest_builder.hexdigest(),
                    byte_count=member.file_size,
                )
                member_candidate = _SourceCandidate(
                    provider=candidate.provider,
                    root=candidate.root,
                    path=member_path,
                    logical_source_id=member_revision.logical_source_id,
                )
                try:
                    if member_path.suffix.lower() in {".jsonl", ".ndjson"}:
                        with open_bounded_zip_entry(archive, member) as handle:
                            rows, member_records, versions, unrecognized = _collect_payload_evidence(
                                member_candidate,
                                member_revision,
                                _iter_jsonl_payloads(handle),
                                dynamic_paths_by_element=dynamic_paths_by_element,
                                include_statistics=include_statistics,
                            )
                    else:
                        rows, member_records, versions, unrecognized = _collect_payload_evidence(
                            member_candidate,
                            member_revision,
                            _iter_document_payloads(
                                partial(open_bounded_zip_entry, archive, member),
                                member.filename,
                                byte_count=member.file_size,
                            ),
                            dynamic_paths_by_element=dynamic_paths_by_element,
                            include_statistics=include_statistics,
                        )
                except (SourceInferenceError, ZipBombError, OSError) as exc:
                    return _CollectedCandidate(
                        candidate, revision, SourceTerminal("decode_failed", byte_count, reason=str(exc))
                    )
                contributions.extend(rows)
                byte_count += member.file_size
                record_count += member_records
                producer_versions.update(versions)
                producer_version_unrecognized = producer_version_unrecognized or unrecognized
    except (OSError, ZipBombError, zipfile.BadZipFile):
        return _CollectedCandidate(candidate, revision, SourceTerminal("decode_failed", reason="invalid_zip"))
    if not contributions:
        return _CollectedCandidate(candidate, revision, SourceTerminal("unsupported", reason="no_schema_zip_members"))
    return _CollectedCandidate(
        candidate,
        revision,
        SourceTerminal("included", byte_count, record_count),
        tuple(contributions),
        tuple(sorted(producer_versions)),
        producer_version_unrecognized,
    )


def _bounded_collected_candidates(
    executor: ProcessPoolExecutor,
    candidates: Iterable[_SourceCandidate],
    *,
    limit: int,
    dynamic_paths_by_element: dict[str, tuple[str, ...]],
    include_statistics: bool = True,
) -> Iterator[_CollectedCandidate]:
    """Drain completed source workers while keeping the submission window bounded."""
    iterator = iter(candidates)
    pending = set()
    for _ in range(limit):
        try:
            pending.add(
                executor.submit(
                    _collect_candidate,
                    next(iterator),
                    dynamic_paths_by_element,
                    include_statistics=include_statistics,
                )
            )
        except StopIteration:
            break
    while pending:
        ready, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in ready:
            yield future.result()
            with suppress(StopIteration):
                pending.add(
                    executor.submit(
                        _collect_candidate,
                        next(iterator),
                        dynamic_paths_by_element,
                        include_statistics=include_statistics,
                    )
                )


def _source_recipe_fingerprint() -> str:
    """Bind cache rows to the source reducer's complete import closure."""
    return _fingerprint_sources(("polylogue/schemas/source_inference.py",), namespace="schema-source-evidence")


def _serialize_contributions(contributions: Iterable[_SourceContribution]) -> JSONDocument:
    return {
        "contributions": [
            {
                "source": item.logical_source_id,
                "revision": item.revision_sha256,
                "elements": cast(JSONValue, item.evidence_by_element),
                "records": item.record_count,
                "updated": list(item.declared_updated_at) if item.declared_updated_at is not None else None,
            }
            for item in sorted(contributions, key=lambda item: (item.logical_source_id, item.revision_sha256))
        ]
    }


def _cached_contributions(payload: JSONDocument) -> Iterator[_SourceContribution]:
    rows = payload.get("contributions")
    if not isinstance(rows, list):
        raise SourceInferenceError("cached source evidence has no contribution list")
    for row in rows:
        if not isinstance(row, dict):
            raise SourceInferenceError("cached source contribution is invalid")
        source, revision, elements, records, updated = (
            row.get("source"),
            row.get("revision"),
            row.get("elements"),
            row.get("records"),
            row.get("updated"),
        )
        if not isinstance(source, str) or not isinstance(revision, str) or not isinstance(elements, dict):
            raise SourceInferenceError("cached source contribution is invalid")
        if not isinstance(records, int) or isinstance(records, bool) or records < 0:
            raise SourceInferenceError("cached source contribution record count is invalid")
        evidence_by_element: dict[str, JSONDocument] = {}
        for kind, evidence in elements.items():
            if not isinstance(evidence, dict):
                raise SourceInferenceError("cached source contribution element is invalid")
            evidence_by_element[kind] = evidence
        update_key: tuple[int, str] | None = None
        if updated is not None:
            if not (
                isinstance(updated, list)
                and len(updated) == 2
                and isinstance(updated[0], int)
                and not isinstance(updated[0], bool)
                and isinstance(updated[1], str)
            ):
                raise SourceInferenceError("cached source contribution update key is invalid")
            update_key = updated[0], updated[1]
        yield _SourceContribution(source, revision, evidence_by_element, records, update_key)


def _historical_payload(payload: JSONDocument) -> JSONDocument:
    evidence = SchemaEvidence.from_json(payload)
    return replace(
        evidence,
        current_structure={},
        fields={},
        current_source_count=0,
        current_record_count=0,
        historical_structure=evidence.structure,
        historical_source_count=1,
        historical_record_count=evidence.current_record_count,
    ).to_json()


def _merge_evidence_by_element(payloads: Iterable[dict[str, JSONDocument]]) -> dict[str, SchemaEvidence]:
    groups: dict[str, list[JSONDocument]] = {}
    for source_payload in payloads:
        for element_kind, payload in source_payload.items():
            groups.setdefault(element_kind, []).append(payload)
    from polylogue.schemas.generation.evidence import merge_evidence

    return {
        element_kind: merge_evidence(SchemaEvidence.from_json(payload) for payload in rows)
        for element_kind, rows in sorted(groups.items())
    }


def _cache_key(
    candidate: _SourceCandidate,
    revision_sha256: str,
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]] | None,
    recipe_fingerprint: str,
) -> str:
    return hash_payload(
        {
            "recipe": recipe_fingerprint,
            "subject": candidate.provider,
            "source_context": hash_payload({"logical_source_id": candidate.logical_source_id}),
            "revision_sha256": revision_sha256,
            "dynamic_paths": (
                {kind: list(paths) for kind, paths in sorted(dynamic_paths_by_element.items())}
                if dynamic_paths_by_element is not None
                else None
            ),
        }
    )


def _cached_versions(metadata: JSONDocument) -> tuple[str, ...]:
    values = metadata.get("producer_versions")
    return tuple(value for value in values if isinstance(value, str)) if isinstance(values, list) else ()


def infer_sources(
    inputs: Iterable[SchemaSourceInput],
    *,
    cache_path: Path,
    max_workers: int = 2,
    progress: Callable[[str, JSONDocument], None] | None = None,
) -> SourceInferenceResult:
    """Collect complete source evidence without retaining every reduced member."""
    from polylogue.schemas.generation.dynamic_keys import dynamic_object_paths
    from polylogue.schemas.generation.evidence import SchemaEvidenceAccumulator

    started = time.monotonic_ns()
    candidates = inventory_schema_sources(inputs)
    inventory_ms = (time.monotonic_ns() - started) / 1_000_000
    terminal_counts: Counter[str] = Counter()
    terminal_reason_counts: Counter[str] = Counter()
    input_bytes_by_candidate: dict[_SourceCandidate, int] = {}
    cache_hits = 0
    cache_misses = 0
    completed = 0
    last_progress_ns = started
    preliminary_records = 0

    def report(phase: str, *, force: bool = False, records: int = 0, input_bytes: int = 0) -> None:
        nonlocal last_progress_ns
        now = time.monotonic_ns()
        if progress is None or not force and now - last_progress_ns < 2_000_000_000:
            return
        progress(
            phase,
            {
                "completed_candidates": completed,
                "total_candidates": len(candidates),
                "input_bytes": input_bytes,
                "record_count": records,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "elapsed_ms": round((now - started) / 1_000_000, 3),
            },
        )
        last_progress_ns = now

    report("inventory", force=True)
    collect_started = time.monotonic_ns()
    recipe_fingerprint = _source_recipe_fingerprint()
    preliminary_by_element: dict[str, SchemaEvidenceAccumulator] = {}
    descriptors: list[_CandidateDescriptor] = []

    def add_preliminary(
        candidate: _SourceCandidate,
        digest: str,
        byte_count: int,
        contributions: Iterable[_SourceContribution],
        versions: tuple[str, ...],
        unrecognized: bool,
    ) -> None:
        nonlocal preliminary_records
        contribution_descriptors: list[_ContributionDescriptor] = []
        for contribution in contributions:
            contribution_descriptors.append(
                _ContributionDescriptor(
                    logical_source_id=contribution.logical_source_id,
                    revision_sha256=contribution.revision_sha256,
                    record_count=contribution.record_count,
                    declared_updated_at=contribution.declared_updated_at,
                )
            )
            preliminary_records += contribution.record_count
            for kind, payload in contribution.evidence_by_element.items():
                preliminary_by_element.setdefault(kind, SchemaEvidenceAccumulator()).add(
                    SchemaEvidence.from_json(payload)
                )
        descriptors.append(
            _CandidateDescriptor(
                candidate=candidate,
                revision_sha256=digest,
                byte_count=byte_count,
                contributions=tuple(contribution_descriptors),
                producer_versions=versions,
                producer_version_unrecognized=unrecognized,
            )
        )

    with SourceContributionCache(cache_path) as cache:
        misses: list[tuple[_SourceCandidate, str, int]] = []
        for candidate in candidates:
            preflight = _preflight_terminal(candidate)
            if preflight is not None:
                terminal_counts[preflight.outcome] += 1
                reason_code = _terminal_reason_code(preflight)
                if reason_code is not None:
                    terminal_reason_counts[reason_code] += 1
                input_bytes_by_candidate[candidate] = preflight.byte_count
                completed += 1
                report("inventory", input_bytes=sum(input_bytes_by_candidate.values()))
                continue
            try:
                digest, byte_count = _stable_file_digest(candidate.path)
            except SourceInferenceError:
                terminal = SourceTerminal("changed_during_read", _candidate_byte_count(candidate.path))
                terminal_counts[terminal.outcome] += 1
                reason_code = _terminal_reason_code(terminal)
                if reason_code is not None:
                    terminal_reason_counts[reason_code] += 1
                input_bytes_by_candidate[candidate] = terminal.byte_count
                completed += 1
                report("hash", input_bytes=sum(input_bytes_by_candidate.values()))
                continue
            except OSError:
                terminal = SourceTerminal("decode_failed", reason="unreadable_source")
                terminal_counts[terminal.outcome] += 1
                reason_code = _terminal_reason_code(terminal)
                if reason_code is not None:
                    terminal_reason_counts[reason_code] += 1
                input_bytes_by_candidate[candidate] = terminal.byte_count
                completed += 1
                report("hash", input_bytes=sum(input_bytes_by_candidate.values()))
                continue
            input_bytes_by_candidate[candidate] = byte_count
            cached = cache.get(
                _cache_key(candidate, digest, dynamic_paths_by_element=None, recipe_fingerprint=recipe_fingerprint)
            )
            if cached is None:
                misses.append((candidate, digest, byte_count))
                continue
            add_preliminary(
                candidate,
                digest,
                byte_count,
                _cached_contributions(cached.evidence),
                _cached_versions(cached.metadata),
                bool(cached.metadata.get("producer_version_unrecognized")),
            )
            cache_hits += 1
            completed += 1
            report(
                "hash",
                records=preliminary_records,
                input_bytes=sum(input_bytes_by_candidate.values()),
            )
        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for item in _bounded_collected_candidates(
                executor,
                (candidate for candidate, _digest, _bytes in misses),
                limit=max(1, max_workers) * 2,
                dynamic_paths_by_element={},
                include_statistics=False,
            ):
                if item.terminal.outcome != "included" or item.revision is None:
                    terminal_counts[item.terminal.outcome] += 1
                    reason_code = _terminal_reason_code(item.terminal)
                    if reason_code is not None:
                        terminal_reason_counts[reason_code] += 1
                    completed += 1
                    report("collect", input_bytes=sum(input_bytes_by_candidate.values()))
                    continue
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(
                            item.candidate,
                            item.revision.revision_sha256,
                            dynamic_paths_by_element=None,
                            recipe_fingerprint=recipe_fingerprint,
                        ),
                        evidence=_serialize_contributions(item.contributions),
                        input_bytes=item.terminal.byte_count,
                        record_count=item.terminal.record_count,
                        metadata={
                            "producer_versions": list(item.producer_versions),
                            "producer_version_unrecognized": item.producer_version_unrecognized,
                        },
                    )
                )
                add_preliminary(
                    item.candidate,
                    item.revision.revision_sha256,
                    item.terminal.byte_count,
                    item.contributions,
                    item.producer_versions,
                    item.producer_version_unrecognized,
                )
                cache_misses += 1
                completed += 1
                report(
                    "collect",
                    records=preliminary_records,
                    input_bytes=sum(input_bytes_by_candidate.values()),
                )

        descriptors.sort(
            key=lambda item: (
                item.candidate.provider,
                item.candidate.logical_source_id,
                str(item.candidate.path),
                item.revision_sha256,
            )
        )
        dynamic_paths_by_element = {
            kind: tuple(sorted(dynamic_object_paths(accumulator.finish().structure)))
            for kind, accumulator in preliminary_by_element.items()
        }
        report(
            "reduce",
            force=True,
            records=preliminary_records,
            input_bytes=sum(input_bytes_by_candidate.values()),
        )

        final: list[_CandidateDescriptor] = []
        final_misses: list[_CandidateDescriptor] = []
        for descriptor in descriptors:
            candidate = descriptor.candidate
            try:
                final_digest, final_byte_count = _stable_file_digest(candidate.path)
            except (OSError, SourceInferenceError):
                final_digest = None
                final_byte_count = _candidate_byte_count(candidate.path)
            if final_digest != descriptor.revision_sha256:
                terminal = SourceTerminal("changed_during_read", final_byte_count)
                terminal_counts[terminal.outcome] += 1
                reason_code = _terminal_reason_code(terminal)
                if reason_code is not None:
                    terminal_reason_counts[reason_code] += 1
                input_bytes_by_candidate[candidate] = terminal.byte_count
                continue
            cached = cache.get(
                _cache_key(
                    candidate,
                    descriptor.revision_sha256,
                    dynamic_paths_by_element=dynamic_paths_by_element,
                    recipe_fingerprint=recipe_fingerprint,
                )
            )
            if cached is None:
                final_misses.append(descriptor)
                continue
            final.append(descriptor)
            cache_hits += 1

        expected_by_candidate = {descriptor.candidate: descriptor for descriptor in final_misses}
        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for item in _bounded_collected_candidates(
                executor,
                (descriptor.candidate for descriptor in final_misses),
                limit=max(1, max_workers) * 2,
                dynamic_paths_by_element=dynamic_paths_by_element,
            ):
                expected = expected_by_candidate[item.candidate]
                if (
                    item.terminal.outcome != "included"
                    or item.revision is None
                    or item.revision.revision_sha256 != expected.revision_sha256
                ):
                    terminal = (
                        item.terminal
                        if item.terminal.outcome != "included"
                        else SourceTerminal("changed_during_read", item.terminal.byte_count)
                    )
                    terminal_counts[terminal.outcome] += 1
                    reason_code = _terminal_reason_code(terminal)
                    if reason_code is not None:
                        terminal_reason_counts[reason_code] += 1
                    input_bytes_by_candidate[item.candidate] = terminal.byte_count
                    continue
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(
                            item.candidate,
                            item.revision.revision_sha256,
                            dynamic_paths_by_element=dynamic_paths_by_element,
                            recipe_fingerprint=recipe_fingerprint,
                        ),
                        evidence=_serialize_contributions(item.contributions),
                        input_bytes=item.terminal.byte_count,
                        record_count=item.terminal.record_count,
                        metadata={
                            "producer_versions": list(item.producer_versions),
                            "producer_version_unrecognized": item.producer_version_unrecognized,
                        },
                    )
                )
                final.append(expected)
                cache_misses += 1

        final.sort(
            key=lambda item: (
                item.candidate.provider,
                item.candidate.logical_source_id,
                str(item.candidate.path),
                item.revision_sha256,
            )
        )
        unique: dict[tuple[str, str], tuple[_CandidateDescriptor, _ContributionDescriptor]] = {}
        for descriptor in final:
            for contribution in descriptor.contributions:
                key = contribution.logical_source_id, contribution.revision_sha256
                prior = unique.get(key)
                if prior is None or str(descriptor.candidate.path) < str(prior[0].candidate.path):
                    unique[key] = descriptor, contribution
        by_identity: dict[str, list[tuple[_CandidateDescriptor, _ContributionDescriptor]]] = {}
        for row in unique.values():
            by_identity.setdefault(row[1].logical_source_id, []).append(row)
        current_rows: list[tuple[_CandidateDescriptor, _ContributionDescriptor]] = []
        historical_rows: list[tuple[_CandidateDescriptor, _ContributionDescriptor]] = []
        for rows in by_identity.values():
            maximal = [
                row
                for row in rows
                if not any(
                    _is_strict_file_prefix(row[0].candidate.path, other[0].candidate.path)
                    for other in rows
                    if other != row
                )
            ]
            selected = max(
                maximal,
                key=lambda row: (
                    row[1].declared_updated_at is not None,
                    row[1].declared_updated_at or (-1, ""),
                    row[1].revision_sha256,
                    str(row[0].candidate.path),
                ),
            )
            current_rows.append(selected)
            historical_rows.extend(row for row in rows if row != selected)

        selected_by_candidate: dict[_CandidateDescriptor, dict[tuple[str, str], bool]] = {}
        for descriptor, contribution in current_rows:
            selected_by_candidate.setdefault(descriptor, {})[
                (contribution.logical_source_id, contribution.revision_sha256)
            ] = True
        for descriptor, contribution in historical_rows:
            selected_by_candidate.setdefault(descriptor, {})[
                (contribution.logical_source_id, contribution.revision_sha256)
            ] = False

        evidence_by_element: dict[str, SchemaEvidenceAccumulator] = {}
        for descriptor in sorted(
            selected_by_candidate,
            key=lambda item: (
                item.candidate.provider,
                item.candidate.logical_source_id,
                str(item.candidate.path),
                item.revision_sha256,
            ),
        ):
            cached = cache.get(
                _cache_key(
                    descriptor.candidate,
                    descriptor.revision_sha256,
                    dynamic_paths_by_element=dynamic_paths_by_element,
                    recipe_fingerprint=recipe_fingerprint,
                )
            )
            if cached is None:
                raise SourceInferenceError("final source evidence disappeared from the private cache")
            selected_contributions = selected_by_candidate[descriptor]
            for cached_contribution in _cached_contributions(cached.evidence):
                current = selected_contributions.pop(
                    (cached_contribution.logical_source_id, cached_contribution.revision_sha256), None
                )
                if current is None:
                    continue
                for kind, payload in cached_contribution.evidence_by_element.items():
                    if not current:
                        payload = _historical_payload(payload)
                    evidence_by_element.setdefault(kind, SchemaEvidenceAccumulator()).add(
                        SchemaEvidence.from_json(payload)
                    )
            if selected_contributions:
                raise SourceInferenceError("final source evidence no longer matches the structure pass")

    collect_ms = (time.monotonic_ns() - collect_started) / 1_000_000
    if unique:
        terminal_counts["included"] += len(unique)
    producer_version_counts: Counter[str] = Counter()
    producer_version_missing_sources = 0
    producer_version_conflicting_sources = 0
    producer_version_unrecognized_sources = 0
    export_metadata: dict[tuple[str, str], tuple[tuple[str, ...], bool]] = {}
    for descriptor in final:
        export_metadata.setdefault(
            (descriptor.candidate.logical_source_id, descriptor.revision_sha256),
            (descriptor.producer_versions, descriptor.producer_version_unrecognized),
        )
    for versions, unrecognized in export_metadata.values():
        producer_version_counts.update(versions)
        producer_version_missing_sources += int(not versions)
        producer_version_conflicting_sources += int(len(versions) > 1)
        producer_version_unrecognized_sources += int(unrecognized)
    input_bytes = sum(input_bytes_by_candidate.values())
    record_count = sum(contribution.record_count for _descriptor, contribution in current_rows)
    report("reduce", force=True, records=record_count, input_bytes=input_bytes)
    return SourceInferenceResult(
        evidence_by_element={
            kind: (accumulator.finish().to_json(),) for kind, accumulator in sorted(evidence_by_element.items())
        },
        terminal_counts=dict(sorted(terminal_counts.items())),
        terminal_reason_counts=dict(sorted(terminal_reason_counts.items())),
        input_bytes=input_bytes,
        record_count=record_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        phase_timings_ms={"inventory": round(inventory_ms, 3), "collect": round(collect_ms, 3)},
        producer_version_counts=dict(sorted(producer_version_counts.items())),
        producer_version_missing_sources=producer_version_missing_sources,
        producer_version_conflicting_sources=producer_version_conflicting_sources,
        producer_version_unrecognized_sources=producer_version_unrecognized_sources,
        input_manifest_digest=hash_payload(
            {
                "recipe": _source_recipe_fingerprint(),
                "inputs": [
                    {"provider": descriptor.candidate.provider, "revision": contribution.revision_sha256}
                    for descriptor, contribution in sorted(
                        unique.values(),
                        key=lambda row: (row[0].candidate.provider, row[1].revision_sha256),
                    )
                ],
            }
        ),
    )


__all__ = [
    "SchemaSourceInput",
    "SourceContributionCache",
    "SourceInferenceError",
    "SourceInferenceResult",
    "SourceObservation",
    "SourceOutcome",
    "SourceRevision",
    "SourceTerminal",
    "default_schema_source_inputs",
    "infer_sources",
    "inventory_schema_sources",
    "parse_schema_source_input",
]
