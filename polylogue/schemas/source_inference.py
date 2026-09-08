"""Explicit ordinary-source inventory and reduced-evidence collection.

The archive-backed sampler remains its own route.  This module observes
operator-selected source roots directly, retains only recipe-bound reduced
evidence in its private cache, and reports every terminal disposition.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDecodeError, JSONValue, is_json_value, loads
from polylogue.schemas.observation import extract_schema_units_from_payload, resolve_provider_config
from polylogue.schemas.source_cache import CachedContribution, SourceContributionCache
from polylogue.sources.decoder_zip import ZipBombError, ZipEntryValidator, open_bounded_zip_entry
from polylogue.sources.live.watcher import WatchSource, default_sources
from polylogue.sources.origin_specs import recognize_source_class
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

_RECIPE_VERSION = "source-evidence-v1"
_MAX_UNSTREAMABLE_DOCUMENT_BYTES = 32 * 1024 * 1024


class SourceInferenceError(RuntimeError):
    """Source inference could not form a complete source observation."""


@dataclass(frozen=True, slots=True)
class SchemaSourceInput:
    """One explicitly selected source root for a provider subject."""

    provider: str
    root: Path

    def __post_init__(self) -> None:
        Provider.from_string(self.provider)


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

    evidence_by_element: dict[str, tuple[dict[str, object], ...]]
    terminal_counts: dict[str, int]
    input_bytes: int
    record_count: int
    cache_hits: int
    cache_misses: int
    phase_timings_ms: dict[str, float]
    producer_version_counts: dict[str, int]
    producer_version_missing_sources: int
    producer_version_conflicting_sources: int
    input_manifest_digest: str

    def provenance(self) -> dict[str, object]:
        """Return aggregate-only source provenance safe for package metadata."""
        return {
            "source_input_bytes": self.input_bytes,
            "source_record_count": self.record_count,
            "source_cache_hits": self.cache_hits,
            "source_cache_misses": self.cache_misses,
            "source_terminal_outcomes": dict(sorted(self.terminal_counts.items())),
            "source_phase_timings_ms": dict(sorted(self.phase_timings_ms.items())),
            "producer_version_counts": dict(sorted(self.producer_version_counts.items())),
            "producer_version_missing_sources": self.producer_version_missing_sources,
            "producer_version_conflicting_sources": self.producer_version_conflicting_sources,
            "source_input_manifest_digest": self.input_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class _SourceCandidate:
    provider: str
    root: Path
    path: Path
    logical_source_id: str


@dataclass(frozen=True, slots=True)
class _CollectedCandidate:
    candidate: _SourceCandidate
    revision: SourceRevision | None
    terminal: SourceTerminal
    evidence_by_element: dict[str, dict[str, object]] | None = None
    producer_versions: tuple[str, ...] = ()


def parse_schema_source_input(value: str) -> SchemaSourceInput:
    """Parse the explicit ``provider=path`` command argument."""
    provider, separator, path_text = value.partition("=")
    if not separator or not provider or not path_text:
        raise ValueError("schema source inputs must use provider=path")
    return SchemaSourceInput(provider=Provider.from_string(provider).value, root=Path(path_text).expanduser())


def default_schema_source_inputs(*, provider: str) -> tuple[SchemaSourceInput, ...]:
    """Adapt executable watcher declarations without inventing path defaults."""
    provider_token = Provider.from_string(provider)
    rows: list[SchemaSourceInput] = []
    for source in default_sources():
        try:
            source_provider = Provider.from_string(source.name)
        except ValueError:
            continue
        if source_provider is provider_token:
            rows.append(SchemaSourceInput(provider=provider_token.value, root=source.root))
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
        provider = Provider.from_string(source_input.provider).value
        key = provider, _root_identity(source_input.root)
        existing = canonical.get(key)
        if existing is None or str(source_input.root) < str(existing.root):
            canonical[key] = SchemaSourceInput(provider=provider, root=source_input.root)
    return tuple(sorted(canonical.values(), key=lambda item: (item.provider, str(item.root))))


def inventory_schema_sources(inputs: Iterable[SchemaSourceInput]) -> tuple[_SourceCandidate, ...]:
    """Enumerate every admitted source candidate through watcher semantics."""
    candidates: list[_SourceCandidate] = []
    for source_input in _canonical_inputs(inputs):
        provider = Provider.from_string(source_input.provider)
        root = source_input.root
        watcher_source = WatchSource(
            name=provider.value,
            root=root,
            suffixes=(".json", ".jsonl", ".ndjson", ".zip", ".db", ".sqlite", ".sqlite3"),
        )
        root_identity = _root_identity(root)
        for path in _iter_source_entries(root):
            try:
                mode = os.stat(path, follow_symlinks=False).st_mode
            except OSError:
                continue
            if not stat.S_ISREG(mode) or not watcher_source.accepts(path):
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            candidates.append(
                _SourceCandidate(
                    provider=provider.value,
                    root=root,
                    path=path,
                    logical_source_id=f"{provider.value}:{root_identity}:{relative.as_posix()}",
                )
            )
    return tuple(sorted(candidates, key=lambda item: (item.provider, item.logical_source_id, str(item.path))))


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
    open_handle: object,
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
                for value in ijson.items(handle, prefix):
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
    return fallback


def _collect_payload_evidence(
    candidate: _SourceCandidate,
    revision: SourceRevision,
    payloads: Iterable[JSONValue],
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, object]], int, tuple[str, ...]]:
    """Reduce payloads as they stream, retaining no decoded source records."""
    from polylogue.schemas.generation.evidence import collect_source_evidence, merge_evidence

    config = resolve_provider_config(Provider.from_string(candidate.provider))
    evidence_rows: dict[str, object] = {}
    record_count = 0
    producer_versions: set[str] = set()
    logical_source_id = candidate.logical_source_id
    for payload_index, payload in enumerate(payloads):
        producer_versions.update(_declared_producer_versions(Provider.from_string(candidate.provider), (payload,)))
        declared_source_id = _native_source_id(
            Provider.from_string(candidate.provider),
            payload,
            f"{candidate.logical_source_id}:{payload_index}",
        )
        if declared_source_id != f"{candidate.logical_source_id}:{payload_index}":
            logical_source_id = declared_source_id
        units = extract_schema_units_from_payload(
            payload,
            source_name=Provider.from_string(candidate.provider),
            source_path=candidate.path,
            raw_id=f"{revision.revision_sha256}:{payload_index}",
            config=config,
            full_corpus=True,
        )
        for unit in units:
            contribution = collect_source_evidence(
                SourceObservation(
                    logical_source_id=logical_source_id,
                    revision_sha256=revision.revision_sha256,
                    subject=candidate.provider,
                    element_kind=unit.artifact_kind,
                    records=unit.schema_samples,
                ),
                dynamic_paths=dynamic_paths_by_element.get(unit.artifact_kind, ()),
            )
            prior = evidence_rows.get(unit.artifact_kind)
            evidence_rows[unit.artifact_kind] = contribution if prior is None else merge_evidence((prior, contribution))
            record_count += len(unit.schema_samples)
    payloads: dict[str, dict[str, object]] = {}
    for element_kind, row in sorted(evidence_rows.items()):
        payload = row.to_json()
        if not isinstance(payload, dict):
            raise SourceInferenceError("source evidence must serialize to a JSON object")
        payloads[element_kind] = payload
    return payloads, record_count, tuple(sorted(producer_versions))


def _declared_producer_versions(provider: Provider, payloads: Iterable[JSONValue]) -> tuple[str, ...]:
    """Extract only provider-declared release evidence, never generic keys."""
    versions: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if provider is Provider.CLAUDE_CODE:
            version = payload.get("version")
            if isinstance(version, str) and version:
                versions.add(version)
        elif provider is Provider.CODEX and payload.get("type") == "session_meta":
            session_payload = payload.get("payload")
            if isinstance(session_payload, dict):
                version = session_payload.get("cli_version")
                if isinstance(version, str) and version:
                    versions.add(version)
    return tuple(sorted(versions))


def _collect_candidate(
    candidate: _SourceCandidate,
    dynamic_paths_by_element: dict[str, tuple[str, ...]] | None = None,
) -> _CollectedCandidate:
    """Read one member fully and construct one-pass evidence observations."""
    dynamic_paths_by_element = dynamic_paths_by_element or {}
    provider = Provider.from_string(candidate.provider)
    recognition = recognize_source_class(provider, candidate.path)
    if recognition is not None and recognition.source_class != "session":
        return _CollectedCandidate(
            candidate,
            None,
            SourceTerminal(
                "intentionally_excluded" if recognition.source_class == "non_session" else "unsupported",
                reason=recognition.reason,
            ),
        )
    if candidate.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return _CollectedCandidate(
            candidate,
            None,
            SourceTerminal("unsupported", reason="sqlite_value_inference_not_supported"),
        )
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
        return _collect_zip_candidate(candidate, revision, dynamic_paths_by_element=dynamic_paths_by_element)
    try:
        evidence_by_element, record_count, producer_versions = _collect_payload_evidence(
            candidate,
            revision,
            _iter_file_payloads(candidate.path, byte_count=byte_count),
            dynamic_paths_by_element=dynamic_paths_by_element,
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
    if not evidence_by_element:
        return _CollectedCandidate(
            candidate, revision, SourceTerminal("unsupported", byte_count, reason="no_schema_units")
        )
    return _CollectedCandidate(
        candidate,
        revision,
        SourceTerminal("included", byte_count, record_count),
        evidence_by_element,
        producer_versions,
    )


def _collect_zip_candidate(
    candidate: _SourceCandidate,
    revision: SourceRevision,
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]],
) -> _CollectedCandidate:
    """Stream supported archive members without extracting them to a source root."""
    evidence_payloads: dict[str, list[dict[str, object]]] = {}
    byte_count = 0
    record_count = 0
    producer_versions: set[str] = set()
    try:
        with zipfile.ZipFile(candidate.path) as archive:
            validator = ZipEntryValidator(candidate.provider, cursor_state=None, zip_path=candidate.path)
            members = validator.filter_entries(archive.infolist(), allowed_suffixes=(".json", ".jsonl", ".ndjson"))
            for member in sorted(members, key=lambda item: item.filename):
                member_path = Path(member.filename)
                with open_bounded_zip_entry(archive, member) as member_handle:
                    member_digest_builder = hashlib.sha256()
                    for chunk in iter(lambda: member_handle.read(1024 * 1024), b""):
                        member_digest_builder.update(chunk)
                member_digest = member_digest_builder.hexdigest()
                member_revision = SourceRevision(
                    provider=revision.provider,
                    path=candidate.path,
                    logical_source_id=f"{candidate.logical_source_id}:zip:{member.filename}",
                    revision_sha256=member_digest,
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
                        with open_bounded_zip_entry(archive, member) as member_handle:
                            payloads, member_records, member_versions = _collect_payload_evidence(
                                member_candidate,
                                member_revision,
                                _iter_jsonl_payloads(member_handle),
                                dynamic_paths_by_element=dynamic_paths_by_element,
                            )
                    else:
                        payloads, member_records, member_versions = _collect_payload_evidence(
                            member_candidate,
                            member_revision,
                            _iter_document_payloads(
                                lambda member=member: open_bounded_zip_entry(archive, member),
                                member.filename,
                                byte_count=member.file_size,
                            ),
                            dynamic_paths_by_element=dynamic_paths_by_element,
                        )
                except (SourceInferenceError, ZipBombError, OSError) as exc:
                    return _CollectedCandidate(
                        candidate,
                        revision,
                        SourceTerminal("decode_failed", byte_count, reason=str(exc)),
                    )
                producer_versions.update(member_versions)
                if not payloads:
                    continue
                for element_kind, payload in payloads.items():
                    evidence_payloads.setdefault(element_kind, []).append(payload)
                byte_count += member.file_size
                record_count += member_records
    except (OSError, ZipBombError, zipfile.BadZipFile):
        return _CollectedCandidate(candidate, revision, SourceTerminal("decode_failed", reason="invalid_zip"))
    if not evidence_payloads:
        return _CollectedCandidate(candidate, revision, SourceTerminal("unsupported", reason="no_schema_zip_members"))
    from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence

    evidence_by_element: dict[str, dict[str, object]] = {}
    for element_kind, payloads in sorted(evidence_payloads.items()):
        evidence_payload = merge_evidence(SchemaEvidence.from_json(payload) for payload in payloads).to_json()
        if not isinstance(evidence_payload, dict):
            raise SourceInferenceError("source evidence must serialize to a JSON object")
        evidence_by_element[element_kind] = evidence_payload
    return _CollectedCandidate(
        candidate,
        revision,
        SourceTerminal("included", byte_count, record_count),
        evidence_by_element,
        tuple(sorted(producer_versions)),
    )


def _cache_key(
    candidate: _SourceCandidate,
    revision_sha256: str,
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]] | None,
) -> str:
    return hash_payload(
        {
            "recipe": _RECIPE_VERSION,
            "subject": candidate.provider,
            "revision_sha256": revision_sha256,
            "dynamic_paths": (
                {kind: list(paths) for kind, paths in sorted(dynamic_paths_by_element.items())}
                if dynamic_paths_by_element is not None
                else None
            ),
        }
    )


def _bounded_collected_candidates(
    executor: ProcessPoolExecutor,
    candidates: Iterable[_SourceCandidate],
    *,
    limit: int,
    dynamic_paths_by_element: dict[str, tuple[str, ...]],
) -> Iterator[_CollectedCandidate]:
    """Yield source-worker results in input order without an unbounded queue."""
    iterator = iter(candidates)
    pending = []
    for _ in range(limit):
        try:
            pending.append(executor.submit(_collect_candidate, next(iterator), dynamic_paths_by_element))
        except StopIteration:
            break
    while pending:
        future = pending.pop(0)
        yield future.result()
        with suppress(StopIteration):
            pending.append(executor.submit(_collect_candidate, next(iterator), dynamic_paths_by_element))


def _evidence_payloads_by_element(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    elements = payload.get("elements")
    if not isinstance(elements, dict):
        raise SourceInferenceError("cached source evidence has no element partition")
    result: dict[str, dict[str, object]] = {}
    for element_kind, evidence in elements.items():
        if not isinstance(element_kind, str) or not isinstance(evidence, dict):
            raise SourceInferenceError("cached source evidence element is invalid")
        result[element_kind] = evidence
    return result


def _merge_evidence_by_element(
    payloads: Iterable[dict[str, dict[str, object]]],
) -> dict[str, object]:
    """Merge only comparable artifact kinds; session and adjunct evidence stay apart."""
    groups: dict[str, list[dict[str, object]]] = {}
    for source_payload in payloads:
        for element_kind, payload in source_payload.items():
            groups.setdefault(element_kind, []).append(payload)
    from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence

    return {
        element_kind: merge_evidence(SchemaEvidence.from_json(payload) for payload in rows)
        for element_kind, rows in sorted(groups.items())
    }


def infer_sources(
    inputs: Iterable[SchemaSourceInput],
    *,
    cache_path: Path,
    max_workers: int = 2,
) -> SourceInferenceResult:
    """Collect cacheable reduced evidence from complete selected source roots.

    ``collect_source_evidence`` is deliberately imported in the worker after
    reading a stable member.  It consumes each record iterator once and
    returns the reduced projection supplied by the statistics owner.
    """
    started = time.monotonic_ns()
    discovered = inventory_schema_sources(inputs)
    # Equal immutable revisions are re-acquisitions of the same source
    # material. Select one deterministic physical representative before any
    # denominator-bearing evidence is collected.
    selected_by_revision: dict[str, _SourceCandidate] = {}
    for candidate in discovered:
        try:
            digest, _byte_count = _stable_file_digest(candidate.path)
        except (OSError, SourceInferenceError):
            continue
        previous = selected_by_revision.get(digest)
        if previous is None or (candidate.logical_source_id, str(candidate.path)) < (
            previous.logical_source_id,
            str(previous.path),
        ):
            selected_by_revision[digest] = candidate
    candidates = tuple(
        candidate
        for _digest, candidate in sorted(selected_by_revision.items(), key=lambda item: item[1].logical_source_id)
    )
    inventory_ms = (time.monotonic_ns() - started) / 1_000_000
    terminal_counts: Counter[str] = Counter()
    input_bytes = 0
    record_count = 0
    cache_hits = 0
    cache_misses = 0
    producer_version_counts: Counter[str] = Counter()
    producer_version_missing_sources = 0
    producer_version_conflicting_sources = 0
    collect_started = time.monotonic_ns()

    # A first reduced-evidence pass determines global dynamic-key paths.  Its
    # cache rows let a warm run derive exactly the same normalization policy
    # without reopening source payloads.  Changed members are read a second
    # time only when that policy is known, so field denominators never merge
    # values collected under incompatible normalization.
    preliminary_payloads: list[dict[str, dict[str, object]]] = []
    revisions: dict[_SourceCandidate, tuple[str, int]] = {}
    with SourceContributionCache(cache_path) as cache:
        preliminary_misses: list[_SourceCandidate] = []
        for candidate in candidates:
            try:
                revision_sha256, byte_count = _stable_file_digest(candidate.path)
            except SourceInferenceError:
                terminal_counts["changed_during_read"] += 1
                continue
            except OSError:
                terminal_counts["decode_failed"] += 1
                continue
            cached = cache.get(_cache_key(candidate, revision_sha256, dynamic_paths_by_element=None))
            if cached is None:
                preliminary_misses.append(candidate)
                continue
            cached_payloads = _evidence_payloads_by_element(cached.evidence)
            preliminary_payloads.append(cached_payloads)
            revisions[candidate] = revision_sha256, byte_count
            terminal_counts["included"] += 1
            input_bytes += byte_count
            record_count += cached.record_count
            cache_hits += 1
            versions = cached.metadata.get("producer_versions", [])
            if isinstance(versions, list) and all(isinstance(version, str) for version in versions):
                producer_version_counts.update(versions)
                producer_version_missing_sources += int(not versions)
                producer_version_conflicting_sources += int(len(versions) > 1)

        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for item in _bounded_collected_candidates(
                executor,
                preliminary_misses,
                limit=max(1, max_workers) * 2,
                dynamic_paths_by_element={},
            ):
                terminal_counts[item.terminal.outcome] += 1
                input_bytes += item.terminal.byte_count
                record_count += item.terminal.record_count
                if item.terminal.outcome != "included":
                    continue
                if item.revision is None or item.evidence_by_element is None:
                    raise SourceInferenceError("included source candidate has no preliminary evidence")
                revisions[item.candidate] = item.revision.revision_sha256, item.terminal.byte_count
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(
                            item.candidate, item.revision.revision_sha256, dynamic_paths_by_element=None
                        ),
                        evidence={"elements": item.evidence_by_element},
                        input_bytes=item.terminal.byte_count,
                        record_count=item.terminal.record_count,
                        metadata={"producer_versions": list(item.producer_versions)},
                    )
                )
                preliminary_payloads.append(item.evidence_by_element)
                cache_misses += 1
                producer_version_counts.update(item.producer_versions)
                producer_version_missing_sources += int(not item.producer_versions)
                producer_version_conflicting_sources += int(len(item.producer_versions) > 1)

        from polylogue.schemas.generation.dynamic_keys import dynamic_object_paths

        preliminary = _merge_evidence_by_element(preliminary_payloads)
        dynamic_paths_by_element = {
            element_kind: tuple(sorted(dynamic_object_paths(evidence.structure)))
            for element_kind, evidence in preliminary.items()
        }
        evidence_by_element: dict[str, list[dict[str, object]]] = {}
        final_misses: list[_SourceCandidate] = []
        for candidate, (revision_sha256, _byte_count) in sorted(
            revisions.items(), key=lambda item: item[0].logical_source_id
        ):
            cached = cache.get(
                _cache_key(candidate, revision_sha256, dynamic_paths_by_element=dynamic_paths_by_element)
            )
            if cached is None:
                final_misses.append(candidate)
            else:
                for element_kind, payload in _evidence_payloads_by_element(cached.evidence).items():
                    evidence_by_element.setdefault(element_kind, []).append(payload)
                cache_hits += 1

        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for item in _bounded_collected_candidates(
                executor,
                final_misses,
                limit=max(1, max_workers) * 2,
                dynamic_paths_by_element=dynamic_paths_by_element,
            ):
                if item.terminal.outcome != "included":
                    raise SourceInferenceError(
                        f"source changed or failed during final evidence pass: {item.terminal.outcome}"
                    )
                if item.revision is None or item.evidence_by_element is None:
                    raise SourceInferenceError("included source candidate has no final evidence")
                initial_revision = revisions.get(item.candidate)
                if initial_revision is None or initial_revision[0] != item.revision.revision_sha256:
                    raise SourceInferenceError("source revision changed between evidence passes")
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(
                            item.candidate,
                            item.revision.revision_sha256,
                            dynamic_paths_by_element=dynamic_paths_by_element,
                        ),
                        evidence={"elements": item.evidence_by_element},
                        input_bytes=item.terminal.byte_count,
                        record_count=item.terminal.record_count,
                        metadata={"producer_versions": list(item.producer_versions)},
                    )
                )
                for element_kind, payload in item.evidence_by_element.items():
                    evidence_by_element.setdefault(element_kind, []).append(payload)
                cache_misses += 1
    collect_ms = (time.monotonic_ns() - collect_started) / 1_000_000
    return SourceInferenceResult(
        evidence_by_element={kind: tuple(payloads) for kind, payloads in sorted(evidence_by_element.items())},
        terminal_counts=dict(sorted(terminal_counts.items())),
        input_bytes=input_bytes,
        record_count=record_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        phase_timings_ms={"inventory": round(inventory_ms, 3), "collect": round(collect_ms, 3)},
        producer_version_counts=dict(sorted(producer_version_counts.items())),
        producer_version_missing_sources=producer_version_missing_sources,
        producer_version_conflicting_sources=producer_version_conflicting_sources,
        input_manifest_digest=hash_payload(
            {
                "recipe": _RECIPE_VERSION,
                "inputs": [
                    {"provider": candidate.provider, "revision": revision_sha256}
                    for candidate, (revision_sha256, _byte_count) in sorted(
                        revisions.items(), key=lambda item: (item[0].provider, item[1][0])
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
