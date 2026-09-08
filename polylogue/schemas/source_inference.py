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
]

_RECIPE_VERSION = "source-evidence-v1"


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

    evidence_payloads: tuple[dict[str, object], ...]
    terminal_counts: dict[str, int]
    input_bytes: int
    record_count: int
    cache_hits: int
    cache_misses: int
    phase_timings_ms: dict[str, float]

    def provenance(self) -> dict[str, object]:
        """Return aggregate-only source provenance safe for package metadata."""
        return {
            "source_input_bytes": self.input_bytes,
            "source_record_count": self.record_count,
            "source_cache_hits": self.cache_hits,
            "source_cache_misses": self.cache_misses,
            "source_terminal_outcomes": dict(sorted(self.terminal_counts.items())),
            "source_phase_timings_ms": dict(sorted(self.phase_timings_ms.items())),
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
    evidence_payload: dict[str, object] | None = None


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


def _read_stable_bytes(path: Path) -> tuple[bytes, str, int]:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    before_identity = before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    after_identity = after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    if before_identity != after_identity or len(data) != before.st_size:
        raise SourceInferenceError("changed_during_read")
    return data, hashlib.sha256(data).hexdigest(), len(data)


def _records_from_jsonl(data: bytes) -> tuple[tuple[JSONValue, ...], SourceTerminal | None]:
    records: list[JSONValue] = []
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not line.endswith((b"\n", b"\r")) and index == len(lines) - 1:
            return (), SourceTerminal("partial_trailing_record", reason="trailing_jsonl_record_not_terminated")
        try:
            value = loads(line)
        except JSONDecodeError:
            return (), SourceTerminal("decode_failed", reason="malformed_jsonl_record")
        if not is_json_value(value):
            return (), SourceTerminal("decode_failed", reason="non_json_value")
        records.append(value)
    return tuple(records), None


def _payloads_from_member_bytes(path: Path, data: bytes) -> tuple[tuple[JSONValue, ...], SourceTerminal | None]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _records_from_jsonl(data)
    try:
        payload = loads(data)
    except JSONDecodeError:
        return (), SourceTerminal("decode_failed", reason="malformed_json")
    if not is_json_value(payload):
        return (), SourceTerminal("decode_failed", reason="non_json_value")
    return (payload,), None


def _observations_for_payloads(
    candidate: _SourceCandidate,
    revision: SourceRevision,
    payloads: Iterable[JSONValue],
) -> tuple[SourceObservation, ...]:
    config = resolve_provider_config(Provider.from_string(candidate.provider))
    observations: list[SourceObservation] = []
    for payload_index, payload in enumerate(payloads):
        units = extract_schema_units_from_payload(
            payload,
            source_name=Provider.from_string(candidate.provider),
            source_path=candidate.path,
            raw_id=f"{revision.revision_sha256}:{payload_index}",
            config=config,
            full_corpus=True,
        )
        for unit_index, unit in enumerate(units):
            observations.append(
                SourceObservation(
                    logical_source_id=f"{candidate.logical_source_id}:{payload_index}:{unit_index}",
                    revision_sha256=revision.revision_sha256,
                    subject=candidate.provider,
                    element_kind=unit.artifact_kind,
                    records=unit.schema_samples,
                )
            )
    return tuple(observations)


def _collect_candidate(candidate: _SourceCandidate) -> _CollectedCandidate:
    """Read one member fully and construct one-pass evidence observations."""
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
        data, digest, byte_count = _read_stable_bytes(candidate.path)
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
        return _collect_zip_candidate(candidate, revision, data)
    payloads, terminal = _payloads_from_member_bytes(candidate.path, data)
    if terminal is not None:
        return _CollectedCandidate(candidate, revision, terminal)
    observations = _observations_for_payloads(candidate, revision, payloads)
    if not observations:
        return _CollectedCandidate(
            candidate, revision, SourceTerminal("unsupported", byte_count, reason="no_schema_units")
        )
    record_count = sum(len(observation.records) for observation in observations)  # type: ignore[arg-type]
    return _CollectedCandidate(
        candidate,
        revision,
        SourceTerminal("included", byte_count, record_count),
        _collect_evidence_payload(observations),
    )


def _collect_zip_candidate(candidate: _SourceCandidate, revision: SourceRevision, data: bytes) -> _CollectedCandidate:
    """Stream supported archive members without extracting them to a source root."""
    del data  # zipfile reopens the verified path; stable read is checked again per member bytes.
    observations: list[SourceObservation] = []
    byte_count = 0
    record_count = 0
    try:
        with zipfile.ZipFile(candidate.path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            for member in sorted(members, key=lambda item: item.filename):
                member_path = Path(member.filename)
                if member_path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
                    continue
                member_data = archive.read(member)
                member_digest = hashlib.sha256(member_data).hexdigest()
                member_revision = SourceRevision(
                    provider=revision.provider,
                    path=candidate.path,
                    logical_source_id=f"{candidate.logical_source_id}:zip:{member.filename}",
                    revision_sha256=member_digest,
                    byte_count=len(member_data),
                )
                payloads, terminal = _payloads_from_member_bytes(member_path, member_data)
                if terminal is not None:
                    continue
                member_candidate = _SourceCandidate(
                    provider=candidate.provider,
                    root=candidate.root,
                    path=member_path,
                    logical_source_id=member_revision.logical_source_id,
                )
                member_observations = _observations_for_payloads(member_candidate, member_revision, payloads)
                observations.extend(member_observations)
                byte_count += len(member_data)
                record_count += sum(len(observation.records) for observation in member_observations)  # type: ignore[arg-type]
    except (OSError, zipfile.BadZipFile):
        return _CollectedCandidate(candidate, revision, SourceTerminal("decode_failed", reason="invalid_zip"))
    if not observations:
        return _CollectedCandidate(candidate, revision, SourceTerminal("unsupported", reason="no_schema_zip_members"))
    return _CollectedCandidate(
        candidate,
        revision,
        SourceTerminal("included", byte_count, record_count),
        _collect_evidence_payload(tuple(observations)),
    )


def _collect_evidence_payload(observations: tuple[SourceObservation, ...]) -> dict[str, object]:
    """Run the statistics-owned collector inside the bounded source worker."""
    from polylogue.schemas.generation.evidence import collect_source_evidence, merge_evidence

    evidence = merge_evidence(collect_source_evidence(observation) for observation in observations)
    payload = evidence.to_json()
    if not isinstance(payload, dict):
        raise SourceInferenceError("source evidence must serialize to a JSON object")
    return payload


def _cache_key(candidate: _SourceCandidate, revision_sha256: str) -> str:
    return hash_payload(
        {
            "recipe": _RECIPE_VERSION,
            "subject": candidate.provider,
            "revision_sha256": revision_sha256,
        }
    )


def _bounded_collected_candidates(
    executor: ProcessPoolExecutor,
    candidates: Iterable[_SourceCandidate],
    *,
    limit: int,
) -> Iterator[_CollectedCandidate]:
    """Yield source-worker results in input order without an unbounded queue."""
    iterator = iter(candidates)
    pending = []
    for _ in range(limit):
        try:
            pending.append(executor.submit(_collect_candidate, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.pop(0)
        yield future.result()
        with suppress(StopIteration):
            pending.append(executor.submit(_collect_candidate, next(iterator)))


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
    candidates = inventory_schema_sources(inputs)
    inventory_ms = (time.monotonic_ns() - started) / 1_000_000
    terminal_counts: Counter[str] = Counter()
    evidence_payloads: list[dict[str, object]] = []
    input_bytes = 0
    record_count = 0
    cache_hits = 0
    cache_misses = 0
    collect_started = time.monotonic_ns()

    # Hashing establishes immutable input identity before cache selection.
    # Cache hits bypass decode and field statistics; misses run those CPU steps
    # in bounded workers.  The coordinator alone owns SQLite transactions.
    misses: list[_SourceCandidate] = []
    with SourceContributionCache(cache_path) as cache:
        for candidate in candidates:
            try:
                _data, revision_sha256, byte_count = _read_stable_bytes(candidate.path)
            except SourceInferenceError:
                terminal_counts["changed_during_read"] += 1
                continue
            except OSError:
                terminal_counts["decode_failed"] += 1
                continue
            cached = cache.get(_cache_key(candidate, revision_sha256))
            if cached is None:
                misses.append(candidate)
                continue
            evidence_payloads.append(cached.evidence)
            terminal_counts["included"] += 1
            input_bytes += byte_count
            record_count += cached.record_count
            cache_hits += 1

        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            collected = _bounded_collected_candidates(executor, misses, limit=max(1, max_workers) * 2)
            for item in collected:
                terminal_counts[item.terminal.outcome] += 1
                input_bytes += item.terminal.byte_count
                record_count += item.terminal.record_count
                if item.terminal.outcome != "included":
                    continue
                if item.revision is None or item.evidence_payload is None:
                    raise SourceInferenceError("included source candidate has no evidence payload")
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(item.candidate, item.revision.revision_sha256),
                        evidence=item.evidence_payload,
                        input_bytes=item.terminal.byte_count,
                        record_count=item.terminal.record_count,
                    )
                )
                evidence_payloads.append(item.evidence_payload)
                cache_misses += 1
    collect_ms = (time.monotonic_ns() - collect_started) / 1_000_000
    return SourceInferenceResult(
        evidence_payloads=tuple(evidence_payloads),
        terminal_counts=dict(sorted(terminal_counts.items())),
        input_bytes=input_bytes,
        record_count=record_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        phase_timings_ms={"inventory": round(inventory_ms, 3), "collect": round(collect_ms, 3)},
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
