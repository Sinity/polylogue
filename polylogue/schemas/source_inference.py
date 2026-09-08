"""Explicit ordinary-source inventory and reduced-evidence collection.

The archive-backed sampler remains its own route.  This module observes
operator-selected source roots directly, retains only recipe-bound reduced
evidence in its private cache, and reports every terminal disposition.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import re
import stat
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from polylogue.core.enums import Provider
from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDecodeError, JSONDocument, JSONValue, is_json_value, loads
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

_RECIPE_VERSION = "source-evidence-v2"
_MAX_UNSTREAMABLE_DOCUMENT_BYTES = 32 * 1024 * 1024
_DECLARED_PRODUCER_VERSION = re.compile(r"v?\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z][0-9A-Za-z.-]{0,63})?")


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
    producer_version_unrecognized_sources: int
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
    evidence_by_element: dict[str, dict[str, object]]
    record_count: int
    declared_updated_at: tuple[int, str] | None


@dataclass(frozen=True, slots=True)
class _CollectedCandidate:
    candidate: _SourceCandidate
    revision: SourceRevision | None
    terminal: SourceTerminal
    contributions: tuple[_SourceContribution, ...] = ()
    producer_versions: tuple[str, ...] = ()
    producer_version_unrecognized: bool = False


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


def _candidate_native_identity(candidate: _SourceCandidate) -> str:
    """Scope one-record exports by their declared native session identity."""
    try:
        byte_count = candidate.path.stat().st_size
        payloads = _iter_file_payloads(candidate.path, byte_count=byte_count)
        first = next(payloads, None)
        second = next(payloads, None)
        if first is not None and second is None:
            native = _native_source_id(Provider.from_string(candidate.provider), first, "")
            if native:
                return native
    except (OSError, JSONDecodeError):
        pass
    return candidate.logical_source_id


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
) -> tuple[tuple[_SourceContribution, ...], int, tuple[str, ...], bool]:
    """Reduce each native source revision without retaining decoded records."""
    from polylogue.schemas.generation.evidence import collect_source_evidence, merge_evidence

    provider = Provider.from_string(candidate.provider)
    config = resolve_provider_config(provider)
    evidence_rows: dict[str, dict[str, object]] = {}
    record_counts: Counter[str] = Counter()
    update_keys: dict[str, tuple[int, str] | None] = {}
    producer_versions: set[str] = set()
    producer_version_unrecognized = False
    source_seen: set[tuple[str, str]] = set()
    header_source_id = candidate.logical_source_id
    for payload_index, payload in enumerate(payloads):
        versions, unrecognized = _declared_producer_versions(provider, (payload,))
        producer_versions.update(versions)
        producer_version_unrecognized = producer_version_unrecognized or unrecognized
        declared = _native_source_id(provider, payload, "")
        if declared:
            header_source_id = declared
        declared_source_id = hash_payload({"source": declared or header_source_id})
        update = _declared_update_key(provider, payload)
        if update is not None and (
            update_keys.get(declared_source_id) is None or update > update_keys[declared_source_id]
        ):
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
            contribution = collect_source_evidence(
                SourceObservation(
                    logical_source_id=declared_source_id,
                    revision_sha256=revision.revision_sha256,
                    subject=candidate.provider,
                    element_kind=unit.artifact_kind,
                    records=unit.schema_samples,
                ),
                dynamic_paths=dynamic_paths_by_element.get(unit.artifact_kind, ()),
            )
            source_key = declared_source_id, unit.artifact_kind
            if source_key in source_seen:
                contribution = replace(contribution, current_source_count=0)
            else:
                source_seen.add(source_key)
            by_kind = evidence_rows.setdefault(declared_source_id, {})
            prior = by_kind.get(unit.artifact_kind)
            by_kind[unit.artifact_kind] = contribution if prior is None else merge_evidence((prior, contribution))
            record_counts[declared_source_id] += len(unit.schema_samples)
    contributions: list[_SourceContribution] = []
    for source_id, rows in sorted(evidence_rows.items()):
        payload_by_element: dict[str, dict[str, object]] = {}
        for element_kind, row in sorted(rows.items()):
            payload = row.to_json()
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
        contributions, record_count, producer_versions, producer_version_unrecognized = _collect_payload_evidence(
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
                            )
                    else:
                        rows, member_records, versions, unrecognized = _collect_payload_evidence(
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


def _recipe_source(function: object) -> str:
    """Fingerprint executable reducer code without comment-only cache churn."""
    source = inspect.getsource(function)
    return ast.dump(ast.parse(source), annotate_fields=False, include_attributes=False)


def _source_recipe_fingerprint() -> str:
    from polylogue.schemas.generation.dynamic_keys import dynamic_object_paths
    from polylogue.schemas.generation.evidence import collect_source_evidence

    return hash_payload(
        {
            "version": _RECIPE_VERSION,
            "collector": _recipe_source(_collect_payload_evidence),
            "observation": _recipe_source(extract_schema_units_from_payload),
            "reducer": _recipe_source(collect_source_evidence),
            "emitter": _recipe_source(dynamic_object_paths),
        }
    )


def _serialize_contributions(contributions: Iterable[_SourceContribution]) -> dict[str, object]:
    return {
        "contributions": [
            {
                "source": item.logical_source_id,
                "revision": item.revision_sha256,
                "elements": item.evidence_by_element,
                "records": item.record_count,
                "updated": list(item.declared_updated_at) if item.declared_updated_at is not None else None,
            }
            for item in sorted(contributions, key=lambda item: (item.logical_source_id, item.revision_sha256))
        ]
    }


def _cached_contributions(payload: dict[str, object]) -> tuple[_SourceContribution, ...]:
    rows = payload.get("contributions")
    if not isinstance(rows, list):
        raise SourceInferenceError("cached source evidence has no contribution list")
    result: list[_SourceContribution] = []
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
        evidence_by_element: dict[str, dict[str, object]] = {}
        for kind, evidence in elements.items():
            if not isinstance(kind, str) or not isinstance(evidence, dict):
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
        result.append(_SourceContribution(source, revision, evidence_by_element, records, update_key))
    return tuple(result)


def _historical_payload(payload: dict[str, object]) -> dict[str, object]:
    from polylogue.schemas.generation.evidence import SchemaEvidence

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


def _merge_evidence_by_element(payloads: Iterable[dict[str, dict[str, object]]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for source_payload in payloads:
        for element_kind, payload in source_payload.items():
            groups.setdefault(element_kind, []).append(payload)
    from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence

    return {
        element_kind: merge_evidence(SchemaEvidence.from_json(payload) for payload in rows)
        for element_kind, rows in sorted(groups.items())
    }


def _cache_key(
    candidate: _SourceCandidate,
    revision_sha256: str,
    *,
    dynamic_paths_by_element: dict[str, tuple[str, ...]] | None,
) -> str:
    return hash_payload(
        {
            "recipe": _source_recipe_fingerprint(),
            "subject": candidate.provider,
            "revision_sha256": revision_sha256,
            "dynamic_paths": (
                {kind: list(paths) for kind, paths in sorted(dynamic_paths_by_element.items())}
                if dynamic_paths_by_element is not None
                else None
            ),
        }
    )


def infer_sources(
    inputs: Iterable[SchemaSourceInput],
    *,
    cache_path: Path,
    max_workers: int = 2,
    progress: Callable[[str, JSONDocument], None] | None = None,
) -> SourceInferenceResult:
    """Collect reduced evidence from complete source members and revisions."""
    started = time.monotonic_ns()
    candidates = inventory_schema_sources(inputs)
    inventory_ms = (time.monotonic_ns() - started) / 1_000_000
    terminal_counts: Counter[str] = Counter()
    cache_hits = 0
    cache_misses = 0
    completed = 0
    last_progress_ns = started

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
    preliminary: list[tuple[_SourceCandidate, str, int, tuple[_SourceContribution, ...], tuple[str, ...], bool]] = []
    with SourceContributionCache(cache_path) as cache:
        misses: list[tuple[_SourceCandidate, str, int]] = []
        for candidate in candidates:
            try:
                digest, byte_count = _stable_file_digest(candidate.path)
            except SourceInferenceError:
                terminal_counts["changed_during_read"] += 1
                continue
            except OSError:
                terminal_counts["decode_failed"] += 1
                continue
            cached = cache.get(_cache_key(candidate, digest, dynamic_paths_by_element=None))
            if cached is None:
                misses.append((candidate, digest, byte_count))
                continue
            preliminary.append(
                (
                    candidate,
                    digest,
                    byte_count,
                    _cached_contributions(cached.evidence),
                    tuple(value for value in cached.metadata.get("producer_versions", []) if isinstance(value, str)),
                    bool(cached.metadata.get("producer_version_unrecognized")),
                )
            )
            cache_hits += 1
            completed += 1
            report(
                "hash",
                records=sum(row.record_count for row in preliminary[-1][3]),
                input_bytes=sum(row[2] for row in preliminary),
            )
        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for item in _bounded_collected_candidates(
                executor,
                (candidate for candidate, _digest, _bytes in misses),
                limit=max(1, max_workers) * 2,
                dynamic_paths_by_element={},
            ):
                if item.terminal.outcome != "included" or item.revision is None:
                    terminal_counts[item.terminal.outcome] += 1
                    continue
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(
                            item.candidate, item.revision.revision_sha256, dynamic_paths_by_element=None
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
                preliminary.append(
                    (
                        item.candidate,
                        item.revision.revision_sha256,
                        item.terminal.byte_count,
                        item.contributions,
                        item.producer_versions,
                        item.producer_version_unrecognized,
                    )
                )
                cache_misses += 1
                completed += 1
                report(
                    "collect",
                    records=sum(
                        row.record_count
                        for _candidate, _digest, _bytes, rows, _versions, _unrecognized in preliminary
                        for row in rows
                    ),
                    input_bytes=sum(row[2] for row in preliminary),
                )
        from polylogue.schemas.generation.dynamic_keys import dynamic_object_paths

        all_preliminary = _merge_evidence_by_element(
            contribution.evidence_by_element
            for _candidate, _digest, _bytes, rows, _versions, _unrecognized in preliminary
            for contribution in rows
        )
        dynamic_paths_by_element = {
            kind: tuple(sorted(dynamic_object_paths(evidence.structure))) for kind, evidence in all_preliminary.items()
        }
        report(
            "reduce",
            force=True,
            records=sum(
                row.record_count
                for _candidate, _digest, _bytes, rows, _versions, _unrecognized in preliminary
                for row in rows
            ),
            input_bytes=sum(row[2] for row in preliminary),
        )
        final: list[tuple[_SourceCandidate, str, int, tuple[_SourceContribution, ...], tuple[str, ...], bool]] = []
        final_misses: list[tuple[_SourceCandidate, str, int]] = []
        for candidate, digest, byte_count, _rows, versions, unrecognized in preliminary:
            cached = cache.get(_cache_key(candidate, digest, dynamic_paths_by_element=dynamic_paths_by_element))
            if cached is None:
                final_misses.append((candidate, digest, byte_count))
                continue
            final.append(
                (candidate, digest, byte_count, _cached_contributions(cached.evidence), versions, unrecognized)
            )
            cache_hits += 1
        with ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for item in _bounded_collected_candidates(
                executor,
                (candidate for candidate, _digest, _bytes in final_misses),
                limit=max(1, max_workers) * 2,
                dynamic_paths_by_element=dynamic_paths_by_element,
            ):
                if item.terminal.outcome != "included" or item.revision is None:
                    raise SourceInferenceError(f"source failed during final evidence pass: {item.terminal.outcome}")
                expected = next(
                    (digest for candidate, digest, _bytes in final_misses if candidate == item.candidate), None
                )
                if expected != item.revision.revision_sha256:
                    raise SourceInferenceError("source revision changed between evidence passes")
                cache.put(
                    CachedContribution(
                        cache_key=_cache_key(
                            item.candidate,
                            item.revision.revision_sha256,
                            dynamic_paths_by_element=dynamic_paths_by_element,
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
                final.append(
                    (
                        item.candidate,
                        item.revision.revision_sha256,
                        item.terminal.byte_count,
                        item.contributions,
                        item.producer_versions,
                        item.producer_version_unrecognized,
                    )
                )
                cache_misses += 1
    collect_ms = (time.monotonic_ns() - collect_started) / 1_000_000
    all_rows = [
        (candidate, contribution)
        for candidate, _digest, _bytes, rows, _versions, _unrecognized in final
        for contribution in rows
    ]
    by_identity: dict[str, list[tuple[_SourceCandidate, _SourceContribution]]] = {}
    unique: dict[tuple[str, str], tuple[_SourceCandidate, _SourceContribution]] = {}
    for candidate, contribution in all_rows:
        key = contribution.logical_source_id, contribution.revision_sha256
        prior = unique.get(key)
        if prior is None or str(candidate.path) < str(prior[0].path):
            unique[key] = candidate, contribution
    for row in unique.values():
        by_identity.setdefault(row[1].logical_source_id, []).append(row)
    current_rows: list[tuple[_SourceCandidate, _SourceContribution]] = []
    historical_rows: list[tuple[_SourceCandidate, _SourceContribution]] = []
    for rows in by_identity.values():
        maximal = [
            row
            for row in rows
            if not any(_is_strict_file_prefix(row[0].path, other[0].path) for other in rows if other != row)
        ]
        selected = max(
            maximal,
            key=lambda row: (
                row[1].declared_updated_at is not None,
                row[1].declared_updated_at or (-1, ""),
                row[1].revision_sha256,
                str(row[0].path),
            ),
        )
        current_rows.append(selected)
        historical_rows.extend(row for row in rows if row != selected)
    evidence_by_element: dict[str, list[dict[str, object]]] = {}
    for _candidate, contribution in current_rows:
        for kind, payload in contribution.evidence_by_element.items():
            evidence_by_element.setdefault(kind, []).append(payload)
    for _candidate, contribution in historical_rows:
        for kind, payload in contribution.evidence_by_element.items():
            evidence_by_element.setdefault(kind, []).append(_historical_payload(payload))
    terminal_counts["included"] += len(unique)
    producer_version_counts: Counter[str] = Counter()
    producer_version_missing_sources = 0
    producer_version_conflicting_sources = 0
    producer_version_unrecognized_sources = 0
    # Producer metadata belongs to a physical export; count each immutable export once.
    export_metadata: dict[tuple[str, str], tuple[tuple[str, ...], bool]] = {}
    for candidate, digest, _bytes, _rows, versions, unrecognized in final:
        export_metadata.setdefault((candidate.logical_source_id, digest), (versions, unrecognized))
    for versions, unrecognized in export_metadata.values():
        producer_version_counts.update(versions)
        producer_version_missing_sources += int(not versions)
        producer_version_conflicting_sources += int(len(versions) > 1)
        producer_version_unrecognized_sources += int(unrecognized)
    input_bytes = sum(
        {
            (candidate.logical_source_id, digest): byte_count
            for candidate, digest, byte_count, _rows, _versions, _unrecognized in final
        }.values()
    )
    record_count = sum(contribution.record_count for _candidate, contribution in current_rows)
    report("reduce", force=True, records=record_count, input_bytes=input_bytes)
    return SourceInferenceResult(
        evidence_by_element={kind: tuple(rows) for kind, rows in sorted(evidence_by_element.items())},
        terminal_counts=dict(sorted(terminal_counts.items())),
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
                    {"provider": candidate.provider, "revision": contribution.revision_sha256}
                    for candidate, contribution in sorted(
                        unique.values(), key=lambda row: (row[0].provider, row[1].revision_sha256)
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
