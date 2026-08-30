"""Read-only normalized comparison for source-divergent blob residue.

This module is deliberately a census tool. It reads a frozen census JSON file,
routes the present source and retained blob through the same detector/parser
branches used by live full ingest, and writes an extended receipt. It never
opens a writable archive connection and never publishes a blob.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Any

from polylogue.archive.session_revision_membership import _relation
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.pipeline.ids import SessionRevisionProjection, session_revision_projection
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import (
    detect_provider_from_raw_bytes_evidence,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.live.batch import _STREAMING_FULL_INGEST_BYTES
from polylogue.sources.live.batch_support import (
    _detect_provider_from_path_sample,
    _parse_path_as_session_artifact,
    _parse_payload_as_session_artifact,
)
from polylogue.sources.parsers import codex_state, hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.sqlite_snapshot import is_sqlite_path, looks_like_sqlite_bytes
from polylogue.storage.blob_store import BlobStore


class ComparisonOutcome(StrEnum):
    REPRODUCED_NORMALIZED = "reproduced_normalized"
    SUPERSEDED_PREFIX = "superseded_prefix"
    CONTENT_DIVERGENT = "content_divergent"


@dataclass(frozen=True, slots=True)
class NormalizedSessionContribution:
    """One session's existing content-only comparison value."""

    provider_session_id: str
    projection: SessionRevisionProjection

    def to_dict(self) -> dict[str, object]:
        projection = self.projection
        return {
            "provider_session_id": self.provider_session_id,
            "message_count": sum(multiplicity for _identity, _content, multiplicity in projection.message_contents),
            "attachment_count": len(projection.attachment_identities),
            "event_count": len(projection.event_contents),
            "normalized_axes": {
                "messages": _axis_digest(
                    sorted(
                        (identity.hex(), content.hex(), multiplicity)
                        for identity, content, multiplicity in projection.message_contents
                    )
                ),
                "attachments": _axis_digest(
                    {
                        "identities": sorted(identity.hex() for identity in projection.attachment_identities),
                        "contents": sorted(
                            (identity.hex(), content.hex()) for identity, content in projection.attachment_contents
                        ),
                    }
                ),
                "session_events": _axis_digest(
                    sorted((identity.hex(), content.hex()) for identity, content in projection.event_contents)
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class NormalizedContribution:
    sessions: tuple[NormalizedSessionContribution, ...]

    @classmethod
    def from_sessions(cls, sessions: Iterable[ParsedSession]) -> NormalizedContribution:
        return cls(
            tuple(
                NormalizedSessionContribution(
                    provider_session_id=session.provider_session_id,
                    projection=session_revision_projection(session),
                )
                for session in sessions
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "session_count": len(self.sessions),
            "sessions": [session.to_dict() for session in self.sessions],
        }


@dataclass(frozen=True, slots=True)
class ContributionComparison:
    outcome: ComparisonOutcome
    differing_fields: tuple[str, ...] = ()
    extended_fields: tuple[str, ...] = ()
    unresolved: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "differing_fields": list(self.differing_fields),
            "extended_fields": list(self.extended_fields),
            "comparison_status": "unresolved" if self.unresolved else "resolved",
        }


@dataclass(frozen=True, slots=True)
class RouteResult:
    provider: Provider
    detector_evidence: str
    route: str
    sessions: tuple[ParsedSession, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "provider": self.provider.value,
            "detector_evidence": self.detector_evidence,
            "route": self.route,
            "status": "error" if self.error is not None else "accepted",
        }
        if self.error is not None:
            result["error"] = self.error[:4096]
        return result


def _axis_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_map(contribution: NormalizedContribution) -> dict[str, NormalizedSessionContribution]:
    return {session.provider_session_id: session for session in contribution.sessions}


def _differing_axes(
    stored: SessionRevisionProjection,
    current: SessionRevisionProjection,
) -> tuple[str, ...]:
    fields: list[str] = []
    if (
        stored.message_contents != current.message_contents
        or stored.mutable_message_identities != current.mutable_message_identities
    ):
        fields.append("messages")
    if (
        stored.attachment_identities != current.attachment_identities
        or stored.attachment_contents != current.attachment_contents
    ):
        fields.append("attachments")
    if stored.event_contents != current.event_contents:
        fields.append("session_events")
    return tuple(fields)


def compare_normalized_contributions(
    stored: NormalizedContribution,
    current: NormalizedContribution,
) -> ContributionComparison:
    """Classify two production-parser contributions.

    The current contribution is a superseding prefix only when it contains
    every stored session and each shared session is equal or contains the
    stored normalized axes. A missing session or a mixed growth direction is
    a content difference and is reported by field name.
    """
    stored_by_id = _session_map(stored)
    current_by_id = _session_map(current)
    differing: set[str] = set()
    extended: set[str] = set()

    if set(stored_by_id) - set(current_by_id):
        differing.add("sessions")
    if set(current_by_id) - set(stored_by_id):
        extended.add("sessions")

    for session_id in sorted(set(stored_by_id) & set(current_by_id)):
        stored_projection = stored_by_id[session_id].projection
        current_projection = current_by_id[session_id].projection
        relation = _relation(current_projection, stored_projection)
        if relation == "equal":
            continue
        if relation == "a_contains_b":
            extended.update(_differing_axes(stored_projection, current_projection))
        else:
            differing.update(_differing_axes(stored_projection, current_projection))
            if not _differing_axes(stored_projection, current_projection):
                differing.add("sessions")

    if differing:
        return ContributionComparison(
            ComparisonOutcome.CONTENT_DIVERGENT, tuple(sorted(differing)), tuple(sorted(extended))
        )
    if extended:
        return ContributionComparison(ComparisonOutcome.SUPERSEDED_PREFIX, (), tuple(sorted(extended)))
    return ContributionComparison(ComparisonOutcome.REPRODUCED_NORMALIZED)


def _stable_bytes(path: Path) -> tuple[bytes, dict[str, object]]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity:
        raise RuntimeError("source changed while being read")
    return payload, {
        "path": str(path.resolve(strict=False)),
        "device": before.st_dev,
        "inode": before.st_ino,
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _stable_stream_observation(path: Path, before: os.stat_result) -> dict[str, object]:
    """Hash a streamed parse input and reject a changed file boundary."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity:
        raise RuntimeError("source changed while being read")
    return {
        "path": str(path.resolve(strict=False)),
        "device": after.st_dev,
        "inode": after.st_ino,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "sha256": hasher.hexdigest(),
    }


def _fallback_provider(record: dict[str, Any]) -> Provider:
    origin = Origin.from_string(record.get("origin"))
    hint = record.get("capture_mode")
    return provider_from_origin(origin, family_hint=hint if isinstance(hint, str) else None)


def _parse_sqlite(path: Path, *, provider: Provider, fallback_id: str) -> RouteResult | None:
    if provider is Provider.HERMES and hermes_state.looks_like_state_db_path(path, immutable=True):
        return RouteResult(
            provider,
            "hermes_state.required_tables",
            "hermes_state.sqlite_snapshot",
            tuple(hermes_state.parse_state_db(path, fallback_id=fallback_id, profile_root=path.parent, immutable=True)),
        )
    if provider is Provider.HERMES and hermes_verification.looks_like_verification_evidence_db_path(
        path, immutable=True
    ):
        return RouteResult(
            provider,
            "hermes_verification.required_tables",
            "hermes_verification.sqlite_snapshot",
            tuple(
                hermes_verification.parse_verification_evidence_db(
                    path, fallback_id=fallback_id, profile_root=path.parent, immutable=True
                )
            ),
        )
    if provider is Provider.CODEX and codex_state.is_in_scope_codex_sqlite_path(path, immutable=True):
        kind = codex_state.classify_codex_sqlite_path(path, immutable=True)
        return RouteResult(provider, f"codex_state.{kind}", f"codex_state.{kind}.non_session", ())
    return None


def parse_production_route(
    path: Path, *, provider_hint: Provider, logical_path: Path | None = None
) -> tuple[RouteResult, dict[str, object]]:
    """Parse one file using the live detector/parser route without writes."""
    logical = logical_path or path
    if path.stat().st_size >= _STREAMING_FULL_INGEST_BYTES:
        return _parse_large_production_route(path, logical=logical, provider_hint=provider_hint)
    payload, observation = _stable_bytes(path)
    fallback_id = logical.stem
    if is_sqlite_path(logical) or looks_like_sqlite_bytes(payload):
        sqlite_result = _parse_sqlite(path, provider=provider_hint, fallback_id=fallback_id)
        if sqlite_result is not None:
            return sqlite_result, observation
        return RouteResult(
            provider_hint, "sqlite.unrecognized", "sqlite.refused", error="unrecognized SQLite source"
        ), observation

    size = len(payload)
    if size >= _STREAMING_FULL_INGEST_BYTES:
        provider, detector_evidence = detect_provider_from_raw_bytes_evidence(
            payload[:8192], logical.name, provider_hint, truncated_tail_ok=True
        )
    else:
        provider, detector_evidence = detect_provider_from_raw_bytes_evidence(payload, logical.name, provider_hint)
    if provider is not Provider.UNKNOWN and not _parse_payload_as_session_artifact(
        logical, provider=provider, payload=payload
    ):
        return RouteResult(
            provider,
            detector_evidence,
            "artifact.refused",
            error="production artifact admission refused session parsing",
        ), observation

    try:
        if is_stream_record_provider(str(logical), provider):
            sessions = parse_stream_payload(
                provider,
                _iter_json_stream(BytesIO(payload), logical.name, fail_on_decode_error=provider is Provider.UNKNOWN),
                fallback_id,
                source_path=str(logical),
            )
            route = "stream.parse"
        else:
            payloads = list(
                _iter_json_stream(BytesIO(payload), logical.name, fail_on_decode_error=provider is Provider.UNKNOWN)
            )
            sessions = parse_payload(provider, payloads, fallback_id, source_path=str(logical))
            route = "document.parse"
        admitted = require_positive_conversational_evidence(sessions, provider=provider, source_path=str(logical))
        if not admitted:
            return RouteResult(
                provider,
                detector_evidence,
                f"{route}.refused",
                error="positive conversational admission produced no sessions",
            ), observation
        return RouteResult(provider, detector_evidence, f"{route}.accepted", tuple(admitted)), observation
    except Exception as exc:
        return RouteResult(
            provider, detector_evidence, "parser.error", error=f"{type(exc).__name__}: {exc}"
        ), observation


def _parse_large_production_route(
    path: Path, *, logical: Path, provider_hint: Provider
) -> tuple[RouteResult, dict[str, object]]:
    """Run the production streaming branch without materializing the file."""
    before = path.stat()
    fallback_id = logical.stem
    provider = provider_hint
    detector_evidence = "path_sample"
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8192)
        if is_sqlite_path(logical) or looks_like_sqlite_bytes(prefix):
            sqlite_result = _parse_sqlite(path, provider=provider_hint, fallback_id=fallback_id)
            route = sqlite_result or RouteResult(
                provider_hint, "sqlite.unrecognized", "sqlite.refused", error="unrecognized SQLite source"
            )
            return route, _stable_stream_observation(path, before)

        if path == logical:
            provider = _detect_provider_from_path_sample(path, provider_hint)
        else:
            provider, detector_evidence = detect_provider_from_raw_bytes_evidence(
                prefix, logical.name, provider_hint, truncated_tail_ok=True
            )
        if (
            path == logical
            and provider is not Provider.UNKNOWN
            and not _parse_path_as_session_artifact(path, provider=provider)
        ):
            route = RouteResult(
                provider,
                detector_evidence,
                "artifact.refused",
                error="production artifact admission refused session parsing",
            )
            return route, _stable_stream_observation(path, before)

        with path.open("rb") as handle:
            payloads = _iter_json_stream(handle, logical.name, fail_on_decode_error=provider is Provider.UNKNOWN)
            if is_stream_record_provider(str(logical), provider):
                sessions = parse_stream_payload(provider, payloads, fallback_id, source_path=str(logical))
                route_name = "stream.parse"
            else:
                sessions = parse_payload(provider, list(payloads), fallback_id, source_path=str(logical))
                route_name = "document.parse"
        admitted = require_positive_conversational_evidence(sessions, provider=provider, source_path=str(logical))
        route = (
            RouteResult(
                provider,
                detector_evidence,
                f"{route_name}.accepted",
                tuple(admitted),
            )
            if admitted
            else RouteResult(
                provider,
                detector_evidence,
                f"{route_name}.refused",
                error="positive conversational admission produced no sessions",
            )
        )
    except Exception as exc:
        route = RouteResult(provider, detector_evidence, "parser.error", error=f"{type(exc).__name__}: {exc}")
    return route, _stable_stream_observation(path, before)


def _normalize_route(route: RouteResult) -> tuple[RouteResult, NormalizedContribution]:
    if route.error is not None:
        return route, NormalizedContribution.from_sessions(())
    try:
        return route, NormalizedContribution.from_sessions(route.sessions)
    except Exception as exc:
        failed = RouteResult(
            route.provider,
            route.detector_evidence,
            "normalized-contribution.error",
            error=f"{type(exc).__name__}: {exc}",
        )
        return failed, NormalizedContribution.from_sessions(())


def _cached_source_matches(path: Path, observation: dict[str, object]) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        stat.st_dev == observation.get("device")
        and stat.st_ino == observation.get("inode")
        and stat.st_size == observation.get("size_bytes")
        and stat.st_mtime_ns == observation.get("mtime_ns")
        and stat.st_ctime_ns == observation.get("ctime_ns")
    )


def _route_status(comparison: dict[str, Any], key: str) -> str:
    route = comparison.get(key)
    return str(route.get("status")) if isinstance(route, dict) else ""


def _candidate_result(
    record: dict[str, Any],
    *,
    blob_store: BlobStore,
    current_cache: dict[tuple[str, str], tuple[dict[str, object], NormalizedContribution, dict[str, object]]],
) -> dict[str, object]:
    blob_hash = record.get("blob_hash")
    source_value = record.get("recorded_source")
    if not isinstance(blob_hash, str) or not isinstance(source_value, str):
        raise ValueError("present-source census record lacks blob_hash or recorded_source")
    source_path = Path(source_value)
    blob_path = blob_store.blob_path(blob_hash)
    provider_hint = _fallback_provider(record)
    cache_key = (str(source_path), provider_hint.value)
    cached = current_cache.get(cache_key)
    if cached is not None and not _cached_source_matches(source_path, cached[2]):
        current_cache.pop(cache_key)
        cached = None
    if cached is None:
        parsed_current_route, source_observation = parse_production_route(
            source_path, provider_hint=provider_hint, logical_path=source_path
        )
        current_route, current_contribution = _normalize_route(parsed_current_route)
        current_route_data = current_route.to_dict()
        current_cache[cache_key] = (current_route_data, current_contribution, source_observation)
    else:
        current_route_data, current_contribution, source_observation = cached
    parsed_stored_route, blob_observation = parse_production_route(
        blob_path, provider_hint=provider_hint, logical_path=source_path
    )
    stored_route, stored_contribution = _normalize_route(parsed_stored_route)
    comparison = (
        ContributionComparison(
            ComparisonOutcome.CONTENT_DIVERGENT,
            ("normalized_contribution",),
            unresolved=True,
        )
        if current_route_data.get("status") == "error" or stored_route.error is not None
        else compare_normalized_contributions(stored_contribution, current_contribution)
    )
    return {
        **record,
        "normalized_comparison": {
            **comparison.to_dict(),
            "stored_route": stored_route.to_dict(),
            "current_route": current_route_data,
            "stored_contribution": stored_contribution.to_dict(),
            "current_contribution": current_contribution.to_dict(),
            "source_observation": source_observation,
            "blob_observation": blob_observation,
        },
        "disposition": comparison.outcome.value,
    }


def extend_census(census: dict[str, Any], *, blob_root: Path) -> dict[str, object]:
    """Extend a census, leaving the source-missing cohort byte-for-byte intact."""
    records = census.get("records")
    if not isinstance(records, list):
        raise ValueError("census records must be a list")
    present = [record for record in records if isinstance(record, dict) and record.get("cohort") != "source_missing"]
    if len(present) != 577:
        raise ValueError(f"expected 577 present-source candidates, found {len(present)}")
    from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

    origins = sorted({str(record.get("origin")) for record in present})
    code_identity = {
        "lowering_fingerprint": lowering_fingerprint(),
        "parser_fingerprints": {origin: parser_fingerprint_for_origin(origin) for origin in origins},
    }
    store = BlobStore(blob_root)
    extended_records: list[dict[str, object]] = []
    current_cache: dict[tuple[str, str], tuple[dict[str, object], NormalizedContribution, dict[str, object]]] = {}
    processed = 0
    for record in records:
        if not isinstance(record, dict) or record.get("cohort") == "source_missing":
            extended_records.append(record)
        else:
            extended_records.append(_candidate_result(record, blob_store=store, current_cache=current_cache))
            processed += 1
            if processed % 25 == 0 or processed == len(present):
                print(f"normalized-comparison progress={processed}/{len(present)}", flush=True)
    comparisons = [
        comparison
        for record in extended_records
        if isinstance((comparison := record.get("normalized_comparison")), dict)
    ]
    outcomes = Counter(str(comparison.get("outcome")) for comparison in comparisons)
    route_errors = sum(
        1
        for comparison in comparisons
        if _route_status(comparison, "stored_route") == "error" or _route_status(comparison, "current_route") == "error"
    )
    return {
        **census,
        "normalized_comparison": {
            "schema_version": 1,
            "input_candidate_hash_digest": census.get("candidate_hash_digest"),
            "present_source_candidate_count": len(present),
            "source_missing_candidate_count_untouched": sum(
                1 for record in records if isinstance(record, dict) and record.get("cohort") == "source_missing"
            ),
            "outcome_counts": dict(sorted(outcomes.items())),
            "route_error_count": route_errors,
            "read_only": True,
            "route": "production_detector_parser_admission_v1",
            "source_cache_unique_paths": len(current_cache),
            "code_identity": code_identity,
        },
        "records": extended_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blob-root", type=Path, required=True)
    args = parser.parse_args(argv)
    census = json.loads(args.census.read_text(encoding="utf-8"))
    receipt = extend_census(census, blob_root=args.blob_root)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    comparison = receipt["normalized_comparison"]
    counts = comparison["outcome_counts"] if isinstance(comparison, dict) else {}
    print(f"normalized-comparison present=577 outcomes={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ComparisonOutcome",
    "ContributionComparison",
    "NormalizedContribution",
    "NormalizedSessionContribution",
    "compare_normalized_contributions",
    "extend_census",
    "main",
    "parse_production_route",
]
