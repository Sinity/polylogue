"""Conservative replay of legacy raw rows into typed revision authority."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import is_stream_record_provider, parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(frozen=True, slots=True)
class RevisionBackfillResult:
    scanned: int
    classified_full: int
    replayed_logical_sources: int
    quarantined: int


def backfill_historical_revision_evidence(
    archive_root: Path,
    *,
    selected_raw_ids: list[str] | None = None,
) -> RevisionBackfillResult:
    """Parse retained legacy bytes, classify only single-session full captures."""
    scanned = 0
    classified = 0
    quarantined = 0
    logical_keys: set[str] = set()
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        unclassified, selected_keys = archive.raw_revision_rebuild_selection(selected_raw_ids)
        logical_keys.update(selected_keys)
        for raw_id, source_index in unclassified:
            scanned += 1
            if source_index < 0:
                quarantined += 1
                continue
            try:
                provider, payload, source_path, _kind = archive.raw_revision_material(raw_id)
                sessions = _parse_one(provider, payload, source_path)
            except Exception:
                quarantined += 1
                continue
            if len(sessions) != 1:
                quarantined += 1
                continue
            session = sessions[0]
            logical_key = f"{provider.value}:{session.provider_session_id}"
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    logical_source_key=logical_key,
                    kind=RawRevisionKind.FULL,
                    source_revision=raw_id,
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
            )
            logical_keys.add(logical_key)
            classified += 1

        replayed = 0
        for logical_key in sorted(logical_keys):
            plan = archive.classify_raw_revision_cohort(logical_key)
            if not plan.accepted_raw_ids:
                continue
            parsed_by_raw_id: dict[str, ParsedSession] = {}
            for raw_id in plan.accepted_raw_ids:
                provider, payload, source_path, _kind = archive.raw_revision_material(raw_id)
                sessions = _parse_one(provider, payload, source_path)
                if len(sessions) != 1:
                    raise RuntimeError(f"classified raw revision {raw_id} no longer parses to one session")
                parsed_by_raw_id[raw_id] = sessions[0]
            archive.apply_raw_revision_replay(plan, parsed_by_raw_id, acquired_at_ms=0)
            replayed += 1
    return RevisionBackfillResult(scanned, classified, replayed, quarantined)


def _parse_one(provider: Provider, payload: bytes, source_path: str) -> list[ParsedSession]:
    source_name = Path(source_path).name
    fallback_id = Path(source_path).stem
    if is_stream_record_provider(source_path, str(provider)):
        return parse_stream_payload(
            provider,
            _iter_json_stream(BytesIO(payload), source_name),
            fallback_id,
        )
    return parse_payload(
        provider,
        list(_iter_json_stream(BytesIO(payload), source_name)),
        fallback_id,
        source_path=source_path,
    )


__all__ = ["RevisionBackfillResult", "backfill_historical_revision_evidence"]
