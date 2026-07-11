"""Conservative replay of legacy raw rows into typed revision authority."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from polylogue.archive.session_revision_membership import MembershipRevision, classify_membership_revisions
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
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
    """Census every retained raw, then replay byte and bundle authority cohorts."""
    scanned = 0
    classified = 0
    quarantined = 0
    logical_keys: set[str] = set()
    parsed_by_raw: dict[str, list[ParsedSession]] = {}
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_id, source_index in archive.raw_membership_census_rows():
            scanned += 1
            if source_index < 0:
                archive.replace_raw_membership_census(
                    raw_id,
                    None,
                    parser_fingerprint="revision-membership-v1",
                    censused_at_ms=0,
                    detail="append fragments are governed by byte revision authority",
                )
                quarantined += 1
                continue
            try:
                provider, payload, source_path, _kind = archive.raw_revision_material(raw_id)
                sessions = _parse_one(provider, payload, source_path)
            except Exception as exc:
                archive.replace_raw_membership_census(
                    raw_id,
                    None,
                    parser_fingerprint="revision-membership-v1",
                    censused_at_ms=0,
                    detail=str(exc),
                )
                quarantined += 1
                continue
            parsed_by_raw[raw_id] = sessions
            archive.replace_raw_membership_census(
                raw_id,
                sessions,
                parser_fingerprint="revision-membership-v1",
                censused_at_ms=0,
            )

        _unclassified, selected_keys = archive.raw_revision_rebuild_selection(selected_raw_ids)
        logical_keys.update(selected_keys)
        classified = sum(1 for sessions in parsed_by_raw.values() if len(sessions) == 1)

        _selected_membership_raws, membership_keys = archive.expand_raw_membership_selection(selected_raw_ids)

        replayed = 0
        for logical_key in sorted(logical_keys):
            plan = archive.classify_raw_revision_cohort(logical_key)
            if not plan.accepted_raw_ids:
                continue
            parsed_by_raw_id: dict[str, ParsedSession] = {}
            for raw_id in plan.accepted_raw_ids:
                sessions = parsed_by_raw.get(raw_id, [])
                if len(sessions) != 1:
                    raise RuntimeError(f"classified raw revision {raw_id} no longer parses to one session")
                parsed_by_raw_id[raw_id] = sessions[0]
            archive.apply_raw_revision_replay(plan, parsed_by_raw_id, acquired_at_ms=0)
            replayed += 1

        for logical_key in membership_keys:
            if logical_key in logical_keys:
                continue
            member_sessions: dict[str, ParsedSession] = {}
            revisions: list[MembershipRevision] = []
            projections = {}
            for raw_id, sessions in parsed_by_raw.items():
                for session in sessions:
                    if f"{session.source_name.value}:{session.provider_session_id}" != logical_key:
                        continue
                    projection = session_revision_projection(session)
                    member_sessions[raw_id] = session
                    projections[raw_id] = projection
                    revisions.append(MembershipRevision(raw_id, projection))
            classification = classify_membership_revisions(revisions)
            if classification.ambiguous_raw_ids:
                quarantined += len(classification.ambiguous_raw_ids)
            archive.apply_raw_membership_classification(
                logical_key,
                classification,
                member_sessions,
                projections,
                acquired_at_ms=0,
            )
            if classification.accepted_raw_ids:
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
