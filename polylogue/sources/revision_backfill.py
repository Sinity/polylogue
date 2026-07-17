"""Conservative replay of legacy raw rows into typed revision authority."""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
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
    adoption_deferred: int = 0


@dataclass(frozen=True, slots=True)
class RevisionCensusResult:
    scanned: int
    classified_full: int
    quarantined: int
    input_raw_ids: tuple[str, ...]
    logical_keys: tuple[str, ...]


@dataclass(slots=True)
class _RevisionCensusState:
    scanned: int
    classified: int
    quarantined: int
    censused: set[str]
    membership_candidates: dict[str, set[str]]
    provisional_full_raw_ids: dict[str, set[str]]


class RawRevisionReplayResourceBlockedError(RuntimeError):
    def __init__(self, raw_ids: list[str], limit_bytes: int, total_bytes: int) -> None:
        self.raw_ids = tuple(raw_ids)
        self.limit_bytes = limit_bytes
        self.total_bytes = total_bytes
        super().__init__(f"{len(raw_ids)} raw revision(s) total {total_bytes} bytes exceed replay limit {limit_bytes}")


def uncensused_historical_revision_raw_ids(archive_root: Path, raw_ids: list[str]) -> tuple[str, ...]:
    """Return inputs whose current parser identity has not been persisted.

    The dedicated receipt proves that the current parser actually observed
    every relevant raw. Durable revision or membership rows alone may have
    been produced by an older parser and therefore cannot establish current
    quiescence.
    """
    if not raw_ids:
        return ()
    placeholders = ",".join("?" for _ in raw_ids)
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            f"""
            SELECT r.raw_id
            FROM raw_sessions AS r
            LEFT JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
            WHERE r.raw_id IN ({placeholders})
              AND NOT COALESCE(
                  c.parser_fingerprint = 'revision-membership-v1'
                  AND c.status = 'complete',
                  0
              )
            ORDER BY r.raw_id
            """,
            raw_ids,
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _record_raw_authority_parser_census(archive_root: Path, raw_ids: tuple[str, ...]) -> None:
    """Persist per-raw current-parser completion without changing governance."""
    if not raw_ids:
        return
    with sqlite3.connect(archive_root / "source.db") as conn, conn:
        for raw_id in raw_ids:
            raw = conn.execute(
                """
                SELECT logical_source_key, revision_kind
                FROM raw_sessions WHERE raw_id = ?
                """,
                (raw_id,),
            ).fetchone()
            membership_census = conn.execute(
                """
                SELECT status, detail FROM raw_membership_census
                WHERE raw_id = ? AND parser_fingerprint = 'revision-membership-v1'
                """,
                (raw_id,),
            ).fetchone()
            membership_keys = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT logical_source_key FROM raw_session_memberships
                    WHERE raw_id = ? ORDER BY logical_source_key
                    """,
                    (raw_id,),
                )
            ]
            typed_key = (
                str(raw[0])
                if raw is not None and raw[0] is not None and str(raw[1]) != RawRevisionKind.UNKNOWN.value
                else None
            )
            complete = typed_key is not None or (
                membership_census is not None and str(membership_census[0]) in {"complete", "non_session"}
            )
            logical_keys = sorted(set(membership_keys) | ({typed_key} if typed_key is not None else set()))
            detail = (
                "current parser established durable authority identity"
                if complete
                else (
                    str(membership_census[1])
                    if membership_census is not None
                    else "current parser produced no durable authority identity"
                )
            )
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census (
                    raw_id, parser_fingerprint, status, logical_keys_json,
                    detail, censused_at_ms
                ) VALUES (?, 'revision-membership-v1', ?, ?, ?, 0)
                ON CONFLICT(raw_id) DO UPDATE SET
                    parser_fingerprint = excluded.parser_fingerprint,
                    status = excluded.status,
                    logical_keys_json = excluded.logical_keys_json,
                    detail = excluded.detail,
                    censused_at_ms = excluded.censused_at_ms
                """,
                (raw_id, "complete" if complete else "failed", json.dumps(logical_keys), detail),
            )


def _census_historical_revision_evidence(
    archive: ArchiveStore,
    spill: _ParsedSessionSpill,
    *,
    selected_raw_ids: list[str] | None,
    max_payload_bytes: int | None,
) -> _RevisionCensusState:
    """Persist a complete bounded parser census without mutating index.db."""
    state = _RevisionCensusState(0, 0, 0, set(), {}, {})
    census_selections: tuple[tuple[str, ...] | None, ...]
    if selected_raw_ids is None:
        census_selections = (None,)
    else:
        census_selections = archive.raw_membership_selection_components(selected_raw_ids)
    for initial_selection in census_selections:
        census_selection = initial_selection
        while True:
            rows = archive.raw_membership_census_rows(census_selection)
            pending_rows = [(raw_id, source_index) for raw_id, source_index in rows if raw_id not in state.censused]
            if max_payload_bytes is not None:
                payload_sizes = archive.raw_payload_sizes([raw_id for raw_id, _index in rows])
                total_payload_bytes = sum(payload_sizes.values())
                oversized = [raw_id for raw_id, size in payload_sizes.items() if size > max_payload_bytes]
                stream_safe = all(_retained_raw_is_stream_safe(archive, raw_id) for raw_id in payload_sizes)
                if (oversized or total_payload_bytes > max_payload_bytes) and not stream_safe:
                    blocked_ids = oversized or list(payload_sizes)
                    raise RawRevisionReplayResourceBlockedError(
                        sorted(blocked_ids), max_payload_bytes, total_payload_bytes
                    )
            for raw_id, source_index in pending_rows:
                state.scanned += 1
                state.censused.add(raw_id)
                if source_index < 0:
                    archive.replace_raw_membership_census(
                        raw_id,
                        None,
                        parser_fingerprint="revision-membership-v1",
                        censused_at_ms=0,
                        detail=BYTE_AUTHORITY_CENSUS_DETAIL,
                    )
                    state.quarantined += 1
                    continue
                try:
                    sessions, payload_bytes, revision_kind = _parse_retained_raw(archive, raw_id)
                except Exception as exc:
                    archive.replace_raw_membership_census(
                        raw_id,
                        None,
                        parser_fingerprint="revision-membership-v1",
                        censused_at_ms=0,
                        detail=str(exc),
                    )
                    state.quarantined += 1
                    continue
                state.classified += int(len(sessions) == 1)
                spill.add(raw_id, sessions, payload_bytes=payload_bytes)
                if len(sessions) == 1 and revision_kind is RawRevisionKind.UNKNOWN:
                    session = sessions[0]
                    logical_key = f"{session.source_name.value}:{session.provider_session_id}"
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
                    state.provisional_full_raw_ids.setdefault(logical_key, set()).add(raw_id)
                elif revision_kind is RawRevisionKind.UNKNOWN:
                    archive.replace_raw_membership_census(
                        raw_id,
                        sessions,
                        parser_fingerprint="revision-membership-v1",
                        censused_at_ms=0,
                    )
                    for session in sessions:
                        logical_key = f"{session.source_name.value}:{session.provider_session_id}"
                        state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
            if census_selection is None:
                break
            expanded, _keys = archive.expand_raw_membership_selection(list(census_selection))
            if set(expanded) == set(census_selection):
                break
            census_selection = expanded
    return state


def census_historical_revision_evidence(
    archive_root: Path,
    *,
    selected_raw_ids: list[str] | None = None,
    max_payload_bytes: int | None = None,
) -> RevisionCensusResult:
    """Complete the source-tier census stage without applying index changes."""
    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        _ParsedSessionSpill(archive_root) as spill,
    ):
        state = _census_historical_revision_evidence(
            archive,
            spill,
            selected_raw_ids=selected_raw_ids,
            max_payload_bytes=max_payload_bytes,
        )
        expanded, logical_keys = archive.expand_raw_membership_selection(selected_raw_ids)
    _record_raw_authority_parser_census(archive_root, tuple(expanded))
    return RevisionCensusResult(
        state.scanned,
        state.classified,
        state.quarantined,
        expanded,
        logical_keys,
    )


def backfill_historical_revision_evidence(
    archive_root: Path,
    *,
    selected_raw_ids: list[str] | None = None,
    owned_inactive_generation: tuple[str, str] | None = None,
    retention_observer: Callable[[int, int], None] | None = None,
    max_payload_bytes: int | None = None,
) -> RevisionBackfillResult:
    """Census every retained raw, then replay byte and bundle authority cohorts.

    Parser output is spilled beside the target archive during the census and
    loaded one logical authority cohort at a time. Peak retained session trees
    therefore follow the largest raw/cohort, not the archive-wide raw count.
    """
    adoption_deferred = 0
    quarantined = 0
    logical_keys: set[str] = set()
    archive_context = (
        ArchiveStore.open_owned_inactive_generation(
            archive_root,
            generation_id=owned_inactive_generation[0],
            owner_id=owned_inactive_generation[1],
        )
        if owned_inactive_generation is not None
        else ArchiveStore.open_existing(archive_root, read_only=False)
    )
    with archive_context as archive, _ParsedSessionSpill(archive_root) as spill:
        census = _census_historical_revision_evidence(
            archive,
            spill,
            selected_raw_ids=selected_raw_ids,
            max_payload_bytes=max_payload_bytes,
        )
        censused_raw_ids, _censused_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        # The direct backfill entry point must publish the same current-parser
        # receipt as the census-only entry point before it assigns or applies
        # any index plan. Commit the source census first so the separate
        # durable receipt writer observes one complete source snapshot.
        archive.commit()
        _record_raw_authority_parser_census(archive_root, tuple(censused_raw_ids))
        membership_candidates = census.membership_candidates
        provisional_full_raw_ids = census.provisional_full_raw_ids

        _unclassified, selected_keys = archive.raw_revision_rebuild_selection(selected_raw_ids)
        logical_keys.update(selected_keys)
        _selected_membership_raws, selected_membership_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        membership_keys = set(selected_membership_keys)

        replayed = 0
        byte_replayed_keys: set[str] = set()
        for logical_key in sorted(logical_keys):
            plan = archive.classify_raw_revision_cohort(logical_key)
            if not plan.accepted_raw_ids:
                # Complete snapshots that are not a unique byte-prefix chain
                # still carry semantic evidence. Move only that full-only
                # cohort to membership governance and let parsed-content
                # prefix rules decide it; append chains remain byte-governed.
                for raw_id in archive.convertible_full_revision_raw_ids(logical_key):
                    sessions, _payload_bytes = spill.for_raw(raw_id)
                    if len(sessions) != 1:
                        raise RuntimeError(f"full revision {raw_id} no longer parses to one session")
                    archive.replace_raw_membership_census(
                        raw_id,
                        sessions,
                        parser_fingerprint="revision-membership-v1",
                        censused_at_ms=0,
                        detail="historical non-prefix full revision governance",
                        retire_full_revision_governance=True,
                    )
                    membership_candidates.setdefault(logical_key, set()).add(raw_id)
                membership_keys.add(logical_key)
                continue
            parsed_by_raw_id: dict[str, ParsedSession] = {}
            retained_bytes = 0
            for raw_id in plan.accepted_raw_ids:
                sessions, payload_bytes = spill.for_raw(raw_id)
                if len(sessions) != 1:
                    raise RuntimeError(f"classified raw revision {raw_id} no longer parses to one session")
                parsed_by_raw_id[raw_id] = sessions[0]
                retained_bytes += payload_bytes
            if retention_observer is not None:
                retention_observer(len(parsed_by_raw_id), retained_bytes)
            accepted_sessions = [parsed_by_raw_id[raw_id] for raw_id in plan.accepted_raw_ids]
            if not archive.raw_revision_replay_adoptable(accepted_sessions):
                archive.defer_raw_revision_adoption(plan.logical_source_key, plan.accepted_raw_ids, accepted_sessions)
                provisional_raw_ids = provisional_full_raw_ids.get(logical_key, set())
                plan_raw_ids = {application.raw_id for application in plan.applications}
                if plan_raw_ids and plan_raw_ids <= provisional_raw_ids:
                    archive.release_provisional_full_revisions(sorted(plan_raw_ids))
                adoption_deferred += len(plan.accepted_raw_ids)
                continue
            archive.apply_raw_revision_replay(plan, parsed_by_raw_id, acquired_at_ms=0)
            replayed += 1
            byte_replayed_keys.add(logical_key)

        for logical_key in sorted(membership_keys):
            if logical_key in byte_replayed_keys:
                continue
            member_sessions: dict[str, ParsedSession] = {}
            revisions: list[MembershipRevision] = []
            projections = {}
            retained_bytes = 0
            candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
            candidate_raw_ids.update(membership_candidates.get(logical_key, ()))
            for raw_id, session, payload_bytes in spill.for_logical_key(logical_key):
                if raw_id not in candidate_raw_ids:
                    continue
                projection = session_revision_projection(session)
                member_sessions[raw_id] = session
                projections[raw_id] = projection
                revisions.append(MembershipRevision(raw_id, projection, session.updated_at))
                retained_bytes += payload_bytes
            if retention_observer is not None:
                retention_observer(len(member_sessions), retained_bytes)
            classification = classify_membership_revisions(revisions)
            if classification.ambiguous_raw_ids:
                quarantined += len(classification.ambiguous_raw_ids)
            accepted_sessions = [member_sessions[raw_id] for raw_id in classification.accepted_raw_ids]
            if accepted_sessions and not archive.raw_revision_replay_adoptable(accepted_sessions):
                archive.defer_raw_revision_adoption(
                    logical_key,
                    classification.accepted_raw_ids,
                    accepted_sessions,
                )
                adoption_deferred += len(classification.accepted_raw_ids)
                continue
            archive.apply_raw_membership_classification(
                logical_key,
                classification,
                member_sessions,
                projections,
                acquired_at_ms=0,
            )
            if classification.accepted_raw_ids:
                replayed += 1
    return RevisionBackfillResult(
        census.scanned,
        census.classified,
        replayed,
        census.quarantined + quarantined,
        adoption_deferred,
    )


def _parse_retained_raw(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
    provider, _blob_hash, source_path, kind, payload_size = archive.raw_revision_descriptor(raw_id)
    if is_stream_record_provider(source_path, str(provider)):
        with archive.open_raw_revision_material(raw_id) as (stream_provider, payload, stream_path, stream_kind):
            return _parse_stream(stream_provider, payload, stream_path), payload_size, stream_kind
    _provider, eager_payload, _source_path, _kind = archive.raw_revision_material(raw_id)
    return _parse_one(provider, eager_payload, source_path), len(eager_payload), kind


class _ParsedSessionSpill:
    """Disk-backed parser output cache bounded by one logical replay cohort."""

    def __init__(self, archive_root: Path) -> None:
        fd, name = tempfile.mkstemp(prefix=".revision-census-", suffix=".sqlite", dir=archive_root)
        os.close(fd)
        self.path = Path(name)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE parsed_sessions (
                raw_id TEXT NOT NULL,
                logical_key TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                parsed BLOB NOT NULL,
                PRIMARY KEY(raw_id, logical_key)
            ) STRICT
            """
        )
        self.conn.execute("CREATE INDEX parsed_sessions_logical ON parsed_sessions(logical_key, raw_id)")

    def __enter__(self) -> _ParsedSessionSpill:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.conn.close()
        self.path.unlink(missing_ok=True)

    def add(self, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT INTO parsed_sessions(raw_id, logical_key, payload_bytes, parsed) VALUES (?, ?, ?, ?)",
                (
                    (
                        raw_id,
                        f"{session.source_name.value}:{session.provider_session_id}",
                        payload_bytes,
                        pickle.dumps(session, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    for session in sessions
                ),
            )

    def for_raw(self, raw_id: str) -> tuple[list[ParsedSession], int]:
        rows = self.conn.execute(
            "SELECT parsed, payload_bytes FROM parsed_sessions WHERE raw_id = ? ORDER BY logical_key", (raw_id,)
        ).fetchall()
        return [pickle.loads(bytes(row[0])) for row in rows], int(rows[0][1]) if rows else 0

    def for_logical_key(self, logical_key: str) -> list[tuple[str, ParsedSession, int]]:
        rows = self.conn.execute(
            "SELECT raw_id, parsed, payload_bytes FROM parsed_sessions WHERE logical_key = ? ORDER BY raw_id",
            (logical_key,),
        ).fetchall()
        return [(str(row[0]), pickle.loads(bytes(row[1])), int(row[2])) for row in rows]


def _parse_one(provider: Provider, payload: bytes, source_path: str) -> list[ParsedSession]:
    source_name = Path(source_path).name
    fallback_id = Path(source_path).stem
    if is_stream_record_provider(source_path, str(provider)):
        return parse_stream_payload(
            provider,
            _iter_json_stream(BytesIO(payload), source_name),
            fallback_id,
            source_path=source_path,
        )
    return parse_payload(
        provider,
        list(_iter_json_stream(BytesIO(payload), source_name)),
        fallback_id,
        source_path=source_path,
    )


def _parse_stream(provider: Provider, payload: BinaryIO, source_path: str) -> list[ParsedSession]:
    source_name = Path(source_path).name
    fallback_id = Path(source_path).stem
    return parse_stream_payload(
        provider,
        _iter_json_stream(payload, source_name),
        fallback_id,
        source_path=source_path,
    )


def _retained_raw_is_stream_safe(archive: ArchiveStore, raw_id: str) -> bool:
    provider, _blob_hash, source_path, _kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    return is_stream_record_provider(source_path, str(provider))


__all__ = [
    "RawRevisionReplayResourceBlockedError",
    "RevisionBackfillResult",
    "RevisionCensusResult",
    "backfill_historical_revision_evidence",
    "census_historical_revision_evidence",
]
