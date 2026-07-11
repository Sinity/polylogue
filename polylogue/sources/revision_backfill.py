"""Conservative replay of legacy raw rows into typed revision authority."""

from __future__ import annotations

import os
import pickle
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType

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


class RawRevisionReplayResourceBlockedError(RuntimeError):
    def __init__(self, raw_ids: list[str], limit_bytes: int) -> None:
        self.raw_ids = tuple(raw_ids)
        self.limit_bytes = limit_bytes
        super().__init__(f"{len(raw_ids)} raw revision(s) exceed replay limit {limit_bytes}")


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
    scanned = 0
    classified = 0
    quarantined = 0
    adoption_deferred = 0
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
        census_selection: tuple[str, ...] | None = None
        if selected_raw_ids is not None:
            census_selection, _keys = archive.expand_raw_membership_selection(selected_raw_ids)
        censused: set[str] = set()
        while True:
            rows = archive.raw_membership_census_rows(census_selection)
            pending_rows = [(raw_id, source_index) for raw_id, source_index in rows if raw_id not in censused]
            if max_payload_bytes is not None:
                oversized = [
                    raw_id
                    for raw_id, size in archive.raw_payload_sizes([raw_id for raw_id, _index in rows]).items()
                    if size > max_payload_bytes
                ]
                if oversized:
                    raise RawRevisionReplayResourceBlockedError(sorted(oversized), max_payload_bytes)
            for raw_id, source_index in pending_rows:
                scanned += 1
                censused.add(raw_id)
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
                    sessions, _payload_bytes = _parse_retained_raw(archive, raw_id)
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
                classified += int(len(sessions) == 1)
                spill.add(raw_id, sessions, payload_bytes=_payload_bytes)
                archive.replace_raw_membership_census(
                    raw_id,
                    sessions,
                    parser_fingerprint="revision-membership-v1",
                    censused_at_ms=0,
                )
            if census_selection is None:
                break
            expanded, _keys = archive.expand_raw_membership_selection(list(census_selection))
            if set(expanded) == set(census_selection):
                break
            census_selection = expanded

        _unclassified, selected_keys = archive.raw_revision_rebuild_selection(selected_raw_ids)
        logical_keys.update(selected_keys)
        _selected_membership_raws, membership_keys = archive.expand_raw_membership_selection(selected_raw_ids)

        replayed = 0
        for logical_key in sorted(logical_keys):
            plan = archive.classify_raw_revision_cohort(logical_key)
            if not plan.accepted_raw_ids:
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
                adoption_deferred += len(plan.accepted_raw_ids)
                continue
            archive.apply_raw_revision_replay(plan, parsed_by_raw_id, acquired_at_ms=0)
            replayed += 1

        for logical_key in membership_keys:
            if logical_key in logical_keys:
                continue
            member_sessions: dict[str, ParsedSession] = {}
            revisions: list[MembershipRevision] = []
            projections = {}
            retained_bytes = 0
            for raw_id, session, payload_bytes in spill.for_logical_key(logical_key):
                projection = session_revision_projection(session)
                member_sessions[raw_id] = session
                projections[raw_id] = projection
                revisions.append(MembershipRevision(raw_id, projection))
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
    return RevisionBackfillResult(scanned, classified, replayed, quarantined, adoption_deferred)


def _parse_retained_raw(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int]:
    provider, payload, source_path, _kind = archive.raw_revision_material(raw_id)
    return _parse_one(provider, payload, source_path), len(payload)


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
        )
    return parse_payload(
        provider,
        list(_iter_json_stream(BytesIO(payload), source_name)),
        fallback_id,
        source_path=source_path,
    )


__all__ = ["RawRevisionReplayResourceBlockedError", "RevisionBackfillResult", "backfill_historical_revision_evidence"]
