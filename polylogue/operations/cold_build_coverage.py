"""Cold-build promotion must not drop a session the active index serves.

An explicit ``polylogued run --cold-build-index`` builds a candidate over a
populated active generation. The candidate is filled by ordinary intake over
the configured source roots, so a session the active generation serves from a
raw outside that baseline (a manual import, a transcript deleted or rotated out
of its source root) never reaches it, and raw materialization stays suspended
while the build is unsettled. Promoting such a candidate would drop that session
from every read with no route re-deriving it. The daemon therefore promotes
through :func:`promote_cold_build_covering_active_index`, which refuses with
:class:`ColdBuildCoverageError` while any such session is missing.

This lives outside ``sources/live/cold_build.py`` on purpose: that module is in
the derived index identity closure, and a daemon settlement policy must not
move the index schema identity.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import WARNING, emit
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from polylogue.sources.live.cold_build import ColdBuildGeneration
    from polylogue.storage.index_generation import IndexGeneration

__all__ = [
    "ColdBuildCoverageError",
    "promote_cold_build_covering_active_index",
    "require_active_coverage",
]

#: Active sessions compared per round trip. A paging size only: every active
#: session is compared, whatever the archive holds.
_COVERAGE_PAGE_SIZE = 512


class ColdBuildCoverageError(RuntimeError):
    """The candidate lacks sessions the active generation serves from retained raws.

    The refusal leaves the candidate inactive and the active generation
    serving. Settlement re-evaluates it when the candidate or the source
    evidence changes.
    """

    def __init__(self, *, missing_count: int, first_missing_session_id: str) -> None:
        self.missing_count = missing_count
        self.first_missing_session_id = first_missing_session_id
        super().__init__(
            f"cold-build candidate lacks {missing_count} session(s) the active index serves from retained raws "
            f"(first: {first_missing_session_id})"
        )


def require_active_coverage(generation: ColdBuildGeneration) -> None:
    """Refuse a candidate that would drop a session the active generation serves.

    Compares every active session whose ``raw_id`` ``source.db`` still retains
    with the candidate, by ``session_id`` (origin plus native id), so a newer
    revision of the same source that the candidate accepted still covers the
    session. A session whose raw is no longer retained has no durable evidence
    any build could replay, so it is not a coverage obligation. The active
    generation is read inside one transaction, a consistent snapshot, a page
    at a time.

    Both index files are opened without schema validation: only
    ``sessions.session_id`` and ``sessions.raw_id`` are read, the candidate
    still has its deferred indexes dropped, and an explicit cold build is how
    an active generation at an older derived identity gets replaced.
    """
    archive_root = generation.archive_root
    active_path = resolve_active_index_path(archive_root)
    candidate_path = Path(generation.generation.index_path)
    if not active_path.exists():
        return
    if generation.promoted or active_path.resolve() == candidate_path.resolve():
        # Readers already resolve this candidate (an earlier promotion swapped
        # the pointer before failing). There is no other generation to cover,
        # and ``promote`` finishes that interrupted publication.
        return
    missing_count = 0
    first_missing: str | None = None
    with (
        closing(
            open_readonly_connection(
                active_path, tier=ArchiveTier.INDEX, validate_schema=False, timeout_class="background-read"
            )
        ) as active,
        closing(
            open_readonly_connection(
                candidate_path, tier=ArchiveTier.INDEX, validate_schema=False, timeout_class="background-read"
            )
        ) as candidate,
        closing(
            open_readonly_connection(
                archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
            )
        ) as source,
    ):
        active.execute("BEGIN")
        after = ""
        while True:
            rows = active.execute(
                "SELECT session_id, raw_id FROM sessions "
                "WHERE session_id > ? AND raw_id IS NOT NULL ORDER BY session_id LIMIT ?",
                (after, _COVERAGE_PAGE_SIZE),
            ).fetchall()
            if not rows:
                break
            after = str(rows[-1][0])
            raw_ids = tuple(dict.fromkeys(str(row[1]) for row in rows))
            retained = {
                str(row[0])
                for row in source.execute(
                    f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({','.join('?' for _ in raw_ids)})",
                    raw_ids,
                )
            }
            owed = tuple(str(row[0]) for row in rows if str(row[1]) in retained)
            if not owed:
                continue
            present = {
                str(row[0])
                for row in candidate.execute(
                    f"SELECT session_id FROM sessions WHERE session_id IN ({','.join('?' for _ in owed)})",
                    owed,
                )
            }
            for session_id in owed:
                if session_id not in present:
                    missing_count += 1
                    if first_missing is None:
                        first_missing = session_id
        active.execute("COMMIT")
    if first_missing is not None:
        emit(
            "daemon.cold_build.coverage_refused",
            level=WARNING,
            outcome="degraded",
            reason="active_coverage_incomplete",
            generation_id=generation.generation_id,
            sessions=missing_count,
        )
        raise ColdBuildCoverageError(missing_count=missing_count, first_missing_session_id=first_missing)


def promote_cold_build_covering_active_index(generation: ColdBuildGeneration) -> IndexGeneration:
    """Promote ``generation`` only if it serves every session the active index serves.

    One writer call runs the comparison and the promotion, so no archive write
    lands between them.
    """
    require_active_coverage(generation)
    return generation.promote()
