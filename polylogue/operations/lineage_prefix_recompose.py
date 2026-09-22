"""Owner for the ``lineage_prefix_recompose`` convergence-debt backlog.

``storage/sqlite/archive_tiers/write.py`` records one convergence-debt row per
child whose recomposed lineage prefix this archive lost -- either because a
provider-session identity contradiction invalidated the edge that carried it,
or because a parent re-parse dropped the message the child had pinned as its
branch point. Both losses are named on one stage,
``IDENTITY_INVALIDATION_DEBT_STAGE``, because they have one remedy: re-derive
the child's inherited prefix from retained source evidence.

This module is that remedy's owner. It is deliberately **subject-scoped**: it
only ever inspects the sessions the debt ledger names. An archive-wide
corrective sweep over dangling branch points is forbidden -- it would repair
whatever ``_resolve_session_graph`` got wrong before anybody could see it, the
mask ``daemon/lineage_startup.py`` exists to refuse.

The re-derivation itself is the ordinary full-replay route
(``backfill_historical_revision_evidence``) scoped to the child's own retained
raw revision, so the write still lands through
``write_parsed_session_to_archive`` -- the single choke point shared by live
ingest and full replay. There is no second lineage write route here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import span
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.write import IDENTITY_INVALIDATION_DEBT_STAGE
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

#: The stage name the writer records its lineage-prefix losses under. Imported,
#: never restated: one spelling owns both the producer and this drain.
LINEAGE_PREFIX_RECOMPOSE_STAGE = IDENTITY_INVALIDATION_DEBT_STAGE

#: Payload budget for one child's retained raw component. A component larger
#: than this refuses with a named reason instead of replaying unbounded bytes
#: inside a convergence pass.
_RECOMPOSE_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

#: SQLite parameter chunk for the keyed debt lookup.
_SUBJECT_CHUNK = 256


class LineagePrefixRecomposeRefusedError(RuntimeError):
    """No requested child could be recomposed, and this names why for each.

    Raised rather than returned so the measured refusal reaches
    ``SessionState.last_error`` and therefore the debt row's ``last_error``.
    Returning ``False`` would record the engine's generic "session stage
    lineage_prefix_recompose returned False" over the diagnostic the writer
    recorded, which is exactly the degradation the drain's unimplemented-stage
    guard was added to prevent.
    """


def _ops_db_path(db_path: Path) -> Path:
    return db_path.with_name("ops.db")


def recorded_loss_subjects(db_path: Path, session_ids: Sequence[str]) -> set[str]:
    """Return the requested sessions this stage actually owns a debt row for.

    The index cannot distinguish "this child's prefix was extracted and then
    lost" from "this child's parent was never ingested": identity invalidation
    NULLs the same columns an edge that never resolved already carries. The
    debt row written at the moment of loss is the only record of which of the
    two happened, so it -- not a shape query over ``session_links`` -- is this
    stage's scope. A session with no row is not this stage's work, which keeps
    a generic ``convergence`` debt subject from minting a lineage row that
    would then never clear.
    """
    ordered = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    ops_db = _ops_db_path(db_path)
    if not ordered or not ops_db.exists():
        return set()
    conn = open_readonly_connection(ops_db, validate_schema=False)
    try:
        if (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'convergence_debt'").fetchone()
            is None
        ):
            return set()
        owned: set[str] = set()
        for start in range(0, len(ordered), _SUBJECT_CHUNK):
            chunk = ordered[start : start + _SUBJECT_CHUNK]
            rows = conn.execute(
                f"""SELECT target_id FROM convergence_debt
                    WHERE stage = ? AND target_type = 'session_id'
                      AND target_id IN ({",".join("?" for _ in chunk)})""",
                (LINEAGE_PREFIX_RECOMPOSE_STAGE, *chunk),
            ).fetchall()
            owned.update(str(row[0]) for row in rows)
        return owned
    finally:
        conn.close()


def unrecomposed_prefix_reason(conn: sqlite3.Connection, session_id: str) -> str | None:
    """Name how this child's prefix is still missing, or ``None`` when it is not.

    Both shapes compose the child to its own tail only: an edge with no
    resolved parent carries no prefix at all, and a composing prefix-sharing
    edge whose branch point names no message row dangles.
    """
    rows = conn.execute(
        """SELECT dst_origin, dst_native_id, resolved_dst_session_id, inheritance, branch_point_message_id
           FROM session_links WHERE src_session_id = ? AND status IS NULL""",
        (session_id,),
    ).fetchall()
    for row in rows:
        if row[2] is None:
            return "unresolved_parent_reference"
    for row in rows:
        if str(row[3] or "") == "prefix-sharing" and row[4] is not None:
            anchored = conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (str(row[4]),)).fetchone()
            if anchored is None:
                return "dangling_branch_point"
    return None


def _static_refusal(conn: sqlite3.Connection, session_id: str) -> str | None:
    """Refuse before replaying when retained evidence cannot settle the edge.

    A parent reference no session claims, or one two sessions still claim, is
    not repaired by re-parsing the child: the contradiction lives in the
    identity claims, not in the child's bytes. Naming it here keeps a
    permanently unrecomposable row from replaying its whole raw component on
    every pass.
    """
    rows = conn.execute(
        """SELECT dst_origin, dst_native_id FROM session_links
           WHERE src_session_id = ? AND status IS NULL AND resolved_dst_session_id IS NULL""",
        (session_id,),
    ).fetchall()
    for origin, native_id in rows:
        claimants = int(
            conn.execute(
                """SELECT COUNT(*) FROM session_identity_claims
                   WHERE origin = ? AND identity_namespace = 'provider-session' AND provider_value = ?""",
                (str(origin), str(native_id)),
            ).fetchone()[0]
        )
        if claimants == 0:
            return f"no session in the archive claims provider-session {native_id!r} on origin {origin!r}"
        if claimants > 1:
            return (
                f"provider-session {native_id!r} on origin {origin!r} is claimed by {claimants} sessions; "
                "the identity contradiction that truncated this child is unresolved"
            )
    return None


def _retained_raw_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    """The retained raw revision this child's stored content came from."""
    row = conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None or not row[0]:
        return ()
    return (str(row[0]),)


def recompose_session_prefix(archive_root: Path, index_path: Path, session_id: str) -> str | None:
    """Re-derive one child's prefix. Return ``None`` on success, else the reason.

    Success is decided by re-reading :func:`unrecomposed_prefix_reason` after
    the replay, never by the replay reporting that it ran. A best-effort
    partial recompose therefore cannot clear the row.
    """
    from polylogue.sources.revision_backfill import backfill_historical_revision_evidence

    conn = open_readonly_connection(index_path)
    try:
        if unrecomposed_prefix_reason(conn, session_id) is None:
            return None
        blocked = _static_refusal(conn, session_id)
        if blocked is not None:
            return blocked
        raw_ids = _retained_raw_ids(conn, session_id)
    finally:
        conn.close()
    if not raw_ids:
        return "no retained raw revision is bound to this session, so its prefix cannot be re-derived"
    try:
        backfill_historical_revision_evidence(
            archive_root,
            active_index_path=index_path,
            selected_raw_ids=list(raw_ids),
            max_payload_bytes=_RECOMPOSE_MAX_PAYLOAD_BYTES,
        )
    except Exception as exc:
        return f"replaying retained raw evidence failed: {type(exc).__name__}: {exc}"
    conn = open_readonly_connection(index_path)
    try:
        reason = unrecomposed_prefix_reason(conn, session_id)
    finally:
        conn.close()
    if reason is None:
        return None
    return f"retained raw evidence was replayed and the prefix is still missing ({reason})"


def make_lineage_prefix_recompose_stage(db_path: Path) -> ConvergenceStage:
    """Build the stage that drains recorded lineage-prefix losses."""
    archive_root = db_path.parent

    def _index_path() -> Path:
        return ArchiveLocation.resolve(archive_root).active_index_path

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        owned = recorded_loss_subjects(db_path, session_ids)
        if not owned:
            return set()
        index_path = _index_path()
        if not index_path.exists():
            return set()
        conn = open_readonly_connection(index_path)
        try:
            return {session_id for session_id in owned if unrecomposed_prefix_reason(conn, session_id) is not None}
        finally:
            conn.close()

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        ordered = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
        index_path = _index_path()
        with span("daemon.stage.execute", stage=LINEAGE_PREFIX_RECOMPOSE_STAGE, sessions=len(ordered)) as work:
            refused: dict[str, str] = {}
            recomposed = 0
            for session_id in ordered:
                reason = recompose_session_prefix(archive_root, index_path, session_id)
                if reason is None:
                    recomposed += 1
                else:
                    refused[session_id] = reason
            if not refused:
                work.ok(sessions=recomposed)
                return True
            work.degraded("prefix_unrecomposable", sessions=recomposed, refused=len(refused))
            if recomposed:
                # A mixed batch: let the engine's recheck clear exactly the
                # children that were recomposed. The refusals keep their row
                # and take the precise reason on the next (now homogeneous)
                # pass rather than dragging the successes down with them.
                return False
            raise LineagePrefixRecomposeRefusedError(
                "lineage prefix recompose refused: "
                + "; ".join(f"{session_id}: {reason}" for session_id, reason in sorted(refused.items()))
            )

    def check(_path: Path) -> bool:
        # This stage's subjects are sessions named by convergence debt, never
        # source files. Reporting work per watched file would make it an
        # archive-wide sweep.
        return False

    def execute(_path: Path) -> StageExecuteReturn:
        return True

    return ConvergenceStage(
        name=LINEAGE_PREFIX_RECOMPOSE_STAGE,
        description="Re-derive a child's lost lineage prefix from its retained raw evidence",
        check=check,
        execute=execute,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
    )


__all__ = [
    "LINEAGE_PREFIX_RECOMPOSE_STAGE",
    "LineagePrefixRecomposeRefusedError",
    "make_lineage_prefix_recompose_stage",
    "recompose_session_prefix",
    "recorded_loss_subjects",
    "unrecomposed_prefix_reason",
]
