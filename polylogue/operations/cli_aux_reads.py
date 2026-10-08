"""Pinned product reads for CLI commands outside the main query grammar."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import cast

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def execute_cli_aux_read(
    name: str, payload: Mapping[str, object], *, archive: ArchiveStore, checkpoint: Callable[[], None] = lambda: None
) -> dict[str, object]:
    """Execute the small CLI-only read contracts against the daemon's pinned reader."""
    from polylogue.surfaces.outcome import decide_outcome

    if name == "session.excision.plan":
        from polylogue.security.excision import LineageDependentsError, plan_session_excision

        session_id = str(payload["session_id"])
        cascade_lineage = bool(payload.get("cascade_lineage", False))
        try:
            plan = plan_session_excision(archive, session_id, cascade_lineage=cascade_lineage)
        except LineageDependentsError as exc:
            return {
                "found": True,
                "lineage_dependent_session_ids": list(exc.dependent_session_ids),
                "refused": True,
                "detail": str(exc),
                "plan": None,
            }
        return {
            "found": bool(plan.found),
            "lineage_dependent_session_ids": list(plan.lineage_dependent_session_ids),
            "refused": False,
            "detail": None,
            "plan": plan.as_dict(),
        }

    if name == "user.assertions.export":
        from polylogue.archive.query.transaction import archive_snapshot_epoch
        from polylogue.storage.sqlite.archive_tiers.user_write import (
            assertion_envelope_to_payload,
            count_assertions_for_export,
            iter_assertions_for_export,
        )

        archive.require_attached_user_tier()
        checkpoint()
        kinds = cast(list[str] | None, payload.get("kinds"))
        statuses = cast(list[str] | None, payload.get("statuses"))
        limit = cast(int | None, payload.get("limit"))
        offset = int(cast(int, payload.get("offset", 0)))
        page_size = int(cast(int, payload.get("page_size", 256)))
        total = count_assertions_for_export(archive._conn, kinds=kinds, statuses=statuses, schema="user_tier")
        if limit is not None:
            total = min(total, limit)
        items = []
        for row in iter_assertions_for_export(
            archive._conn,
            kinds=kinds,
            statuses=statuses,
            limit=min(page_size, max(0, total - offset)),
            offset=min(offset, total),
            schema="user_tier",
        ):
            checkpoint()
            items.append(assertion_envelope_to_payload(row))
        next_offset = offset + len(items)
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "next_offset": next_offset if next_offset < total else None,
            "snapshot_epoch": f"{archive.index_db_path.resolve()}:{archive_snapshot_epoch(archive)}",
            "outcome": decide_outcome(matched=total).to_dict(),
        }

    raise ValueError(f"unsupported CLI auxiliary read operation: {name}")


def _resolve_session_prefixes(archive: ArchiveStore, tokens: list[str]) -> list[str]:
    """Resolve reset targets using literal prefixes and preserve typo semantics."""
    if not archive.index_db_path.exists():
        return list(dict.fromkeys(tokens))
    resolved: list[str] = []
    for token in dict.fromkeys(tokens):
        exact = archive._conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (token,)).fetchone()
        if exact is not None:
            resolved.append(str(exact[0]))
            continue
        rows = archive._conn.execute(
            "SELECT session_id FROM sessions WHERE substr(session_id, 1, ?) = ? ORDER BY session_id LIMIT 2",
            (len(token), token),
        ).fetchall()
        if not rows:
            continue
        if len(rows) > 1:
            raise ValueError(f"session id prefix {token!r} is ambiguous")
        resolved.append(str(rows[0][0]))
    return list(dict.fromkeys(resolved))


def _sessions_from_source_path(archive: ArchiveStore, path: Path) -> tuple[str, ...]:
    attached = {str(row[1]) for row in archive._conn.execute("PRAGMA database_list")}
    if "source_tier" not in attached:
        raise ArchiveTierUnavailableError(
            tier="source.db",
            path=str(archive.source_db_path.resolve(strict=False)),
            reason="source tier is not attached to the pinned operation snapshot",
            guidance="restore or initialize the durable source tier, then retry the query; "
            "the reader will not open a replacement tier during an operation",
        )
    from polylogue.archive.query.path_prefix import escaped_sql_path_prefix_patterns

    exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(path)
    rows = archive._conn.execute(
        """SELECT s.session_id FROM sessions s
           JOIN source_tier.raw_sessions r ON r.raw_id = s.raw_id
           WHERE REPLACE(r.source_path, char(92), '/') = ?
              OR REPLACE(r.source_path, char(92), '/') LIKE ? ESCAPE '\\'
           ORDER BY s.session_id""",
        (exact_prefix, child_prefix),
    )
    from polylogue.core.compute_cancel import check_compute_cancelled

    def selected_ids() -> Iterator[str]:
        for row in rows:
            check_compute_cancelled()
            yield str(row[0])

    try:
        return tuple(selected_ids())
    finally:
        rows.close()


__all__ = ["execute_cli_aux_read"]
