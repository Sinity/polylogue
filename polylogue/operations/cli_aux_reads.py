"""Pinned product reads for CLI commands outside the main query grammar."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.operations.assertion_export import AssertionExportImages
    from polylogue.operations.mutation_transaction import MutationPrincipal

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def execute_cli_aux_read(
    name: str,
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    checkpoint: Callable[[], None] = lambda: None,
    assertion_exports: AssertionExportImages | None = None,
    principal: MutationPrincipal | None = None,
) -> dict[str, object]:
    """Execute the small CLI-only read contracts against the daemon's pinned reader."""
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
        archive.require_attached_user_tier()
        if assertion_exports is None or principal is None:
            raise ValueError("assertion export requires its resident selection owner")
        return assertion_exports.page(payload, archive=archive, principal=principal, checkpoint=checkpoint)

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


def _iter_sessions_from_source_path(archive: ArchiveStore, path: Path) -> Iterator[str]:
    attached = {str(row[1]) for row in archive._conn.execute("PRAGMA database_list")}
    if "source_tier" not in attached:
        raise ArchiveTierUnavailableError(
            tier="source.db",
            path=str(archive.source_db_path.resolve(strict=False)),
            reason="source tier is not attached to the pinned operation snapshot",
            guidance="restore or initialize the durable source tier, then retry the query; "
            "the reader will not open a replacement tier during an operation",
        )
    rows = archive._conn.execute(
        """SELECT s.session_id FROM sessions s
           JOIN source_tier.raw_sessions r ON r.raw_id = s.raw_id
           WHERE pl_path_prefix(r.source_path, ?)
           ORDER BY s.session_id""",
        (str(path),),
    )
    from polylogue.core.compute_cancel import check_compute_cancelled

    try:
        for row in rows:
            check_compute_cancelled()
            yield str(row[0])
    finally:
        rows.close()


__all__ = ["execute_cli_aux_read"]
