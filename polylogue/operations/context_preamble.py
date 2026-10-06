"""Context preamble composition against a caller-pinned archive snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.analysis.resume import _rank_resume_profiles
from polylogue.archive.hydration import archive_envelope_to_session
from polylogue.context.preamble import build_context_preamble_payload
from polylogue.context.scheduler import ContextAssembly
from polylogue.core.async_bridge import complete_without_suspension
from polylogue.surfaces.payloads import AssertionClaimPayload, ContextPreamble, ContextPreambleProjectState

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(frozen=True, slots=True)
class ContextPreambleResult:
    payload: ContextPreamble | None
    ledger: ContextAssembly | None
    observed_at_ms: int

    def to_wire(self) -> dict[str, object]:
        return {
            "view": "context",
            "payload": self.payload.model_dump(mode="json", exclude_none=True) if self.payload is not None else None,
            "ledger": {
                "build_ref": self.ledger.build_ref,
                "ledger_rows": [row.as_dict() for row in self.ledger.ledger],
                "observed_at_ms": self.observed_at_ms,
            }
            if self.ledger is not None
            else None,
        }


class _PinnedPreambleReader:
    def __init__(self, archive: ArchiveStore, *, observed_at_ms: int) -> None:
        self.archive = archive
        self.observed_at_ms = observed_at_ms

    async def get_session(self, session_id: str) -> object | None:
        try:
            return archive_envelope_to_session(self.archive.read_session(session_id))
        except KeyError:
            return None

    async def compact_lineage(
        self, session_id: str, *, node_limit: int | None, edge_limit: int | None, include_accounting: bool
    ) -> object | None:
        return self.archive.read_compact_lineage(
            session_id, node_limit=node_limit, edge_limit=edge_limit, include_accounting=include_accounting
        )

    async def find_resume_candidates(
        self, *, repo_path: str, cwd: str | None, recent_files: tuple[str, ...], limit: int
    ) -> object:
        profiles = self.archive.list_session_profile_insights(sort="last-message", tier="merged", limit=None)
        return _rank_resume_profiles(profiles, repo_path=repo_path, cwd=cwd, recent_files=recent_files, limit=limit)

    async def list_assertion_claim_payloads(
        self, *, target_ref: str, statuses: tuple[str, ...], context_inject: bool, limit: int
    ) -> list[AssertionClaimPayload]:
        from polylogue.storage.sqlite.archive_tiers.user_write import (
            _ASSERTION_COLUMNS,
            ASSERTION_CLAIM_KINDS,
            _assertion_row_to_envelope,
        )

        self.archive.require_user_tier()
        connection = self.archive.index_connection
        if connection is None:
            raise ValueError("context preamble requires an index snapshot")
        kinds = tuple(str(kind.value) for kind in ASSERTION_CLAIM_KINDS)
        placeholders = ", ".join("?" for _ in kinds)
        status_placeholders = ", ".join("?" for _ in statuses)
        rows = connection.execute(
            f"SELECT {_ASSERTION_COLUMNS} FROM user_tier.assertions "
            f"WHERE kind IN ({placeholders}) AND target_ref = ? "
            f"AND COALESCE(status, 'active') IN ({status_placeholders}) "
            "AND (staleness_json IS NULL OR json_extract(staleness_json, '$.expires_at_ms') IS NULL "
            "OR json_extract(staleness_json, '$.expires_at_ms') > ?) "
            "ORDER BY updated_at_ms DESC, assertion_id",
            (*kinds, target_ref, *statuses, self.observed_at_ms),
        ).fetchall()
        claims = (_assertion_row_to_envelope(row) for row in rows)
        return [
            AssertionClaimPayload.from_envelope(claim)
            for claim in claims
            if bool(claim.context_policy.get("inject")) is context_inject
        ][:limit]


def execute_context_preamble(
    archive: ArchiveStore,
    *,
    session_id: str | None,
    observed_at: datetime,
    observed_project_state: tuple[ContextPreambleProjectState | None, str | None],
    cwd: str | None = None,
    repo_path: str | None = None,
    recent_files: tuple[str, ...] = (),
    related_limit: int = 5,
    require_session: bool = True,
    boundary: str = "session_start",
    token_budget: int | None = None,
    source_tool_calls: dict[str, str] | None = None,
) -> ContextPreambleResult:
    """Return a preamble and a ledger intent; the writer owns persistence.

    This runs on the thread that holds the pinned snapshot, which may be a
    compute worker already driving an event loop (the HTTP reader's nested
    admitted read). The shared builder is a coroutine only because the
    facade reader is asynchronous; over the pinned reader it never suspends,
    so it is driven to completion here rather than through a second loop.
    """

    ledger: ContextAssembly | None = None

    def retain(assembly: ContextAssembly) -> None:
        nonlocal ledger
        ledger = assembly

    payload = complete_without_suspension(
        build_context_preamble_payload(
            _PinnedPreambleReader(archive, observed_at_ms=int(observed_at.timestamp() * 1000)),
            session_id=session_id,
            related_limit=related_limit,
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            source_tool_calls=source_tool_calls,
            require_session=require_session,
            boundary=boundary,
            token_budget=token_budget,
            observed_project_state=observed_project_state,
            observed_at=observed_at,
            ledger_sink=retain,
        )
    )
    return ContextPreambleResult(payload, ledger, int(observed_at.timestamp() * 1000))


def execute_context_preamble_read(payload: dict[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Execute the declared machine read without resolving ambient state."""

    observed_at = datetime.fromisoformat(str(payload["observed_at"]))
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must include a timezone")
    project_raw = payload.get("observed_project_state")
    project = ContextPreambleProjectState.model_validate(project_raw) if project_raw is not None else None
    source_raw = payload.get("source_tool_calls")
    source = {str(key): str(value) for key, value in source_raw.items()} if isinstance(source_raw, dict) else None
    recent_raw = payload.get("recent_files", ())
    if not isinstance(recent_raw, (list, tuple)):
        raise ValueError("recent_files must be a list")
    raw_session_id = str(payload["session_id"]) if payload.get("session_id") is not None else None
    if raw_session_id is None:
        session_id = None
    else:
        try:
            session_id = archive.resolve_session_id(raw_session_id)
        except KeyError:
            # Optional context targets can be non-session assertion targets;
            # preserve their spelling when no session resolves.
            session_id = raw_session_id
    result = execute_context_preamble(
        archive,
        session_id=session_id,
        observed_at=observed_at,
        observed_project_state=(
            project,
            str(payload["project_failure"]) if payload.get("project_failure") else None,
        ),
        cwd=str(payload["cwd"]) if payload.get("cwd") is not None else None,
        repo_path=str(payload["repo_path"]) if payload.get("repo_path") is not None else None,
        recent_files=tuple(str(value) for value in recent_raw),
        related_limit=int(str(payload.get("related_limit", 5))),
        require_session=bool(payload.get("require_session", True)),
        boundary=str(payload.get("boundary", "session_start")),
        token_budget=int(str(payload["token_budget"])) if payload.get("token_budget") is not None else None,
        source_tool_calls=source,
    )
    return result.to_wire()
