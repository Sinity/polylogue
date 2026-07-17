"""Leaf adapters for named-source freshness CLI, MCP, and status surfaces.

The substrate projection remains in :mod:`polylogue.archive.query.source_freshness`.
These adapters intentionally avoid importing Click, Rich, or an MCP runtime so
the full checkout can register them without moving source-freshness semantics
into a leaf surface.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

from polylogue.archive.query.source_freshness import (
    NamedSourceFreshness,
    NamedSourceOperationalState,
    NamedSourceStage,
    ProjectionLimits,
    project_named_source_freshness,
)

_MCP_TOOL_NAME = "named_source_freshness"
_MCP_TOOL_DESCRIPTION = (
    "Trace one exact source path from filesystem/cursor evidence through "
    "raw, parse, index, FTS, and insight convergence."
)


class NamedSourceFreshnessStatusPayload(TypedDict):
    """Compact additive status record for one exact source."""

    source_path: str
    stage: str
    operational_state: str
    operational_reason: str
    source_exists: bool
    source_size_bytes: int | None
    source_mtime_ns: int | None
    source_stat_error: str | None
    cursor_observed_size_bytes: int | None
    cursor_byte_offset: int | None
    excluded: bool
    failure_count: int
    pending_bytes: int | None
    byte_lag: dict[str, object]
    unobserved_growth_bytes: int | None
    cursor_ahead_bytes: int | None
    observed_size_ahead_bytes: int | None
    reason: str | None
    accepted_raw_id: str | None
    accepted_by_acquisition: bool | None
    revision_authority: str | None
    revision_authority_owner: str
    replay_prevention_owner: str
    revision_application_count: int
    parse_state: str
    accepted_raw_indexed: bool
    broken_head: bool
    source_raw_scope_truncated: bool
    index_high_water_ms: int | None
    fts_converged: bool
    insights_converged: bool
    unsafe_scan_rejections: int
    projection_error_count: int
    projection_sha256: str


def source_freshness_status_payload(
    freshness: NamedSourceFreshness,
) -> NamedSourceFreshnessStatusPayload:
    """Return a compact additive record suitable for aggregate status payloads."""
    raw = freshness.accepted_raw_revision
    return {
        "source_path": freshness.source_path,
        "stage": freshness.stage.value,
        "operational_state": freshness.operational_state.value,
        "operational_reason": freshness.operational_reason.value,
        "source_exists": freshness.source_stat.exists,
        "source_size_bytes": freshness.source_stat.size_bytes,
        "source_mtime_ns": freshness.source_stat.mtime_ns,
        "source_stat_error": freshness.source_stat.error,
        "cursor_observed_size_bytes": freshness.cursor.observed_size_bytes,
        "cursor_byte_offset": freshness.cursor.byte_offset,
        "excluded": freshness.cursor.excluded,
        "failure_count": freshness.cursor.failure_count,
        "pending_bytes": freshness.cursor.pending_bytes,
        "byte_lag": freshness.byte_lag.to_dict(),
        "unobserved_growth_bytes": freshness.cursor.unobserved_growth_bytes,
        "cursor_ahead_bytes": freshness.cursor.cursor_ahead_bytes,
        "observed_size_ahead_bytes": freshness.cursor.observed_size_ahead_bytes,
        "reason": freshness.retry.reason,
        "accepted_raw_id": None if raw is None else raw.raw_id,
        "accepted_by_acquisition": None if raw is None else raw.accepted_by_acquisition,
        "revision_authority": None if raw is None else raw.revision_authority,
        "revision_authority_owner": freshness.ownership.raw_authority_owner,
        "replay_prevention_owner": freshness.ownership.replay_prevention_owner,
        "revision_application_count": len(freshness.revision_applications),
        "parse_state": freshness.parse.state,
        "accepted_raw_indexed": freshness.index.accepted_raw_indexed,
        "broken_head": freshness.index.broken_head,
        "source_raw_scope_truncated": freshness.index.source_raw_scope_truncated,
        "index_high_water_ms": freshness.index.high_water_ms,
        "fts_converged": freshness.fts.converged,
        "insights_converged": freshness.insights.converged,
        "unsafe_scan_rejections": len(freshness.receipt.unsafe_scan_rejections),
        "projection_error_count": len(freshness.errors),
        "projection_sha256": freshness.projection_sha256,
    }


def render_source_freshness_status(freshness: NamedSourceFreshness) -> str:
    """Render an operator-readable checkpoint receipt without archive totals."""
    raw = freshness.accepted_raw_revision
    reason = (
        freshness.retry.reason
        or freshness.source_stat.error
        or freshness.parse.error
        or freshness.index.reason
        or (freshness.fts.reason if not freshness.fts.converged else None)
        or (freshness.insights.reason if not freshness.insights.converged else None)
    )
    byte_lag = (
        str(freshness.byte_lag.value) if freshness.byte_lag.value_state == "known" else freshness.byte_lag.value_state
    )
    lines = [
        (
            f"{freshness.source_path}: stage={freshness.stage.value} "
            f"operational={freshness.operational_state.value} "
            f"operational_reason={freshness.operational_reason.value} "
            f"exists={str(freshness.source_stat.exists).lower()} "
            f"file_size={freshness.source_stat.size_bytes} "
            f"cursor_observed={freshness.cursor.observed_size_bytes} "
            f"cursor_offset={freshness.cursor.byte_offset} "
            f"excluded={str(freshness.cursor.excluded).lower()} "
            f"pending_bytes={byte_lag} "
            f"cursor_ahead_bytes={freshness.cursor.cursor_ahead_bytes}"
        ),
        (
            f"  byte_lag_evidence={freshness.byte_lag.freshness.state} "
            f"definition={freshness.byte_lag.definition_ref.format()}"
        ),
        (
            "  raw="
            + ("none" if raw is None else raw.raw_id)
            + f" authority={None if raw is None else raw.revision_authority}"
            + f" parse={freshness.parse.state} indexed={freshness.index.accepted_raw_indexed}"
            + f" broken_head={freshness.index.broken_head}"
        ),
        (
            f"  index_high_water_ms={freshness.index.high_water_ms} "
            f"fts={freshness.fts.converged} insights={freshness.insights.converged}"
        ),
        (
            f"  queries={freshness.receipt.query_count} "
            f"unsafe_scans={len(freshness.receipt.unsafe_scan_rejections)} "
            f"errors={len(freshness.errors)} receipt={freshness.projection_sha256}"
        ),
    ]
    if reason:
        lines.insert(1, f"  reason={reason}")
    return "\n".join(lines)


def make_source_freshness_mcp_handler(
    archive_root: Path | Callable[[], Path],
    *,
    cursor_export: Path | None = None,
    attempt_log: Path | None = None,
    limits: ProjectionLimits | None = None,
) -> Callable[..., Awaitable[dict[str, object]]]:
    """Build an async MCP handler while keeping registration in the leaf layer.

    Fallback evidence paths are owner-configured at handler construction. MCP
    callers can name only the source being diagnosed; they cannot turn this
    read-only tool into an arbitrary bounded file reader.
    """

    async def named_source_freshness(source_path: str) -> dict[str, object]:
        root = archive_root() if callable(archive_root) else archive_root
        projection = project_named_source_freshness(
            root,
            Path(source_path),
            cursor_export=cursor_export,
            attempt_log=attempt_log,
            limits=limits,
        )
        return projection.to_dict()

    named_source_freshness.__name__ = _MCP_TOOL_NAME
    named_source_freshness.__doc__ = _MCP_TOOL_DESCRIPTION
    return named_source_freshness


def register_source_freshness_mcp_tool(
    server: object,
    handler: Callable[..., Awaitable[object]],
) -> object:
    """Register the handler on a FastMCP-style server.

    The owning MCP module still controls tool inventory and contracts.  This
    adapter exists so that registration is one line once the full checkout's
    server module is available.
    """
    tool = getattr(server, "tool", None)
    if not callable(tool):
        raise TypeError("MCP server does not expose a callable tool() registrar")
    try:
        decorator = tool(name=_MCP_TOOL_NAME, description=_MCP_TOOL_DESCRIPTION)
    except TypeError:
        decorator = tool(name=_MCP_TOOL_NAME)
    return cast(Callable[[Callable[..., Any]], object], decorator)(handler)


def source_freshness_cli(argv: Sequence[str] | None = None) -> int:
    """Execute the integration-ready CLI adapter.

    The canonical query-first Click tree can call this function after parsing,
    while the module also remains directly executable for a live receipt.
    """
    parser = argparse.ArgumentParser(prog="polylogue-source-freshness")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cursor-export", type=Path)
    parser.add_argument("--attempt-log", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return 2 when the source is degraded or not searchable",
    )
    args = parser.parse_args(argv)
    projection = project_named_source_freshness(
        args.archive_root,
        args.source,
        cursor_export=args.cursor_export,
        attempt_log=args.attempt_log,
    )
    if args.format == "json":
        print(json.dumps(projection.to_dict(), sort_keys=True, indent=2))
    else:
        print(render_source_freshness_status(projection))
    if projection.receipt.unsafe_scan_rejections or projection.errors:
        return 3
    if args.strict and (
        projection.operational_state is NamedSourceOperationalState.DEGRADED
        or projection.stage is not NamedSourceStage.SEARCHABLE
    ):
        return 2
    return 0


def main() -> int:
    return source_freshness_cli()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NamedSourceFreshnessStatusPayload",
    "make_source_freshness_mcp_handler",
    "register_source_freshness_mcp_tool",
    "render_source_freshness_status",
    "source_freshness_cli",
    "source_freshness_status_payload",
]
