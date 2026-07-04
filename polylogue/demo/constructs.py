"""Declared construct coverage for the deterministic demo archive."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DemoConstruct:
    """A demo-world construct that must be present after seeding."""

    construct_id: str
    label: str
    description: str
    sql: str
    minimum: int = 1


@dataclass(frozen=True, slots=True)
class DemoConstructCoverage:
    """Measured coverage for one declared demo-world construct."""

    construct_id: str
    label: str
    observed: int
    minimum: int
    ok: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "construct_id": self.construct_id,
            "label": self.label,
            "observed": self.observed,
            "minimum": self.minimum,
            "ok": self.ok,
        }


DEMO_CONSTRUCTS: tuple[DemoConstruct, ...] = (
    DemoConstruct(
        construct_id="multi_origin_sessions",
        label="Multi-origin sessions",
        description="The demo archive contains at least ChatGPT, Claude Code, and Codex origins.",
        sql="SELECT COUNT(DISTINCT origin) FROM sessions",
        minimum=3,
    ),
    DemoConstruct(
        construct_id="session_profiles",
        label="Session profiles",
        description="Derived session-profile rows are materialized for read/postmortem surfaces.",
        sql="SELECT COUNT(*) FROM session_profiles",
        minimum=3,
    ),
    DemoConstruct(
        construct_id="tool_use_blocks",
        label="Tool-use blocks",
        description="Agent tool invocations are present for action and evidence demos.",
        sql="SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_use'",
    ),
    DemoConstruct(
        construct_id="tool_result_blocks",
        label="Tool-result blocks",
        description="Tool outcomes are present for action-result and claim-vs-evidence demos.",
        sql="SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_result'",
    ),
    DemoConstruct(
        construct_id="failed_tool_results",
        label="Failed tool results",
        description="At least one provider-reported failed tool result exists.",
        sql="SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_result' AND tool_result_is_error = 1",
    ),
    DemoConstruct(
        construct_id="provider_usage_messages",
        label="Provider usage messages",
        description="At least one message has token/cost lanes for usage demos.",
        sql=(
            "SELECT COUNT(*) FROM messages WHERE "
            "COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0) + "
            "COALESCE(cache_read_tokens, 0) + COALESCE(cache_write_tokens, 0) > 0"
        ),
    ),
    DemoConstruct(
        construct_id="attachment_rows",
        label="Attachment rows",
        description="At least one parsed attachment exists in the demo archive.",
        sql="SELECT COUNT(*) FROM attachments",
    ),
    DemoConstruct(
        construct_id="acquired_attachment_rows",
        label="Acquired attachment rows",
        description="At least one attachment has source-provided bytes stored in the blob store.",
        sql="SELECT COUNT(*) FROM attachments WHERE acquisition_status = 'acquired' AND blob_hash IS NOT NULL",
    ),
    DemoConstruct(
        construct_id="run_projection_rows",
        label="Run projection rows",
        description="Run-projection read models are populated for temporal demos.",
        sql="SELECT COUNT(*) FROM session_runs",
    ),
    DemoConstruct(
        construct_id="observed_event_rows",
        label="Observed-event rows",
        description="Observed events are populated for temporal/action analysis.",
        sql="SELECT COUNT(*) FROM session_observed_events",
    ),
    DemoConstruct(
        construct_id="context_snapshot_rows",
        label="Context snapshot rows",
        description="Session context snapshots are populated for resume/context demos.",
        sql="SELECT COUNT(*) FROM session_context_snapshots",
    ),
)


def evaluate_demo_constructs(archive_root: Path) -> tuple[DemoConstructCoverage, ...]:
    """Measure declared construct coverage against the demo index tier."""

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        rows: list[DemoConstructCoverage] = []
        for construct in DEMO_CONSTRUCTS:
            observed = int(conn.execute(construct.sql).fetchone()[0])
            rows.append(
                DemoConstructCoverage(
                    construct_id=construct.construct_id,
                    label=construct.label,
                    observed=observed,
                    minimum=construct.minimum,
                    ok=observed >= construct.minimum,
                )
            )
        return tuple(rows)
    finally:
        conn.close()


def construct_problem_messages(coverage: tuple[DemoConstructCoverage, ...]) -> tuple[str, ...]:
    """Return verifier problem strings for missing declared constructs."""

    return tuple(
        f"declared demo construct {row.construct_id!r} has {row.observed}, expected >= {row.minimum}"
        for row in coverage
        if not row.ok
    )


__all__ = [
    "DEMO_CONSTRUCTS",
    "DemoConstruct",
    "DemoConstructCoverage",
    "construct_problem_messages",
    "evaluate_demo_constructs",
]
