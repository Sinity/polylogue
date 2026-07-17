"""Declared construct coverage for the deterministic demo archive."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.embeddings.materialization import archive_embeddable_message_where, message_prose_sql
from polylogue.storage.sqlite.run_projection_relations import (
    context_snapshot_relation_sql,
    observed_event_relation_sql,
    run_relation_sql,
)

# Canonical message prose SQL for embedding candidate selection (used in verification constructs).
_MESSAGE_PROSE_EMBEDDING = message_prose_sql("m", separator="char(10)||char(10)", block_types=("text",))


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
        construct_id="temporary_session_rows",
        label="Temporary session rows",
        description="At least one source-declared temporary session exists.",
        sql="SELECT COUNT(*) FROM sessions WHERE session_kind = 'temporary'",
    ),
    DemoConstruct(
        construct_id="token_budget_web_constructs",
        label="Token-budget web constructs",
        description="At least one provider-native token-budget construct is preserved.",
        sql="SELECT COUNT(*) FROM web_content_constructs WHERE construct_type = 'token_budget'",
    ),
    DemoConstruct(
        construct_id="capture_gap_events",
        label="Capture-gap events",
        description="At least one lower-precedence browser capture is recorded as a gap event.",
        sql="SELECT COUNT(*) FROM session_events WHERE event_type = 'capture_gap'",
    ),
    DemoConstruct(
        construct_id="browser_capture_raw_variants",
        label="Browser-capture raw variants",
        description="Multiple raw source captures for the same ChatGPT session remain durable source evidence.",
        sql="""
            SELECT COUNT(*)
            FROM source.raw_sessions
            WHERE origin = 'chatgpt-export'
              AND native_id = 'dc13ca54-0bba-4298-a38f-09068c2ef2c5'
        """,
        minimum=3,
    ),
    DemoConstruct(
        construct_id="browser_capture_coalesced_session",
        label="Browser-capture coalesced session",
        description="Direct export and browser captures coalesce into one canonical indexed ChatGPT session.",
        sql="""
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE s.session_id = 'chatgpt-export:dc13ca54-0bba-4298-a38f-09068c2ef2c5'
              AND EXISTS (
                  SELECT 1
                  FROM source.raw_sessions AS r
                  WHERE r.raw_id = s.raw_id
              )
              AND (
                  SELECT COUNT(*)
                  FROM sessions AS duplicates
                  WHERE duplicates.origin = 'chatgpt-export'
                    AND duplicates.native_id = 'dc13ca54-0bba-4298-a38f-09068c2ef2c5'
              ) = 1
        """,
    ),
    DemoConstruct(
        construct_id="source_outage_interval_events",
        label="Source-outage interval events",
        description=(
            "A capture adapter declares a bounded interval during which it was not observing the "
            "session, distinct from ordinary conversational silence, which leaves no signal at all."
        ),
        sql="""
            SELECT COUNT(*)
            FROM session_events
            WHERE event_type = 'source_outage'
              AND json_extract(payload_json, '$.started_at') IS NOT NULL
              AND json_extract(payload_json, '$.ended_at') IS NOT NULL
        """,
    ),
    DemoConstruct(
        construct_id="ambiguous_cross_material_duplicate",
        label="Ambiguous cross-material duplicate",
        description=(
            "The same logical conversation content arrives via two materials that do not share a "
            "native id, so occurrence-identity tooling must resolve the duplicate from content."
        ),
        sql="""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT b1.text
                FROM blocks AS b1
                JOIN messages AS m1 ON m1.message_id = b1.message_id AND m1.session_id = b1.session_id
                JOIN blocks AS b2
                  ON b2.text = b1.text
                 AND b2.block_type = 'text'
                JOIN messages AS m2 ON m2.message_id = b2.message_id AND m2.session_id = b2.session_id
                WHERE b1.block_type = 'text'
                  AND m1.session_id = 'chatgpt-export:cross-material-duplicate-01'
                  AND m2.session_id = 'chatgpt-export:cross-material-duplicate-02'
            )
        """,
    ),
    DemoConstruct(
        construct_id="compaction_omits_failed_attempt",
        label="Compaction omits a failed attempt",
        description=(
            "A structurally failed tool result precedes a compaction boundary whose summary text "
            "omits any mention of the failure, so compaction-honesty demos must diff full session "
            "evidence against the summary rather than trusting the summary alone."
        ),
        sql="""
            SELECT CASE WHEN
                EXISTS (
                    SELECT 1 FROM actions
                    WHERE session_id = 'claude-code-session:63705dcc-f3e5-4378-8118-8bc21e53bbb6:agent-acompact-demo'
                      AND is_error = 1
                )
                AND EXISTS (
                    SELECT 1 FROM session_events
                    WHERE session_id = 'claude-code-session:63705dcc-f3e5-4378-8118-8bc21e53bbb6:agent-acompact-demo'
                      AND event_type = 'compaction'
                      AND LOWER(summary) NOT LIKE '%fail%'
                      AND LOWER(summary) NOT LIKE '%error%'
                )
            THEN 1 ELSE 0 END
        """,
    ),
    DemoConstruct(
        construct_id="session_link_rows",
        label="Session-link rows",
        description="At least one parser-declared parent relationship is persisted.",
        sql="SELECT COUNT(*) FROM session_links",
    ),
    DemoConstruct(
        construct_id="generic_branch_links",
        label="Generic branch links",
        description="Ambiguous parent links are preserved without over-claiming fork-vs-resume semantics.",
        sql="SELECT COUNT(*) FROM session_links WHERE link_type = 'branch'",
    ),
    DemoConstruct(
        construct_id="prefix_sharing_links",
        label="Prefix-sharing lineage links",
        description="At least one child stores only its divergent tail and composes the inherited prefix.",
        sql="SELECT COUNT(*) FROM session_links WHERE inheritance = 'prefix-sharing'",
    ),
    DemoConstruct(
        construct_id="continuation_links",
        label="Continuation links",
        description="At least one auto-compaction or continuation relationship is persisted as a typed link.",
        sql="SELECT COUNT(*) FROM session_links WHERE link_type = 'continuation'",
    ),
    DemoConstruct(
        construct_id="subagent_links",
        label="Subagent links",
        description="At least one spawned subagent relationship is persisted as a typed link.",
        sql="SELECT COUNT(*) FROM session_links WHERE link_type = 'subagent'",
    ),
    DemoConstruct(
        construct_id="sidechain_sessions",
        label="Sidechain sessions",
        description="At least one provider-declared sidechain session is preserved as typed session state.",
        sql="SELECT COUNT(*) FROM sessions WHERE branch_type = 'sidechain'",
    ),
    DemoConstruct(
        construct_id="compaction_events",
        label="Compaction events",
        description="At least one provider-declared compaction boundary is preserved as a session event.",
        sql="SELECT COUNT(*) FROM session_events WHERE event_type = 'compaction'",
    ),
    DemoConstruct(
        construct_id="run_projection_rows",
        label="Run projection rows",
        description="Run-projection read models are populated for temporal demos.",
        sql=f"{run_relation_sql()}\nSELECT COUNT(*) FROM runs",
    ),
    DemoConstruct(
        construct_id="observed_event_rows",
        label="Observed-event rows",
        description="Observed events are populated for temporal/action analysis.",
        sql=f"{observed_event_relation_sql(source_where='1')}\nSELECT COUNT(*) FROM observed_events",
    ),
    DemoConstruct(
        construct_id="context_snapshot_rows",
        label="Context snapshot rows",
        description="Session context snapshots are populated for resume/context demos.",
        sql=f"{context_snapshot_relation_sql()}\nSELECT COUNT(*) FROM context_snapshots",
    ),
    DemoConstruct(
        construct_id="subagent_context_snapshots",
        label="Subagent context snapshots",
        description="Context snapshots expose the subagent-start boundary.",
        sql=f"{context_snapshot_relation_sql()}\nSELECT COUNT(*) FROM context_snapshots WHERE boundary = 'subagent_start'",
    ),
    DemoConstruct(
        construct_id="subagent_run_rows",
        label="Subagent run rows",
        description="Parent-side subagent executions are preserved as distinct runs from child main runs.",
        sql=f"{run_relation_sql()}\nSELECT COUNT(*) FROM runs WHERE role = 'subagent'",
    ),
    DemoConstruct(
        construct_id="unfinished_terminal_state_rows",
        label="Unfinished terminal-state rows",
        description="At least one session profile exposes an unfinished structural terminal state for resume demos.",
        sql="SELECT COUNT(*) FROM session_profiles WHERE terminal_state IN ('question_left', 'tool_left')",
    ),
    DemoConstruct(
        construct_id="error_terminal_state_rows",
        label="Error terminal-state rows",
        description="At least one session profile exposes an unrecovered structural error boundary.",
        sql="SELECT COUNT(*) FROM session_profiles WHERE terminal_state = 'error_left'",
    ),
    DemoConstruct(
        construct_id="receipts_failed_test_action",
        label="Receipts failed test action",
        description="The Evidence Lab incident contains a structurally failed pytest action before the claim.",
        sql="""
            SELECT COUNT(*)
            FROM actions
            WHERE session_id = 'codex-session:demo-receipts'
              AND is_error = 1
              AND exit_code = 1
              AND tool_input LIKE '%pytest tests/test_clock.py%'
        """,
    ),
    DemoConstruct(
        construct_id="receipts_successful_recovery_action",
        label="Receipts successful recovery action",
        description="The incident later contains a structurally successful verification run.",
        sql="""
            SELECT COUNT(*)
            FROM actions
            WHERE session_id = 'codex-session:demo-receipts'
              AND is_error = 0
              AND exit_code = 0
              AND tool_input LIKE '%pytest tests/test_clock.py%'
        """,
    ),
    DemoConstruct(
        construct_id="receipts_conflicting_claim",
        label="Receipts conflicting claim",
        description="Assistant prose claims success after the failed run and before the successful recovery.",
        sql="""
            SELECT COUNT(*)
            FROM blocks
            WHERE session_id = 'codex-session:demo-receipts'
              AND block_type = 'text'
              AND text = 'All tests pass. The clock fix is complete.'
        """,
    ),
    DemoConstruct(
        construct_id="anti_grep_control",
        label="Anti-grep negative control",
        description="The word error appears in prose while the control session contains zero failed actions.",
        sql="""
            SELECT CASE WHEN
                EXISTS (
                    SELECT 1 FROM blocks
                    WHERE session_id = 'codex-session:demo-anti-grep'
                      AND block_type = 'text'
                      AND LOWER(text) LIKE '%error%'
                )
                AND NOT EXISTS (
                    SELECT 1 FROM actions
                    WHERE session_id = 'codex-session:demo-anti-grep'
                      AND is_error = 1
                )
            THEN 1 ELSE 0 END
        """,
    ),
    DemoConstruct(
        construct_id="embedding_candidate_prose_messages",
        label="Embedding candidate prose messages",
        description="Authored prose rows exist for the paid embedding selector without counting tool/protocol rows.",
        sql=f"""
            SELECT COUNT(*)
            FROM (
                SELECT m.message_id, {_MESSAGE_PROSE_EMBEDDING} AS text
                FROM messages AS m
                JOIN blocks AS b
                  ON b.session_id = m.session_id
                 AND b.message_id = m.message_id
                 AND b.block_type = 'text'
                 AND b.text IS NOT NULL
                WHERE {archive_embeddable_message_where("m")}
                GROUP BY m.message_id, m.position, m.variant_index
                HAVING LENGTH(TRIM(COALESCE(text, ''))) >= 20
            ) embeddable_messages
        """,
    ),
    DemoConstruct(
        construct_id="synthetic_message_embedding_rows",
        label="Synthetic message embedding rows",
        description="Deterministic synthetic embedding vectors are present in embeddings.db for demo surfaces.",
        sql="""
            SELECT COUNT(*)
            FROM embeddings.message_embeddings_meta
            WHERE model = 'demo-synthetic-embedding'
              AND needs_reindex = 0
        """,
    ),
    DemoConstruct(
        construct_id="embedding_status_rows",
        label="Embedding status rows",
        description="The embeddings tier records at least one completed session-level status row.",
        sql="""
            SELECT COUNT(*)
            FROM embeddings.embedding_status
            WHERE message_count_embedded > 0
              AND needs_reindex = 0
              AND error_message IS NULL
        """,
    ),
)


def evaluate_demo_constructs(archive_root: Path) -> tuple[DemoConstructCoverage, ...]:
    """Measure declared construct coverage against the demo index tier."""

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        embeddings_db = archive_root / "embeddings.db"
        if embeddings_db.exists():
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        source_db = archive_root / "source.db"
        if source_db.exists():
            conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
        rows: list[DemoConstructCoverage] = []
        for construct in DEMO_CONSTRUCTS:
            result_rows = conn.execute(construct.sql).fetchall()
            observed = sum(int(row[0] or 0) for row in result_rows)
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
