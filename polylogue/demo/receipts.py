"""Deterministic claim-versus-evidence proof over the demo archive."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from polylogue.insights.cohorts import CohortCandidate, CohortManifest, CohortSpec, compile_cohort_manifest
from polylogue.scenarios import (
    DEMO_CODEX_ANTI_GREP_SESSION_ID,
    DEMO_CODEX_RECEIPTS_SESSION_ID,
)

_EXPECTED_CLAIM = "All tests pass. The clock fix is complete."
_COMPLETION_CLAIM_SAMPLE_SIZE = 250
_COMPLETION_CLAIM_SEED = "polylogue-demo-receipts-v1"
_COMPLETION_CLAIM_QUERY = (
    "assistant text blocks matching the declared high-specificity completion phrases: "
    "all tests pass | tests are passing | implementation is complete | task is complete | "
    "fix is complete | completed successfully"
)
_COMPLETION_FTS_QUERY = (
    '"all tests pass" OR "tests are passing" OR "implementation is complete" OR '
    '"task is complete" OR "fix is complete" OR "completed successfully"'
)
_COMPLETION_MARKERS = (
    "all tests pass",
    "tests are passing",
    "implementation is complete",
    "task is complete",
    "fix is complete",
    "completed successfully",
)


@dataclass(frozen=True, slots=True)
class DemoActionReceipt:
    """One structural tool-action receipt used by the evidence proof."""

    block_ref: str
    tool_name: str
    semantic_type: str
    command: str
    exit_code: int
    is_error: bool
    output: str

    def to_payload(self) -> dict[str, object]:
        return {
            "block_ref": self.block_ref,
            "tool_name": self.tool_name,
            "semantic_type": self.semantic_type,
            "command": self.command,
            "exit_code": self.exit_code,
            "is_error": self.is_error,
            "output": self.output,
        }


@dataclass(frozen=True, slots=True)
class DemoReceiptsResult:
    """Result of the deterministic claim-versus-evidence proof."""

    archive_root: Path
    ok: bool
    verdict: str
    session_ref: str
    claim_ref: str
    claim_text: str
    failed: DemoActionReceipt | None
    recovery: DemoActionReceipt | None
    anti_grep_session_ref: str
    anti_grep_text_hits: int
    anti_grep_failed_actions: int
    raw_id: str | None
    raw_blob_sha256: str | None
    completion_claims: CompletionClaimExperimentResult | None = None
    problems: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "ok": self.ok,
            "verdict": self.verdict,
            "session_ref": self.session_ref,
            "claim_ref": self.claim_ref,
            "claim_text": self.claim_text,
            "failed": self.failed.to_payload() if self.failed is not None else None,
            "recovery": self.recovery.to_payload() if self.recovery is not None else None,
            "anti_grep_control": {
                "session_ref": self.anti_grep_session_ref,
                "text_hits_for_error": self.anti_grep_text_hits,
                "failed_actions": self.anti_grep_failed_actions,
            },
            "source_material": {
                "raw_id": self.raw_id,
                "blob_sha256": self.raw_blob_sha256,
            },
            "completion_claim_experiment": (
                self.completion_claims.to_payload() if self.completion_claims is not None else None
            ),
            "problems": list(self.problems),
        }


@dataclass(frozen=True, slots=True)
class CompletionClaimEvidence:
    """One sampled completion claim resolved against typed action outcomes."""

    claim_ref: str
    session_ref: str
    origin: str
    classification: str
    prior_action_ref: str | None
    repair_action_ref: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "claim_ref": self.claim_ref,
            "session_ref": self.session_ref,
            "origin": self.origin,
            "classification": self.classification,
            "prior_action_ref": self.prior_action_ref,
            "repair_action_ref": self.repair_action_ref,
        }


@dataclass(frozen=True, slots=True)
class CompletionClaimExperimentResult:
    """Aggregate completion-claim result with a deterministic sample manifest."""

    archive_root: Path
    manifest: CohortManifest
    evidence: tuple[CompletionClaimEvidence, ...]
    unsupported_count: int
    contradicted_then_repaired_count: int

    @property
    def sample_size(self) -> int:
        return len(self.evidence)

    def to_payload(self) -> dict[str, object]:
        denominator = self.sample_size
        return {
            "archive_root": str(self.archive_root),
            "method": {
                "population_query": _COMPLETION_CLAIM_QUERY,
                "unsupported_definition": (
                    "No structurally recorded action outcome precedes the selected completion claim."
                ),
                "contradicted_then_repaired_definition": (
                    "The latest structurally recorded action before the claim failed, and a later action "
                    "with the same tool and command succeeded."
                ),
            },
            "manifest": self.manifest.to_payload(),
            "headline": {
                "denominator": denominator,
                "unsupported_count": self.unsupported_count,
                "unsupported_rate": self.unsupported_count / denominator if denominator else None,
                "contradicted_then_repaired_count": self.contradicted_then_repaired_count,
                "contradicted_then_repaired_rate": (
                    self.contradicted_then_repaired_count / denominator if denominator else None
                ),
            },
            "evidence": [item.to_payload() for item in self.evidence],
        }


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _command_from_row(row: sqlite3.Row) -> str:
    command = row["tool_command"]
    if isinstance(command, str) and command:
        return command
    raw_input = row["tool_input"]
    if not isinstance(raw_input, str) or not raw_input:
        return ""
    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError:
        return raw_input
    if isinstance(payload, dict):
        for key in ("cmd", "command"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return raw_input


def _action_receipt(row: sqlite3.Row) -> DemoActionReceipt:
    block_id = str(row["tool_result_block_id"])
    return DemoActionReceipt(
        block_ref=f"block:{block_id}",
        tool_name=str(row["tool_name"] or "unknown"),
        semantic_type=str(row["semantic_type"] or "unknown"),
        command=_command_from_row(row),
        exit_code=int(row["exit_code"]),
        is_error=bool(row["is_error"]),
        output=str(row["output_text"] or ""),
    )


def _completion_template(text: str) -> str:
    lowered = text.lower()
    return next((marker for marker in _COMPLETION_MARKERS if marker in lowered), "other")


def _archive_cursor(conn: sqlite3.Connection) -> str:
    schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    session_count, message_count, last_message_id = conn.execute(
        "SELECT (SELECT COUNT(*) FROM sessions), COUNT(*), COALESCE(MAX(message_id), '') FROM messages"
    ).fetchone()
    payload = f"index-v{schema_version}:sessions={session_count}:messages={message_count}:last={last_message_id}"
    return f"sha256:{sha256(payload.encode()).hexdigest()}"


def _action_rows(conn: sqlite3.Connection, session_id: str, position_clause: str, position: int) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT a.*, m.position AS message_position
        FROM actions AS a
        JOIN messages AS m ON m.message_id = a.message_id
        WHERE a.session_id = ?
          AND a.exit_code IS NOT NULL
          AND m.position {position_clause} ?
        ORDER BY m.position, a.tool_result_block_id
        """,
        (session_id, position),
    ).fetchall()


def _is_failed_action(row: sqlite3.Row) -> bool:
    return int(row["is_error"] or 0) == 1 or int(row["exit_code"] or 0) != 0


def _is_matching_repair(failed: sqlite3.Row, candidate: sqlite3.Row) -> bool:
    return (
        not _is_failed_action(candidate)
        and str(candidate["tool_name"] or "") == str(failed["tool_name"] or "")
        and _command_from_row(candidate) == _command_from_row(failed)
    )


def inspect_completion_claims(
    archive_root: Path,
    *,
    sample_size: int = _COMPLETION_CLAIM_SAMPLE_SIZE,
) -> CompletionClaimExperimentResult:
    """Sample completion claims and resolve them against typed action evidence.

    The population is deliberately lexical and declared in the result rather
    than inferred as every possible claim. Action outcomes, not prose, decide
    whether a sampled claim lacks evidence or was contradicted then repaired.
    """

    with _connect(archive_root / "index.db") as conn:
        candidates_rows = conn.execute(
            """
            SELECT b.block_id, b.text, m.session_id, m.position, s.origin, m.model_name
            FROM blocks AS b
            JOIN messages AS m ON m.message_id = b.message_id
            JOIN sessions AS s ON s.session_id = m.session_id
            WHERE b.block_type = 'text'
              AND m.role = 'assistant'
              AND b.rowid IN (
                  SELECT rowid
                  FROM messages_fts
                  WHERE messages_fts MATCH ?
              )
            ORDER BY b.block_id
            """,
            (_COMPLETION_FTS_QUERY,),
        ).fetchall()
        candidates = [
            CohortCandidate(
                object_ref=f"block:{row['block_id']}",
                dimensions={"origin": str(row["origin"]), "model": str(row["model_name"] or "unknown")},
                template_key=_completion_template(str(row["text"] or "")),
            )
            for row in candidates_rows
        ]
        manifest = compile_cohort_manifest(
            CohortSpec(
                population_query=_COMPLETION_CLAIM_QUERY,
                archive_cursor=_archive_cursor(conn),
                seed=_COMPLETION_CLAIM_SEED,
                requested_size=sample_size,
                strata=("origin",),
                exact_template_cap=25,
            ),
            candidates,
        )
        selected = {ref.removeprefix("block:") for ref in manifest.selected_refs}
        evidence: list[CompletionClaimEvidence] = []
        for row in candidates_rows:
            block_id = str(row["block_id"])
            if block_id not in selected:
                continue
            prior = _action_rows(conn, str(row["session_id"]), "<", int(row["position"]))
            prior_action = prior[-1] if prior else None
            failed = prior_action if prior_action is not None and _is_failed_action(prior_action) else None
            repair = None
            classification = "supported_at_claim_time"
            if prior_action is None:
                classification = "unsupported_by_structural_tool_evidence"
            elif failed is not None:
                later = _action_rows(conn, str(row["session_id"]), ">", int(row["position"]))
                repair = next((candidate for candidate in later if _is_matching_repair(failed, candidate)), None)
                classification = (
                    "contradicted_then_repaired" if repair is not None else "contradicted_without_recorded_repair"
                )
            evidence.append(
                CompletionClaimEvidence(
                    claim_ref=f"block:{block_id}",
                    session_ref=f"session:{row['session_id']}",
                    origin=str(row["origin"]),
                    classification=classification,
                    prior_action_ref=(
                        f"block:{prior_action['tool_result_block_id']}" if prior_action is not None else None
                    ),
                    repair_action_ref=f"block:{repair['tool_result_block_id']}" if repair is not None else None,
                )
            )
    unsupported_count = sum(item.classification == "unsupported_by_structural_tool_evidence" for item in evidence)
    repaired_count = sum(item.classification == "contradicted_then_repaired" for item in evidence)
    return CompletionClaimExperimentResult(
        archive_root=archive_root,
        manifest=manifest,
        evidence=tuple(evidence),
        unsupported_count=unsupported_count,
        contradicted_then_repaired_count=repaired_count,
    )


def inspect_demo_receipts(archive_root: Path) -> DemoReceiptsResult:
    """Compare one assistant claim with typed tool evidence and a lexical control."""

    problems: list[str] = []
    claim_ref = ""
    claim_text = ""
    failed: DemoActionReceipt | None = None
    recovery: DemoActionReceipt | None = None
    anti_grep_text_hits = 0
    anti_grep_failed_actions = 0
    raw_id: str | None = None
    raw_blob_sha256: str | None = None

    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    try:
        with _connect(index_db) as conn:
            action_rows = conn.execute(
                """
                SELECT a.*, m.position AS message_position
                FROM actions AS a
                JOIN messages AS m ON m.message_id = a.message_id
                WHERE a.session_id = ?
                  AND a.exit_code IS NOT NULL
                  AND a.tool_input LIKE '%pytest tests/test_clock.py%'
                ORDER BY m.position, a.tool_result_block_id
                """,
                (DEMO_CODEX_RECEIPTS_SESSION_ID,),
            ).fetchall()
            failed_row = next((row for row in action_rows if int(row["is_error"] or 0) == 1), None)
            recovery_row = next(
                (
                    row
                    for row in action_rows
                    if int(row["is_error"] or 0) == 0
                    and failed_row is not None
                    and int(row["message_position"]) > int(failed_row["message_position"])
                ),
                None,
            )
            if failed_row is None:
                problems.append("missing structurally failed pytest action")
            else:
                failed = _action_receipt(failed_row)
            if recovery_row is None:
                problems.append("missing later successful pytest recovery action")
            else:
                recovery = _action_receipt(recovery_row)

            claim_row = conn.execute(
                """
                SELECT b.block_id, b.text, m.position
                FROM blocks AS b
                JOIN messages AS m ON m.message_id = b.message_id
                WHERE b.session_id = ?
                  AND b.block_type = 'text'
                  AND b.text = ?
                ORDER BY m.position
                LIMIT 1
                """,
                (DEMO_CODEX_RECEIPTS_SESSION_ID, _EXPECTED_CLAIM),
            ).fetchone()
            if claim_row is None:
                problems.append("missing conflicting assistant claim")
            else:
                claim_ref = f"block:{claim_row['block_id']}"
                claim_text = str(claim_row["text"])
                if failed_row is not None and int(claim_row["position"]) <= int(failed_row["message_position"]):
                    problems.append("assistant claim does not occur after the failed action")
                if recovery_row is not None and int(claim_row["position"]) >= int(recovery_row["message_position"]):
                    problems.append("assistant claim does not occur before the successful recovery")

            anti_grep_text_hits = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM blocks
                    WHERE session_id = ?
                      AND block_type = 'text'
                      AND LOWER(text) LIKE '%error%'
                    """,
                    (DEMO_CODEX_ANTI_GREP_SESSION_ID,),
                ).fetchone()[0]
            )
            anti_grep_failed_actions = int(
                conn.execute(
                    "SELECT COUNT(*) FROM actions WHERE session_id = ? AND is_error = 1",
                    (DEMO_CODEX_ANTI_GREP_SESSION_ID,),
                ).fetchone()[0]
            )
            if anti_grep_text_hits < 1:
                problems.append("anti-grep control contains no lexical error hit")
            if anti_grep_failed_actions != 0:
                problems.append("anti-grep control unexpectedly contains a failed action")
    except (OSError, sqlite3.Error) as exc:
        problems.append(f"archive evidence unreadable: {exc}")

    try:
        with _connect(source_db) as conn:
            row = conn.execute(
                "SELECT raw_id, hex(blob_hash) AS blob_hash FROM raw_sessions WHERE origin = 'codex-session' AND native_id = ?",
                (DEMO_CODEX_RECEIPTS_SESSION_ID.removeprefix("codex-session:"),),
            ).fetchone()
            if row is None:
                problems.append("source material row for receipts session is missing")
            else:
                raw_id = str(row["raw_id"])
                raw_blob_sha256 = str(row["blob_hash"]).lower()
    except (OSError, sqlite3.Error) as exc:
        problems.append(f"source evidence unreadable: {exc}")

    verdict = "contradicted_at_claim_time_then_repaired" if not problems else "invalid_demo_evidence"
    return DemoReceiptsResult(
        archive_root=archive_root,
        ok=not problems,
        verdict=verdict,
        session_ref=f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}",
        claim_ref=claim_ref,
        claim_text=claim_text,
        failed=failed,
        recovery=recovery,
        anti_grep_session_ref=f"session:{DEMO_CODEX_ANTI_GREP_SESSION_ID}",
        anti_grep_text_hits=anti_grep_text_hits,
        anti_grep_failed_actions=anti_grep_failed_actions,
        raw_id=raw_id,
        raw_blob_sha256=raw_blob_sha256,
        completion_claims=inspect_completion_claims(archive_root),
        problems=tuple(problems),
    )


def render_demo_receipts(result: DemoReceiptsResult) -> str:
    """Render the proof as a compact human-readable receipt."""

    failed = result.failed
    recovery = result.recovery
    lines = [
        "Polylogue evidence receipt",
        f"archive: {result.archive_root}",
        f"verdict: {result.verdict}",
        "",
        f"assistant claim: {result.claim_text}",
        f"claim evidence: {result.claim_ref}",
        "",
        "at claim time:",
    ]
    if failed is None:
        lines.append("  unavailable")
    else:
        lines.extend(
            [
                f"  tool: {failed.semantic_type} ({failed.tool_name})",
                f"  command: {failed.command}",
                f"  exit: {failed.exit_code} (failed={str(failed.is_error).lower()})",
                f"  result: {failed.output}",
                f"  evidence: {failed.block_ref}",
            ]
        )
    lines.extend(["", "later recovery:"])
    if recovery is None:
        lines.append("  unavailable")
    else:
        lines.extend(
            [
                f"  tool: {recovery.semantic_type} ({recovery.tool_name})",
                f"  command: {recovery.command}",
                f"  exit: {recovery.exit_code} (failed={str(recovery.is_error).lower()})",
                f"  result: {recovery.output}",
                f"  evidence: {recovery.block_ref}",
            ]
        )
    lines.extend(
        [
            "",
            "anti-grep control:",
            f"  prose hits for 'error': {result.anti_grep_text_hits}",
            f"  structurally failed actions: {result.anti_grep_failed_actions}",
            f"  control session: {result.anti_grep_session_ref}",
            "",
            "source material:",
            f"  raw_id: {result.raw_id or 'unavailable'}",
            f"  blob_sha256: {result.raw_blob_sha256 or 'unavailable'}",
        ]
    )
    experiment = result.completion_claims
    if experiment is not None:
        headline = experiment.to_payload()["headline"]
        assert isinstance(headline, dict)
        lines.extend(
            [
                "",
                "completion-claim experiment:",
                f"  sample manifest: {experiment.manifest.manifest_id}",
                f"  denominator: {headline['denominator']}",
                f"  unsupported by structural evidence: {headline['unsupported_count']}",
                f"  contradicted then repaired: {headline['contradicted_then_repaired_count']}",
            ]
        )
    if result.problems:
        lines.extend(["", "problems:", *(f"  - {problem}" for problem in result.problems)])
    return "\n".join(lines) + "\n"


__all__ = [
    "DemoActionReceipt",
    "CompletionClaimEvidence",
    "CompletionClaimExperimentResult",
    "DemoReceiptsResult",
    "inspect_completion_claims",
    "inspect_demo_receipts",
    "render_demo_receipts",
]
