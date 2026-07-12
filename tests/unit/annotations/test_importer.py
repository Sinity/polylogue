"""Production-route tests for bounded annotation batch import."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.annotations.importer import AnnotationBatchImportRequest, import_annotation_batch
from polylogue.annotations.schema import (
    DELEGATION_DISCOURSE_SCHEMA,
    AnnotationField,
    AnnotationSchema,
    AnnotationSchemaRegistry,
)
from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.core.enums import AssertionKind, BlockType, BranchType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidate


def _schema() -> AnnotationSchema:
    return AnnotationSchema(
        schema_id="test.import",
        version=1,
        title="Import fixture",
        fields=(
            AnnotationField(name="label", value_type="enum", enum_values=("yes", "no")),
            AnnotationField(name="confidence", value_type="number", minimum=0, maximum=1),
            AnnotationField(name="abstain", value_type="boolean", required=False),
        ),
        target_ref_kinds=("session",),
        abstain_field="abstain",
        evidence_policy="required",
        status="active",
    )


def _request(batch_id: str, jsonl: str) -> AnnotationBatchImportRequest:
    return AnnotationBatchImportRequest(
        jsonl=jsonl,
        batch_id=batch_id,
        schema_id="test.import",
        schema_version=1,
        target_ref="session:codex-session:annotation-target",
        source_result_ref="result-set:annotation-evidence",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref="block:prompt:0",
        created_at_ms=1_000,
    )


def _delegation_value() -> dict[str, object]:
    return {
        "directive_mode": "imperative",
        "prohibitions": "explicit",
        "autonomy": "bounded",
        "output_contract": "structured",
        "scope_control": "owned_paths",
        "verification_demand": "focused_tests",
        "checkpoint_escalation": "escalation",
        "relational_frame": "collaborative",
        "rationale_visibility": "explicit",
        "applicable": True,
        "confidence": 0.9,
        "abstain": False,
        "rationale": "The dispatch names scope, checks, and escalation.",
    }


def _delegation_parent() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="import-parent",
        title="Annotation import delegation parent",
        messages=[
            ParsedMessage(
                provider_message_id="dispatch",
                role=Role.ASSISTANT,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Task",
                        tool_id="task-import",
                        tool_input={"prompt": "audit the importer", "subagent_type": "general-purpose"},
                    )
                ],
            ),
            ParsedMessage(
                provider_message_id="result",
                role=Role.USER,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="task-import",
                        text="no gaps",
                        is_error=False,
                        exit_code=0,
                    )
                ],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_import_roundtrip_keeps_failures_candidates_and_independent_batches(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="annotation-target",
                title="Annotation target",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
            )
        )
    index_db = archive_root / "index.db"
    assert session_id == "codex-session:annotation-target"
    rows = [
        {
            "row_key": f"row-{index}",
            "value": {"label": "yes", "confidence": 0.9},
            "evidence_refs": ["codex-session:annotation-target"],
        }
        for index in range(4)
    ]
    rows.append(
        {"row_key": "bad-evidence", "value": {"label": "no", "confidence": 0.4}, "evidence_refs": ["missing-session"]}
    )
    jsonl = "\n".join(json.dumps(row) for row in rows)
    registry = AnnotationSchemaRegistry()
    registry.register(_schema())

    async with Polylogue(archive_root=archive_root, db_path=index_db) as poly:
        first = await import_annotation_batch(poly, _request("batch-one", jsonl), registry=registry)
        second = await import_annotation_batch(poly, _request("batch-two", jsonl), registry=registry)

    assert (first.total_count, first.valid_count, first.invalid_count) == (5, 4, 1)
    assert first.status == second.status == "partial"
    assert first.batch_ref != second.batch_ref
    assert first.rows[-1].errors == ("evidence_ref 'missing-session' does not resolve in the live archive",)

    with ArchiveStore.open_existing(archive_root) as archive:
        batches = archive.list_annotation_batches(schema_id="test.import")
    assert {batch.batch_ref for batch in batches} == {"annotation-batch:batch-one", "annotation-batch:batch-two"}
    assert all(batch.validation_failures for batch in batches)

    with sqlite3.connect(archive_root / "user.db") as conn:
        assertion_rows = conn.execute(
            "SELECT status, context_policy_json, scope_ref FROM assertions WHERE key LIKE 'row-%'"
        ).fetchall()
    assert len(assertion_rows) == 8
    assert {row[0] for row in assertion_rows} == {"candidate"}
    policies = [json.loads(row[1]) for row in assertion_rows]
    assert all(policy["inject"] is False and policy["promotion_required"] is True for policy in policies)
    assert {row[2] for row in assertion_rows} == {"annotation-batch:batch-one", "annotation-batch:batch-two"}

    judgment_refs = [row.assertion_ref for row in first.rows if row.status == "imported"][:3]
    with sqlite3.connect(archive_root / "user.db") as conn:
        for candidate_ref, decision in zip(judgment_refs, ("accept", "reject", "defer"), strict=True):
            assert candidate_ref is not None
            judge_assertion_candidate(conn, candidate_ref=candidate_ref, decision=decision, now_ms=2_000)
        conn.commit()
        judged_statuses = {
            row[0]
            for row in conn.execute(
                "SELECT status FROM assertions WHERE assertion_id IN (?, ?, ?)",
                tuple(ref.removeprefix("assertion:") for ref in judgment_refs if ref is not None),
            )
        }
        active_count = conn.execute(
            "SELECT count(*) FROM assertions WHERE status = 'active' AND kind = 'annotation'"
        ).fetchone()[0]
    assert judged_statuses == {"accepted", "rejected", "deferred"}
    assert active_count == 1

    with ArchiveStore.open_existing(archive_root) as archive:
        typed_source = parse_unit_source_expression(
            "assertions where kind:annotation AND status:active AND value.confidence:>=0.8"
        )
        assert typed_source is not None
        active_rows = archive.query_assertions(typed_source.predicate, limit=100)
    assert len(active_rows) == 1
    assert active_rows[0].target_ref == "session:codex-session:annotation-target"
    assert active_rows[0].status == "active"


@pytest.mark.asyncio
async def test_import_uses_concrete_delegation_schema_and_exact_retry_is_idempotent(
    workspace_env: dict[str, Path],
) -> None:
    """The real resolver, built-in schema, writer, and retry path compose.

    Anti-vacuity: removing delegation resolution, schema validation, batch
    timestamp reuse, or batch-scoped assertion insert-once semantics breaks
    this production-route test.
    """
    archive_root = workspace_env["archive_root"]
    with ArchiveStore(archive_root) as archive:
        parent_session_id = archive.write_parsed(_delegation_parent())
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="import-child",
                title="Annotation import delegation child",
                messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="working")],
                parent_session_provider_id="import-parent",
                branch_type=BranchType.SUBAGENT,
            )
        )
    instruction_block_id = f"{parent_session_id}:dispatch:0"
    target_ref = f"delegation:{instruction_block_id}"
    evidence_ref = f"block:{instruction_block_id}"
    evidence_span = f"{parent_session_id}::{parent_session_id}:dispatch::0"
    valid_rows = [
        {
            "row_key": f"delegation-{index}",
            "value": {**_delegation_value(), "confidence": 0.9 - index / 20},
            "evidence_refs": [evidence_span],
        }
        for index in range(5)
    ]
    invalid_rows = [
        {
            "row_key": "wrong-lineage",
            "value": _delegation_value(),
            "evidence_refs": [f"claude-code-session:wrong::{parent_session_id}:dispatch::0"],
        },
        {
            "row_key": "delegation-0",
            "value": _delegation_value(),
            "evidence_refs": [evidence_span],
        },
    ]
    jsonl = "\n".join(json.dumps(row) for row in (*valid_rows, *invalid_rows))
    request = AnnotationBatchImportRequest(
        jsonl=jsonl,
        batch_id="delegation-retry",
        schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
        schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
        target_ref=target_ref,
        source_result_ref="result-set:delegation-review",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref=evidence_ref,
    )

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        first = await import_annotation_batch(poly, request)
        replayed = await import_annotation_batch(poly, request)
        disagreement_rows: list[dict[str, object]] = []
        for row in valid_rows:
            value = dict(cast(dict[str, object], row["value"]))
            value["directive_mode"] = "collaborative"
            disagreement_rows.append({**row, "value": value})
        second = await import_annotation_batch(
            poly,
            request.model_copy(
                update={
                    "batch_id": "delegation-disagreement",
                    "jsonl": "\n".join(json.dumps(row) for row in disagreement_rows),
                }
            ),
        )
        missing_target = request.model_copy(update={"batch_id": "missing-target", "target_ref": "delegation:missing"})
        with pytest.raises(ValueError, match="does not resolve"):
            await import_annotation_batch(poly, missing_target)

        first_refs = [row.assertion_ref for row in first.rows if row.status == "imported"]
        judgments = []
        for candidate_ref, decision in zip(first_refs[:3], ("accept", "reject", "defer"), strict=True):
            assert candidate_ref is not None
            judgments.append(await poly.judge_assertion_candidate(candidate_ref=candidate_ref, decision=decision))

        typed = await poly.query_units("assertions where kind:annotation AND status:active AND value.confidence:>=0.8")
        assert judgments[0].resulting_assertion is not None
        active_render = await poly.resolve_ref(f"assertion:{judgments[0].resulting_assertion.assertion_id}")
        unresolved = await poly.list_assertion_candidate_reviews(kinds=(AssertionKind.ANNOTATION,))

    assert replayed == first
    assert first.status == "partial"
    assert (first.total_count, first.valid_count, first.invalid_count) == (7, 5, 2)
    assert first.qualified_schema_id == "delegation.discourse@v1"
    assert second.status == "ok"
    assert second.valid_count == 5
    assert second.batch_ref != first.batch_ref
    assert len(typed.items) == 1
    assert active_render.resolved is True
    assert active_render.payload is not None and active_render.payload["status"] == "active"
    unresolved_payload = unresolved.model_dump(mode="json")
    assert unresolved_payload["total"] == 10
    assert {item["candidate"]["status"] for item in unresolved_payload["items"]} == {
        "candidate",
        "accepted",
        "deferred",
        "rejected",
    }
    with sqlite3.connect(archive_root / "user.db") as conn:
        assert (
            conn.execute("SELECT count(*) FROM annotation_batches WHERE batch_id = 'delegation-retry'").fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM assertions WHERE scope_ref = ? AND kind = 'annotation' AND status != 'active'",
                (first.batch_ref,),
            ).fetchone()[0]
            == 5
        )
