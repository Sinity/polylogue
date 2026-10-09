"""Production-route tests for bounded annotation batch import."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast

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
from polylogue.core.json import require_json_document
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.operations.bindings import OperationBinding
from polylogue.operations.mutation_transaction import OperationExecutor
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidate
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.live_ingest import write_index_session
from tests.infra.user_tier import connect_user_db


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


def _request(batch_id: str) -> AnnotationBatchImportRequest:
    return AnnotationBatchImportRequest(
        batch_id=batch_id,
        schema_id="test.import",
        schema_version=1,
        target_ref="session:codex-session:annotation-target",
        source_result_ref="result-set:annotation-evidence",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref="block:prompt:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
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
async def test_import_streams_complete_population_and_legal_unicode_without_old_caps(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The ordinary writer retains every row, including one formerly oversized row."""
    archive_root = workspace_env["archive_root"]

    def seed() -> None:
        with ArchiveStore(archive_root) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    run_off_event_loop(seed)
    schema = AnnotationSchema(
        schema_id="test.import",
        version=1,
        title="Uncapped import",
        fields=(AnnotationField(name="label", value_type="string"),),
        target_ref_kinds=("session",),
        evidence_policy="required",
        status="active",
    )
    registry = AnnotationSchemaRegistry()
    registry.register(schema)
    special = "before\u0085middle\u2028paragraph\u2029after" + "x" * 70_000
    path = tmp_path / "labels.jsonl"
    with path.open("wb") as output:
        for index in range(10_001):
            output.write(
                (
                    json.dumps(
                        {
                            "row_key": f"row-{index}",
                            "value": {"label": special if index == 0 else "yes"},
                            "evidence_refs": ["codex-session:annotation-target"],
                        },
                        ensure_ascii=False,
                    )
                    + "\r\n"
                ).encode("utf-8")
            )
    assert path.stat().st_size > 1_048_576
    async with Polylogue(archive_root=archive_root) as poly:
        with path.open("rb") as input:
            result = await import_annotation_batch(poly, _request("uncapped"), input=input, registry=registry)
            assert not input.closed
    assert (result.status, result.total_count, result.valid_count, result.invalid_count) == ("ok", 10_001, 10_001, 0)
    with connect_user_db(archive_root / "user.db") as conn:
        first = conn.execute("SELECT value_json FROM assertions WHERE key='row-0'").fetchone()
        assert first is not None
        assert json.loads(first[0])["label"] == special
        assert (
            conn.execute("SELECT count(*) FROM assertions WHERE scope_ref=?", (result.batch_ref,)).fetchone()[0]
            == 10_001
        )
    with ArchiveStore.open_existing(archive_root) as archive:
        page = archive.get_annotation_batch_page("uncapped", limit=2, offset=9_999)
    assert page is not None
    assert (page.total, page.next_offset) == (10_001, None)
    assert [item["ordinal"] for item in page.items] == [9_999, 10_000]


@pytest.mark.asyncio
async def test_import_cancelled_at_acceptance_keeps_validated_rows_and_schema_unpublished(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]

    def seed() -> None:
        with ArchiveStore(archive_root) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    run_off_event_loop(seed)
    registry = AnnotationSchemaRegistry()
    registry.register(_schema())
    body = io.BytesIO(
        json.dumps(
            {
                "row_key": "cancelled",
                "value": {"label": "yes", "confidence": 0.8},
                "evidence_refs": ["codex-session:annotation-target"],
            }
        ).encode("utf-8")
    )

    class CancelledBeforeAcceptanceError(RuntimeError):
        pass

    def cancel() -> None:
        assert body.tell() == len(body.getbuffer())
        raise CancelledBeforeAcceptanceError

    async with Polylogue(archive_root=archive_root) as poly:
        with pytest.raises(CancelledBeforeAcceptanceError):
            await import_annotation_batch(
                poly, _request("cancelled"), input=body, registry=registry, before_durable_execution=cancel
            )
    assert not body.closed
    with connect_user_db(archive_root / "user.db") as connection:
        assert (
            connection.execute("SELECT count(*) FROM annotation_batches WHERE batch_id='cancelled'").fetchone()[0] == 0
        )
        assert connection.execute("SELECT count(*) FROM assertions WHERE key='cancelled'").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM annotation_schemas WHERE schema_id='test.import'").fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_import_classifies_unrepresentable_probability_without_losing_exact_payload_numbers(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]

    def seed() -> None:
        with ArchiveStore(archive_root) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    run_off_event_loop(seed)
    schema = AnnotationSchema(
        schema_id="test.import",
        version=1,
        title="Probability extraction",
        fields=(
            AnnotationField(name="confidence", value_type="number", required=False),
            AnnotationField(name="counter", value_type="integer"),
        ),
        target_ref_kinds=("session",),
        evidence_policy="required",
        status="active",
    )
    registry = AnnotationSchemaRegistry()
    registry.register(schema)
    exact = 10**1000
    values = ({"confidence": 0, "counter": exact}, {"counter": exact}, {"confidence": exact, "counter": 1})
    body = io.BytesIO(
        "\n".join(
            json.dumps(
                {
                    "row_key": f"row-{index}",
                    "value": value,
                    "evidence_refs": ["codex-session:annotation-target"],
                }
            )
            for index, value in enumerate(values)
        ).encode("utf-8")
    )
    resolved: list[str] = []
    async with Polylogue(archive_root=archive_root) as poly:

        async def resolve(ref: str) -> bool:
            resolved.append(ref)
            return (await poly.resolve_ref(ref)).resolved

        result = await import_annotation_batch(
            poly, _request("probabilities"), input=body, registry=registry, resolve_ref=resolve
        )
    assert (result.status, result.total_count, result.valid_count, result.invalid_count) == ("partial", 3, 2, 1)
    assert resolved == ["session:codex-session:annotation-target", "codex-session:annotation-target"]
    with connect_user_db(archive_root / "user.db") as connection:
        rows = connection.execute("SELECT key, value_json, confidence FROM assertions ORDER BY key").fetchall()
        assert [(row[0], json.loads(row[1])["counter"], row[2]) for row in rows] == [
            ("row-0", exact, 0.0),
            ("row-1", exact, None),
        ]
    with ArchiveStore.open_existing(archive_root) as archive:
        page = archive.get_annotation_batch_page("probabilities", limit=10, offset=0)
    assert page is not None
    invalid = [item for item in page.items if item["kind"] == "validation-error"]
    assert len(invalid) == 1
    assert require_json_document(invalid[0]["failure"], context="validation failure")["row_key"] == "row-2"
    assert invalid[0]["error"] == "value.confidence must be a finite probability between 0 and 1"


@pytest.mark.asyncio
async def test_import_roundtrip_keeps_failures_candidates_and_independent_batches(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registered import route writes real user-tier provenance and candidates.

    Anti-vacuity: ``import_annotation_batch`` must dispatch the real
    ``AnnotationBatchImportActuator`` through ``OperationExecutor.execute_bound`` before the
    transaction reaches ``user.db``. Removing that executor dispatch or
    restoring a direct persistence call leaves ``executed`` empty even though
    a toy persistence stub could still appear green.
    """
    executed: list[str] = []
    original_execute_bound = OperationExecutor.execute_bound

    def spy(
        self: OperationExecutor,
        binding: OperationBinding[object, object],
        preview: object,
        authorization: object,
        args: object,
    ) -> object:
        executed.append(type(binding.actuator).__name__)
        return original_execute_bound(self, binding, preview, authorization, args)  # type: ignore[arg-type]

    monkeypatch.setattr(OperationExecutor, "execute_bound", spy)

    archive_root = workspace_env["archive_root"]

    def _seed_archive_1() -> Any:
        with ArchiveStore(archive_root) as archive:
            session_id = write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    title="Annotation target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )
            return (session_id,)

    (session_id,) = run_off_event_loop(_seed_archive_1)
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
        first = await import_annotation_batch(
            poly, _request("batch-one"), input=io.BytesIO(jsonl.encode("utf-8")), registry=registry
        )
        second = await import_annotation_batch(
            poly, _request("batch-two"), input=io.BytesIO(jsonl.encode("utf-8")), registry=registry
        )

    assert executed == ["AnnotationBatchImportActuator", "AnnotationBatchImportActuator"]
    assert (first.total_count, first.valid_count, first.invalid_count) == (5, 4, 1)
    assert first.status == second.status == "partial"
    assert first.batch_ref != second.batch_ref
    assert "rows" not in first.model_dump()

    with ArchiveStore.open_existing(archive_root) as archive:
        batches = archive.list_annotation_batches(schema_id="test.import")
    assert {batch.batch_ref for batch in batches} == {"annotation-batch:batch-one", "annotation-batch:batch-two"}
    assert all(batch.validation_failures for batch in batches)

    with connect_user_db(archive_root / "user.db") as conn:
        assertion_rows = conn.execute(
            "SELECT status, context_policy_json, scope_ref FROM assertions WHERE key LIKE 'row-%'"
        ).fetchall()
    assert len(assertion_rows) == 8
    assert {row[0] for row in assertion_rows} == {"candidate"}
    policies = [json.loads(row[1]) for row in assertion_rows]
    assert all(policy["inject"] is False and policy["promotion_required"] is True for policy in policies)
    assert {row[2] for row in assertion_rows} == {"annotation-batch:batch-one", "annotation-batch:batch-two"}

    judgment_refs = list(next(batch.assertion_refs for batch in batches if batch.batch_ref == first.batch_ref))[:3]
    with connect_user_db(archive_root / "user.db") as conn:
        for candidate_ref, decision in zip(judgment_refs, ("accept", "reject", "defer"), strict=True):
            assert candidate_ref is not None
            judge_assertion_candidate(conn, candidate_ref=candidate_ref, decision=decision, now_ms=2_000)
        conn.commit()
        judged_statuses = {
            row[0]
            for row in conn.execute(
                "SELECT status FROM assertions WHERE assertion_id IN (?, ?, ?)",
                tuple(ref.removeprefix("assertion:") for ref in judgment_refs),
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

    def _seed_archive_2() -> Any:
        with ArchiveStore(archive_root) as archive:
            parent_session_id = write_index_session(archive, _delegation_parent())
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="import-child",
                    title="Annotation import delegation child",
                    messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="working")],
                    parent_session_provider_id="import-parent",
                    branch_type=BranchType.SUBAGENT,
                ),
            )
            return (parent_session_id,)

    (parent_session_id,) = run_off_event_loop(_seed_archive_2)
    instruction_block_id = f"{parent_session_id}:n:dispatch:0"
    target_ref = f"delegation:{instruction_block_id}"
    evidence_ref = f"block:{instruction_block_id}"
    # Mirrors instruction_block_id above: the message id carries the `n:`
    # native-id discriminator, so a span built without it resolves to nothing
    # and every row is rejected as unresolvable evidence.
    evidence_span = f"{parent_session_id}::{parent_session_id}:n:dispatch::0"
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
        batch_id="delegation-retry",
        schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
        schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
        target_ref=target_ref,
        source_result_ref="result-set:delegation-review",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref=evidence_ref,
    )

    with running_daemon_operations(archive_root, socket_path=daemon_socket_path(archive_root)):
        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
            first = await import_annotation_batch(poly, request, input=io.BytesIO(jsonl.encode("utf-8")))
            replayed = await import_annotation_batch(poly, request, input=io.BytesIO(jsonl.encode("utf-8")))
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
                    }
                ),
                input=io.BytesIO("\n".join(json.dumps(row) for row in disagreement_rows).encode("utf-8")),
            )
            missing_target = request.model_copy(
                update={"batch_id": "missing-target", "target_ref": "delegation:missing"}
            )
            with pytest.raises(ValueError, match="does not resolve"):
                await import_annotation_batch(poly, missing_target, input=io.BytesIO(jsonl.encode("utf-8")))

            resolution = await poly.resolve_ref(first.batch_ref)
            assert resolution.payload is not None
            first_refs = [item["assertion_ref"] for item in resolution.payload["items"] if item["kind"] == "assertion"]
            judgments = []
            for candidate_ref, decision in zip(first_refs[:3], ("accept", "reject", "defer"), strict=True):
                assert candidate_ref is not None
                judgments.append(await poly.judge_assertion_candidate(candidate_ref=candidate_ref, decision=decision))

            typed = await poly.query_units(
                "assertions where kind:annotation AND status:active AND value.confidence:>=0.8"
            )
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
    with connect_user_db(archive_root / "user.db") as conn:
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


class _Killed(BaseException):
    """Stands in for the process dying mid-import; not an ``Exception`` any route may catch."""


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", ["before-apply", "after-apply"])
async def test_interrupted_import_is_resolved_complete_or_absent_at_restart(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, crash: str
) -> None:
    """A killed batch import is resolved from its one atomic transaction, never ``unknown``.

    Anti-vacuity: make ``AnnotationBatchImportActuator.recover`` report
    ``complete`` without reading ``annotation_batches`` and the before-apply
    row turns red; drop the route from ``recoverable_actuators`` and both rows
    end ``recovery_not_replayable``.
    """
    import sqlite3

    from tests.infra.operation_recovery import recover_on_admitted_owner

    started: list[str] = []

    def die_mid_import(
        self: OperationExecutor,
        binding: OperationBinding[object, object],
        preview: object,
        authorization: object,
        args: object,
    ) -> object:
        assert self._audit is not None
        started.append(self._audit.consume_authorization_and_start(preview, authorization))  # type: ignore[arg-type]
        if crash == "after-apply":
            binding.actuator.apply(preview.plan, args)  # type: ignore[attr-defined]
        raise _Killed

    archive_root = workspace_env["archive_root"]

    def _seed_archive_3() -> Any:
        with ArchiveStore(archive_root) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    title="Annotation target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    run_off_event_loop(_seed_archive_3)
    registry = AnnotationSchemaRegistry()
    registry.register(_schema())
    jsonl = json.dumps(
        {
            "row_key": "row-0",
            "value": {"label": "yes", "confidence": 0.9},
            "evidence_refs": ["codex-session:annotation-target"],
        }
    )
    monkeypatch.setattr(OperationExecutor, "execute_bound", die_mid_import)
    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(_Killed):
            await import_annotation_batch(
                poly, _request("batch-killed"), input=io.BytesIO(jsonl.encode("utf-8")), registry=registry
            )
    (operation_id,) = started
    with sqlite3.connect(archive_root / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()

    recover_on_admitted_owner(archive_root)

    with sqlite3.connect(archive_root / "audit.db") as conn:
        status, reason = conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone()
    with ArchiveStore.open_existing(archive_root) as archive:
        committed = archive.get_annotation_batch("batch-killed") is not None
    if crash == "after-apply":
        assert (status, reason, committed) == ("completed", "recovered_complete", True)
    else:
        assert (status, reason, committed) == ("failed", "recovered_absent", False)


@pytest.mark.asyncio
async def test_interrupted_import_reusing_a_batch_id_is_not_mistaken_for_its_predecessor(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier batch under the same id does not make a killed import complete.

    Anti-vacuity: resolve by ``batch_id`` existence alone in
    ``AnnotationBatchImportActuator.recover`` and this run ends
    ``recovered_complete`` though none of its rows committed.
    """
    import sqlite3

    from tests.infra.operation_recovery import recover_on_admitted_owner

    archive_root = workspace_env["archive_root"]

    def _seed_archive_4() -> Any:
        with ArchiveStore(archive_root) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-target",
                    title="Annotation target",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                ),
            )

    run_off_event_loop(_seed_archive_4)
    registry = AnnotationSchemaRegistry()
    registry.register(_schema())

    def row(key: str) -> str:
        return json.dumps(
            {
                "row_key": key,
                "value": {"label": "yes", "confidence": 0.9},
                "evidence_refs": ["codex-session:annotation-target"],
            }
        )

    started: list[str] = []

    def die_before_apply(
        self: OperationExecutor,
        binding: OperationBinding[object, object],
        preview: object,
        authorization: object,
        args: object,
    ) -> object:
        assert self._audit is not None
        started.append(self._audit.consume_authorization_and_start(preview, authorization))  # type: ignore[arg-type]
        raise _Killed

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        await import_annotation_batch(
            poly, _request("batch-reused"), input=io.BytesIO(row("first").encode("utf-8")), registry=registry
        )
        monkeypatch.setattr(OperationExecutor, "execute_bound", die_before_apply)
        with pytest.raises(_Killed):
            await import_annotation_batch(
                poly, _request("batch-reused"), input=io.BytesIO(row("second").encode("utf-8")), registry=registry
            )
    (operation_id,) = started
    with sqlite3.connect(archive_root / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()

    recover_on_admitted_owner(archive_root)

    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "recovered_absent")
