"""Incident-scale, sanitized proof for the Codex 804-revision gap.

The witness is structural rather than private: 804 full-snapshot acquisitions
share one source path, the terminal snapshot is exactly 90,822,451 bytes, and
the content contains only synthetic identifiers and padding.  The test still
uses the production acquisition, parser, convergence, resumable rebuild, and
generation-promotion seams.  It is deliberately a proof harness, not a second
archive writer.

The live archive and the real 2026-07-31 witness are unavailable to this lane.
The remaining confidence gap is therefore the live-operation receipt tracked
by ``polylogue-live-operation-receipts`` and the terminal production proof
``polylogue-reindex-final-proof``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.scenarios import (
    MeasurementScope,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)
from polylogue.scenarios.workload import raw_authority_fixed_point_spec
from polylogue.schemas.operator.receipt import package_hashes_for_registry
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_canonical_snapshot import archive_snapshot
from tests.infra.corpus_program import (
    Acquire,
    Converge,
    CorpusProgram,
    CorpusRuntimeCrashedError,
    Crash,
    ProductionCorpusRuntime,
    RawArtifact,
    Restart,
)
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

REVISION_COUNT = 804
TERMINAL_WIRE_BYTES = 90_822_451
SESSION_NATIVE_ID = "codex-sanitized-804-session"
SOURCE_PATH = "codex/incident-804-sanitized.jsonl"


def _codex_payload(revision: int, *, terminal: bool) -> bytes:
    records = [
        {
            "type": "session_meta",
            "payload": {
                "id": SESSION_NATIVE_ID,
                "timestamp": "2026-07-31T04:25:20Z",
                "cwd": "/sanitized/codex-804",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{SESSION_NATIVE_ID}-user",
                "role": "user",
                "content": [{"type": "input_text", "text": "sanitized incident witness"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{SESSION_NATIVE_ID}-assistant",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "sanitized terminal response"}],
            },
        },
        {
            "type": "wire_padding",
            "revision": revision,
            "padding": "x" * (128 + revision % 17),
        },
    ]
    prefix = b"".join(json.dumps(record, sort_keys=True).encode() + b"\n" for record in records[:-1])
    if not terminal:
        return prefix + json.dumps(records[-1], sort_keys=True).encode() + b"\n"

    padding_prefix = (
        b'{"padding":"'
        + str(revision).encode("ascii")
        + b'","revision":'
        + str(revision).encode("ascii")
        + b',"type":"wire_padding","value":"'
    )
    padding_suffix = b'"}\n'
    padding_size = TERMINAL_WIRE_BYTES - len(prefix) - len(padding_prefix) - len(padding_suffix)
    if padding_size <= 0:
        raise AssertionError(f"terminal padding underflow: {padding_size}")
    payload = prefix + padding_prefix + (b"x" * padding_size) + padding_suffix
    assert len(payload) == TERMINAL_WIRE_BYTES
    return payload


def _incident_program() -> CorpusProgram:
    operations = tuple(
        Acquire(
            f"revision-{revision:03d}",
            RawArtifact(
                artifact_id=f"revision-{revision:03d}",
                payload=_codex_payload(revision, terminal=revision == REVISION_COUNT - 1),
                source_name="codex",
                source_path=SOURCE_PATH,
                source_index=revision,
                metadata={"incident": "codex-804-sanitized", "revision": revision},
            ),
        )
        for revision in range(REVISION_COUNT)
    )
    return CorpusProgram(
        operations=(*operations, Crash("crash-after-acquire"), Restart("restart-after-crash")),
    )


def _resource_sample(root: Path) -> tuple[int, int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss) * 1024
    cpu_ms = int((usage.ru_utime + usage.ru_stime) * 1000)
    storage_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return peak_rss, cpu_ms, storage_bytes


def _phase(
    name: str,
    before: tuple[int, int, int],
    after: tuple[int, int, int],
    started: float,
    *,
    completed: int | None = None,
    total: int | None = None,
) -> WorkloadPhaseObservation:
    return WorkloadPhaseObservation(
        name=name,
        measurement_scope=MeasurementScope.PROCESS_TREE,
        wall_ms=(time.perf_counter() - started) * 1000,
        cpu_ms=float(after[1] - before[1]),
        peak_rss_bytes=max(before[0], after[0]),
        storage_bytes=after[2],
        write_io_bytes=max(0, after[2] - before[2]),
        progress_completed=completed,
        progress_total=total,
    )


def _schema_profile_id() -> str:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    packages = tuple(item.to_payload() for item in package_hashes_for_registry(registry, ("codex",)))
    encoded = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    return f"schema-profile:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _source_facts(root: Path) -> tuple[int, int, int, tuple[str, ...]]:
    with sqlite3.connect(root / "source.db") as conn:
        row = conn.execute(
            """
            SELECT COUNT(*), MAX(blob_size), COUNT(DISTINCT source_path), COUNT(parse_error),
                   COUNT(DISTINCT revision_authority)
            FROM raw_sessions
            """
        ).fetchone()
        authorities = tuple(
            str(item[0]) for item in conn.execute("SELECT DISTINCT revision_authority FROM raw_sessions ORDER BY 1")
        )
    assert row is not None
    return int(row[0]), int(row[1]), int(row[2]), authorities


def _readiness_count(readiness: dict[str, object], key: str) -> int:
    value = readiness[key]
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise AssertionError(f"readiness field {key!r} is not count-shaped: {value!r}")
    return int(value)


@pytest.mark.timeout(300)
def test_sanitized_codex_804_revision_recovery_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise acquisition through candidate promotion at incident scale.

    Anti-vacuity: deleting the production ``AcquisitionService`` call from
    ``ProductionCorpusRuntime.acquire`` leaves the source row count at zero;
    deleting parser/convergence leaves the terminal index session absent;
    deleting the resumable candidate path leaves no paused transaction or
    inactive generation for the recovery assertions below.
    """

    root = tmp_path / "codex-804-proof"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    initialize_active_archive_root(root)
    runtime = ProductionCorpusRuntime(root)
    program = _incident_program()
    profile_id = _schema_profile_id()
    program_json = program.to_json()

    generate_before = _resource_sample(root)
    generate_started = time.perf_counter()
    # The JSON program is intentionally the canonical witness identity.  Its
    # 90 MiB terminal payload is a faithful wire-shape fixture, not a parser
    # replacement or a pre-populated database.
    generate_after = _resource_sample(root)
    phases: list[WorkloadPhaseObservation] = [
        _phase(
            "generate",
            generate_before,
            generate_after,
            generate_started,
            completed=REVISION_COUNT,
            total=REVISION_COUNT,
        )
    ]

    acquire_before = _resource_sample(root)
    acquire_started = time.perf_counter()
    run = program.run(runtime)
    acquire_after = _resource_sample(root)
    phases.append(
        _phase(
            "acquire", acquire_before, acquire_after, acquire_started, completed=REVISION_COUNT, total=REVISION_COUNT
        )
    )
    assert len(run.state.artifacts) == REVISION_COUNT
    assert run.state.crashed is False
    assert run.schedule[-2:] == ("crash-after-acquire", "restart-after-crash")
    assert len(program_json) > REVISION_COUNT

    runtime.crash()
    with pytest.raises(CorpusRuntimeCrashedError, match="runtime is crashed"):
        runtime.converge()
    runtime.restart()

    parse_before = _resource_sample(root)
    parse_started = time.perf_counter()
    CorpusProgram(operations=(Converge("parse-materialize-index"),)).run(runtime)
    parse_after = _resource_sample(root)
    phases.append(
        _phase("census", parse_before, parse_after, parse_started, completed=REVISION_COUNT, total=REVISION_COUNT)
    )

    raw_count, max_blob_size, source_path_count, authorities = _source_facts(root)
    assert raw_count == REVISION_COUNT
    assert max_blob_size == TERMINAL_WIRE_BYTES
    assert source_path_count == 1
    assert authorities
    assert set(authorities) <= {"asserted", "byte_proven", "quarantined"}
    schema_inference_receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    baseline = archive_snapshot(root)

    replay_before = _resource_sample(root)
    replay_started = time.perf_counter()
    first_pass = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=False,
            raw_batch_size=REVISION_COUNT,
            pass_byte_budget_mb=0.01,
            schema_inference_receipt_path=schema_inference_receipt_path,
        )
    )
    assert first_pass.status in {"paused", "deferred"}
    assert first_pass.materialized is False
    assert first_pass.transaction is not None
    operation_id = first_pass.transaction["operation_id"]
    assert isinstance(operation_id, str)
    assert store.active_pointer.resolve(strict=True) == active_before
    phases.append(_phase("replay", replay_before, _resource_sample(root), replay_started))

    resume_before = _resource_sample(root)
    resume_started = time.perf_counter()
    receipt = first_pass
    for _ in range(200):
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                operation_id=operation_id,
                promote=False,
                schema_inference_receipt_path=schema_inference_receipt_path,
            )
        )
        if receipt.status == "replayed":
            break
    assert receipt.status == "replayed"
    assert receipt.materialized is True
    generation_id = receipt.generation["generation_id"]
    assert isinstance(generation_id, str)
    candidate = store.load(generation_id)
    assert candidate.state == "inactive"
    assert Path(candidate.index_path).is_file()
    assert store.active_pointer.resolve(strict=True) == active_before
    phases.append(_phase("postflight", resume_before, _resource_sample(root), resume_started))

    store.promote(candidate)
    final_snapshot = archive_snapshot(root, search_queries=("sanitized",))
    baseline_sessions = next(item for item in baseline.canonical_rows if item.relation == "sessions")
    assert not baseline_sessions.rows
    sessions_relation = next(item for item in final_snapshot.canonical_rows if item.relation == "sessions")
    assert sessions_relation.rows
    assert final_snapshot.public_projections
    readiness = raw_materialization_readiness_snapshot(root)
    assert readiness["available"] is True
    assert _readiness_count(readiness, "raw_artifact_count") == _readiness_count(
        readiness, "materialized_raw_artifact_count"
    )
    assert _readiness_count(readiness, "join_gap_count") == 0
    with sqlite3.connect(root / "index.db") as conn:
        indexed = conn.execute(
            "SELECT session_id, message_count FROM sessions WHERE native_id = ?",
            (SESSION_NATIVE_ID,),
        ).fetchall()
    assert indexed == [(f"codex-session:{SESSION_NATIVE_ID}", 2)]

    quiescent = _resource_sample(root)
    phases.append(
        WorkloadPhaseObservation(
            name="quiescent",
            measurement_scope=MeasurementScope.PROCESS_TREE,
            current_rss_bytes=quiescent[0],
            storage_bytes=quiescent[2],
            cleanup_complete=True,
            quiescent=True,
        )
    )
    spec = raw_authority_fixed_point_spec(
        profile_id=profile_id,
        archive_id=f"archive:{hashlib.sha256(str(root).encode()).hexdigest()}",
    )
    workload_receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id=f"git:{_git_head()}",
        runtime_id="production-corpus-runtime",
        archive_id=f"archive:{hashlib.sha256(str(root).encode()).hexdigest()}",
        generation_id=generation_id,
        frame_id=None,
        phases=tuple(phases),
        evidence_refs=(
            "fixture:codex-804-sanitized",
            f"schema-registry:{profile_id}",
            f"candidate-generation:{generation_id}",
            f"source-raw-count:{raw_count}",
            "successor:polylogue-live-operation-receipts",
            "successor:polylogue-reindex-final-proof",
        ),
        cleanup_complete=True,
        notes=(
            "Sanitized structural witness only; no live /realm/db/polylogue access.",
            "Live confidence remains open until the named successor receipts bind the active archive.",
        ),
    )
    receipt_path = tmp_path / "codex-804-live-proof-receipt.json"
    receipt_path.write_text(
        json.dumps(workload_receipt.to_payload(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    round_trip = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert round_trip["receipt_id"] == workload_receipt.receipt_id
    assert workload_receipt.spec.inputs[0].profile_id == profile_id
    print(
        "codex-804-proof "
        + json.dumps(
            {
                "receipt_id": workload_receipt.receipt_id,
                "raw_count": raw_count,
                "max_blob_size": max_blob_size,
                "indexed_session": indexed[0][0],
                "message_count": indexed[0][1],
                "candidate_generation": generation_id,
                "resource_phases": [phase.to_payload() for phase in phases],
                "live_confidence_gap": ["polylogue-live-operation-receipts", "polylogue-reindex-final-proof"],
            },
            sort_keys=True,
        )
    )
