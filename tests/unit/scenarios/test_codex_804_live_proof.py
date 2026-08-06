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
import os
import resource
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
                "content": [{"type": "input_text", "text": f"sanitized incident witness revision {revision}"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{SESSION_NATIVE_ID}-assistant",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"sanitized terminal response revision {revision}"}],
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


def _current_rss_bytes() -> int | None:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _resource_sample(root: Path) -> tuple[int | None, int, int]:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    children_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_ms = int((self_usage.ru_utime + self_usage.ru_stime + children_usage.ru_utime + children_usage.ru_stime) * 1000)
    storage_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return _current_rss_bytes(), cpu_ms, storage_bytes


def _phase(
    name: str,
    before: tuple[int | None, int, int],
    after: tuple[int | None, int, int],
    started: float,
    *,
    completed: int | None = None,
    total: int | None = None,
) -> WorkloadPhaseObservation:
    unavailable = ["peak_rss_bytes"]
    if after[0] is None:
        unavailable.append("current_rss_bytes")
    return WorkloadPhaseObservation(
        name=name,
        measurement_scope=MeasurementScope.PROCESS_TREE,
        wall_ms=(time.perf_counter() - started) * 1000,
        cpu_ms=float(after[1] - before[1]),
        current_rss_bytes=after[0],
        storage_bytes=after[2],
        write_io_bytes=max(0, after[2] - before[2]),
        progress_completed=completed,
        progress_total=total,
        unavailable=tuple(unavailable),
    )


def _schema_profile_id() -> str:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    packages = tuple(item.to_payload() for item in package_hashes_for_registry(registry, ("codex",)))
    encoded = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    return f"schema-profile:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _source_facts(root: Path) -> tuple[int, int, int, int, tuple[str, ...]]:
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
    return int(row[0]), int(row[1]), int(row[2]), int(row[3]), authorities


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

    setup_before = _resource_sample(root)
    setup_started = time.perf_counter()
    program = _incident_program()
    profile_id = _schema_profile_id()
    program_json = program.to_json()
    # The JSON program is intentionally the canonical witness identity.  Its
    # 90 MiB terminal payload is a faithful wire-shape fixture, not a parser
    # replacement or a pre-populated database.
    setup_after = _resource_sample(root)
    phases: list[WorkloadPhaseObservation] = [
        _phase(
            "generate",
            setup_before,
            setup_after,
            setup_started,
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
    first_revision_payload = _codex_payload(0, terminal=False)
    second_revision_payload = _codex_payload(1, terminal=False)
    assert first_revision_payload != second_revision_payload
    assert b"incident witness revision 0" in first_revision_payload
    assert b"incident witness revision 1" in second_revision_payload

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

    raw_count, max_blob_size, source_path_count, parse_error_count, authorities = _source_facts(root)
    assert raw_count == REVISION_COUNT
    assert max_blob_size == TERMINAL_WIRE_BYTES
    assert source_path_count == 1
    assert parse_error_count == 0
    assert authorities
    assert set(authorities) <= {"asserted", "byte_proven", "quarantined"}
    schema_inference_receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    baseline = archive_snapshot(root)

    crash_script = """
import os
import sys
from pathlib import Path

from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

root = Path(sys.argv[1])
receipt = Path(sys.argv[2])
marker = Path(sys.argv[3])
result = rebuild_index_from_source_sync(
    RebuildIndexRequest(
        archive_root=root,
        promote=False,
        raw_batch_size=804,
        pass_byte_budget_mb=1.0,
        schema_inference_receipt_path=receipt,
    )
)
if result.status not in {"paused", "deferred"} or result.materialized:
    raise SystemExit(f"unexpected crash-boundary result: {result.status} {result.materialized}")
assert result.transaction is not None
marker.write_text(str(result.transaction["operation_id"]), encoding="ascii")
with marker.open("a", encoding="ascii") as handle:
    handle.flush()
    os.fsync(handle.fileno())
os._exit(137)
"""
    operation_marker = tmp_path / "codex-804-operation-id.txt"
    replay_before = _resource_sample(root)
    replay_started = time.perf_counter()
    crashed_process = subprocess.run(
        [sys.executable, "-c", crash_script, str(root), str(schema_inference_receipt_path), str(operation_marker)],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert crashed_process.returncode == 137, crashed_process.stderr
    operation_id = operation_marker.read_text(encoding="ascii")
    assert operation_id
    persisted = store.load_transaction(operation_id)
    assert persisted.status in {"paused", "deferred"}
    assert persisted.processed_raw_count > 0
    assert tuple(store.transactions_root.joinpath(f"{operation_id}.receipts").glob("pass-*.json"))
    assert store.active_pointer.resolve(strict=True) == active_before
    phases.append(
        _phase(
            "replay",
            replay_before,
            _resource_sample(root),
            replay_started,
            completed=persisted.processed_raw_count,
            total=REVISION_COUNT,
        )
    )

    resume_before = _resource_sample(root)
    resume_started = time.perf_counter()
    resume_script = """
import sys
from pathlib import Path

from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

root = Path(sys.argv[1])
operation_id = sys.argv[2]
receipt = Path(sys.argv[3])
for _ in range(200):
    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            operation_id=operation_id,
            promote=False,
            schema_inference_receipt_path=receipt,
        )
    )
    if result.status == "replayed" and result.materialized:
        print(result.generation["generation_id"])
        break
else:
    raise SystemExit("persisted rebuild did not reach replayed")
"""
    resumed_process = subprocess.run(
        [sys.executable, "-c", resume_script, str(root), operation_id, str(schema_inference_receipt_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    generation_id = resumed_process.stdout.strip().splitlines()[-1]
    assert generation_id.startswith("gen-")
    persisted_after_restart = store.load_transaction(operation_id)
    assert persisted_after_restart.status == "ready"
    candidate = store.load(generation_id)
    assert candidate.state == "inactive"
    assert Path(candidate.index_path).is_file()
    assert store.active_pointer.resolve(strict=True) == active_before
    phases.append(_phase("postflight", resume_before, _resource_sample(root), resume_started))

    store.promote(candidate)
    final_snapshot = archive_snapshot(root, search_queries=("sanitized", "revision"))
    baseline_sessions = next(item for item in baseline.canonical_rows if item.relation == "sessions")
    assert not baseline_sessions.rows
    sessions_relation = next(item for item in final_snapshot.canonical_rows if item.relation == "sessions")
    assert sessions_relation.rows
    public = dict(final_snapshot.public_projections)
    indexed_session_id = f"codex-session:{SESSION_NATIVE_ID}"
    assert public[f"summary:{indexed_session_id}"]
    assert public[f"tree:{indexed_session_id}"]
    assert public["search:revision"]
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
            unavailable=("peak_rss_bytes",) if quiescent[0] is not None else ("current_rss_bytes", "peak_rss_bytes"),
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
            f"source-parse-error-count:{parse_error_count}",
            "restart-boundary:subprocess-persisted-transaction",
            "successor:polylogue-live-operation-receipts",
            "successor:polylogue-reindex-final-proof",
        ),
        cleanup_complete=True,
        notes=(
            "Sanitized structural witness only; no live /realm/db/polylogue access.",
            "Fixture setup includes 804 payload construction, schema hashing, and canonical program serialization.",
            "Current RSS is sampled per phase; phase-local peak RSS is unavailable and marked explicitly.",
            "The crash boundary hard-exits after a persisted bounded pass; a fresh process resumes the transaction.",
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
                "parse_error_count": parse_error_count,
                "indexed_session": indexed[0][0],
                "message_count": indexed[0][1],
                "candidate_generation": generation_id,
                "resource_phases": [phase.to_payload() for phase in phases],
                "live_confidence_gap": ["polylogue-live-operation-receipts", "polylogue-reindex-final-proof"],
            },
            sort_keys=True,
        )
    )
