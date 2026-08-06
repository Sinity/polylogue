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
from dataclasses import replace
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
from polylogue.storage.index_generation import IndexGenerationStore, rebuild_source_evidence_snapshot
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_canonical_snapshot import archive_snapshot
from tests.infra.corpus_program import (
    Acquire,
    CorpusProgram,
    ProductionCorpusRuntime,
    RawArtifact,
)
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

REVISION_COUNT = 804
TERMINAL_WIRE_BYTES = 90_822_451
SESSION_NATIVE_ID = "codex-sanitized-804-session"
SOURCE_PATH = "codex/incident-804-sanitized.jsonl"
NEAR_TERMINAL_PREDECESSOR_BYTES = 32 * 1024 * 1024


def _wire_target_bytes(revision: int) -> int:
    if revision < REVISION_COUNT - 4:
        # Keep every prefix strictly larger than its predecessor without
        # making the 800 ordinary snapshots consume hundreds of megabytes.
        return 4_096 + revision * 256
    return {
        REVISION_COUNT - 4: 8 * 1024 * 1024,
        REVISION_COUNT - 3: 16 * 1024 * 1024,
        REVISION_COUNT - 2: NEAR_TERMINAL_PREDECESSOR_BYTES,
        REVISION_COUNT - 1: TERMINAL_WIRE_BYTES,
    }[revision]


def _codex_payload(revision: int, *, terminal: bool) -> bytes:
    revision_timestamp = f"2026-07-31T04:{25 + revision // 60:02d}:{20 + revision % 60:02d}Z"
    records: list[dict[str, object]] = [
        {
            "type": "session_meta",
            "payload": {
                "id": SESSION_NATIVE_ID,
                "timestamp": "2026-07-31T04:25:20Z",
                "cwd": "/sanitized/codex-804",
            },
        },
    ]
    for message_index, role in enumerate(("user", "assistant")):
        text = f"sanitized incident witness {role} baseline"
        records.append(
            {
                "type": "response_item",
                "timestamp": revision_timestamp,
                "payload": {
                    "type": "message",
                    "id": f"{SESSION_NATIVE_ID}-message-{message_index}",
                    "role": role,
                    "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}],
                },
            }
        )
    # Parsed content grows by containment.  The first milestone proves that
    # revisions differ semantically, while the later milestones keep the
    # production membership classifier's accepted frontier moving toward the
    # terminal snapshot without making the fixture 804 messages wide.
    milestone_revisions = (1, 800, 801, 802)
    for milestone in milestone_revisions:
        if revision >= milestone:
            records.append(
                {
                    "type": "response_item",
                    "timestamp": revision_timestamp,
                    "payload": {
                        "type": "message",
                        "id": f"{SESSION_NATIVE_ID}-milestone-{milestone:03d}",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"sanitized parsed milestone revision {milestone}",
                            }
                        ],
                    },
                }
            )
    if terminal:
        records.append(
            {
                "type": "response_item",
                "timestamp": revision_timestamp,
                "payload": {
                    "type": "message",
                    "id": f"{SESSION_NATIVE_ID}-terminal-803",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "sanitized terminal response revision 803"}],
                },
            }
        )
    prefix = b"".join(json.dumps(record, sort_keys=True).encode() + b"\n" for record in records)
    wire_template = json.dumps({"padding": "", "revision": revision, "type": "wire_padding"}, sort_keys=True).encode()
    padding_prefix, padding_suffix = wire_template.split(b'""', maxsplit=1)
    padding_size = _wire_target_bytes(revision) - len(prefix) - len(padding_prefix) - len(padding_suffix) - 1
    if padding_size <= 0:
        raise AssertionError(f"terminal padding underflow: {padding_size}")
    payload = prefix + padding_prefix + (b"x" * padding_size) + padding_suffix + b"\n"
    if terminal:
        assert len(payload) == TERMINAL_WIRE_BYTES
    else:
        assert len(payload) == _wire_target_bytes(revision)
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
    return CorpusProgram(operations=operations)


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
    rss_available: bool = True,
) -> WorkloadPhaseObservation:
    unavailable = ["peak_rss_bytes"]
    current_rss = after[0] if rss_available else None
    if current_rss is None:
        unavailable.append("current_rss_bytes")
    unavailable.append("write_io_bytes")
    return WorkloadPhaseObservation(
        name=name,
        measurement_scope=MeasurementScope.PROCESS_TREE,
        wall_ms=(time.perf_counter() - started) * 1000,
        cpu_ms=float(after[1] - before[1]),
        current_rss_bytes=current_rss,
        storage_bytes=after[2],
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


def _source_facts(
    root: Path,
) -> tuple[int, int, int, int, tuple[str, ...], tuple[tuple[int, int, str, str | None, str], ...]]:
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
        raw_rows = tuple(
            (
                int(row[0]),
                int(row[1]),
                str(row[2]),
                None if row[3] is None else str(row[3]),
                str(row[4]),
            )
            for row in conn.execute(
                "SELECT source_index, blob_size, raw_id, logical_source_key, lower(hex(blob_hash)) "
                "FROM raw_sessions ORDER BY blob_size, raw_id"
            )
        )
    assert row is not None
    return int(row[0]), int(row[1]), int(row[2]), int(row[3]), authorities, raw_rows


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
    assert len(program_json) > REVISION_COUNT
    first_revision_payload = _codex_payload(0, terminal=False)
    second_revision_payload = _codex_payload(1, terminal=False)
    assert first_revision_payload != second_revision_payload
    assert b"incident witness user baseline" in first_revision_payload
    assert b"parsed milestone revision 1" in second_revision_payload

    program_digest = hashlib.sha256(program_json.encode("utf-8")).hexdigest()

    census_before = _resource_sample(root)
    census_started = time.perf_counter()
    with sqlite3.connect(root / "source.db") as conn:
        pre_recovery_raw_count = int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
        pre_recovery_unresolved_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE revision_authority = 'quarantined'").fetchone()[0]
        )
        pre_recovery_membership_count = int(
            conn.execute("SELECT COUNT(DISTINCT raw_id) FROM raw_session_memberships").fetchone()[0]
        )
        pre_recovery_census_count = int(conn.execute("SELECT COUNT(*) FROM raw_membership_census").fetchone()[0])
    assert pre_recovery_raw_count == REVISION_COUNT
    assert pre_recovery_unresolved_count == REVISION_COUNT
    assert pre_recovery_membership_count == 0
    assert pre_recovery_census_count == 0
    phases.append(
        _phase(
            "census",
            census_before,
            _resource_sample(root),
            census_started,
            completed=pre_recovery_unresolved_count,
            total=REVISION_COUNT,
        )
    )

    schema_inference_receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    baseline = archive_snapshot(root)

    transaction = store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        pass_byte_budget=64 * 1024 * 1024,
    )
    operation_id = transaction.operation_id
    assert transaction.status == "running"
    transaction_path = store.transactions_root / f"{operation_id}.json"
    assert transaction_path.is_file()
    crash_script = """
import os
import sys
from pathlib import Path

from polylogue.storage.index_generation import IndexGenerationStore

root = Path(sys.argv[1])
operation_id = sys.argv[2]
transaction = IndexGenerationStore.for_archive_root(root).load_transaction(operation_id)
if transaction.status != "running" or transaction.processed_raw_count != 0:
    raise SystemExit(f"unexpected pre-replay transaction: {transaction.status} {transaction.processed_raw_count}")
os._exit(137)
"""
    replay_before = _resource_sample(root)
    replay_started = time.perf_counter()
    crashed_process = subprocess.run(
        [sys.executable, "-c", crash_script, str(root), operation_id],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert crashed_process.returncode == 137, crashed_process.stderr
    persisted = store.load_transaction(operation_id)
    assert persisted.status == "running"
    assert persisted.processed_raw_count == 0
    with sqlite3.connect(root / "source.db") as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM raw_session_memberships").fetchone()[0]) == 0
        assert int(conn.execute("SELECT COUNT(*) FROM raw_membership_census").fetchone()[0]) == 0
    assert store.active_pointer.resolve(strict=True) == active_before
    phases.append(
        _phase(
            "replay",
            replay_before,
            _resource_sample(root),
            replay_started,
            completed=0,
            total=REVISION_COUNT,
            rss_available=False,
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
    phases.append(
        _phase(
            "postflight",
            resume_before,
            _resource_sample(root),
            resume_started,
            completed=REVISION_COUNT,
            total=REVISION_COUNT,
            rss_available=False,
        )
    )

    raw_count, max_blob_size, source_path_count, parse_error_count, authorities, raw_rows = _source_facts(root)
    assert raw_count == REVISION_COUNT
    assert max_blob_size == TERMINAL_WIRE_BYTES
    assert source_path_count == 1
    assert parse_error_count == 0
    assert authorities
    assert set(authorities) <= {"asserted", "byte_proven", "quarantined"}
    assert tuple(row[1] for row in raw_rows) == tuple(
        _wire_target_bytes(revision) for revision in range(REVISION_COUNT)
    )
    assert raw_rows[-1][1] == TERMINAL_WIRE_BYTES
    assert raw_rows[-2][1] >= NEAR_TERMINAL_PREDECESSOR_BYTES
    terminal_blob_hash = raw_rows[-1][4]
    assert raw_rows[-1][4] == hashlib.sha256(_codex_payload(REVISION_COUNT - 1, terminal=True)).hexdigest()
    assert raw_rows[-1][3] is None
    with sqlite3.connect(root / "source.db") as conn:
        post_recovery_membership_count = int(
            conn.execute("SELECT COUNT(DISTINCT raw_id) FROM raw_session_memberships").fetchone()[0]
        )
        membership_keys = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT logical_source_key FROM raw_session_memberships ORDER BY logical_source_key"
            )
        )
    assert post_recovery_membership_count == REVISION_COUNT
    assert len(membership_keys) == 1
    terminal_logical_source_key = membership_keys[0]

    store.promote(candidate)
    final_snapshot = archive_snapshot(root, search_queries=("sanitized",))
    baseline_sessions = next(item for item in baseline.canonical_rows if item.relation == "sessions")
    assert not baseline_sessions.rows
    sessions_relation = next(item for item in final_snapshot.canonical_rows if item.relation == "sessions")
    assert sessions_relation.rows
    public = dict(final_snapshot.public_projections)
    indexed_session_id = f"codex-session:{SESSION_NATIVE_ID}"
    assert public[f"summary:{indexed_session_id}"]
    assert public[f"tree:{indexed_session_id}"]
    assert public["search:sanitized"]
    readiness = raw_materialization_readiness_snapshot(root)
    assert readiness["available"] is True
    assert _readiness_count(readiness, "raw_artifact_count") == REVISION_COUNT
    materialized_raw_count = _readiness_count(readiness, "materialized_raw_artifact_count")
    assert 0 < materialized_raw_count <= REVISION_COUNT
    assert _readiness_count(readiness, "join_gap_count") == REVISION_COUNT - materialized_raw_count
    with sqlite3.connect(root / "index.db") as conn:
        indexed = conn.execute(
            "SELECT session_id, raw_id, message_count FROM sessions WHERE native_id = ?",
            (SESSION_NATIVE_ID,),
        ).fetchall()
        terminal_block_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE search_text LIKE ?",
                ("%sanitized terminal response revision 803%",),
            ).fetchone()[0]
        )
        selected_heads = conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            (terminal_logical_source_key,),
        ).fetchall()
    assert len(indexed) == 1
    assert len(selected_heads) == 1
    selected_raw_id = str(selected_heads[0][0])
    assert indexed == [(f"codex-session:{SESSION_NATIVE_ID}", selected_raw_id, 7)]
    with sqlite3.connect(root / "source.db") as conn:
        selected_source_row = conn.execute(
            "SELECT blob_size, lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?",
            (selected_raw_id,),
        ).fetchone()
    assert selected_source_row == (TERMINAL_WIRE_BYTES, terminal_blob_hash)
    assert terminal_block_count > 0

    quiescent = _resource_sample(root)
    phases.append(
        WorkloadPhaseObservation(
            name="quiescent",
            measurement_scope=MeasurementScope.PROCESS_TREE,
            current_rss_bytes=quiescent[0],
            storage_bytes=quiescent[2],
            cleanup_complete=True,
            quiescent=True,
            unavailable=("peak_rss_bytes", "write_io_bytes")
            if quiescent[0] is not None
            else ("current_rss_bytes", "peak_rss_bytes", "write_io_bytes"),
        )
    )
    spec = raw_authority_fixed_point_spec(
        profile_id=profile_id,
        archive_id=f"archive:{hashlib.sha256(str(root).encode()).hexdigest()}",
    )
    input_ref = spec.inputs[0]
    spec = replace(
        spec,
        inputs=(
            replace(
                input_ref,
                input_id=f"{input_ref.input_id}:program-json-sha256:{program_digest}",
                corpus_id=f"program-json:sha256:{program_digest}",
            ),
        ),
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
            f"program-json-sha256:{program_digest}",
            f"source-raw-count:{raw_count}",
            f"source-parse-error-count:{parse_error_count}",
            f"recovery-unresolved-before-crash:{pre_recovery_unresolved_count}",
            f"recovery-memberships-after-restart:{post_recovery_membership_count}",
            "restart-boundary:subprocess-persisted-transaction",
            "successor:polylogue-live-operation-receipts",
            "successor:polylogue-reindex-final-proof",
        ),
        cleanup_complete=True,
        notes=(
            "Sanitized structural witness only; no live /realm/db/polylogue access.",
            "Fixture setup includes 804 payload construction, schema hashing, and canonical program serialization.",
            f"Serialized program digest is sha256:{program_digest} and is bound into the receipt input identity.",
            "Replay and postflight subprocess RSS is unavailable because statm samples only the pytest parent; storage growth is not reported as write I/O.",
            "The crash boundary hard-exits after durable transaction creation with all 804 authority rows unresolved; a fresh process resumes and resolves the cohort.",
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
                "program_digest": program_digest,
                "parse_error_count": parse_error_count,
                "terminal_raw_id": selected_raw_id,
                "indexed_session": indexed[0][0],
                "message_count": indexed[0][2],
                "candidate_generation": generation_id,
                "resource_phases": [phase.to_payload() for phase in phases],
                "live_confidence_gap": ["polylogue-live-operation-receipts", "polylogue-reindex-final-proof"],
            },
            sort_keys=True,
        )
    )
