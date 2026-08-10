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
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.product import raw_authority
from polylogue.scenarios import (
    MeasurementScope,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)
from polylogue.scenarios.workload import raw_authority_fixed_point_spec
from polylogue.schemas.operator.receipt import package_hashes_for_registry
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.sources.revision_backfill import RAW_AUTHORITY_PARSER_FINGERPRINT, backfill_historical_revision_evidence
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.index_generation import IndexGenerationStore, rebuild_source_evidence_snapshot
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_canonical_snapshot import archive_snapshot
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt
from tests.infra.whale_fixtures import (
    WHALE_FIXTURE_DIMENSIONS,
    CodexRevisionChainFixture,
    acquire_codex_revision_chain,
)

REVISION_COUNT = WHALE_FIXTURE_DIMENSIONS.revision_count
TERMINAL_WIRE_BYTES = WHALE_FIXTURE_DIMENSIONS.terminal_wire_bytes
SESSION_NATIVE_ID = CodexRevisionChainFixture().session_native_id
SOURCE_PATH = "codex/incident-804-sanitized.jsonl"
_BASELINE_MESSAGE_IDS = tuple(f"{SESSION_NATIVE_ID}-message-{index}" for index in range(2))
_BASELINE_MESSAGE_TIMESTAMPS = ("2026-07-31T04:25:20Z", "2026-07-31T04:25:20Z")


def _baseline_message_timestamps(path: Path) -> tuple[str, ...]:
    timestamps: dict[str, str] = {}
    with path.open("rb") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            record_payload = record.get("payload")
            if not isinstance(record_payload, dict) or record_payload.get("type") != "message":
                continue
            message_id = record_payload.get("id")
            timestamp = record.get("timestamp")
            if isinstance(message_id, str) and message_id in _BASELINE_MESSAGE_IDS:
                if not isinstance(timestamp, str):
                    raise AssertionError(f"baseline message {message_id} has no string timestamp")
                timestamps[message_id] = timestamp
                if len(timestamps) == len(_BASELINE_MESSAGE_IDS):
                    break
    return tuple(timestamps[message_id] for message_id in _BASELINE_MESSAGE_IDS)


def _assert_baseline_timestamps_are_stable(actual: tuple[str, ...]) -> None:
    """Require every observed wire revision to retain the baseline timestamps."""
    assert actual == _BASELINE_MESSAGE_TIMESTAMPS


def _assert_precheckpoint_state(
    *,
    status: str,
    processed_raw_count: int,
    last_raw_id: str | None,
    last_blob_hash_hex: str | None,
    receipt_names: tuple[str, ...],
) -> None:
    """Require a kill before the durable paused checkpoint to stay running."""
    assert status == "running"
    assert processed_raw_count == 0
    assert last_raw_id is None
    assert last_blob_hash_hex is None
    assert receipt_names == ()


def _assert_exact_authority_census(
    *,
    raw_ids: set[str],
    authority_rows: tuple[tuple[str, str, int], ...],
    authority_counts: tuple[tuple[str, int], ...],
    parser_census_rows: tuple[tuple[str, str, str, str, str], ...],
    application_rows: tuple[tuple[str, str, str | None], ...],
    head_rows: tuple[tuple[str, str, str, str, int], ...],
    source_key: str,
    session_id: str,
    terminal_raw_id: str,
) -> None:
    """Require one unambiguous authority and application for every raw."""
    assert len(raw_ids) == REVISION_COUNT
    assert len(authority_rows) == REVISION_COUNT
    assert {row[0] for row in authority_rows} == raw_ids
    assert all(row[1] == "byte_proven" and row[2] == 1 for row in authority_rows)
    assert authority_counts == (("byte_proven", REVISION_COUNT),)
    assert len(parser_census_rows) == REVISION_COUNT
    assert {row[0] for row in parser_census_rows} == raw_ids
    assert all(
        row[1] == RAW_AUTHORITY_PARSER_FINGERPRINT
        and row[2] == "complete"
        and row[4].startswith("parser-observed:")
        and isinstance(json.loads(row[3]), list)
        and len(json.loads(row[3])) == 1
        for row in parser_census_rows
    )
    assert len(application_rows) == REVISION_COUNT
    assert {row[0] for row in application_rows} == raw_ids
    assert {row[1] for row in application_rows} <= {"selected_baseline", "applied_append", "superseded"}
    assert all(row[2] == terminal_raw_id for row in application_rows)
    assert head_rows == ((source_key, session_id, terminal_raw_id, "byte", TERMINAL_WIRE_BYTES),)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@pytest.mark.timeout(900)
@pytest.mark.uses_real_clock("waits for a real subprocess replay checkpoint and kill/resume boundary")
def test_sanitized_codex_804_revision_recovery_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise acquisition through candidate promotion at incident scale.

    Anti-vacuity: deleting the production ``AcquisitionService`` call from
    ``acquire_codex_revision_chain`` leaves the source row count at zero;
    deleting parser/convergence leaves the terminal index session absent;
    deleting the resumable candidate path leaves no paused transaction or
    inactive generation for the recovery assertions below.
    """

    root = tmp_path / "codex-804-proof"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    initialize_active_archive_root(root)
    fixture = CodexRevisionChainFixture()
    source_path = root / "fixture-sources" / SOURCE_PATH

    setup_before = _resource_sample(root)
    setup_started = time.perf_counter()
    profile_id = _schema_profile_id()
    setup_after = _resource_sample(root)
    phases: list[WorkloadPhaseObservation] = [
        _phase(
            "generate",
            setup_before,
            setup_after,
            setup_started,
            completed=0,
            total=REVISION_COUNT,
        )
    ]

    acquire_before = _resource_sample(root)
    acquire_started = time.perf_counter()
    observed_revisions: list[int] = []

    def observe_revision(revision: int, path: Path) -> None:
        assert revision == len(observed_revisions)
        _assert_baseline_timestamps_are_stable(_baseline_message_timestamps(path))
        observed_revisions.append(revision)

    acquired_raw_ids, sizes, fixture_sha256s = acquire_codex_revision_chain(
        root,
        fixture,
        source_path,
        revision_observer=observe_revision,
    )
    with sqlite3.connect(root / "source.db") as conn:
        persisted_sizes = tuple(
            int(row[0]) for row in conn.execute("SELECT blob_size FROM raw_sessions ORDER BY blob_size")
        )
    manifest_path = fixture.write_manifest(root / "fixture-manifest.json", sizes, fixture_sha256s)
    fixture_manifest = manifest_path.read_text(encoding="utf-8")
    acquire_after = _resource_sample(root)
    phases.append(
        _phase(
            "acquire", acquire_before, acquire_after, acquire_started, completed=REVISION_COUNT, total=REVISION_COUNT
        )
    )
    assert len(acquired_raw_ids) == REVISION_COUNT
    assert observed_revisions == list(range(REVISION_COUNT))
    assert len(set(acquired_raw_ids)) == REVISION_COUNT
    assert len(sizes) == REVISION_COUNT
    assert sizes[0] == 4_096
    assert sizes[-2] == WHALE_FIXTURE_DIMENSIONS.near_terminal_predecessor_bytes
    assert sizes[-1] == TERMINAL_WIRE_BYTES
    assert all(current > previous for previous, current in pairwise(sizes))
    assert persisted_sizes == sizes
    manifest_payload = json.loads(fixture_manifest)
    assert manifest_payload["fixture_id"] == WHALE_FIXTURE_DIMENSIONS.fixture_id
    assert manifest_payload["session_native_id"] == SESSION_NATIVE_ID
    assert manifest_payload["revision_sizes"] == list(sizes)
    assert manifest_payload["revision_sha256"] == list(fixture_sha256s)
    assert manifest_payload["dimensions"] == dict(WHALE_FIXTURE_DIMENSIONS.manifest_dimensions())
    with source_path.open("rb") as handle:
        first_source_bytes = handle.read(8_192)
    assert b"sanitized incident witness user baseline" in first_source_bytes
    assert b"sanitized parsed milestone revision 1" in first_source_bytes
    _assert_baseline_timestamps_are_stable(_baseline_message_timestamps(source_path))

    fixture_manifest_digest = hashlib.sha256(fixture_manifest.encode("utf-8")).hexdigest()

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
    whale_candidate = raw_authority.whale_pass_candidate(
        Config(archive_root=root, render_root=root / "render", sources=[]),
        ordinary_max_payload_bytes=WHALE_FIXTURE_DIMENSIONS.ordinary_blob_limit_bytes,
        whale_max_payload_bytes=WHALE_FIXTURE_DIMENSIONS.whale_blob_limit_bytes,
    )
    assert whale_candidate is not None
    assert whale_candidate == acquired_raw_ids[0]
    whale_component_bytes = sum(sizes)
    assert (
        WHALE_FIXTURE_DIMENSIONS.ordinary_blob_limit_bytes
        < whale_component_bytes
        <= WHALE_FIXTURE_DIMENSIONS.whale_blob_limit_bytes
    )
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

    # Source remediation is a phase-2 input to candidate construction. Run
    # the production remediation route in an isolated archive, then carry its
    # finalized durable source tier into this fresh-index candidate fixture.
    # This keeps the rebuild receipt frozen after remediation, so a candidate
    # replay cannot hide a source mutation behind its own provenance gate.
    # The replay phase intentionally begins before source remediation and the
    # crash boundary so its receipt covers the complete recovery envelope.
    replay_before = _resource_sample(root)
    replay_started = time.perf_counter()
    source_ready_root = tmp_path / "codex-804-source-ready"
    initialize_active_archive_root(source_ready_root)
    shutil.copy2(root / "source.db", source_ready_root / "source.db")
    shutil.copytree(root / "blob", source_ready_root / "blob")
    backfill_historical_revision_evidence(source_ready_root, ingest_workers=1, bulk_fts=True)
    shutil.copy2(source_ready_root / "source.db", root / "source.db")

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
    precheckpoint_script = """
import os
import sys
from pathlib import Path

from polylogue.storage.index_generation import IndexGenerationStore

original_checkpoint = IndexGenerationStore.checkpoint_transaction

def terminate_before_page_checkpoint(self, transaction, **kwargs):
    if kwargs.get("status") == "paused" and kwargs.get("processed_raw_count", 0) > 0:
        os._exit(97)
    return original_checkpoint(self, transaction, **kwargs)

IndexGenerationStore.checkpoint_transaction = terminate_before_page_checkpoint
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

root = Path(sys.argv[1])
operation_id = sys.argv[2]
receipt = Path(sys.argv[3])
rebuild_index_from_source_sync(
    RebuildIndexRequest(
        archive_root=root,
        operation_id=operation_id,
        promote=False,
        schema_inference_receipt_path=receipt,
        raw_batch_size=8,
    )
)
raise SystemExit("checkpoint seam was not reached")
"""
    precheckpoint_process = subprocess.run(
        [sys.executable, "-c", precheckpoint_script, str(root), operation_id, str(schema_inference_receipt_path)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert precheckpoint_process.returncode == 97, (
        f"pre-checkpoint boundary did not terminate at the checkpoint seam: "
        f"returncode={precheckpoint_process.returncode}; "
        f"stdout={precheckpoint_process.stdout}; stderr={precheckpoint_process.stderr}"
    )
    persisted_before_page = store.load_transaction(operation_id)
    # Mutation that advances the cursor before checkpoint_transaction returns
    # fails these pre-checkpoint invariants and the receipt census below.
    receipt_directory = store.transactions_root / f"{operation_id}.receipts"
    _assert_precheckpoint_state(
        status=persisted_before_page.status,
        processed_raw_count=persisted_before_page.processed_raw_count,
        last_raw_id=persisted_before_page.last_raw_id,
        last_blob_hash_hex=persisted_before_page.last_blob_hash_hex,
        receipt_names=tuple(path.name for path in receipt_directory.glob("pass-*.json")),
    )
    precheckpoint_generation = store.load(persisted_before_page.generation_id)
    with sqlite3.connect(precheckpoint_generation.index_path) as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]) > 0
    committed_page = store.next_raw_page(persisted_before_page, limit=8)
    committed_page_raw_ids = tuple(row[0] for row in committed_page.rows)
    assert len(committed_page_raw_ids) == 8
    replay_script = """
import sys
from pathlib import Path

from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

root = Path(sys.argv[1])
operation_id = sys.argv[2]
receipt = Path(sys.argv[3])
result = rebuild_index_from_source_sync(
    RebuildIndexRequest(
        archive_root=root,
        operation_id=operation_id,
        promote=False,
        schema_inference_receipt_path=receipt,
        raw_batch_size=8,
    )
)
print(result.status)
"""
    replay_process = subprocess.Popen(
        [sys.executable, "-c", replay_script, str(root), operation_id, str(schema_inference_receipt_path)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    replay_deadline = time.monotonic() + 120
    while time.monotonic() < replay_deadline:
        persisted = store.load_transaction(operation_id)
        if persisted.processed_raw_count > 0:
            replay_process.kill()
            break
        returncode = replay_process.poll()
        if returncode is not None:
            stdout = replay_process.stdout.read() if replay_process.stdout is not None else ""
            stderr = replay_process.stderr.read() if replay_process.stderr is not None else ""
            raise AssertionError(
                f"replay exited before durable progress: {returncode}; stdout={stdout}; stderr={stderr}"
            )
        time.sleep(0.1)
    else:
        replay_process.kill()
        raise AssertionError("replay did not checkpoint durable progress before the interruption deadline")
    replay_returncode = replay_process.wait(timeout=30)
    assert replay_returncode == -9
    persisted = store.load_transaction(operation_id)
    assert persisted.status == "paused"
    assert persisted.processed_raw_count == len(committed_page_raw_ids)
    interrupted_generation = store.load(persisted.generation_id)
    with sqlite3.connect(interrupted_generation.index_path) as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]) > 0
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
import json
import sys
from pathlib import Path

import polylogue.maintenance.replay as replay_module

trace = Path(sys.argv[4])
real_replay = replay_module.rebuild_index_from_source

async def recording_replay(*args, **kwargs):
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(list(kwargs["raw_ids"]), sort_keys=True) + "\\n")
    return await real_replay(*args, **kwargs)

replay_module.rebuild_index_from_source = recording_replay
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
    replay_trace_path = tmp_path / "rebuild-replay-trace.jsonl"
    resumed_process = subprocess.run(
        [
            sys.executable,
            "-c",
            resume_script,
            str(root),
            operation_id,
            str(schema_inference_receipt_path),
            str(replay_trace_path),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed_process.returncode == 0, (
        f"restart subprocess failed: returncode={resumed_process.returncode}; "
        f"stdout={resumed_process.stdout}; stderr={resumed_process.stderr}"
    )
    generation_id = resumed_process.stdout.strip().splitlines()[-1]
    assert generation_id.startswith("gen-")
    persisted_after_restart = store.load_transaction(operation_id)
    assert persisted_after_restart.status == "ready"
    candidate = store.load(generation_id)
    assert candidate.state == "inactive"
    assert Path(candidate.index_path).is_file()
    assert store.active_pointer.resolve(strict=True) == active_before
    resumed_raw_pages = tuple(
        tuple(json.loads(line)) for line in replay_trace_path.read_text(encoding="utf-8").splitlines()
    )
    assert resumed_raw_pages, "restart observed no production replay selections for the suffix"
    resumed_raw_sequence = tuple(raw_id for page in resumed_raw_pages for raw_id in page)
    with sqlite3.connect(root / "source.db") as conn:
        all_raw_sequence = tuple(
            str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY blob_hash, raw_id")
        )
    assert all(isinstance(raw_id, str) for page in resumed_raw_pages for raw_id in page)
    assert len(resumed_raw_sequence) == len(set(resumed_raw_sequence)), "restart replayed a raw more than once"
    assert len(all_raw_sequence) == len(set(all_raw_sequence))
    assert set(committed_page_raw_ids).isdisjoint(set(resumed_raw_sequence))
    assert resumed_raw_sequence == all_raw_sequence[len(committed_page_raw_ids) :]

    idempotence_script = """
import json
import sys
from pathlib import Path

import polylogue.maintenance.replay as replay_module

trace = Path(sys.argv[4])
real_replay = replay_module.rebuild_index_from_source

async def recording_replay(*args, **kwargs):
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(list(kwargs["raw_ids"]), sort_keys=True) + "\\n")
    return await real_replay(*args, **kwargs)

replay_module.rebuild_index_from_source = recording_replay
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

root = Path(sys.argv[1])
operation_id = sys.argv[2]
receipt = Path(sys.argv[3])
result = rebuild_index_from_source_sync(
    RebuildIndexRequest(
        archive_root=root,
        operation_id=operation_id,
        promote=False,
        schema_inference_receipt_path=receipt,
    )
)
print(json.dumps({"status": result.status, "generation_id": result.generation["generation_id"]}))
"""
    idempotence_trace_path = tmp_path / "rebuild-idempotence-trace.jsonl"
    idempotent_process = subprocess.run(
        [
            sys.executable,
            "-c",
            idempotence_script,
            str(root),
            operation_id,
            str(schema_inference_receipt_path),
            str(idempotence_trace_path),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert idempotent_process.returncode == 0, (
        f"idempotent restart failed: returncode={idempotent_process.returncode}; "
        f"stdout={idempotent_process.stdout}; stderr={idempotent_process.stderr}"
    )
    idempotent_result = json.loads(idempotent_process.stdout.strip().splitlines()[-1])
    assert idempotent_result == {"status": "replayed", "generation_id": generation_id}
    idempotent_pages = tuple(
        tuple(json.loads(line)) for line in idempotence_trace_path.read_text(encoding="utf-8").splitlines()
    )
    assert all(not page for page in idempotent_pages), "idempotent restart replayed a raw revision"
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
    assert authorities == ("byte_proven",)
    assert tuple(row[1] for row in raw_rows) == sizes
    assert tuple(row[4] for row in raw_rows) == fixture_sha256s
    assert raw_rows[-1][1] == TERMINAL_WIRE_BYTES
    assert raw_rows[-2][1] == WHALE_FIXTURE_DIMENSIONS.near_terminal_predecessor_bytes
    terminal_raw_id = raw_rows[-1][2]
    terminal_blob_hash = raw_rows[-1][4]
    assert raw_rows[-1][4] == fixture_sha256s[-1] == _file_sha256(source_path)
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
        source_keys = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT logical_source_key FROM raw_sessions "
                "WHERE logical_source_key IS NOT NULL ORDER BY logical_source_key"
            )
        )
        raw_ids = {str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions")}
        source_authority_rows = tuple(
            (str(row[0]), str(row[1]))
            for row in conn.execute("SELECT raw_id, revision_authority FROM raw_sessions ORDER BY raw_id")
        )
        authority_counts = tuple(
            (str(row[0]), int(row[1]))
            for row in conn.execute("SELECT revision_authority, COUNT(*) FROM raw_sessions GROUP BY revision_authority")
        )
        membership_rows = tuple(
            (str(row[0]), str(row[1]), None if row[2] is None else str(row[2]))
            for row in conn.execute(
                "SELECT raw_id, revision_authority, decision FROM raw_session_memberships ORDER BY raw_id"
            )
        )
        census_rows = tuple(
            (str(row[0]), str(row[1]))
            for row in conn.execute("SELECT raw_id, status FROM raw_membership_census ORDER BY raw_id")
        )
        parser_census_rows = tuple(
            (str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]))
            for row in conn.execute(
                "SELECT raw_id, parser_fingerprint, status, logical_keys_json, detail "
                "FROM raw_authority_parser_census ORDER BY raw_id"
            )
        )
    # This fixture is the byte-revision authority route, so semantic
    # membership census remains exactly empty. The application ledger below
    # is the production coverage relation for all 804 byte revisions.
    assert post_recovery_membership_count == 0
    # Mutation that omits one authority row from the final census fails the
    # exact raw, membership, and complete-census conservation below.
    assert membership_rows == ()
    assert census_rows == ()
    assert len(source_keys) == 1
    terminal_logical_source_key = membership_keys[0] if membership_keys else source_keys[0]

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
        compaction_event_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_events WHERE session_id = ? AND event_type = 'compaction'",
                (f"codex-session:{SESSION_NATIVE_ID}",),
            ).fetchone()[0]
        )
        attachment_summary_count = int(
            conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text LIKE '%sha256_base64=%'").fetchone()[0]
        )
        attachment_image_block_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE block_type = 'image' AND search_text LIKE '%sha256_base64=%'"
            ).fetchone()[0]
        )
        selected_heads = conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            (terminal_logical_source_key,),
        ).fetchall()
        application_rows = conn.execute(
            "SELECT raw_id, decision, accepted_raw_id FROM raw_revision_applications ORDER BY raw_id"
        ).fetchall()
        head_rows = tuple(
            (str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]))
            for row in conn.execute(
                "SELECT logical_source_key, session_id, accepted_raw_id, accepted_frontier_kind, accepted_frontier "
                "FROM raw_revision_heads ORDER BY logical_source_key"
            )
        )
    application_counts = {
        str(raw_id): sum(1 for row in application_rows if str(row[0]) == str(raw_id))
        for raw_id, _ in source_authority_rows
    }
    authority_rows = tuple(
        (raw_id, authority, application_counts.get(raw_id, 0)) for raw_id, authority in source_authority_rows
    )
    _assert_exact_authority_census(
        raw_ids=raw_ids,
        authority_rows=authority_rows,
        authority_counts=authority_counts,
        parser_census_rows=parser_census_rows,
        application_rows=tuple(
            (str(row[0]), str(row[1]), None if row[2] is None else str(row[2])) for row in application_rows
        ),
        head_rows=head_rows,
        source_key=source_keys[0],
        session_id=f"codex-session:{SESSION_NATIVE_ID}",
        terminal_raw_id=terminal_raw_id,
    )
    assert len(application_rows) == REVISION_COUNT
    assert {str(row[0]) for row in application_rows} == raw_ids
    assert {str(row[1]) for row in application_rows} <= {"selected_baseline", "applied_append", "superseded"}
    accepted_application_ids = {str(row[2]) for row in application_rows if row[2] is not None}
    assert len(accepted_application_ids) == 1
    assert accepted_application_ids <= raw_ids
    assert accepted_application_ids == {terminal_raw_id}
    assert all(row[2] is not None for row in application_rows)
    assert len(indexed) == 1
    assert len(selected_heads) == 1
    selected_raw_id = str(selected_heads[0][0])
    assert selected_raw_id == terminal_raw_id
    assert indexed == [(f"codex-session:{SESSION_NATIVE_ID}", selected_raw_id, 8)]
    with sqlite3.connect(root / "source.db") as conn:
        selected_source_row = conn.execute(
            "SELECT blob_size, lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?",
            (selected_raw_id,),
        ).fetchone()
    assert selected_source_row == (TERMINAL_WIRE_BYTES, terminal_blob_hash)
    assert terminal_block_count > 0
    assert compaction_event_count == 1
    assert attachment_summary_count > 0
    assert attachment_image_block_count > 0
    expected_baseline_timestamps = tuple(
        int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
        for timestamp in _BASELINE_MESSAGE_TIMESTAMPS
    )
    with sqlite3.connect(candidate.index_path) as conn:
        persisted_baseline_timestamps = tuple(
            int(row[0])
            for row in conn.execute(
                "SELECT occurred_at_ms FROM messages WHERE message_id LIKE ? ORDER BY message_id",
                (f"%{SESSION_NATIVE_ID}-message-%",),
            )
        )
    assert persisted_baseline_timestamps == expected_baseline_timestamps

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
                input_id=f"{input_ref.input_id}:fixture-manifest-sha256:{fixture_manifest_digest}",
                corpus_id=f"fixture-manifest:sha256:{fixture_manifest_digest}",
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
            f"fixture-manifest-sha256:{fixture_manifest_digest}",
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
            "Fixture setup includes 804 on-disk payload revisions, schema hashing, and a canonical fixture manifest.",
            f"Serialized fixture manifest digest is sha256:{fixture_manifest_digest} and is bound into the receipt input identity.",
            "Replay and postflight subprocess RSS is unavailable because statm samples only the pytest parent; storage growth is not reported as write I/O.",
            "The crash boundary hard-exits after durable transaction creation with all 804 authority rows unresolved; a fresh process resumes and materializes the cohort into an inactive candidate.",
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
                "fixture_manifest_digest": fixture_manifest_digest,
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


def test_codex_804_timestamp_red_mutation_is_rejected() -> None:
    mutated = (_BASELINE_MESSAGE_TIMESTAMPS[0], "2026-07-31T04:25:21Z")
    with pytest.raises(AssertionError):
        _assert_baseline_timestamps_are_stable(mutated)


def test_codex_804_false_paused_state_red_mutation_is_rejected() -> None:
    with pytest.raises(AssertionError):
        _assert_precheckpoint_state(
            status="paused",
            processed_raw_count=8,
            last_raw_id="raw-008",
            last_blob_hash_hex="deadbeef",
            receipt_names=("pass-001.json",),
        )


def test_codex_804_incomplete_authority_red_mutation_is_rejected() -> None:
    with pytest.raises(AssertionError):
        _assert_exact_authority_census(
            raw_ids={"raw-000", "raw-001"},
            authority_rows=(("raw-000", "byte_proven", 1),),
            authority_counts=(("byte_proven", 1),),
            parser_census_rows=(
                ("raw-000", RAW_AUTHORITY_PARSER_FINGERPRINT, "complete", "[]", "parser-observed: inherited"),
            ),
            application_rows=(("raw-000", "selected_baseline", "raw-001"),),
            head_rows=(("codex-session:fixture", "codex-session:fixture", "raw-001", "byte", 1),),
            source_key="codex-session:fixture",
            session_id="codex-session:fixture",
            terminal_raw_id="raw-001",
        )
