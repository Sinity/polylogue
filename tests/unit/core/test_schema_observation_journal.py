"""Behavioral contracts for the schema ObservationJournal lifecycle."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.scenarios import (
    MeasurementScope,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)
from polylogue.schemas.generation import observation_journal as observation_journal_module
from polylogue.schemas.generation.observation_journal import (
    ObservationJournal,
    recover_stale_journals,
)
from polylogue.schemas.generation.replay import MembershipSessionIds
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.workload_tiers import WorkloadScaleTier


@dataclass(frozen=True, slots=True)
class _InferenceProbeResult:
    sample_count: int
    peak_rss_bytes: int
    peak_journal_bytes: int
    phase_snapshots: tuple[dict[str, object], ...]
    journal_method_metrics: dict[str, dict[str, int]]
    receipt: WorkloadReceipt


def _process_rss_bytes(pid: int) -> int:
    try:
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _journal_bytes(root: Path) -> int:
    journal_root = root / "cache" / "polylogue" / "schema-observation-journals"
    return sum(path.stat().st_size for path in journal_root.glob("run-*.sqlite3*") if path.is_file())


def _run_inference_probe(
    tmp_path: Path,
    *,
    count: int,
    scale_tier: WorkloadScaleTier,
    provider: str = "chatgpt",
    record_count: int | None = None,
) -> _InferenceProbeResult:
    run_root = tmp_path / scale_tier.value
    archive_root = run_root / "archive"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.infra.schema_inference_memory_probe",
            "--archive-root",
            str(archive_root),
            "--count",
            str(count),
            "--provider",
            provider,
            *([] if record_count is None else ["--record-count", str(record_count)]),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdin is not None
        assert process.stderr is not None
        assert process.stdout.readline().strip() == "READY"
        process.stdin.write("run\n")
        process.stdin.flush()

        started = time.perf_counter()
        peak_rss_bytes = 0
        peak_journal_bytes = 0
        while process.poll() is None:
            peak_rss_bytes = max(peak_rss_bytes, _process_rss_bytes(process.pid))
            peak_journal_bytes = max(peak_journal_bytes, _journal_bytes(run_root))
            time.sleep(0.01)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        payload = json.loads(stdout)
        assert payload["success"] is True
        assert payload["phase_receipt"]["status"] == "succeeded"
        assert payload["journal_remaining"] == []
        phase_snapshots = tuple(item for item in payload["phases"] if isinstance(item, dict))
        assert phase_snapshots
        journal_method_metrics = payload["journal_method_metrics"]
        assert isinstance(journal_method_metrics, dict)
        wall_ms = (time.perf_counter() - started) * 1_000
        spec = WorkloadEnvelopeSpec(
            workload_id="canary:schema-inference-memory-scaling",
            family_id="schema-profile-production-canary",
            version=1,
            inputs=(
                WorkloadInputRef(
                    input_id=f"synthetic-{provider}:{scale_tier.value}:{record_count or count}",
                    corpus_id=f"archive:test:schema-inference:{provider}:{scale_tier.value}",
                    profile_id="workload-profile:schema-inference-memory",
                    scale_tier=scale_tier.value,
                    seed=0,
                    distribution_refs=("provider-package.artifact_count", "provider-package.sample_count"),
                ),
            ),
            phases=("infer", "quiescent"),
            measurement_scope=MeasurementScope.PROCESS_TREE,
        )
        receipt = WorkloadReceipt.from_observations(
            spec=spec,
            status=WorkloadRunStatus.SUCCEEDED,
            build_id="git:test",
            runtime_id=f"python:{sys.version_info.major}.{sys.version_info.minor}",
            archive_id=f"archive:test:schema-inference:{provider}:{scale_tier.value}",
            generation_id=f"synthetic:{provider}:{record_count or count}",
            frame_id=None,
            phases=(
                WorkloadPhaseObservation(
                    name="infer",
                    measurement_scope=MeasurementScope.PROCESS_TREE,
                    wall_ms=wall_ms,
                    peak_rss_bytes=peak_rss_bytes,
                    temp_storage_bytes=peak_journal_bytes,
                    progress_completed=int(payload["sample_count"]),
                    progress_total=record_count or count,
                ),
                WorkloadPhaseObservation(
                    name="quiescent",
                    measurement_scope=MeasurementScope.PROCESS_TREE,
                    cleanup_complete=True,
                    quiescent=True,
                ),
            ),
            cleanup_complete=True,
            notes=("The probe process owns no child workers; process RSS is the complete process tree.",),
        )
        return _InferenceProbeResult(
            sample_count=int(payload["sample_count"]),
            peak_rss_bytes=peak_rss_bytes,
            peak_journal_bytes=peak_journal_bytes,
            phase_snapshots=phase_snapshots,
            journal_method_metrics={
                str(name): {str(key): int(value) for key, value in metric.items()}
                for name, metric in journal_method_metrics.items()
                if isinstance(metric, dict)
            },
            receipt=receipt,
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _unit() -> SchemaUnit:
    return SchemaUnit(
        cluster_payload={"type": "message", "content": {"kind": "tool"}},
        schema_samples=[{"type": "message"}, {"type": "tool_result", "exit_code": 0}],
        artifact_kind="session_record_stream",
        session_id="session-1",
        raw_id="raw-1",
        source_path="/private/source.jsonl",
        bundle_scope="scope-1",
        observed_at="2026-07-16T12:00:00+00:00",
        exact_structure_id="structure-1",
        profile_tokens=("field:type", "record:message"),
    )


def test_journal_roundtrips_units_samples_and_terminal_evidence(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    with ObservationJournal.create(root=root) as journal:
        journal.append_unit(_unit())
        journal.record_terminal(
            raw_id="raw-1",
            status="included",
            artifact_kind="session_record_stream",
            source_path="/private/source.jsonl",
            reason=None,
        )
        journal.record_terminal(
            raw_id="raw-quarantined",
            status="quarantined",
            artifact_kind="sqlite_evidence",
            source_path="/private/evidence.db",
            reason="artifact taxonomy excludes provider-wire decoding",
        )

        assert list(journal.iter_units()) == [_unit()]
        terminals = list(journal.iter_terminals())
        assert [(item.raw_id, item.status) for item in terminals] == [
            ("raw-1", "included"),
            ("raw-quarantined", "quarantined"),
        ]
        assert journal.unit_count == 1
        assert journal.sample_count == 2
        assert journal.terminal_counts() == {"included": 1, "quarantined": 1}
        assert journal.terminal_summary() == {
            "total": 2,
            "status_counts": {"included": 1, "quarantined": 1},
            "reason_counts": {"artifact taxonomy excludes provider-wire decoding": 1},
        }

        journal_path = journal.path
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600

    assert not journal_path.exists()
    assert not journal_path.with_name(f"{journal_path.name}-wal").exists()
    assert not journal_path.with_name(f"{journal_path.name}-shm").exists()


def test_journal_replays_lone_surrogate_provider_values_without_utf8_failure(tmp_path: Path) -> None:
    unit = SchemaUnit(
        cluster_payload={"text": "broken \udce2 provider value"},
        schema_samples=[{"text": "broken \udce2 provider value"}],
        artifact_kind="session_record_stream",
        session_id="session-surrogate",
        raw_id="raw-surrogate",
        source_path="/private/source.jsonl",
        bundle_scope="scope-surrogate",
        observed_at="2026-07-16T12:00:00+00:00",
        exact_structure_id="structure-surrogate",
        profile_tokens=("field:text",),
    )

    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        journal.append_unit(unit)
        assert list(journal.iter_units()) == [unit]


def test_journal_flushes_private_ingest_before_replay(tmp_path: Path) -> None:
    """A long generation may observe indefinitely without making one giant WAL."""
    root = tmp_path / "journals"
    with ObservationJournal.create(root=root) as journal:
        journal.append_unit(_unit())
        journal.flush()

        with sqlite3.connect(journal.path) as reader:
            assert reader.execute("SELECT COUNT(*) FROM units").fetchone() == (1,)
            assert reader.execute("SELECT COUNT(*) FROM samples").fetchone() == (2,)


def test_large_single_unit_commits_sample_batches_before_unit_finishes(tmp_path: Path) -> None:
    """One transcript cannot turn the complete private WAL budget into one transaction."""
    unit = SchemaUnit(
        cluster_payload={},
        schema_samples=[{"payload": "x" * (1024 * 1024)} for _ in range(33)],
        artifact_kind="session_record_stream",
        exact_structure_id="large-stream",
        profile_tokens=("field:payload",),
    )
    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        journal.append_unit(unit)

        with sqlite3.connect(journal.path) as reader:
            committed_samples = reader.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        assert 0 < committed_samples < len(unit.schema_samples)


def test_selective_membership_replay_uses_units_before_samples(tmp_path: Path) -> None:
    """Package replay must not scan every sample before applying its package filter."""
    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        for index in range(32):
            unit_id = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=[{"index": index}],
                    artifact_kind="session_record_stream",
                    exact_structure_id=f"shape-{index}",
                    profile_tokens=("field:index",),
                ),
                retain_cluster_payload=False,
            )
            journal.assign_profile_family(unit_id, "selected" if index == 0 else "other")
            journal.assign_package_family(unit_id, "selected" if index == 0 else "other")
        journal.flush()

        where, parameters = journal._membership_where(
            profile_family_id="selected",
            package_family_id="selected",
            artifact_kind="session_record_stream",
        )
        plan = [
            str(row[-1])
            for row in journal._connection.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT units.*, samples.position, samples.sample_json "
                "FROM units JOIN samples ON samples.unit_id = units.unit_id "
                f"WHERE {where} ORDER BY samples.unit_id, samples.position",
                parameters,
            )
        ]

        assert any("SEARCH units USING INDEX units_package_family_idx" in detail for detail in plan)
        assert any("SEARCH samples USING PRIMARY KEY (unit_id=?)" in detail for detail in plan)
        assert list(journal.memberships(package_family_id="selected"))[0].unit.schema_samples == [{"index": 0}]


def test_journal_cleanup_runs_for_exceptions(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    journal_path: Path | None = None

    with pytest.raises(RuntimeError, match="cancelled"):
        with ObservationJournal.create(root=root) as journal:
            journal_path = journal.path
            journal.append_unit(_unit())
            raise RuntimeError("cancelled")

    assert journal_path is not None
    assert not journal_path.exists()


def test_journal_cleanup_runs_for_sigterm(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    journal = ObservationJournal.create(root=root)

    with pytest.raises(SystemExit) as raised:
        with journal:
            journal.append_unit(_unit())
            signal.raise_signal(signal.SIGTERM)

    assert raised.value.code == 128 + signal.SIGTERM
    assert not list(root.glob("run-*.sqlite3*"))


def test_journal_cleanup_runs_for_real_terminated_process(tmp_path: Path) -> None:
    """A launcher SIGTERM must unwind the journal owner, not only a unit test frame."""
    root = tmp_path / "journals"
    script = """
import signal
import sys
from pathlib import Path
from polylogue.schemas.generation.observation_journal import ObservationJournal
from polylogue.schemas.observation import SchemaUnit

with ObservationJournal.create(root=Path(sys.argv[1])) as journal:
    journal.append_unit(SchemaUnit(
        cluster_payload={"x": 1}, schema_samples=[{"x": 1}],
        artifact_kind="session_document", exact_structure_id="x",
        profile_tokens=("field:x",),
    ))
    print(journal.path, flush=True)
    signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    journal_path = Path(process.stdout.readline().strip())
    process.terminate()
    assert process.wait(timeout=10) == 128 + signal.SIGTERM
    assert not journal_path.exists()
    assert not list(root.glob("run-*.sqlite3*"))


def test_journal_replays_assignments_without_retaining_cluster_payloads(tmp_path: Path) -> None:
    first = _unit()
    second = SchemaUnit(
        cluster_payload={"large": ["private clustering content"] * 10},
        schema_samples=[{"type": "assistant"}],
        artifact_kind="session_record_stream",
        session_id="session-2",
        exact_structure_id="structure-2",
        profile_tokens=("field:type", "record:assistant"),
    )
    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        first_id = journal.append_unit(first, retain_cluster_payload=False)
        second_id = journal.append_unit(second, retain_cluster_payload=False)
        journal.assign_profile_family(first_id, "family-a")
        journal.assign_profile_family(second_id, "family-b")

        # Simultaneous normalization must not cascade family-a through family-b.
        journal.normalize_profile_families({"family-a": "family-b", "family-b": "family-c"})
        journal.assign_package_family(first_id, "package-1")

        identified = list(journal.iter_identified_memberships())
        assert [membership.profile_family_id for _unit_id, membership in identified] == [
            "family-b",
            "family-c",
        ]
        assert all(membership.unit.cluster_payload == {} for _unit_id, membership in identified)
        package = journal.memberships(package_family_id="package-1")
        assert len(package) == 1
        assert package[0].unit.schema_samples == first.schema_samples
        assert list(package.iter_distinct_values("bundle_scope")) == ["scope-1"]
        assert list(package.iter_distinct_values("exact_structure_id")) == ["structure-1"]
        assert list(package.iter_distinct_values("profile_family_id")) == ["family-b"]
        assert package.distinct_count("bundle_scope") == 1


def test_metadata_membership_passes_do_not_decode_sample_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_decode = observation_journal_module._decode_json

    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        unit_id = journal.append_unit(_unit(), retain_cluster_payload=False)
        journal.assign_profile_family(unit_id, "family-a")
        journal.assign_package_family(unit_id, "package-a")
        memberships = journal.memberships(package_family_id="package-a")

        def reject_sample_decode(value: bytes) -> object:
            if b'"tool_result"' in value:
                raise AssertionError("metadata pass decoded a schema sample")
            return original_decode(value)

        monkeypatch.setattr(observation_journal_module, "_decode_json", reject_sample_decode)

        metadata = list(memberships.metadata())
        assert len(metadata) == 1
        assert metadata[0].unit.schema_samples == []
        identified_metadata = list(journal.iter_identified_unit_metadata())
        assert [
            (unit_id, len(unit.schema_samples), sample_count) for unit_id, unit, sample_count in identified_metadata
        ] == [(unit_id, 0, 2)]
        assert memberships.sample_count == 2
        assert list(MembershipSessionIds(memberships)) == ["session-1", "session-1"]
        with pytest.raises(AssertionError, match="decoded a schema sample"):
            list(memberships)


def test_journal_assigns_one_canonical_package_member_per_scope_structure(tmp_path: Path) -> None:
    """One pathological scope must not become a Python-sized assembly batch."""
    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        for index in range(512):
            unit_id = journal.append_unit(
                SchemaUnit(
                    cluster_payload={"index": index},
                    schema_samples=[{"index": index}],
                    artifact_kind="session_document",
                    bundle_scope="one-large-scope",
                    exact_structure_id=f"shape-{index % 4}",
                    profile_tokens=("field:index",),
                ),
                retain_cluster_payload=False,
            )
            journal.assign_profile_family(unit_id, "anchor-family")
        adjunct_id = journal.append_unit(
            SchemaUnit(
                cluster_payload={"meta": True},
                schema_samples=[{"meta": True}],
                artifact_kind="metadata_document",
                bundle_scope="one-large-scope",
                exact_structure_id="metadata-shape",
                profile_tokens=("field:meta",),
            ),
            retain_cluster_payload=False,
        )
        journal.assign_profile_family(adjunct_id, "adjunct-family")

        assert journal.assign_canonical_package_families(frozenset({"session_document"})) == {}
        package = journal.memberships(package_family_id="anchor-family")
        assert len(package) == 5
        assert package.scope_count() == 1
        assert package.sample_count == 5


def test_journal_rejects_archive_or_broad_permission_roots(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        ObservationJournal.create(root=archive_root / "journal", forbidden_roots=(archive_root,))

    broad_root = tmp_path / "broad"
    broad_root.mkdir(mode=0o777)
    broad_root.chmod(0o777)
    with pytest.raises(PermissionError, match="permissions"):
        ObservationJournal.create(root=broad_root, tighten_permissions=False)


def test_stale_recovery_removes_dead_owner_and_preserves_live_owner(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    code = """
import os
from pathlib import Path
from polylogue.schemas.generation.observation_journal import ObservationJournal
journal = ObservationJournal.create(root=Path(os.environ["JOURNAL_ROOT"]))
os.write(1, str(journal.path).encode("utf-8"))
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=os.environ | {"JOURNAL_ROOT": str(root)},
    )
    stale_path = Path(result.stdout.decode("utf-8"))
    os.utime(stale_path, (100.0, 100.0))

    live = ObservationJournal.create(root=root)
    live_path = live.path
    os.utime(live_path, (100.0, 100.0))

    removed = recover_stale_journals(root, now_s=10_000.0, minimum_age_s=60.0)

    assert removed == [stale_path]
    assert not stale_path.exists()
    assert live_path.exists()
    live.close()


def test_full_generation_10x_scales_counts_without_10x_python_memory(tmp_path: Path) -> None:
    one_x = _run_inference_probe(tmp_path, count=32, scale_tier=WorkloadScaleTier.ARCHIVE_1X)
    ten_x = _run_inference_probe(tmp_path, count=320, scale_tier=WorkloadScaleTier.ARCHIVE_10X)

    assert ten_x.sample_count == one_x.sample_count * 10
    assert one_x.peak_journal_bytes > 0
    assert ten_x.peak_journal_bytes > 0
    assert {snapshot["phase"] for snapshot in one_x.phase_snapshots} >= {
        "before-provider-bundle",
        "before-_collect_cluster_accumulators",
        "after-_collect_cluster_accumulators",
        "before-_build_package_candidates",
        "after-_build_package_candidates",
        "before-build_provider_catalog_artifacts",
        "after-build_provider_catalog_artifacts",
        "after-provider-bundle",
    }
    assert all(
        {"monotonic_ns", "rchar", "wchar", "read_bytes", "write_bytes", "journal_db_bytes", "journal_wal_bytes"}
        <= snapshot.keys()
        for snapshot in one_x.phase_snapshots
    )
    assert one_x.journal_method_metrics["append_unit"]["calls"] == 32
    assert ten_x.journal_method_metrics["append_unit"]["calls"] == 320
    assert one_x.journal_method_metrics["flush"]["calls"] >= 1
    assert one_x.journal_method_metrics["assign_canonical_package_families"]["calls"] == 1
    assert ten_x.peak_rss_bytes <= one_x.peak_rss_bytes + 96 * 1024 * 1024
    assert one_x.receipt.spec.inputs[0].scale_tier == WorkloadScaleTier.ARCHIVE_1X.value
    assert ten_x.receipt.spec.inputs[0].scale_tier == WorkloadScaleTier.ARCHIVE_10X.value
    assert one_x.receipt.cleanup_complete is True
    assert ten_x.receipt.cleanup_complete is True


def test_single_jsonl_10x_replays_all_records_without_10x_python_memory(tmp_path: Path) -> None:
    one_x = _run_inference_probe(
        tmp_path,
        count=1,
        record_count=1_024,
        provider="codex",
        scale_tier=WorkloadScaleTier.ARCHIVE_1X,
    )
    ten_x = _run_inference_probe(
        tmp_path,
        count=1,
        record_count=10_240,
        provider="codex",
        scale_tier=WorkloadScaleTier.ARCHIVE_10X,
    )

    assert ten_x.sample_count == one_x.sample_count * 10
    assert ten_x.peak_rss_bytes <= one_x.peak_rss_bytes + 96 * 1024 * 1024
    assert one_x.peak_journal_bytes > 0
    assert ten_x.peak_journal_bytes > 0
    assert one_x.receipt.cleanup_complete is True
    assert ten_x.receipt.cleanup_complete is True
