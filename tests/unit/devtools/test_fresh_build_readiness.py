"""Readiness uses the actual arm's code, never the driver's recipe oracle."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from devtools.fresh_build_bench.readiness_probe import CandidateReadinessProbe, ReadinessProbeError
from devtools.fresh_build_bench.report import readiness_evidence
from devtools.fresh_build_bench.run import REQUIRED_READINESS_DOMAINS, observe


def _candidate(root: Path, ready: bool) -> Path:
    package = root / "polylogue" / "storage"
    package.mkdir(parents=True)
    for parent in (package, package.parent):
        (parent / "__init__.py").write_text("")
    (package / "archive_readiness.py").write_text(
        "import os\n"
        "def archive_readiness_status(archive):\n"
        " assert 'POLYLOGUE_BENCH_STACKS' not in os.environ\n"
        " assert 'POLYLOGUE_TRACE_COMPUTE' not in os.environ\n"
        " assert os.environ['POLYLOGUE_CONFIG'] == 'isolated-config'\n"
        f" return {{'checked': True, 'surfaces': {{name: {{'ready': {ready!r}}} for name in {sorted(REQUIRED_READINESS_DOMAINS)!r}}}}}\n"
    )
    return root


def _probe(candidate: Path, work: Path) -> CandidateReadinessProbe:
    work.mkdir(exist_ok=True)
    return CandidateReadinessProbe(
        candidate=candidate,
        python=sys.executable,
        work=work,
        environment={
            **os.environ,
            "POLYLOGUE_CONFIG": "isolated-config",
            "POLYLOGUE_BENCH_STACKS": "1",
            "POLYLOGUE_TRACE_COMPUTE": "1",
        },
    )


def test_readiness_imports_actual_candidate_and_qualifies_its_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.archive_readiness as driver

    def wrong_oracle(*args: object) -> None:
        raise AssertionError("driver readiness must never decide another candidate's recipes")

    monkeypatch.setattr(driver, "archive_readiness_status", wrong_oracle)
    candidate = _candidate(tmp_path / "candidate", True)
    probe = _probe(candidate, tmp_path / "work")
    result = probe(tmp_path / "archive")
    assert result["outcome"] == "bound"
    assert all(row["ready"] for row in result["readiness"]["surfaces"].values())
    assert result["helper_wall_s"] >= 0
    assert result["helper_cpu_s"]["user"] >= 0
    assert (
        readiness_evidence(
            probe.work, candidate=str(candidate), python=os.path.realpath(sys.executable), helper_hash=probe.helper_hash
        )["outcome"]
        == "bound"
    )
    assert (
        readiness_evidence(
            probe.work,
            candidate=str(tmp_path / "wrong"),
            python=os.path.realpath(sys.executable),
            helper_hash=probe.helper_hash,
        )["outcome"]
        != "bound"
    )


def _archive(archive: Path) -> None:
    archive.mkdir()
    with sqlite3.connect(archive / "ops.db") as conn:
        conn.execute(
            "CREATE TABLE ingest_cursor(excluded INTEGER,byte_offset INTEGER,stat_size INTEGER,failure_count INTEGER,deferred_end_offset INTEGER,next_retry_at TEXT)"
        )
        conn.execute("INSERT INTO ingest_cursor VALUES(0,0,5,0,5,NULL)")
    with sqlite3.connect(archive / "source.db") as conn:
        conn.execute("CREATE TABLE empty(value INTEGER)")
    generation = archive / ".index-generations" / "gen" / "index.db"
    generation.parent.mkdir(parents=True)
    generation.touch()
    (archive / "index.db").symlink_to(generation)


@pytest.mark.uses_real_clock("readiness helper and observer share the host monotonic clock across processes")
def test_readiness_cache_binds_candidate_and_source_and_exposes_ready_before_cursor_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.fresh_build_bench import run

    monkeypatch.setattr(run, "_last_readiness", {})
    archive = tmp_path / "archive"
    _archive(archive)
    a = _probe(_candidate(tmp_path / "A", True), tmp_path / "work-A")
    b = _probe(_candidate(tmp_path / "B", False), tmp_path / "work-B")
    started = time.monotonic()
    first = observe(archive, started, readiness_probe=a)
    assert first.readiness_complete and not first.intake_complete and not first.terminal
    cached = observe(archive, started, readiness_probe=a)
    assert cached.readiness_cached
    changed = observe(archive, started, readiness_probe=b)
    assert not changed.readiness_cached and not changed.readiness_complete
    source = b.candidate / "polylogue/storage/archive_readiness.py"
    source.write_text(source.read_text().replace("'ready': False", "'ready': True"))
    changed_source = observe(archive, started, readiness_probe=b)
    assert not changed_source.readiness_cached and changed_source.readiness_complete


@pytest.mark.uses_real_clock("observe consumes a real subprocess readiness refusal")
def test_broken_readiness_candidate_refuses_instead_of_reporting_ready(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "broken", True)
    (candidate / "polylogue/storage/archive_readiness.py").write_text(
        "raise ImportError('synthetic broken candidate')\n"
    )
    probe = _probe(candidate, tmp_path / "work")
    with pytest.raises(ReadinessProbeError):
        probe(tmp_path / "archive")
    evidence = json.loads((probe.work / "readiness.json").read_text())
    assert evidence["censuses"][0]["outcome"] == "refused"
    assert (
        readiness_evidence(
            probe.work, candidate=str(candidate), python=os.path.realpath(sys.executable), helper_hash=probe.helper_hash
        )["outcome"]
        != "bound"
    )
    archive = tmp_path / "archive"
    _archive(archive)
    observation = observe(archive, time.monotonic(), readiness_probe=probe)
    assert observation.error is not None
    assert not observation.error_retryable and not observation.terminal


@pytest.mark.parametrize("reason,retryable", [("database is locked", True), ("no such table: sessions", False)])
@pytest.mark.uses_real_clock("observe consumes an actual subprocess census read failure")
def test_unchecked_readiness_preserves_transient_reads_but_refuses_schema_errors(
    tmp_path: Path, reason: str, retryable: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.fresh_build_bench import run

    monkeypatch.setattr(run, "_last_readiness", {})
    candidate = _candidate(tmp_path / "candidate", True)
    source = candidate / "polylogue/storage/archive_readiness.py"
    source.write_text(
        "def archive_readiness_status(archive):\n"
        f" return {{'checked': False, 'reason': {reason!r}, 'surfaces': {{}}}}\n"
    )
    probe = _probe(candidate, tmp_path / "work")
    archive = tmp_path / "archive"
    _archive(archive)
    observation = observe(archive, time.monotonic(), readiness_probe=probe)
    assert observation.error is not None and observation.error_retryable is retryable
    assert not observation.readiness and not observation.terminal
    assert not run._last_readiness
    evidence = readiness_evidence(
        probe.work, candidate=str(candidate), python=os.path.realpath(sys.executable), helper_hash=probe.helper_hash
    )
    assert evidence["outcome"] != "bound"
    if retryable:
        source.write_text(
            "def archive_readiness_status(archive):\n"
            f" return {{'checked': True, 'surfaces': {{name: {{'ready': True}} for name in {sorted(REQUIRED_READINESS_DOMAINS)!r}}}}}\n"
        )
        recovered = observe(archive, time.monotonic(), readiness_probe=probe)
        assert recovered.error is None and recovered.readiness_complete
        assert (
            readiness_evidence(
                probe.work,
                candidate=str(candidate),
                python=os.path.realpath(sys.executable),
                helper_hash=probe.helper_hash,
            )["outcome"]
            == "bound"
        )
