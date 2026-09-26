"""Tests for ``devtools bench ingest-throughput``.

Throughput is host-variable, so these assertions only cover the report shape,
type wellformedness, and the *deterministic* count fields — never wall-clock
durations or messages/second values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.ingest_throughput_probe import (
    REPORT_VERSION,
    main,
    measure_ingest_throughput,
)

_PER_BATCH_MS_KEYS = {"min", "max", "mean", "p90"}


@pytest.mark.parametrize("provider", ["codex", "chatgpt"])
def test_measure_emits_expected_shape(provider: str, tmp_path: Path) -> None:
    report = measure_ingest_throughput(provider=provider, batches=3, seed=7, workdir=tmp_path)

    assert report["ok"] is True
    assert report["report_version"] == REPORT_VERSION
    assert report["tool"] == "bench ingest-throughput"
    assert report["provider"] == provider
    assert report["batches"] == 3
    assert report["seed"] == 7

    # Deterministic count fields are non-negative ints.
    for key in ("total_sessions", "total_messages", "batches", "seed", "messages_min", "messages_max"):
        assert isinstance(report[key], int)
        assert report[key] >= 0
    assert report["total_sessions"] == 3
    assert report["total_messages"] >= 3

    # Timing fields are non-negative floats; values themselves are not asserted.
    for key in ("total_wall_s", "messages_per_s", "sessions_per_s"):
        assert isinstance(report[key], float)
        assert report[key] >= 0.0

    per_batch_ms = report["per_batch_ms"]
    assert set(per_batch_ms) == _PER_BATCH_MS_KEYS
    for value in per_batch_ms.values():
        assert isinstance(value, float)
        assert value >= 0.0
    # min <= mean <= max (allow equality).
    assert per_batch_ms["min"] <= per_batch_ms["mean"] <= per_batch_ms["max"]

    per_batch = report["per_batch"]
    assert len(per_batch) == 3
    for index, batch in enumerate(per_batch):
        assert batch["batch_index"] == index
        assert batch["sessions_ingested"] == 1
        assert batch["messages_ingested"] >= 1
        assert isinstance(batch["batch_ms"], float)
        assert batch["batch_ms"] >= 0.0

    # Workload tag is the corpus path for these fixtures.
    assert report["workload"] == "corpus"

    # The legacy report and the shared receipt are one observation, not two
    # independently measured workloads.  This catches unit drift (MiB vs
    # bytes) and prevents a mutation from dropping the common adapter.
    receipt = report["workload_receipt"]
    assert receipt["spec"]["measurement_scope"] == "process"
    phase = receipt["phases"][0]
    assert phase["peak_rss_bytes"] == round(report["peak_rss_mb"] * 1024 * 1024)
    assert phase["cpu_ms"] == report["cpu_seconds_total"] * 1000.0
    assert phase["progress_completed"] == report["total_messages"]
    assert phase["progress_total"] == report["expected_messages"]
    assert report["output_membership"]
    assert all(item["matched"] for item in report["output_membership"])
    assert report["peak_rss_scope"] == "process-lifetime ru_maxrss"
    if report["proc_io_available"]:
        assert phase["read_io_bytes"] == report["proc_io"]["read_bytes"]
        assert phase["write_io_bytes"] == report["proc_io"]["write_bytes"]
    else:
        assert "read_io_bytes" in phase["unavailable"]
        assert "write_io_bytes" in phase["unavailable"]

    # Per-stage attribution is populated for a real ingest run.
    stage_timings = report["stage_timings_s"]
    assert isinstance(stage_timings, dict)
    assert stage_timings  # non-empty
    for stage_name, seconds in stage_timings.items():
        assert isinstance(stage_name, str)
        assert isinstance(seconds, float)
        assert seconds >= 0.0

    top_stages = report["top_stages"]
    assert isinstance(top_stages, list)
    assert top_stages  # non-empty when stage_timings is non-empty
    for entry in top_stages:
        assert set(entry) >= {"stage", "seconds", "pct_of_total_stage_time"}
        assert isinstance(entry["stage"], str)
        assert isinstance(entry["seconds"], float)
        assert entry["seconds"] >= 0.0
        assert 0.0 <= entry["pct_of_total_stage_time"] <= 100.0
    assert len(top_stages) <= 8

    # CPU / memory headline metrics — shape and non-negativity only.
    assert isinstance(report["cpu_seconds_total"], float)
    assert report["cpu_seconds_total"] >= 0.0
    assert isinstance(report["cpu_utilization"], float)
    assert report["cpu_utilization"] >= 0.0
    assert isinstance(report["peak_rss_mb"], float)
    assert report["peak_rss_mb"] > 0.0
    assert report["workload_receipt"]["status"] == "succeeded"

    # rusage-derived resource deltas are present and well-typed.
    resources = report["resources"]
    assert isinstance(resources, dict)
    for float_key in ("ru_utime_s", "ru_stime_s"):
        assert isinstance(resources[float_key], float)
        assert resources[float_key] >= 0.0
    for int_key in (
        "ru_minflt_delta",
        "ru_majflt_delta",
        "ru_inblock_delta",
        "ru_oublock_delta",
    ):
        assert isinstance(resources[int_key], int)

    # /proc/self/io is best-effort; the flag governs whether the dict is filled.
    assert isinstance(report["proc_io_available"], bool)
    proc_io = report["proc_io"]
    assert isinstance(proc_io, dict)
    if report["proc_io_available"]:
        for field in ("rchar", "wchar", "read_bytes", "write_bytes", "syscr", "syscw"):
            assert isinstance(proc_io[field], int)
        assert isinstance(proc_io["write_mb"], float)
        assert isinstance(proc_io["read_mb"], float)

    # SQLite storage growth is reported with non-negative sizes.
    storage = report["storage"]
    assert isinstance(storage, dict)
    for size_key in ("index_db_bytes", "index_wal_final_bytes", "source_db_bytes"):
        assert isinstance(storage[size_key], int)
        assert storage[size_key] >= 0
    for ratio_key in ("bytes_written_per_message", "db_growth_per_message"):
        assert isinstance(storage[ratio_key], float)


def test_counts_are_deterministic(tmp_path: Path) -> None:
    first = measure_ingest_throughput(provider="codex", batches=3, seed=99, workdir=tmp_path / "a")
    second = measure_ingest_throughput(provider="codex", batches=3, seed=99, workdir=tmp_path / "b")

    # Counts are deterministic for fixed (provider, batches, seed); timings are not.
    assert first["total_sessions"] == second["total_sessions"]
    assert first["total_messages"] == second["total_messages"]
    assert [b["messages_ingested"] for b in first["per_batch"]] == [b["messages_ingested"] for b in second["per_batch"]]


def test_main_json_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--json", "--batches", "2", "--seed", "3", "--workdir", str(tmp_path)])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["tool"] == "bench ingest-throughput"
    assert len(payload["per_batch"]) == 2


def test_rejects_unavailable_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not available"):
        measure_ingest_throughput(provider="nope-not-real", batches=1, workdir=tmp_path)


def test_refuses_populated_workdir_archive_before_mutation(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    sentinel = archive / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="archive destination is not empty"):
        measure_ingest_throughput(batches=1, workdir=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_lineage_rejects_mislabeled_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lineage currently uses the codex provider"):
        measure_ingest_throughput(lineage=True, provider="chatgpt", workdir=tmp_path)


def test_receipt_build_identity_uses_the_imported_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    import devtools.ingest_throughput_probe as probe

    repository_root = Path(probe.__file__).resolve().parents[1]
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    assert probe._current_build_id().startswith(f"git:{expected_head}:tracked-diff:")


def test_unreportable_private_failure_cleans_its_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import devtools.ingest_throughput_probe as probe

    scratch = tmp_path / "unreportable"
    monkeypatch.setattr(probe.tempfile, "mkdtemp", lambda **kwargs: str(scratch))

    def fail_before_report(*args: object, **kwargs: object) -> object:
        raise RuntimeError("controlled route failure")

    monkeypatch.setattr(probe, "_run_corpus_workload", fail_before_report)
    with pytest.raises(RuntimeError, match="controlled route failure"):
        measure_ingest_throughput(batches=1)

    assert not scratch.exists()


def test_parse_failure_is_a_failed_cli_receipt_with_expected_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from polylogue.pipeline.services import archive_ingest
    from polylogue.pipeline.services.parsing_models import ParseResult

    async def failed_parse(*args: object, **kwargs: object) -> ParseResult:
        result = ParseResult()
        result.parse_failures = 1
        result.counts["skipped_sessions"] = 1
        return result

    monkeypatch.setattr(archive_ingest, "parse_sources_archive", failed_parse)
    exit_code = main(
        ["--json", "--batches", "1", "--messages-min", "2", "--messages-max", "2", "--workdir", str(tmp_path)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["total_messages"] == 0
    assert payload["messages_per_s"] is None
    assert payload["expected_messages"] == 2
    assert payload["parse_outcomes"]["parse_failures"] == 1
    assert payload["workload_receipt"]["status"] == "failed"
    assert payload["workload_receipt"]["phases"][0]["progress_total"] == 2
    assert payload["workdir_disposition"] == "retained-by-caller"


def test_partial_population_keeps_expected_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from polylogue.pipeline.services import archive_ingest
    from polylogue.pipeline.services.parsing_models import ParseResult

    async def partial_parse(*args: object, **kwargs: object) -> ParseResult:
        result = ParseResult()
        result.parse_failures = 1
        result.counts["sessions"] = 1
        result.counts["messages"] = 1
        return result

    monkeypatch.setattr(archive_ingest, "parse_sources_archive", partial_parse)
    exit_code = main(
        ["--json", "--batches", "2", "--messages-min", "2", "--messages-max", "2", "--workdir", str(tmp_path)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["total_messages"] == 2
    assert payload["expected_messages"] == 4
    assert payload["workload_receipt"]["phases"][0]["progress_total"] == 4
    assert payload["workload_receipt"]["status"] == "failed"


def test_retained_raw_hash_detects_same_count_content_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import devtools.ingest_throughput_probe as probe
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    original = ArchiveStore.raw_revision_material

    def substituted(self: ArchiveStore, raw_id: str) -> tuple[object, bytes, str, object]:
        provider, raw_bytes, source_path, kind = original(self, raw_id)
        return provider, raw_bytes + b" altered", source_path, kind

    monkeypatch.setattr(ArchiveStore, "raw_revision_material", substituted)
    report = probe.measure_ingest_throughput(batches=1, messages_min=2, messages_max=2, workdir=tmp_path)

    assert report["total_sessions"] == report["expected_sessions"] == 1
    assert report["total_messages"] == report["expected_messages"] == 2
    assert report["output_membership"][0]["source_matches"] is False
    assert report["ok"] is False
    assert report["messages_per_s"] is None


def test_success_claim_explicitly_excludes_normalized_text_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import devtools.ingest_throughput_probe as probe
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope

    original = ArchiveStore.read_session

    def alter_text(self: ArchiveStore, session_id: str) -> ArchiveSessionEnvelope:
        envelope = original(self, session_id)
        message = envelope.messages[0]
        assert message.blocks
        altered_block = replace(message.blocks[0], text=(message.blocks[0].text or "") + " altered")
        altered_message = replace(message, blocks=(altered_block, *message.blocks[1:]))
        return replace(envelope, messages=(altered_message, *envelope.messages[1:]))

    monkeypatch.setattr(ArchiveStore, "read_session", alter_text)
    report = probe.measure_ingest_throughput(batches=1, messages_min=2, messages_max=2, workdir=tmp_path)

    assert report["output_membership"][0]["source_matches"] is True
    assert report["output_membership"][0]["matched"] is True
    assert report["semantic_content_verified"] is False
    assert "canonical-parser preflight" in report["verification_scope"]
    assert report["ok"] is True


def test_ordered_message_identity_mismatch_fails_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import devtools.ingest_throughput_probe as probe
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    original = ArchiveStore.read_session

    def alter_identity(self: ArchiveStore, session_id: str) -> object:
        envelope = original(self, session_id)
        message = envelope.messages[0]
        return replace(
            envelope,
            messages=(replace(message, message_id=message.message_id + ":altered"), *envelope.messages[1:]),
        )

    monkeypatch.setattr(ArchiveStore, "read_session", alter_identity)
    report = probe.measure_ingest_throughput(batches=1, messages_min=2, messages_max=2, workdir=tmp_path)

    assert report["output_membership"][0]["source_matches"] is True
    assert report["output_membership"][0]["matched"] is False
    assert report["semantic_content_verified"] is False
    assert report["ok"] is False


def test_excision_and_budget_evidence_survive_in_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from polylogue.pipeline.services import archive_ingest
    from polylogue.pipeline.services.parsing_models import ParseResult

    async def incomplete_parse(*args: object, **kwargs: object) -> ParseResult:
        result = ParseResult()
        result.excised_skips = 1
        result.time_budget_exceeded = True
        return result

    monkeypatch.setattr(archive_ingest, "parse_sources_archive", incomplete_parse)
    exit_code = main(["--json", "--batches", "1", "--workdir", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["parse_outcomes"]["excised_skips"] == 1
    assert payload["parse_outcomes"]["budget_exhausted"] == 1
    assert payload["workload_receipt"]["status"] == "failed"
    assert "excised_skips" in " ".join(payload["workload_receipt"]["notes"])


def test_lineage_workload_composes(tmp_path: Path) -> None:
    report = measure_ingest_throughput(
        lineage=True,
        batches=4,
        seed=7,
        messages_max=12,
        workdir=tmp_path,
    )

    assert report["ok"] is True
    assert report["workload"] == "lineage"
    # One parent + four forks were written.
    assert report["total_sessions"] == 5
    assert len(report["per_batch"]) == 4
    # total_messages reflects parent prefix + every fork's replayed prefix + tail.
    assert report["total_messages"] > report["messages_max"]
    # Stage attribution still populated through the direct ArchiveStore path.
    assert isinstance(report["stage_timings_s"], dict)
    assert report["stage_timings_s"]
    assert report["cpu_utilization"] >= 0.0
    assert report["peak_rss_mb"] > 0.0
