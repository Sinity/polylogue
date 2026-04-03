"""Tests for devtools.pipeline_probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.pipeline_probe import _write_probe_sources, main, run_probe
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.backends import create_backend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord


class _Args:
    input_mode = "synthetic"
    provider = "chatgpt"
    count = 1
    messages_min = 3
    messages_max = 4
    seed = 7
    sample_per_provider = 50
    source_filters = None
    source_db = None
    source_blob_root = None
    manifest_out = None
    manifest_in = None
    stage = "parse"
    raw_batch_size = None
    ingest_workers = None
    measure_ingest_result_size = False
    json_out = None
    max_total_ms = None
    max_peak_rss_mb = None

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir


class _ArchiveArgs:
    input_mode = "archive-subset"
    provider = None
    count = 1
    messages_min = 3
    messages_max = 4
    seed = 11
    sample_per_provider = 1
    source_filters = None
    manifest_out = None
    manifest_in = None
    stage = "parse"
    raw_batch_size = None
    ingest_workers = None
    measure_ingest_result_size = False
    json_out = None
    max_total_ms = None
    max_peak_rss_mb = None

    def __init__(
        self,
        *,
        workdir: Path,
        source_db: Path,
        source_blob_root: Path,
        manifest_out: Path | None = None,
        manifest_in: Path | None = None,
    ) -> None:
        self.workdir = workdir
        self.source_db = source_db
        self.source_blob_root = source_blob_root
        self.manifest_out = manifest_out
        self.manifest_in = manifest_in


class _SourceSubsetArgs:
    input_mode = "source-subset"
    provider = None
    count = 1
    messages_min = 3
    messages_max = 4
    seed = 13
    sample_per_provider = 50
    source_filters = None
    source_db = None
    source_blob_root = None
    manifest_out = None
    manifest_in = None
    stage = "parse"
    raw_batch_size = None
    ingest_workers = None
    measure_ingest_result_size = False
    json_out = None
    max_total_ms = None
    max_peak_rss_mb = None

    def __init__(
        self,
        *,
        workdir: Path,
        source_paths: list[Path],
        source_name: str = "inbox",
    ) -> None:
        self.workdir = workdir
        self.source_paths = source_paths
        self.source_name = source_name


async def _seed_archive_source(tmp_path: Path) -> tuple[Path, Path]:
    source_db = tmp_path / "source.db"
    source_blob_root = tmp_path / "source-blobs"
    blob_store = BlobStore(source_blob_root)
    backend = create_backend(db_path=source_db)
    repository = ConversationRepository(backend=backend)
    corpus = SyntheticCorpus.for_provider("chatgpt")
    codex_corpus = SyntheticCorpus.for_provider("codex")

    try:
        raw_payloads = [
            ("chatgpt", "chatgpt-main", corpus.generate(count=1, seed=100, messages_per_conversation=range(3, 4))[0]),
            ("chatgpt", "chatgpt-sidecar", corpus.generate(count=1, seed=101, messages_per_conversation=range(3, 4))[0]),
            ("codex", "codex-main", codex_corpus.generate(count=1, seed=200, messages_per_conversation=range(3, 4))[0]),
            ("codex", "codex-sidecar", codex_corpus.generate(count=1, seed=201, messages_per_conversation=range(3, 4))[0]),
        ]
        for index, (provider_name, source_name, raw_bytes) in enumerate(raw_payloads):
            raw_id, blob_size = blob_store.write_from_bytes(raw_bytes)
            await repository.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name=provider_name,
                    payload_provider=provider_name,
                    source_name=source_name,
                    source_path=f"/tmp/{source_name}-{index}",
                    source_index=index,
                    blob_size=blob_size,
                    acquired_at=f"2026-04-0{index + 1}T12:00:00Z",
                    file_mtime=f"2026-04-0{index + 1}T11:00:00Z",
                    parsed_at="2026-04-01T00:00:00Z",
                    validated_at="2026-04-01T00:00:00Z",
                    validation_status="passed",
                )
            )
    finally:
        await repository.close()

    return source_db, source_blob_root


async def test_run_probe_emits_real_pipeline_summary(tmp_path) -> None:
    summary = await run_probe(_Args(tmp_path / "probe"))
    ingest_details = summary["run_payload"]["metrics"]["stages"]["ingest"]["details"]["batch_observations"]

    assert summary["probe"]["provider"] == "chatgpt"
    assert summary["result"]["run_path"] is not None
    assert summary["run_payload"]["metrics"]["total_duration_ms"] is not None
    assert summary["run_payload"]["metrics"]["peak_rss_self_mb"] is not None
    assert summary["run_payload"]["metrics"]["peak_rss_children_mb"] is not None
    assert "index" in summary["run_payload"]["metrics"]["stages"]
    assert ingest_details["batch_count"] == 1
    assert ingest_details["max_current_rss_mb"] is not None
    assert len(ingest_details["batches"]) == 1
    assert ingest_details["batches"][0]["max_current_rss_mb"] is not None
    assert summary["db_stats"]["raw_conversations_count"] >= 1
    assert len(summary["raw_fanout"]) == summary["db_stats"]["raw_conversations_count"]


async def test_run_probe_can_stage_real_source_subset(tmp_path) -> None:
    source_input_root = tmp_path / "inputs"
    files, total_bytes = _write_probe_sources(
        provider="chatgpt",
        count=2,
        messages_min=3,
        messages_max=4,
        seed=17,
        source_root=source_input_root,
    )
    args = _SourceSubsetArgs(
        workdir=tmp_path / "source-subset-probe",
        source_paths=files,
        source_name="inbox",
    )

    summary = await run_probe(args)

    assert summary["probe"]["input_mode"] == "source-subset"
    assert summary["probe"]["source_name"] == "inbox"
    assert summary["source_inputs"]["input_count"] == 2
    assert summary["source_inputs"]["staged_file_count"] == 2
    assert summary["source_inputs"]["total_bytes"] == total_bytes
    assert summary["db_stats"]["raw_conversations_count"] == 2
    assert summary["db_stats"]["conversations_count"] == 2
    assert len(summary["raw_fanout"]) == 2
    assert summary["result"]["run_path"] is not None


async def test_run_probe_applies_ingest_tuning_overrides(tmp_path) -> None:
    source_input_root = tmp_path / "inputs"
    files, _ = _write_probe_sources(
        provider="chatgpt",
        count=2,
        messages_min=3,
        messages_max=4,
        seed=29,
        source_root=source_input_root,
    )
    args = _SourceSubsetArgs(
        workdir=tmp_path / "source-subset-overrides",
        source_paths=files,
    )
    args.raw_batch_size = 1
    args.ingest_workers = 1

    summary = await run_probe(args)
    ingest_details = summary["run_payload"]["metrics"]["stages"]["ingest"]["details"]["batch_observations"]

    assert summary["probe"]["raw_batch_size"] == 1
    assert summary["probe"]["ingest_workers"] == 1
    assert ingest_details["batch_count"] == 2
    assert all(batch["workers"] == 1 for batch in ingest_details["batches"])


async def test_run_probe_can_measure_ingest_result_sizes(tmp_path) -> None:
    source_input_root = tmp_path / "inputs"
    files, _ = _write_probe_sources(
        provider="chatgpt",
        count=1,
        messages_min=3,
        messages_max=4,
        seed=31,
        source_root=source_input_root,
    )
    args = _SourceSubsetArgs(
        workdir=tmp_path / "source-subset-size-probe",
        source_paths=files,
    )
    args.measure_ingest_result_size = True

    summary = await run_probe(args)
    batch = summary["run_payload"]["metrics"]["stages"]["ingest"]["details"]["batch_observations"]

    assert summary["probe"]["measure_ingest_result_size"] is True
    assert batch["max_result_mb"] is not None
    assert batch["batches"][0]["result_mb"] > 0
    assert batch["batches"][0]["max_result_mb"] > 0


async def test_run_probe_rejects_source_subset_without_source_paths(tmp_path) -> None:
    args = _SourceSubsetArgs(
        workdir=tmp_path / "missing-source-paths",
        source_paths=[],
    )

    with pytest.raises(ValueError, match="--source-path is required"):
        await run_probe(args)


def test_main_writes_json_summary(tmp_path, capsys) -> None:
    workdir = tmp_path / "probe-main"
    json_out = tmp_path / "probe-summary.json"

    exit_code = main([
        "--provider",
        "chatgpt",
        "--count",
        "1",
        "--messages-min",
        "3",
        "--messages-max",
        "4",
        "--stage",
        "parse",
        "--workdir",
        str(workdir),
        "--json-out",
        str(json_out),
    ])

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(json_out.read_text())

    assert exit_code == 0
    assert printed["paths"]["workdir"] == str(workdir.resolve())
    assert written["result"]["run_path"] is not None


def test_main_runs_source_subset_mode(tmp_path, capsys) -> None:
    source_input_root = tmp_path / "inputs"
    files, _ = _write_probe_sources(
        provider="chatgpt",
        count=1,
        messages_min=3,
        messages_max=4,
        seed=23,
        source_root=source_input_root,
    )

    exit_code = main([
        "--input-mode",
        "source-subset",
        "--source-path",
        str(files[0]),
        "--stage",
        "parse",
        "--workdir",
        str(tmp_path / "probe-main-source-subset"),
    ])

    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert printed["probe"]["input_mode"] == "source-subset"
    assert printed["db_stats"]["raw_conversations_count"] == 1


def test_main_uses_ephemeral_workdir_when_omitted(capsys) -> None:
    exit_code = main([
        "--provider",
        "chatgpt",
        "--count",
        "1",
        "--messages-min",
        "3",
        "--messages-max",
        "4",
        "--stage",
        "parse",
    ])

    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "polylogue-pipeline-probe-" in printed["paths"]["workdir"]
    assert printed["result"]["run_path"] is not None


def test_main_returns_nonzero_when_budget_is_exceeded(capsys) -> None:
    exit_code = main([
        "--provider",
        "chatgpt",
        "--count",
        "1",
        "--messages-min",
        "3",
        "--messages-max",
        "4",
        "--stage",
        "parse",
        "--max-total-ms",
        "0.0",
        "--max-peak-rss-mb",
        "0.0",
    ])

    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert printed["budgets"]["ok"] is False
    assert len(printed["budgets"]["violations"]) >= 1


async def test_run_probe_can_sample_archive_subset_and_persist_manifest(tmp_path) -> None:
    source_db, source_blob_root = await _seed_archive_source(tmp_path)
    manifest_out = tmp_path / "subset-manifest.json"
    args = _ArchiveArgs(
        workdir=tmp_path / "archive-probe",
        source_db=source_db,
        source_blob_root=source_blob_root,
        manifest_out=manifest_out,
    )

    summary = await run_probe(args)
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

    assert summary["probe"]["input_mode"] == "archive-subset"
    assert summary["sample"]["selected_count"] == 2
    assert summary["sample"]["provider_counts"] == {"chatgpt": 1, "codex": 1}
    assert summary["db_stats"]["raw_conversations_count"] == 2
    assert summary["db_stats"]["conversations_count"] == 2
    assert len(summary["raw_fanout"]) == 2
    assert sum(item["conversation_count"] for item in summary["raw_fanout"]) == 2
    assert summary["paths"]["manifest_path"].endswith("archive-subset-manifest.json")
    assert manifest["sample_per_provider"] == 1
    assert len(manifest["records"]) == 2


async def test_run_probe_can_replay_archive_subset_manifest(tmp_path) -> None:
    source_db, source_blob_root = await _seed_archive_source(tmp_path)
    manifest_out = tmp_path / "subset-manifest.json"
    first_args = _ArchiveArgs(
        workdir=tmp_path / "archive-probe-first",
        source_db=source_db,
        source_blob_root=source_blob_root,
        manifest_out=manifest_out,
    )
    await run_probe(first_args)

    replay_args = _ArchiveArgs(
        workdir=tmp_path / "archive-probe-replay",
        source_db=source_db,
        source_blob_root=source_blob_root,
        manifest_in=manifest_out,
    )
    replay_args.sample_per_provider = 99
    replay_args.seed = 999

    summary = await run_probe(replay_args)

    assert summary["sample"]["selected_count"] == 2
    assert summary["sample"]["sample_per_provider"] == 1
    assert summary["sample"]["provider_counts"] == {"chatgpt": 1, "codex": 1}


async def test_run_probe_rejects_empty_archive_subset(tmp_path) -> None:
    empty_db = tmp_path / "empty-source.db"
    with open_connection(empty_db):
        pass
    empty_blob_root = tmp_path / "empty-blobs"
    args = _ArchiveArgs(
        workdir=tmp_path / "archive-probe-empty",
        source_db=empty_db,
        source_blob_root=empty_blob_root,
    )

    with pytest.raises(ValueError, match="found no raw conversations"):
        await run_probe(args)
