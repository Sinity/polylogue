"""Tests for devtools.pipeline_probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import pytest

from devtools.pipeline_probe import ProbeSummary, _build_budget_report, _write_probe_sources, main, run_probe
from devtools.regression_cases import RegressionCase
from polylogue.scenarios import CorpusRequest, CorpusScenario, CorpusSpec, PipelineProbeRequest
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import RawConversationRecord
from polylogue.storage.sqlite import create_backend
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import Provider, ValidationStatus

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonArray: TypeAlias = list[JsonValue]


def _require_json_object(value: object) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _require_json_array(value: object) -> JsonArray:
    assert isinstance(value, list)
    return value


def _require_int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _require_str(value: object) -> str:
    assert isinstance(value, str)
    return value


def _require_bool(value: object) -> bool:
    assert isinstance(value, bool)
    return value


def _require_number(value: object) -> int | float:
    assert isinstance(value, int | float)
    return value


def _json_path(value: object, *keys: str) -> object:
    current: object = value
    for key in keys:
        current = _require_json_object(current)[key]
    return current


def _load_json_object(text: str) -> JsonObject:
    return _require_json_object(json.loads(text))


def test_build_budget_report_accounts_for_recorded_child_rss() -> None:
    request = PipelineProbeRequest(max_peak_rss_mb=100.0)
    summary: ProbeSummary = {
        "run_payload": {
            "metrics": {
                "peak_rss_self_mb": 70.0,
                "peak_rss_children_mb": 40.0,
            }
        },
        "result": {},
        "probe": {},
        "paths": {},
        "provenance": {"git_commit": None, "worktree_dirty": None},
        "db_stats": {},
        "raw_fanout": [],
    }

    report = _build_budget_report(summary, request)

    assert report is not None
    assert report["ok"] is False
    assert report["observed_peak_rss_mb"] == 110.0
    assert report["observed_peak_rss_self_mb"] == 70.0
    assert report["observed_peak_rss_children_mb"] == 40.0
    assert report["violations"] == ["peak RSS 110.0 MiB exceeded budget 100.0 MiB"]


def _synthetic_request(workdir: Path) -> PipelineProbeRequest:
    return PipelineProbeRequest(
        stage="parse",
        workdir=str(workdir),
        corpus_request=CorpusRequest(
            providers=("chatgpt",),
            source="default",
            count=1,
            messages_min=3,
            messages_max=4,
            seed=7,
            style="default",
        ),
    )


def _archive_request(
    *,
    workdir: Path,
    source_db: Path,
    source_blob_root: Path,
    manifest_out: Path | None = None,
    manifest_in: Path | None = None,
    sample_per_provider: int = 1,
    seed: int = 11,
) -> PipelineProbeRequest:
    return PipelineProbeRequest(
        stage="parse",
        input_mode="archive-subset",
        workdir=str(workdir),
        source_db=str(source_db),
        source_blob_root=str(source_blob_root),
        manifest_out=str(manifest_out) if manifest_out is not None else None,
        manifest_in=str(manifest_in) if manifest_in is not None else None,
        sample_per_provider=sample_per_provider,
        corpus_request=CorpusRequest(
            count=1,
            messages_min=3,
            messages_max=4,
            seed=seed,
            style="default",
        ),
    )


def _source_subset_request(
    *,
    workdir: Path,
    source_paths: list[Path],
    source_name: str = "inbox",
    raw_batch_size: int | None = None,
    ingest_workers: int | None = None,
    measure_ingest_result_size: bool = False,
) -> PipelineProbeRequest:
    return PipelineProbeRequest(
        stage="parse",
        input_mode="source-subset",
        workdir=str(workdir),
        source_paths=tuple(str(path) for path in source_paths),
        source_name=source_name,
        raw_batch_size=raw_batch_size,
        ingest_workers=ingest_workers,
        measure_ingest_result_size=measure_ingest_result_size,
        corpus_request=CorpusRequest(
            count=1,
            messages_min=3,
            messages_max=4,
            seed=13,
            style="default",
        ),
    )


async def _seed_archive_source(tmp_path: Path) -> tuple[Path, Path]:
    source_db = tmp_path / "source.db"
    source_blob_root = tmp_path / "source-blobs"
    blob_store = BlobStore(source_blob_root)
    backend = create_backend(db_path=source_db)
    repository = ConversationRepository(backend=backend)
    try:
        raw_payloads = [
            (
                "chatgpt",
                "chatgpt-main",
                SyntheticCorpus.generate_for_spec(
                    CorpusSpec.for_provider(
                        "chatgpt",
                        count=1,
                        messages_min=3,
                        messages_max=3,
                        seed=100,
                        origin="generated.test-pipeline-probe",
                        tags=("synthetic", "test", "pipeline-probe"),
                    )
                )[0],
            ),
            (
                "chatgpt",
                "chatgpt-sidecar",
                SyntheticCorpus.generate_for_spec(
                    CorpusSpec.for_provider(
                        "chatgpt",
                        count=1,
                        messages_min=3,
                        messages_max=3,
                        seed=101,
                        origin="generated.test-pipeline-probe",
                        tags=("synthetic", "test", "pipeline-probe"),
                    )
                )[0],
            ),
            (
                "codex",
                "codex-main",
                SyntheticCorpus.generate_for_spec(
                    CorpusSpec.for_provider(
                        "codex",
                        count=1,
                        messages_min=3,
                        messages_max=3,
                        seed=200,
                        origin="generated.test-pipeline-probe",
                        tags=("synthetic", "test", "pipeline-probe"),
                    )
                )[0],
            ),
            (
                "codex",
                "codex-sidecar",
                SyntheticCorpus.generate_for_spec(
                    CorpusSpec.for_provider(
                        "codex",
                        count=1,
                        messages_min=3,
                        messages_max=3,
                        seed=201,
                        origin="generated.test-pipeline-probe",
                        tags=("synthetic", "test", "pipeline-probe"),
                    )
                )[0],
            ),
        ]
        for index, (provider_name, source_name, raw_bytes) in enumerate(raw_payloads):
            raw_id, blob_size = blob_store.write_from_bytes(raw_bytes)
            await repository.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name=provider_name,
                    payload_provider=Provider.from_string(provider_name),
                    source_name=source_name,
                    source_path=f"/tmp/{source_name}-{index}",
                    source_index=index,
                    blob_size=blob_size,
                    acquired_at=f"2026-04-0{index + 1}T12:00:00Z",
                    file_mtime=f"2026-04-0{index + 1}T11:00:00Z",
                    parsed_at="2026-04-01T00:00:00Z",
                    validated_at="2026-04-01T00:00:00Z",
                    validation_status=ValidationStatus.PASSED,
                )
            )
    finally:
        await repository.close()

    return source_db, source_blob_root


async def test_run_probe_emits_real_pipeline_summary(tmp_path: Path) -> None:
    summary = await run_probe(_synthetic_request(tmp_path / "probe"))
    run_payload = _require_json_object(summary["run_payload"])
    metrics = _require_json_object(run_payload["metrics"])
    stages = _require_json_object(metrics["stages"])
    ingest = _require_json_object(stages["ingest"])
    details = _require_json_object(ingest["details"])
    acquisition_details = _require_json_object(details["acquisition"])
    ingest_details = _require_json_object(details["batch_observations"])

    assert _json_path(summary, "probe", "provider") == "chatgpt"
    assert _json_path(summary, "probe", "corpus_source") == "default"
    assert _json_path(summary, "probe", "stage_sequence") == ["acquire", "parse"]
    assert _json_path(summary, "result", "run_path") is not None
    assert metrics["total_duration_ms"] is not None
    assert metrics["peak_rss_self_mb"] is not None
    assert metrics["peak_rss_children_mb"] is not None
    assert "index" not in stages
    assert isinstance(acquisition_details, dict)
    assert _require_int(ingest_details["batch_count"]) == 1
    assert ingest_details["max_current_rss_mb"] is not None
    batches = _require_json_array(ingest_details["batches"])
    assert len(batches) == 1
    first_batch = _require_json_object(batches[0])
    assert first_batch["max_current_rss_mb"] is not None
    assert _require_number(first_batch["result_wait_elapsed_ms"]) >= 0
    assert _require_number(first_batch["write_elapsed_ms"]) >= 0
    assert _require_number(first_batch["commit_elapsed_ms"]) >= 0
    assert _require_number(first_batch["raw_state_update_elapsed_ms"]) >= 0
    raw_count = _require_int(_json_path(summary, "db_stats", "raw_conversations_count"))
    assert raw_count >= 1
    assert len(_require_json_array(summary["raw_fanout"])) == raw_count


async def test_run_probe_can_stage_real_source_subset(tmp_path: Path) -> None:
    source_input_root = tmp_path / "inputs"
    files, total_bytes = _write_probe_sources(
        request=CorpusRequest(
            providers=("chatgpt",),
            source="default",
            count=2,
            messages_min=3,
            messages_max=4,
            seed=17,
            style="default",
        ),
        source_root=source_input_root,
    )
    request = _source_subset_request(
        workdir=tmp_path / "source-subset-probe",
        source_paths=files,
        source_name="inbox",
    )

    summary = await run_probe(request)

    source_inputs = _require_json_object(summary["source_inputs"])
    provenance = _require_json_object(summary["provenance"])

    assert _json_path(summary, "probe", "input_mode") == "source-subset"
    assert _json_path(summary, "probe", "source_name") == "inbox"
    assert _json_path(summary, "probe", "stage_sequence") == ["acquire", "parse"]
    assert _require_int(source_inputs["input_count"]) == 2
    assert _require_int(source_inputs["staged_file_count"]) == 2
    assert _require_int(source_inputs["total_bytes"]) == total_bytes
    assert provenance["git_commit"] is not None
    assert isinstance(_require_bool(provenance["worktree_dirty"]), bool)
    assert len(_require_str(provenance["source_inputs_sha256"])) == 64
    fingerprints = _require_json_array(provenance["source_input_fingerprints"])
    assert len(fingerprints) == 2
    assert all(len(_require_str(_require_json_object(entry)["sha256"])) == 64 for entry in fingerprints)


def test_write_probe_sources_threads_through_corpus_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[CorpusRequest] = []

    def fake_resolve_scenarios(self: CorpusRequest, **kwargs: object) -> tuple[CorpusScenario, ...]:
        assert kwargs["origin"] == "compiled.pipeline-probe"
        assert kwargs["tags"] == ("synthetic", "probe", "scenario")
        calls.append(self)
        return (
            CorpusScenario(
                provider="chatgpt",
                corpus_specs=(CorpusSpec.for_provider("chatgpt", count=1, origin="test"),),
                origin="compiled.pipeline-probe",
            ),
        )

    class _Artifact:
        raw_bytes = b"{}"

    class _Batch:
        artifacts = (_Artifact(),)

    class _Written:
        files = (tmp_path / "source.json",)
        batch = _Batch()

    monkeypatch.setattr("polylogue.scenarios.corpus.CorpusRequest.resolve_scenarios", fake_resolve_scenarios)
    monkeypatch.setattr(
        "devtools.pipeline_probe.staging.SyntheticCorpus.write_specs_artifacts",
        lambda *_args, **_kwargs: [_Written()],
    )

    files, total_bytes = _write_probe_sources(
        request=CorpusRequest(
            providers=("chatgpt",),
            source="inferred",
            count=1,
            messages_min=3,
            messages_max=4,
            seed=17,
            style="default",
        ),
        source_root=tmp_path / "sources",
    )

    assert len(calls) == 1
    assert calls[0].source_kind.value == "inferred"
    assert files == [tmp_path / "source.json"]
    assert total_bytes == 2


async def test_run_probe_applies_ingest_tuning_overrides(tmp_path: Path) -> None:
    source_input_root = tmp_path / "inputs"
    files, _ = _write_probe_sources(
        request=CorpusRequest(
            providers=("chatgpt",),
            source="default",
            count=2,
            messages_min=3,
            messages_max=4,
            seed=29,
            style="default",
        ),
        source_root=source_input_root,
    )
    request = _source_subset_request(
        workdir=tmp_path / "source-subset-overrides",
        source_paths=files,
        raw_batch_size=1,
        ingest_workers=1,
    )

    summary = await run_probe(request)
    ingest_details = _require_json_object(
        _json_path(summary, "run_payload", "metrics", "stages", "ingest", "details", "batch_observations")
    )

    assert _json_path(summary, "probe", "raw_batch_size") == 1
    assert _json_path(summary, "probe", "ingest_workers") == 1
    assert _require_int(ingest_details["batch_count"]) == 2
    assert all(_require_json_object(batch)["workers"] == 1 for batch in _require_json_array(ingest_details["batches"]))


async def test_run_probe_can_measure_ingest_result_sizes(tmp_path: Path) -> None:
    source_input_root = tmp_path / "inputs"
    files, _ = _write_probe_sources(
        request=CorpusRequest(
            providers=("chatgpt",),
            source="default",
            count=1,
            messages_min=3,
            messages_max=4,
            seed=31,
            style="default",
        ),
        source_root=source_input_root,
    )
    request = _source_subset_request(
        workdir=tmp_path / "source-subset-size-probe",
        source_paths=files,
        measure_ingest_result_size=True,
    )

    summary = await run_probe(request)
    batch = _require_json_object(
        _json_path(summary, "run_payload", "metrics", "stages", "ingest", "details", "batch_observations")
    )

    assert _json_path(summary, "probe", "measure_ingest_result_size") is True
    assert batch["max_result_mb"] is not None
    first_batch = _require_json_object(_require_json_array(batch["batches"])[0])
    assert _require_number(first_batch["result_mb"]) > 0
    assert _require_number(first_batch["max_result_mb"]) > 0


async def test_run_probe_rejects_source_subset_without_source_paths(tmp_path: Path) -> None:
    request = _source_subset_request(
        workdir=tmp_path / "missing-source-paths",
        source_paths=[],
    )

    with pytest.raises(ValueError, match="--source-path is required"):
        await run_probe(request)


def test_main_writes_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workdir = tmp_path / "probe-main"
    json_out = tmp_path / "probe-summary.json"

    exit_code = main(
        [
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
        ]
    )

    printed = _load_json_object(capsys.readouterr().out)
    written = _load_json_object(json_out.read_text())

    assert exit_code == 0
    assert _json_path(printed, "paths", "workdir") == str(workdir.resolve())
    assert _json_path(written, "result", "run_path") is not None


def test_main_runs_source_subset_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source_input_root = tmp_path / "inputs"
    files, _ = _write_probe_sources(
        request=CorpusRequest(
            providers=("chatgpt",),
            source="default",
            count=1,
            messages_min=3,
            messages_max=4,
            seed=23,
            style="default",
        ),
        source_root=source_input_root,
    )

    exit_code = main(
        [
            "--input-mode",
            "source-subset",
            "--source-path",
            str(files[0]),
            "--stage",
            "parse",
            "--workdir",
            str(tmp_path / "probe-main-source-subset"),
        ]
    )

    printed = _load_json_object(capsys.readouterr().out)

    assert exit_code == 0
    assert _json_path(printed, "probe", "input_mode") == "source-subset"
    assert _json_path(printed, "db_stats", "raw_conversations_count") == 1


def test_main_uses_ephemeral_workdir_when_omitted(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
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
        ]
    )

    printed = _load_json_object(capsys.readouterr().out)

    assert exit_code == 0
    assert "polylogue-pipeline-probe-" in _require_str(_json_path(printed, "paths", "workdir"))
    assert _json_path(printed, "result", "run_path") is not None


def test_main_returns_nonzero_when_budget_is_exceeded(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
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
        ]
    )

    printed = _load_json_object(capsys.readouterr().out)

    assert exit_code == 1
    assert _json_path(printed, "budgets", "ok") is False
    assert len(_require_json_array(_json_path(printed, "budgets", "violations"))) >= 1


def test_main_can_capture_probe_summary_as_regression_case(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "regression-cases"

    exit_code = main(
        [
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
            "--capture-regression",
            "Probe Budget Drift",
            "--regression-output-dir",
            str(output_dir),
            "--regression-tag",
            "budget",
            "--regression-note",
            "captured by pipeline-probe",
        ]
    )

    printed = _load_json_object(capsys.readouterr().out)
    regression_case = _require_json_object(_json_path(printed, "regression_case"))
    case_path = Path(_require_str(regression_case["path"]))
    case = RegressionCase.read(case_path)

    assert exit_code == 1
    assert case_path.parent == output_dir
    assert case.name == "Probe Budget Drift"
    assert case.tags == ("budget",)
    assert case.notes == ("captured by pipeline-probe",)
    assert _json_path(case.summary, "budgets", "ok") is False


async def test_run_probe_can_sample_archive_subset_and_persist_manifest(tmp_path: Path) -> None:
    source_db, source_blob_root = await _seed_archive_source(tmp_path)
    manifest_out = tmp_path / "subset-manifest.json"
    request = _archive_request(
        workdir=tmp_path / "archive-probe",
        source_db=source_db,
        source_blob_root=source_blob_root,
        manifest_out=manifest_out,
    )

    summary = await run_probe(request)
    manifest = _load_json_object(manifest_out.read_text(encoding="utf-8"))

    assert _json_path(summary, "probe", "input_mode") == "archive-subset"
    assert _json_path(summary, "sample", "selected_count") == 2
    assert _json_path(summary, "sample", "provider_counts") == {"chatgpt": 1, "codex": 1}
    assert _json_path(summary, "db_stats", "raw_conversations_count") == 2
    assert _json_path(summary, "db_stats", "conversations_count") == 2
    raw_fanout = _require_json_array(summary["raw_fanout"])
    assert len(raw_fanout) == 2
    assert sum(_require_int(_require_json_object(item)["conversation_count"]) for item in raw_fanout) == 2
    assert _require_str(_json_path(summary, "paths", "manifest_path")).endswith("archive-subset-manifest.json")
    assert len(_require_str(_json_path(summary, "provenance", "manifest_sha256"))) == 64
    assert manifest["sample_per_provider"] == 1
    assert len(_require_json_array(manifest["records"])) == 2


async def test_run_probe_can_replay_archive_subset_manifest(tmp_path: Path) -> None:
    source_db, source_blob_root = await _seed_archive_source(tmp_path)
    manifest_out = tmp_path / "subset-manifest.json"
    first_request = _archive_request(
        workdir=tmp_path / "archive-probe-first",
        source_db=source_db,
        source_blob_root=source_blob_root,
        manifest_out=manifest_out,
    )
    await run_probe(first_request)

    replay_request = _archive_request(
        workdir=tmp_path / "archive-probe-replay",
        source_db=source_db,
        source_blob_root=source_blob_root,
        manifest_in=manifest_out,
        sample_per_provider=99,
        seed=999,
    )

    summary = await run_probe(replay_request)

    assert _json_path(summary, "sample", "selected_count") == 2
    assert _json_path(summary, "sample", "sample_per_provider") == 1
    assert _json_path(summary, "sample", "provider_counts") == {"chatgpt": 1, "codex": 1}


async def test_run_probe_rejects_empty_archive_subset(tmp_path: Path) -> None:
    empty_db = tmp_path / "empty-source.db"
    with open_connection(empty_db):
        pass
    empty_blob_root = tmp_path / "empty-blobs"
    request = _archive_request(
        workdir=tmp_path / "archive-probe-empty",
        source_db=empty_db,
        source_blob_root=empty_blob_root,
    )

    with pytest.raises(ValueError, match="found no raw conversations"):
        await run_probe(request)
