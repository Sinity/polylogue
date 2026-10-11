"""Shared matched runner and immutable component reduction discriminators."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from devtools.fresh_build_bench.components import refresh_component_experiment
from devtools.fresh_build_bench.experiment import matched_schedule, run_matched_sequence


@pytest.fixture
def external_component_scratch() -> Iterator[Path]:
    from devtools.pytest_slot import ISOLATED_ARCHIVE_PARENT

    ISOLATED_ARCHIVE_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="component-seed-", dir=ISOLATED_ARCHIVE_PARENT) as directory:
        yield Path(directory)


def test_component_recipe_refuses_silent_domain_substitution() -> None:
    from devtools.fresh_build_bench.components import _require_loose_jsonl_recipe

    route = {
        "domain": "complete-loose-jsonl",
        "validation_mode": "advisory",
        "jsonl": True,
        "captured_zip_coordinate": None,
        "schema_resolution": None,
        "schema_resolution_is_explicit": False,
    }
    recipe = {"route": route, "provider": "codex", "sidecar_context": "absent"}
    _require_loose_jsonl_recipe(recipe, "validation")
    _require_loose_jsonl_recipe(recipe, "preparation")
    for changed in (
        {},
        {**route, "validation_mode": "strict"},
        {**route, "captured_zip_coordinate": "member"},
        {**route, "schema_resolution_is_explicit": True},
    ):
        with pytest.raises(ValueError):
            _require_loose_jsonl_recipe({**recipe, "route": changed}, "validation")
    with pytest.raises(ValueError):
        _require_loose_jsonl_recipe({**recipe, "sidecar_context": "unavailable"}, "preparation")
    page = {
        "route": {"domain": "synthetic-retained-page", "validation_mode": "advisory"},
        "provider": "codex",
        "workers": 2,
        "queue_units": 8,
        "queue_bytes": 1024 * 1024,
    }
    _require_loose_jsonl_recipe(page, "page")
    for changed in ({**page, "provider": "claude-code"}, {**page, "queue_bytes": 0}, {**page, "route": route}):
        with pytest.raises(ValueError):
            _require_loose_jsonl_recipe(changed, "page")


def test_preparation_consumers_close_before_backing_retirement_and_no_ambient_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from devtools.fresh_build_bench.components import _prepare_observation
    from polylogue.pipeline import ids
    from polylogue.sources import prepared_jsonl

    events = []

    class Attempt:
        blob_hash = "input"
        error = deferred = unsupported_shape = missing_profile_identity = False
        codex_state_kind = shard_path = resolved_provider = parsed_prefix_size = None

        def iter_sessions(self):
            try:
                yield SimpleNamespace(session_events=[])
            finally:
                events.append("reader-closed")

        def discard(self):
            events.append("retired")

    def prepare(*args: Any, **kwargs: Any) -> Attempt:
        resolver = kwargs["sidecar_resolver"]
        assert not resolver.claude_code_scope(tmp_path / "unbound-external.jsonl").available
        assert not resolver.gemini_cli_scope(tmp_path / "unbound-external.jsonl", "session").available
        return Attempt()

    def fail_hash(_session: Any) -> str:
        raise ValueError("consumer failed")

    monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", prepare)
    monkeypatch.setattr(ids, "session_content_hash", fail_hash)
    with pytest.raises(ValueError, match="consumer failed"):
        _prepare_observation(
            tmp_path / "input.jsonl",
            {"provider": "codex", "source_path": "neutral.jsonl", "raw_id": "raw", "input_sha256": "input"},
            tmp_path,
            None,
        )
    assert events == ["reader-closed", "retired"]


def test_matched_sequence_preserves_actual_bytes_and_stops_on_invalid_work(tmp_path: Path) -> None:
    seen = []

    def invoke(ordinal: int, arm: str) -> Path:
        seen.append(arm)
        path = tmp_path / f"{ordinal}.json"
        path.write_text(json.dumps({"valid": ordinal < 3, "arm": arm}))
        return path

    samples = run_matched_sequence(
        order="ABBA", repetitions=1, invoke=invoke, valid=lambda receipt: receipt["valid"], progress=lambda _line: None
    )
    assert seen == ["A", "B", "B"]
    assert [sample["ordinal"] for sample in samples] == [1, 2, 3]
    assert all(hashlib.sha256(Path(sample["path"]).read_bytes()).hexdigest() == sample["sha256"] for sample in samples)
    with pytest.raises(ValueError):
        list(matched_schedule("AAB", 1))


def test_component_report_recomputes_equivalence_and_refuses_changed_original(tmp_path: Path) -> None:
    launch = {
        "order": "ABBA",
        "repetitions": 1,
        "candidates": {"A": {"git_sha": "a"}, "B": {"git_sha": "b"}},
        "product_trees": {"A": "tree-a", "B": "tree-b"},
        "input_sha256": "input",
        "recipe_sha256": "recipe",
        "runtime": {"sha256": "runtime"},
    }
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(json.dumps(launch))
    launch_digest = hashlib.sha256(launch_path.read_bytes()).hexdigest()
    samples = []
    for ordinal, arm in matched_schedule("ABBA", 1):
        work = tmp_path / f"{ordinal:03d}-{arm}"
        work.mkdir()
        receipt = {
            "launch_binding": {"sha256": launch_digest, "ordinal": ordinal, "arm": arm},
            "candidate": launch["candidates"][arm],
            "product_tree": launch["product_trees"][arm],
            "input_sha256": "input",
            "recipe_sha256": "recipe",
            "runtime": {"sha256": "runtime"},
            "timing": {"wall_s": ordinal, "thread_cpu_s": ordinal, "process_cpu_s": ordinal},
            "output": {"count": 3 if arm == "A" else 4},
        }
        path = work / "receipt.json"
        path.write_text(json.dumps(receipt))
        samples.append(
            {"ordinal": ordinal, "arm": arm, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps({"launch_sha256": launch_digest, "samples": samples, "complete": True, "equivalent_outputs": True})
    )
    refreshed = refresh_component_experiment(experiment)
    assert refreshed["complete"] and not refreshed["equivalent_outputs"]
    first = Path(samples[0]["path"])
    first.write_text(first.read_text() + " ")
    with pytest.raises(ValueError, match="constituent changed"):
        refresh_component_experiment(experiment)


def test_component_runtime_binding_refuses_gil_change(monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.fresh_build_bench import components

    declared = {"sha256": "binary", "version": "3.14t", "gil_enabled": False}
    monkeypatch.setattr(components, "_runtime_identity", lambda: {**declared, "gil_enabled": True})
    with pytest.raises(ValueError, match="GIL state"):
        components._require_runtime(declared)


def test_product_content_binding_uses_original_launch_identity(tmp_path: Path) -> None:
    from devtools.fresh_build_bench.report import product_content_bound

    binding = tmp_path / "product-tree.json"
    binding.write_text(json.dumps({"product_tree": "original"}))
    assert product_content_bound(tmp_path, "original")
    binding.write_text(json.dumps({"product_tree": "changed"}))
    assert not product_content_bound(tmp_path, "original")
    binding.write_text("not-json")
    assert not product_content_bound(tmp_path, "original")


def test_work_ledger_cli_adopts_existing_export_owner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools.fresh_build_bench import work_ledger
    from devtools.fresh_build_bench.cli import main

    calls = []

    def export(source: Path, output: Path) -> dict[str, Any]:
        calls.append((source, output))
        return {"format": "polylogue.work-ledger-export.v1", "state": "complete"}

    monkeypatch.setattr(work_ledger, "export_work_ledger", export)
    assert main(["ledger", "/neutral/events.jsonl", "--out", "/neutral/result"]) == 0
    assert calls == [(Path("/neutral/events.jsonl"), Path("/neutral/result"))]
    assert json.loads(capsys.readouterr().out)["state"] == "complete"


def test_prepared_seed_refuses_changed_receipt_before_adoption(external_component_scratch: Path) -> None:
    from devtools.fresh_build_bench.run import RunConfig, _seed_artifact, bind_seed

    tmp_path = external_component_scratch
    seed = tmp_path / "receipt.json"
    seed.write_text("{}")
    config = RunConfig(
        corpus=tmp_path / "corpus",
        work=tmp_path / "work",
        candidate=tmp_path / "candidate",
        python="python",
        label="neutral",
        route="restart",
        seed_receipt=seed,
        seed_receipt_sha256="different-original",
    )
    with pytest.raises(ValueError, match="changed after launch"):
        _seed_artifact(config)
    with pytest.raises(ValueError, match="requires --seed-receipt"):
        bind_seed(
            RunConfig(
                corpus=tmp_path,
                work=tmp_path / "work",
                candidate=tmp_path,
                python="python",
                label="neutral",
                route="upgrade",
            )
        )


def test_seeded_route_config_digest_binds_route_and_original_seed(tmp_path: Path) -> None:
    from dataclasses import replace

    from devtools.fresh_build_bench.report import config_digest
    from devtools.fresh_build_bench.run import RunConfig

    base = RunConfig(corpus=tmp_path, work=tmp_path / "work", candidate=tmp_path, python="python", label="neutral")
    restart = replace(base, route="restart", seed_receipt_sha256="receipt", seed_manifest_id="manifest")
    assert (
        len(
            {
                config_digest(base),
                config_digest(restart),
                config_digest(replace(restart, route="upgrade")),
                config_digest(replace(restart, seed_manifest_id="changed")),
            }
        )
        == 4
    )


def test_direct_validation_cli_launches_clean_child_before_product_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from devtools.fresh_build_bench import components

    seen = []

    def run(command: list[str], *, check: bool) -> SimpleNamespace:
        seen.append(command)
        assert not check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(components.subprocess, "run", run)
    assert components.main(["validation", "--candidate", "/neutral/candidate"]) == 7
    assert seen[0][1] == "-B" and seen[0][3:] == ["validation", "--candidate", "/neutral/candidate"]


def test_component_phase_unknown_cleanup_uses_nullable_scope_field() -> None:
    from devtools.fresh_build_bench.components import _validation_observation

    phase = _validation_observation({"wall_s": 0.001, "process_cpu_s": 0.001}, 1)
    assert phase.cleanup_complete is None
    assert phase.peak_rss_bytes is None


def test_component_calibration_uses_complete_product_tree_not_selected_imports() -> None:
    from devtools.fresh_build_bench.components import _reduce_components

    def sample(arm: str, tree: str) -> dict[str, Any]:
        return {
            "arm": arm,
            "receipt": {
                "product_tree": tree,
                "imported_product_modules": {"polylogue.partial": {"sha256": "same"}},
                "output": {},
                "timing": {"wall_s": 1, "thread_cpu_s": 1, "process_cpu_s": 1},
            },
        }

    assert (
        _reduce_components([sample("A", "one"), sample("B", "two")], complete=True)["treatment"]
        == "insufficient-executed-treatment-proof"
    )
    assert (
        _reduce_components([sample("A", "one"), sample("B", "one")], complete=True)["treatment"]
        == "A/A-product-content-calibration"
    )


def test_page_owner_construction_failure_still_settles_compute_and_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from devtools.fresh_build_bench.components import _page_observation
    from polylogue.core import compute
    from polylogue.daemon import raw_observation_owner, write_coordinator
    from tests.infra import archive_templates, selected_raw_preparation, workload_artifacts

    events: list[str] = []

    class Kernel:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def shutdown(self, *, wait: bool) -> None:
            assert wait
            events.append("compute-settled")

    class Coordinator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def shutdown(self, *, timeout: float) -> bool:
            events.append("writer-settled")
            return True

    async def write(root: Path, callback: Any) -> Any:
        return callback()

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("constructor-refusal")

    monkeypatch.setattr(compute, "BoundedComputeAdapter", Kernel)
    monkeypatch.setattr(write_coordinator, "DaemonWriteCoordinator", Coordinator)
    monkeypatch.setattr(write_coordinator, "DaemonWriteThreadBridge", lambda *args: object())
    monkeypatch.setattr(raw_observation_owner, "RawObservationConvergenceOwner", unavailable)
    monkeypatch.setattr(archive_templates, "run_archive_fixture_write", write)
    monkeypatch.setattr(archive_templates, "bootstrap_archive_root", lambda root: None)
    monkeypatch.setattr(selected_raw_preparation, "acquire_independent_codex_raws", lambda *args, **kwargs: ("r",))
    monkeypatch.setattr(workload_artifacts, "seal_fixture_tree", lambda root: events.append("fixture-sealed"))
    monkeypatch.setattr(workload_artifacts.ImmutableTreeArtifact, "adopt", lambda *args, **kwargs: SimpleNamespace())

    def clone(*args: Any) -> Any:
        import asyncio

        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return nullcontext()

    monkeypatch.setattr(workload_artifacts, "clone_immutable_tree", clone)
    owned = tmp_path / "input.json"
    owned.write_text(json.dumps({"count": 1, "messages": 1, "text_size": 8, "namespace": "neutral"}))
    with pytest.raises(RuntimeError, match="constructor-refusal"):
        _page_observation(
            owned, {"workers": 2, "queue_units": 8, "queue_bytes": 1024, "input_sha256": "input"}, tmp_path
        )
    assert events == ["fixture-sealed", "compute-settled", "writer-settled"]


def test_component_output_equivalence_excludes_descriptive_physical_resources() -> None:
    from devtools.fresh_build_bench.components import _reduce_components

    samples = [
        {
            "arm": arm,
            "receipt": {
                "product_tree": "same",
                "output": {"census": {"rows": {"sessions": 2}}, "output_fingerprint": {"digest": "bound"}},
                "physical_census": {"index_bytes": size},
                "timing": {"wall_s": 1, "thread_cpu_s": 1, "process_cpu_s": 1},
            },
        }
        for arm, size in (("A", 4096), ("B", 8192))
    ]
    assert _reduce_components(samples, complete=True)["equivalent_outputs"]
    samples[1]["receipt"]["output"]["census"]["rows"]["sessions"] = 1
    assert not _reduce_components(samples, complete=True)["equivalent_outputs"]
