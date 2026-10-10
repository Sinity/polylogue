"""Small corpus closure and paired experiment qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from devtools.fresh_build_bench.corpus import SampleSource, select_real, verify_manifest
from devtools.fresh_build_bench.experiment import reduce_experiment, run_experiment
from devtools.fresh_build_bench.run import RunConfig


def test_selection_keeps_shared_dependencies_once_and_excludes_whale(tmp_path: Path) -> None:
    root = tmp_path / "input" / ".claude" / "projects"
    project = root / "-synthetic"
    project.mkdir(parents=True)
    for name in ("small-a", "small-b"):
        (project / f"{name}.jsonl").write_bytes(b"{}\n")
    sidecar = project / "small-a" / "tool-results" / "result.txt"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"evidence")
    os.utime(sidecar, ns=(1_000_000_000, 1_000_000_000))
    (project / "whale.jsonl").write_bytes(b"x" * 10_000)
    history = root.parent / "history.jsonl"
    history.write_bytes(b"h" * 300)
    index = project / "sessions-index.json"
    index.write_bytes(b"i" * 300)
    sources = (SampleSource("claude-code", root, "home/.claude/projects", True),)
    first = select_real(tmp_path / "selected", seed=2, max_bytes=620, max_unit_bytes=20, max_units=10, sources=sources)
    again = select_real(tmp_path / "again", seed=2, max_bytes=620, max_unit_bytes=20, max_units=10, sources=sources)
    assert first["files"] == again["files"]
    assert first["parameters"]["provenance"] == again["parameters"]["provenance"]
    assert first["total_bytes"] == 614  # shared history/index charged once
    assert first["parameters"]["selected_units"] == 2
    assert sum(row["oversized_units"] for row in first["parameters"]["strata"]) == 1
    assert any(row["missing"] for row in first["parameters"]["strata"])
    assert "codex" in first["parameters"]["missing_providers"]
    assert (
        first["sidecar_mtimes_ns"]["home/.claude/projects/-synthetic/small-a/tool-results/result.txt"] == 1_000_000_000
    )
    assert all("whale" not in row[0] for row in first["files"])
    verify_manifest(tmp_path / "selected", first)


def _sample(arm: str, seconds: float, **changes: Any) -> dict[str, Any]:
    receipt = {
        "qualified": True,
        "outcome": "terminal",
        "checks": {},
        "corpus": {"digest": "fixed"},
        "config": {"digest": "fixed"},
        "environment": {},
        "effective_compute_policy": {
            "outcome": "bound",
            "states": [{"capacity_slots": 2}],
            "product_binding": {"outcome": "bound"},
        },
        "timing_s": {"promotion": seconds, "terminal": seconds + 1},
        "output_fingerprint": {"digest": "same"},
    }
    receipt.update(changes)
    return {"arm": arm, "receipt": receipt}


def test_distribution_does_not_claim_speed_inside_observed_noise() -> None:
    result = reduce_experiment([_sample("A", 10), _sample("B", 9), _sample("B", 12), _sample("A", 11)])
    assert result["qualified"]
    assert result["metrics"]["promotion"]["assessment"] == "within-observed-noise"
    assert result["metrics"]["promotion"]["arms"]["B"]["raw_s"] == [9, 12]
    single = reduce_experiment([_sample("A", 10), _sample("B", 1)])
    assert single["metrics"]["promotion"]["assessment"] == "single-sample"
    separated = reduce_experiment([_sample("A", 10), _sample("B", 1), _sample("B", 2), _sample("A", 11)])
    assert separated["metrics"]["promotion"]["assessment"] == "B-faster-than-observed-spread"


def test_output_or_policy_changes_refuse_paired_qualification() -> None:
    changed = _sample("B", 1, output_fingerprint={"digest": "different"})
    assert not reduce_experiment([_sample("A", 10), changed])["qualified"]
    assert not reduce_experiment(
        [
            _sample("A", 10),
            _sample("B", 1, effective_compute_policy={"outcome": "bound", "states": [{"capacity_slots": 12}]}),
        ]
    )["qualified"]
    assert not reduce_experiment([_sample("A", 10), _sample("B", 1, effective_compute_policy=None)])["qualified"]


def test_experiment_uses_new_state_and_binds_receipt_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.fresh_build_bench import experiment
    from devtools.fresh_build_bench.corpus import seal

    corpus = tmp_path / "corpus"
    file = corpus / "home" / ".codex" / "sessions" / "rollout.jsonl"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"{}\n")
    seal(corpus, kind="synthetic", parameters={"morphology": "tiny"})
    seen = []

    def build(config: RunConfig, **kwargs: Any) -> dict[str, Any]:
        assert not config.work.exists()
        config.work.mkdir()
        seen.append((config.label, config.candidate, config.work))
        receipt = _sample(config.label, 2)["receipt"]
        (config.work / "receipt.json").write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(experiment, "run_build", build)
    config = RunConfig(corpus, tmp_path / "work", tmp_path / "driver", "python", "test")
    result = run_experiment(config, before=tmp_path / "a", after=tmp_path / "b", repetitions=2)
    assert result["qualified"] and result["complete"]
    assert [arm for arm, _candidate, _work in seen] == list("ABBAABBA")
    assert len({work for _arm, _candidate, work in seen}) == 8
    for sample in result["samples"]:
        assert sample["sha256"] == hashlib.sha256(Path(sample["path"]).read_bytes()).hexdigest()


def test_policy_observer_reads_owner_without_constructing(monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.fresh_build_bench.policy_observer import policy_state
    from polylogue.core import compute

    monkeypatch.setattr(compute, "_SHARED_COMPUTE_ADAPTER", None)
    assert policy_state() is None
    adapter = compute.BoundedComputeAdapter(max_workers=2, queue_units=3, queue_bytes=100)
    try:
        monkeypatch.setattr(compute, "_SHARED_COMPUTE_ADAPTER", adapter)
        state = policy_state()
        assert state is not None
        assert (state["capacity_slots"], state["capacity_units"], state["capacity_bytes"]) == (2, 5, 100)
        assert len(state["classes"]) == 4
        assert "gil_enabled_after_product_imports" in state
    finally:
        adapter.shutdown()


def test_unselected_live_growth_does_not_invalidate_selected_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.fresh_build_bench import corpus

    root = tmp_path / "input" / ".codex" / "sessions"
    day = root / "2026" / "10" / "11"
    day.mkdir(parents=True)
    (day / "rollout-small.jsonl").write_bytes(b"{}\n")
    unselected = day / "rollout-whale.jsonl"
    unselected.write_bytes(b"x" * 2000)
    copy = corpus._copy_private

    def growing_copy(source: Path, destination: Path) -> None:
        copy(source, destination)
        unselected.write_bytes(b"x" * 3000)

    monkeypatch.setattr(corpus, "_copy_private", growing_copy)
    manifest = select_real(
        tmp_path / "out",
        seed=1,
        max_bytes=20,
        max_unit_bytes=20,
        max_units=1,
        sources=(SampleSource("codex", root, "home/.codex/sessions"),),
    )
    assert manifest["total_bytes"] == 3
    assert manifest["parameters"]["population"]["codex"]["bytes"] == 2003
    assert "non-atomic" in manifest["parameters"]["population_observation"]["coherence"]


def test_new_selected_dependency_during_copy_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.fresh_build_bench import corpus

    root = tmp_path / "input" / ".codex" / "sessions"
    day = root / "2026" / "10" / "11"
    day.mkdir(parents=True)
    (day / "rollout-small.jsonl").write_bytes(b"{}\n")
    copy = corpus._copy_private

    def new_dependency(source: Path, destination: Path) -> None:
        copy(source, destination)
        (root.parent / "history.jsonl").write_bytes(b"{}\n")

    monkeypatch.setattr(corpus, "_copy_private", new_dependency)
    with pytest.raises(ValueError, match="dependency membership changed"):
        select_real(
            tmp_path / "out",
            seed=1,
            max_bytes=20,
            max_unit_bytes=20,
            max_units=1,
            sources=(SampleSource("codex", root, "home/.codex/sessions"),),
        )


@pytest.mark.uses_real_clock
def test_launcher_rebinds_candidate_when_interpreter_wrapper_unsets_pythonpath(tmp_path: Path) -> None:
    """A launcher that drops PYTHONPATH must still import the requested candidate."""
    import subprocess
    import sys

    from devtools.fresh_build_bench.run import _daemon_command

    package = tmp_path / "candidate" / "polylogue"
    commands = package / "daemon" / "commands.py"
    commands.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (commands.parent / "__init__.py").write_text("")
    commands.write_text("def main():\n    print('requested-candidate')\n")
    observer = tmp_path / "observer.py"
    observer.write_text("def start_from_environment():\n    pass\n")
    launcher = tmp_path / "python-wrapper"
    # Synthetic wrapper intentionally models the installed wrapper's unset.
    launcher.write_text(f'#!/bin/sh\nunset PYTHONPATH\nexec "{sys.executable}" "$@"\n')
    launcher.chmod(0o700)
    config = RunConfig(tmp_path, tmp_path, package.parent, str(launcher), "binding")
    env = dict(os.environ, POLYLOGUE_BENCH_CANDIDATE=str(package.parent), POLYLOGUE_BENCH_POLICY_OBSERVER=str(observer))
    env.pop("POLYLOGUE_BENCH_SAMPLER", None)
    result = subprocess.run(_daemon_command(config), env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "requested-candidate"


def test_policy_binding_refuses_imported_product_outside_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.fresh_build_bench.policy_observer import product_binding
    from polylogue.core import compute
    from polylogue.daemon import commands

    monkeypatch.setenv("POLYLOGUE_BENCH_CANDIDATE", str(tmp_path / "wrong-candidate"))
    assert compute.__file__ and commands.__file__
    binding = product_binding()
    assert binding["outcome"] == "mismatch"
    assert not binding["modules"]["polylogue.core.compute"]["bound"]
    assert binding["modules"]["polylogue.core.compute"]["sha256"]
