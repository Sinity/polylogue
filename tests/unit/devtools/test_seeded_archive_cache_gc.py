from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path

import pytest

from devtools import seeded_archive_cache_gc as command
from polylogue.scenarios import CorpusSpec
from tests.infra import workload_artifacts
from tests.infra.workload_artifacts import (
    ArtifactGcDisposition,
    SeededArchiveArtifact,
    SeededArchiveReachabilityInventory,
    build_seeded_archive,
    copy_seeded_archive,
    current_seeded_archive_reachability,
    gc_seeded_archive_artifacts,
)
from tests.infra.workload_declarations import c03_semantic_corpus_spec

pytest_plugins = ("tests.infra.corpus_fixtures",)


def _stale_specs(seed: int) -> tuple[CorpusSpec, ...]:
    """A small unreachable recipe; collection semantics do not depend on its size."""
    return (
        dataclasses.replace(
            c03_semantic_corpus_spec(), seed=seed, count=2, session_native_ids=("c03-target", "c03-irrelevant-000")
        ),
    )


def _age_artifact(root: Path, *, now: float = 10_000.0) -> None:
    old = now - 10_000
    os.utime(root, (old, old))
    os.utime(root / "manifest.json", (old, old))


def test_route_refuses_a_partial_implicit_inventory_before_gc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = current_seeded_archive_reachability()
    partial = SeededArchiveReachabilityInventory(inventory.entries[:-1])
    monkeypatch.setattr(command, "current_seeded_archive_reachability", lambda: partial)
    monkeypatch.setattr(command, "gc_seeded_archive_artifacts", pytest.fail)
    output = io.StringIO()

    assert command.main(["--cache-root", str(tmp_path)], stdout=output) == 1
    assert "incomplete" in output.getvalue()


def test_route_preview_apply_and_repeat_apply_use_generated_keys(
    c03_seeded_artifact: SeededArchiveArtifact, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    current = copy_seeded_archive(c03_seeded_artifact, cache_root=cache_root)
    stale = build_seeded_archive(_stale_specs(999), cache_root=cache_root)
    _age_artifact(stale.root)
    preview_receipt = tmp_path / "preview.json"
    preview_output = io.StringIO()

    assert (
        command.main(
            [
                "--cache-root",
                str(cache_root),
                "--receipt",
                str(preview_receipt),
                "--grace-period-s",
                "1",
                "--json",
            ],
            stdout=preview_output,
        )
        == 0
    )
    preview = json.loads(preview_output.getvalue())
    assert preview["dry_run"] is True
    assert current.manifest.key in preview["reachable_keys"]
    assert preview["reachability"]["kinds"] == {"benchmark": 4, "default": 3, "named": 5}
    stale_preview = next(entry for entry in preview["entries"] if entry["name"] == stale.root.name)
    assert stale_preview["disposition"] == ArtifactGcDisposition.STALE.value
    assert stale.root.exists()
    assert json.loads(preview_receipt.read_text(encoding="utf-8"))["dry_run"] is True

    apply_receipt = tmp_path / "apply.json"
    applied_output = io.StringIO()
    assert (
        command.main(
            [
                "--cache-root",
                str(cache_root),
                "--receipt",
                str(apply_receipt),
                "--grace-period-s",
                "1",
                "--apply",
                "--json",
            ],
            stdout=applied_output,
        )
        == 0
    )
    applied = json.loads(applied_output.getvalue())
    assert applied["dry_run"] is False
    assert applied["deleted_bytes"] > 0
    assert not stale.root.exists()
    assert current.root.exists()
    assert json.loads(apply_receipt.read_text(encoding="utf-8"))["deleted_bytes"] == applied["deleted_bytes"]

    repeated_output = io.StringIO()
    assert (
        command.main(
            ["--cache-root", str(cache_root), "--grace-period-s", "1", "--apply", "--json"],
            stdout=repeated_output,
        )
        == 0
    )
    repeated = json.loads(repeated_output.getvalue())
    assert repeated["deleted_bytes"] == 0
    assert repeated["dispositions"].get(ArtifactGcDisposition.DELETED.value, 0) == 0
    assert current.root.exists()


def test_declared_agentctl_operation_is_bounded_and_previewable() -> None:
    import tomllib

    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["seeded_archive_cache_gc"]

    assert operation["exec"] == ["devtools", "cache", "gc", "--json"]
    assert operation["result"] == "json"
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 0


def test_gc_rejects_non_finite_grace_period(tmp_path: Path) -> None:
    """A NaN grace period must never bypass the age gate.

    Anti-vacuity: removing the ``math.isfinite`` check (leaving only
    ``grace_period_s < 0``) makes this pass ``nan`` straight through, since
    every comparison against NaN is False, and this test would then fail.
    """
    cache_root = tmp_path / "cache"

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            gc_seeded_archive_artifacts(
                cache_root=cache_root,
                reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
                grace_period_s=bad,
                dry_run=False,
            )


def test_route_refuses_receipt_beneath_artifact_candidates(tmp_path: Path) -> None:
    """A GC receipt cannot recreate debris in the namespace GC scans.

    Anti-vacuity: removing the containment refusal lets the call proceed and
    creates ``artifacts/gc.json``, which the next pass classifies as corrupt.
    """
    cache_root = tmp_path / "cache"
    artifact_root = cache_root / "artifacts"
    artifact_root.mkdir(parents=True)
    receipt = artifact_root / "gc.json"
    output = io.StringIO()
    assert command.main(["--cache-root", str(cache_root), "--receipt", str(receipt), "--json"], stdout=output) == 1
    assert "outside the managed artifacts directory" in output.getvalue()
    assert not receipt.exists()


def test_route_returns_nonzero_when_deletion_fails(
    c03_seeded_artifact: SeededArchiveArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deletion failure must fail the operation, not silently succeed.

    Anti-vacuity: reverting the CLI's ``return`` to unconditional ``0`` makes
    this fail, since dispositions still contain ``deletion-failed`` but the
    command would report success.
    """
    cache_root = tmp_path / "cache"
    current = copy_seeded_archive(c03_seeded_artifact, cache_root=cache_root)
    stale = build_seeded_archive(_stale_specs(999), cache_root=cache_root)
    _age_artifact(stale.root)

    def _boom(path: Path, **kwargs: object) -> None:
        raise OSError("simulated deletion failure")

    monkeypatch.setattr(workload_artifacts, "_remove_tree", _boom)
    output = io.StringIO()

    exit_code = command.main(
        ["--cache-root", str(cache_root), "--grace-period-s", "1", "--apply", "--json"],
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert payload["dispositions"].get(ArtifactGcDisposition.DELETION_FAILED.value) == 1
    assert exit_code == 1
    assert not stale.root.exists()
    assert any((cache_root / ".staging").glob(f"{stale.root.name}.gc.*"))
    assert current.root.exists()


def test_route_emits_json_error_payload_for_refusals_in_json_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--json`` refusals must stay parsable JSON, not a plain-text line.

    Anti-vacuity: reverting to the unconditional ``print(f"refused: {exc}")``
    makes ``json.loads`` on the output raise, and this test would fail.
    """

    def _explode() -> SeededArchiveReachabilityInventory:
        raise RuntimeError("simulated inventory failure")

    monkeypatch.setattr(command, "current_seeded_archive_reachability", _explode)
    output = io.StringIO()

    exit_code = command.main(["--cache-root", str(tmp_path), "--json"], stdout=output)

    assert exit_code == 1
    payload = json.loads(output.getvalue())
    assert "simulated inventory failure" in payload["refused"]


def test_json_refusal_payload_bounds_untrusted_error_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: oversized exception text cannot make an unbounded receipt."""

    def _explode() -> SeededArchiveReachabilityInventory:
        raise RuntimeError("x" * 5_000)

    monkeypatch.setattr(command, "current_seeded_archive_reachability", _explode)
    output = io.StringIO()
    assert command.main(["--cache-root", str(tmp_path), "--json"], stdout=output) == 1
    payload = json.loads(output.getvalue())
    assert len(payload["refused"]) == 2_048


def test_gc_sigterm_after_partial_deletion_is_receipted_and_resumable(
    c03_seeded_artifact: SeededArchiveArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production command must retire before unlinking and preserve a resumable interruption."""
    import signal

    cache_root = tmp_path / "cache"
    current = copy_seeded_archive(c03_seeded_artifact, cache_root=cache_root)
    stale = build_seeded_archive(_stale_specs(991), cache_root=cache_root)
    _age_artifact(stale.root)
    receipt = tmp_path / "gc.json"
    original = workload_artifacts._remove_tree
    previous_handler = signal.getsignal(signal.SIGTERM)
    sent = False

    def interrupt(path: Path, *, budget: int | None, progress: object) -> None:
        from collections.abc import Callable
        from typing import cast

        assert path.parent == cache_root / ".staging"
        assert not stale.root.exists()
        before = tuple(path.iterdir())

        def tick() -> None:
            nonlocal sent
            if not sent and any(not item.exists() for item in before):
                sent = True
                signal.raise_signal(signal.SIGTERM)
            cast(Callable[[], None], progress)()

        original(path, budget=budget, progress=tick)

    monkeypatch.setattr(workload_artifacts, "_remove_tree", interrupt)
    output = io.StringIO()
    assert (
        command.main(
            ["--cache-root", str(cache_root), "--receipt", str(receipt), "--grace-period-s", "1", "--apply", "--json"],
            stdout=output,
        )
        == 143
    )
    interrupted = json.loads(output.getvalue())
    assert sent and interrupted["interrupted"] and not interrupted["complete"]
    assert interrupted["deletion_bytes_complete"] is False
    assert interrupted["next_cursor"] is None
    checkpoint = json.loads(receipt.read_text())
    assert checkpoint["interrupted"] and not checkpoint["complete"]
    assert checkpoint["entries"] == interrupted["entries"]
    assert any(entry["disposition"] == "retired" for entry in interrupted["entries"])
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert current.root.exists()
    monkeypatch.setattr(workload_artifacts, "_remove_tree", original)
    resumed_output = io.StringIO()
    assert (
        command.main(
            ["--cache-root", str(cache_root), "--receipt", str(receipt), "--grace-period-s", "1", "--apply", "--json"],
            stdout=resumed_output,
        )
        == 0
    )
    resumed = json.loads(resumed_output.getvalue())
    assert resumed["complete"] and not resumed["interrupted"]
    assert resumed["dispositions"].get("corrupt", 0) == 0
    assert any(entry["disposition"] == "deleted" for entry in resumed["entries"])
    assert not list((cache_root / ".staging").glob("*.gc.*"))
    assert current.root.exists()


def test_gc_pages_do_not_skip_candidates_when_prior_artifacts_disappear(tmp_path: Path) -> None:
    """Lexical continuation follows deletion without the shrinking-offset defect."""
    cache_root = tmp_path / "cache"
    artifacts = [
        build_seeded_archive(
            (
                dataclasses.replace(
                    c03_semantic_corpus_spec(),
                    seed=seed,
                    count=2,
                    session_native_ids=("c03-target", "c03-irrelevant-000"),
                ),
            ),
            cache_root=cache_root,
        )
        for seed in (991, 992, 993)
    ]
    for artifact in artifacts:
        _age_artifact(artifact.root)
    after = None
    seen = set()
    complete = False
    while not complete:
        report = gc_seeded_archive_artifacts(
            cache_root=cache_root,
            reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
            grace_period_s=1,
            dry_run=False,
            page_size=1,
            after=after,
        )
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.disposition is ArtifactGcDisposition.DELETED
        assert entry.key not in seen
        seen.add(entry.key)
        complete = report.complete
        after = report.next_cursor
        assert complete == (after is None)
    assert seen == {artifact.manifest.key for artifact in artifacts}


def test_gc_retired_deletion_has_no_valid_tree_node_cap(tmp_path: Path) -> None:
    """A retired tree exceeding the predecessor 10,000-node cap still completes by streaming."""
    cache = tmp_path / "cache"
    (cache / "artifacts").mkdir(parents=True)
    (cache / ".locks").mkdir()
    retired = cache / ".staging" / ("a" * 64 + ".gc." + "b" * 32)
    retired.mkdir(parents=True)
    for index in range(10_001):
        (retired / str(index)).touch()
    report = gc_seeded_archive_artifacts(
        cache_root=cache, reachable_keys=("seeded-archive:sha256:" + "f" * 64,), dry_run=False
    )
    assert report.complete and not report.interrupted
    assert [entry.disposition for entry in report.entries] == [ArtifactGcDisposition.DELETED]
    assert not retired.exists()


def test_refused_retirement_preserves_published_tree_and_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed directory move must not leave an otherwise valid shared artifact writable."""
    import stat

    cache = tmp_path / "cache"
    spec = dataclasses.replace(
        c03_semantic_corpus_spec(), seed=996, count=2, session_native_ids=("c03-target", "c03-irrelevant-000")
    )
    artifact = build_seeded_archive((spec,), cache_root=cache)
    _age_artifact(artifact.root)
    original_mode = artifact.root.stat().st_mode

    def refuse(_source: Path, _destination: Path) -> None:
        raise PermissionError("injected retirement refusal")

    monkeypatch.setattr(workload_artifacts, "_safe_replace", refuse)
    report = gc_seeded_archive_artifacts(
        cache_root=cache, reachable_keys=("seeded-archive:sha256:" + "f" * 64,), grace_period_s=1, dry_run=False
    )
    assert [entry.disposition for entry in report.entries] == [ArtifactGcDisposition.DELETION_FAILED]
    assert not report.complete and report.next_cursor is None
    assert artifact.root.exists()
    assert artifact.root.stat().st_mode == original_mode
    assert not original_mode & stat.S_IWUSR
    assert workload_artifacts._gc_manifest_integrity(artifact.root)[2] is None
