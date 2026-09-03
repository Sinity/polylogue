from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path

import pytest

from devtools import seeded_archive_cache_gc as command
from tests.infra import workload_artifacts
from tests.infra.workload_artifacts import (
    ArtifactGcDisposition,
    SeededArchiveReachabilityInventory,
    build_seeded_archive,
    c03_semantic_corpus_spec,
    current_seeded_archive_reachability,
    gc_seeded_archive_artifacts,
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


def test_route_preview_apply_and_repeat_apply_use_generated_keys(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    current = build_seeded_archive(cache_root=cache_root)
    stale_specs = (dataclasses.replace(c03_semantic_corpus_spec(), seed=999),)
    stale = build_seeded_archive(stale_specs, cache_root=cache_root)
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
    assert operation["timeout_seconds"] == 900


def test_gc_rejects_non_finite_grace_period(tmp_path: Path) -> None:
    """A NaN grace period must never bypass the age gate.

    Anti-vacuity: removing the ``math.isfinite`` check (leaving only
    ``grace_period_s < 0``) makes this pass ``nan`` straight through, since
    every comparison against NaN is False, and this test would then fail.
    """
    cache_root = tmp_path / "cache"
    current = build_seeded_archive(cache_root=cache_root)

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            gc_seeded_archive_artifacts(
                cache_root=cache_root,
                reachable_keys=(current.manifest.key,),
                grace_period_s=bad,
                dry_run=False,
            )


def test_route_returns_nonzero_when_deletion_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A deletion failure must fail the operation, not silently succeed.

    Anti-vacuity: reverting the CLI's ``return`` to unconditional ``0`` makes
    this fail, since dispositions still contain ``deletion-failed`` but the
    command would report success.
    """
    cache_root = tmp_path / "cache"
    current = build_seeded_archive(cache_root=cache_root)
    stale_specs = (dataclasses.replace(c03_semantic_corpus_spec(), seed=999),)
    stale = build_seeded_archive(stale_specs, cache_root=cache_root)
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
    assert stale.root.exists()
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
