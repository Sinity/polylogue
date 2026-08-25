from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path

import pytest

from devtools import seeded_archive_cache_gc as command
from tests.infra.workload_artifacts import (
    ArtifactGcDisposition,
    SeededArchiveReachabilityInventory,
    build_seeded_archive,
    c03_semantic_corpus_spec,
    current_seeded_archive_reachability,
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
    assert preview["reachability"]["kinds"] == {"benchmark": 4, "default": 2, "named": 5}
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

    assert operation["exec"] == ["devtools", "workspace", "seeded-archive-cache-gc", "--json"]
    assert operation["scratch"] == "nvme"
    assert operation["exclusive_keys"] == ["polylogue:seeded-archive-cache-gc"]
    assert operation["timeout_seconds"] == 900
    assert operation["parameters"]["apply"]["flag"] == "--apply"
