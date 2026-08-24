"""Tests for the one shared real-pipeline seeded archive adapter."""

from __future__ import annotations

import fcntl
import gc
import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore, ReadOnlyArchiveError
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import DurableChangeTrainError
from tests.infra.workload_artifacts import (
    BENCHMARK_WORKLOAD_PROFILES,
    NAMED_WORKLOAD_PROFILES,
    ArtifactGcDisposition,
    ArtifactGcReport,
    BenchmarkWorkloadTier,
    SeededArchiveClone,
    WorkloadProfile,
    _assert_lock_identity,
    _journal_mode_delete_with_retry,
    _open_no_follow,
    _recover_obsolete_staging,
    _recover_stale_handoffs,
    _recover_stale_staging,
    acquire_query_only_seeded_archive,
    benchmark_corpus_specs,
    benchmark_workload_profile,
    benchmark_workload_tier,
    build_seeded_archive,
    c03_semantic_corpus_spec,
    clone_seeded_archive,
    gc_seeded_archive_artifacts,
    named_corpus_specs,
    named_workload_profile,
    seeded_archive_key,
)

pytest_plugins = ("tests.infra.corpus_fixtures",)


def test_benchmark_profiles_are_semantic_mixed_origin_exact_message_projections() -> None:
    """Benchmark targets are named workload contracts, not direct index seeds.

    Replacing the catalog with one provider or a round-count-only map makes the
    provider and exact-message assertions fail before any benchmark runs.
    """
    assert tuple(profile.tier for profile in BENCHMARK_WORKLOAD_PROFILES) == (
        BenchmarkWorkloadTier.SMOKE,
        BenchmarkWorkloadTier.REPRESENTATIVE,
        BenchmarkWorkloadTier.ARCHIVE_SCALE,
        BenchmarkWorkloadTier.STRESS,
    )
    assert tuple(profile.target_messages for profile in BENCHMARK_WORKLOAD_PROFILES) == (1_000, 5_000, 10_000, 50_000)
    for profile in BENCHMARK_WORKLOAD_PROFILES:
        specs = benchmark_corpus_specs(profile.tier)
        assert sum(spec.count * spec.messages_min for spec in specs) == profile.target_messages
        assert {spec.provider for spec in specs} == {"chatgpt", "claude-ai", "claude-code", "codex", "gemini"}
        assert {spec.messages_min for spec in specs} == {2, 8, profile.messages_per_session, 100}
        assert {spec.messages_max for spec in specs} == {2, 8, profile.messages_per_session, 100}
        assert {spec.profile.primary_family_id for spec in specs} == {"benchmark-archive"}
        assert {profile.tier.value for spec in specs if profile.tier.value in spec.tags} == {profile.tier.value}


def test_benchmark_profile_selection_is_deterministic_and_rejects_legacy_ad_hoc_targets() -> None:
    """The adapter has one stable corpus identity per supported semantic tier."""
    first = benchmark_corpus_specs(BenchmarkWorkloadTier.REPRESENTATIVE, seed=91)
    second = benchmark_corpus_specs("representative", seed=91)

    assert first == second
    assert benchmark_workload_tier(5_000) is BenchmarkWorkloadTier.REPRESENTATIVE
    assert benchmark_workload_profile("representative").target_messages == 5_000
    with pytest.raises(ValueError, match="no named benchmark workload"):
        benchmark_workload_tier(7_500)


def test_named_workload_profiles_are_semantic_and_build_deterministic_provider_specs() -> None:
    """Fixture workloads retain purpose and provider-native artifact identity."""
    assert {profile.name for profile in NAMED_WORKLOAD_PROFILES} == {
        "schema-small",
        "schema-medium",
        "cli-chatgpt",
        "cli-mixed",
        "completion",
    }

    profile = named_workload_profile("completion")
    first = named_corpus_specs(profile.name)
    second = profile.corpus_specs()

    assert first == second
    assert profile.purpose == "completion"
    assert {spec.provider for spec in first} == {"chatgpt", "claude-ai"}
    assert {spec.seed for spec in first} == {1271}
    assert {spec.origin for spec in first} == {"generated.test-workload-completion"}
    assert {spec.profile.primary_family_id for spec in first} == {"test-workload"}
    assert {"completion", "provider-native"}.issubset(set(first[0].profile.profile_tokens))
    with pytest.raises(ValueError, match="unknown named seeded archive workload"):
        named_workload_profile("unknown")


def test_named_and_benchmark_catalogs_share_one_semantic_spec_contract() -> None:
    """Both catalog adapters retain a common identity and native-spec constructor."""
    catalog_profiles = (*NAMED_WORKLOAD_PROFILES, *BENCHMARK_WORKLOAD_PROFILES)

    assert all(isinstance(profile.workload, WorkloadProfile) for profile in catalog_profiles)
    assert all(profile.workload.name and profile.workload.purpose for profile in catalog_profiles)
    assert all("provider-native" in profile.workload.profile_tokens for profile in catalog_profiles)

    named = named_workload_profile("cli-mixed")
    benchmark = benchmark_workload_profile(BenchmarkWorkloadTier.SMOKE)
    assert {spec.origin for spec in named.corpus_specs()} == {named.workload.origin}
    assert {spec.origin for spec in benchmark_corpus_specs(benchmark.tier)} == {benchmark.workload.origin}


def test_seeded_archive_publishes_valid_immutable_real_pipeline_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.pipeline.services.archive_ingest import (
        parse_sources_archive as real_parse_sources_archive,
    )

    observed_parse_workers: list[int | None] = []

    async def record_parse_workers(*args: Any, **kwargs: Any) -> Any:
        observed_parse_workers.append(kwargs.get("parse_workers"))
        return await real_parse_sources_archive(*args, **kwargs)

    monkeypatch.setattr("tests.infra.workload_artifacts.parse_sources_archive", record_parse_workers)
    cache_root = tmp_path / "cache"

    first = build_seeded_archive(cache_root=cache_root)
    second = build_seeded_archive(cache_root=cache_root)

    assert observed_parse_workers == [1]

    assert first.root == second.root
    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.manifest.receipt["status"] == "succeeded"
    assert len(first.facts) == 64
    assert first.facts[0].expected_session_id == "codex-session:c03-target"
    assert first.root.joinpath("index.db").exists()
    assert raw_materialization_ready(raw_materialization_readiness_snapshot(first.root))
    phases = first.manifest.receipt["phases"]
    assert isinstance(phases, list)
    assert any(isinstance(phase, dict) and phase.get("name") == "raw_authority_frontier" for phase in phases)
    assert not (first.root.stat().st_mode & stat.S_IWUSR)
    lock_path = first.root / ".index-rebuild.lock"
    if lock_path.exists():
        first.root.chmod(first.root.stat().st_mode | stat.S_IWUSR)
        lock_path.unlink()
        first.root.chmod(first.root.stat().st_mode & ~stat.S_IWUSR)
    with pytest.raises(PermissionError):
        lock_path.touch()
    assert not lock_path.exists()


def test_seeded_archive_rejects_unsafe_or_unknown_provider_before_recipe_path_use() -> None:
    import dataclasses

    base = c03_semantic_corpus_spec()
    for provider in ("../outside", "chatgpt/escape", "unknown-provider"):
        forged = dataclasses.replace(base, provider=provider)
        with pytest.raises(ValueError, match="provider"):
            seeded_archive_key((forged,))


def test_seeded_archive_rejects_symlinked_cache_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        build_seeded_archive(cache_root=cache_root)
    assert not (outside / "artifacts").exists()


def test_seeded_archive_clone_rejects_symlink_inside_published_tree(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    artifact.root.chmod(artifact.root.stat().st_mode | stat.S_IWUSR)
    link = artifact.root / "hostile-link"
    link.symlink_to(tmp_path / "outside")
    artifacts._VALIDATED_ARTIFACTS.clear()

    with pytest.raises(ValueError, match="symlink"):
        clone_seeded_archive(artifact, tmp_path / "clone")


def test_lock_inode_replacement_is_detected_after_flock(tmp_path: Path) -> None:
    import fcntl

    lock = tmp_path / "lock"
    lock.write_text("owner", encoding="utf-8")
    fd = _open_no_follow(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        replacement = tmp_path / "replacement"
        replacement.write_text("attacker", encoding="utf-8")
        os.replace(replacement, lock)
        with pytest.raises(OSError, match="lock pathname was replaced"):
            _assert_lock_identity(fd, lock)
    finally:
        os.close(fd)


def test_seeded_archive_rejects_unsupported_cache_node_and_rebuilds(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    original = build_seeded_archive(cache_root=cache_root)
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    hostile = original.root / "u"
    os.mkfifo(hostile)
    artifacts._VALIDATED_ARTIFACTS.clear()
    rebuilt = build_seeded_archive(cache_root=cache_root)
    assert not (rebuilt.root / "u").exists()


def test_clone_cleans_partial_output_after_short_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    destination = tmp_path / "partial-clone"

    def fail_write(fd: int, data: bytes) -> None:
        raise OSError("injected short write")

    monkeypatch.setattr(artifacts, "_write_all", fail_write)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )
    with pytest.raises(OSError, match="short write"):
        clone_seeded_archive(artifact, destination)
    assert not destination.exists()


def test_clone_rejects_tampered_copy_and_cleans_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    destination = tmp_path / "tampered-clone"
    original_copy = artifacts._copy_tree

    def tamper(source: Path, target: Path) -> None:
        original_copy(source, target)
        tampered = target.joinpath("manifest.json")
        tampered.chmod(tampered.stat().st_mode | stat.S_IWUSR)
        original_bytes = tampered.read_bytes()
        tampered.write_bytes(bytes((original_bytes[0] ^ 1,)) + original_bytes[1:])

    monkeypatch.setattr(artifacts, "_copy_tree", tamper)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )
    with pytest.raises(ValueError, match="manifest bytes mismatch"):
        clone_seeded_archive(artifact, destination)
    assert not destination.exists()


def test_seeded_archive_key_changes_with_source_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(artifacts, "lowering_fingerprint", lambda: "emitter-semantics:first")
    first = seeded_archive_key(())
    monkeypatch.setattr(artifacts, "lowering_fingerprint", lambda: "emitter-semantics:second")
    second = seeded_archive_key(())

    assert first.value != second.value


def test_seeded_archive_clone_is_private_full_root_and_preserves_base(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    base_manifest = artifact.root.joinpath("manifest.json").read_bytes()
    marker_relative = Path(".maintenance-state/durable-change-trains/.bootstrap")
    base_marker = artifact.root.joinpath(marker_relative).read_bytes()

    clone = clone_seeded_archive(artifact, tmp_path / "clone")
    clone.root.joinpath("private-mutation.txt").write_text("private")
    with ArchiveStore.open_existing(clone.root, read_only=False) as archive:
        assert archive.count_sessions() == 64

    assert clone.clone_method in {"reflink", "copy"}
    assert clone.source_manifest_id == artifact.manifest.manifest_id
    assert clone.root.joinpath("source.db").exists()
    assert clone.root.joinpath("index.db").exists()
    assert artifact.root.joinpath("manifest.json").read_bytes() == base_manifest
    assert artifact.root.joinpath(marker_relative).read_bytes() == base_marker
    assert clone.root.joinpath(marker_relative).read_bytes() != base_marker
    assert not artifact.root.joinpath("private-mutation.txt").exists()

    clone.root.joinpath(marker_relative).write_bytes(base_marker)
    with pytest.raises(DurableChangeTrainError, match="durable identity mismatch"):
        ArchiveStore.open_existing(clone.root, read_only=False)


def test_seeded_archive_copy_fallback_rebinds_durable_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")

    def reject_reflink(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["cp"])

    monkeypatch.setattr(subprocess, "run", reject_reflink)
    clone = clone_seeded_archive(artifact, tmp_path / "clone")

    assert clone.clone_method == "copy"
    with ArchiveStore.open_existing(clone.root, read_only=False) as archive:
        assert archive.count_sessions() == 64
    clone.close()


def test_seeded_archive_clone_leaves_unrelated_siblings_live(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    parent = tmp_path / "consumer-work"
    sibling = parent / "unrelated-sibling"
    sibling.mkdir(parents=True)

    with clone_seeded_archive(artifact, parent / "seeded-archive-clone"):
        sibling.joinpath("still-live.txt").write_text("independent", encoding="utf-8")
        assert sibling.joinpath("still-live.txt").read_text(encoding="utf-8") == "independent"


def test_seeded_archive_clone_preserves_caller_directory_modes(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    parent = tmp_path / "consumer-work"
    parent.mkdir(mode=0o750)
    ancestor_mode = stat.S_IMODE(tmp_path.stat().st_mode)
    parent_mode = stat.S_IMODE(parent.stat().st_mode)

    with clone_seeded_archive(artifact, parent / "seeded-archive-clone"):
        assert stat.S_IMODE(tmp_path.stat().st_mode) == ancestor_mode
        assert stat.S_IMODE(parent.stat().st_mode) == parent_mode

    assert stat.S_IMODE(tmp_path.stat().st_mode) == ancestor_mode
    assert stat.S_IMODE(parent.stat().st_mode) == parent_mode


def test_seeded_archive_reflink_and_copy_clones_are_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    attempted: list[list[str]] = []
    real_run = subprocess.run

    def record_reflink_attempt(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        attempted.append(args)
        return cast(subprocess.CompletedProcess[bytes], real_run(args, **kwargs))

    monkeypatch.setattr(subprocess, "run", record_reflink_attempt)
    fast = clone_seeded_archive(artifact, tmp_path / "fast")

    def reject_reflink(*_args: object, **_kwargs: object) -> None:
        late = tmp_path / "fallback"
        late.mkdir(exist_ok=True)
        late.joinpath("late-copy-residue").write_text("remove me", encoding="utf-8")
        raise subprocess.CalledProcessError(1, ["cp"])

    monkeypatch.setattr(subprocess, "run", reject_reflink)
    fallback = clone_seeded_archive(artifact, tmp_path / "fallback")
    ignored = ".maintenance-state/durable-change-trains/.bootstrap"

    def files(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and str(path.relative_to(root)) != ignored
        }

    try:
        assert attempted == [["cp", "-a", "--reflink=always", str(artifact.root), str(fast.root)]]
        assert fast.clone_method in {"reflink", "copy"}
        assert fallback.clone_method == "copy"
        assert not fallback.root.joinpath("late-copy-residue").exists()
        assert files(fast.root) == files(fallback.root)
    finally:
        fast.close()
        fallback.close()


def test_seeded_archive_rejects_corrupt_published_cache_and_rebuilds(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    index_path.chmod(index_path.stat().st_mode | stat.S_IWUSR)
    index_path.unlink()

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root == original.root
    assert rebuilt.root.joinpath("index.db").is_file()
    assert rebuilt.manifest.key == original.manifest.key
    assert rebuilt.manifest.profile_id == original.manifest.profile_id
    assert rebuilt.manifest.recipe_id == original.manifest.recipe_id
    assert rebuilt.facts == original.facts
    assert not (rebuilt.root.stat().st_mode & stat.S_IWUSR)


def test_seeded_archive_rejects_unexpected_published_files(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    artifact = build_seeded_archive(cache_root=cache_root)
    artifact.root.chmod(artifact.root.stat().st_mode | stat.S_IWUSR)
    extra = artifact.root / "unexpected.txt"
    extra.write_text("contamination", encoding="utf-8")

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert not rebuilt.root.joinpath("unexpected.txt").exists()
    assert not (rebuilt.root.stat().st_mode & stat.S_IWUSR)


def test_seeded_archive_memo_rejects_same_size_database_corruption(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    original_bytes = index_path.read_bytes()
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    index_path.chmod(index_path.stat().st_mode | stat.S_IWUSR)
    poison_offset = len(original_bytes) // 2
    poisoned_bytes = original_bytes[:poison_offset] + b"poison" + original_bytes[poison_offset + 6 :]
    with index_path.open("r+b") as handle:
        handle.seek(poison_offset)
        handle.write(b"poison")
    index_path.chmod(index_path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    original.root.chmod(original.root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root.joinpath("index.db").read_bytes() != poisoned_bytes


def test_seeded_archive_rejects_forged_receipt_with_recomputed_manifest(
    tmp_path: Path,
) -> None:
    import dataclasses

    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    receipt = dict(original.manifest.receipt)
    receipt["evidence_refs"] = ["forged-evidence"]
    forged = dataclasses.replace(original.manifest, receipt=receipt)
    manifest_path = original.root / "manifest.json"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    manifest_path.chmod(manifest_path.stat().st_mode | stat.S_IWUSR)
    manifest_path.write_text(json.dumps(forged.to_payload(), sort_keys=True) + "\\n", encoding="utf-8")
    manifest_path.chmod(manifest_path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    original.root.chmod(original.root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.manifest.receipt["evidence_refs"] == []


def test_seeded_archive_rebuilds_malformed_build_provenance_without_raising(tmp_path: Path) -> None:
    import dataclasses

    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    forged = dataclasses.replace(original.manifest, build_id="git:not-a-commit")
    manifest_path = original.root / "manifest.json"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    manifest_path.chmod(manifest_path.stat().st_mode | stat.S_IWUSR)
    manifest_path.write_text(json.dumps(forged.to_payload(), sort_keys=True) + "\\n", encoding="utf-8")
    manifest_path.chmod(manifest_path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    original.root.chmod(original.root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.manifest.build_id != "git:not-a-commit"
    assert rebuilt.manifest.manifest_id == artifacts._read_manifest(rebuilt.root / "manifest.json").manifest_id


def test_cleanup_fails_closed_on_replaced_lock_path(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    staging_root = cache_root / ".staging"
    staging_root.mkdir(parents=True)
    (staging_root / "old-key.dead").mkdir()
    (cache_root / ".locks").mkdir()
    (cache_root / ".cleanup.lock").symlink_to(tmp_path / "active.lock")

    assert artifacts._recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root) == ()
    assert (staging_root / "old-key.dead").exists()
    assert (cache_root / ".cleanup.lock").is_symlink()


def test_build_aborts_when_key_lock_path_is_replaced_during_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    original = artifacts._recover_stale_staging

    def replace_lock(*, staging_root: Path, artifact_name: str) -> tuple[str, ...]:
        result = original(staging_root=staging_root, artifact_name=artifact_name)
        lock = next((cache_root / ".locks").glob("*.lock"))
        replacement = cache_root / "replacement.lock"
        replacement.write_text("replacement", encoding="utf-8")
        os.replace(replacement, lock)
        return result

    monkeypatch.setattr(artifacts, "_recover_stale_staging", replace_lock)
    with pytest.raises(OSError, match="lock pathname was replaced|Permission denied"):
        build_seeded_archive(cache_root=cache_root)


def test_cleanup_removes_symlink_nodes_without_following_targets(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    staging_root = cache_root / ".staging"
    artifacts_root = cache_root / "artifacts"
    staging_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    (cache_root / ".locks").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep", encoding="utf-8")
    staging_link = staging_root / "key.dead"
    handoff_link = artifacts_root / ".key.handoff"
    staging_link.symlink_to(outside, target_is_directory=True)
    handoff_link.symlink_to(outside, target_is_directory=True)

    assert _recover_stale_staging(staging_root=staging_root, artifact_name="key") == ("key.dead",)
    assert _recover_stale_handoffs(cache_root=cache_root, artifacts_root=artifacts_root) == (".key.handoff",)
    assert not staging_link.exists()
    assert not handoff_link.exists()
    assert (outside / "keep").read_text(encoding="utf-8") == "keep"


def test_obsolete_cleanup_advances_cursor_past_irrelevant_entries(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    staging_root = cache_root / ".staging"
    (staging_root / "irrelevant").mkdir(parents=True)
    (staging_root / "also-irrelevant").write_text("x", encoding="utf-8")
    (cache_root / ".locks").mkdir()

    assert _recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root, budget=2) == ()
    cursor = (cache_root / ".cleanup.cursor").read_text(encoding="utf-8").strip()
    assert cursor in {"irrelevant", "also-irrelevant"}


def test_clone_rejects_hardlinked_leaf_and_cleans_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    destination = tmp_path / "hardlinked-clone"
    original_copy = artifacts._copy_tree

    def hardlink(source: Path, target: Path) -> None:
        original_copy(source, target)
        leaf = target / "source.db"
        leaf.unlink()
        os.link(source / "source.db", leaf)

    monkeypatch.setattr(artifacts, "_copy_tree", hardlink)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )
    with pytest.raises(ValueError, match="inode"):
        clone_seeded_archive(artifact, destination)
    assert not destination.exists()


def test_seeded_archive_rejects_malformed_self_hashed_file_entry(
    tmp_path: Path,
) -> None:
    import dataclasses

    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    malformed_files = [dict(item) for item in original.manifest.files]
    malformed_files[0].pop("path")
    forged = dataclasses.replace(original.manifest, files=tuple(malformed_files))
    manifest_path = original.root / "manifest.json"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    manifest_path.chmod(manifest_path.stat().st_mode | stat.S_IWUSR)
    manifest_path.write_text(json.dumps(forged.to_payload(), sort_keys=True) + "\\n", encoding="utf-8")
    manifest_path.chmod(manifest_path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    original.root.chmod(original.root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.manifest.files[0]["path"] == original.manifest.files[0]["path"]


def test_seeded_archive_memo_rejects_manifest_replacement(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    manifest_path = original.root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["manifest_id"] = "poisoned-manifest"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    manifest_path.chmod(manifest_path.stat().st_mode | stat.S_IWUSR)
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
    manifest_path.chmod(manifest_path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    original.root.chmod(original.root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    rebuilt = build_seeded_archive(cache_root=cache_root)

    disk_manifest = json.loads(rebuilt.root.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["manifest_id"] == rebuilt.manifest.manifest_id


def test_seeded_archive_memo_rejects_nested_write_bits(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    nested = original.root / "wire"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    nested.chmod(nested.stat().st_mode | stat.S_IWGRP)
    index_path = original.root / "index.db"
    index_path.chmod(index_path.stat().st_mode | stat.S_IWOTH)
    original.root.chmod(original.root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert all(
        not (path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) for path in rebuilt.root.rglob("*")
    )


def test_publish_attempts_rename_with_a_sealed_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    staging = tmp_path / "staging"
    final_root = tmp_path / "final"
    staging.mkdir()
    staging.joinpath("payload").write_text("payload", encoding="utf-8")
    observed_modes: list[int] = []

    def reject_rename(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: object,
        **kwargs: object,
    ) -> None:
        observed_modes.append(
            os.stat(source, dir_fd=cast(int, kwargs.get("src_dir_fd")), follow_symlinks=False).st_mode
        )
        raise PermissionError("injected sealed rename failure")

    monkeypatch.setattr(os, "replace", reject_rename)
    with pytest.raises(PermissionError, match="injected sealed rename failure"):
        artifacts._publish_sealed_staging(staging, final_root)

    assert observed_modes
    assert not (observed_modes[0] & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def test_sealed_fallback_publishes_only_sealed_final_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    staging = tmp_path / "staging"
    final_root = tmp_path / "final"
    staging.mkdir()
    staging.joinpath("payload").write_text("payload", encoding="utf-8")
    real_replace = os.replace
    calls = 0
    observed: list[tuple[Path, bool]] = []

    def fail_once_then_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        **kwargs: object,
    ) -> None:
        nonlocal calls
        calls += 1
        source_path = Path(os.readlink(f"/proc/self/fd/{kwargs['src_dir_fd']}")) / os.fsdecode(source)
        destination_path = Path(os.readlink(f"/proc/self/fd/{kwargs['dst_dir_fd']}")) / os.fsdecode(destination)
        observed.append((source_path, bool(source_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))))
        if calls == 1:
            raise PermissionError("injected first rename failure")
        assert not destination_path.exists()
        real_replace(
            source,
            destination,
            src_dir_fd=cast(int, kwargs.get("src_dir_fd")),
            dst_dir_fd=cast(int, kwargs.get("dst_dir_fd")),
        )

    monkeypatch.setattr(os, "replace", fail_once_then_replace)
    artifacts._publish_sealed_staging(staging, final_root)

    assert calls == 2
    assert observed[0] == (staging, False)
    assert observed[1][0].parent == final_root.parent
    assert not observed[1][1]
    assert final_root.exists()
    assert not staging.exists()
    assert not (final_root.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def test_sealed_fallback_kill_injection_leaves_no_visible_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    staging = tmp_path / "staging"
    final_root = tmp_path / "final"
    staging.mkdir()
    staging.joinpath("payload").write_text("payload", encoding="utf-8")
    calls = 0

    def fail_then_interrupt(source: object, destination: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("injected first rename failure")
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", fail_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        artifacts._publish_sealed_staging(staging, final_root)

    assert not final_root.exists()
    handoffs = tuple(final_root.parent.glob(f".{final_root.name}.*.handoff"))
    assert len(handoffs) == 1
    assert not (handoffs[0].stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    (tmp_path / ".locks").mkdir()
    removed = artifacts._recover_stale_handoffs(cache_root=tmp_path, artifacts_root=tmp_path)
    assert removed == (handoffs[0].name,)
    assert not handoffs[0].exists()


def test_archive_and_clone_reject_symlink_nodes_without_following_targets(
    tmp_path: Path,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "outside"
    target.write_text("do not touch", encoding="utf-8")
    root.joinpath("link").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        artifacts._archive_files(root)
    with pytest.raises(ValueError, match="symlink"):
        artifacts._make_read_only(root)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.clone_seeded_archive(type("Artifact", (), {"root": root})(), tmp_path / "clone")
    assert target.read_text(encoding="utf-8") == "do not touch"


def test_seeded_archive_sealing_failure_never_publishes_writable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    def fail_sealing(_root: Path) -> None:
        raise RuntimeError("injected sealing failure")

    monkeypatch.setattr(artifacts, "_make_read_only", fail_sealing)
    cache_root = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="injected sealing failure"):
        build_seeded_archive(cache_root=cache_root)

    assert not list((cache_root / "artifacts").iterdir())
    assert not list((cache_root / ".staging").iterdir())


def test_seeded_archive_failure_never_publishes_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    async def fail_parse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected ingest failure")

    monkeypatch.setattr(artifacts, "parse_sources_archive", fail_parse)

    with pytest.raises(RuntimeError, match="injected ingest failure"):
        build_seeded_archive(cache_root=tmp_path / "cache")

    cache_root = tmp_path / "cache"
    assert not list((cache_root / "artifacts").iterdir())
    assert not list((cache_root / ".staging").iterdir())


def test_seeded_archive_recovers_crash_left_staging_before_rebuild(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    cache_root.joinpath("artifacts").mkdir(parents=True)
    cache_root.joinpath(".locks").mkdir()
    staging_root = cache_root / ".staging"
    staging_root.mkdir()
    stale = staging_root / "dead-build.123"
    stale.mkdir()
    stale.joinpath("index.db").write_bytes(b"partial sqlite")
    stale.joinpath(".build.done").write_text("written before the crash", encoding="utf-8")

    removed = artifacts._recover_stale_staging(staging_root=staging_root, artifact_name="dead-build")

    assert removed == ("dead-build.123",)
    assert not stale.exists()


def test_obsolete_staging_sweep_honors_budget_and_continues(
    tmp_path: Path,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    staging_root = cache_root / ".staging"
    lock_root = cache_root / ".locks"
    staging_root.mkdir(parents=True)
    lock_root.mkdir()
    for index in range(5):
        (staging_root / f"key-{index}.build").mkdir()

    removed: list[str] = []
    for _ in range(12):
        batch = artifacts._recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root, budget=2)
        removed.extend(batch)
        if not tuple(staging_root.iterdir()):
            break

    assert len(removed) == 5
    assert not tuple(staging_root.iterdir())


def test_obsolete_staging_sweep_does_not_remove_an_active_key(
    tmp_path: Path,
) -> None:
    import fcntl

    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    staging_root = cache_root / ".staging"
    lock_root = cache_root / ".locks"
    staging_root.mkdir(parents=True)
    lock_root.mkdir()
    candidate = staging_root / "active-key.123"
    candidate.mkdir()
    lock_path = lock_root / "active-key.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        removed = artifacts._recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root)
        assert removed == ()
        assert candidate.exists()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    removed = artifacts._recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root)
    assert removed == ("active-key.123",)
    assert not candidate.exists()


class _FlakyLockConnection:
    """Fakes ``PRAGMA journal_mode=DELETE`` raising a transient same-process lock.

    Mirrors CPython's ``sqlite3_close_v2`` zombie-connection footgun
    (polylogue-lbgc): a not-yet-finalized cursor/connection from earlier in
    the same worker process keeps SQLite's per-process shared pager-cache
    entry alive, so ``PRAGMA journal_mode=DELETE`` on a legitimate connection
    raises ``sqlite3.OperationalError: database is locked`` until that zombie
    is garbage-collected.
    """

    def __init__(self, *, locked_attempts: int) -> None:
        self.locked_attempts = locked_attempts
        self.attempts = 0

    def execute(self, sql: str) -> _FlakyLockConnection:
        assert sql == "PRAGMA journal_mode=DELETE"
        self.attempts += 1
        if self.attempts <= self.locked_attempts:
            raise sqlite3.OperationalError("database is locked")
        return self

    def fetchone(self) -> tuple[str]:
        return ("delete",)


def test_journal_mode_delete_retry_survives_a_transient_same_process_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: reverting to a bare ``conn.execute(...)`` call (no retry/gc.collect)
    makes this fail on the very first simulated lock, since there would be no
    mechanism left to absorb it."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    gc_collect_calls = 0

    def fake_collect() -> int:
        nonlocal gc_collect_calls
        gc_collect_calls += 1
        return 0

    monkeypatch.setattr(gc, "collect", fake_collect)

    conn = _FlakyLockConnection(locked_attempts=2)
    _journal_mode_delete_with_retry(conn, name="index.db")  # type: ignore[arg-type]

    assert conn.attempts == 3
    assert gc_collect_calls == 2


def test_journal_mode_delete_does_not_retry_non_lock_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-lock OperationalError (e.g. real corruption/IO error) must propagate
    immediately -- retrying it would hide a genuine failure, not a transient
    same-process race."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", sleep_calls.append)

    class _BrokenConnection:
        def execute(self, sql: str) -> Any:
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        _journal_mode_delete_with_retry(_BrokenConnection(), name="index.db")  # type: ignore[arg-type]

    assert sleep_calls == []


def test_journal_mode_delete_reraises_lock_once_deadline_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent (non-transient) same-process lock must still surface as a
    real failure rather than retry forever; the fake clock jumps straight past
    the deadline so the test doesn't need to sleep for real."""
    clock = iter([0.0, 10.0, 10.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    class _AlwaysLockedConnection:
        def execute(self, sql: str) -> Any:
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _journal_mode_delete_with_retry(_AlwaysLockedConnection(), name="index.db")  # type: ignore[arg-type]


def test_validate_artifact_preserves_transient_sqlite_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock means "try again", never "delete and republish".

    Anti-vacuity: folding ``sqlite3.OperationalError`` back into
    ``_validate_artifact``'s invalid-artifact catch-all makes this return
    ``None`` and routes a good shared artifact into the expensive rebuild path.
    """
    import tests.infra.workload_artifacts as artifacts

    key = seeded_archive_key(())
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    class _Manifest:
        protocol_version = artifacts._ARTIFACT_PROTOCOL_VERSION
        key: str
        files: tuple[object, ...] = ()
        facts: tuple[object, ...] = ()

    manifest = _Manifest()
    manifest.key = key.value
    monkeypatch.setattr(artifacts, "_read_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        artifacts,
        "_sqlite_integrity",
        lambda _root: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    with pytest.raises(artifacts._ArtifactValidationContentionError, match="contended"):
        artifacts._validate_artifact(tmp_path, key)


def test_artifact_validation_retries_transient_contention(monkeypatch: pytest.MonkeyPatch) -> None:
    """A brief validation lock settles before the builder considers republishing.

    Anti-vacuity: removing the retry helper or turning its contention exception
    into ``None`` leaves no path that retries validation before a cache miss can
    delete the published artifact.
    """
    import tests.infra.workload_artifacts as artifacts

    attempts = 0

    def flaky_validate(_root: Path, _key: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise artifacts._ArtifactValidationContentionError("locked")
        return None

    monkeypatch.setattr(artifacts, "_validate_artifact", flaky_validate)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    assert artifacts._validate_artifact_with_retry(Path("unused"), seeded_archive_key(())) is None
    assert attempts == 3


# ---------------------------------------------------------------------------
# Cache identity: what a published artifact is, and is not, a function of
# ---------------------------------------------------------------------------


_REUSE_PROBE = """
import json, os, sys
from pathlib import Path
import tests.infra.workload_artifacts as artifacts

artifacts._build_id = lambda: os.environ["FAKE_BUILD_ID"]
cache_root = Path(sys.argv[1])
artifact = artifacts.build_seeded_archive(cache_root=cache_root)
print(json.dumps({
    "key": artifact.manifest.key,
    "root": str(artifact.root),
    "published": len(list((cache_root / "artifacts").iterdir())),
}))
"""


def test_seeded_archive_key_does_not_carry_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new commit must not change the cache key.

    Anti-vacuity: restoring ``build_id`` (``git rev-parse HEAD``) to
    :class:`SeededArchiveKey` makes both assertions fail. That was the
    measured behavior before polylogue-1xc.14.1 -- 223 immutable artifact
    directories, 560 MB, for a catalog of about six distinct workloads,
    because each commit republished every workload it touched.
    """
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(artifacts, "_build_id", lambda: "git:" + "0" * 40)
    first = seeded_archive_key(())
    monkeypatch.setattr(artifacts, "_build_id", lambda: "git:" + "f" * 40)
    second = seeded_archive_key(())

    assert first.value == second.value
    assert not hasattr(first, "build_id")


@pytest.mark.uses_real_clock("spawns fresh interpreters; no timestamp assertions")
def test_seeded_archive_is_reused_by_a_later_commit(tmp_path: Path) -> None:
    """Two commits must share one published artifact.

    Runs each build in its own interpreter rather than twice in this process.
    That is both the case that matters -- a later commit's test run is always
    a new process -- and the only way to test reuse without the
    ``sqlite3_close_v2`` zombie-connection footgun documented on
    :func:`_journal_mode_delete_with_retry` (polylogue-lbgc): revalidating an
    artifact in the same process that just wrote it re-opens tiers whose
    connections may not be finalized yet, which under load raises
    ``database is locked``.

    Anti-vacuity: returning ``build_id`` to the key makes ``published`` come
    back as 2 with two different roots.
    """
    cache_root = tmp_path / "cache"
    repo_root = Path(__file__).resolve().parents[3]

    def build(fake_commit: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-c", _REUSE_PROBE, str(cache_root)],
            cwd=repo_root,
            env={**os.environ, "FAKE_BUILD_ID": f"git:{fake_commit}"},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr
        payload: dict[str, object] = json.loads(result.stdout.strip().splitlines()[-1])
        return payload

    first = build("0" * 40)
    second = build("f" * 40)

    assert first["key"] == second["key"]
    assert first["root"] == second["root"]
    assert second["published"] == 1
    # The commit survives as provenance on the manifest, where it records
    # which checkout published the bytes without gating their reuse.
    manifest = json.loads((Path(str(first["root"])) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_id"] == "git:" + "0" * 40


def test_seeded_archive_key_includes_artifact_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    baseline = seeded_archive_key(())
    monkeypatch.setattr(artifacts, "_ARTIFACT_PROTOCOL_VERSION", baseline.artifact_protocol_version + 1)

    changed = seeded_archive_key(())

    assert changed.value != baseline.value
    assert changed.artifact_protocol_version != baseline.artifact_protocol_version


def test_recipe_id_tracks_transitive_source_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    dependency_root = tmp_path / "dependencies"
    dependency_root.mkdir()
    nested = dependency_root / "nested_helper.py"
    nested.write_text("VALUE = 1\\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "_SOURCE_DEPENDENCY_ROOTS", (dependency_root,))
    first = artifacts._recipe_id()
    nested.write_text("VALUE = 2\\n", encoding="utf-8")

    assert artifacts._recipe_id() != first


def test_recipe_id_tracks_runtime_schema_inputs_but_ignores_unrelated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    inputs = tmp_path / "schema-inputs"
    inputs.mkdir()
    catalog = inputs / "catalog.json"
    unrelated = inputs / "README.txt"
    catalog.write_text('{"version": 1}\\n', encoding="utf-8")
    unrelated.write_text("unrelated\\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "_SOURCE_DEPENDENCY_ROOTS", ())
    monkeypatch.setattr(artifacts, "_RECIPE_INPUT_ROOTS", (inputs,))

    baseline = artifacts._recipe_id()
    unrelated.write_text("changed\\n", encoding="utf-8")
    assert artifacts._recipe_id() == baseline
    catalog.write_text('{"version": 2}\\n', encoding="utf-8")
    assert artifacts._recipe_id() != baseline


def test_recipe_id_only_tracks_selected_provider_catalogs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    providers = tmp_path / "providers"
    (providers / "codex").mkdir(parents=True)
    (providers / "chatgpt").mkdir()
    codex = providers / "codex" / "catalog.json"
    chatgpt = providers / "chatgpt" / "catalog.json"
    codex.write_text('{"version": 1}\\n', encoding="utf-8")
    chatgpt.write_text('{"version": 1}\\n', encoding="utf-8")
    monkeypatch.setattr(artifacts, "_SOURCE_DEPENDENCY_ROOTS", ())
    monkeypatch.setattr(artifacts, "_RECIPE_INPUT_ROOTS", ())
    monkeypatch.setattr(artifacts, "_RECIPE_PROVIDER_ROOT", providers)

    baseline = artifacts._recipe_id(("codex",))
    chatgpt.write_text('{"version": 2}\\n', encoding="utf-8")
    assert artifacts._recipe_id(("codex",)) == baseline
    codex.write_text('{"version": 2}\\n', encoding="utf-8")
    assert artifacts._recipe_id(("codex",)) != baseline


def test_seeded_archive_key_changes_with_archive_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Archive DDL is part of the artifact's identity.

    ``recipe_id`` hashes a fixed six-file list that names ``bootstrap.py`` but
    none of the ``archive_tiers`` DDL modules, so before this key component a
    schema change arriving through ``index.py`` (the normal route) left the
    key untouched and a stale-schema artifact reusable.
    """
    baseline = seeded_archive_key(())

    bumped = dict(ARCHIVE_DDL_BY_TIER)
    bumped[ArchiveTier.INDEX] = ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX] + "\nCREATE TABLE later_addition(id TEXT);"
    monkeypatch.setattr("tests.infra.workload_artifacts.ARCHIVE_DDL_BY_TIER", bumped)

    assert seeded_archive_key(()).value != baseline.value
    assert seeded_archive_key(()).archive_schema_id != baseline.archive_schema_id


def test_seeded_archive_key_ignores_ddl_reordering() -> None:
    """The schema component names the DDL, not the module text around it.

    Hashing the rendered per-tier DDL rather than the Python source of the
    modules that build it keeps a comment or docstring edit in ``index.py``
    from invalidating every cached artifact -- the same over-invalidation, one
    layer down, that dropping ``build_id`` exists to stop.
    """
    import tests.infra.workload_artifacts as artifacts

    assert artifacts._archive_schema_id() == artifacts._archive_schema_id()
    assert artifacts._archive_schema_id().startswith("archive-schema:sha256:")


# ---------------------------------------------------------------------------
# Validate-once-per-process memo
# ---------------------------------------------------------------------------


def test_seeded_archive_memo_skips_revalidation_within_a_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second hit in a process must not re-run the full validation.

    Anti-vacuity: deleting the memo lookup in ``build_seeded_archive`` makes
    this fail, because ``_validate_artifact`` is then called on every hit --
    which is exactly the per-cache-hit cost (re-SHA256 of every tier, five
    ``PRAGMA quick_check`` runs, the planted-facts query, and the
    frontier-convergence read) the memo exists to stop paying per test.
    """
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    first = build_seeded_archive(cache_root=cache_root)

    calls = 0
    real_validate = artifacts._validate_artifact

    def counting_validate(root: Path, key: artifacts.SeededArchiveKey) -> object:
        nonlocal calls
        calls += 1
        return real_validate(root, key)

    monkeypatch.setattr(artifacts, "_validate_artifact", counting_validate)
    second = build_seeded_archive(cache_root=cache_root)

    assert calls == 0
    assert second.root == first.root
    assert second.manifest.manifest_id == first.manifest.manifest_id


def test_seeded_archive_memo_is_dropped_when_the_artifact_is_unplaced(tmp_path: Path) -> None:
    """A memo must not survive the artifact being deleted under a live process."""
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    original.root.chmod(original.root.stat().st_mode | stat.S_IWUSR)
    index_path.chmod(index_path.stat().st_mode | stat.S_IWUSR)
    index_path.unlink()

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root == original.root
    assert rebuilt.root.joinpath("index.db").is_file()


_FRESH_PROCESS_PROBE = """
import json, sys
from pathlib import Path
from tests.infra.workload_artifacts import build_seeded_archive

cache_root = Path(sys.argv[1])
artifact = build_seeded_archive(cache_root=cache_root)
print(json.dumps({"key": artifact.manifest.key, "root": str(artifact.root)}))
"""


@pytest.mark.uses_real_clock("spawns a fresh interpreter; no timestamp assertions")
def test_seeded_archive_corruption_is_refused_by_a_fresh_process(tmp_path: Path) -> None:
    """The memo is per-process: a NEW process still validates in full.

    Red twin for the validate-once memo. Corrupts ``index.db`` in place
    without changing its size, so the cheap presence/size check a warm
    process uses cannot see it -- only the full SHA-256 comparison can. A
    freshly spawned interpreter has no memo, must therefore run that full
    validation, must reject the artifact, and must republish it.

    Anti-vacuity: making the memo process-global (a file on disk, or an
    unconditional trust of the published manifest without re-hashing) makes
    this fail, because the fresh process would accept the corrupted bytes.
    """
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)

    index_path = original.root / "index.db"
    # ``stat.S_IWUSR``, not ``os.W_OK``: the latter is 2, which as a mode bit
    # is ``S_IWOTH`` and grants this process nothing.
    index_path.chmod(index_path.stat().st_mode | stat.S_IWUSR)
    size_before = index_path.stat().st_size
    with index_path.open("r+b") as handle:
        handle.seek(size_before // 2)
        handle.write(b"\xde\xad\xbe\xef")
    assert index_path.stat().st_size == size_before

    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_PROBE, str(cache_root)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    republished = json.loads(result.stdout.strip().splitlines()[-1])
    # Same cache identity, freshly generated bytes: the artifact is rebuilt in
    # place, not accepted. ``manifest_id`` deliberately is NOT compared -- it
    # digests the per-file SHA-256 list, and a rebuild produces byte-different
    # SQLite files for identical logical content, so equality there would be
    # asserting determinism the pipeline never promised.
    assert republished["key"] == original.manifest.key
    assert republished["root"] == str(original.root)
    with index_path.open("rb") as handle:
        handle.seek(size_before // 2)
        assert handle.read(4) != b"\xde\xad\xbe\xef"


# ---------------------------------------------------------------------------
# Cache-root placement
# ---------------------------------------------------------------------------


def test_default_cache_root_falls_back_when_realm_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without ``/realm`` must not crash every consumer of this module.

    ``mkdir(parents=True)`` cannot create a directory under a nonexistent
    mount point, so the previously hard-coded ``/realm/tmp`` root made the
    seeded-archive cache raise ``OSError`` on any cloud sandbox.
    """
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    assert artifacts.default_cache_root() == artifacts._CLOUD_CACHE_ROOT

    monkeypatch.undo()
    if artifacts._SCRATCH_CACHE_ROOT.parent.is_dir():
        assert artifacts.default_cache_root() == artifacts._SCRATCH_CACHE_ROOT


# ---------------------------------------------------------------------------
# Read-only fixture path
# ---------------------------------------------------------------------------


def test_named_seeded_archive_ro_serves_a_readable_private_clone(
    named_seeded_archive_ro: Callable[[str], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-only fixture hands back a private clone for path-based consumers.

    Anti-vacuity: returning the shared artifact would make the ``artifacts/``
    containment assertion fail, exposing a mutation leak into the shared cache.
    """
    monkeypatch.setattr(
        "tests.infra.corpus_fixtures.build_seeded_archive",
        lambda specs: build_seeded_archive(specs, cache_root=tmp_path / "cache"),
    )
    db_path = named_seeded_archive_ro("cli-chatgpt")

    assert db_path.is_file()
    assert db_path.name == "index.db"
    assert "artifacts" not in db_path.parts
    assert os.environ["POLYLOGUE_ARCHIVE_ROOT"] == str(db_path.parent)
    with ArchiveStore.open_existing(db_path.parent, read_only=True) as archive:
        assert archive.count_sessions() > 0


@pytest.mark.parametrize("mutation", ("content", "sidecar", "replacement", "symlink"))
def test_query_only_lease_refuses_mutated_or_replaced_source(tmp_path: Path, mutation: str) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    lease = acquire_query_only_seeded_archive(artifact, seeded_archive_key((c03_semantic_corpus_spec(),)))
    root = artifact.root
    try:
        if mutation == "content":
            index = root / "index.db"
            root.chmod(root.stat().st_mode | stat.S_IWUSR)
            index.chmod(index.stat().st_mode | stat.S_IWUSR)
            payload = index.read_bytes()
            index.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
        elif mutation == "sidecar":
            root.chmod(root.stat().st_mode | stat.S_IWUSR)
            root.joinpath("index.db-wal").write_bytes(b"injected")
        else:
            parked = root.with_name(f"{root.name}-{mutation}")
            root.rename(parked)
            if mutation == "symlink":
                root.symlink_to(parked, target_is_directory=True)

        with pytest.raises(RuntimeError, match="query-only capability source"):
            lease.open()
    finally:
        lease.close()


def test_query_only_lease_allows_only_authenticated_read_use_and_finalization(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    lease = acquire_query_only_seeded_archive(artifact, seeded_archive_key((c03_semantic_corpus_spec(),)))

    with lease.open() as archive:
        assert archive.count_sessions() == 64
        with pytest.raises(ReadOnlyArchiveError, match="read-only"):
            archive.delete_sessions(("codex-session:c03-target",))
    with pytest.raises(RuntimeError, match="write-capable"):
        lease.open(read_only=False)

    lease.close()
    with pytest.raises(RuntimeError, match="finalized"):
        lease.open()


# ---------------------------------------------------------------------------
# Build-time same-process lock retry (polylogue-lbgc class)
# ---------------------------------------------------------------------------


def test_build_retries_a_transient_same_process_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient SQLITE_LOCKED during real ingest must not fail the consumer.

    Observed as two setup ERRORs in a full bootstrap run: the fixture calls
    ``build_seeded_archive``, the ingest inside it raises
    ``sqlite3.OperationalError: database is locked`` from
    ``introspection.table_exists``, and every test depending on that fixture
    errors out. The cause is the ``sqlite3_close_v2`` zombie-connection
    footgun already documented on :func:`_journal_mode_delete_with_retry`
    (polylogue-lbgc): SQLITE_LOCKED is a same-process conflict that the
    busy-timeout does not retry.

    Anti-vacuity: removing the ``except sqlite3.OperationalError`` arm makes
    this fail on the first injected lock, because nothing else absorbs it.
    """
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive as real_parse

    attempts = 0

    async def lock_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return await real_parse(*args, **kwargs)

    monkeypatch.setattr(artifacts, "parse_sources_archive", lock_once)
    artifacts._VALIDATED_ARTIFACTS.clear()

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")

    assert attempts == 2
    assert artifact.root.joinpath("index.db").is_file()
    # The abandoned first attempt must leave no staging tree behind.
    assert not list((tmp_path / "cache" / ".staging").iterdir())


def test_build_does_not_retry_a_non_lock_database_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying a real failure would hide it; only same-process locks qualify."""
    import tests.infra.workload_artifacts as artifacts

    attempts = 0

    async def always_broken(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(artifacts, "parse_sources_archive", always_broken)
    artifacts._VALIDATED_ARTIFACTS.clear()

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        build_seeded_archive(cache_root=tmp_path / "cache")

    assert attempts == 1


def _age_artifact(root: Path, *, now: float = 10_000.0) -> None:
    old = now - 10_000
    os.utime(root, (old, old))
    os.utime(root / "manifest.json", (old, old))


def test_artifact_gc_uses_current_manifest_key_not_build_id_and_respects_grace(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    with pytest.raises(ValueError, match="complete seeded archive keys"):
        gc_seeded_archive_artifacts(
            cache_root=cache_root,
            reachable_keys=("git:" + "0" * 40,),
        )
    reachable = build_seeded_archive(cache_root=cache_root)
    stale = build_seeded_archive(named_corpus_specs("cli-chatgpt"), cache_root=cache_root)
    _age_artifact(stale.root)

    preview = gc_seeded_archive_artifacts(
        cache_root=cache_root,
        reachable_keys=(reachable.manifest.key,),
        grace_period_s=1,
        now=10_000,
        dry_run=True,
    )

    by_name = {entry.name: entry for entry in preview.entries}
    assert by_name[reachable.root.name].disposition is ArtifactGcDisposition.REACHABLE
    assert by_name[stale.root.name].disposition is ArtifactGcDisposition.STALE
    assert by_name[stale.root.name].key == stale.manifest.key
    assert by_name[stale.root.name].manifest_id == stale.manifest.manifest_id
    assert stale.root.exists()


def test_artifact_gc_preview_and_apply_write_receipts_and_delete_only_stale_final_tree(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    stale = build_seeded_archive(cache_root=cache_root)
    _age_artifact(stale.root)
    preview_receipt = tmp_path / "preview.json"
    preview = gc_seeded_archive_artifacts(
        cache_root=cache_root,
        reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
        grace_period_s=1,
        now=10_000,
        dry_run=True,
        receipt_path=preview_receipt,
    )
    assert preview.entries[0].disposition is ArtifactGcDisposition.STALE
    assert json.loads(preview_receipt.read_text())["dry_run"] is True
    assert stale.root.exists()

    receipt = tmp_path / "apply.json"
    applied = gc_seeded_archive_artifacts(
        cache_root=cache_root,
        reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
        grace_period_s=1,
        now=10_000,
        dry_run=False,
        receipt_path=receipt,
    )
    assert applied.entries[0].disposition is ArtifactGcDisposition.DELETED
    assert not stale.root.exists()
    assert json.loads(receipt.read_text())["deleted_bytes"] == applied.deleted_bytes


def test_artifact_gc_preserves_active_per_key_lock_and_query_lease(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    artifact = build_seeded_archive(cache_root=cache_root)
    _age_artifact(artifact.root)
    lock_path = cache_root / ".locks" / f"{artifact.root.name}.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = gc_seeded_archive_artifacts(
            cache_root=cache_root,
            reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
            grace_period_s=1,
            now=10_000,
            dry_run=False,
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert locked.entries[0].disposition is ArtifactGcDisposition.ACTIVE_LOCK
    assert artifact.root.exists()

    cache_fd = os.open(cache_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(cache_fd, fcntl.LOCK_EX)
        cache_locked = gc_seeded_archive_artifacts(
            cache_root=cache_root,
            reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
            grace_period_s=1,
            now=10_000,
            dry_run=False,
        )
    finally:
        fcntl.flock(cache_fd, fcntl.LOCK_UN)
        os.close(cache_fd)
    assert cache_locked.entries[0].disposition is ArtifactGcDisposition.ACTIVE_LOCK
    assert cache_locked.entries[0].name == "<cache>"

    lease = acquire_query_only_seeded_archive(artifact, seeded_archive_key((c03_semantic_corpus_spec(),)))
    try:
        leased = gc_seeded_archive_artifacts(
            cache_root=cache_root,
            reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
            grace_period_s=1,
            now=10_000,
            dry_run=False,
        )
    finally:
        lease.close()
    assert leased.entries[0].disposition is ArtifactGcDisposition.ACTIVE_LEASE
    assert artifact.root.exists()


def test_artifact_gc_cannot_delete_source_while_clone_is_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clone's shared lease wins the race before its first source read."""
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    artifact = build_seeded_archive(cache_root=cache_root)
    _age_artifact(artifact.root)
    copy_started = threading.Event()
    allow_copy = threading.Event()
    clone_result: list[object] = []
    gc_result: list[object] = []
    original_copy = artifacts._copy_tree

    def blocked_copy(source: Path, destination: Path) -> None:
        copy_started.set()
        assert allow_copy.wait(5), "clone did not receive its release signal"
        original_copy(source, destination)

    monkeypatch.setattr(artifacts, "_copy_tree", blocked_copy)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )

    def run_clone() -> None:
        try:
            clone_result.append(clone_seeded_archive(artifact, tmp_path / "clone"))
        except BaseException as exc:  # surfaced below with the thread result
            clone_result.append(exc)

    def collect() -> None:
        gc_result.append(
            gc_seeded_archive_artifacts(
                cache_root=cache_root,
                reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
                grace_period_s=1,
                now=10_000,
                dry_run=False,
            )
        )

    clone_thread = threading.Thread(target=run_clone)
    clone_thread.start()
    assert copy_started.wait(5), "clone did not reach its source read barrier"
    gc_thread = threading.Thread(target=collect)
    gc_thread.start()
    gc_thread.join(5)
    allow_copy.set()
    clone_thread.join(5)

    assert not gc_thread.is_alive()
    assert not clone_thread.is_alive()
    assert len(gc_result) == 1
    gc_report = cast(ArtifactGcReport, gc_result[0])
    assert gc_report.entries[0].disposition is ArtifactGcDisposition.ACTIVE_LOCK
    assert len(clone_result) == 1 and not isinstance(clone_result[0], BaseException)
    cloned = cast(SeededArchiveClone, clone_result[0])
    assert cloned.root.exists()
    cloned.close()
    assert artifact.root.exists()


def test_rejected_hardlink_clone_cleans_residue_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.infra.workload_artifacts as artifacts

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    source_manifest = (artifact.root / "manifest.json").read_bytes()
    source_index = artifact.root / "index.db"
    source_mode = stat.S_IMODE(source_index.stat().st_mode)
    destination = tmp_path / "hardlink-clone"
    original_copy = artifacts._copy_tree

    def inject_hardlink(source: Path, target: Path) -> None:
        original_copy(source, target)
        target.chmod(target.stat().st_mode | stat.S_IWUSR)
        target_index = target / "index.db"
        target_index.unlink()
        os.link(source / "index.db", target_index)

    monkeypatch.setattr(artifacts, "_copy_tree", inject_hardlink)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )
    with pytest.raises(ValueError, match="inode was not detached"):
        clone_seeded_archive(artifact, destination)

    assert not destination.exists()
    assert source_index.exists()
    assert stat.S_IMODE(source_index.stat().st_mode) == source_mode
    assert (artifact.root / "manifest.json").read_bytes() == source_manifest


def test_rejected_symlink_clone_cleans_residue_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.infra.workload_artifacts as artifacts

    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    source_manifest = (artifact.root / "manifest.json").read_bytes()
    destination = tmp_path / "symlink-clone"
    outside = tmp_path / "outside"
    outside.write_text("must remain", encoding="utf-8")
    original_copy = artifacts._copy_tree

    def inject_symlink(source: Path, target: Path) -> None:
        original_copy(source, target)
        target.chmod(target.stat().st_mode | stat.S_IWUSR)
        (target / "unsafe-link").symlink_to(outside)

    monkeypatch.setattr(artifacts, "_copy_tree", inject_symlink)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["cp"])),
    )
    with pytest.raises(ValueError, match="symlink"):
        clone_seeded_archive(artifact, destination)

    assert not destination.exists()
    assert outside.read_text(encoding="utf-8") == "must remain"
    assert (artifact.root / "manifest.json").read_bytes() == source_manifest


def test_artifact_gc_retains_corruption_and_explicit_worktree_protection(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    corrupt = build_seeded_archive(cache_root=cache_root)
    _age_artifact(corrupt.root)
    corrupt.root.chmod(corrupt.root.stat().st_mode | stat.S_IWUSR)
    index = corrupt.root / "index.db"
    index.chmod(index.stat().st_mode | stat.S_IWUSR)
    content = index.read_bytes()
    index.write_bytes(bytes((content[0] ^ 1,)) + content[1:])
    corrupt.root.chmod(corrupt.root.stat().st_mode & ~stat.S_IWUSR)
    corrupted = gc_seeded_archive_artifacts(
        cache_root=cache_root,
        reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
        grace_period_s=1,
        now=10_000,
        dry_run=False,
    )
    assert corrupted.entries[0].disposition is ArtifactGcDisposition.CORRUPT
    assert corrupt.root.exists()

    protected = build_seeded_archive(named_corpus_specs("cli-chatgpt"), cache_root=cache_root)
    _age_artifact(protected.root)
    guarded = gc_seeded_archive_artifacts(
        cache_root=cache_root,
        reachable_keys=("seeded-archive:sha256:" + "f" * 64,),
        grace_period_s=1,
        now=10_000,
        dry_run=False,
        protected_worktrees=(protected.root,),
    )
    protected_entry = next(entry for entry in guarded.entries if entry.name == protected.root.name)
    assert protected_entry.disposition is ArtifactGcDisposition.ACTIVE_WORKTREE
    assert protected.root.exists()


def test_build_gives_up_on_a_persistent_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lock that never clears must surface, not spin forever."""
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    attempts = 0

    async def always_locked(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(artifacts, "parse_sources_archive", always_locked)
    artifacts._VALIDATED_ARTIFACTS.clear()

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        build_seeded_archive(cache_root=tmp_path / "cache")

    assert attempts == artifacts._BUILD_LOCK_ATTEMPTS
    assert not list((tmp_path / "cache" / ".staging").iterdir())


@pytest.mark.parametrize(
    ("error_code", "error_name", "message"),
    (
        (sqlite3.SQLITE_LOCKED | (1 << 8), None, "opaque SQLite error"),
        (None, "SQLITE_LOCKED_SHAREDCACHE", "opaque SQLite error"),
        (None, None, "database table is locked: sqlite_master"),
        (sqlite3.SQLITE_BUSY, "SQLITE_BUSY", "database is busy"),
    ),
)
def test_transient_sqlite_lock_recognizes_shared_cache_variants(
    error_code: int | None,
    error_name: str | None,
    message: str,
) -> None:
    exc = sqlite3.OperationalError(message)
    if error_code is not None:
        exc.sqlite_errorcode = error_code
    if error_name is not None:
        exc.sqlite_errorname = error_name

    assert is_transient_sqlite_lock(exc)


@pytest.mark.parametrize(
    ("error_code", "message"),
    (
        (sqlite3.SQLITE_CORRUPT, "database disk image is malformed"),
        (sqlite3.SQLITE_CORRUPT, "database is locked while reading a corrupt page"),
        (sqlite3.SQLITE_IOERR, "disk I/O error"),
        (None, "no such table: sqlite_master"),
    ),
)
def test_transient_sqlite_lock_rejects_non_contention_errors(
    error_code: int | None,
    message: str,
) -> None:
    exc = sqlite3.OperationalError(message)
    if error_code is not None:
        exc.sqlite_errorcode = error_code

    assert not is_transient_sqlite_lock(exc)


def test_contention_during_cache_validation_reuses_published_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient validation lock must not turn valid bytes into a rebuild."""
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    published = build_seeded_archive(cache_root=cache_root)
    artifacts._VALIDATED_ARTIFACTS.clear()
    real_validate = artifacts._validate_artifact
    attempts = 0

    def contend_once(root: Path, key: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            cause = sqlite3.OperationalError("database table is locked: sqlite_master")
            raise artifacts._ArtifactValidationContentionError from cause
        return real_validate(root, key)

    def forbid_republish(path: Path) -> None:
        raise AssertionError(f"valid artifact was republished: {path}")

    monkeypatch.setattr(artifacts, "_validate_artifact", contend_once)
    monkeypatch.setattr(artifacts, "_remove_tree", forbid_republish)
    reused = build_seeded_archive(cache_root=cache_root)

    assert attempts >= 2
    assert reused.root == published.root
    assert reused.manifest.manifest_id == published.manifest.manifest_id
