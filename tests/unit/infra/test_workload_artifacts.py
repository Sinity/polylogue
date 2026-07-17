"""Tests for the one shared real-pipeline seeded archive adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.infra.workload_artifacts import build_seeded_archive, clone_seeded_archive


def test_seeded_archive_publishes_valid_immutable_real_pipeline_artifact(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"

    first = build_seeded_archive(cache_root=cache_root)
    second = build_seeded_archive(cache_root=cache_root)

    assert first.root == second.root
    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.manifest.receipt["status"] == "succeeded"
    assert len(first.facts) == 64
    assert first.facts[0].expected_session_id == "codex-session:c03-target"
    assert first.root.joinpath("index.db").exists()
    assert not (first.root.stat().st_mode & os.W_OK)


def test_seeded_archive_clone_is_private_full_root_and_preserves_base(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    base_manifest = artifact.root.joinpath("manifest.json").read_bytes()

    clone = clone_seeded_archive(artifact, tmp_path / "clone")
    clone.root.joinpath("private-mutation.txt").write_text("private")

    assert clone.clone_method in {"reflink", "copy"}
    assert clone.source_manifest_id == artifact.manifest.manifest_id
    assert clone.root.joinpath("source.db").exists()
    assert clone.root.joinpath("index.db").exists()
    assert artifact.root.joinpath("manifest.json").read_bytes() == base_manifest
    assert not artifact.root.joinpath("private-mutation.txt").exists()


def test_seeded_archive_rejects_corrupt_published_cache_and_rebuilds(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    index_path.chmod(index_path.stat().st_mode | os.W_OK)
    index_path.unlink()

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root == original.root
    assert rebuilt.root.joinpath("index.db").is_file()
    assert rebuilt.manifest.key == original.manifest.key
    assert rebuilt.manifest.profile_id == original.manifest.profile_id
    assert rebuilt.manifest.recipe_id == original.manifest.recipe_id
    assert rebuilt.facts == original.facts
    assert not (rebuilt.root.stat().st_mode & os.W_OK)


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
