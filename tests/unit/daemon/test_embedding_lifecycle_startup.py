"""Synthetic laws for the actual daemon embedding lifecycle startup route."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.daemon.cli import _run_startup_embedding_lifecycle
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.storage.embeddings.generations import EmbeddingGenerationError, EmbeddingGenerationStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.embedding_backup_fixture import embedding_vector_rows
from tests.infra.embedding_lifecycle import seed_retained_vector


@pytest.mark.parametrize("split_embedding_root", [False, True])
def test_startup_preserves_canonical_paid_tier_through_configured_farm(
    tmp_path: Path,
    split_embedding_root: bool,
) -> None:
    canonical = bootstrap_archive_root(tmp_path / "canonical")
    seed_retained_vector(canonical / "embeddings.db")
    paid_rows = embedding_vector_rows(canonical / "embeddings.db")
    physical = canonical
    if split_embedding_root:
        physical = tmp_path / "purchased-tier"
        physical.mkdir()
        (canonical / "embeddings.db").rename(physical / "embeddings.db")
    configured = tmp_path / "configured"
    configured.mkdir()
    for tier in ("source", "index", "embeddings", "user", "audit", "ops"):
        target_root = physical if tier == "embeddings" else canonical
        (configured / f"{tier}.db").symlink_to(target_root / f"{tier}.db")
    alias_identity = (configured / "embeddings.db").lstat().st_ino
    original = (physical / "embeddings.db").read_bytes()

    async def startup() -> Path:
        return await _run_startup_embedding_lifecycle(DaemonWriteCoordinator(archive_root=configured), configured)

    active = asyncio.run(startup())
    assert active == physical / "embeddings.db"
    assert active.is_symlink()
    assert active.resolve().read_bytes() == original
    assert embedding_vector_rows(active.resolve()) == paid_rows
    assert (configured / "embeddings.db").lstat().st_ino == alias_identity
    assert (configured / "embeddings.db").resolve() == active.resolve()
    metadata = json.loads(next((physical / ".embeddings-generations").glob("gen-*/generation.json")).read_text())
    assert metadata["archive_root"] == str(physical)
    assert metadata["state"] == "active"
    first_generation = active.resolve()
    assert asyncio.run(startup()).resolve() == first_generation
    with EmbeddingGenerationStore(configured).writer_lock() as binding:
        assert binding.archive_root == str(configured)
        assert Path(binding.database_path) == first_generation
    with EmbeddingGenerationStore(physical).writer_lock() as binding:
        assert Path(binding.database_path) == first_generation


@pytest.mark.parametrize("phase", ["copy", "contract", "metadata"])
def test_actual_startup_retries_first_adoption_after_unpublished_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    root = bootstrap_archive_root(tmp_path / "archive")
    seed_retained_vector(root / "embeddings.db")
    original = (root / "embeddings.db").read_bytes()
    original_copy = shutil.copyfileobj
    original_contract = EmbeddingGenerationStore._database_contract
    original_write = EmbeddingGenerationStore._write_staged_generation

    def fail_copy(source: object, target: object, *args: object) -> None:
        original_copy(source, target, *args)  # type: ignore[arg-type]
        raise OSError("synthetic transient copy failure")

    def fail_contract(self: EmbeddingGenerationStore, *args: object, **kwargs: object) -> object:
        raise OSError("synthetic transient contract failure")

    def fail_metadata(self: EmbeddingGenerationStore, generation: object, directory: Path) -> None:
        # A partially written metadata file is still unpublished state.
        (directory / "generation.json").write_text("{", encoding="utf-8")
        raise OSError("synthetic transient metadata failure")

    if phase == "copy":
        monkeypatch.setattr(shutil, "copyfileobj", fail_copy)
    elif phase == "contract":
        monkeypatch.setattr(EmbeddingGenerationStore, "_database_contract", fail_contract)
    else:
        monkeypatch.setattr(EmbeddingGenerationStore, "_write_staged_generation", fail_metadata)

    async def startup() -> Path:
        return await _run_startup_embedding_lifecycle(DaemonWriteCoordinator(archive_root=root), root)

    with pytest.raises(OSError, match="synthetic transient"):
        asyncio.run(startup())
    assert (root / "embeddings.db").read_bytes() == original
    assert not (root / "embeddings.db").is_symlink()
    assert not list((root / ".embeddings-generations").glob("gen-*"))
    assert not list((root / ".embeddings-generations").glob("unpublished-*"))
    monkeypatch.setattr(shutil, "copyfileobj", original_copy)
    monkeypatch.setattr(EmbeddingGenerationStore, "_database_contract", original_contract)
    monkeypatch.setattr(EmbeddingGenerationStore, "_write_staged_generation", original_write)
    active = asyncio.run(startup())
    assert active.is_symlink()
    assert active.resolve().read_bytes() == original
    assert len(embedding_vector_rows(active.resolve())[0]) == 1
    assert len(list((root / ".embeddings-generations").glob("gen-*/generation.json"))) == 1


def test_split_tier_alias_retarget_refuses_admitted_binding(tmp_path: Path) -> None:
    canonical = bootstrap_archive_root(tmp_path / "canonical")
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "embeddings.db").symlink_to(canonical / "embeddings.db")
    store = EmbeddingGenerationStore(configured)
    # Admission must reject a retarget even when both locators are valid tiers.
    foreign = bootstrap_archive_root(tmp_path / "foreign")
    (configured / "embeddings.db").rename(configured / "original-link")
    (configured / "embeddings.db").symlink_to(foreign / "embeddings.db")
    with pytest.raises(EmbeddingGenerationError, match="configured alias changed"):
        store.ensure_active()
    assert not list((canonical / ".embeddings-generations").glob("gen-*"))
    assert not (foreign / "embeddings.db").is_symlink()


@pytest.mark.uses_real_clock("child process interruption exercises actual OS writer ownership")
@pytest.mark.parametrize("boundary", ["copy", "metadata", "published_intent"])
def test_process_exit_during_first_adoption_cannot_poison_fresh_owner_restart(tmp_path: Path, boundary: str) -> None:
    root = bootstrap_archive_root(tmp_path / "archive")
    seed_retained_vector(root / "embeddings.db")
    original = (root / "embeddings.db").read_bytes()
    program = """
import asyncio, os, shutil, sys
from pathlib import Path
from polylogue.daemon.cli import _run_startup_embedding_lifecycle
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.storage.embeddings.generations import EmbeddingGenerationStore
root = Path(sys.argv[1])
boundary = sys.argv[2]
def interrupt_copy(source, target, *args):
    target.write(source.read(128))
    target.flush()
    os.fsync(target.fileno())
    os._exit(73)
if boundary == "copy":
    shutil.copyfileobj = interrupt_copy
else:
    original_publish = EmbeddingGenerationStore._publish_staged_generation
    def interrupt_publish(self, generation, directory, identity):
        if boundary == "published_intent":
            original_publish(self, generation, directory, identity)
        os._exit(73)
    EmbeddingGenerationStore._publish_staged_generation = interrupt_publish
asyncio.run(_run_startup_embedding_lifecycle(DaemonWriteCoordinator(archive_root=root), root))
"""
    child = subprocess.run([sys.executable, "-c", program, str(root), boundary], capture_output=True, text=True)
    assert child.returncode == 73, child.stderr
    assert (root / "embeddings.db").read_bytes() == original
    published = list((root / ".embeddings-generations").glob("gen-*/embeddings.db"))
    assert len(published) == (1 if boundary == "published_intent" else 0)
    orphans = list((root / ".embeddings-generations").glob("unpublished-*/embeddings.db"))
    assert len(orphans) == (0 if boundary == "published_intent" else 1)
    orphan_bytes = original[:128] if boundary == "copy" else original
    for orphan in orphans:
        assert orphan.read_bytes() == orphan_bytes

    async def startup() -> Path:
        return await _run_startup_embedding_lifecycle(DaemonWriteCoordinator(archive_root=root), root)

    active = asyncio.run(startup())
    assert active.resolve().read_bytes() == original
    assert len(embedding_vector_rows(active.resolve())[0]) == 1
    for orphan in orphans:
        assert orphan.read_bytes() == orphan_bytes
    if published:
        assert active.resolve() == published[0]
    assert len(list((root / ".embeddings-generations").glob("gen-*/generation.json"))) == 1
    from unittest.mock import patch

    from polylogue.daemon.embedding_readiness import embedding_readiness_info
    from tests.infra.embedding_config import embedding_config

    with patch("polylogue.config.load_polylogue_config", return_value=embedding_config(embedding_enabled=False)):
        readiness = embedding_readiness_info(root / "index.db")
    assert readiness["embedding_unmeasurable_reason"] is None
    assert readiness["embedding_status"] != "unknown"
