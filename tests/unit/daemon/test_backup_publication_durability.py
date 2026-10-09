"""Neutral syscall-ordering proofs for backup and restore publication."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.operations.archive_backup import backup_archive, restore_verified_backup
from polylogue.storage.backup_attestation import load_or_mint_attestation_key
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.population_admission import POPULATION_PENDING


@pytest.fixture
def preservation_source(workspace_env: dict[str, Path]) -> tuple[Path, str]:
    root = workspace_env["archive_root"]
    payload = b'{"synthetic_record":"publication-barriers"}\n'
    BlobStore(root / "blob").write_from_bytes(payload)
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            capture_mode=Provider.CODEX,
            source_path="/synthetic/publication",
            canonical_source_path="/synthetic/publication",
            source_index=0,
            native_id=None,
            payload=payload,
            acquired_at_ms=2,
        )
    digest = hashlib.sha256(payload).hexdigest()
    return root, f"blob/{digest[:2]}/{digest[2:]}"


@pytest.mark.parametrize("verify", [False, True])
def test_backup_dependencies_precede_success_receipt(
    preservation_source: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify: bool,
) -> None:
    root, blob = preservation_source
    events: list[Path] = []
    actual = os.fsync

    def observe(fd: int) -> None:
        events.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        actual(fd)

    monkeypatch.setattr(os, "fsync", observe)
    result = backup_archive(output_dir=tmp_path / "new" / "packages", verify=verify, archive_root_path=root)
    assert result.ok and result.verified == verify and result.output_path
    package = Path(result.output_path)
    publication = next(
        (i for i, path in enumerate(events) if path.parent == package and path.name.startswith(".publish-")),
        len(events),
    )
    for relative in ("source.db", "user.db", "audit.db", "embeddings.db", "manifest.json", blob):
        path = package / relative
        assert events.count(path) == 1
        assert events.index(path) < publication
    for path in ((package / blob).parent, package / "blob", package, package.parent, package.parent.parent):
        assert events.index(path) < publication


@pytest.mark.parametrize("failure", ["tier", "blob", "directory", "receipt", "receipt_directory"])
def test_backup_barrier_failure_cannot_publish_success(
    preservation_source: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root, blob = preservation_source
    output = tmp_path / "packages"
    actual = os.fsync
    failed = False

    def inject(fd: int) -> None:
        nonlocal failed
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if output in path.parents:
            relative = path.relative_to(output)
            name = relative.parts[1:]  # generated package root
            selected = (
                (failure == "tier" and name == ("source.db",))
                or (failure == "blob" and name == tuple(Path(blob).parts))
                or (failure == "directory" and name == ("blob", Path(blob).parts[1]))
                or (failure == "receipt" and len(name) == 1 and name[0].startswith(".publish-"))
                or (failure == "receipt_directory" and not name and (path / "verification-receipt.json").exists())
            )
            if selected and not failed:
                failed = True
                raise OSError("synthetic publication barrier failure")
        actual(fd)

    monkeypatch.setattr(os, "fsync", inject)
    result = backup_archive(output_dir=output, verify=True, archive_root_path=root)
    assert failed
    assert not result.ok and not result.verified
    assert result.output_path
    assert not (Path(result.output_path) / "verification-receipt.json").exists()


def test_first_attestation_key_persists_new_ancestors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = tmp_path / "state" / "backup-attestations" / "neutral.key"
    monkeypatch.setattr("polylogue.storage.backup_attestation.attestation_key_path", lambda _: key)
    synced: list[Path] = []
    actual = os.fsync

    def observe(fd: int) -> None:
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        actual(fd)

    monkeypatch.setattr(os, "fsync", observe)
    assert len(load_or_mint_attestation_key(tmp_path / "source.db")) == 32
    assert key.parent in synced and key.parent.parent in synced and tmp_path in synced


@pytest.mark.parametrize("failure", [None, "tier", "blob", "directory", "fence", "fence_interrupt"])
def test_restore_dependencies_precede_fence_retirement_and_fail_closed(
    preservation_source: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    from polylogue.core import durable_fs
    from polylogue.storage.sqlite.archive_tiers import archive_plan

    root, blob = preservation_source
    package_result = backup_archive(output_dir=tmp_path / "packages", verify=True, archive_root_path=root)
    assert package_result.ok and package_result.output_path
    destination = tmp_path / "new" / "restored"
    actual_sync_tree = durable_fs.sync_tree
    actual_fsync = os.fsync
    actual_unlink = Path.unlink
    actual_sync_directory = archive_plan._fsync_directory
    events: list[Path | str] = []
    in_barrier = False
    failed = False

    def tree(path: Path) -> None:
        nonlocal in_barrier
        assert path == destination
        in_barrier = True
        try:
            actual_sync_tree(path)
        finally:
            in_barrier = False

    def observe(fd: int) -> None:
        nonlocal failed
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if in_barrier:
            events.append(path)
            targets = {
                "tier": destination / "embeddings.db",
                "blob": destination / blob,
                "directory": (destination / blob).parent,
            }
            if path == targets.get(failure) and not failed:
                failed = True
                raise OSError("synthetic restore dependency failure")
        actual_fsync(fd)

    def unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if path == destination / POPULATION_PENDING:
            events.append("fence-retired")
        actual_unlink(path, *args, **kwargs)
        if path == destination / POPULATION_PENDING and failure == "fence_interrupt" and not failed:
            failed = True
            raise KeyboardInterrupt

    def sync_directory(path: Path) -> None:
        nonlocal failed
        if failure == "fence" and path == destination and not (path / POPULATION_PENDING).exists() and not failed:
            failed = True
            raise OSError("synthetic fence retirement failure")
        actual_sync_directory(path)

    monkeypatch.setattr(durable_fs, "sync_tree", tree)
    monkeypatch.setattr(os, "fsync", observe)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(archive_plan, "_fsync_directory", sync_directory)
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == "fence_interrupt" else OSError):
            restore_verified_backup(backup_dir=Path(package_result.output_path), destination=destination)
        assert failed
        assert (destination / POPULATION_PENDING).is_file()
        assert (destination / "embeddings.db").is_file()
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
        from polylogue.storage.sqlite.population_admission import ArchivePopulationPendingError

        with pytest.raises(ArchivePopulationPendingError):
            initialize_active_archive_root(destination)
    else:
        restore_verified_backup(backup_dir=Path(package_result.output_path), destination=destination)
        retirement = events.index("fence-retired")
        for path in [destination / tier for tier in ("source.db", "user.db", "audit.db", "embeddings.db")] + [
            destination / blob,
            (destination / blob).parent,
            destination,
            destination.parent,
        ]:
            assert events.count(path) == 1
            assert events.index(path) < retirement
        assert not (destination / POPULATION_PENDING).exists()
