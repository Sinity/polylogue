from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import tests.conftest as conftest


def test_managed_pytest_temp_root_defaults_to_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(conftest, "_DEFAULT_SCRATCH_ROOT", scratch)
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_TMPFS", raising=False)

    root, label = conftest._managed_pytest_temp_root()

    assert root == scratch
    assert label == "scratch"
    assert root.is_dir()


def test_managed_pytest_temp_root_honors_explicit_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured"
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(configured))
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS", "1")

    root, label = conftest._managed_pytest_temp_root()

    assert root == configured
    assert label == "configured"
    assert root.is_dir()


def test_managed_pytest_temp_root_uses_tmpfs_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(conftest, "_DEFAULT_SCRATCH_ROOT", scratch)
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", raising=False)
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS", "1")

    root, label = conftest._managed_pytest_temp_root()

    shm = Path("/dev/shm")
    if conftest._is_tmpfs(shm):
        assert root == shm
        assert label == "tmpfs opt-in"
    else:
        assert root == scratch
        assert label == "scratch"


def test_sweep_stale_polylogue_basetemps_preserves_seeded_and_recent(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "pytest-polylogue-dead-123"
    seeded = tmp_path / "pytest-polylogue-seeded-dead"
    recent = tmp_path / "pytest-polylogue-live-123"
    unrelated = tmp_path / "pytest-other"
    for path in (stale, seeded, recent, unrelated):
        path.mkdir()

    old = 1.0
    os.utime(stale, (old, old))
    os.utime(seeded, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert not stale.exists()
    assert seeded.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_sessionfinish_leaves_xdist_basetemp_for_supervisor_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    basetemp = tmp_path / "pytest-polylogue-run-123"
    basetemp.mkdir()
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(basetemp), numprocesses=8)))
    monkeypatch.setenv("POLYLOGUE_PYTEST_RUN_ID", "run-123")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    conftest.pytest_sessionfinish(cast("pytest.Session", session), 0)

    assert basetemp.exists()


def test_archive_template_clone_is_private(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "index.db").write_bytes(b"immutable-template")

    conftest._clone_archive_template(source, destination)
    (destination / "index.db").write_bytes(b"private-mutation")

    assert (source / "index.db").read_bytes() == b"immutable-template"
