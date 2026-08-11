from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import tests.conftest as conftest
from devtools import verify_runs


def _make_real_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, realm_mounted: bool = True
) -> tuple[Path, Path]:
    """Point every placement-policy constant at real, tmp_path-backed dirs.

    ``tests/conftest.py`` delegates placement entirely to
    ``devtools.verify_runs.resolve_pytest_basetemp_root`` — these tests exist
    to prove the delegation and the mkdir/no-CoW side effects, not to
    re-implement candidate selection (covered by
    ``tests/unit/devtools/test_verify.py``). Repointing
    ``PYTEST_TMPFS_ROOT``/``DEFAULT_PYTEST_BASETEMP_ROOT``/
    ``_CLOUD_PYTEST_BASETEMP_ROOT`` (not just the internal candidate check)
    matters here specifically: ``pytest_configure`` also runs the stale-sweep
    across every *known* root, and without repointing all three, a test in
    this module could otherwise glob and delete real basetemps under a
    shared host's actual ``/dev/shm`` or ``/tmp/polylogue-pytest``.
    """
    shm = tmp_path / "dev-shm"
    shm.mkdir()
    scratch_parent = tmp_path / "realm-tmp"
    scratch = scratch_parent / "polylogue-pytest"
    cloud_fallback = tmp_path / "tmp" / "polylogue-pytest"
    cloud_fallback.parent.mkdir(parents=True)
    if realm_mounted:
        scratch_parent.mkdir()
    monkeypatch.setattr(verify_runs, "PYTEST_TMPFS_ROOT", shm)
    monkeypatch.setattr(verify_runs, "DEFAULT_PYTEST_BASETEMP_ROOT", scratch)
    monkeypatch.setattr(verify_runs, "_CLOUD_PYTEST_BASETEMP_ROOT", cloud_fallback)
    monkeypatch.setattr(verify_runs, "_is_tmpfs_dir", lambda path: path == shm)
    monkeypatch.setattr(
        verify_runs,
        "_fs_usage",
        lambda path: {"used_kb": 0, "free_kb": 32 * 1024 * 1024} if path in (shm, scratch_parent) else None,
    )
    return shm, scratch


def test_managed_pytest_temp_root_defaults_to_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _shm, scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setattr(verify_runs, "_is_tmpfs_dir", lambda path: False)  # force past the tmpfs candidate
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
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 32 * 1024 * 1024})
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(configured))
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS", "1")

    root, label = conftest._managed_pytest_temp_root()

    assert root == configured
    assert label == "configured"
    assert root.is_dir()


def test_managed_pytest_temp_root_uses_tmpfs_when_requested_and_it_has_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, _scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", raising=False)
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS", "1")

    root, label = conftest._managed_pytest_temp_root()

    assert root == shm
    assert label == "tmpfs opt-in"


def test_managed_pytest_temp_root_uses_scratch_when_tmpfs_budget_leaves_no_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setattr(
        verify_runs,
        "_fs_usage",
        lambda path: {"used_kb": 0, "free_kb": 1_500 * 1024} if path in (shm, scratch.parent) else None,
    )
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", raising=False)
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS", "1")
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS_MAX_MB", "512")

    root, label = verify_runs.resolve_pytest_basetemp_root(
        {
            "POLYLOGUE_PYTEST_TMPFS": "1",
            "POLYLOGUE_PYTEST_TMPFS_MAX_MB": "512",
        }
    )

    assert root == scratch
    assert label == "scratch"


def test_managed_pytest_temp_root_refuses_when_every_candidate_is_full(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Demonstrates the low-space path deliberately: every candidate reports
    starved free space, and the run refuses loudly instead of silently
    picking a too-small location that fills up mid-run."""
    _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 1024})  # 1 MiB anywhere
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", raising=False)

    with pytest.raises(verify_runs.PytestResourceError, match="no pytest basetemp location has enough free space"):
        conftest._managed_pytest_temp_root()


def test_negative_declared_basetemp_demand_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB", "-1")

    with pytest.raises(verify_runs.PytestResourceError, match="invalid POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB"):
        verify_runs.pytest_basetemp_required_kb(os.environ)


def test_pytest_configure_reports_low_space_as_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``pytest_configure`` turns a starved-basetemp refusal into a clean
    ``UsageError`` instead of an uncaught traceback, so a low-space run fails
    at startup with an explicit reason rather than mid-collection."""
    _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 1024})
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", raising=False)
    # These are set as a side effect of the real pytest_configure() called
    # below; pre-registering them with monkeypatch (even absent) guarantees
    # teardown reverts the leak regardless of what the call under test does.
    monkeypatch.delenv("POLYLOGUE_PYTEST_RUN_ID", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_CHECKOUT", raising=False)
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with pytest.raises(pytest.UsageError, match="no pytest basetemp location has enough free space"):
        conftest.pytest_configure(cast("pytest.Config", config))


def test_bare_pytest_configure_defaults_to_scratch_without_a_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _shm, scratch = _make_real_candidates(monkeypatch, tmp_path)
    for name in (
        "POLYLOGUE_VERIFY_RUN_ID",
        "POLYLOGUE_PYTEST_BASETEMP_ROOT",
        "POLYLOGUE_PYTEST_TMPFS",
        "POLYLOGUE_PYTEST_RUN_ID",
        "POLYLOGUE_PYTEST_CHECKOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    conftest.pytest_configure(cast("pytest.Config", config))

    assert Path(str(config.option.basetemp)).parent == scratch
    assert os.environ["POLYLOGUE_PYTEST_TMPFS"] == "0"


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


def test_sweep_stale_polylogue_basetemps_never_deletes_a_live_owner(
    tmp_path: Path,
) -> None:
    """Critical safety invariant: age alone must never justify deletion — a
    long-running lane's basetemp must survive the sweep even once it is
    older than the stale-age threshold, as long as its owning process is
    still alive."""
    live = tmp_path / "pytest-polylogue-live-owner-123"
    live.mkdir()
    conftest._mark_basetemp_owner(live)
    old = 1.0
    os.utime(live, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert live.exists()


def test_sweep_stale_polylogue_basetemps_reclaims_a_confirmed_dead_owner(
    tmp_path: Path,
) -> None:
    dead = tmp_path / "pytest-polylogue-dead-owner-123"
    dead.mkdir()
    # A pid that is guaranteed not to be alive right now (max pid + 1 territory
    # would flake on hosts near pid rollover; /proc simply never has this one).
    (dead / conftest._OWNER_PID_MARKER).write_text("999999999", encoding="utf-8")
    old = 1.0
    os.utime(dead, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert not dead.exists()


def test_sweep_stale_polylogue_basetemps_gives_unknown_owner_a_long_grace_period(
    tmp_path: Path,
) -> None:
    """A directory with no owner marker (pre-fix leftover, or a startup
    race) cannot be confirmed dead, so it gets a much longer grace period
    rather than the normal stale-age cutoff."""
    unmarked = tmp_path / "pytest-polylogue-unmarked-123"
    unmarked.mkdir()
    # Past the normal (60s, for this test) stale-age cutoff, but nowhere near
    # the multi-hour unknown-owner grace period. Derive "now" from the
    # directory's own just-created mtime (filesystem metadata) rather than a
    # direct host-clock read, which test code may not perform (clock_guard).
    now = unmarked.stat().st_mtime
    past_normal_cutoff = now - 120
    os.utime(unmarked, (past_normal_cutoff, past_normal_cutoff))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert unmarked.exists()


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


def test_archive_template_clone_rebinds_durable_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    marker_relative = Path(".maintenance-state/durable-change-trains/.bootstrap")
    with ArchiveStore(source):
        pass
    source_marker = (source / marker_relative).read_bytes()

    conftest._clone_archive_template(source, destination)

    assert (destination / marker_relative).read_bytes() != source_marker
    with ArchiveStore(destination):
        pass
    assert (source / marker_relative).read_bytes() == source_marker
