from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import tests.conftest as conftest
from devtools import verify_runs
from tests.infra.frozen_clock import FrozenClock


@contextmanager
def _configured_pytest(config: Any) -> Generator[None, None, None]:
    """Run the configure hook directly without leaking its claim lock."""
    conftest.pytest_configure(cast("pytest.Config", config))
    basetemp = Path(str(config.option.basetemp))
    try:
        yield
    finally:
        conftest.pytest_unconfigure(cast("pytest.Config", config))
        conftest._release_basetemp_claim_lock(basetemp)


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


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), RuntimeError("teardown failure")])
def test_test_tmp_path_reclamation_runs_after_failure_or_interrupt(
    tmp_path: Path,
    exception: BaseException,
) -> None:
    tree = tmp_path / "test-private"
    tree.mkdir()
    fixture_generator = cast(Any, conftest._reclaim_test_tmp_path).__wrapped__
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=None)))
    cleanup = cast("Generator[None, BaseException, None]", fixture_generator(tree, request))

    assert next(cleanup) is None
    with pytest.raises(type(exception)):
        cleanup.throw(exception)

    assert not tree.exists()


def test_test_tmp_path_reclamation_keeps_explicit_diagnostic_basetemp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "diagnostic"
    tree = explicit / "test-private"
    tree.mkdir(parents=True)
    fixture_generator = cast(Any, conftest._reclaim_test_tmp_path).__wrapped__
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(explicit))))
    monkeypatch.delenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", raising=False)
    cleanup = cast("Generator[None, BaseException, None]", fixture_generator(tree, request))

    assert next(cleanup) is None
    with pytest.raises(RuntimeError):
        cleanup.throw(RuntimeError("failed diagnostic rerun"))

    assert tree.exists()


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
    configured.mkdir()
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
    monkeypatch.delenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", raising=False)
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
        "POLYLOGUE_PYTEST_MANAGED_BASETEMP",
    ):
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(config):
        assert Path(str(config.option.basetemp)).parent == scratch
        assert os.environ["POLYLOGUE_PYTEST_TMPFS"] == "0"


def test_bare_pytest_ignores_leaked_cloud_basetemp_on_workstation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _shm, scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(verify_runs._CLOUD_PYTEST_BASETEMP_ROOT))
    monkeypatch.delenv("POLYLOGUE_VERIFY_RUN_ID", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_TMPFS", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_RUN_ID", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_CHECKOUT", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", raising=False)
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(config):
        assert Path(str(config.option.basetemp)).parent == scratch
        assert "POLYLOGUE_PYTEST_BASETEMP_ROOT" not in os.environ
        assert os.environ["POLYLOGUE_PYTEST_TMPFS"] == "0"


def test_bare_pytest_routes_an_environment_configured_tmpfs_root_to_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(shm / "configured"))
    monkeypatch.delenv("POLYLOGUE_VERIFY_RUN_ID", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_TMPFS", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_RUN_ID", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_CHECKOUT", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", raising=False)
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(config):
        assert Path(str(config.option.basetemp)).parent == scratch
        assert "POLYLOGUE_PYTEST_BASETEMP_ROOT" not in os.environ
        assert os.environ["POLYLOGUE_PYTEST_TMPFS"] == "0"


def test_sweep_stale_polylogue_basetemps_preserves_unknown_seeded_and_recent(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    stale = tmp_path / "pytest-polylogue-dead-123"
    seeded = tmp_path / "pytest-polylogue-seeded-dead"
    recent = tmp_path / "pytest-polylogue-live-123"
    unrelated = tmp_path / "pytest-other"
    for path in (stale, seeded, recent, unrelated):
        path.mkdir()

    old = frozen_clock.time() - 24 * 60 * 60
    os.utime(stale, (old, old))
    os.utime(seeded, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert stale.exists()
    assert seeded.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_explicit_basetemp_remains_outside_a_later_startup_stale_sweep(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    explicit = tmp_path / "pytest-polylogue-debug"
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(explicit)),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(config):
        assert verify_runs.pytest_basetemp_claim_path(explicit, kind="caller-owned").is_file()
        old = frozen_clock.time() - 24 * 60 * 60
        os.utime(explicit, (old, old))

        conftest._sweep_stale_polylogue_basetemps(roots=(tmp_path,))

        assert explicit.exists()


def test_explicit_basetemp_clears_stale_managed_identity_from_prior_in_process_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "pytest-polylogue-debug"
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(explicit)),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_SCOPES", [])
    monkeypatch.setenv("POLYLOGUE_PYTEST_RUN_ID", "prior-run")
    monkeypatch.setenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", str(explicit))

    with _configured_pytest(config):
        assert "POLYLOGUE_PYTEST_RUN_ID" not in os.environ
        assert "POLYLOGUE_PYTEST_MANAGED_BASETEMP" not in os.environ
        explicit.mkdir(exist_ok=True)

    assert explicit.exists()


def test_nested_explicit_basetemp_restores_active_outer_managed_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ambient_identity = (
        os.environ.get("POLYLOGUE_PYTEST_RUN_ID"),
        os.environ.get("POLYLOGUE_PYTEST_MANAGED_BASETEMP"),
    )
    _shm, _scratch = _make_real_candidates(monkeypatch, tmp_path)
    for name in (
        "POLYLOGUE_VERIFY_RUN_ID",
        "POLYLOGUE_PYTEST_BASETEMP_ROOT",
        "POLYLOGUE_PYTEST_TMPFS",
        "POLYLOGUE_PYTEST_RUN_ID",
        "POLYLOGUE_PYTEST_CHECKOUT",
        "POLYLOGUE_PYTEST_MANAGED_BASETEMP",
    ):
        monkeypatch.delenv(name, raising=False)
    outer = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    explicit = tmp_path / "nested-diagnostic"
    nested = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(explicit)),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(outer):
        outer_identity = (
            os.environ["POLYLOGUE_PYTEST_RUN_ID"],
            os.environ["POLYLOGUE_PYTEST_MANAGED_BASETEMP"],
        )
        with _configured_pytest(nested):
            assert "POLYLOGUE_PYTEST_RUN_ID" not in os.environ
            assert "POLYLOGUE_PYTEST_MANAGED_BASETEMP" not in os.environ
        assert os.environ["POLYLOGUE_PYTEST_RUN_ID"] == outer_identity[0]
        assert os.environ["POLYLOGUE_PYTEST_MANAGED_BASETEMP"] == outer_identity[1]

    assert os.environ.get("POLYLOGUE_PYTEST_RUN_ID") == ambient_identity[0]
    assert os.environ.get("POLYLOGUE_PYTEST_MANAGED_BASETEMP") == ambient_identity[1]


@pytest.mark.parametrize("alias", [False, True])
def test_nested_explicit_basetemp_reuse_is_rejected_before_the_nonreentrant_claim_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    alias: bool,
) -> None:
    active = tmp_path / "active-basetemp"
    requested = active
    if alias:
        active.mkdir()
        requested = tmp_path / "active-basetemp-alias"
        requested.symlink_to(active, target_is_directory=True)
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_SCOPES", [])
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_BASETEMPS", set())
    outer = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(active)),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    nested = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(requested)),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(outer):
        with pytest.raises(pytest.UsageError, match="already active in this pytest process"):
            conftest.pytest_configure(cast("pytest.Config", nested))


def test_nested_supervised_explicit_basetemp_reuse_is_rejected_before_claiming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active = tmp_path / "active-basetemp"
    active.mkdir()
    run_id = "supervised-run"
    config = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(active)),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_SCOPES", [])
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_BASETEMPS", set())
    monkeypatch.setenv("POLYLOGUE_VERIFY_RUN_ID", run_id)
    monkeypatch.setenv("POLYLOGUE_PYTEST_RUN_ID", run_id)
    monkeypatch.setenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", str(active))
    conftest._mark_basetemp_owner(active)
    try:
        with _configured_pytest(config):
            with pytest.raises(pytest.UsageError, match="already active in this pytest process"):
                conftest.pytest_configure(cast("pytest.Config", config))
    finally:
        conftest._release_basetemp_claim_lock(active)


def test_nested_managed_pytest_forces_scratch_outside_the_outer_tmpfs_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_SCOPES", [])
    monkeypatch.setenv("POLYLOGUE_VERIFY_RUN_ID", "outer-supervisor")
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(shm))
    monkeypatch.setenv("POLYLOGUE_PYTEST_TMPFS", "1")
    outer = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    nested = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(outer):
        assert Path(str(outer.option.basetemp)).parent == shm
        with _configured_pytest(nested):
            assert Path(str(nested.option.basetemp)).parent == scratch
            assert os.environ["POLYLOGUE_PYTEST_TMPFS"] == "0"
        assert os.environ["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(shm)
        assert os.environ["POLYLOGUE_PYTEST_TMPFS"] == "1"


def test_nested_unmanaged_scopes_do_not_reclaim_the_live_outer_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _shm, _scratch = _make_real_candidates(monkeypatch, tmp_path)
    monkeypatch.setattr(conftest, "_ACTIVE_PYTEST_SCOPES", [])
    for name in (
        "POLYLOGUE_VERIFY_RUN_ID",
        "POLYLOGUE_PYTEST_BASETEMP_ROOT",
        "POLYLOGUE_PYTEST_TMPFS",
        "POLYLOGUE_PYTEST_RUN_ID",
        "POLYLOGUE_PYTEST_CHECKOUT",
        "POLYLOGUE_PYTEST_MANAGED_BASETEMP",
    ):
        monkeypatch.delenv(name, raising=False)
    outer = SimpleNamespace(
        option=SimpleNamespace(basetemp=None),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    middle = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(tmp_path / "middle-diagnostic")),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )
    inner = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(tmp_path / "inner-diagnostic")),
        addinivalue_line=lambda *args, **kwargs: None,
        rootpath=tmp_path,
    )

    with _configured_pytest(outer):
        outer_basetemp = Path(str(outer.option.basetemp))
        live_outer_tree = outer_basetemp / "still-live"
        live_outer_tree.mkdir(parents=True)
        with _configured_pytest(middle):
            with _configured_pytest(inner):
                assert "POLYLOGUE_PYTEST_MANAGED_BASETEMP" not in os.environ
            assert "POLYLOGUE_PYTEST_MANAGED_BASETEMP" not in os.environ
            fixture_generator = cast(Any, conftest._reclaim_test_tmp_path).__wrapped__
            request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(outer_basetemp))))
            cleanup = cast("Generator[None, BaseException, None]", fixture_generator(live_outer_tree, request))
            assert next(cleanup) is None
            cleanup.close()
            assert live_outer_tree.exists()


def test_explicit_basetemp_claim_survives_real_pytest_basetemp_replacement(tmp_path: Path) -> None:
    """Exercise pytest's lazy TempPathFactory clearing against our real conftest."""
    explicit = tmp_path / "pytest-polylogue-diagnostic"
    explicit.mkdir()
    cleared_by_pytest = explicit / "cleared-by-temp-path-factory"
    cleared_by_pytest.write_text("old", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if not key.startswith("POLYLOGUE_PYTEST_")}
    env.pop("POLYLOGUE_VERIFY_RUN_ID", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--basetemp",
            str(explicit),
            "tests/unit/infra/test_archive_templates.py::test_clone_fallback_is_private_writable_and_symlink_safe",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not cleared_by_pytest.exists()
    assert verify_runs.pytest_basetemp_claim_path(explicit, kind="caller-owned").is_file()


def test_claim_lock_inode_stays_contended_after_managed_claim_clear(tmp_path: Path) -> None:
    """A second process cannot lock a replacement inode for this basetemp."""
    basetemp = tmp_path / "pytest-polylogue-lock-inode"
    conftest._mark_caller_owned_basetemp(basetemp)
    lock_path = verify_runs.pytest_basetemp_claim_path(basetemp, kind="lock")
    try:
        verify_runs.clear_managed_pytest_basetemp_claim(basetemp)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import fcntl, sys\n"
                    "with open(sys.argv[1], 'a+', encoding='utf-8') as handle:\n"
                    "    try:\n"
                    "        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                    "    except BlockingIOError:\n"
                    "        raise SystemExit(0)\n"
                    "raise SystemExit(1)\n"
                ),
                str(lock_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        conftest._release_basetemp_claim_lock(basetemp)

    assert result.returncode == 0, result.stdout + result.stderr
    assert lock_path.is_file()


def test_claim_lock_failure_closes_handle_and_releases_thread_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basetemp = tmp_path / "pytest-polylogue-lock-failure"
    lock_path = verify_runs.pytest_basetemp_claim_path(basetemp, kind="lock")

    with monkeypatch.context() as scoped:

        def fail_lock(*_args: object) -> None:
            raise OSError("lock failed")

        scoped.setattr(fcntl, "flock", fail_lock)
        assert conftest._acquire_basetemp_claim_lock(basetemp, blocking=True) is None

    assert not conftest._BASE_TEMP_CLAIM_THREAD_LOCKS[lock_path].locked()
    handle = conftest._acquire_basetemp_claim_lock(basetemp, blocking=True)
    assert handle is not None
    conftest._release_basetemp_claim_lock(basetemp)


def test_explicit_basetemp_claim_failure_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basetemp = tmp_path / "unclaimable"
    monkeypatch.setattr(conftest, "_acquire_basetemp_claim_lock", lambda *_args, **_kwargs: None)

    with pytest.raises(pytest.UsageError, match="cannot claim the explicit basetemp"):
        conftest._mark_caller_owned_basetemp(basetemp)


def test_managed_basetemp_claim_collision_is_rejected_across_processes(tmp_path: Path) -> None:
    basetemp = tmp_path / "pytest-polylogue-managed-collision"
    conftest._mark_basetemp_owner(basetemp)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "import sys\n"
                    "import tests.conftest as conftest\n"
                    "from devtools.verify_runs import PytestResourceError\n"
                    "try:\n"
                    "    conftest._mark_basetemp_owner(Path(sys.argv[1]))\n"
                    "except PytestResourceError:\n"
                    "    raise SystemExit(0)\n"
                    "raise SystemExit(1)\n"
                ),
                str(basetemp),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        conftest._release_basetemp_claim_lock(basetemp)

    assert result.returncode == 0, result.stdout + result.stderr


def test_stale_sweep_and_explicit_claim_are_atomic_for_one_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    basetemp = tmp_path / "pytest-polylogue-race-123"
    basetemp.mkdir()
    conftest._mark_basetemp_owner(basetemp)
    verify_runs.pytest_basetemp_claim_path(basetemp, kind="managed").write_text("999999999", encoding="utf-8")
    conftest._release_basetemp_claim_lock(basetemp)
    old = frozen_clock.time() - 120
    os.utime(basetemp, (old, old))
    sweep_checked = threading.Event()
    allow_sweep = threading.Event()
    caller_claimed = threading.Event()
    original_owner_alive = verify_runs.managed_pytest_basetemp_owner_alive

    def pause_after_admission(entry: Path) -> bool | None:
        sweep_checked.set()
        assert allow_sweep.wait(timeout=2)
        return original_owner_alive(entry)

    monkeypatch.setattr(verify_runs, "managed_pytest_basetemp_owner_alive", pause_after_admission)
    sweeper = threading.Thread(
        target=conftest._sweep_stale_polylogue_basetemps,
        kwargs={"max_age_s": 60, "roots": (tmp_path,)},
    )
    sweeper.start()
    assert sweep_checked.wait(timeout=2)

    def claim_and_use() -> None:
        conftest._mark_caller_owned_basetemp(basetemp)
        basetemp.mkdir(exist_ok=True)
        caller_claimed.set()

    caller = threading.Thread(target=claim_and_use)
    caller.start()
    assert not caller_claimed.wait(timeout=0.1)
    allow_sweep.set()
    sweeper.join(timeout=2)
    caller.join(timeout=2)

    assert not sweeper.is_alive()
    assert not caller.is_alive()
    assert caller_claimed.is_set()
    assert basetemp.exists()
    assert verify_runs.pytest_basetemp_claim_path(basetemp, kind="caller-owned").is_file()


def test_sweep_stale_polylogue_basetemps_never_deletes_a_live_owner(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    """Critical safety invariant: age alone must never justify deletion — a
    long-running lane's basetemp must survive the sweep even once it is
    older than the stale-age threshold, as long as its owning process is
    still alive."""
    live = tmp_path / "pytest-polylogue-live-owner-123"
    live.mkdir()
    conftest._mark_basetemp_owner(live)
    old = frozen_clock.time() - 120
    os.utime(live, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert live.exists()


def test_sweep_stale_polylogue_basetemps_reclaims_a_confirmed_dead_owner(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    dead = tmp_path / "pytest-polylogue-dead-owner-123"
    dead.mkdir()
    # A pid that is guaranteed not to be alive right now (max pid + 1 territory
    # would flake on hosts near pid rollover; /proc simply never has this one).
    verify_runs.pytest_basetemp_claim_path(dead, kind="managed").write_text("999999999", encoding="utf-8")
    old = frozen_clock.time() - 120
    os.utime(dead, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert not dead.exists()


def test_sweep_stale_polylogue_basetemps_reclaims_reused_pid_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    stale = tmp_path / "pytest-polylogue-reused-pid-123"
    stale.mkdir()
    verify_runs.pytest_basetemp_claim_path(stale, kind="managed").write_text(f"{os.getpid()}:100", encoding="utf-8")
    monkeypatch.setattr(conftest, "_process_start_ticks", lambda _pid: 200)
    old = frozen_clock.time() - 120
    os.utime(stale, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert not stale.exists()


def test_sweep_stale_polylogue_basetemps_reclaims_read_only_fixture_tree(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    stale = tmp_path / "pytest-polylogue-read-only-123"
    nested = stale / "published" / "artifact"
    nested.mkdir(parents=True)
    payload = nested / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    payload.chmod(0o400)
    nested.chmod(0o500)
    (stale / "published").chmod(0o500)
    verify_runs.pytest_basetemp_claim_path(stale, kind="managed").write_text("999999999", encoding="utf-8")
    old = frozen_clock.time() - 120
    os.utime(stale, (old, old))

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert not stale.exists()


def test_sweep_stale_polylogue_basetemps_does_not_follow_top_level_symlink(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    target = tmp_path / "external-fixture"
    nested = target / "published"
    nested.mkdir(parents=True)
    payload = nested / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    payload.chmod(0o400)
    nested.chmod(0o500)
    target.chmod(0o500)
    old = frozen_clock.time() - 24 * 60 * 60
    os.utime(target, (old, old))

    link = tmp_path / "pytest-polylogue-stale-symlink-123"
    link.symlink_to(target, target_is_directory=True)
    before_modes = (target.stat().st_mode, nested.stat().st_mode, payload.stat().st_mode)

    conftest._sweep_stale_polylogue_basetemps(max_age_s=60, roots=(tmp_path,))

    assert link.is_symlink()
    assert target.is_dir()
    assert payload.read_text(encoding="utf-8") == "{}"
    assert (target.stat().st_mode, nested.stat().st_mode, payload.stat().st_mode) == before_modes


def test_sweep_stale_polylogue_basetemps_never_deletes_an_unknown_owner(
    tmp_path: Path,
) -> None:
    """Unknown paths may be an explicit caller racing a sweep, so retain them."""
    unmarked = tmp_path / "pytest-polylogue-unmarked-123"
    unmarked.mkdir()
    # Derive "now" from the directory's own just-created mtime (filesystem
    # metadata) rather than a direct host-clock read, which test code may not
    # perform (clock_guard).
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


@pytest.mark.parametrize("worker_id", [None, "gw0"])
def test_sessionfinish_releases_explicit_claim_without_managed_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker_id: str | None,
) -> None:
    basetemp = tmp_path / "pytest-polylogue-explicit"
    conftest._mark_caller_owned_basetemp(basetemp)
    lock_path = verify_runs.pytest_basetemp_claim_path(basetemp, kind="lock")
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(basetemp), numprocesses=0)))
    monkeypatch.delenv("POLYLOGUE_PYTEST_RUN_ID", raising=False)
    if worker_id is None:
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    else:
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker_id)

    conftest.pytest_sessionfinish(cast("pytest.Session", session), 0)

    assert not conftest._BASE_TEMP_CLAIM_THREAD_LOCKS[lock_path].locked()


def test_sessionfinish_reclaims_only_its_managed_basetemp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    basetemp = tmp_path / "pytest-polylogue-run-123"
    basetemp.mkdir()
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(basetemp), numprocesses=0)))
    monkeypatch.setenv("POLYLOGUE_PYTEST_RUN_ID", "run-123")
    monkeypatch.setenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", str(basetemp))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    conftest.pytest_sessionfinish(cast("pytest.Session", session), 1)

    assert not basetemp.exists()


def test_sessionfinish_retains_managed_claim_after_failed_rmtree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    basetemp = tmp_path / "pytest-polylogue-rmtree-failure"
    basetemp.mkdir()
    managed_claim = verify_runs.pytest_basetemp_claim_path(basetemp, kind="managed")
    managed_claim.write_text("999999999", encoding="utf-8")
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(basetemp), numprocesses=0)))
    monkeypatch.setenv("POLYLOGUE_PYTEST_RUN_ID", "run-123")
    monkeypatch.setenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", str(basetemp))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setattr(shutil, "rmtree", lambda _path, **_kwargs: None)

    conftest.pytest_sessionfinish(cast("pytest.Session", session), 1)

    assert basetemp.exists()
    assert managed_claim.is_file()


def test_sessionfinish_retains_explicit_diagnostic_basetemp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "pytest-polylogue-diagnostic"
    explicit.mkdir()
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(basetemp=str(explicit), numprocesses=0)))
    monkeypatch.setenv("POLYLOGUE_PYTEST_RUN_ID", "run-123")
    monkeypatch.setenv("POLYLOGUE_PYTEST_MANAGED_BASETEMP", str(tmp_path / "pytest-polylogue-other-run"))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    conftest.pytest_sessionfinish(cast("pytest.Session", session), 1)

    assert explicit.exists()
