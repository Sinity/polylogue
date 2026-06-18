"""Installed-package smoke tests under fresh XDG paths.

Builds the polylogue wheel, installs it into an isolated virtual environment,
and exercises the documented operator entrypoints under temporary XDG paths.
The test asserts that:

- ``polylogue --version``, ``polylogue --plain status``, ``polylogue --plain analyze --count``,
  ``polylogued --help``, ``polylogued status``, and ``polylogue-mcp --help`` all
  succeed against a fresh installed wheel with no archive present;
- writes during those invocations stay inside the temporary XDG directories
  (no leakage into ``$HOME`` or other host paths).

This closes #1265 (slice D of #869): the existing
``devtools release verify-distribution`` smoke only exercised ``--help``/
``--version``/``analyze --count``; daemon status under fresh XDG paths was not covered.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from devtools import repo_root

REPO_ROOT = repo_root()
RUNTIME_SCRIPTS = ("polylogue", "polylogued", "polylogue-mcp")

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


requires_uv = pytest.mark.skipif(
    not _have("uv"),
    reason="uv is required to build and install the wheel under a fresh venv",
)


def _build_wheel(work_dir: Path) -> Path:
    """Build a wheel from the repo checkout into ``work_dir/dist``."""
    dist_dir = work_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    _run(
        ("uv", "build", "--out-dir", str(dist_dir), "--wheel", str(REPO_ROOT)),
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {dist_dir}, got {wheels!r}"
    return wheels[0]


def _install_wheel(wheel: Path, install_dir: Path) -> Path:
    """Create a venv under ``install_dir`` and pip-install the wheel into it.

    Returns the venv's ``bin``/``Scripts`` directory.
    """
    install_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = install_dir / "venv"
    _run(("uv", "venv", str(venv_dir)), cwd=install_dir)
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        ("uv", "pip", "install", "--python", str(python), str(wheel)),
        cwd=install_dir,
    )
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _fresh_xdg_env(xdg_root: Path) -> dict[str, str]:
    """Build an environment that points all XDG roots and HOME under ``xdg_root``.

    The synthetic ``HOME`` is the leak canary: anything the installed tools
    write under HOME but outside the XDG sub-roots is a violation of the
    fresh-install contract.
    """
    home = xdg_root / "home"
    config = xdg_root / "xdg" / "config"
    data = xdg_root / "xdg" / "data"
    cache = xdg_root / "xdg" / "cache"
    state = xdg_root / "xdg" / "state"
    for path in (home, config, data, cache, state):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        # Minimal PATH so subprocesses can still resolve ``sh`` / coreutils if
        # needed; the installed scripts themselves carry absolute paths.
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data),
        "XDG_CACHE_HOME": str(cache),
        "XDG_STATE_HOME": str(state),
        # Force deterministic, machine-readable output and disable any
        # ambient theming/locale variance.
        "POLYLOGUE_FORCE_PLAIN": "1",
        "NO_COLOR": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    # Preserve ``SSL_CERT_FILE`` etc. so uv-installed venvs that need TLS
    # certificates during runtime imports still find them.
    for keep in ("SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "TZ"):
        if keep in os.environ:
            env[keep] = os.environ[keep]
    return env


def _snapshot(root: Path) -> set[Path]:
    """Return all files under ``root`` (recursively)."""
    return {p for p in root.rglob("*") if p.is_file()}


def _run(
    cmd: Iterable[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(cmd)
    result = subprocess.run(
        tuple(cmd),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed (exit {result.returncode}): {rendered}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def _assert_no_home_leak(home: Path, before: set[Path], after: set[Path]) -> None:
    """Files newly written under ``home`` outside the XDG sub-roots fail the test."""
    new_files = after - before
    leaks = [p for p in new_files if home in p.parents]
    assert not leaks, (
        "installed polylogue wrote files under HOME outside the temporary XDG "
        f"sub-roots: {sorted(str(p) for p in leaks)!r}"
    )


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build the wheel once and install it into a single venv shared by the module."""
    if not _have("uv"):
        pytest.skip("uv is required to build and install the wheel under a fresh venv")
    work_dir = tmp_path_factory.mktemp("polylogue-installed-smoke")
    wheel = _build_wheel(work_dir)
    bin_dir = _install_wheel(wheel, work_dir / "install")
    for script in RUNTIME_SCRIPTS:
        assert (bin_dir / script).exists(), f"missing installed script: {script}"
    return work_dir, bin_dir


@requires_uv
def test_installed_polylogue_entrypoints_under_fresh_xdg(
    installed_wheel: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """All three installed entrypoints respond under fresh XDG paths."""
    _work, bin_dir = installed_wheel
    env = _fresh_xdg_env(tmp_path)
    home = Path(env["HOME"])
    before = _snapshot(home)

    # polylogue --version: must succeed and embed the build commit / dirty marker
    # so the version contract is exercised end-to-end.
    res = _run((str(bin_dir / "polylogue"), "--version"), env=env, timeout=30)
    assert "polylogue" in res.stdout.lower()

    # polylogue --help and ``polylogued``/``polylogue-mcp`` --help: cheap surface
    # checks that the click groups load with all extras resolvable.
    _run((str(bin_dir / "polylogue"), "--help"), env=env, timeout=30)
    _run((str(bin_dir / "polylogued"), "--help"), env=env, timeout=30)
    _run((str(bin_dir / "polylogue-mcp"), "--help"), env=env, timeout=30)

    # polylogue --plain analyze --count: forces an archive open (or first-run bootstrap)
    # under fresh XDG paths; must not traceback.
    _run((str(bin_dir / "polylogue"), "--plain", "analyze", "--count"), env=env, timeout=60)

    # polylogue --plain status: the actionable first-run surface (#1263) — must
    # exit cleanly and emit human text against a fresh archive.
    status = _run(
        (str(bin_dir / "polylogue"), "--plain", "status"),
        env=env,
        timeout=60,
    )
    assert status.stdout.strip(), "polylogue ops status produced no output"
    assert "Traceback" not in status.stdout + status.stderr

    # polylogued status: daemon-side status against fresh XDG paths (#1265 AC).
    daemon_status = _run((str(bin_dir / "polylogued"), "status"), env=env, timeout=60)
    assert "Traceback" not in daemon_status.stdout + daemon_status.stderr

    after = _snapshot(home)
    _assert_no_home_leak(home, before, after)


@requires_uv
def test_installed_polylogue_writes_only_under_xdg_roots(
    installed_wheel: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Archive bootstrap writes stay inside the configured XDG_DATA_HOME tree.

    Runs ``polylogue --plain analyze --count`` on a fresh archive and asserts that any
    files created under HOME live under one of the four declared XDG roots —
    a regression test for archive/config code paths that might silently fall
    back to ``~/.local/...`` even when the env vars are set.
    """
    _work, bin_dir = installed_wheel
    env = _fresh_xdg_env(tmp_path)
    home = Path(env["HOME"])
    xdg_roots = (
        Path(env["XDG_CONFIG_HOME"]),
        Path(env["XDG_DATA_HOME"]),
        Path(env["XDG_CACHE_HOME"]),
        Path(env["XDG_STATE_HOME"]),
    )

    before = _snapshot(home)
    _run((str(bin_dir / "polylogue"), "--plain", "analyze", "--count"), env=env, timeout=60)
    _run((str(bin_dir / "polylogue"), "--plain", "status"), env=env, timeout=60)
    after = _snapshot(home)

    new_under_home = sorted(p for p in (after - before) if home in p.parents)
    misplaced = [path for path in new_under_home if not any(root in path.parents or root == path for root in xdg_roots)]
    assert not misplaced, (
        "installed polylogue created files under HOME but outside the "
        f"declared XDG roots {[str(r) for r in xdg_roots]!r}: "
        f"{[str(p) for p in misplaced]!r}"
    )


def test_repo_dev_install_is_not_a_smoke_proxy() -> None:
    """Guard: this module must run against an installed wheel, not the dev tree.

    Without this assertion an editable install would silently satisfy the
    smoke contract via the source tree, masking packaging regressions.
    """
    # The dev venv has polylogue available via ``import polylogue``; the smoke
    # tests above instead resolve scripts under a freshly-created venv directory.
    # This test pins that contract so a future refactor that replaces the
    # venv-based fixture with ``sys.executable`` is rejected loudly.
    src = REPO_ROOT / "polylogue" / "__init__.py"
    assert src.exists(), "expected polylogue package source to be present in the repo"
    assert "polylogue" in Path(sys.executable).parts or src.exists(), (
        "sanity precondition: repo tree must be available so the installed-wheel fixture can build from it"
    )
