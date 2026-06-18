"""Regression tests for the `devtools` CLI shell wrapper (`nix/devtools-wrapper.sh`).

The wrapper resolves the polylogue ops doctorout from the caller's cwd via
`git rev-parse --show-toplevel` so it is worktree-aware (issue #1209,
companion to #1193). These tests guard that contract directly against the
shell script — the same script is read verbatim into the flake's
`writeShellScriptBin`, so they cover what Nix actually installs.

The tests do not require Nix or the flake to be built. They invoke
`bash <wrapper>` in subprocesses with controlled environments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Path to the wrapper script, relative to the repo root the wrapper itself
# would discover (this test file lives under tests/unit/devtools/).
REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = REPO_ROOT / "nix" / "devtools-wrapper.sh"


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="wrapper test requires bash and git",
)


def _run_wrapper(
    *args: str,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    base_env = {
        # Provide PATH so bash, python, and git are discoverable; otherwise
        # set HOME to a tmp value so git doesn't read the developer's config.
        "PATH": os.environ["PATH"],
        "HOME": str(cwd),
    }
    if env is not None:
        base_env.update(env)
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=str(cwd),
        env=base_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _make_fake_checkout(root: Path) -> None:
    """Create a minimal git repo with a stub devtools/__main__.py that
    just prints its own __file__ on stdout so the caller can verify which
    checkout the wrapper resolved."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    devtools = root / "devtools"
    devtools.mkdir()
    (devtools / "__main__.py").write_text("import sys\nprint(__file__)\nprint('args=' + repr(sys.argv[1:]))\n")


def test_wrapper_resolves_from_cwd_git_root(tmp_path: Path) -> None:
    """Calling the wrapper from inside a git checkout should resolve to
    that checkout, regardless of POLYLOGUE_REPO_ROOT."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _make_fake_checkout(checkout)

    result = _run_wrapper("hello", cwd=checkout)
    assert result.returncode == 0, result.stderr
    expected = str(checkout / "devtools" / "__main__.py")
    assert expected in result.stdout, result.stdout
    assert "args=['hello']" in result.stdout


def test_wrapper_prefers_cwd_over_env_var(tmp_path: Path) -> None:
    """When POLYLOGUE_REPO_ROOT points to a different checkout but the
    cwd is inside a valid git checkout, cwd wins. This is the worktree
    scenario from #1193."""
    main_checkout = tmp_path / "main"
    worktree = tmp_path / "worktree"
    main_checkout.mkdir()
    worktree.mkdir()
    _make_fake_checkout(main_checkout)
    _make_fake_checkout(worktree)

    result = _run_wrapper(
        cwd=worktree,
        env={"POLYLOGUE_REPO_ROOT": str(main_checkout)},
    )
    assert result.returncode == 0, result.stderr
    assert str(worktree / "devtools" / "__main__.py") in result.stdout
    assert str(main_checkout / "devtools" / "__main__.py") not in result.stdout


def test_wrapper_falls_back_to_env_var_outside_git(tmp_path: Path) -> None:
    """When cwd is not in any git checkout, fall back to
    POLYLOGUE_REPO_ROOT if it points to a real checkout."""
    fake_checkout = tmp_path / "fake-checkout"
    fake_checkout.mkdir()
    devtools = fake_checkout / "devtools"
    devtools.mkdir()
    (devtools / "__main__.py").write_text("print(__file__)\n")

    outside_git = tmp_path / "elsewhere"
    outside_git.mkdir()

    result = _run_wrapper(
        cwd=outside_git,
        env={"POLYLOGUE_REPO_ROOT": str(fake_checkout)},
    )
    assert result.returncode == 0, result.stderr
    assert str(fake_checkout / "devtools" / "__main__.py") in result.stdout


def test_wrapper_errors_when_no_checkout_resolvable(tmp_path: Path) -> None:
    """With neither a git checkout nor a valid POLYLOGUE_REPO_ROOT, the
    wrapper must fail loudly rather than silently picking a wrong path."""
    outside_git = tmp_path / "elsewhere"
    outside_git.mkdir()

    result = _run_wrapper(cwd=outside_git, env={})
    assert result.returncode != 0
    assert "cannot locate a polylogue ops doctorout" in result.stderr


def test_wrapper_ignores_invalid_env_var(tmp_path: Path) -> None:
    """A stale POLYLOGUE_REPO_ROOT that no longer points at a checkout
    must not cause the wrapper to invoke a missing __main__.py."""
    git_root = tmp_path / "real"
    git_root.mkdir()
    _make_fake_checkout(git_root)

    stale = tmp_path / "stale-removed"
    # Note: stale never created.

    result = _run_wrapper(
        cwd=git_root,
        env={"POLYLOGUE_REPO_ROOT": str(stale)},
    )
    # cwd resolution wins here, env var is ignored because it's invalid.
    assert result.returncode == 0, result.stderr
    assert str(git_root / "devtools" / "__main__.py") in result.stdout


def test_repo_wrapper_script_has_no_hardcoded_paths() -> None:
    """The wrapper script itself must not bake in any
    /realm/project/polylogue path or other developer-specific absolute
    path. This is the static side of the audit AC for #1209."""
    body = WRAPPER.read_text()
    assert "/realm/project/polylogue" not in body
    # The wrapper may mention POLYLOGUE_REPO_ROOT (as a documented
    # overridable fallback), but it must not require it.
    assert "POLYLOGUE_REPO_ROOT is not set" not in body
