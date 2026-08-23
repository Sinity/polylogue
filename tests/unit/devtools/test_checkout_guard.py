"""Tests for the worktree-import guard (devtools/checkout_guard.py).

Covers the 2026-07-30/31 hazard: a shared/editable venv's `.pth` entry can
point at a different checkout than the one a tool is actually invoked from
(most concretely, a linked git worktree reusing the main checkout's
`.venv`), silently resolving `import polylogue` to the wrong tree. See
``devtools/checkout_guard.py`` module docstring for the full writeup.
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import cast

import pytest

import devtools.click_dispatch as click_dispatch
import devtools.run_tests as run_tests
import devtools.verify as verify
import polylogue
from devtools.checkout_guard import (
    CheckoutEnvironmentMismatchError,
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
    checkout_environment_fingerprint,
    find_git_worktree_root,
    quick_gate_toolchain_fingerprint,
    resolved_polylogue_path,
)
from devtools.verify_runs import VerifyRun


def test_resolved_polylogue_path_matches_the_running_package() -> None:
    assert resolved_polylogue_path() == Path(polylogue.__file__).resolve()


def test_assert_matches_checkout_passes_for_the_real_checkout() -> None:
    repo_root = Path(polylogue.__file__).resolve().parents[1]
    fingerprint = assert_polylogue_matches_checkout(repo_root, context="test")
    assert fingerprint.polylogue_import_path == Path(polylogue.__file__).resolve()


def test_assert_matches_checkout_raises_for_an_unrelated_root(tmp_path: Path) -> None:
    """The core failure mode: repo_root doesn't contain the resolved package."""
    with pytest.raises(CheckoutImportMismatchError) as excinfo:
        assert_polylogue_matches_checkout(tmp_path, context="unit test probe")

    message = str(excinfo.value)
    # Both paths must be named -- this is the whole point of the guard: an
    # agent staring at the failure should immediately see which two trees
    # disagree, not just that "something" is wrong.
    assert str(tmp_path.resolve()) in message
    assert str(Path(polylogue.__file__).resolve()) in message
    assert "unit test probe" in message


def test_assert_matches_checkout_rejects_a_sibling_directory(tmp_path: Path) -> None:
    """A near-miss root (sibling, not ancestor) must still be rejected.

    ``Path.relative_to`` is the load-bearing check; guard against a regression
    that swaps it for a weaker string-prefix comparison, which a sibling path
    sharing a prefix (``/realm/project/polylogue-2`` vs
    ``/realm/project/polylogue``) would incorrectly pass.
    """
    package_root = Path(polylogue.__file__).resolve().parents[1]
    sibling = package_root.parent / f"{package_root.name}-decoy"
    with pytest.raises(CheckoutImportMismatchError):
        assert_polylogue_matches_checkout(sibling, context="sibling probe")


def test_find_git_worktree_root_finds_the_real_polylogue_checkout() -> None:
    """The happy path: cwd inside this actual checkout resolves to its root."""
    repo_root = Path(polylogue.__file__).resolve().parents[1]
    assert find_git_worktree_root(repo_root) == repo_root


def test_find_git_worktree_root_no_ops_for_an_unrelated_git_repo(tmp_path: Path) -> None:
    """Regression (polylogue-373yt): an ordinary, unrelated git repo must not be

    mistaken for "the invoking Polylogue checkout". Before this fix,
    ``find_git_worktree_root`` returned the first ``.git``-containing
    ancestor unconditionally, so `cd ~/some-other-project && polylogue
    --version` found that repo's root, and the mismatch guard fired against
    it (exit 125) for every single command -- a normally pip-installed
    ``polylogue`` invoked from any unrelated git repository was completely
    unusable.

    Mutation this guards against: dropping the
    ``_is_polylogue_checkout_root`` gate from ``find_git_worktree_root`` (i.e.
    reverting to "return the first ``.git`` ancestor, full stop") makes this
    test fail because it would return ``unrelated_repo`` instead of ``None``.
    """
    unrelated_repo = tmp_path / "some-other-project"
    unrelated_repo.mkdir()
    (unrelated_repo / ".git").mkdir()
    # Deliberately no pyproject.toml and no polylogue/ package tree here --
    # this is what makes it "unrelated" rather than a Polylogue checkout.

    nested_cwd = unrelated_repo / "src" / "pkg"
    nested_cwd.mkdir(parents=True)

    assert find_git_worktree_root(nested_cwd) is None


def test_find_git_worktree_root_no_ops_with_no_git_ancestry_at_all(tmp_path: Path) -> None:
    """A plain directory tree with no ``.git`` anywhere -- e.g. a real installed

    ``polylogue`` invocation with no dev checkout in cwd's ancestry at all.
    """
    plain_dir = tmp_path / "no-git-here"
    plain_dir.mkdir()
    assert find_git_worktree_root(plain_dir) is None


def test_click_dispatch_main_refuses_on_checkout_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``devtools`` (the CLI entry point) refuses before dispatching any command."""

    def _boom(repo_root: Path, *, context: str) -> Path:
        raise CheckoutImportMismatchError(
            f"{context}: `import polylogue` resolved OUTSIDE this checkout.\n"
            f"  invoking checkout : {repo_root}\n"
            "  resolved package  : /some/other/checkout/polylogue/__init__.py\n"
        )

    monkeypatch.setattr(click_dispatch, "assert_polylogue_matches_checkout", _boom)
    exit_code = click_dispatch.main(["status", "--json"])
    assert exit_code == 125
    captured = capsys.readouterr()
    assert "resolved OUTSIDE this checkout" in captured.err
    # And critically: it must not have produced the command's normal JSON
    # output -- the whole point is refusing *before* doing any real work.
    assert captured.out == ""


def test_run_tests_main_refuses_on_checkout_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(repo_root: Path, *, context: str) -> Path:
        raise CheckoutImportMismatchError(f"{context}: mismatch against {repo_root}")

    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", _boom)
    exit_code = run_tests.main(["tests/unit/devtools/test_checkout_guard.py"])
    assert exit_code == 125
    assert "mismatch against" in capsys.readouterr().err


def test_verify_main_refuses_on_checkout_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(repo_root: Path, *, context: str) -> Path:
        raise CheckoutImportMismatchError(f"{context}: mismatch against {repo_root}")

    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", _boom)
    exit_code = verify.main(["--quick"])
    assert exit_code == 125
    assert "mismatch against" in capsys.readouterr().err


def _fake_linked_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "lane"
    root.mkdir()
    (root / ".git").write_text("gitdir: /main/.git/worktrees/lane\n")
    return root


def test_checkout_environment_fingerprint_accepts_clean_linked_worktree(tmp_path: Path) -> None:
    root = _fake_linked_checkout(tmp_path)
    fingerprint = checkout_environment_fingerprint(
        root,
        polylogue_import_path=root / "polylogue" / "__init__.py",
        python_executable=root / ".venv" / "bin" / "python",
    )

    assert fingerprint.clean
    assert fingerprint.linked_worktree is True
    assert fingerprint.python_environment_root == root


def test_checkout_guard_leaves_derived_native_testmon_state_for_verify_to_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fake_linked_checkout(tmp_path)
    data = root / ".cache" / "testmon" / "testmondata"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"interrupted or invalid derived SQLite state")
    package_path = root / "polylogue" / "__init__.py"
    monkeypatch.setattr("devtools.checkout_guard.resolved_polylogue_path", lambda: package_path)

    fingerprint = assert_polylogue_matches_checkout(
        root,
        context="plain verify repair",
        python_executable=root / ".venv" / "bin" / "python",
    )

    assert fingerprint.clean
    assert data.read_bytes().startswith(b"interrupted")


def test_checkout_preflight_reports_runtime_provenance_but_not_testmon_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fake_linked_checkout(tmp_path)
    main = tmp_path / "main-checkout"
    main_python = main / ".venv" / "bin" / "python"
    main_python.parent.mkdir(parents=True)
    (root / ".venv").mkdir()
    (root / "node_modules").mkdir()
    data = root / ".cache" / "testmon" / "testmondata"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"repairable")
    (root / ".cache" / "verify").mkdir(parents=True)
    verify_marker = root / ".cache" / "verify" / "current-run.json"
    verify_marker.write_text(json.dumps({"checkout_root": str(main)}))
    package_path = root / "polylogue" / "__init__.py"
    monkeypatch.setattr("devtools.checkout_guard.resolved_polylogue_path", lambda: package_path)

    with pytest.raises(CheckoutEnvironmentMismatchError) as excinfo:
        assert_polylogue_matches_checkout(root, context="lane", python_executable=main_python)

    message = str(excinfo.value)
    assert str(main_python) in message
    assert str(root / ".venv") in message
    assert str(root / "node_modules") in message
    assert str(verify_marker) in message
    assert str(data) not in message


def test_verify_run_persists_environment_fingerprint(tmp_path: Path) -> None:
    root = _fake_linked_checkout(tmp_path)
    (root / "node_modules").mkdir()
    fingerprint = checkout_environment_fingerprint(
        root,
        polylogue_import_path=root / "polylogue" / "__init__.py",
        python_executable=Path("/usr/bin/python"),
    )
    run = VerifyRun(
        tier="environment-fingerprint",
        argv=["--quick"],
        git_head="head",
        root=root,
        environment_fingerprint=fingerprint.as_dict(),
    )

    payload = json.loads((run.run_dir / "run.json").read_text())
    assert fingerprint.artifacts
    assert payload["environment_fingerprint"] == fingerprint.as_dict()
    assert payload["checkout_root"] == str(root.resolve())


def test_verify_run_marker_is_attributable_without_a_fingerprint(tmp_path: Path) -> None:
    root = _fake_linked_checkout(tmp_path)
    run = VerifyRun(tier="legacy-marker", argv=[], git_head=None, root=root)

    fingerprint = checkout_environment_fingerprint(
        root,
        polylogue_import_path=root / "polylogue" / "__init__.py",
        python_executable=root / ".venv" / "bin" / "python",
    )

    assert fingerprint.clean
    assert fingerprint.verify_state_origin == root.resolve()
    assert json.loads((root / ".cache" / "verify" / "current-run.json").read_text())["run_id"] == run.run_id


_MYPY_WRAPPER_BODY = (
    "# -*- coding: utf-8 -*-\n"
    "import sys\n"
    "from mypy.__main__ import console_entry\n"
    "if __name__ == '__main__':\n"
    "    sys.exit(console_entry())\n"
)


def _write_console_script(path: Path, *, shebang: str, body: str = _MYPY_WRAPPER_BODY) -> None:
    """Write a pip/uv-style console-script wrapper: a shebang line naming the
    owning venv's own interpreter, followed by boilerplate identical across
    every install of the same package version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{shebang}\n{body}", encoding="utf-8")
    path.chmod(0o755)


def _executables(fingerprint: dict[str, object]) -> dict[str, dict[str, str | None]]:
    return cast("dict[str, dict[str, str | None]]", fingerprint["executables"])


def test_quick_gate_toolchain_fingerprint_ignores_relocated_wrapper_shebang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two functionally identical mypy installs, resolved from two different
    venv roots, must fingerprint identically.

    Real pip/uv-generated console-script wrappers for the same package
    version share an identical body but hardcode the absolute path to their
    *own* venv's interpreter in the leading ``#!`` line -- an install-location
    accident (worktree-local `.venv` vs. a shared checkout's `.venv`, both
    synced from the same lockfile), not a genuine toolchain difference. A
    fingerprint that fails to ignore this defeats pre-push receipt reuse
    every time a lane transitions between a borrowed and a dedicated venv,
    even though nothing about what would actually run changed.
    """
    venv_a_root = tmp_path / "venv_a"
    venv_b_root = tmp_path / "venv_b"
    mypy_a = venv_a_root / ".venv" / "bin" / "mypy"
    mypy_b = venv_b_root / ".venv" / "bin" / "mypy"
    _write_console_script(mypy_a, shebang=f"#!{venv_a_root}/.venv/bin/python")
    _write_console_script(mypy_b, shebang=f"#!{venv_b_root}/.venv/bin/python")

    monkeypatch.setattr(metadata, "version", lambda name: "2.3.0")

    monkeypatch.setattr(shutil, "which", lambda name: str(mypy_a) if name == "mypy" else None)
    fingerprint_a = quick_gate_toolchain_fingerprint(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: str(mypy_b) if name == "mypy" else None)
    fingerprint_b = quick_gate_toolchain_fingerprint(tmp_path)

    assert _executables(fingerprint_a)["mypy"] == _executables(fingerprint_b)["mypy"]


def test_quick_gate_toolchain_fingerprint_detects_genuine_wrapper_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real change to a console-script wrapper's body (not just its
    venv-root shebang) must still change the fingerprint -- shebang
    normalization must not blind the gate to actual tool drift."""
    venv_root = tmp_path / "venv"
    mypy_path = venv_root / ".venv" / "bin" / "mypy"
    _write_console_script(mypy_path, shebang=f"#!{venv_root}/.venv/bin/python")
    monkeypatch.setattr(metadata, "version", lambda name: "2.3.0")
    monkeypatch.setattr(shutil, "which", lambda name: str(mypy_path) if name == "mypy" else None)
    fingerprint_before = quick_gate_toolchain_fingerprint(tmp_path)

    _write_console_script(
        mypy_path,
        shebang=f"#!{venv_root}/.venv/bin/python",
        body=_MYPY_WRAPPER_BODY + "# tampered\n",
    )
    fingerprint_after = quick_gate_toolchain_fingerprint(tmp_path)

    assert _executables(fingerprint_before)["mypy"]["sha256"] != _executables(fingerprint_after)["mypy"]["sha256"]


def test_quick_gate_toolchain_fingerprint_detects_native_binary_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native/compiled executable (no shebang -- ruff's real shape) with
    genuinely different bytes must still fingerprint differently; shebang
    normalization is a no-op for content that never starts with ``#!``."""
    ruff_path = tmp_path / "venv" / ".venv" / "bin" / "ruff"
    ruff_path.parent.mkdir(parents=True, exist_ok=True)
    ruff_path.write_bytes(b"\x7fELF" + b"\x00" * 32)
    ruff_path.chmod(0o755)
    monkeypatch.setattr(metadata, "version", lambda name: "0.16.3")
    monkeypatch.setattr(shutil, "which", lambda name: str(ruff_path) if name == "ruff" else None)
    fingerprint_before = quick_gate_toolchain_fingerprint(tmp_path)

    ruff_path.write_bytes(b"\x7fELF" + b"\xff" * 32)
    fingerprint_after = quick_gate_toolchain_fingerprint(tmp_path)

    assert _executables(fingerprint_before)["ruff"]["sha256"] != _executables(fingerprint_after)["ruff"]["sha256"]


def _python_fields(fingerprint: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", fingerprint["python"])


def test_quick_gate_toolchain_fingerprint_ignores_relocated_venv_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``python.prefix`` carries the same install-root noise the executable
    identity fix already removed: it is the raw absolute venv root, so a
    worktree's own dedicated ``.venv`` and a shared checkout's ``.venv`` --
    both synced from the same lockfile, functionally identical -- report
    different prefixes and defeat receipt reuse on install location alone.
    """
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(metadata, "version", lambda name: "2.3.0")

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv_a" / ".venv"))
    fingerprint_a = quick_gate_toolchain_fingerprint(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv_b" / ".venv"))
    fingerprint_b = quick_gate_toolchain_fingerprint(tmp_path)

    assert _python_fields(fingerprint_a)["prefix"] == _python_fields(fingerprint_b)["prefix"]


def test_quick_gate_toolchain_fingerprint_detects_genuine_python_build_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalizing the venv-root component of ``prefix`` must not blind the
    gate to a real interpreter/build difference carried by the other
    ``python`` fields (``base_prefix``, ``version``)."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(metadata, "version", lambda name: "2.3.0")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv" / ".venv"))
    monkeypatch.setattr(sys, "base_prefix", "/nix/store/aaaa-python3-3.14.4")
    fingerprint_before = quick_gate_toolchain_fingerprint(tmp_path)

    monkeypatch.setattr(sys, "base_prefix", "/nix/store/bbbb-python3-3.15.0")
    fingerprint_after = quick_gate_toolchain_fingerprint(tmp_path)

    assert _python_fields(fingerprint_before)["prefix"] == _python_fields(fingerprint_after)["prefix"]
    assert _python_fields(fingerprint_before)["base_prefix"] != _python_fields(fingerprint_after)["base_prefix"]
    assert fingerprint_before["python"] != fingerprint_after["python"]
