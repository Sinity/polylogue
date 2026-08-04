"""Tests for the worktree-import guard (devtools/checkout_guard.py).

Covers the 2026-07-30/31 hazard: a shared/editable venv's `.pth` entry can
point at a different checkout than the one a tool is actually invoked from
(most concretely, a linked git worktree reusing the main checkout's
`.venv`), silently resolving `import polylogue` to the wrong tree. See
``devtools/checkout_guard.py`` module docstring for the full writeup.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    resolved_polylogue_path,
)
from devtools.verify_runs import VerifyRun


def test_resolved_polylogue_path_matches_the_running_package() -> None:
    assert resolved_polylogue_path() == Path(polylogue.__file__).resolve()


def test_assert_matches_checkout_passes_for_the_real_checkout() -> None:
    repo_root = Path(polylogue.__file__).resolve().parents[1]
    resolved = assert_polylogue_matches_checkout(repo_root, context="test")
    assert resolved == Path(polylogue.__file__).resolve()


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


def test_checkout_preflight_reports_seeded_artifacts_and_main_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_linked_checkout(tmp_path)
    main = tmp_path / "main-checkout"
    main_venv = main / ".venv" / "bin"
    main_venv.mkdir(parents=True)
    main_python = main_venv / "python"
    (root / ".venv").mkdir()
    (root / "node_modules").mkdir()
    (root / ".cache" / "testmon").mkdir(parents=True)
    (root / ".cache" / "testmon" / "seed.json").write_text(json.dumps({"status": "complete"}))
    (root / ".cache" / "verify").mkdir(parents=True)
    verify_marker = root / ".cache" / "verify" / "current-run.json"
    verify_marker.write_text(json.dumps({"checkout_root": str(main)}))
    package_path = root / "polylogue" / "__init__.py"
    monkeypatch.setattr("devtools.checkout_guard.resolved_polylogue_path", lambda: package_path)

    with pytest.raises(CheckoutEnvironmentMismatchError) as excinfo:
        assert_polylogue_matches_checkout(root, context="seeded lane", python_executable=main_python)

    message = str(excinfo.value)
    assert str(main_python) in message
    assert str(root / ".venv") in message
    assert str(root / "node_modules") in message
    assert str(root / ".cache" / "testmon" / "seed.json") in message
    assert str(verify_marker) in message
    assert "direnv allow" in message
    assert "remediation" in message


def test_verify_run_persists_environment_fingerprint(tmp_path: Path) -> None:
    fingerprint = checkout_environment_fingerprint(
        tmp_path,
        polylogue_import_path=tmp_path / "polylogue" / "__init__.py",
        python_executable=Path("/usr/bin/python"),
    )
    run = VerifyRun(
        tier="environment-fingerprint",
        argv=["--quick"],
        git_head="head",
        root=tmp_path,
        environment_fingerprint=fingerprint.as_dict(),
    )

    payload = json.loads((run.run_dir / "run.json").read_text())
    assert payload["environment_fingerprint"] == fingerprint.as_dict()
