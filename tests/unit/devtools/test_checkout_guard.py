"""Tests for the retained Polylogue import-root assertion."""

from __future__ import annotations

from pathlib import Path

import pytest

import devtools.click_dispatch as click_dispatch
import devtools.run_tests as run_tests
import devtools.verify as verify
import polylogue
from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
    find_git_worktree_root,
    resolved_polylogue_path,
)


def test_resolved_polylogue_path_matches_the_running_package() -> None:
    assert resolved_polylogue_path() == Path(polylogue.__file__).resolve()


def test_assert_matches_checkout_passes_for_the_real_checkout() -> None:
    repo_root = Path(polylogue.__file__).resolve().parents[1]

    assert assert_polylogue_matches_checkout(repo_root, context="test") == Path(polylogue.__file__).resolve()


def test_assert_matches_checkout_raises_for_an_unrelated_root(tmp_path: Path) -> None:
    with pytest.raises(CheckoutImportMismatchError) as excinfo:
        assert_polylogue_matches_checkout(tmp_path, context="unit test probe")

    message = str(excinfo.value)
    assert str(tmp_path.resolve()) in message
    assert str(Path(polylogue.__file__).resolve()) in message
    assert "unit test probe" in message


def test_assert_matches_checkout_rejects_a_sibling_directory() -> None:
    """Containment must use path semantics, never a string-prefix comparison."""
    package_root = Path(polylogue.__file__).resolve().parents[1]
    sibling = package_root.parent / f"{package_root.name}-decoy"

    with pytest.raises(CheckoutImportMismatchError):
        assert_polylogue_matches_checkout(sibling, context="sibling probe")


def test_find_git_worktree_root_finds_the_real_polylogue_checkout() -> None:
    repo_root = Path(polylogue.__file__).resolve().parents[1]

    assert find_git_worktree_root(repo_root) == repo_root


def test_find_git_worktree_root_no_ops_for_an_unrelated_git_repo(tmp_path: Path) -> None:
    unrelated_repo = tmp_path / "some-other-project"
    unrelated_repo.mkdir()
    (unrelated_repo / ".git").mkdir()
    nested_cwd = unrelated_repo / "src" / "pkg"
    nested_cwd.mkdir(parents=True)

    assert find_git_worktree_root(nested_cwd) is None


def test_find_git_worktree_root_no_ops_with_no_git_ancestry_at_all(tmp_path: Path) -> None:
    plain_dir = tmp_path / "no-git-here"
    plain_dir.mkdir()

    assert find_git_worktree_root(plain_dir) is None


def test_click_dispatch_main_refuses_on_checkout_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(repo_root: Path, *, context: str) -> Path:
        raise CheckoutImportMismatchError(
            f"{context}: `import polylogue` resolved OUTSIDE this checkout.\n"
            f"  invoking checkout : {repo_root}\n"
            "  resolved package  : /some/other/checkout/polylogue/__init__.py\n"
        )

    monkeypatch.setattr(click_dispatch, "assert_polylogue_matches_checkout", _boom)

    assert click_dispatch.main(["status", "--json"]) == 125

    captured = capsys.readouterr()
    assert "resolved OUTSIDE this checkout" in captured.err
    assert captured.out == ""


def test_run_tests_main_refuses_on_checkout_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(repo_root: Path, *, context: str) -> Path:
        raise CheckoutImportMismatchError(f"{context}: mismatch against {repo_root}")

    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", _boom)

    assert run_tests.main(["tests/unit/devtools/test_checkout_guard.py"]) == 125
    assert "mismatch against" in capsys.readouterr().err


def test_verify_main_refuses_before_creating_receipt_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifier startup must validate the retained import-root contract first."""
    cache = tmp_path / ".cache" / "verify"

    def _reject_after_observing_clean_cache(repo_root: Path, *, context: str) -> Path:
        assert repo_root == tmp_path
        assert context == "devtools verify"
        assert not cache.exists()
        raise CheckoutImportMismatchError("simulated checkout mismatch")

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", _reject_after_observing_clean_cache)

    assert verify.main(["--quick"]) == 125
    assert not cache.exists()
    assert "simulated checkout mismatch" in capsys.readouterr().err
