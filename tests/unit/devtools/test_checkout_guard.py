"""Tests for the worktree-import guard (devtools/checkout_guard.py).

Covers the 2026-07-30/31 hazard: a shared/editable venv's `.pth` entry can
point at a different checkout than the one a tool is actually invoked from
(most concretely, a linked git worktree reusing the main checkout's
`.venv`), silently resolving `import polylogue` to the wrong tree. See
``devtools/checkout_guard.py`` module docstring for the full writeup.
"""

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
    resolved_polylogue_path,
)


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
