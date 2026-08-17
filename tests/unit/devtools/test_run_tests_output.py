"""Contracts for how much `devtools test` says on an ordinary run.

A wrapper that is noisier than the thing it wraps gets bypassed, and a bypassed
wrapper takes the checkout guard, containment and receipts with it. The property
here is that diagnostic material is suppressed only where it is uninformative --
never where it would change what the reader does.
"""

from __future__ import annotations

from pathlib import Path

from devtools.run_tests import ROOT, _import_path_is_unsurprising


def test_package_inside_this_checkout_is_unsurprising() -> None:
    assert _import_path_is_unsurprising(ROOT / "polylogue" / "__init__.py")


def test_package_from_another_checkout_is_surprising_and_must_be_announced() -> None:
    """The 2026-07-31 incident: a worktree silently ran the main checkout's code.

    That is precisely when the resolved-package line must survive the quieting,
    because it contradicts the checkout the reader believes they are testing.
    """
    assert not _import_path_is_unsurprising(Path("/realm/project/polylogue-other/polylogue/__init__.py"))


def test_unresolvable_path_is_treated_as_surprising() -> None:
    """Fail toward announcing. An unreadable path is not evidence of correctness."""
    assert not _import_path_is_unsurprising("")
