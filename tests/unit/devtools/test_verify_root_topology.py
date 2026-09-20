"""The root-topology gate must catch a non-kernel module at ``polylogue/``.

``devtools/verify_topology.py`` enforced this until it was deleted with the
per-file placement-judgment machinery it was entangled with (3d65277de,
#3653); nothing has refused a root stray since (polylogue-8a060).

Anti-vacuity: empty ``ROOT_KERNEL`` and the check goes green on any input --
that is the vacuous state, so ``test_a_stray_root_module_is_reported`` seeds
``scratch_helper.py`` into a synthetic root and requires it by name, and
``test_gate_passes_on_this_checkout`` requires the real eleven-module root to
be recognized rather than trivially empty. Deleting the stale-entry check
lets the allowlist keep a name after its module moved, which
``test_a_stale_allowlist_entry_is_reported`` catches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.gate import GATES_BY_NAME
from devtools.verify_root_topology import (
    PACKAGE_ROOT,
    ROOT_KERNEL,
    main,
    root_modules,
    stale_allowlist_entries,
    strays,
)


def _kernel_root(tmp_path: Path) -> Path:
    """A synthetic package root holding exactly the declared kernel modules."""
    for name in ROOT_KERNEL:
        (tmp_path / name).write_text("", encoding="utf-8")
    return tmp_path


def test_gate_is_registered_against_its_module() -> None:
    gate = GATES_BY_NAME["root-topology"]
    assert gate.args == ("devtools.verify_root_topology",)
    assert gate.kind == "module"
    assert gate.in_quick is True
    assert gate.blocking is True


def test_a_kernel_only_root_reports_nothing(tmp_path: Path) -> None:
    root = _kernel_root(tmp_path)

    assert strays(root) == []
    assert stale_allowlist_entries(root) == []


def test_a_stray_root_module_is_reported(tmp_path: Path) -> None:
    """The concrete input the bead names: an undeclared file at the root."""
    root = _kernel_root(tmp_path)
    (root / "scratch_helper.py").write_text("", encoding="utf-8")

    assert strays(root) == ["scratch_helper.py"]


def test_a_stale_allowlist_entry_is_reported(tmp_path: Path) -> None:
    """A declared kernel module that moved away must not stay declared."""
    root = _kernel_root(tmp_path)
    (root / "config.py").unlink()

    assert stale_allowlist_entries(root) == ["config.py"]
    assert strays(root) == []


def test_the_inventory_is_flat_and_carries_no_placement_projection(tmp_path: Path) -> None:
    """The declaration is a name-to-owner map, not a per-file target record.

    #3653 deleted the per-file target/reason/owner projection; rebuilding it
    is the failure mode this bead must not repeat.
    """
    root = _kernel_root(tmp_path)
    (root / "nested").mkdir()
    (root / "nested" / "module.py").write_text("", encoding="utf-8")

    assert root_modules(root) == sorted(ROOT_KERNEL)
    assert all(owner.strip() for owner in ROOT_KERNEL.values())


def test_gate_passes_on_this_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    """The real package root satisfies the invariant the gate declares."""
    assert main([]) == 0
    assert "no strays" in capsys.readouterr().out
    # Not vacuous: the real root actually holds the declared kernel modules.
    assert set(root_modules(PACKAGE_ROOT)) == set(ROOT_KERNEL)
