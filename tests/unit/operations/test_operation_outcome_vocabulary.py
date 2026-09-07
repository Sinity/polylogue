"""One vocabulary names an operation's terminal state.

``OperationStatus`` is the canonical enum: it backs ``operation_runs.status``
in the durable audit tier and it is what the daemon operation envelope puts on
the wire. A second enum over the same states is what let a daemon report
``complete`` while the audit row said ``completed``.

Anti-vacuity: reintroducing a protocol-local outcome enum — any StrEnum in the
operation or surface layers whose members cover the same terminal states — turns
:func:`test_no_second_terminal_state_enum_exists` red.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from polylogue.core.enums import OperationStatus
from polylogue.operations.daemon_protocol import DaemonOperationEnvelope

_TERMINAL_STATES = {"complete", "completed", "failed", "cancelled", "interrupted", "indeterminate"}
_SCANNED = (Path("polylogue/operations"), Path("polylogue/surfaces"), Path("polylogue/cli"))


def test_envelope_outcome_is_the_canonical_operation_status() -> None:
    envelope = DaemonOperationEnvelope(
        operation="status",
        archive={},
        generation={},
        readiness={},
        authority={},
        progress={},
    )
    assert envelope.outcome is OperationStatus.COMPLETED
    assert envelope.to_dict()["outcome"] == "completed"


def test_interrupted_is_the_only_cancelled_terminal_state() -> None:
    """The wire has no separate ``cancelled`` member to diverge from."""
    members = {member.value for member in OperationStatus}
    assert "interrupted" in members
    assert "cancelled" not in members


def _enum_members(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
        bases |= {base.attr for base in node.bases if isinstance(base, ast.Attribute)}
        if not bases & {"StrEnum", "PolylogueStrEnum", "Enum"}:
            continue
        values: set[str] = set()
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                values.add(statement.value.value)
        found[node.name] = values
    return found


@pytest.mark.parametrize("root", _SCANNED, ids=lambda path: path.name)
def test_no_second_terminal_state_enum_exists(root: Path) -> None:
    """No enum outside ``core.enums`` may describe operation terminal states."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for name, values in _enum_members(path).items():
            overlap = {value for value in values if value in _TERMINAL_STATES}
            if len(overlap) >= 2:
                offenders.append(f"{path}:{name} covers {sorted(overlap)}")
    assert offenders == [], offenders
