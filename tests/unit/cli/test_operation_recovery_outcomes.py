"""Unit tests for the operation-recovery CLI's target-outcome parsing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.commands.maintenance import _operation_recovery
from polylogue.cli.commands.maintenance._operation_recovery import _outcomes


def test_outcomes_splits_from_the_last_equals_sign() -> None:
    """A target ref containing "=" must not be mis-split at the first "=".

    ``--target-outcome`` values are ``target_ref=applied|not-applied|unknown``.
    A target ref built from a native id that itself contains "=" must still
    parse to the exact original ref, with only the outcome suffix stripped
    off the end.

    Anti-vacuity: reverting ``rpartition`` to ``partition`` (split at the
    FIRST "=") makes this test fail -- it would instead parse the ref as
    ``session:native`` with a corrupted, truncated identity.
    """

    result = _outcomes(("session:native=id=applied",))
    assert result == {"session:native=id": "applied"}


def test_outcomes_still_rejects_a_plain_malformed_value() -> None:
    with pytest.raises(click.ClickException, match="target_ref=applied"):
        _outcomes(("no-equals-sign",))


def test_outcomes_rejects_unknown_outcome_vocabulary() -> None:
    with pytest.raises(click.ClickException, match="target_ref=applied"):
        _outcomes(("session:fixture=maybe",))


def test_outcomes_parses_ordinary_refs_without_embedded_equals() -> None:
    result = _outcomes(("session:one=applied", "session:two=not-applied", "session:three=unknown"))
    assert result == {"session:one": "applied", "session:two": "not-applied", "session:three": "unknown"}


def test_operation_recovery_list_outputs_unresolved_runs_and_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Audit:
        def __init__(self) -> None:
            self.archive_root: Path | None = None

        def list_recovery_operations(self) -> tuple[dict[str, object], ...]:
            return (
                {
                    "operation": {"operation_id": "operation:interrupted", "status": "interrupted"},
                    "targets": ({"target_ref": "session:one", "state": "unknown"},),
                },
            )

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._operation_recovery.AuditRepository.for_archive_root",
        lambda root: _Audit(),
    )
    env = SimpleNamespace(config=SimpleNamespace(archive_root=Path("/archive")))
    result = CliRunner().invoke(
        _operation_recovery.operation_recovery_command, ["--list", "--output-format", "json"], obj=env
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "operations": [
            {
                "operation": {"operation_id": "operation:interrupted", "status": "interrupted"},
                "targets": [{"target_ref": "session:one", "state": "unknown"}],
            }
        ]
    }
