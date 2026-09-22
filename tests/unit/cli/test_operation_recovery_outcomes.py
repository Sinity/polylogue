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
    """``--list --output-format json`` emits the repository's rows verbatim.

    Anti-vacuity: bypassing the ``--list`` branch, or emitting anything other
    than the repository result (an empty envelope, a summary count, a
    re-derived shape), makes the exact-equality assertion red. The fake audit
    repository returns one interrupted operation with one unknown target, so
    both the operation record and its target list must survive the surface.
    """

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


def test_adjudication_reconciles_pending_continuity_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prepared-but-uncommitted audit command must not block recovery.

    The crash this command exists to recover from can leave a prepared audit
    continuity command behind. Adjudication is itself an audit mutation, so
    without reconciling first it failed with "another audit continuity
    mutation is already pending" -- refusing to recover exactly the operation
    the operator came here to close.

    Anti-vacuity: removing the `reconcile_continuity()` call makes `calls`
    start with `adjudicate` and the stub raises that pending error, so the
    command exits non-zero.
    """
    calls: list[str] = []

    class _Audit:
        def reconcile_continuity(self) -> None:
            calls.append("reconcile")

        def adjudicate_recovery(self, operation_id: str, **_: object) -> None:
            calls.append("adjudicate")
            if calls[0] != "reconcile":
                raise ValueError("another audit continuity mutation is already pending")

        def get_operation(self, operation_id: str) -> dict[str, object]:
            return {"operation_id": operation_id, "status": "recovered"}

        def list_events(self, operation_id: str) -> tuple[object, ...]:
            return ()

        def list_targets(self, operation_id: str) -> tuple[dict[str, object], ...]:
            return ({"target_ref": "session:one", "state": "applied"},)

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._operation_recovery.AuditRepository.for_archive_root",
        lambda root: _Audit(),
    )
    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._operation_recovery.offline_maintenance_block_reason",
        lambda *_args, **_kwargs: None,
    )
    env = SimpleNamespace(config=SimpleNamespace(archive_root=Path("/archive")))
    result = CliRunner().invoke(
        _operation_recovery.operation_recovery_command,
        [
            "--operation-id",
            "operation:interrupted",
            "--target-outcome",
            "session:one=applied",
            "--reason",
            "operator evidence",
            "--confirm",
            "--output-format",
            "json",
        ],
        obj=env,
    )

    assert result.exit_code == 0, result.output
    assert calls == ["reconcile", "adjudicate"]


def test_inspection_never_performs_the_durable_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opposite direction: a bare inspection stays a read."""
    calls: list[str] = []

    class _Audit:
        def reconcile_continuity(self) -> None:
            calls.append("reconcile")

        def get_operation(self, operation_id: str) -> dict[str, object]:
            return {"operation_id": operation_id, "status": "interrupted"}

        def list_events(self, operation_id: str) -> tuple[object, ...]:
            return ()

        def list_targets(self, operation_id: str) -> tuple[dict[str, object], ...]:
            return ()

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance._operation_recovery.AuditRepository.for_archive_root",
        lambda root: _Audit(),
    )
    env = SimpleNamespace(config=SimpleNamespace(archive_root=Path("/archive")))
    result = CliRunner().invoke(
        _operation_recovery.operation_recovery_command,
        ["--operation-id", "operation:interrupted", "--output-format", "json"],
        obj=env,
    )

    assert result.exit_code == 0, result.output
    assert calls == []
