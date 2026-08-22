"""Fail-closed operation-ID validation at maintenance Click boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.maintenance import _raw_authority_recovery as raw_module
from polylogue.cli.commands.maintenance import _rebuild_index_status as rebuild_module
from polylogue.cli.commands.maintenance import _status as status_module
from polylogue.cli.shared.types import AppEnv

INVALID_OPERATION_ID = "../escape"


def test_status_rejects_invalid_operation_id_before_registry_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(status_module, "archive_root", lambda: tmp_path)

    def fail_lookup(*args: object, **kwargs: object) -> object:
        raise AssertionError("registry lookup must not run for invalid operation IDs")

    from polylogue.maintenance import registry

    monkeypatch.setattr(registry.MaintenanceOperationRegistry, "get_operation", fail_lookup)
    result = CliRunner().invoke(
        status_module.status_command,
        ["--operation-id", INVALID_OPERATION_ID],
        obj=AppEnv(),
    )

    assert result.exit_code != 0
    assert result.exception is not None


def test_rebuild_index_status_rejects_invalid_operation_id_before_status_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rebuild_module, "archive_root", lambda: tmp_path)

    def fail_status(*args: object, **kwargs: object) -> object:
        raise AssertionError("status lookup must not run for invalid operation IDs")

    from polylogue.maintenance import rebuild_index

    monkeypatch.setattr(rebuild_index, "rebuild_status", fail_status)
    result = CliRunner().invoke(
        rebuild_module.rebuild_index_status_command,
        ["--operation-id", INVALID_OPERATION_ID],
    )

    assert result.exit_code != 0
    assert result.exception is not None


def test_raw_authority_recovery_rejects_invalid_operation_id_before_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_recovery(*args: object, **kwargs: object) -> object:
        raise AssertionError("recovery execution must not run for invalid operation IDs")

    monkeypatch.setattr(raw_module, "inspect_raw_authority_recovery", fail_recovery)
    monkeypatch.setattr(raw_module, "resume_raw_authority_recovery", fail_recovery)
    result = CliRunner().invoke(
        raw_module.raw_authority_recovery_command,
        ["--operation", "reset", "--operation-id", INVALID_OPERATION_ID],
    )

    assert result.exit_code != 0
    assert result.exception is not None
