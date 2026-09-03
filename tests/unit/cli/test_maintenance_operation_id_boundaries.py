"""Fail-closed operation-ID validation at maintenance Click boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.maintenance import _rebuild_index_status as rebuild_module
from polylogue.cli.commands.maintenance import _status as status_module
from polylogue.cli.shared.types import AppEnv

INVALID_OPERATION_ID = "../not-an-opaque-id"


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
    assert "operation_id must not contain path separators" in result.output


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
    assert "operation_id must not contain path separators" in result.output
