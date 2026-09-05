from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.daemon.maintenance_registry_http import handle_status
from polylogue.maintenance.operation_ids import validate_operation_id
from polylogue.maintenance.registry import MaintenanceOperationRegistry


@pytest.mark.parametrize("value", ["", "/tmp/escape", "../escape", "..", ".", "a/b", r"a\\b", 0, False, [], Path("op")])
def test_validate_operation_id_rejects_untrusted_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_operation_id(value)


def test_validate_operation_id_preserves_opaque_ids() -> None:
    assert validate_operation_id("operation:abc-123") == "operation:abc-123"


def test_validate_operation_id_rejects_none_generation_sentinel() -> None:
    with pytest.raises(ValueError):
        validate_operation_id(None)


def test_registry_rejects_hostile_id_before_path_lookup(tmp_path: Path) -> None:
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    registry = MaintenanceOperationRegistry(config=config)
    with pytest.raises(ValueError):
        registry.get_operation("../escape")


def test_daemon_status_rejects_url_decoded_hostile_id() -> None:
    class Handler:
        def __init__(self) -> None:
            self.error: tuple[object, ...] | None = None

        def _send_error(self, *args: object) -> None:
            self.error = args

    handler = Handler()
    handle_status(handler, "../escape")
    assert handler.error is not None
    assert handler.error[0] is HTTPStatus.BAD_REQUEST
    assert handler.error[1] == "invalid_operation_id"
