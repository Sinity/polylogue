"""Saved-view name normalization preserves exact mutation and replay scope."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from polylogue.operations.audit import AuditRepository
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.facade_mutations import facade_save_view
from polylogue.operations.mutation_actuators import SavedViewSaveActuator, SavedViewSaveArgs
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import assertion_id_for_saved_view
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.operation_recovery import recover_on_admitted_owner
from tests.unit.operations.test_mutation_actuators import _seed_archive_session
from tests.unit.operations.test_mutation_crash_recovery import _crash_mid_mutation, _principal, _Scenario


def test_facade_saved_view_authorizes_the_normalized_name_collision(tmp_path: Path) -> None:
    """Removing prepare's normalization drops the tombstoned view from authority."""
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.save_view("existing", "Shared", '{"query":"old"}')

    principal = _principal(runtime_operation_binding(SavedViewSaveActuator())).on_surface("api")
    context = SimpleNamespace(runtime=object(), archive_root=root, principal=principal)
    request = SimpleNamespace(
        operation="mutation.facade.save_view",
        payload={"view_id": "new", "name": " Shared ", "query_json": '{"query":"new"}', "watch": False},
    )
    audit = AuditRepository.for_archive_root(root)
    with write_lease("test.saved-view.scope", archive_root=root):
        result = facade_save_view(request, context, audit, None)
    operation_id = result["receipt_ref"].split(":", 1)[1]
    with audit.settled_machine_read():
        plan = audit.operation_plan(operation_id)
    assert plan.target_refs == ("saved_view:new", "saved_view:existing")
    assert plan.context["name"] == "Shared"
    assert plan.context["collision_view_id"] == "existing"
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        assert archive.get_view("existing") is None
        assert archive.get_view("new")["name"] == "Shared"


@pytest.mark.parametrize("crash", ("before-apply", "after-apply"))
def test_saved_view_padded_name_recovery_preserves_the_committed_effect(
    tmp_path: Path, crash: Literal["before-apply", "after-apply"]
) -> None:
    """The admitted canonical name also drives recovery's exact-effect inspection."""
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.save_view("existing", "Shared", '{"query":"old"}')
    scenario = _Scenario(
        "padded-view",
        SavedViewSaveActuator(),
        lambda _root, archive: SavedViewSaveArgs(archive, "new", " Shared ", '{"query":"new"}'),
        lambda _root, archive: archive.get_view("new") is not None,
    )
    operation_id = _crash_mid_mutation(root, scenario, crash)
    assertion_id = assertion_id_for_saved_view("new")
    if crash == "after-apply":
        with sqlite3.connect(root / "user.db") as conn:
            conn.execute("UPDATE assertions SET updated_at_ms=1234 WHERE assertion_id=?", (assertion_id,))
    recover_on_admitted_owner(root)
    with sqlite3.connect(root / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id=?", (operation_id,)
        ).fetchone() == ("completed", "recovered_complete")
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        assert archive.get_view("new")["name"] == "Shared"
        assert archive.get_view("existing") is None
    if crash == "after-apply":
        with sqlite3.connect(root / "user.db") as conn:
            assert conn.execute(
                "SELECT updated_at_ms FROM assertions WHERE assertion_id=?", (assertion_id,)
            ).fetchone() == (1234,)
