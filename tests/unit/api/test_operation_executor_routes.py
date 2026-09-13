"""Production facade routes for derived maintenance mutations.

These tests deliberately enter the public async facade. They fail if a route
stops constructing the real actuator or bypasses ``OperationExecutor`` and
calls the storage rebuild primitive directly.

Session-insight maintenance is the deliberate exception: it has no facade
execution route at all. The facade must refuse it with a typed, actionable
error and must never reach ``OperationExecutor`` -- see the test below.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.errors import InsightMaintenanceRequiresDaemonError
from polylogue.operations.mutation_transaction import MutationTransactionError, OperationExecutor
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _seed_archive(archive_root: Path, *, native_id: str) -> str:
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    raw_id = f"raw-{native_id}"
    session_id = f"codex-session:{native_id}"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, 1000)
            """,
            (raw_id, native_id, str(archive_root / f"{native_id}.jsonl")),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)
            """,
            (native_id, raw_id, f"Maintenance route {native_id}"),
        )
    return session_id


@pytest.mark.asyncio
async def test_facade_rebuild_insights_refuses_without_entering_the_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The facade names the sealed owner instead of staging execution authority.

    An insight sweep is authorized page by page: the scope is frozen into a
    manifest, staged as immutable previews, sealed into accepted parts, and
    started one ordinal at a time. A library process holds none of that
    authority, so the facade refuses up front.

    Anti-vacuity: if the facade is ever rewired back onto the generic
    prepare/authorize/execute path, ``calls`` records ``mutate-rebuild-insights``
    and the raised type is ``MutationTransactionError`` (the internal guard
    leaking to a public caller) rather than the typed refusal -- both assertions
    go red. Staging a preview before refusing also goes red, because
    ``prepare_bound_for_archive`` would have run.
    """

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_id = _seed_archive(archive_root, native_id="route-insights")
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    calls: list[str] = []
    original_execute_bound = OperationExecutor.execute_bound
    original_prepare_bound = OperationExecutor.prepare_bound_for_archive

    def record_execute_bound(self: OperationExecutor, binding, preview, authorization, args):  # type: ignore[no-untyped-def]
        calls.append(binding.actuator.operation)
        return original_execute_bound(self, binding, preview, authorization, args)

    def record_prepare_bound(self: OperationExecutor, binding, args, principal, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(f"prepare:{binding.actuator.operation}")
        return original_prepare_bound(self, binding, args, principal, **kwargs)

    monkeypatch.setattr(OperationExecutor, "execute_bound", record_execute_bound)
    monkeypatch.setattr(OperationExecutor, "prepare_bound_for_archive", record_prepare_bound)
    try:
        with pytest.raises(InsightMaintenanceRequiresDaemonError) as raised:
            await archive.rebuild_insights(session_ids=[session_id])
    finally:
        await archive.close()

    assert "maintenance.insights.rebuild" in str(raised.value)
    assert "polylogued run" in str(raised.value)
    assert not isinstance(raised.value, MutationTransactionError)
    assert calls == []
