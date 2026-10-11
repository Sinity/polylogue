"""Retry the real frontier when its committed condition changes."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon import cli as daemon_cli
from polylogue.operations import raw_frontier_inspection
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.connection_profile import owned_daemon_connection
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


@pytest.mark.asyncio
@pytest.mark.frozen_clock_modules(
    "polylogue.daemon.cli",
    "polylogue.sources.live.cursor",
    "polylogue.sources.live.convergence_debt_retry",
)
@pytest.mark.parametrize("failed,unavailable", [(False, False), (True, False), (False, True)])
async def test_changed_frontier_releases_only_its_deferred_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: Any,
    failed: bool,
    unavailable: bool,
) -> None:
    """A committed cursor condition invalidates the real blocked mark.

    Unchanged waiting must not run another full inspection. Relevant progress
    should release deferred frontier eligibility, while true failures and
    another stage keep their deadlines. All inspection and ledger work uses
    the real preparation, compute and writer owners.
    """
    root = tmp_path / "archive"
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))
    calls = 0
    real_inspect = raw_frontier_inspection.inspect_prepared_raw_authority_frontier

    def inspect(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(raw_frontier_inspection, "inspect_prepared_raw_authority_frontier", inspect)
    async with prepared_live_convergence_owner(root) as owner:
        cursor = await owner._write_coordinator.run_sync("fixture.frontier.cursor", CursorStore, root / "index.db")
        stage = raw_frontier_inspection.make_raw_frontier_inspection_stage(
            root / "index.db",
            compute_adapter=owner._compute_adapter,
        )
        monkeypatch.setattr(
            "polylogue.daemon.convergence_stages.make_default_convergence_stages",
            lambda _db, **_kwargs: (stage,),
        )

        def change_cursor(remove: bool) -> None:
            with owned_daemon_connection(root / "ops.db", archive_root=root) as conn, conn:
                query = (
                    "DELETE FROM ingest_cursor WHERE source_path=?"
                    if remove
                    else "INSERT INTO ingest_cursor(source_path,byte_offset,updated_at_ms) VALUES(?,0,1)"
                )
                with closing(conn.execute(query, ("neutral/missing.jsonl",))):
                    pass

        async def drain() -> None:
            await owner.run_convergence_sync(
                "fixture.frontier.debt",
                daemon_cli._drain_convergence_debt_and_archive,
                root / "index.db",
                compute_adapter=owner._compute_adapter,
            )

        await owner._write_coordinator.run_sync("fixture.frontier.gap", change_cursor, False)
        await drain()
        debt = cursor.list_convergence_debt(stage=stage.name)[0]
        assert debt.status == "deferred" and debt.failure_count == 1
        assert debt.next_retry_at is not None
        assert datetime.fromisoformat(debt.next_retry_at).timestamp() == frozen_clock.time() + 60
        assert calls == 1
        unchanged = raw_frontier_inspection.frontier_coverage_for_archive(root)
        assert unchanged["available"] and unchanged["current"] and not unchanged["healthy"]

        frozen_clock.advance(10)
        await drain()
        assert calls == 1 and cursor.list_convergence_debt(stage=stage.name)[0] == debt
        frozen_clock.advance(50)
        await drain()
        debt = cursor.list_convergence_debt(stage=stage.name)[0]
        assert calls == 2 and debt.failure_count == 2
        assert debt.next_retry_at is not None
        deadline = datetime.fromisoformat(debt.next_retry_at).timestamp()
        assert deadline == frozen_clock.time() + 120
        if failed:
            await owner._write_coordinator.run_sync(
                "fixture.frontier.failure",
                cursor.record_convergence_debt,
                stage=stage.name,
                subject_type=debt.subject_type,
                subject_id=debt.subject_id,
                error="neutral physical failure",
                deferred=False,
            )
            debt = cursor.list_convergence_debt(stage=stage.name)[0]
            assert debt.status == "failed"

        await owner._write_coordinator.run_sync(
            "fixture.frontier.other-stage",
            cursor.record_convergence_debt,
            stage="neutral_other_stage",
            subject_type="source_path",
            subject_id="neutral/other",
            error="stage state: pending",
            deferred=True,
        )
        other = cursor.list_convergence_debt(stage="neutral_other_stage")[0]
        frozen_clock.advance(10)
        await owner._write_coordinator.run_sync("fixture.frontier.condition", change_cursor, True)
        changed = raw_frontier_inspection.frontier_coverage_for_archive(root)
        assert changed["available"] and not changed["current"] and not changed["healthy"]
        if unavailable:
            monkeypatch.setattr(
                raw_frontier_inspection,
                "frontier_coverage_for_archive",
                lambda _root: {"available": False, "current": False, "healthy": False},
            )
        await drain()
        after = cursor.list_convergence_debt(stage=stage.name)
        assert cursor.list_convergence_debt(stage="neutral_other_stage") == [other]
        if failed or unavailable:
            assert calls == 2 and after == [debt]
        else:
            assert calls == 3 and after == []
            current = raw_frontier_inspection.frontier_coverage_for_archive(root)
            assert current["current"] and current["healthy"]
