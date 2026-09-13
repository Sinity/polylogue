"""Operation debt is visible but cannot replace inspected insight readiness."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.sources.live.cursor import CursorStore
from tests.infra.convergence_harness import (
    build_converged_archive,
    rich_convergence_sources,
)

_STALL_ERROR = "derived stage stalled for the debt-surface fixture"


@pytest.mark.asyncio
async def test_insight_readiness_ignores_poisoned_operation_debt(tmp_path: Path) -> None:
    archive = build_converged_archive(tmp_path / "archive", rich_convergence_sources())
    index_db = archive.root / "index.db"
    ops_db = archive.root / "ops.db"

    polylogue = Polylogue(archive_root=archive.root, db_path=index_db)
    try:
        converged = await polylogue.insight_readiness_report()
        assert converged.converged is True, "a converged archive must report caught-up convergence"
        assert converged.debt_stages == ()

        CursorStore(index_db, ops_db_path=ops_db).record_convergence_debt(
            stage="derived",
            subject_type="session_id",
            subject_id=archive.session_ids[0],
            error=_STALL_ERROR,
        )

        debt_report = await polylogue.insight_readiness_report()
    finally:
        await polylogue.close()

    assert debt_report.converged is True, "a debt row cannot replace inspected derived output"
    assert "derived" in debt_report.debt_stages, "operation health remains visible separately"
