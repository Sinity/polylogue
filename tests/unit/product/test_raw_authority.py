from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.config import Config
from polylogue.product import raw_authority


def test_apply_frontier_rejects_incoherent_actuator_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )
    response = SimpleNamespace(
        census_id="apply-census-1",
        preview_census_id="preview-census-1",
        selected_plan_count=1,
        executed_plan_count=2,
        retryable_plan_count=0,
        post_inventory_digest="digest",
        post_plan_count=0,
        outcome_refs=("detail",),
    )
    monkeypatch.setattr(
        "polylogue.storage.raw_reconciler.apply_raw_authority_frontier",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ValueError, match="incoherent plan counts"):
        raw_authority.apply_frontier(
            config,
            preview_census_id="preview-census-1",
            selected_plan_ids=("safe-1",),
        )
