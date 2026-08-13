from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.config import Config
from polylogue.product import raw_authority
from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError
from polylogue.storage.raw_authority import raw_authority_detail_query_handle
from polylogue.storage.raw_reconciler import RawAuthorityFrontierApplyReport


def test_apply_frontier_rejects_incoherent_actuator_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("polylogue.daemon.write_coordinator.daemon_write_lease_active", lambda: True)
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


def test_apply_frontier_rejects_untyped_actuator_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("polylogue.daemon.write_coordinator.daemon_write_lease_active", lambda: True)
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )
    response = SimpleNamespace(
        selected_plan_count=1,
        executed_plan_count=1,
        retryable_plan_count=0,
        outcome_refs=("detail",),
    )
    monkeypatch.setattr(
        "polylogue.storage.raw_reconciler.apply_raw_authority_frontier",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(TypeError, match="untyped apply report"):
        raw_authority.apply_frontier(
            config,
            preview_census_id="preview-census-1",
            selected_plan_ids=("safe-1",),
        )


def test_frontier_apply_report_rejects_incoherent_counts() -> None:
    with pytest.raises(ValueError, match="incoherent plan counts"):
        RawAuthorityFrontierApplyReport(
            census_id="apply-census-1",
            preview_census_id="preview-census-1",
            selected_plan_count=1,
            executed_plan_count=2,
            retryable_plan_count=0,
            post_inventory_digest="digest",
            post_plan_count=0,
            outcome_refs=("detail",),
        )


def test_materialization_generation_lease_pins_active_index_and_excludes_promotion(tmp_path: Path) -> None:
    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    active_index.touch()
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])

    with raw_authority.materialization_generation_lease(config) as index_db:
        assert index_db == active_index
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass


def test_materialization_generation_lease_uses_explicit_split_root(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    active_root = tmp_path / "active"
    configured_root.mkdir()
    active_root.mkdir()
    active_index = active_root / "index.db"
    active_index.touch()
    config = Config(
        archive_root=configured_root,
        render_root=tmp_path / "render",
        sources=[],
        db_path=active_index,
    )

    with raw_authority.materialization_generation_lease(config) as index_db:
        assert index_db == active_index
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(active_root):
                pass
        with RebuildLease(configured_root):
            pass


@pytest.mark.parametrize(
    ("selected_plan_ids", "preview_census_id", "outcome_plan_id", "message"),
    [
        (("safe-1", "safe-2"), "preview-census-1", "safe-1", "selected plan count does not match"),
        (("safe-1",), "preview-census-2", "safe-1", "preview census does not match"),
        (("safe-1",), "preview-census-1", "safe-2", "outcome references do not match"),
    ],
)
def test_apply_frontier_rejects_response_bound_to_a_different_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selected_plan_ids: tuple[str, ...],
    preview_census_id: str,
    outcome_plan_id: str,
    message: str,
) -> None:
    monkeypatch.setattr("polylogue.daemon.write_coordinator.daemon_write_lease_active", lambda: True)
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )
    response = RawAuthorityFrontierApplyReport(
        census_id="apply-census-1",
        preview_census_id="preview-census-1",
        selected_plan_count=1,
        executed_plan_count=1,
        retryable_plan_count=0,
        post_inventory_digest="digest",
        post_plan_count=0,
        outcome_refs=(raw_authority_detail_query_handle("apply-census-1", outcome_plan_id),),
    )
    monkeypatch.setattr(
        "polylogue.storage.raw_reconciler.apply_raw_authority_frontier",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(ValueError, match=message):
        raw_authority.apply_frontier(
            config,
            preview_census_id=preview_census_id,
            selected_plan_ids=selected_plan_ids,
        )


def test_apply_frontier_requires_daemon_writer_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )
    monkeypatch.setattr(
        "polylogue.storage.raw_reconciler.apply_raw_authority_frontier",
        lambda *_args, **_kwargs: pytest.fail("the actuator must not run without the daemon writer lease"),
    )

    with pytest.raises(RuntimeError, match="daemon writer lease"):
        raw_authority.apply_frontier(
            config,
            preview_census_id="preview-census-1",
            selected_plan_ids=("safe-1",),
        )
