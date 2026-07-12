"""Planner narrows ``affected_rows`` when the scope filter narrows the scope (#1303).

The planner's contract with the typed
:class:`MaintenanceScopeFilter` is that a narrower filter must
produce a narrower preview — the operator must never see a single-
session plan advertise the full archive's debt as its work.

Pins:

* a ``session_ids`` filter clamps ``affected_rows`` to the size
  of the filter set;
* the filter is threaded onto the returned :class:`MaintenanceScope`
  so the envelope echoes it back unchanged;
* an empty filter does not narrow the preview;
* a filter with zero session ids cannot mask a broader debt by
  accident (the underlying debt count is preserved when there is no
  session-id narrowing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.maintenance.planner import preview_backfill
from polylogue.maintenance.scope import MaintenanceScopeFilter
from tests.infra.storage_records import DbFactory, db_setup


def _seeded_config(workspace_env: dict[str, Path], *, sessions: int = 3) -> Config:
    index_db = db_setup(workspace_env)
    factory = DbFactory(index_db)
    for index in range(sessions):
        factory.create_session(id=f"empty-{index}")
    return Config(
        archive_root=workspace_env["archive_root"],
        render_root=workspace_env["data_root"] / "render",
        sources=[],
        db_path=index_db,
    )


class TestPlannerNarrowsBySessionIds:
    """A ``session_ids`` filter clamps preview rows to the filter size."""

    def test_single_session_filter_clamps_real_archive_debt(self, workspace_env: dict[str, Path]) -> None:
        config = _seeded_config(workspace_env)
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=("only-one",)),
        )
        assert narrow.affected_rows == 1
        # And the filter is echoed back on the returned scope so the
        # envelope can serialize it.
        assert narrow.scope is not None
        assert narrow.scope.filter.session_ids == ("only-one",)

    def test_multi_session_filter_clamps_to_filter_size(self, workspace_env: dict[str, Path]) -> None:
        config = _seeded_config(workspace_env)
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=("c1", "c2", "c3")),
        )
        assert narrow.affected_rows == 3
        assert narrow.scope is not None
        assert narrow.scope.filter.session_ids == ("c1", "c2", "c3")

    def test_session_filter_does_not_inflate_when_debt_is_smaller(self, workspace_env: dict[str, Path]) -> None:
        """A filter naming 100 ids cannot inflate a 2-row debt to 100."""
        config = _seeded_config(workspace_env, sessions=2)
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=tuple(f"c{i}" for i in range(100))),
        )
        assert narrow.affected_rows == 2


class TestPlannerLeavesFilterPassthroughDimensionsIntact:
    """Filters on dimensions the planner does not yet narrow on are passthrough."""

    @pytest.mark.parametrize(
        "scope_filter",
        [
            MaintenanceScopeFilter(provider="claude"),
            MaintenanceScopeFilter(source_family="claude-code-session"),
            MaintenanceScopeFilter(failure_kind="ValidationError"),
            MaintenanceScopeFilter(parser_version="v3"),
        ],
    )
    def test_non_session_filters_do_not_change_affected_rows(
        self, workspace_env: dict[str, Path], scope_filter: MaintenanceScopeFilter
    ) -> None:
        """Filter dimensions the planner doesn't yet honor pass through unchanged.

        The repair-fn boundary is advisory (see ``scope.py`` module
        docstring): non-session_ids filters reach the planner and
        are surfaced on the returned scope, but the preview row count
        comes from the full debt status until a repair fn learns to
        narrow on that dimension.
        """
        config = _seeded_config(workspace_env)
        broad = preview_backfill(config, targets=("empty_sessions",))
        scoped = preview_backfill(config, targets=("empty_sessions",), scope_filter=scope_filter)
        assert broad.affected_rows == 3
        assert scoped.affected_rows == 3
        # And the filter still rides through on the scope so the
        # envelope can echo it.
        assert scoped.scope is not None
        assert scoped.scope.filter == scope_filter


class TestPlannerWithEmptyFilter:
    """An empty / default filter must not narrow the preview."""

    def test_default_filter_preserves_full_debt(self, workspace_env: dict[str, Path]) -> None:
        config = _seeded_config(workspace_env)
        broad = preview_backfill(config, targets=("empty_sessions",))
        explicit_empty = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(),
        )
        assert broad.affected_rows == 3
        assert explicit_empty.affected_rows == 3
        # Both ought to attach an empty filter onto the scope.
        assert broad.scope is not None and broad.scope.filter.is_empty()
        assert explicit_empty.scope is not None and explicit_empty.scope.filter.is_empty()
