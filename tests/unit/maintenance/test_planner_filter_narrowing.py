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

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.enums import OperationStatus
from polylogue.maintenance.planner import preview_backfill
from polylogue.maintenance.scope import MaintenanceScopeFilter
from tests.infra.storage_records import DbFactory, db_setup


def _seeded_config(workspace_env: dict[str, Path], *, sessions: int = 3) -> Config:
    index_db = db_setup(workspace_env)
    factory = DbFactory(index_db)
    for index in range(sessions):
        native_id = f"empty-{index}"
        factory.create_session(id=native_id)
        # A message-less session with no raw artifact at all (raw_id IS
        # NULL) is "no evidence either way" to the empty_sessions debt
        # classifier and is never counted as debt -- give each one a
        # phantom raw artifact the classifier positively refuses, so this
        # fixture actually produces real archive debt for the tests below
        # to narrow (polylogue-9rdky).
        factory.mark_as_phantom_debris(native_id)
    return Config(
        archive_root=workspace_env["archive_root"],
        render_root=workspace_env["data_root"] / "render",
        sources=[],
        db_path=index_db,
    )


def _debris_session_ids(config: Config) -> tuple[str, ...]:
    with closing(sqlite3.connect(config.db_path)) as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))


class TestPlannerNarrowsBySessionIds:
    """A ``session_ids`` filter previews exactly the requested sessions."""

    def test_single_session_filter_counts_that_session(self, workspace_env: dict[str, Path]) -> None:
        config = _seeded_config(workspace_env)
        requested = _debris_session_ids(config)[:1]
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=requested),
        )
        assert narrow.affected_rows == 1
        # And the filter is echoed back on the returned scope so the
        # envelope can serialize it.
        assert narrow.scope is not None
        assert narrow.scope.filter.session_ids == requested

    def test_multi_session_filter_counts_the_requested_sessions(self, workspace_env: dict[str, Path]) -> None:
        config = _seeded_config(workspace_env)
        requested = _debris_session_ids(config)
        assert len(requested) == 3
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=requested),
        )
        assert narrow.affected_rows == 3
        assert narrow.scope is not None
        assert narrow.scope.filter.session_ids == requested

    def test_session_filter_does_not_inflate_when_debt_is_smaller(self, workspace_env: dict[str, Path]) -> None:
        """A filter naming 100 ids cannot inflate a 2-row debt to 100."""
        config = _seeded_config(workspace_env, sessions=2)
        requested = _debris_session_ids(config) + tuple(f"c{i}" for i in range(100))
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=requested),
        )
        assert narrow.affected_rows == 2

    def test_healthy_requested_sessions_preview_no_rows(self, workspace_env: dict[str, Path]) -> None:
        """Rows are counted over the requested sessions, not clamped to the request size.

        Anti-vacuity: restoring the ``min(total_rows, len(session_ids))``
        clamp makes this red -- three unrelated debris sessions plus a
        one-session request would preview 1 affected row while
        ``repair_empty_sessions`` scoped to that session deletes none.
        """
        config = _seeded_config(workspace_env)
        narrow = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(session_ids=("test:healthy-and-unrelated",)),
        )
        assert narrow.affected_rows == 0
        assert narrow.estimated_time_s == 0.0


class TestPlannerRefusesUnsupportedFilters:
    """Filters a target cannot apply produce no executable preview rows."""

    @pytest.mark.parametrize(
        "scope_filter",
        [
            MaintenanceScopeFilter(origin="claude-code-session"),
            MaintenanceScopeFilter(source_family="claude-code-session"),
            MaintenanceScopeFilter(failure_kind="ValidationError"),
            MaintenanceScopeFilter(parser_version="v3"),
        ],
    )
    def test_unsupported_filters_refuse_the_target_without_rows(
        self, workspace_env: dict[str, Path], scope_filter: MaintenanceScopeFilter
    ) -> None:
        """Anti-vacuity: counting full debt for a refused target goes red."""
        config = _seeded_config(workspace_env)
        broad = preview_backfill(config, targets=("empty_sessions",))
        scoped = preview_backfill(config, targets=("empty_sessions",), scope_filter=scope_filter)
        assert broad.affected_rows == 3
        assert scoped.status is OperationStatus.FAILED
        assert scoped.affected_rows == 0
        assert scoped.scope is not None
        assert scoped.scope.filter == scope_filter
        assert scoped.failure_samples.samples[0].kind == "UnsupportedScopeDimension"

    def test_refused_preview_carries_an_error_message(self, workspace_env: dict[str, Path]) -> None:
        """A refused preview names its refusal in ``error``.

        Anti-vacuity: dropping the ``error=`` argument from
        ``preview_backfill``'s refusal receipt makes this red -- the plain
        CLI reads ``result.error`` and would print nothing but
        "Affected: 0 rows" for a permanently refused request.
        """
        config = _seeded_config(workspace_env)
        scoped = preview_backfill(
            config,
            targets=("empty_sessions",),
            scope_filter=MaintenanceScopeFilter(origin="claude-code-session"),
        )
        assert scoped.status is OperationStatus.FAILED
        assert scoped.error is not None
        assert "origin" in scoped.error


class TestOmittedSessionFilterIsNotAScope:
    """An omitted repeatable CLI option is no narrowing at all."""

    def test_empty_session_id_tuple_normalizes_to_none(self, workspace_env: dict[str, Path]) -> None:
        """``--session-id`` omitted must not refuse a target that ignores it.

        Anti-vacuity: restoring ``_coerce_session_ids``'s ``tuple(...)``
        without the ``or None`` makes this red -- Click hands ``()`` for an
        omitted repeatable option, and ``superseded_raw_snapshots`` honors no
        scope dimension, so the default plan would be refused.
        """
        config = _seeded_config(workspace_env)
        scope_filter = MaintenanceScopeFilter.from_surface_args(session_ids=())
        assert scope_filter.session_ids is None
        assert scope_filter.is_empty()
        plan = preview_backfill(
            config,
            targets=("superseded_raw_snapshots",),
            scope_filter=scope_filter,
        )
        assert plan.status is OperationStatus.PENDING
        assert plan.failure_samples.samples == ()


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
