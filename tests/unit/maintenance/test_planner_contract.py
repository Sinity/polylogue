"""Contract tests for the typed maintenance planner (issue #1144).

These exercise the typed shape of ``BackfillOperation`` and its
sub-types — the rest of the maintenance cluster (resume/idempotency,
operation-envelope wiring, embedding/cost backfills) builds on top of
this contract and depends on the shape staying stable.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config
from polylogue.maintenance.invalidation import InvalidationReason
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.maintenance.planner import (
    MAX_FAILURE_SAMPLES,
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    BoundedFailureSamples,
    FailureSample,
    MaintenanceScope,
    _derive_invalidation_reason,
    execute_backfill,
    preview_backfill,
)
from polylogue.maintenance.targets import build_maintenance_target_catalog
from tests.infra.storage_records import SessionBuilder, db_setup


def _make_config(tmp_path: Path) -> Config:
    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    return Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[],
        db_path=tmp_path / "archive.db",
    )


class TestBackfillKindCoverage:
    """Every typed-taxonomy kind from #1144 must be representable."""

    @pytest.mark.parametrize(
        "kind",
        [
            BackfillKind.ARCHIVE_SUBSET,
            BackfillKind.DERIVED_REBUILD,
            BackfillKind.INDEX_REPAIR,
            BackfillKind.SEMANTIC_REMATERIALIZE,
            BackfillKind.CONFIG_DRIVEN,
        ],
    )
    def test_kind_roundtrips_through_to_dict(self, kind: BackfillKind) -> None:
        op = BackfillOperation(
            operation_id="op-1",
            kind=kind,
            targets=("session_insights",),
        )
        payload = op.to_dict()
        assert payload["kind"] == kind.value

    @pytest.mark.parametrize(
        ("stored_kind", "expected"),
        [
            ("backfill", BackfillKind.DERIVED_REBUILD),
            ("rebuild", BackfillKind.DERIVED_REBUILD),
            ("reindex", BackfillKind.INDEX_REPAIR),
            ("reset", BackfillKind.CONFIG_DRIVEN),
        ],
    )
    def test_retired_stored_kind_values_rehydrate_to_typed_kind(self, stored_kind: str, expected: BackfillKind) -> None:
        op = BackfillOperation.from_dict(
            {
                "operation_id": "op-1",
                "kind": stored_kind,
                "targets": ["session_insights"],
            }
        )
        assert op.kind is expected


class TestInvalidationReasonCoverage:
    """Every InvalidationReason value from #1144 must be representable
    and must survive the BackfillOperation -> to_dict roundtrip."""

    @pytest.mark.parametrize(
        "reason",
        [
            InvalidationReason.MISSING,
            InvalidationReason.STALE_MATERIALIZER_VERSION,
            InvalidationReason.SOURCE_CHANGED,
            InvalidationReason.PARSER_OR_SCHEMA_CHANGED,
            InvalidationReason.CONFIG_OR_MODEL_SNAPSHOT_CHANGED,
            InvalidationReason.UNKNOWN,
        ],
    )
    def test_reason_roundtrips_through_to_dict(self, reason: InvalidationReason) -> None:
        op = BackfillOperation(
            operation_id="op-1",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
            reason=reason,
        )
        assert op.to_dict()["reason"] == reason.value


class TestMaintenanceScope:
    def test_scope_default_filter_is_empty(self) -> None:
        from polylogue.maintenance.scope import MaintenanceScopeFilter

        scope = MaintenanceScope(targets=("a", "b"))
        assert scope.filter == MaintenanceScopeFilter()
        assert scope.filter.is_empty()
        payload = scope.to_dict()
        assert payload["targets"] == ["a", "b"]
        filter_payload = cast(dict[str, object], payload["filter"])
        # Every typed scope dimension is present with a None default.
        assert filter_payload["session_ids"] is None
        assert filter_payload["provider"] is None

    def test_scope_filter_roundtrips(self) -> None:
        from polylogue.maintenance.scope import MaintenanceScopeFilter

        scope = MaintenanceScope(
            targets=("session_insights",),
            filter=MaintenanceScopeFilter(
                session_ids=("c1", "c2"),
                provider="claude",
            ),
        )
        payload = scope.to_dict()
        filter_payload = cast(dict[str, object], payload["filter"])
        assert filter_payload["session_ids"] == ["c1", "c2"]
        assert filter_payload["provider"] == "claude"
        # to_dict / from_dict round-trips the scope back to itself.
        scope_again = MaintenanceScope.from_dict(cast(dict[str, object], payload))
        assert scope_again == scope

    def test_operation_synthesizes_scope_from_targets(self) -> None:
        op = BackfillOperation(
            operation_id="op-x",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("a", "b"),
        )
        assert op.scope is not None
        assert op.scope.targets == ("a", "b")

    def test_operation_trusts_explicit_scope(self) -> None:
        scope = MaintenanceScope(targets=("only_scope",))
        op = BackfillOperation(
            operation_id="op-x",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("ignored",),
            scope=scope,
        )
        # scope wins; .targets is realigned for caller convenience.
        assert op.targets == ("only_scope",)


class TestBoundedFailureSamples:
    def test_under_limit_passes_through(self) -> None:
        samples = [FailureSample(kind="X", locator=str(i), message="m") for i in range(3)]
        envelope = BoundedFailureSamples.from_samples(samples)
        assert len(envelope.samples) == 3
        assert envelope.truncated is False

    def test_over_limit_truncates_and_flags(self) -> None:
        samples = [FailureSample(kind="X", locator=str(i), message="m") for i in range(MAX_FAILURE_SAMPLES + 5)]
        envelope = BoundedFailureSamples.from_samples(samples)
        assert len(envelope.samples) == MAX_FAILURE_SAMPLES
        assert envelope.truncated is True

    def test_envelope_to_dict_includes_truncated_flag(self) -> None:
        envelope = BoundedFailureSamples.from_samples([FailureSample(kind="K", locator="row-1", message="boom")])
        payload = envelope.to_dict()
        assert payload["truncated"] is False
        samples_payload = cast(list[dict[str, object]], payload["samples"])
        assert samples_payload == [{"kind": "K", "locator": "row-1", "message": "boom"}]


class TestResumeCursorRoundtrip:
    def test_cursor_survives_to_dict(self) -> None:
        op = BackfillOperation(
            operation_id="op-1",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
            resume_cursor="rowid:12345",
        )
        assert op.to_dict()["resume_cursor"] == "rowid:12345"

    def test_missing_cursor_is_none(self) -> None:
        op = BackfillOperation(
            operation_id="op-1",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
        )
        assert op.to_dict()["resume_cursor"] is None


class TestMetricsRoundtrip:
    def test_metrics_are_independent_dict(self) -> None:
        metrics = {"rows_per_s": 123.4, "passes": 2.0}
        op = BackfillOperation(
            operation_id="op-1",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
            metrics=metrics,
        )
        payload = op.to_dict()
        metrics_payload = cast(dict[str, float], payload["metrics"])
        assert metrics_payload == metrics
        # Mutating the payload must not affect the source dict.
        metrics_payload["rows_per_s"] = 0.0
        assert op.metrics["rows_per_s"] == 123.4


class TestInvalidationKeysOnTargets:
    """The #1144 AC requires invalidation_keys on the listed targets."""

    def test_target_specs_have_required_invalidation_keys(self) -> None:
        catalog = build_maintenance_target_catalog()
        by_name = catalog.by_name()

        assert "session.profile" in by_name["session_insights"].invalidation_keys

    def test_invalidation_keys_surface_through_to_dict(self) -> None:
        spec = build_maintenance_target_catalog().by_name()["session_insights"]
        payload = spec.to_dict()
        invalidation_keys = cast(list[str], payload["invalidation_keys"])
        assert "session.profile" in invalidation_keys


class TestDeriveInvalidationReason:
    def test_ready_status_yields_no_reason(self) -> None:
        status = DerivedModelStatus(name="x", ready=True, detail="ok")
        assert _derive_invalidation_reason(status) is None

    def test_missing_materialized_documents_yields_missing(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="empty",
            source_documents=10,
            materialized_documents=0,
        )
        assert _derive_invalidation_reason(status) is InvalidationReason.MISSING

    def test_explicit_invalidated_reason_wins(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="stale",
            source_documents=5,
            materialized_documents=5,
            invalidated_reason=InvalidationReason.CONFIG_OR_MODEL_SNAPSHOT_CHANGED,
        )
        assert _derive_invalidation_reason(status) is InvalidationReason.CONFIG_OR_MODEL_SNAPSHOT_CHANGED

    def test_version_mismatch_yields_stale_materializer_version(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="version skew",
            source_documents=5,
            materialized_documents=5,
            matches_version=False,
        )
        assert _derive_invalidation_reason(status) is InvalidationReason.STALE_MATERIALIZER_VERSION

    def test_stale_rows_yields_source_changed(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="some stale rows",
            source_documents=5,
            materialized_documents=5,
            stale_rows=2,
        )
        assert _derive_invalidation_reason(status) is InvalidationReason.SOURCE_CHANGED

    def test_missing_provenance_yields_parser_or_schema_changed(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="provenance gone",
            source_documents=5,
            materialized_documents=5,
            missing_provenance_rows=3,
        )
        assert _derive_invalidation_reason(status) is InvalidationReason.PARSER_OR_SCHEMA_CHANGED

    def test_unclassified_unready_falls_back_to_unknown(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="no signal",
            source_documents=5,
            materialized_documents=5,
        )
        assert _derive_invalidation_reason(status) is InvalidationReason.UNKNOWN

    def test_non_status_input_is_ignored(self) -> None:
        assert _derive_invalidation_reason("not a status") is None


class TestEmptyTargetsFastFail:
    def test_preview_with_no_resolvable_targets_returns_failed(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        op = preview_backfill(config, targets=("does-not-exist",))
        assert op.status is BackfillStatus.FAILED
        assert op.targets == ()
        assert op.scope is not None
        assert op.scope.targets == ()

    def test_execute_with_no_resolvable_targets_returns_failed(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        op = execute_backfill(config, targets=("does-not-exist",))
        assert op.status is BackfillStatus.FAILED
        assert op.targets == ()


class TestConfigThreading:
    """Regression: the planner must resolve the archive `index.db` from the
    caller's ``config`` (archive_root/db_path), not rely on ambient defaults
    (the original scaffold did the latter and broke multi-archive tests).
    """

    def test_preview_reads_the_callers_seeded_archive(self, workspace_env: dict[str, Path]) -> None:
        """Planner debt comes from the supplied archive, not ambient paths."""
        index_db = db_setup(workspace_env)
        SessionBuilder(index_db, "planner-config").provider("chatgpt").save()
        config = Config(
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["data_root"] / "render",
            sources=[],
            db_path=index_db,
        )

        preview = preview_backfill(config, targets=("empty_sessions",))

        assert preview.targets == ("empty_sessions",)
        assert preview.affected_rows == 1
        assert preview.results

    def test_execute_threads_config_db_path(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        config = _make_config(tmp_path)
        initialize_active_archive_root(config.archive_root)
        expected_index = config.archive_root / "index.db"

        seen: list[Path] = []

        def fake_open(path: object, **_: object) -> MagicMock:
            seen.append(Path(str(path)))
            return MagicMock()

        with (
            patch(
                "polylogue.storage.sqlite.connection_profile.open_readonly_connection",
                side_effect=fake_open,
            ),
            patch(
                "polylogue.storage.repair.collect_archive_debt_statuses_sync",
                return_value={},
            ),
            patch(
                "polylogue.storage.repair.preview_counts_from_archive_debt",
                return_value={},
            ),
            patch(
                "polylogue.storage.repair.run_selected_maintenance",
                return_value=[],
            ),
        ):
            op = execute_backfill(config, targets=("session_insights",))

        assert seen == [expected_index]
        # An execute that succeeded with zero results is COMPLETED (all([]) is True).
        assert op.status is BackfillStatus.COMPLETED


class TestExecuteFailureSamples:
    """When the executor raises, the failure must surface as a bounded sample."""

    def test_unexpected_exception_populates_failure_samples(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)

        @contextmanager
        def fake_connection_context(db_path: object) -> Iterator[object]:
            yield object()

        with (
            patch(
                "polylogue.storage.sqlite.connection.connection_context",
                fake_connection_context,
            ),
            patch(
                "polylogue.storage.repair.collect_archive_debt_statuses_sync",
                return_value={},
            ),
            patch(
                "polylogue.storage.repair.preview_counts_from_archive_debt",
                return_value={},
            ),
            patch(
                "polylogue.storage.repair.run_selected_maintenance",
                side_effect=RuntimeError("simulated repair failure"),
            ),
        ):
            op = execute_backfill(config, targets=("session_insights",))

        assert op.status is BackfillStatus.FAILED
        assert op.failure_samples.truncated is False
        assert len(op.failure_samples.samples) == 1
        sample = op.failure_samples.samples[0]
        assert sample.kind == "RuntimeError"
        assert "simulated repair failure" in sample.message


class TestDerivedModelStatusInvalidatedReason:
    def test_status_to_dict_includes_invalidated_reason(self) -> None:
        status = DerivedModelStatus(
            name="x",
            ready=False,
            detail="d",
            invalidated_reason=InvalidationReason.MISSING,
        )
        payload = status.to_dict()
        assert payload["invalidated_reason"] == "missing"

    def test_status_to_dict_omits_reason_when_none(self) -> None:
        status = DerivedModelStatus(name="x", ready=True, detail="ok")
        payload = status.to_dict()
        assert payload["invalidated_reason"] is None
