"""Tests for cardinality enforcement in mark / analyze / delete query verbs.

Cardinality rules (from #1814):
  - mark:   requires exactly one result unless --all or --first.
  - analyze: no cardinality restriction (applies to result set).
  - delete: requires --dry-run for preview; --yes plus --all for multi-match.

All three verbs share the single :func:`check_cardinality` path from
``polylogue.cli.verb_cardinality``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli import query_verbs
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.verb_cardinality import CardinalityError, check_cardinality

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context_pair(
    *,
    params: dict[str, object] | None = None,
    query_terms: tuple[str, ...] = (),
) -> tuple[click.Context, click.Context]:
    """Build a (parent, child) context pair that mimics the query verb tree."""
    parent = click.Context(click.Command("query"))
    parent.params = {"query_term": query_terms, **(params or {})}
    parent.meta["polylogue_query_terms"] = query_terms
    child = click.Context(click.Command("verb"), parent=parent)
    child.obj = SimpleNamespace()
    return parent, child


# ---------------------------------------------------------------------------
# check_cardinality — pure-function tests
# ---------------------------------------------------------------------------


class TestCheckCardinality:
    """Tests for the shared cardinality guard."""

    def test_singleton_always_passes(self) -> None:
        # No exception raised.
        check_cardinality(1, allow_all=False, first_only=False, operation="mark")

    def test_zero_always_raises(self) -> None:
        with pytest.raises(CardinalityError, match="No sessions matched"):
            check_cardinality(0, allow_all=False, first_only=False, operation="mark")

    def test_zero_with_all_still_raises(self) -> None:
        with pytest.raises(CardinalityError, match="No sessions matched"):
            check_cardinality(0, allow_all=True, first_only=False, operation="mark")

    def test_multi_without_all_or_first_raises(self) -> None:
        with pytest.raises(CardinalityError, match="--all"):
            check_cardinality(3, allow_all=False, first_only=False, operation="mark")

    def test_multi_with_allow_all_passes(self) -> None:
        # No exception raised.
        check_cardinality(3, allow_all=True, first_only=False, operation="mark")

    def test_multi_with_first_only_passes(self) -> None:
        # No exception raised.
        check_cardinality(3, allow_all=False, first_only=True, operation="mark")

    def test_error_message_includes_operation(self) -> None:
        with pytest.raises(CardinalityError, match="delete"):
            check_cardinality(2, allow_all=False, first_only=False, operation="delete")

    def test_error_message_includes_count(self) -> None:
        with pytest.raises(CardinalityError, match="5 sessions"):
            check_cardinality(5, allow_all=False, first_only=False, operation="mark")

    def test_cardinality_error_is_usage_error(self) -> None:
        with pytest.raises(click.UsageError):
            check_cardinality(2, allow_all=False, first_only=False, operation="test")


# ---------------------------------------------------------------------------
# mark_verb — cardinality enforcement
# ---------------------------------------------------------------------------


class TestMarkVerbCardinality:
    """mark_verb enforces the singleton / --all / --first contract."""

    def _mark_callback(self) -> object:
        cb = getattr(query_verbs.mark_verb.callback, "__wrapped__", None)
        assert callable(cb), "mark_verb.callback must be a context-decorated function"
        return cb

    def _call_mark(
        self,
        child: click.Context,
        *,
        tags_to_add: tuple[str, ...] = (),
        tags_to_remove: tuple[str, ...] = (),
        star: bool = False,
        unstar: bool = False,
        pin: bool = False,
        unpin: bool = False,
        do_archive: bool = False,
        do_unarchive: bool = False,
        note_text: str | None = None,
        apply_all: bool = False,
        first_only: bool = False,
    ) -> None:
        cb = self._mark_callback()
        cb(  # type: ignore[operator]
            child,
            tags_to_add,
            tags_to_remove,
            star,
            unstar,
            pin,
            unpin,
            do_archive,
            do_unarchive,
            note_text,
            apply_all,
            first_only,
        )

    def test_singleton_match_applies_operations(self) -> None:
        _, child = _context_pair()
        mock_poly = MagicMock()
        child.obj = SimpleNamespace(polylogue=mock_poly, config=MagicMock())

        add_tag_future: MagicMock = MagicMock()
        add_tag_future.__await__ = lambda self: iter([None])

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["session-abc"],
            ),
            patch(
                "polylogue.api.sync.bridge.run_coroutine_sync",
                side_effect=lambda coro: None,
            ),
        ):
            # Should not raise.
            self._call_mark(child, tags_to_add=("reviewed",), apply_all=False)

    def test_multi_match_without_all_raises(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2", "id3"],
            ),
        ):
            with pytest.raises(click.UsageError, match="--all"):
                self._call_mark(child, tags_to_add=("reviewed",), apply_all=False)

    def test_multi_match_with_all_passes_cardinality(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(polylogue=MagicMock(), config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2"],
            ),
            patch(
                "polylogue.api.sync.bridge.run_coroutine_sync",
                side_effect=lambda coro: None,
            ),
        ):
            # Should not raise — --all is present.
            self._call_mark(child, tags_to_add=("sprint",), apply_all=True)

    def test_multi_match_with_first_uses_first_only(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(polylogue=MagicMock(), config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2", "id3"],
            ),
            patch(
                "polylogue.api.sync.bridge.run_coroutine_sync",
                side_effect=lambda coro: None,
            ),
        ):
            # Should not raise — --first is present.
            self._call_mark(child, tags_to_add=("sprint",), first_only=True)

    def test_zero_matches_raises(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=[],
            ),
        ):
            with pytest.raises(click.UsageError, match="No sessions matched"):
                self._call_mark(child, tags_to_add=("sprint",))

    def test_mark_uses_shared_check_cardinality(self) -> None:
        """mark_verb must call check_cardinality (the shared path)."""
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2"],
            ),
            patch(
                "polylogue.cli.verb_cardinality.check_cardinality",
                side_effect=CardinalityError("mocked error"),
            ) as mock_check,
        ):
            with pytest.raises(click.UsageError, match="mocked error"):
                self._call_mark(child, tags_to_add=("t",), apply_all=False)

        mock_check.assert_called_once_with(
            2,
            allow_all=False,
            first_only=False,
            operation="mark",
        )


# ---------------------------------------------------------------------------
# delete_verb — cardinality enforcement (updated verb)
# ---------------------------------------------------------------------------


class TestDeleteVerbCardinality:
    """delete_verb enforces cardinality and --dry-run / --yes / --all contract."""

    def _delete_callback(self) -> object:
        cb = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
        assert callable(cb), "delete_verb.callback must be a context-decorated function"
        return cb

    def _call_delete(
        self,
        child: click.Context,
        *,
        dry_run: bool = False,
        yes_flag: bool = False,
        all_flag: bool = False,
        force: bool = False,
    ) -> None:
        cb = self._delete_callback()
        cb(child, dry_run, yes_flag, all_flag, force)  # type: ignore[operator]

    def test_dry_run_skips_cardinality_check(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2", "id3"],
            ),
            patch("polylogue.cli.verb_cardinality.check_cardinality") as mock_card,
            patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as mock_exec,
        ):
            self._call_delete(child, dry_run=True)

        # The cardinality guard must NOT run for a dry-run (it is a preview, not a
        # destructive action), but the preview must still go through the real
        # full-set delete path.
        mock_card.assert_not_called()
        mock_exec.assert_called_once()

    def test_dry_run_previews_full_resolved_set_not_truncated_query(self) -> None:
        """Dry-run previews the full pre-resolved ID set.

        Regression for the #1873 truncation: dry-run must NOT re-run the query
        through ``_execute_query_verb`` (which caps at the default limit of 20 and
        would preview fewer sessions than ``--yes --all`` actually deletes). It
        must use the same resolution + delete path the real delete uses, with
        ``dry_run=True``, so the previewed set equals the deleted set.
        """
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        resolved = [f"id{i}" for i in range(60)]
        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=resolved,
            ),
            patch("polylogue.cli.query_verbs._execute_query_verb") as mock_query,
            patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as mock_exec,
        ):
            self._call_delete(child, dry_run=True)

        mock_query.assert_not_called()
        args, kwargs = mock_exec.call_args
        assert list(args[1]) == resolved, "dry-run must preview every resolved id, not a page"
        assert kwargs.get("dry_run") is True

    def test_multi_match_without_all_raises(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2"],
            ),
        ):
            with pytest.raises(click.UsageError, match="--all"):
                self._call_delete(child, yes_flag=True, all_flag=False)

    def test_multi_match_with_yes_and_all_delegates(self) -> None:
        # Bug 1 fix: delete uses execute_delete_by_session_ids (not _execute_query_verb)
        # so all resolved IDs are deleted rather than only the first limit=20.
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2"],
            ),
            patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as mock_exec,
        ):
            self._call_delete(child, yes_flag=True, all_flag=True)

        mock_exec.assert_called_once()

    def test_zero_matches_raises(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=[],
            ),
        ):
            with pytest.raises(click.UsageError, match="No sessions matched"):
                self._call_delete(child, yes_flag=True, all_flag=True)

    def test_delete_uses_shared_check_cardinality(self) -> None:
        """delete_verb must call check_cardinality (the shared path)."""
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2"],
            ),
            patch(
                "polylogue.cli.verb_cardinality.check_cardinality",
                side_effect=CardinalityError("mocked cardinality error"),
            ) as mock_check,
        ):
            with pytest.raises(click.UsageError, match="mocked cardinality error"):
                self._call_delete(child, yes_flag=True, all_flag=False)

        mock_check.assert_called_once_with(
            2,
            allow_all=False,
            first_only=False,
            operation="delete",
        )

    def test_yes_flag_sets_force_on_delegated_request(self) -> None:
        # Bug 1 fix: delete passes force=True to execute_delete_by_session_ids.
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        captured_kwargs: list[dict[str, object]] = []

        def _capture(env: object, ids: list[str], *, force: bool) -> None:
            captured_kwargs.append({"ids": ids, "force": force})

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1"],
            ),
            patch(
                "polylogue.cli.archive_query.execute_delete_by_session_ids",
                side_effect=_capture,
            ),
        ):
            self._call_delete(child, yes_flag=True)

        assert captured_kwargs[0]["force"] is True, "--yes must propagate force=True to execute_delete_by_session_ids"


# ---------------------------------------------------------------------------
# resolve_session_ids_for_verb — --sample is rejected, never silently ignored
# ---------------------------------------------------------------------------


class TestSampleRejectedForMutatingVerbs:
    """``--sample`` must not silently widen a mutating verb's blast radius.

    ``--sample N`` is a display-window random subset; the verb resolution path
    deliberately resolves the COMPLETE matched set. Honoring it would mean a
    destructive ``delete``/``mark`` operated on every match while the operator
    believed only N rows were in scope. The shared resolver rejects the
    combination up front rather than ignoring it.
    """

    def test_resolver_rejects_sample(self) -> None:
        from polylogue.cli.verb_cardinality import resolve_session_ids_for_verb

        request = RootModeRequest.from_params({"sample": 5})
        assert request.query_spec().sample == 5

        with patch("polylogue.api.sync.bridge.run_coroutine_sync") as mock_run:
            with pytest.raises(click.UsageError, match="--sample"):
                resolve_session_ids_for_verb(cast(object, MagicMock()), request)  # type: ignore[arg-type]

        # The guard fires before any DB resolution.
        mock_run.assert_not_called()

    def test_resolver_allows_absent_sample(self) -> None:
        """Without --sample the resolver proceeds to the async DB path."""
        from polylogue.cli.verb_cardinality import resolve_session_ids_for_verb

        request = RootModeRequest.from_params({})
        assert request.query_spec().sample is None

        with (
            # _async_resolve_ids is stubbed so no coroutine is created and the
            # MagicMock env is never dereferenced for real DB work.
            patch(
                "polylogue.cli.verb_cardinality._async_resolve_ids",
                new=lambda env, request: object(),
            ),
            patch(
                "polylogue.api.sync.bridge.run_coroutine_sync",
                return_value=["id1"],
            ) as mock_run,
        ):
            result = resolve_session_ids_for_verb(cast(object, MagicMock()), request)  # type: ignore[arg-type]

        assert result == ["id1"]
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# analyze_verb — no cardinality restriction
# ---------------------------------------------------------------------------


class TestAnalyzeVerbNoBcardinality:
    """analyze_verb applies to the full result set without cardinality guards."""

    def _analyze_callback(self) -> object:
        cb = getattr(query_verbs.analyze_verb.callback, "__wrapped__", None)
        assert callable(cb), "analyze_verb.callback must be a context-decorated function"
        return cb

    def test_analyze_default_delegates_stats_only(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock(), polylogue=MagicMock())

        captured: list[RootModeRequest] = []

        def _capture(ctx: click.Context, req: RootModeRequest) -> None:
            captured.append(req)

        cb = self._analyze_callback()
        with patch("polylogue.cli.query_verbs._execute_query_verb", side_effect=_capture):
            cb(child, None, False, False, None)  # type: ignore[operator]

        assert captured, "analyze_verb must call _execute_query_verb for default stats"
        assert captured[0].params.get("stats_only") is True

    def test_analyze_by_dimension_delegates_stats_by(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock(), polylogue=MagicMock())

        captured: list[RootModeRequest] = []

        def _capture(ctx: click.Context, req: RootModeRequest) -> None:
            captured.append(req)

        cb = self._analyze_callback()
        with patch("polylogue.cli.query_verbs._execute_query_verb", side_effect=_capture):
            cb(child, "origin", False, False, None)  # type: ignore[operator]

        assert captured[0].params.get("stats_by") == "origin"

    def test_analyze_does_not_call_check_cardinality(self) -> None:
        """analyze_verb must not call check_cardinality — no restriction."""
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock(), polylogue=MagicMock())

        cb = self._analyze_callback()
        with (
            patch("polylogue.cli.verb_cardinality.check_cardinality") as mock_check,
            patch("polylogue.cli.query_verbs._execute_query_verb"),
        ):
            cb(child, None, False, False, None)  # type: ignore[operator]

        mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# Shared path: mark and delete both use check_cardinality from the same module
# ---------------------------------------------------------------------------


class TestSharedCardinalityPath:
    """Structural tests: mark and delete must import from the same cardinality module."""

    def test_mark_and_delete_import_check_cardinality_from_same_module(self) -> None:
        import inspect

        mark_src = inspect.getsource(query_verbs.mark_verb.callback)  # type: ignore[arg-type]
        delete_src = inspect.getsource(query_verbs.delete_verb.callback)  # type: ignore[arg-type]

        # Both verbs must reference the shared module (not inline implementations).
        assert "verb_cardinality" in mark_src, "mark_verb must import from verb_cardinality"
        assert "verb_cardinality" in delete_src, "delete_verb must import from verb_cardinality"
        assert "check_cardinality" in mark_src, "mark_verb must call check_cardinality"
        assert "check_cardinality" in delete_src, "delete_verb must call check_cardinality"

    def test_analyze_does_not_use_cardinality_module(self) -> None:
        import inspect

        analyze_src = inspect.getsource(query_verbs.analyze_verb.callback)  # type: ignore[arg-type]
        # analyze_verb deliberately has no cardinality restriction.
        assert "check_cardinality" not in analyze_src, "analyze_verb must not call check_cardinality"


# ---------------------------------------------------------------------------
# Bug 1 regression: delete truncation (#1873)
# ---------------------------------------------------------------------------


class TestDeleteUsesPreResolvedIds:
    """delete_verb must operate on all pre-resolved IDs, not re-query with limit=20."""

    def _delete_callback(self) -> object:
        cb = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
        assert callable(cb)
        return cb

    def test_all_resolved_ids_are_deleted_not_truncated(self) -> None:
        """Bug 1: execute_delete_by_session_ids is called with all resolved IDs.

        Before the fix, _execute_query_verb re-ran the query with limit=20,
        silently truncating large result sets.  After the fix, the pre-resolved
        IDs are forwarded directly.
        """
        many_ids = [f"id{i}" for i in range(50)]  # more than the old default limit of 20
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        captured: list[list[str]] = []

        def _capture(env: object, ids: list[str], *, force: bool) -> None:
            captured.append(list(ids))

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=many_ids,
            ),
            patch(
                "polylogue.cli.archive_query.execute_delete_by_session_ids",
                side_effect=_capture,
            ),
        ):
            cb = self._delete_callback()
            cb(child, False, True, True, False)  # type: ignore[operator]  # dry_run=F, yes=T, all=T, force=F

        assert captured, "execute_delete_by_session_ids must be called"
        assert len(captured[0]) == 50, (
            f"Expected all 50 IDs to be deleted but got {len(captured[0])}. "
            "delete_verb may be re-querying with a limit instead of using pre-resolved IDs."
        )

    def test_delete_does_not_call_execute_query_verb_for_non_dry_run(self) -> None:
        """After the fix, the non-dry-run delete path must NOT call _execute_query_verb."""
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1"],
            ),
            patch("polylogue.cli.archive_query.execute_delete_by_session_ids"),
            patch("polylogue.cli.query_verbs._execute_query_verb") as mock_exec,
        ):
            cb = self._delete_callback()
            cb(child, False, True, False, False)  # type: ignore[operator]  # yes=T

        mock_exec.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 2 regression: _async_resolve_ids uses compiled spec (#1873)
# ---------------------------------------------------------------------------


class TestResolveIdsUsesCompiledSpec:
    """_async_resolve_ids must compile DSL expressions, not pass them as FTS text."""

    def test_resolve_ids_calls_query_spec_not_raw_params(self) -> None:
        """Bug 2: resolve path uses request.query_spec() (compiled DSL) not query_params()."""
        import inspect

        from polylogue.cli.verb_cardinality import _async_resolve_ids

        src = inspect.getsource(_async_resolve_ids)
        # After the fix the function must call query_spec() for DSL parsing.
        assert "query_spec()" in src, "_async_resolve_ids must call request.query_spec()"
        # The old broken path used build_query_execution_plan(request.query_params()).
        assert "build_query_execution_plan" not in src, (
            "_async_resolve_ids must not use build_query_execution_plan (passes raw text as FTS)"
        )
        assert "query_params()" not in src, "_async_resolve_ids must not call query_params() (bypasses DSL parsing)"


# ---------------------------------------------------------------------------
# Non-mocked >50-session delete cardinality evidence (#1873 recovery pack)
# ---------------------------------------------------------------------------


class TestDeleteCardinalityLargeNonMocked:
    """End-to-end evidence over a real seeded archive (no mocks on resolution/delete).

    The invariant that makes ``delete --yes --all`` safe is that three sets are
    identical and none is silently page-limited:

        guard set (cardinality)  ==  dry-run preview set  ==  deleted set

    The default query page limit is 20 (50 in some paths); seeding 60 matching
    sessions makes any truncation observable. This exercises the real
    ``resolve_session_ids_for_verb`` + ``delete_verb`` + ``ArchiveStore`` path.
    """

    TOKEN = "zzbulkdeletetoken"
    COUNT = 60
    _capsys: pytest.CaptureFixture[str]

    def _seed(self, index_db: Path) -> None:
        from tests.infra.storage_records import SessionBuilder

        for i in range(self.COUNT):
            (
                SessionBuilder(index_db, f"bulk-{i:03d}")
                .provider("claude-code")
                .title(f"{self.TOKEN} session {i}")
                .add_message(f"m{i}", role="user", text=f"{self.TOKEN} body line {i}")
                .save()
            )

    def _delete_callback(self) -> object:
        cb = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
        assert callable(cb), "delete_verb.callback must be a context-decorated function"
        return cb

    def _invoke_delete(self, env: object, *, dry_run: bool, yes_flag: bool, all_flag: bool) -> dict[str, object]:
        import json

        _, child = _context_pair(query_terms=(self.TOKEN,))
        child.obj = env
        self._delete_callback()(child, dry_run, yes_flag, all_flag, False)  # type: ignore[operator]
        # _emit_delete prints exactly one JSON document to stdout.
        captured = self._capsys.readouterr().out.strip()
        return cast(dict[str, object], json.loads(captured))

    @pytest.fixture(autouse=True)
    def _bind_capsys(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._capsys = capsys

    def test_guard_dry_run_and_deleted_sets_are_identical_and_unlimited(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.cli.verb_cardinality import resolve_session_ids_for_verb
        from tests.infra.app_env import make_app_env

        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db)

        env = make_app_env()
        request = RootModeRequest.from_params({"query": (self.TOKEN,)})

        # 1. Guard set: the full matched set, not a default page.
        guard = resolve_session_ids_for_verb(env, request)
        assert len(guard) == self.COUNT, f"cardinality guard truncated to {len(guard)} (expected {self.COUNT})"
        assert len(set(guard)) == self.COUNT, "guard set has duplicates"

        # 2. Dry-run preview set: must equal the guard set (the #1873 bug previewed
        #    only the first page while --yes --all deleted everything).
        preview = self._invoke_delete(env, dry_run=True, yes_flag=False, all_flag=False)
        assert preview["status"] == "preview"
        assert preview["session_count"] == self.COUNT
        assert preview["affected_count"] == 0
        preview_ids = preview["session_ids"]
        assert isinstance(preview_ids, list)
        assert set(preview_ids) == set(guard), "dry-run preview set diverges from the guard set"

        # Dry-run mutates nothing.
        assert len(resolve_session_ids_for_verb(env, request)) == self.COUNT

        # 3. Deleted set: --yes --all removes the entire matched set.
        result = self._invoke_delete(env, dry_run=False, yes_flag=True, all_flag=True)
        assert result["session_count"] == self.COUNT
        assert result["affected_count"] == self.COUNT, (
            f"delete truncated to {result['affected_count']} (expected {self.COUNT})"
        )

        # The archive no longer matches the query: deleted set == guard set.
        assert resolve_session_ids_for_verb(env, request) == []
