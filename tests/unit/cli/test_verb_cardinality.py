"""Tests for cardinality enforcement in mark / analyze / delete query verbs.

Cardinality rules (from #1814):
  - mark:   requires exactly one result unless --all or --first.
  - analyze: no cardinality restriction (applies to result set).
  - delete: requires --dry-run for preview; --yes plus --all for multi-match.

All three verbs share the single :func:`check_cardinality` path from
``polylogue.cli.verb_cardinality``.
"""

from __future__ import annotations

from types import SimpleNamespace
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
            ) as mock_resolve,
            patch(
                "polylogue.cli.query_verbs._execute_query_verb",
            ) as mock_exec,
        ):
            self._call_delete(child, dry_run=True)

        # resolve must NOT be called — dry-run skips the cardinality check.
        mock_resolve.assert_not_called()
        mock_exec.assert_called_once()

    def test_dry_run_delegates_with_dry_run_true(self) -> None:
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        captured_request: list[RootModeRequest] = []

        def _capture(ctx: click.Context, req: RootModeRequest) -> None:
            captured_request.append(req)

        with (
            patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb"),
            patch("polylogue.cli.query_verbs._execute_query_verb", side_effect=_capture),
        ):
            self._call_delete(child, dry_run=True)

        assert captured_request, "delete_verb must call _execute_query_verb for dry-run"
        assert captured_request[0].params.get("dry_run") is True

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
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1", "id2"],
            ),
            patch("polylogue.cli.query_verbs._execute_query_verb") as mock_exec,
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
        _, child = _context_pair()
        child.obj = SimpleNamespace(config=MagicMock())

        captured: list[RootModeRequest] = []

        def _capture(ctx: click.Context, req: RootModeRequest) -> None:
            captured.append(req)

        with (
            patch(
                "polylogue.cli.verb_cardinality.resolve_session_ids_for_verb",
                return_value=["id1"],
            ),
            patch("polylogue.cli.query_verbs._execute_query_verb", side_effect=_capture),
        ):
            self._call_delete(child, yes_flag=True)

        assert captured[0].params.get("force") is True, "--yes must set force on the delegated request"


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
            cb(child, None, False, None)  # type: ignore[operator]

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
            cb(child, "origin", False, None)  # type: ignore[operator]

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
            cb(child, None, False, None)  # type: ignore[operator]

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
