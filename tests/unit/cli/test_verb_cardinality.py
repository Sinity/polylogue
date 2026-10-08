"""Tests for cardinality enforcement in read / mark / analyze / delete verbs.

Cardinality rules (from #1814):
  - read:   requires exactly one result unless --all or --first.
  - mark:   requires exactly one result unless --all or --first.
  - analyze: no cardinality restriction (applies to result set).
  - delete: requires --dry-run for preview; --yes plus --all for multi-match.

Read cardinality uses the shared guard. Mutating verbs delegate matching and
cardinality to the resident operation before accepting any durable effect.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import Result

from polylogue.cli import query_verbs
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.select import SelectSessionRow
from polylogue.cli.verb_cardinality import CardinalityError, check_cardinality
from tests.infra.cli_selection import selection_for_rows
from tests.infra.daemon_operations import cli_daemon_archive

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
# read_verb — cardinality enforcement
# ---------------------------------------------------------------------------


class TestReadVerbCardinality:
    """read_verb enforces the singleton / --all / --first contract."""

    def _read_callback(self) -> object:
        cb = getattr(query_verbs.read_verb.callback, "__wrapped__", None)
        assert callable(cb), "read_verb.callback must be a context-decorated function"
        return cb

    def _call_read(
        self,
        child: click.Context,
        *,
        view: str = "summary",
        all_matches: bool = False,
        first_only: bool = False,
    ) -> None:
        cb = self._read_callback()
        cb(  # type: ignore[operator]
            child,
            view=view,
            destination="terminal",
            output_format=None,
            render_expr=None,
            projection_expr=None,
            render_layout=None,
            timestamp_policy=None,
            out_path=None,
            all_matches=all_matches,
            full=False,
            limit=None,
            offset=0,
            window_hours=24,
            repo_path=None,
            since_hours=2,
            confidence_threshold=0.3,
            github_api=False,
            related_limit=5,
            max_sessions=5,
            max_tokens=None,
            include_assertions=False,
            no_redact=False,
            fields=None,
            first_only=first_only,
            show_spec=False,
            show_views=False,
        )

    def _read_resolution(self, session_ids: list[str]) -> AbstractContextManager[MagicMock]:
        """Stand in for the ``cli.query`` rows the read verb resolves against.

        Resolution is now one declared operation through the kernel, so the
        seam a cardinality test controls is the selector rows that operation
        reports, not a coroutine runner.
        """
        rows = [
            SelectSessionRow(session_id=session_id, origin="claude-code-session", title=session_id, date=None)
            for session_id in session_ids
        ]
        return patch("polylogue.cli.session_rows.query_session_selection", return_value=selection_for_rows(rows))

    def test_single_session_view_multi_match_without_first_or_all_raises(self) -> None:
        _, child = _context_pair(query_terms=("needle",))
        child.obj = SimpleNamespace(config=MagicMock())

        with self._read_resolution(["id1", "id2"]):
            with pytest.raises(click.UsageError, match="select first"):
                self._call_read(child, view="messages")

    def test_query_set_view_multi_match_projects_without_cardinality_guard(self) -> None:
        _, child = _context_pair(query_terms=("needle",))
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            patch("polylogue.cli.session_rows.query_session_selection") as query_rows,
            patch("polylogue.cli.query_verbs.run_read_view") as run_read_view,
        ):
            self._call_read(child, view="temporal")

        query_rows.assert_not_called()
        invocation = run_read_view.call_args.args[2]
        assert invocation.view == "temporal"
        assert invocation.session_id is None

    def test_multi_match_with_first_reads_first_match(self) -> None:
        _, child = _context_pair(query_terms=("needle",))
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            self._read_resolution(["id1"]),
            patch("polylogue.cli.query_verbs.run_read_view") as run_read_view,
        ):
            self._call_read(child, first_only=True)

        invocation = run_read_view.call_args.args[2]
        assert invocation.session_id == "id1"

    def test_read_all_and_first_are_mutually_exclusive(self) -> None:
        _, child = _context_pair(query_terms=("needle",))
        child.obj = SimpleNamespace(config=MagicMock())

        with pytest.raises(click.UsageError, match="mutually exclusive"):
            self._call_read(child, all_matches=True, first_only=True)

    def test_read_uses_shared_check_cardinality(self) -> None:
        _, child = _context_pair(query_terms=("needle",))
        child.obj = SimpleNamespace(config=MagicMock())

        with (
            self._read_resolution(["id1", "id2"]),
            patch(
                "polylogue.cli.verb_cardinality.check_cardinality",
                side_effect=CardinalityError("mocked error"),
            ) as mock_check,
        ):
            with pytest.raises(click.UsageError, match="select first"):
                self._call_read(child, view="messages")
        mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# mark_verb — cardinality enforcement
# ---------------------------------------------------------------------------


@pytest.fixture
def resident_cardinality_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.storage_records import SessionBuilder

    initialize_active_archive_root(tmp_path)
    for index in range(3):
        SessionBuilder(tmp_path / "index.db", f"conv-{index}").provider("claude-ai").add_message(
            f"m{index}", role="user", text="cardinality evidence"
        ).save()
    with cli_daemon_archive(tmp_path, monkeypatch):
        yield tmp_path


def _resident_verb(root: Path, expression: str, verb: str, *flags: str) -> Result:
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    return CliRunner().invoke(cli, ["--plain", "--format", "json", "find", expression, "then", verb, *flags])


def _resident_marks(root: Path) -> list[dict[str, str]]:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore.open_existing(root, read_only=True) as archive:
        return list(archive.list_marks())


class TestMarkVerbCardinality:
    """The resident canonical query decides cardinality before any User effect."""

    @pytest.mark.parametrize(
        ("expression", "flags", "count"),
        [
            ("id:ext-conv-0", (), 1),
            ("origin:claude-ai-export", ("--all",), 3),
            ("origin:claude-ai-export", ("--first",), 1),
        ],
    )
    def test_singleton_all_and_first_apply_the_exact_scope(
        self, resident_cardinality_archive: Path, expression: str, flags: tuple[str, ...], count: int
    ) -> None:
        root = resident_cardinality_archive
        result = _resident_verb(root, expression, "mark", "--star", *flags)
        assert result.exit_code == 0, (result.output, result.exception)
        assert json.loads(result.output)["session_count"] == count
        assert len(_resident_marks(root)) == count

    @pytest.mark.parametrize("expression", ["origin:chatgpt-export", "origin:claude-ai-export"])
    def test_empty_and_ambiguous_default_refuse_without_effect(
        self, resident_cardinality_archive: Path, expression: str
    ) -> None:
        root = resident_cardinality_archive
        result = _resident_verb(root, expression, "mark", "--star")
        assert result.exit_code != 0, result.output
        assert _resident_marks(root) == []

    @pytest.mark.parametrize(
        "flags",
        [("--first", "--all", "--star"), ("--star", "--unstar"), ("--tag-add", "x", "--tag-remove", "x")],
    )
    def test_all_incompatible_intents_refuse_before_any_effect(
        self, resident_cardinality_archive: Path, flags: tuple[str, ...]
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        root = resident_cardinality_archive
        result = _resident_verb(root, "id:ext-conv-0", "mark", *flags)
        assert result.exit_code != 0, result.output
        assert _resident_marks(root) == []
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            assert archive.read_summary("claude-ai-export:ext-conv-0").tags == ()

    def test_no_flags_is_an_explicit_no_op(self, resident_cardinality_archive: Path) -> None:
        root = resident_cardinality_archive
        result = _resident_verb(root, "origin:claude-ai-export", "mark")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["affected_count"] == 0
        assert _resident_marks(root) == []

    @pytest.mark.parametrize(
        ("add", "remove", "kind"),
        [("--star", "--unstar", "star"), ("--pin", "--unpin", "pin"), ("--archive", "--unarchive", "archive")],
    )
    def test_every_mark_type_adds_and_removes_on_the_same_scope(
        self, resident_cardinality_archive: Path, add: str, remove: str, kind: str
    ) -> None:
        root = resident_cardinality_archive
        added = _resident_verb(root, "id:ext-conv-0", "mark", add)
        assert added.exit_code == 0, added.output
        assert [row["mark_type"] for row in _resident_marks(root)] == [kind]
        removed = _resident_verb(root, "id:ext-conv-0", "mark", remove)
        assert removed.exit_code == 0, removed.output
        assert _resident_marks(root) == []

    def test_tag_removal_and_note_share_the_authoritative_batch(self, resident_cardinality_archive: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        root = resident_cardinality_archive
        added = _resident_verb(root, "id:ext-conv-0", "mark", "--tag-add", "reviewed")
        assert added.exit_code == 0, added.output
        removed = _resident_verb(root, "id:ext-conv-0", "mark", "--tag-remove", "reviewed", "--note", "neutral note")
        assert removed.exit_code == 0, removed.output
        payload = json.loads(removed.output)
        assert payload["reference"]["part_count"] == 2
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            assert archive.read_summary("claude-ai-export:ext-conv-0").tags == ()
            assert len(archive.list_annotations()) == 1


class TestDeleteVerbCardinality:
    """Dry runs and committed deletes use the same resident selection contract."""

    @pytest.mark.parametrize(
        ("expression", "flags", "count"), [("id:ext-conv-0", (), 1), ("origin:claude-ai-export", ("--all",), 3)]
    )
    def test_preview_and_apply_have_identical_scope(
        self, resident_cardinality_archive: Path, expression: str, flags: tuple[str, ...], count: int
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        root = resident_cardinality_archive
        preview = _resident_verb(root, expression, "delete", "--dry-run", *flags)
        assert preview.exit_code == 0, preview.output
        data = json.loads(preview.output)
        assert (data["status"], data["session_count"], data["affected_count"]) == ("preview", count, 0)
        applied = _resident_verb(root, expression, "delete", "--yes", *flags)
        assert applied.exit_code == 0, applied.output
        assert json.loads(applied.output)["affected_count"] == count
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            assert archive.count_sessions() == 3 - count

    @pytest.mark.parametrize("flags", [("--dry-run",), ("--yes",), ()])
    def test_ambiguous_or_unconfirmed_delete_preserves_every_session(
        self, resident_cardinality_archive: Path, flags: tuple[str, ...]
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        root = resident_cardinality_archive
        result = _resident_verb(root, "origin:claude-ai-export", "delete", *flags)
        assert result.exit_code != 0, result.output
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            assert archive.count_sessions() == 3

    def test_empty_preview_reports_zero_without_authority(self, resident_cardinality_archive: Path) -> None:
        root = resident_cardinality_archive
        result = _resident_verb(root, "origin:chatgpt-export", "delete", "--dry-run", "--all")
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["session_count"] == payload["affected_count"] == 0
        assert "reference" not in payload


# ---------------------------------------------------------------------------
# delete_verb — cardinality enforcement (updated verb)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# require_exact_mutation_selection — --sample is rejected, never silently ignored
# ---------------------------------------------------------------------------


class TestSampleRejectedForMutatingVerbs:
    """``--sample`` must not silently widen a mutating verb's blast radius.

    ``--sample N`` is a display-window random subset; the resident selection
    walk deliberately resolves the COMPLETE matched set. Honoring it would mean
    a destructive ``delete``/``mark`` operated on every match while the
    operator believed only N rows were in scope. The verb guard rejects the
    combination up front rather than ignoring it.
    """

    def test_guard_rejects_sample(self) -> None:
        from polylogue.cli.verb_cardinality import require_exact_mutation_selection

        request = RootModeRequest.from_params({"sample": 5})
        assert request.query_spec().sample == 5

        with pytest.raises(click.UsageError, match="--sample"):
            require_exact_mutation_selection(request, allow_all=True, operation="delete")

    def test_guard_allows_absent_sample(self) -> None:
        """Without --sample the guard lets the verb reach its resident walk."""
        from polylogue.cli.verb_cardinality import require_exact_mutation_selection

        request = RootModeRequest.from_params({})
        assert request.query_spec().sample is None

        require_exact_mutation_selection(request, allow_all=True, operation="delete")


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
            cb(child, False, None, False, False, None, "linear", False, None, None)  # type: ignore[operator]

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
            cb(child, False, "origin", False, False, None, "linear", False, None, None)  # type: ignore[operator]

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
            cb(child)  # type: ignore[operator]

        mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 1 regression: delete truncation (#1873)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Non-mocked >50-session delete cardinality evidence (#1873 recovery pack)
# ---------------------------------------------------------------------------


class TestDeleteCardinalityLargeNonMocked:
    """End-to-end evidence over a real seeded archive (no mocks on resolution/delete).

    The invariant that makes ``delete --yes --all`` safe is that three sets are
    identical and none is silently page-limited:

        canonical query set  ==  resident preview set  ==  deleted set

    The default query page limit is 20 (50 in some paths); seeding 60 matching
    sessions makes any truncation observable. This exercises the real resident preview, authorization and bound executor.
    The CLI carries a bounded sample and durable reference for the complete set.
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

    def _invoke_delete(
        self,
        env: object,
        *,
        dry_run: bool,
        yes_flag: bool,
        all_flag: bool,
        output_format: str | None = None,
    ) -> dict[str, object]:

        _, child = _context_pair(query_terms=(self.TOKEN,))
        child.obj = env
        self._delete_callback()(child, dry_run, yes_flag, all_flag, output_format)  # type: ignore[operator]
        # The resident delete adapter prints exactly one JSON document to stdout.
        captured = self._capsys.readouterr().out.strip()
        return cast(dict[str, object], json.loads(captured))

    @pytest.fixture(autouse=True)
    def _bind_capsys(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._capsys = capsys

    def test_guard_dry_run_and_deleted_sets_are_identical_and_unlimited(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polylogue.cli.session_rows import query_complete_session_selection
        from tests.infra.app_env import make_app_env

        index_db = workspace_env["archive_root"] / "index.db"
        self._seed(index_db)

        with cli_daemon_archive(workspace_env["archive_root"], monkeypatch):
            env = make_app_env(archive_root=workspace_env["archive_root"])
            request = RootModeRequest.from_params({"query": (self.TOKEN,)})

            # 1. Guard set: the full matched set, not a default page.
            guard = query_complete_session_selection(env.config, request).ids
            assert len(guard) == self.COUNT, f"cardinality guard truncated to {len(guard)} (expected {self.COUNT})"
            assert len(set(guard)) == self.COUNT, "guard set has duplicates"

            # 2. Dry-run preview set: must equal the guard set (the #1873 bug previewed
            #    only the first page while --yes --all deleted everything).
            with pytest.raises(click.UsageError) as ambiguous:
                self._invoke_delete(env, dry_run=True, yes_flag=False, all_flag=False)
            from polylogue.cli.verb_cardinality import AmbiguousCardinalityError

            assert isinstance(ambiguous.value, AmbiguousCardinalityError)
            assert ambiguous.value.bounded is True

            preview = self._invoke_delete(env, dry_run=True, yes_flag=False, all_flag=True)
            assert preview["status"] == "preview"
            assert preview["session_count"] == self.COUNT
            assert preview["affected_count"] == 0
            sample = preview["session_ids_sample"]
            assert isinstance(sample, list) and len(sample) == 20
            assert set(sample).issubset(guard)
            reference = preview["reference"]
            assert isinstance(reference, dict)
            assert reference["artifact_kind"] == "preview-batch"

            # Dry-run mutates nothing.
            assert len(query_complete_session_selection(env.config, request).ids) == self.COUNT

            # 3. Deleted set: --yes --all removes the entire matched set.
            result = self._invoke_delete(env, dry_run=False, yes_flag=True, all_flag=True)
            assert result["session_count"] == self.COUNT
            assert result["affected_count"] == self.COUNT, (
                f"delete truncated to {result['affected_count']} (expected {self.COUNT})"
            )

            # The archive no longer matches the query: deleted set == guard set.
            assert query_complete_session_selection(env.config, request).ids == []


def test_temporal_cli_first_with_text_keeps_only_the_resolved_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.storage_records import SessionBuilder

    bootstrap_archive_root(tmp_path)
    for index in range(2):
        builder = SessionBuilder(tmp_path / "index.db", f"temporal-{index}").provider("codex")
        builder.created_at(f"2026-01-0{index + 1}T00:00:00Z").updated_at(f"2026-01-0{index + 1}T00:00:00Z")
        builder.add_message(text="needle temporal evidence").save()
    from polylogue.archive.query.expression import compile_expression
    from polylogue.archive.query.filter_kwargs import plan_filter_kwargs

    with ArchiveStore.open_existing(tmp_path) as archive:
        expected = archive.list_summaries(limit=1, **plan_filter_kwargs(compile_expression("needle").to_plan()))[
            0
        ].session_id
    with cli_daemon_archive(tmp_path, monkeypatch):
        result = _resident_verb(tmp_path, "needle", "read", "--view", "temporal", "--first")
    assert result.exit_code == 0, result.output
    events = json.loads(result.output)["temporal_window"]["events"]
    refs = {ref for event in events for ref in event["evidence_refs"] if ref.startswith("session:")}
    assert refs == {f"session:{expected}"}
