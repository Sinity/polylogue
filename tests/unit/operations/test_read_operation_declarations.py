"""Behaviour of the read operations declared for the CLI's remaining reads.

Every test here runs the real handler over a pinned reader on a seeded
synthetic archive, so a handler that answered from its own query logic instead
of the shared executors would have to reproduce these numbers by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, get_args

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.operations.daemon_protocol import (
    OperationResultContractError,
    daemon_operation_spec,
    validate_operation_result,
)
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.operations.read_contracts import SessionReadKind

_DECLARED_KINDS: tuple[str, ...] = tuple(get_args(SessionReadKind))
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path, *, count: int = 3, messages: int = 4) -> tuple[str, ...]:
    bootstrap_archive_root(root)
    session_ids: list[str] = []
    for number in range(count):
        builder = SessionBuilder(root / "index.db", f"declared-read-{number}").provider("codex").title(f"Read {number}")
        for position in range(messages):
            builder.add_message(text=f"session {number} message {position} needle")
        builder.save()
        session_ids.append(builder.native_session_id())
    return tuple(session_ids)


def _seed_hook_events(root: Path, *, origin: Origin, session_native_id: str, event_types: tuple[str, ...]) -> None:
    """Attach hook events to a seeded session by its own origin/native identity.

    Hook rows key on ``(origin, session_native_id)`` and carry no ``raw_id``
    link, so they must be written against the session's identity rather than
    any acquisition record.
    """

    with ArchiveStore(root) as archive:
        for index, event_type in enumerate(event_types):
            archive.write_hook_event(
                provider=Provider.CODEX,
                payload=b'{"event":"' + event_type.encode() + b'"}',
                source_path=f"/hooks/{index}.json",
                acquired_at_ms=1000 + index,
                hook_event=ArchiveHookEvent(
                    hook_event_id=f"hook:{session_native_id}:{index}",
                    origin=origin,
                    source_path=f"/hooks/{index}.json",
                    event_type=event_type,
                    payload={"event": event_type},
                    observed_at_ms=1000 + index,
                    native_id=f"native-{index}",
                    session_native_id=session_native_id,
                ),
                carrier_source_id="primary",
                carrier_relative_path=f"{index}.json",
            )


def _run(root: Path, name: str, payload: dict[str, object]) -> dict[str, object]:
    with open_operation_read(root) as pinned:
        result = execute_read_operation(name, payload, archive=pinned.archive, serving_identity="daemon")
    validate_operation_result(name, result)
    return result


class TestDeclaration:
    def test_every_read_requires_the_daemon(self) -> None:
        """Reads have one execution route: the resident daemon."""

        for name in ("query.aggregate", "session.read", "session.reference"):
            spec = daemon_operation_spec(name)
            assert spec is not None, name
            assert spec.fallback.value == "never", name
            assert spec.capability == "read", name


class TestQueryAggregate:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_scalar_lexical_scope_preserves_first_ranked_session_without_rich_payloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reverse: bool
    ) -> None:
        from contextlib import closing

        from polylogue.api.archive import _archive_session_identities_for_spec
        from polylogue.archive.query.spec import SessionQuerySpec

        bootstrap_archive_root(tmp_path)
        identifiers: list[str] = []
        for number, repeats in enumerate((1, 8, 3)):
            builder = SessionBuilder(tmp_path / "index.db", f"ranked-{number}").provider("codex")
            builder.working_directories([f"/synthetic/project-{part}" for part in range(20)])
            builder.add_message(text="needle " * repeats)
            builder.add_message(text="needle " + "other " * (30 - repeats))
            builder.save()
            identifiers.append(builder.native_session_id())
        with ArchiveStore(tmp_path) as writer:
            writer.add_user_tags(
                tuple(identifiers),
                tuple(f"synthetic-tag-{part}" for part in range(20)),
                author_ref="user:synthetic-scalar-scope",
                author_kind="user",
            )
        with open_operation_read(tmp_path) as pinned:
            archive = pinned.archive
            with closing(archive.iter_search_summaries("needle", reverse=reverse)) as hits:
                expected = list(dict.fromkeys(hit.session_id for hit in hits))

            def no_rich_payload(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("scalar scope loaded an unused rich payload")

            monkeypatch.setattr(ArchiveStore, "iter_search_summaries", no_rich_payload)
            monkeypatch.setattr(ArchiveStore, "read_summary", no_rich_payload)
            selected = _archive_session_identities_for_spec(
                archive, SessionQuerySpec(query_terms=("needle",), reverse=reverse)
            )
            assert [row.session_id for row in selected] == expected

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({"exclude_text": ("absent",)}, 3),
            ({"exclude_text": ("absent",), "limit": 1}, 1),
            ({"exclude_text": ("absent",), "limit": 0}, 0),
            ({"exclude_text": ("absent",), "offset": 2, "limit": 2}, 1),
            ({"exclude_text": ("needle",)}, 0),
            ({"exclude_text": ("absent",), "sample": 1}, 1),
        ],
    )
    def test_content_excluded_count_reduces_the_requested_survivor_window(
        self, tmp_path: Path, params: dict[str, object], expected: int
    ) -> None:
        _seed(tmp_path)
        result = _run(tmp_path, "query.aggregate", {"mode": "count", "params": params})
        assert result["count"] == expected

    def test_content_excluded_list_total_ignores_the_presentation_window_without_rich_hydration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polylogue.api.archive import _archive_count_sessions_for_spec
        from polylogue.archive.query.spec import SessionQuerySpec

        _seed(tmp_path)

        def no_summary(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("scalar total hydrated unused summary metadata")

        monkeypatch.setattr(ArchiveStore, "iter_summaries", no_summary)
        monkeypatch.setattr(ArchiveStore, "read_summary", no_summary)
        with open_operation_read(tmp_path) as pinned:
            total = _archive_count_sessions_for_spec(
                pinned.archive, SessionQuerySpec(exclude_text_terms=("absent",), limit=1, offset=1)
            )
        assert total == 3

    @pytest.mark.parametrize("mode", ["count", "stats", "stats_by"])
    @pytest.mark.parametrize(
        "params,expected",
        [
            ({}, 3),
            ({"query": ("needle",)}, 3),
            ({"query": ("absent-term",)}, 0),
            ({"query": ("needle",), "limit": 1}, 1),
            ({"query": ("needle",), "limit": 0}, 0),
            ({"query": ('(title:"Read 0" OR title:"absent")',)}, 1),
        ],
    )
    def test_aggregate_modes_share_actual_compound_selected_membership(
        self, tmp_path: Path, mode: str, params: dict[str, object], expected: int
    ) -> None:
        _seed(tmp_path, count=3)
        result = _run(tmp_path, "query.aggregate", {"mode": mode, "group_by": "origin", "params": params})
        if mode == "count":
            actual = result["count"]
        elif mode == "stats":
            stats = cast("dict[str, Any]", result["stats"])
            actual = stats["total_sessions"]
            assert stats["total_messages"] == expected * 4
        else:
            actual = sum(cast("dict[str, int]", result["groups"]).values())
        assert actual == expected

    @pytest.mark.parametrize("mode", ["count", "stats", "stats_by"])
    @pytest.mark.parametrize("term,expected", [("needle", 1), ("absent-term", 0)])
    def test_explicit_id_does_not_bypass_text_membership(
        self, tmp_path: Path, mode: str, term: str, expected: int
    ) -> None:
        identifiers = _seed(tmp_path, count=2)
        result = _run(
            tmp_path,
            "query.aggregate",
            {"mode": mode, "group_by": "origin", "params": {"conv_id": identifiers[0], "query": (term,)}},
        )
        if mode == "count":
            actual = result["count"]
        elif mode == "stats":
            actual = cast("dict[str, Any]", result["stats"])["total_sessions"]
        else:
            actual = sum(cast("dict[str, int]", result["groups"]).values())
        assert actual == expected

    @pytest.mark.parametrize("mode", ["count", "stats", "stats_by"])
    def test_explicit_selection_order_and_offset_precede_every_aggregate(self, tmp_path: Path, mode: str) -> None:
        bootstrap_archive_root(tmp_path)
        for number in range(3):
            builder = SessionBuilder(tmp_path / "index.db", f"ordered-{number}").provider("codex")
            for position in range(number + 1):
                builder.add_message(text=f"ordered session {number} message {position}")
            builder.save()
        result = _run(
            tmp_path,
            "query.aggregate",
            {
                "mode": mode,
                "group_by": "origin",
                "params": {"sort": "messages", "reverse": True, "limit": 1, "offset": 1},
            },
        )
        if mode == "count":
            assert result["count"] == 1
        elif mode == "stats":
            assert cast("dict[str, Any]", result["stats"])["total_messages"] == 2
        else:
            assert result["groups"] == {"codex-session": 1}

    def test_count_matches_the_archive_count_executor(self, tmp_path: Path) -> None:
        """Mutation: count from the returned page length instead of the count
        executor and a count larger than one page reads as the page size."""

        _seed(tmp_path, count=3)
        result = _run(tmp_path, "query.aggregate", {"mode": "count", "params": {}})
        assert result["count"] == 3
        assert result["mode"] == "count"

    def test_stats_by_groups_through_the_shared_aggregate(self, tmp_path: Path) -> None:
        """Grouping consumes the complete shared selection instead of narrowing filters."""

        _seed(tmp_path, count=2)
        result = _run(tmp_path, "query.aggregate", {"mode": "stats_by", "group_by": "origin", "params": {}})
        assert result["groups"] == {"codex-session": 2}

    def test_text_selection_scopes_the_aggregate_to_matched_sessions(self, tmp_path: Path) -> None:
        """Mutation: ignore the matched-session scope and a text query reports the
        whole archive's totals as if the query had selected nothing."""

        _seed(tmp_path, count=2)
        result = _run(tmp_path, "query.aggregate", {"mode": "stats", "params": {"query": ("absent-term",)}})
        stats = cast("dict[str, Any]", result["stats"])
        assert stats["total_sessions"] == 0

    def test_unknown_group_by_is_refused(self, tmp_path: Path) -> None:
        """Mutation: swallow the executor's ValueError and an unknown grouping
        silently returns an empty result instead of a typed refusal."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError):
            _run(tmp_path, "query.aggregate", {"mode": "stats_by", "group_by": "not-a-field", "params": {}})

    def test_semantic_selection_is_refused_rather_than_silently_lexical(self, tmp_path: Path) -> None:
        """Mutation: accept similar_text here and an aggregate over a semantic
        selection is computed from the lexical filters alone."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError):
            _run(tmp_path, "query.aggregate", {"mode": "count", "params": {"similar_text": "needle"}})


class TestSessionRead:
    def test_window_is_bounded_and_continues(self, tmp_path: Path) -> None:
        """Mutation: return the whole transcript and ``complete`` is true on the
        first window, so a reader never asks for the rest."""

        sessions = _seed(tmp_path, count=1, messages=6)
        first = _run(tmp_path, "session.read", {"ref": sessions[0], "limit": 2})
        assert first["total"] == 6
        assert first["next_offset"] == 2
        assert first["complete"] is False
        assert len(cast("dict[str, Any]", first["session"])["messages"]) == 2

        second = _run(tmp_path, "session.read", {"ref": sessions[0], "continuation": first["continuation"]})
        assert second["offset"] == 2
        assert second["limit"] == 2
        assert len(cast("dict[str, Any]", second["session"])["messages"]) == 2

    def test_final_window_carries_no_continuation(self, tmp_path: Path) -> None:
        """Mutation: always mint a continuation and a reader loops forever on an
        exhausted transcript."""

        sessions = _seed(tmp_path, count=1, messages=3)
        result = _run(tmp_path, "session.read", {"ref": sessions[0], "limit": 50})
        assert result["complete"] is True
        assert result["continuation"] is None
        assert result["next_offset"] is None

    def test_continuation_from_another_session_is_refused(self, tmp_path: Path) -> None:
        """Mutation: skip the continuation identity check and a token minted for
        one session pages through a different one."""

        sessions = _seed(tmp_path, count=2, messages=4)
        first = _run(tmp_path, "session.read", {"ref": sessions[0], "limit": 2})
        with pytest.raises(Exception, match="continuation"):
            _run(tmp_path, "session.read", {"ref": sessions[1], "continuation": first["continuation"]})

    def test_missing_session_is_a_typed_refusal(self, tmp_path: Path) -> None:
        """Mutation: let the KeyError escape and the operation reports an internal
        failure rather than a missing reference."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError, match="session not found"):
            _run(tmp_path, "session.read", {"ref": "codex-session:absent"})


class TestSessionReference:
    def test_a_non_reference_expression_is_refused(self, tmp_path: Path) -> None:
        """Mutation: fall through to a normal query and a mistyped reference root
        quietly returns session rows instead of naming the error."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError, match="not a reference operand"):
            _run(tmp_path, "session.reference", {"expression": "origin:codex-session"})

    def test_an_unknown_reference_is_named(self, tmp_path: Path) -> None:
        """Mutation: return an empty member list and an unknown reference is
        indistinguishable from an empty one."""

        _seed(tmp_path, count=1)
        with pytest.raises(ValueError, match="reference not found"):
            _run(tmp_path, "session.reference", {"expression": f"from query:{'a' * 64}"})


class TestResultContracts:
    def test_an_aggregate_body_from_another_mode_is_rejected(self) -> None:
        """Mutation: relax the result model and a count result can carry a stats
        body that no renderer would ever show."""

        with pytest.raises(OperationResultContractError):
            validate_operation_result(
                "query.aggregate",
                {"outcome": {"state": "ok"}, "mode": "count", "count": 1, "groups": {"a": 1}},
            )

    def test_a_window_cannot_claim_completeness_and_a_next_offset(self) -> None:
        """Mutation: relax the result model and a truncated read can report
        itself complete."""

        with pytest.raises(OperationResultContractError):
            validate_operation_result(
                "session.read",
                {
                    "outcome": {"state": "ok"},
                    "session": {},
                    "session_id": "codex-session:x",
                    "total": 10,
                    "limit": 2,
                    "offset": 0,
                    "next_offset": 2,
                    "continuation": "q2.token",
                    "complete": True,
                },
            )


class TestSessionReadEvidenceKinds:
    """``session.read`` also serves per-session evidence relations (design D3).

    These relations have no query-grammar unit of their own, so routing them
    through ``session.read`` is what lets their read views stop opening an
    archive.
    """

    def test_hooks_kind_returns_the_pinned_hook_read_model_whole(self, tmp_path: Path) -> None:
        """Mutation: answer the hooks kind from the transcript branch and the
        result carries a message window with no hook evidence at all."""

        sessions = _seed(tmp_path, count=1, messages=2)
        _seed_hook_events(
            tmp_path,
            origin=Origin.CODEX_SESSION,
            session_native_id="ext-declared-read-0",
            event_types=("PreToolUse", "PostToolUse", "PostToolUse"),
        )
        result = _run(tmp_path, "session.read", {"ref": sessions[0], "kind": "hooks"})

        assert result["kind"] == "hooks"
        assert result["complete"] is True
        assert result["next_offset"] is None
        assert result["continuation"] is None
        evidence = cast("dict[str, Any]", result["evidence"])
        assert evidence["total"] == 3
        assert evidence["by_event_type"] == {"PostToolUse": 2, "PreToolUse": 1}
        # An evidence read is not a message page.
        assert cast("dict[str, Any]", result["session"])["messages"] == []

    def test_a_session_with_no_hooks_is_an_empty_outcome_not_a_refusal(self, tmp_path: Path) -> None:
        """Mutation: raise on an absent hook spool and a session that simply
        recorded no hooks becomes indistinguishable from a missing session."""

        sessions = _seed(tmp_path, count=1, messages=2)
        result = _run(tmp_path, "session.read", {"ref": sessions[0], "kind": "hooks"})
        assert cast("dict[str, Any]", result["evidence"])["total"] == 0
        assert cast("dict[str, Any]", result["outcome"])["state"] == "empty"

    def test_a_missing_session_is_refused_rather_than_answered_empty(self, tmp_path: Path) -> None:
        """Mutation: return an empty evidence body for an unknown reference and
        a typo reads as a session that recorded no hooks."""

        _seed(tmp_path, count=1, messages=1)
        with pytest.raises(ValueError):
            _run(tmp_path, "session.read", {"ref": "codex-session:absent", "kind": "hooks"})

    def test_an_unserved_kind_is_refused_by_name(self, tmp_path: Path) -> None:
        """Mutation: fall through to the transcript branch for an unknown kind
        and a caller silently receives a transcript it did not ask for.

        The kind has to be one ``session.read`` genuinely does not serve.
        ``events`` was used here until it graduated onto the windowed evidence
        contract (``SESSION_EVIDENCE_PAGE_READERS``), after which a seeded session
        answered it with a valid empty page and this assertion could never
        hold -- so the refusal it exists to pin went unchecked.
        """

        sessions = _seed(tmp_path, count=1, messages=1)
        assert "attachments" not in _DECLARED_KINDS
        with pytest.raises(ValueError):
            _run(tmp_path, "session.read", {"ref": sessions[0], "kind": "attachments"})

    def test_every_declared_kind_is_actually_served(self, tmp_path: Path) -> None:
        """The opposite direction, so the refusal above cannot pass by refusing
        everything: each kind the contract declares answers for a real session.

        Mutation: drop a kind's reader from ``_SESSION_EVIDENCE_READERS`` or
        ``SESSION_EVIDENCE_PAGE_READERS`` while leaving it in ``SessionReadKind``
        and this goes red naming that kind, which is the exact drift that left
        the refusal test above vacuous in the other direction.
        """

        sessions = _seed(tmp_path, count=1, messages=1)
        served = {
            kind: _run(tmp_path, "session.read", {"ref": sessions[0], "kind": kind})["session_id"]
            for kind in _DECLARED_KINDS
        }
        assert set(served) == set(_DECLARED_KINDS)
        assert set(served.values()) == {sessions[0]}

    def test_a_transcript_result_cannot_carry_an_evidence_body(self) -> None:
        """Mutation: relax the result model and a transcript window can smuggle
        an evidence body no renderer would ever show."""

        with pytest.raises(OperationResultContractError):
            validate_operation_result(
                "session.read",
                {
                    "outcome": {"state": "ok"},
                    "session": {},
                    "session_id": "codex-session:x",
                    "kind": "transcript",
                    "evidence": {"total": 0},
                    "total": 0,
                    "limit": 1,
                    "offset": 0,
                    "next_offset": None,
                    "continuation": None,
                    "complete": True,
                },
            )

    def test_an_evidence_result_must_carry_its_body(self) -> None:
        """Mutation: relax the result model and an evidence read can report
        success while returning nothing."""

        with pytest.raises(OperationResultContractError):
            validate_operation_result(
                "session.read",
                {
                    "outcome": {"state": "ok"},
                    "session": {},
                    "session_id": "codex-session:x",
                    "kind": "hooks",
                    "total": 0,
                    "limit": 1,
                    "offset": 0,
                    "next_offset": None,
                    "continuation": None,
                    "complete": True,
                },
            )


class TestSelectedDomainRead:
    def test_domain_pages_bind_the_real_query_view(self, tmp_path: Path) -> None:
        from polylogue.archive.session.domain_models import Session

        sessions = _seed(tmp_path, count=1, messages=3)
        selected = _run(tmp_path, "cli.query", {"params": {}})
        epoch = selected["snapshot_epoch"]
        payload = {"ref": sessions[0], "limit": 2, "session_projection": "domain", "selection_epoch": epoch}
        first = _run(tmp_path, "session.read", payload)
        domain = Session.model_validate(first["session"])
        assert domain.id == sessions[0]
        assert len(domain.messages) == 2
        assert first["selection_epoch"] == epoch
        assert first["next_offset"] == 2
        second = _run(tmp_path, "session.read", {**payload, "offset": 2, "continuation": first["continuation"]})
        assert len(Session.model_validate(second["session"]).messages) == 1
        assert second["complete"] is True
        assert cast(dict[str, object], second["outcome"])["state"] == "ok"

    def test_stale_selection_refuses_before_hydration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from polylogue.archive.query.transaction import QueryContinuationStaleError

        sessions = _seed(tmp_path, count=1)
        reached = []

        def read(*_a, **_k):  # type: ignore[no-untyped-def]
            reached.append(True)
            raise AssertionError("stale selection reached hydration")

        monkeypatch.setattr(ArchiveStore, "read_session_page", read)
        with pytest.raises(QueryContinuationStaleError) as refused:
            _run(
                tmp_path,
                "session.read",
                {"ref": sessions[0], "session_projection": "domain", "selection_epoch": "old-view"},
            )
        assert refused.value.issued_epoch == "old-view"
        assert refused.value.current_epoch != "old-view"
        assert reached == []

    def test_domain_continuation_refuses_projection_change(self, tmp_path: Path) -> None:
        from polylogue.archive.query.transaction import QueryContinuationInvalidError

        sessions = _seed(tmp_path, count=1, messages=3)
        first = _run(tmp_path, "session.read", {"ref": sessions[0], "session_projection": "domain", "limit": 1})
        with pytest.raises(QueryContinuationInvalidError):
            _run(tmp_path, "session.read", {"ref": sessions[0], "continuation": first["continuation"]})

    def test_every_selected_registered_route_rejects_a_stale_view(self, tmp_path: Path) -> None:
        from polylogue.archive.query.transaction import QueryContinuationStaleError

        sessions = _seed(tmp_path, count=1)
        routes: dict[str, dict[str, object]] = {
            "read.dialogue": {"session_id": sessions[0]},
            "read.temporal": {"session_id": sessions[0]},
            "read.effective_context": {"session_id": sessions[0]},
            "read.orchestration": {"session_id": sessions[0]},
            "read.lineage": {"session_id": sessions[0]},
            "read.topology": {"session_id": sessions[0]},
            "read.neighbors": {"session_id": sessions[0]},
            "read.correlation": {"session_id": sessions[0]},
            "read.context": {"session_id": sessions[0], "observed_at": "2026-01-01T00:00:00+00:00"},
            "read.context-image": {"seed_session_id": sessions[0], "observed_at_ms": 1},
            "session.read": {"ref": sessions[0], "kind": "messages"},
        }
        for operation, operands in routes.items():
            payload: dict[str, object] = {**operands, "selection_epoch": "stale-selected-view"}
            declaration = daemon_operation_spec(operation)
            assert declaration is not None and declaration.request_model is not None
            declaration.request_model.model_validate(payload)
            with pytest.raises(QueryContinuationStaleError) as refused:
                _run(tmp_path, operation, payload)
            assert refused.value.issued_epoch == "stale-selected-view"

    def test_domain_projection_preserves_tool_blocks(self, tmp_path: Path) -> None:
        from polylogue.archive.session.domain_models import Session

        bootstrap_archive_root(tmp_path)
        builder = SessionBuilder(tmp_path / "index.db", "domain-tools").provider("codex")
        builder.add_message(
            role="assistant",
            text="calling tool",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Read",
                    "tool_id": "original-tool-id",
                    "tool_input": '{"path":"/synthetic/input.txt"}',
                }
            ],
        )
        builder.save()
        result = _run(tmp_path, "session.read", {"ref": builder.native_session_id(), "session_projection": "domain"})
        session = Session.model_validate(result["session"])
        from polylogue.archive.message.models import Message

        blocks = Message.model_validate(list(session.messages)[0]).blocks
        assert any(block.get("tool_id") == "original-tool-id" and block.get("tool_name") == "Read" for block in blocks)


@pytest.mark.parametrize("selected", [True, False])
def test_temporal_text_selection_preserves_an_explicit_session_reference(tmp_path: Path, selected: bool) -> None:
    sessions = _seed(tmp_path, count=2, messages=1)
    result = _run(
        tmp_path,
        "read.temporal",
        {"session_id": sessions[1] if selected else None, "params": {"contains": ["needle"]}},
    )
    body = cast(dict[str, Any], result["payload"])
    events = body["temporal_window"]["events"]
    refs = {ref for event in events for ref in event["evidence_refs"] if ref.startswith("session:")}
    assert refs == {f"session:{sid}" for sid in (sessions[1:] if selected else sessions)}
    assert {event["family"] for event in events} == {"archive-session", "archive-message"}


def test_temporal_missing_selected_reference_does_not_widen_to_text_matches(tmp_path: Path) -> None:
    _seed(tmp_path, count=2, messages=1)
    result = _run(
        tmp_path,
        "read.temporal",
        {"session_id": "codex-session:missing", "params": {"contains": ["needle"]}},
    )
    body = cast(dict[str, Any], result["payload"])
    assert body["temporal_window"]["events"] == []
