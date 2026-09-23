"""Every terminal action carries the canonical outcome, on every surface.

The envelope field is required, so the mutation the acceptance contract names —
dropping the outcome assignment on the empty path — cannot ship: the builder
raises ``ValidationError`` before any surface can serialize the result. The
adapter cases below then pin that CLI, MCP, and daemon HTTP report the same
four states over the same production routes, and that a bad input is an error
on all three rather than a silent empty on one of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from polylogue.surfaces.outcome import (
    OUTCOME_EXIT_CODES,
    UNNAMED_GAP,
    OutcomeEnvelope,
    TerminalOutcomeState,
    combine_outcomes,
    decide_outcome,
    lineage_page_outcome,
    outcome_exit_code,
    outcome_http_status,
    render_outcome_line,
)
from tests.infra.mcp import build_tools, installed_runtime_services, invoke_surface_async

ALL_STATES = ("ok", "empty", "degraded", "error")


def _seed_session(workspace_env: dict[str, Path], native_id: str = "outcome-1") -> str:
    from tests.infra.storage_records import SessionBuilder, db_setup

    builder = (
        SessionBuilder(db_setup(workspace_env), native_id)
        .provider("claude-code")
        .title("terminal outcome probe")
        .add_message(message_id=f"m-{native_id}", role="user", text="terminal outcome probe message")
    )
    builder.save()
    return builder.native_session_id()


# ---------------------------------------------------------------------------
# The one decision
# ---------------------------------------------------------------------------


class TestOutcomeDecision:
    def test_rows_present_is_ok(self) -> None:
        assert decide_outcome(matched=3).state == "ok"

    def test_no_rows_is_empty_with_a_reason(self) -> None:
        outcome = decide_outcome(matched=0)
        assert outcome.state == "empty"
        assert outcome.reason == "no_rows_in_scope"

    def test_named_gap_outranks_empty(self) -> None:
        # The point of the type: zero rows behind a gap is not an empty scope.
        outcome = decide_outcome(matched=0, degraded=("lane_unavailable:semantic",))
        assert outcome.state == "degraded"
        assert outcome.detail["gaps"] == ["lane_unavailable:semantic"]

    def test_named_gap_outranks_rows(self) -> None:
        assert decide_outcome(matched=5, degraded=("lane_failed:actions",)).state == "degraded"

    def test_error_outranks_everything(self) -> None:
        outcome = decide_outcome(matched=5, degraded=("gap",), error="index_unreadable")
        assert outcome.state == "error"
        assert outcome.reason == "index_unreadable"

    @pytest.mark.parametrize("error", [RuntimeError("SECRET_MARKER"), "internal failure: SECRET_MARKER"])
    def test_error_reason_is_a_public_code_not_exception_text(self, error: object) -> None:
        """Serialized outcomes never publish an internal exception message.

        Anti-vacuity: removing reason normalization makes the marker appear in
        the envelope JSON (and in the human renderer for the error state).
        """

        outcome = decide_outcome(matched=0, error=error)  # type: ignore[arg-type]
        serialized = json.dumps(outcome.to_dict())
        assert "SECRET_MARKER" not in serialized
        assert outcome.reason == "operation_failed"

    def test_nested_raw_gap_is_normalized_when_composed(self) -> None:
        inner = OutcomeEnvelope(
            state="degraded",
            reason="safe_gap",
            detail={"gaps": ["internal failure: SECRET_MARKER"]},
        )
        combined = combine_outcomes([inner])
        assert "SECRET_MARKER" not in json.dumps(combined.to_dict())
        assert combined.detail["gaps"] == ["safe_gap", "unnamed_gap"]

    def test_only_ok_and_empty_are_authoritative(self) -> None:
        assert decide_outcome(matched=1).rows_are_authoritative
        assert decide_outcome(matched=0).rows_are_authoritative
        assert not decide_outcome(matched=0, degraded=("gap",)).rows_are_authoritative
        assert not decide_outcome(matched=0, error="boom").rows_are_authoritative

    def test_combine_reports_the_worst_part(self) -> None:
        parts = [decide_outcome(matched=2), decide_outcome(matched=0, error="insight_unavailable:timeline")]
        combined = combine_outcomes(parts)
        assert combined.state == "degraded"
        assert combined.reason == "insight_unavailable:timeline"

    def test_combine_of_all_empty_stays_empty(self) -> None:
        assert combine_outcomes([decide_outcome(matched=0), decide_outcome(matched=0)]).state == "empty"

    def test_unnamed_error_part_cannot_compose_into_ok(self) -> None:
        """An error part with no reason still shapes the answer (polylogue-xvwpi).

        Anti-vacuity: restore the ``and entry.reason`` filter in
        ``combine_outcomes`` and this composite reads ``ok`` again.
        """

        parts = [decide_outcome(matched=2), OutcomeEnvelope(state="error", reason=None)]
        combined = combine_outcomes(parts)
        assert combined.state == "degraded"
        assert combined.reason == UNNAMED_GAP
        assert combined.detail["gaps"] == [UNNAMED_GAP]

    def test_unnamed_degraded_part_cannot_compose_into_empty(self) -> None:
        """A blank-reason degraded part outranks a genuinely empty sibling."""

        parts = [decide_outcome(matched=0), OutcomeEnvelope(state="degraded", reason="")]
        assert combine_outcomes(parts).state == "degraded"

    def test_blank_gap_reason_still_degrades_a_single_operation(self) -> None:
        """A caller that knows it has a gap but cannot name it is still degraded.

        Anti-vacuity: restore ``if reason`` filtering in ``decide_outcome``
        and this returns ``empty``.
        """

        outcome = decide_outcome(matched=0, degraded=("",))
        assert outcome.state == "degraded"
        assert outcome.detail["gaps"] == [UNNAMED_GAP]

    def test_nesting_a_composite_keeps_every_subordinate_gap(self) -> None:
        """Re-combining a composite must not drop the gaps past the first.

        Anti-vacuity: stop extending from ``entry.detail['gaps']`` and the
        outer envelope names one gap where the inner named two.
        """

        inner = combine_outcomes(
            [
                decide_outcome(matched=0, error="insight_unavailable:timeline"),
                decide_outcome(matched=0, degraded=("lane_unavailable:semantic",)),
            ]
        )
        assert inner.detail["gaps"] == ["insight_unavailable:timeline", "lane_unavailable:semantic"]
        outer = combine_outcomes([inner, decide_outcome(matched=4)])
        assert outer.state == "degraded"
        assert outer.detail["gaps"] == [
            "insight_unavailable:timeline",
            "lane_unavailable:semantic",
        ]

    def test_optional_unavailable_part_beside_a_real_empty_scope(self) -> None:
        """An optional component that could not be read is a gap, not an empty scope.

        The sibling scope really did complete with zero rows; the composite
        must still refuse to call the whole answer authoritative.
        """

        parts = [
            decide_outcome(matched=0),
            decide_outcome(matched=0, degraded=("component_unavailable:embeddings",)),
        ]
        combined = combine_outcomes(parts)
        assert combined.state == "degraded"
        assert not combined.rows_are_authoritative

    def test_truncated_lineage_page_is_degraded(self) -> None:
        outcome = lineage_page_outcome(matched=4, complete=False, truncation_reason="dangling_branch_point")
        assert outcome.state == "degraded"
        assert outcome.reason == "lineage_truncated:dangling_branch_point"

    def test_complete_lineage_page_is_ok(self) -> None:
        assert lineage_page_outcome(matched=4, complete=True, truncation_reason=None).state == "ok"


class TestTransportMapping:
    @pytest.mark.parametrize(("state", "code"), [("ok", 0), ("empty", 2), ("degraded", 1), ("error", 1)])
    def test_exit_codes(self, state: TerminalOutcomeState, code: int) -> None:
        assert outcome_exit_code(OutcomeEnvelope(state=state)) == code

    def test_exit_table_covers_the_closed_vocabulary(self) -> None:
        assert set(OUTCOME_EXIT_CODES) == set(ALL_STATES)

    def test_empty_never_exits_zero(self) -> None:
        assert outcome_exit_code(decide_outcome(matched=0)) != 0

    @pytest.mark.parametrize("state", ["ok", "empty", "degraded"])
    def test_served_envelopes_are_200(self, state: TerminalOutcomeState) -> None:
        assert outcome_http_status(OutcomeEnvelope(state=state)) == 200

    def test_error_is_a_server_failure(self) -> None:
        assert outcome_http_status(OutcomeEnvelope(state="error")) == 500

    def test_ok_renders_no_line_every_other_state_does(self) -> None:
        assert render_outcome_line(decide_outcome(matched=1)) is None
        for outcome in (
            decide_outcome(matched=0),
            decide_outcome(matched=0, degraded=("gap",)),
            decide_outcome(matched=0, error="boom"),
        ):
            line = render_outcome_line(outcome)
            assert line is not None and line.startswith("outcome: ")


# ---------------------------------------------------------------------------
# The controlled mutation: the field is required, so it cannot be dropped
# ---------------------------------------------------------------------------


class TestEnvelopesRequireAnOutcome:
    """Removing the outcome assignment on the empty path fails execution."""

    def test_query_unit_envelope_refuses_construction_without_one(self) -> None:
        from polylogue.surfaces.payloads import QueryUnitEnvelope

        with pytest.raises(ValidationError, match="outcome"):
            QueryUnitEnvelope(unit="message", query="", items=(), total=0, limit=10, offset=0)  # type: ignore[call-arg]

    def test_search_envelope_refuses_construction_without_one(self) -> None:
        from polylogue.surfaces.payloads import SearchEnvelope

        with pytest.raises(ValidationError, match="outcome"):
            SearchEnvelope(hits=(), total=0, limit=10, offset=0, query="", retrieval_lane="auto")  # type: ignore[call-arg]

    def test_session_list_response_refuses_construction_without_one(self) -> None:
        from polylogue.surfaces.payloads import SessionListResponse

        with pytest.raises(ValidationError, match="outcome"):
            SessionListResponse(items=(), total=0, limit=10, offset=0)  # type: ignore[call-arg]

    def test_facets_response_refuses_construction_without_one(self) -> None:
        from polylogue.surfaces.payloads import FacetsResponse

        with pytest.raises(ValidationError, match="outcome"):
            FacetsResponse()  # type: ignore[call-arg]

    def test_builders_supply_it_for_the_empty_page(self) -> None:
        from polylogue.surfaces.payloads import build_query_unit_envelope, build_search_envelope

        assert (
            build_query_unit_envelope([], unit="message", query="", limit=10, offset=0, has_next=False).outcome.state
            == "empty"
        )
        assert (
            build_search_envelope([], total=0, limit=10, offset=0, query="", retrieval_lane="auto").outcome.state
            == "empty"
        )


# ---------------------------------------------------------------------------
# CLI adapter
# ---------------------------------------------------------------------------


class TestCliTerminalOutcomes:
    @staticmethod
    def _run(args: list[str]) -> tuple[int, str]:
        from polylogue.cli import cli

        result = CliRunner().invoke(cli, ["--plain", "--no-daemon", *args])
        return result.exit_code, result.output

    def test_facets_over_an_empty_archive_states_its_outcome(self, workspace_env: dict[str, Path]) -> None:
        # The defect was a BARE empty, not exit 0: an aggregate that answered
        # over an empty scope succeeded, and a shell pipeline may rely on that.
        # What must never happen again is a zero-row render that says nothing.
        exit_code, output = self._run(["facets"])
        assert exit_code == 0, output
        assert "outcome: empty" in output

    def test_facets_json_carries_the_outcome(self, workspace_env: dict[str, Path]) -> None:
        exit_code, output = self._run(["facets", "--format", "json"])
        assert exit_code == 0, output
        assert json.loads(output)["outcome"]["state"] == "empty"

    def test_facets_over_a_populated_archive_is_ok(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        exit_code, output = self._run(["facets", "--format", "json"])
        assert exit_code == 0, output
        assert json.loads(output)["outcome"]["state"] == "ok"

    def test_unknown_origin_is_rejected_not_answered_empty(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        exit_code, output = self._run(["--origin", "bogus-origin", "find", "then", "select"])
        assert exit_code == 2
        assert "bogus-origin" in output

    def test_missing_session_is_an_error_not_an_empty_read(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        exit_code, output = self._run(["-i", "nonexistent-xyz", "read"])
        assert exit_code != 0, output

    def test_daemon_degraded_empty_is_not_translated_to_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        from polylogue.cli.render.outcome import emit_empty_page as _emit_no_results

        with pytest.raises(SystemExit) as exc_info:
            _emit_no_results(
                {
                    "mode": "search",
                    "outcome": decide_outcome(matched=0, degraded=("lane_unavailable:semantic",)).to_dict(),
                },
                output_format="json",
            )

        assert exc_info.value.code == OUTCOME_EXIT_CODES["degraded"]
        payload = json.loads(capsys.readouterr().out)
        assert payload["outcome"]["state"] == "degraded"


# ---------------------------------------------------------------------------
# MCP adapter
# ---------------------------------------------------------------------------


class TestMcpTerminalOutcomes:
    @pytest.mark.asyncio
    async def test_populated_scope_is_ok(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        query_fn = build_tools()["query"]
        with installed_runtime_services(workspace_env["archive_root"]):
            payload = json.loads(await invoke_surface_async(query_fn, expression="messages where role:user"))
        assert payload["outcome"]["state"] == "ok"

    @pytest.mark.asyncio
    async def test_empty_scope_is_empty_not_absent(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        query_fn = build_tools()["query"]
        with installed_runtime_services(workspace_env["archive_root"]):
            payload = json.loads(
                await invoke_surface_async(query_fn, expression="messages where text:no-such-token-anywhere")
            )
        assert payload["items"] == []
        assert payload["outcome"]["state"] == "empty"
        assert payload["outcome"]["reason"] == "no_rows_in_scope"

    @pytest.mark.asyncio
    async def test_unknown_origin_is_rejected_not_answered_empty(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        query_fn = build_tools()["query"]
        with installed_runtime_services(workspace_env["archive_root"]):
            payload = json.loads(
                await invoke_surface_async(query_fn, expression="messages where role:user", origin="bogus-origin")
            )
        assert payload.get("is_error") is True
        assert payload.get("code") == "invalid_argument"


# ---------------------------------------------------------------------------
# Daemon HTTP adapter
# ---------------------------------------------------------------------------


class TestHttpTerminalOutcomes:
    @staticmethod
    def _get(path: str) -> tuple[int, dict[str, object]]:
        from tests.infra.daemon_http_harness import capture_responses, make_daemon_handler

        handler = make_daemon_handler("GET", path)
        send_error, send_json = capture_responses(handler)
        handler.do_GET()
        if send_json.call_args is not None:
            status, payload = send_json.call_args.args
            return int(status), cast(dict[str, object], payload)
        status, code = send_error.call_args.args
        return int(status), {"error": code}

    def test_session_list_carries_the_outcome(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        status, payload = self._get("/api/sessions")
        assert status == 200
        assert cast(dict[str, object], payload["outcome"])["state"] == "ok"

    def test_empty_result_is_explicit_not_a_bare_empty_list(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        status, payload = self._get("/api/sessions?query=no-such-token-anywhere")
        assert status == 200
        assert cast(dict[str, object], payload["outcome"])["state"] == "empty"

    def test_session_messages_carry_the_outcome(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_session(workspace_env)
        status, payload = self._get(f"/api/sessions/{session_id}/messages")
        assert status == 200
        assert cast(dict[str, object], payload["outcome"])["state"] == "ok"

    def test_unknown_origin_is_rejected_not_answered_empty(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        status, payload = self._get("/api/sessions?query=probe&origin=bogus-origin")
        assert status == 400, payload
        assert payload.get("outcome") is None

    def test_missing_session_is_an_error_not_an_empty_read(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        status, payload = self._get("/api/sessions/nonexistent-xyz")
        assert status == 404, payload
