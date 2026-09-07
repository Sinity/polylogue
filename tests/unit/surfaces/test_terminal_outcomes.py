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

    def test_facets_over_an_empty_archive_is_never_bare_exit_zero(self, workspace_env: dict[str, Path]) -> None:
        exit_code, output = self._run(["facets"])
        assert exit_code == OUTCOME_EXIT_CODES["empty"]
        assert "outcome: empty" in output

    def test_facets_json_carries_the_outcome(self, workspace_env: dict[str, Path]) -> None:
        exit_code, output = self._run(["facets", "--format", "json"])
        assert exit_code == OUTCOME_EXIT_CODES["empty"]
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

    def test_unknown_origin_is_rejected_not_answered_empty(self, workspace_env: dict[str, Path]) -> None:
        _seed_session(workspace_env)
        status, payload = self._get("/api/sessions?query=probe&origin=bogus-origin")
        assert status == 400, payload
        assert payload.get("outcome") is None
