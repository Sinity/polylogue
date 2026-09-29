"""Worker 12 regression sources through CLI adapters and the real lowering seam."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from polylogue import Polylogue
from polylogue.cli import archive_query, operation_kernel
from polylogue.cli.click_app import cli
from polylogue.cli.commands.maintenance._raw_identity import raw_authority_frontier_command
from polylogue.cli.operation_kernel import OperationFailedError, OperationRequest, OperationUnavailableError
from polylogue.config import Config
from tests.infra.frozen_clock import FrozenClock


def test_session_window_continuation_keeps_remaining_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """60.01: dropping the remaining limit asks for the oversized, unrequested sixth row."""
    monkeypatch.setattr(archive_query, "_SESSION_READ_WINDOW", 4)
    requests: list[OperationRequest] = []

    def read(_config: Config, request: OperationRequest, **_kwargs):
        requests.append(request)
        resumed = request.payload.get("continuation") is not None
        if resumed and request.payload.get("limit") != 1:
            raise OperationFailedError("result_too_large", "unrequested sixth message is oversized")
        start, count = (4, 1) if resumed else (0, 4)
        return {
            "session": {"session_id": "w12", "messages": [{"position": i} for i in range(start, start + count)]},
            "complete": False,
            "continuation": "fixture-after-five" if resumed else "fixture-after-four",
        }, {}

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    result = archive_query._read_session_windows(
        cast(Config, object()), "session:w12", daemon_disabled=True, message_limit=5
    )
    assert [request.payload.get("limit") for request in requests] == [4, 1]
    assert result["messages"] == [{"position": i} for i in range(5)]


def test_session_window_retries_resumed_pages_without_widening(monkeypatch: pytest.MonkeyPatch) -> None:
    """60.01: an oversized continuation must narrow in place and keep the successful bound."""
    monkeypatch.setattr(archive_query, "_SESSION_READ_WINDOW", 4)
    requests: list[tuple[int, object]] = []

    def read(_config: Config, request: OperationRequest, **_kwargs):
        size = request.payload.get("limit")
        assert isinstance(size, int)
        cursor = request.payload.get("continuation")
        start = int(str(cursor).removeprefix("after-")) if cursor is not None else 0
        requests.append((size, cursor))
        if start == 4 and size > 1:
            raise OperationFailedError("result_too_large")
        end = min(start + size, 7)
        return {
            "session": {"session_id": "w12", "messages": [{"position": i} for i in range(start, end)]},
            "complete": end == 7,
            "continuation": f"after-{end}" if end < 7 else None,
        }, {}

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    result = archive_query._read_session_windows(cast(Config, object()), "session:w12", daemon_disabled=True)
    assert requests == [(4, None), (4, "after-4"), (2, "after-4"), (1, "after-4"), (1, "after-5"), (1, "after-6")]
    assert result["messages"] == [{"position": i} for i in range(7)]


def test_raw_frontier_labels_all_plans_not_outstanding_obligations(monkeypatch: pytest.MonkeyPatch) -> None:
    """60.20: a discharged plan must not be described as an outstanding obligation."""
    from polylogue.maintenance import raw_authority

    payload = {
        "pass_id": "w12-pass",
        "accepted_head_count": 1,
        "plan_count": 1,
        "state_counts": {"discharged": 1},
    }
    monkeypatch.setattr(raw_authority, "inspect_frontier", lambda _config: SimpleNamespace(to_dict=lambda: payload))
    result = CliRunner().invoke(raw_authority_frontier_command, [], obj=SimpleNamespace(config=object()))
    assert result.exit_code == 0, result.output
    assert "plans=1" in result.output
    assert "obligations=" not in result.output


@pytest.mark.parametrize("projection", ["postmortem", "portfolio"])
def test_named_analysis_renders_inherited_json(projection: str) -> None:
    """62.01: reading the local None option selects plain text instead of the inherited JSON renderer."""
    with (
        patch.object(Polylogue, f"{projection}_bundle", new_callable=AsyncMock, return_value=object()),
        patch("polylogue.surfaces.payloads.model_json_document", return_value={"fixture": "w12"}),
    ):
        result = CliRunner().invoke(cli, ["--format", "json", "find", "repo:w12", "then", "analyze", projection])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["status"] == "ok"
    assert "w12" in result.output


def test_named_cost_outlook_renders_inherited_json() -> None:
    """62.01: a no-window outlook still owes the root caller a JSON availability document."""
    with patch.object(Polylogue, "cost_outlook", new_callable=AsyncMock, return_value=None):
        result = CliRunner().invoke(
            cli,
            ["--format", "json", "find", "repo:w12", "then", "analyze", "cost-outlook", "--plan", "claude-pro"],
        )
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["outlook"] is None
    assert "availability" in document


def test_named_analysis_local_format_overrides_inherited_json() -> None:
    """The format inheritance fix must not override an explicit child renderer."""
    with (
        patch.object(Polylogue, "postmortem_bundle", new_callable=AsyncMock, return_value=object()),
        patch("polylogue.analysis.postmortem.render_postmortem_markdown", return_value="# W12 explicit markdown"),
    ):
        result = CliRunner().invoke(
            cli,
            ["--format", "json", "find", "repo:w12", "then", "analyze", "postmortem", "--format", "markdown"],
        )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "# W12 explicit markdown"


def test_completion_preserves_observed_empty_answer_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    """62.16: collapsing an observed empty answer into a cache miss incorrectly emits daemon guidance."""
    from polylogue.cli import shell_completion_values as completion

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    answer: object = {"value_completions": {"values": [{"value": "release-tag"}]}}

    def online(*_args, **_kwargs):
        return SimpleNamespace(value=answer)

    def offline(*_args, **_kwargs):
        raise OperationUnavailableError("daemon unavailable")

    monkeypatch.setattr(operation_kernel, "dispatch", online)
    assert [item.value for item in completion.completion_values("tag", "rel", limit=5)] == ["release-tag"]
    answer = {"value_completions": {"values": []}}
    assert completion.completion_values("tag", "rel", limit=5) == []
    monkeypatch.setattr(operation_kernel, "dispatch", offline)
    assert completion.completion_values("tag", "rel", limit=5) == []
    assert completion.completion_values("tag", "unseen", limit=5)[0].value == completion.DAEMON_REQUIRED_COMPLETION_MESSAGE
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-b"))
    assert completion.completion_values("tag", "rel", limit=5)[0].value == completion.DAEMON_REQUIRED_COMPLETION_MESSAGE
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    frozen_clock.advance(completion._COMPLETION_CACHE_TTL_SECONDS + 1)
    assert completion.completion_values("tag", "rel", limit=5)[0].value == completion.DAEMON_REQUIRED_COMPLETION_MESSAGE


def test_completion_nonempty_answer_invalidates_disproved_empty_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty all-values observation must not hide a subsequently observed candidate."""
    from polylogue.cli import shell_completion_values as completion

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    answer: object = {"value_completions": {"values": []}}
    monkeypatch.setattr(operation_kernel, "dispatch", lambda *_args, **_kwargs: SimpleNamespace(value=answer))
    assert completion.completion_values("tag", "", limit=5) == []
    answer = {"value_completions": {"values": [{"value": "release-tag"}]}}
    assert [item.value for item in completion.completion_values("tag", "rel", limit=5)] == ["release-tag"]

    def offline(*_args, **_kwargs):
        raise OperationUnavailableError("daemon unavailable")

    monkeypatch.setattr(operation_kernel, "dispatch", offline)
    assert [item.value for item in completion.completion_values("tag", "", limit=5)] == ["release-tag"]


def test_completion_malformed_result_is_not_an_empty_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing values payload is not authoritative evidence of no matching values."""
    from polylogue.cli import shell_completion_values as completion

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setattr(operation_kernel, "dispatch", lambda *_args, **_kwargs: SimpleNamespace(value={}))
    assert completion.completion_values("tag", "rel", limit=5) == []

    def offline(*_args, **_kwargs):
        raise OperationUnavailableError("daemon unavailable")

    monkeypatch.setattr(operation_kernel, "dispatch", offline)
    assert completion.completion_values("tag", "rel", limit=5)[0].value == completion.DAEMON_REQUIRED_COMPLETION_MESSAGE
