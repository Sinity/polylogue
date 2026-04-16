from __future__ import annotations

from polylogue.showcase.qa_runner_request import QASessionRequest


def test_qa_session_request_needs_workspace_when_exercises_run() -> None:
    assert QASessionRequest().needs_workspace is True


def test_qa_session_request_drops_workspace_when_exercises_are_skipped() -> None:
    assert QASessionRequest(skip_exercises=True).needs_workspace is False
