"""Public QA orchestration surface."""

from __future__ import annotations

from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_reporting import format_qa_summary, save_qa_reports
from polylogue.showcase.qa_runner_request import QASessionRequest
from polylogue.showcase.qa_runner_workflow import run_qa_session

_save_qa_reports = save_qa_reports

__all__ = [
    "QAResult",
    "QASessionRequest",
    "_save_qa_reports",
    "format_qa_summary",
    "run_qa_session",
    "save_qa_reports",
]
