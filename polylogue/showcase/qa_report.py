"""QA-specific report rendering built on showcase session payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from polylogue.showcase.qa_markdown import generate_qa_markdown
from polylogue.showcase.qa_session_payload import build_qa_session_payload
from polylogue.showcase.qa_summary import generate_qa_summary
from polylogue.showcase.showcase_report_payloads import build_showcase_session_record

if TYPE_CHECKING:
    from polylogue.showcase.qa_runner import QAResult


def generate_qa_session(result: QAResult) -> dict[str, object]:
    """Generate a structured full QA session record.

    The wrapper lives here instead of only in ``qa_session_payload`` so tests
    and callers can patch ``polylogue.showcase.qa_report.datetime`` as the
    stable clock seam for deterministic report output.
    """

    timestamp = datetime.now(timezone.utc).isoformat()
    showcase_session = (
        build_showcase_session_record(result.showcase_result, timestamp=timestamp)
        if result.showcase_result is not None
        else None
    )
    return build_qa_session_payload(
        result,
        timestamp=timestamp,
        showcase_session=showcase_session,
    )


__all__ = [
    "build_qa_session_payload",
    "generate_qa_markdown",
    "generate_qa_session",
    "generate_qa_summary",
]
