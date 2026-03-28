"""QA-specific report rendering built on showcase session payloads."""

from datetime import datetime, timezone

from polylogue.showcase.qa_markdown import generate_qa_markdown
from polylogue.showcase.qa_session_payload import build_qa_session_payload
from polylogue.showcase.qa_summary import generate_qa_summary


def generate_qa_session(result):
    """Generate a structured full QA session record."""
    from polylogue.showcase.showcase_report_payloads import generate_showcase_session

    showcase_session = (
        generate_showcase_session(result.showcase_result)
        if result.showcase_result is not None
        else None
    )
    return build_qa_session_payload(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
        showcase_session=showcase_session,
    )

__all__ = [
    "build_qa_session_payload",
    "generate_qa_markdown",
    "generate_qa_session",
    "generate_qa_summary",
]
