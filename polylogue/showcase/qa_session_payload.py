"""Structured QA session payload builders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from polylogue.insights.authored_payloads import PayloadDict
from polylogue.showcase.report_models import (
    QASessionRecord,
    ShowcaseSessionRecord,
    canonical_qa_session,
)

if TYPE_CHECKING:
    from polylogue.showcase.qa_runner import QAResult


def generate_qa_session(result: QAResult) -> PayloadDict:
    """Generate a structured full QA session record."""
    from polylogue.showcase.showcase_report_payloads import build_showcase_session_record

    showcase_session = (
        build_showcase_session_record(
            result.showcase_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if result.showcase_result is not None
        else None
    )
    return build_qa_session_payload(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
        showcase_session=showcase_session,
    )


def build_qa_session_payload(
    result: QAResult,
    *,
    timestamp: str,
    showcase_session: ShowcaseSessionRecord | None,
) -> PayloadDict:
    return build_qa_session_record(
        result,
        timestamp=timestamp,
        showcase_session=showcase_session,
    ).to_payload()


def build_qa_session_record(
    result: QAResult,
    *,
    timestamp: str,
    showcase_session: ShowcaseSessionRecord | None,
) -> QASessionRecord:
    """Return the typed QA session record before JSON serialization."""
    return canonical_qa_session(
        result,
        timestamp=timestamp,
        showcase_session=showcase_session,
    )


__all__ = [
    "build_qa_session_payload",
    "build_qa_session_record",
    "generate_qa_session",
]
