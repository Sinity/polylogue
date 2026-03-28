"""Showcase-specific report rendering and session payloads."""

from __future__ import annotations

from polylogue.showcase.showcase_report_payloads import (
    build_showcase_session_payload,
    generate_json_report,
    generate_showcase_session,
    serialize_showcase_exercise,
    showcase_summary_payload,
    write_showcase_session,
)
from polylogue.showcase.showcase_report_text import (
    generate_cookbook,
    generate_showcase_markdown,
    generate_summary,
)

__all__ = [
    "build_showcase_session_payload",
    "generate_cookbook",
    "generate_json_report",
    "generate_showcase_markdown",
    "generate_showcase_session",
    "generate_summary",
    "serialize_showcase_exercise",
    "showcase_summary_payload",
    "write_showcase_session",
]
