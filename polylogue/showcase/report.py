"""Public showcase report facade."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.showcase.runner import ShowcaseResult

from .qa_report import (
    build_qa_session_payload,
)
from .qa_report import (
    generate_qa_markdown as _generate_qa_markdown,
)
from .qa_report import (
    generate_qa_summary as _generate_qa_summary,
)
from .report_files import (
    generate_manifest as _generate_manifest,
)
from .report_files import (
    save_reports as _save_reports,
)
from .showcase_report import (
    build_showcase_session_payload,
)
from .showcase_report import (
    generate_cookbook as _generate_cookbook,
)
from .showcase_report import (
    generate_json_report as _generate_json_report,
)
from .showcase_report import (
    generate_showcase_markdown as _generate_showcase_markdown,
)
from .showcase_report import (
    generate_summary as _generate_summary,
)

if TYPE_CHECKING:
    from polylogue.showcase.qa_runner import QAResult


def generate_summary(result: ShowcaseResult) -> str:
    return _generate_summary(result)


def generate_json_report(result: ShowcaseResult) -> str:
    return _generate_json_report(result)


def generate_cookbook(result: ShowcaseResult) -> str:
    return _generate_cookbook(result)


def generate_showcase_session(result: ShowcaseResult) -> dict[str, Any]:
    """Generate a structured showcase session record."""
    return build_showcase_session_payload(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def write_showcase_session(result: ShowcaseResult, audit_dir: Path) -> Path:
    """Write a showcase session record to audit_dir and return the written path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = audit_dir / f"showcase-{ts}.json"
    session = generate_showcase_session(result)
    out_path.write_text(json.dumps(session, indent=2))
    return out_path


def generate_showcase_markdown(result: ShowcaseResult, *, git_sha: str | None = None) -> str:
    return _generate_showcase_markdown(result, git_sha=git_sha)


def generate_qa_session(result: QAResult) -> dict[str, Any]:
    """Generate a structured full QA session record."""
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


def generate_qa_summary(result: QAResult) -> str:
    """Generate a human-readable summary for a full QA run."""
    session = generate_qa_session(result)
    return _generate_qa_summary(result, session=session)


def generate_qa_markdown(result: QAResult, *, git_sha: str | None = None) -> str:
    """Generate a stable, diffable Markdown report for a full QA run."""
    session = generate_qa_session(result)
    return _generate_qa_markdown(result, session=session, git_sha=git_sha)


def generate_manifest(
    result: ShowcaseResult,
    *,
    include_hashes: bool = True,
) -> dict[str, Any]:
    return _generate_manifest(result, include_hashes=include_hashes)


def save_reports(result: ShowcaseResult) -> None:
    _save_reports(result)
