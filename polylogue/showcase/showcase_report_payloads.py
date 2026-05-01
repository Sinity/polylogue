"""Structured showcase session payload helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polylogue.insights.authored_payloads import PayloadDict
from polylogue.showcase.report_models import (
    ShowcaseExerciseRecord,
    ShowcaseSessionRecord,
    canonical_showcase_session,
)
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult


def serialize_showcase_exercise(
    result: ExerciseResult,
    *,
    include_description: bool = True,
    include_tier: bool = False,
) -> PayloadDict:
    return ShowcaseExerciseRecord.from_result(
        result,
        include_description=include_description,
        include_tier=include_tier,
    ).to_payload()


def showcase_summary_payload(result: ShowcaseResult) -> dict[str, int | float]:
    return result.summary().to_payload()


def build_showcase_session_payload(
    result: ShowcaseResult,
    *,
    timestamp: str,
) -> PayloadDict:
    return canonical_showcase_session(result, timestamp=timestamp).to_payload()


def generate_showcase_session(result: ShowcaseResult) -> PayloadDict:
    return build_showcase_session_payload(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def write_showcase_session(result: ShowcaseResult, audit_dir: Path) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = audit_dir / f"showcase-{ts}.json"
    out_path.write_text(json.dumps(generate_showcase_session(result), indent=2))
    return out_path


def build_showcase_session_record(result: ShowcaseResult, *, timestamp: str) -> ShowcaseSessionRecord:
    """Return the typed showcase session record before JSON serialization."""
    return canonical_showcase_session(result, timestamp=timestamp)


def generate_json_report(result: ShowcaseResult) -> str:
    report = {
        **showcase_summary_payload(result),
        "exercises": [serialize_showcase_exercise(entry) for entry in result.results],
    }
    return json.dumps(report, indent=2)


__all__ = [
    "build_showcase_session_payload",
    "build_showcase_session_record",
    "generate_json_report",
    "generate_showcase_session",
    "serialize_showcase_exercise",
    "showcase_summary_payload",
    "write_showcase_session",
]
