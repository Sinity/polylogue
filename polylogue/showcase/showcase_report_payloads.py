"""Structured showcase session payload helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.showcase.runner import ExerciseResult, ShowcaseResult


def _optional_string(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _optional_string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def serialize_showcase_exercise(
    result: ExerciseResult,
    *,
    include_description: bool = True,
    include_tier: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": result.exercise.name,
        "group": result.exercise.group,
        "passed": result.passed,
        "exit_code": result.exit_code,
        "duration_ms": round(result.duration_ms, 1),
    }
    if include_description:
        entry["description"] = result.exercise.description
    if include_tier:
        entry["tier"] = result.exercise.tier
    entry["origin"] = _optional_string(getattr(result.exercise, "origin", "authored"), "authored")
    artifact_targets = _optional_string_list(getattr(result.exercise, "artifact_targets", ()))
    if artifact_targets:
        entry["artifact_targets"] = artifact_targets
    operation_targets = _optional_string_list(getattr(result.exercise, "operation_targets", ()))
    if operation_targets:
        entry["operation_targets"] = operation_targets
    tags = _optional_string_list(getattr(result.exercise, "tags", ()))
    if tags:
        entry["tags"] = tags
    if result.skipped:
        entry["skipped"] = True
        entry["skip_reason"] = result.skip_reason
    if result.error:
        entry["error"] = result.error
    return entry


def showcase_summary_payload(result: ShowcaseResult) -> dict[str, int | float]:
    return {
        "total": len(result.results),
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "total_duration_ms": round(result.total_duration_ms, 1),
    }


def build_showcase_session_payload(
    result: ShowcaseResult,
    *,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": timestamp,
        "summary": showcase_summary_payload(result),
        "group_counts": result.group_counts(),
        "exercises": [
            serialize_showcase_exercise(
                report,
                include_description=False,
                include_tier=True,
            )
            for report in result.results
        ],
    }


def generate_showcase_session(result: ShowcaseResult) -> dict[str, Any]:
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


def generate_json_report(result: ShowcaseResult) -> str:
    report = {
        **showcase_summary_payload(result),
        "exercises": [serialize_showcase_exercise(entry) for entry in result.results],
    }
    return json.dumps(report, indent=2)


__all__ = [
    "build_showcase_session_payload",
    "generate_json_report",
    "generate_showcase_session",
    "serialize_showcase_exercise",
    "showcase_summary_payload",
    "write_showcase_session",
]
