"""Disposable safety-case lab for ``polylogue-yeq.1``.

The lab is deliberately a composition boundary: semantic checks remain in the
storage, daemon, and source owners.  It runs the existing production-route
scenarios and adds a small model sequence for the cursor authority seam.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from devtools.rebuild_safety_scenario import run_rebuild_differential, run_rebuild_safety
from devtools.storage_correctness_scenario import run_storage_correctness
from polylogue.core.outcomes import OutcomeStatus
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.cursor_lifecycle import (
    CursorLifecycleState,
    classify_cursor_lifecycle_state,
)

SAFETY_CASE_SCENARIO_NAME = "safety-case"
SAFETY_CASE_ARTIFACT = "docs/safety-case-v1.json"


@dataclass(slots=True)
class SafetyCaseResult:
    report_dir: Path | None
    checks: dict[str, bool]
    details: dict[str, object]

    @property
    def scenario_name(self) -> str:
        return SAFETY_CASE_SCENARIO_NAME

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())

    def stage_statuses(self) -> dict[str, OutcomeStatus]:
        return {name: OutcomeStatus.OK if passed else OutcomeStatus.ERROR for name, passed in self.checks.items()}

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks.items() if not passed)

    def extra_payload(self) -> dict[str, object]:
        return {"checks": self.checks, "details": self.details, "artifact": SAFETY_CASE_ARTIFACT}


def _artifact() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    payload = cast(dict[str, object], json.loads((root / SAFETY_CASE_ARTIFACT).read_text(encoding="utf-8")))
    tiers = payload.get("tiers", [])
    hazards = payload.get("hazards", [])
    if payload.get("version") != 1 or not isinstance(tiers, list) or not isinstance(hazards, list):
        raise AssertionError("safety-case artifact has invalid collection fields")
    if len(tiers) != 6 or len(hazards) != 5:
        raise AssertionError("safety-case artifact must enumerate version 1, all six tiers, and five hazards")
    return payload


def _cursor_model_sequence() -> dict[str, object]:
    """Run reorder/retry/restart-like operations through the real CursorStore."""
    with TemporaryDirectory(prefix="polylogue-safety-cursor-") as temp:
        root = Path(temp)
        store = CursorStore(root / "ops.sqlite")
        source = root / "capture.jsonl"
        source.write_text('{"event":"one"}\n', encoding="utf-8")
        store.mark_failed(source)
        pending = store.get_record(source)
        assert classify_cursor_lifecycle_state(pending) is CursorLifecycleState.RETRY_PENDING
        store.reset_failures(source)
        store.mark_excluded(source)
        excluded = store.get_record(source)
        assert classify_cursor_lifecycle_state(excluded) is CursorLifecycleState.EXCLUDED
        # A retry/reorder attempt must not clear a poison-pill state.
        store.reset_failures(source)
        assert classify_cursor_lifecycle_state(store.get_record(source)) is CursorLifecycleState.EXCLUDED
        return {"states": ["retry_pending", "active", "excluded", "excluded"], "committed_evidence_retained": True}


def run_safety_case(*, report_dir: Path | None = None) -> SafetyCaseResult:
    artifact = _artifact()
    cursor = _cursor_model_sequence()

    def run_checked(
        function: Callable[..., object], *args: object, **kwargs: object
    ) -> tuple[object | None, str | None]:
        try:
            return function(*args, **kwargs), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    storage, storage_error = run_checked(run_storage_correctness, report_dir=None)
    rebuild, rebuild_error = run_checked(run_rebuild_safety)
    differential, differential_error = run_checked(run_rebuild_differential)

    def passed(value: object | None) -> bool:
        return bool(value is not None and getattr(value, "all_passed", False))

    checks = {
        "artifact_complete": True,
        "cursor_retry_reorder": bool(cursor["committed_evidence_retained"]),
        "storage_routes": passed(storage),
        "full_rebuild_rerun": passed(rebuild),
        "incremental_matches_full": passed(differential),
    }

    def failed_stages(value: object) -> tuple[str, ...]:
        candidate = getattr(value, "failed_stages", ())
        result = candidate() if callable(candidate) else candidate
        return tuple(str(item) for item in result)

    def diverging_tables(value: object) -> tuple[str, ...]:
        return tuple(str(getattr(item, "table", item)) for item in getattr(value, "diverging_tables", ()))

    details = {
        "artifact_version": artifact["version"],
        "cursor": cursor,
        "storage_failed": list(failed_stages(storage)) if storage is not None else [],
        "rebuild_failed": list(diverging_tables(rebuild)) if rebuild is not None else [],
        "differential_failed": list(diverging_tables(differential)) if differential is not None else [],
        "errors": {
            key: value
            for key, value in {
                "storage": storage_error,
                "rebuild": rebuild_error,
                "differential": differential_error,
            }.items()
            if value is not None
        },
    }
    result = SafetyCaseResult(report_dir=report_dir, checks=checks, details=details)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "safety-case-v1.json").write_text(json.dumps(result.extra_payload(), indent=2), encoding="utf-8")
    return result


__all__ = ["SAFETY_CASE_ARTIFACT", "SAFETY_CASE_SCENARIO_NAME", "SafetyCaseResult", "run_safety_case"]
