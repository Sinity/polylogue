"""Typed request helpers for the QA command surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from polylogue.showcase import qa_runner_request


class QACaptureMode(str, Enum):
    NONE = "none"
    VHS = "vhs"


@dataclass(frozen=True, slots=True)
class QASnapshotPlan:
    """Normalized snapshot intent for a QA invocation."""

    label: str
    source_dir: Path | None = None

    @property
    def skips_qa(self) -> bool:
        return self.source_dir is not None

    def resolve_source_dir(self, fallback: Path | None) -> Path | None:
        return self.source_dir or fallback


@dataclass(frozen=True, slots=True)
class QAFinalizationPlan:
    """Normalized post-run intent for one QA invocation."""

    capture_mode: QACaptureMode = QACaptureMode.NONE
    json_output: bool = False
    snapshot_plan: QASnapshotPlan | None = None


@dataclass(frozen=True, slots=True)
class QAInvocationPlan:
    """Canonical CLI-facing QA invocation plan."""

    session_request: qa_runner_request.QASessionRequest | None
    finalization_plan: QAFinalizationPlan
    snapshot_plan: QASnapshotPlan | None = None

    @property
    def snapshot_only(self) -> bool:
        return self.snapshot_plan is not None and self.snapshot_plan.skips_qa


def build_qa_snapshot_plan(*, snapshot_label: str | None, snapshot_from: Path | None) -> QASnapshotPlan | None:
    """Normalize the optional snapshot intent from QA command flags."""
    if snapshot_label is None and snapshot_from is None:
        return None
    return QASnapshotPlan(
        label=snapshot_label or "snapshot",
        source_dir=snapshot_from,
    )


def build_qa_finalization_plan(
    *,
    capture_mode: QACaptureMode,
    json_output: bool,
    snapshot_plan: QASnapshotPlan | None,
) -> QAFinalizationPlan:
    """Normalize post-run capture/output intent for QA command surfaces."""
    return QAFinalizationPlan(
        capture_mode=capture_mode,
        json_output=json_output,
        snapshot_plan=snapshot_plan,
    )


def build_qa_invocation_plan(
    *,
    synthetic: bool,
    source_names: tuple[str, ...] | None,
    fresh: bool | None,
    ingest: bool | None,
    regenerate_schemas: bool,
    only_stage: str | None,
    skip_stages: tuple[str, ...],
    workspace: Path | None,
    report_dir: Path | None,
    verbose: bool,
    fail_fast: bool,
    tier_filter: int | None,
    capture: str,
    json_output: bool,
    snapshot_label: str | None,
    snapshot_from: Path | None,
) -> QAInvocationPlan:
    """Normalize all CLI-facing QA invocation options into one typed plan."""
    capture_mode = QACaptureMode(capture.lower())
    snapshot_plan = build_qa_snapshot_plan(
        snapshot_label=snapshot_label,
        snapshot_from=snapshot_from,
    )
    finalization_plan = build_qa_finalization_plan(
        capture_mode=capture_mode,
        json_output=json_output,
        snapshot_plan=snapshot_plan,
    )
    if snapshot_plan and snapshot_plan.skips_qa:
        return QAInvocationPlan(
            session_request=None,
            finalization_plan=finalization_plan,
            snapshot_plan=snapshot_plan,
        )
    return QAInvocationPlan(
        session_request=qa_runner_request.build_qa_session_request(
            synthetic=synthetic,
            source_names=source_names,
            fresh=fresh,
            ingest=ingest,
            regenerate_schemas=regenerate_schemas,
            only_stage=qa_runner_request.QAStage(only_stage) if only_stage else None,
            skip_stages=tuple(qa_runner_request.QAStage(stage) for stage in skip_stages),
            workspace=workspace,
            report_dir=report_dir,
            verbose=verbose,
            fail_fast=fail_fast,
            tier_filter=tier_filter,
        ),
        finalization_plan=finalization_plan,
        snapshot_plan=snapshot_plan,
    )


__all__ = [
    "QAInvocationPlan",
    "QAFinalizationPlan",
    "QACaptureMode",
    "QASnapshotPlan",
    "build_qa_finalization_plan",
    "build_qa_invocation_plan",
    "build_qa_snapshot_plan",
]
