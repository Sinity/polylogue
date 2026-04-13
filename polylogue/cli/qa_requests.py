"""Typed request helpers for the QA command surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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


def build_qa_snapshot_plan(*, snapshot_label: str | None, snapshot_from: Path | None) -> QASnapshotPlan | None:
    """Normalize the optional snapshot intent from QA command flags."""
    if snapshot_label is None and snapshot_from is None:
        return None
    return QASnapshotPlan(
        label=snapshot_label or "snapshot",
        source_dir=snapshot_from,
    )


__all__ = [
    "QACaptureMode",
    "QASnapshotPlan",
    "build_qa_snapshot_plan",
]
