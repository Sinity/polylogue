"""Typed payloads for the deterministic Polylogue demo archive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constructs import DemoConstructCoverage


@dataclass(frozen=True, slots=True)
class DemoSeedResult:
    """Result of materializing and ingesting the deterministic demo archive."""

    archive_root: Path
    source_root: Path
    session_count: int
    message_count: int
    session_ids: tuple[str, ...]
    overlays_seeded: bool
    assertion_count: int
    construct_coverage: tuple[DemoConstructCoverage, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Serialize the seed result for CLI JSON output and tests."""

        return {
            "archive_root": str(self.archive_root),
            "source_root": str(self.source_root),
            "session_count": self.session_count,
            "message_count": self.message_count,
            "session_ids": list(self.session_ids),
            "overlays_seeded": self.overlays_seeded,
            "assertion_count": self.assertion_count,
            "construct_coverage": [row.to_payload() for row in self.construct_coverage],
        }


@dataclass(frozen=True, slots=True)
class DemoVerifyResult:
    """Semantic verification result for the deterministic demo archive."""

    archive_root: Path
    ok: bool
    session_count: int
    message_count: int
    query_hits: tuple[str, ...]
    overlays_present: bool
    absolute_path_leaks: tuple[str, ...]
    construct_coverage: tuple[DemoConstructCoverage, ...] = ()
    problems: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Serialize the verification result for CLI JSON output and tests."""

        return {
            "archive_root": str(self.archive_root),
            "ok": self.ok,
            "session_count": self.session_count,
            "message_count": self.message_count,
            "query_hits": list(self.query_hits),
            "overlays_present": self.overlays_present,
            "absolute_path_leaks": list(self.absolute_path_leaks),
            "construct_coverage": [row.to_payload() for row in self.construct_coverage],
            "problems": list(self.problems),
        }


@dataclass(frozen=True, slots=True)
class DemoTourStep:
    """One command executed by the one-command demo tour."""

    name: str
    command: tuple[str, ...]
    exit_code: int
    duration_s: float
    output_path: Path
    bytes_written: int

    def to_payload(self) -> dict[str, object]:
        """Serialize a tour step for report JSON."""

        return {
            "name": self.name,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "output_path": str(self.output_path),
            "bytes_written": self.bytes_written,
        }


@dataclass(frozen=True, slots=True)
class DemoTourResult:
    """Result of the one-command public demo tour."""

    archive_root: Path
    output_dir: Path
    ok: bool
    first_result_s: float
    total_duration_s: float
    report_json_path: Path
    report_markdown_path: Path
    transcript_path: Path
    recording_tape_path: Path
    seed: DemoSeedResult
    verify: DemoVerifyResult
    steps: tuple[DemoTourStep, ...]
    problems: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Serialize the tour result for CLI JSON output and artifacts."""

        return {
            "archive_root": str(self.archive_root),
            "output_dir": str(self.output_dir),
            "ok": self.ok,
            "first_result_s": round(self.first_result_s, 3),
            "total_duration_s": round(self.total_duration_s, 3),
            "report_json_path": str(self.report_json_path),
            "report_markdown_path": str(self.report_markdown_path),
            "transcript_path": str(self.transcript_path),
            "recording_tape_path": str(self.recording_tape_path),
            "seed": self.seed.to_payload(),
            "verify": self.verify.to_payload(),
            "steps": [step.to_payload() for step in self.steps],
            "problems": list(self.problems),
        }


__all__ = ["DemoSeedResult", "DemoTourResult", "DemoTourStep", "DemoVerifyResult"]
