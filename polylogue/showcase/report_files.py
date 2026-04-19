"""Filesystem-facing showcase report helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.publication import OutputManifest
from polylogue.showcase.runner import ShowcaseResult

from .showcase_report_payloads import generate_json_report
from .showcase_report_text import generate_cookbook, generate_showcase_markdown, generate_summary


@dataclass(frozen=True, slots=True)
class ShowcaseReportArtifact:
    """One generated showcase report artifact."""

    relative_path: str
    content: str

    def write_to(self, output_dir: Path) -> None:
        (output_dir / self.relative_path).write_text(self.content)


def scan_manifest(result: ShowcaseResult, *, include_hashes: bool = True) -> OutputManifest:
    """Produce the typed manifest for all generated showcase artifacts."""
    if not result.output_dir:
        return OutputManifest()
    return OutputManifest.scan(
        result.output_dir,
        include_hashes=include_hashes,
        exclude_paths={"showcase-manifest.json"},
    )


def generate_manifest(result: ShowcaseResult, *, include_hashes: bool = True) -> dict[str, Any]:
    """Produce a JSON-serializable manifest payload for showcase artifacts."""
    return scan_manifest(result, include_hashes=include_hashes).model_dump(mode="json", exclude_none=True)


def _report_artifacts(result: ShowcaseResult) -> tuple[ShowcaseReportArtifact, ...]:
    return (
        ShowcaseReportArtifact("showcase-summary.txt", generate_summary(result)),
        ShowcaseReportArtifact("showcase-report.json", generate_json_report(result)),
        ShowcaseReportArtifact("showcase-cookbook.md", generate_cookbook(result)),
        ShowcaseReportArtifact("showcase-session.md", generate_showcase_markdown(result)),
    )


def save_reports(result: ShowcaseResult) -> None:
    """Save all report artifacts to the output directory."""
    if not result.output_dir:
        return

    output_dir = result.output_dir
    for artifact in _report_artifacts(result):
        artifact.write_to(output_dir)
    manifest = scan_manifest(result, include_hashes=True)
    (output_dir / "showcase-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True)
    )


__all__ = ["ShowcaseReportArtifact", "generate_manifest", "save_reports", "scan_manifest"]
