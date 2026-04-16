"""Filesystem-facing showcase report helpers."""

from __future__ import annotations

import json
from typing import Any

from polylogue.publication import OutputManifest
from polylogue.showcase.runner import ShowcaseResult

from .showcase_report_payloads import generate_json_report
from .showcase_report_text import (
    generate_cookbook,
    generate_showcase_markdown,
    generate_summary,
)


def generate_manifest(
    result: ShowcaseResult,
    *,
    include_hashes: bool = True,
) -> dict[str, Any]:
    """Produce a manifest with file hashes for all generated artifacts."""
    if not result.output_dir:
        return OutputManifest().model_dump(mode="json")
    return OutputManifest.scan(
        result.output_dir,
        include_hashes=include_hashes,
        exclude_paths={"showcase-manifest.json"},
    ).model_dump(mode="json", exclude_none=True)


def save_reports(result: ShowcaseResult) -> None:
    """Save all report artifacts to the output directory."""
    if not result.output_dir:
        return

    out = result.output_dir
    (out / "showcase-summary.txt").write_text(generate_summary(result))
    (out / "showcase-report.json").write_text(generate_json_report(result))
    (out / "showcase-cookbook.md").write_text(generate_cookbook(result))
    (out / "showcase-session.md").write_text(generate_showcase_markdown(result))
    manifest = generate_manifest(result, include_hashes=True)
    (out / "showcase-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
