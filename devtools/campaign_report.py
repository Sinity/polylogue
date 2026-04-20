"""Generate Markdown and JSON reports from campaign results."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from devtools.synthetic_benchmark_runtime import CampaignResult


def generate_campaign_markdown(results: list[CampaignResult]) -> str:
    """Produce a Markdown report from a list of campaign results.

    The report includes scale level, per-campaign timing, DB size,
    row counts, and a comparison table for easy diffing.
    """
    if not results:
        return "# Benchmark Campaign Report\n\nNo results.\n"

    scale_level = results[0].scale_level or "unknown"
    timestamp = results[0].timestamp

    lines = [
        "# Benchmark Campaign Report",
        "",
        f"- **Scale level**: {scale_level}",
        f"- **Generated**: {timestamp}",
        "",
        "## Summary",
        "",
        "| Campaign | Key Metric | Value |",
        "| --- | --- | ---: |",
    ]

    for result in results:
        key_metric, value = _pick_key_metric(result)
        lines.append(f"| {result.campaign_name} | {key_metric} | {value} |")

    lines.extend(["", "## Detailed Results", ""])

    for result in results:
        lines.extend(_render_campaign_section(result))

    # DB stats comparison table
    lines.extend(["", "## Database Statistics", ""])
    lines.extend(_render_db_stats_table(results))

    lines.append("")
    return "\n".join(lines)


def _pick_key_metric(result: CampaignResult) -> tuple[str, str]:
    """Pick the single most important metric from a campaign result."""
    m = result.metrics
    match result.campaign_name:
        case "fts-rebuild":
            return "rebuild_wall_s", f"{m.get('rebuild_wall_s', 0):.3f}s"
        case "incremental-index":
            return "total_wall_s", f"{m.get('total_wall_s', 0):.3f}s"
        case "filter-scan":
            return "list_50_wall_s", f"{m.get('list_50_wall_s', 0):.4f}s"
        case "startup-readiness":
            return "total_readiness_s", f"{m.get('total_readiness_s', 0):.4f}s"
        case _:
            if m:
                key = next(iter(m))
                return key, f"{m[key]}"
            return "n/a", "n/a"


def _render_campaign_section(result: CampaignResult) -> list[str]:
    """Render a single campaign's detailed section."""
    lines = [
        f"### {result.campaign_name}",
        "",
    ]
    if result.path_targets or result.artifact_targets or result.operation_targets or result.tags:
        lines.extend(
            [
                "| Scenario metadata | Value |",
                "| --- | --- |",
                f"| origin | `{result.origin}` |",
            ]
        )
        if result.path_targets:
            lines.append(f"| path targets | `{', '.join(result.path_targets)}` |")
        if result.artifact_targets:
            lines.append(f"| artifact targets | `{', '.join(result.artifact_targets)}` |")
        if result.operation_targets:
            lines.append(f"| operation targets | `{', '.join(result.operation_targets)}` |")
        if result.tags:
            lines.append(f"| tags | `{', '.join(result.tags)}` |")
        lines.append("")

    if result.metrics:
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in sorted(result.metrics.items()):
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

    return lines


def _render_db_stats_table(results: list[CampaignResult]) -> list[str]:
    """Render a DB statistics comparison table."""
    # Collect all unique stat keys
    all_keys: list[str] = []
    seen: set[str] = set()
    for result in results:
        for key in result.db_stats:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    if not all_keys:
        return ["_No database statistics collected._"]

    # Header
    campaign_names = [r.campaign_name for r in results]
    header = "| Stat | " + " | ".join(campaign_names) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in campaign_names) + " |"
    lines = [header, separator]

    for key in all_keys:
        values = []
        for result in results:
            val = result.db_stats.get(key, "")
            if isinstance(val, int) and key.endswith("_bytes"):
                values.append(f"{val / 1024 / 1024:.1f} MB")
            else:
                values.append(str(val))
        lines.append(f"| {key} | " + " | ".join(values) + " |")

    return lines


def generate_campaign_json(results: list[CampaignResult]) -> str:
    """Produce a JSON report from a list of campaign results."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scale_level": results[0].scale_level if results else "unknown",
        "campaigns": [asdict(r) for r in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def save_campaign_reports(results: list[CampaignResult], output_dir: Path) -> list[Path]:
    """Save both Markdown and JSON reports to the output directory.

    Returns:
        List of paths to saved report files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = results[0].scale_level if results else "unknown"
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    stem = f"{date}-{scale}"

    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"

    md_path.write_text(generate_campaign_markdown(results), encoding="utf-8")
    json_path.write_text(generate_campaign_json(results), encoding="utf-8")

    return [md_path, json_path]


__all__ = [
    "generate_campaign_json",
    "generate_campaign_markdown",
    "save_campaign_reports",
]
