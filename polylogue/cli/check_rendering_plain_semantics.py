"""Roundtrip section for plain check output."""

from __future__ import annotations

from polylogue.cli.check_workflow import CheckCommandResult


def append_roundtrip_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.roundtrip_report is None:
        return
    roundtrip_summary = result.roundtrip_report.summary
    lines.extend(
        [
            "",
            f"Roundtrip proof: {roundtrip_summary['provider_count']:,} providers "
            f"(clean={roundtrip_summary['clean_providers']:,}, "
            f"failed={roundtrip_summary['failed_providers']:,}, "
            f"artifacts={roundtrip_summary['artifact_count']:,}, "
            f"parsed_conversations={roundtrip_summary['parsed_conversations']:,}, "
            f"persisted_conversations={roundtrip_summary['persisted_conversations']:,})",
        ]
    )
    for provider, provider_report in sorted(result.roundtrip_report.provider_reports.items()):
        summary = provider_report.summary
        status = "clean" if provider_report.is_clean else "failed"
        lines.append(
            f"  {provider}: {status}, package={provider_report.package_version}, "
            f"element={provider_report.element_kind or '-'}, "
            f"failed_stages={','.join(summary['failed_stages']) or '-'}"
        )


__all__ = ["append_roundtrip_lines"]
