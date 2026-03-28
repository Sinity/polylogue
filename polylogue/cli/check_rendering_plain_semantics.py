"""Semantic/roundtrip sections for plain check output."""

from __future__ import annotations

from polylogue.cli.check_support import format_semantic_metric_summary
from polylogue.cli.check_workflow import CheckCommandResult


def append_semantic_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.semantic_report is not None:
        semantic_summary = result.semantic_report.to_dict()["summary"]
        lines.extend(
            [
                "",
                f"Semantic proof: {semantic_summary['surface_count']:,} surfaces "
                f"(clean={semantic_summary['clean_surfaces']:,}, "
                f"critical={semantic_summary['critical_surfaces']:,}, "
                f"total_conversations={semantic_summary['total_conversations']:,}, "
                f"preserved_checks={semantic_summary['preserved_checks']:,}, "
                f"declared_loss_checks={semantic_summary['declared_loss_checks']:,}, "
                f"critical_loss_checks={semantic_summary['critical_loss_checks']:,})",
            ]
        )
        if semantic_summary["metric_summary"]:
            lines.append(
                f"  Metrics: {format_semantic_metric_summary(semantic_summary['metric_summary'])}"
            )
        for surface, surface_report in sorted(result.semantic_report.surfaces.items()):
            surface_summary = surface_report.to_dict()["summary"]
            lines.append(
                f"  {surface}: conversations={surface_summary['total_conversations']:,} "
                f"clean={surface_summary['clean_conversations']:,} "
                f"critical={surface_summary['critical_conversations']:,} "
                f"preserved_checks={surface_summary['preserved_checks']:,} "
                f"declared_loss_checks={surface_summary['declared_loss_checks']:,} "
                f"critical_loss_checks={surface_summary['critical_loss_checks']:,}"
            )
            for provider, stats in sorted(surface_report.providers.items()):
                lines.append(
                    f"    {provider}: conversations={stats.total_conversations:,} "
                    f"clean={stats.clean_conversations:,} "
                    f"critical={stats.critical_conversations:,} "
                    f"preserved_checks={stats.preserved_checks:,} "
                    f"declared_loss_checks={stats.declared_loss_checks:,} "
                    f"critical_loss_checks={stats.critical_loss_checks:,}"
                )

    if result.semantic_contracts is not None:
        lines.extend(["", f"Semantic contracts: {len(result.semantic_contracts):,} surfaces"])
        for spec in result.semantic_contracts:
            details: list[str] = [f"category={spec.category}"]
            if spec.aliases:
                details.append(f"aliases={','.join(spec.aliases)}")
            if spec.export_format:
                details.append(f"export_format={spec.export_format}")
            if spec.stream_format:
                details.append(f"stream_format={spec.stream_format}")
            details.append(f"contracts={len(spec.contracts)}")
            lines.append(f"  {spec.name}: {'; '.join(details)}")
            metric_bits = [f"{contract.metric}:{contract.mode}" for contract in spec.contracts]
            lines.append(f"    metrics={', '.join(metric_bits)}")


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


__all__ = ["append_roundtrip_lines", "append_semantic_lines"]
