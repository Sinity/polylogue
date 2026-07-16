"""CLI summary rendering helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.helper_support import load_effective_config
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config, Source
from polylogue.insights.archive import ArchiveCoverageInsight
from polylogue.logging import get_logger
from polylogue.readiness import ReadinessReport
from polylogue.services import RuntimeServices
from polylogue.storage.embeddings import embedding_status_payload
from polylogue.ui.theme import provider_color

logger = get_logger(__name__)


class GetOriginCountsFn(Protocol):
    def __call__(
        self,
        *,
        services: RuntimeServices | None = None,
        db_path: Path | None = None,
    ) -> Awaitable[list[tuple[str, int]]]: ...


class ListArchiveCoverageInsightsFn(Protocol):
    def __call__(
        self,
        *,
        services: RuntimeServices | None = None,
        db_path: Path | None = None,
    ) -> Awaitable[list[ArchiveCoverageInsight]]: ...


def print_summary_impl(
    env: AppEnv,
    *,
    verbose: bool = False,
    format_sources_summary_fn: Callable[[list[Source]], str],
    quick_readiness_summary_fn: Callable[[Path], str],
    get_readiness_fn: Callable[[Config], ReadinessReport],
    get_origin_counts_fn: GetOriginCountsFn,
    list_archive_coverage_insights_fn: ListArchiveCoverageInsightsFn,
) -> None:
    ui = env.ui
    config = load_effective_config(env)
    embedding_stats = None

    try:
        embedding_stats = embedding_status_payload(env)
    except Exception:
        logger.warning("Embedding status computation failed; summary will omit stats", exc_info=True)

    lines = [
        f"Archive: {config.archive_root}",
        f"Render: {config.render_root}",
        f"Sources: {format_sources_summary_fn(config.sources)}",
        "Ingestion: owned by polylogued",
    ]
    if embedding_stats is not None:
        embedding_line = (
            f"Embeddings: {embedding_stats['embedded_sessions']:,}/{embedding_stats['total_sessions']:,} convs, "
            f"{embedding_stats['embedded_messages']:,} msgs ({embedding_stats['embedding_coverage_percent']:.1f}%)"
        )
        pending_embedding_sessions = embedding_stats["pending_sessions"]
        stale_embedding_messages = embedding_stats["stale_messages"]
        missing_embedding_provenance = embedding_stats["messages_missing_provenance"]
        if pending_embedding_sessions:
            embedding_line += f", pending {pending_embedding_sessions:,}"
        if stale_embedding_messages:
            embedding_line += f", stale {stale_embedding_messages:,}"
        if missing_embedding_provenance:
            embedding_line += f", missing provenance {missing_embedding_provenance:,}"
        lines.append(embedding_line)

    if verbose:
        report = get_readiness_fn(config)
        provenance = report.provenance
        source_val = getattr(provenance.source, "value", provenance.source) if hasattr(provenance, "source") else "live"
        readiness_header = f"Readiness (source={source_val})"
        lines.append(readiness_header)
        for check in report.checks:
            status_str = str(check.status) if check.status else "?"
            icon = {"ok": "[green]✓[/green]", "warning": "[yellow]![/yellow]", "error": "[red]✗[/red]"}.get(
                status_str,
                "?",
            )
            if ui.plain:
                icon = {"ok": "OK", "warning": "WARN", "error": "ERR"}.get(status_str, "?")
            lines.append(f"  {icon} {check.name}: {check.detail}")
    else:
        lines.append(f"Readiness: {quick_readiness_summary_fn(config.archive_root)}")

    ui.summary("Polylogue", lines)

    try:
        if verbose:
            metrics = run_coroutine_sync(list_archive_coverage_insights_fn(services=env.services))
            counts: list[tuple[str, int]] = [
                (metric.origin or metric.bucket, metric.session_count) for metric in metrics
            ]
        else:
            counts = run_coroutine_sync(get_origin_counts_fn(services=env.services))
            metrics = []

        if counts:
            ui.console.print()
            total_convs = sum(count for _, count in counts)
            ui.console.print(f"[bold]Archive:[/bold] {total_convs:,} sessions")

            if total_convs == 0:
                ui.console.print()
                inbox = config.archive_root / "inbox"
                ui.console.print(
                    "[yellow]No sessions yet. Drop export files in[/yellow] "
                    f"[bold]{inbox}[/bold]"
                    " [yellow]and start[/yellow] [bold]polylogued run[/bold]"
                )

            max_width = 30
            for source_name, conv_count in counts:
                if total_convs > 0:
                    pct = (conv_count / total_convs) * 100
                    bar_len = int((conv_count / total_convs) * max_width)
                else:
                    pct = 0
                    bar_len = 0

                bar = "█" * bar_len
                color = provider_color(source_name).hex

                name_padded = f"{source_name}:".ljust(14)
                count_padded = f"{conv_count:,}".rjust(5)
                pct_padded = f"({pct:.0f}%)".rjust(5)

                ui.console.print(f"  {name_padded} {count_padded} {pct_padded}  │  [{color}]{bar}[/{color}]")

            if verbose and metrics:
                ui.console.print()
                ui.console.print("[bold]Deep Dive:[/bold]")
                for metric in metrics:
                    ui.console.print(f"[bold]{metric.origin}[/bold]")
                    avg_msgs = (
                        "n/a" if metric.avg_messages_per_session is None else f"{metric.avg_messages_per_session:.1f}"
                    )
                    ui.console.print(f"  Messages: {metric.message_count:,} (avg {avg_msgs}/conv)")
                    avg_user = "n/a" if metric.avg_user_words is None else str(int(metric.avg_user_words))
                    avg_asst = "n/a" if metric.avg_assistant_words is None else str(int(metric.avg_assistant_words))
                    ui.console.print(f"  Words: {avg_user} user / {avg_asst} asst (avg)")
                    if metric.avg_authored_user_words is not None and (
                        metric.avg_user_words is None
                        or int(metric.avg_authored_user_words) != int(metric.avg_user_words)
                    ):
                        ui.console.print(f"  Authored user words: {int(metric.avg_authored_user_words)} (avg)")
                    if metric.tool_use_count > 0:
                        tool_pct = "n/a" if metric.tool_use_percentage is None else f"{metric.tool_use_percentage:.1f}"
                        ui.console.print(f"  Tool Use: {metric.tool_use_count:,} ({tool_pct}% of convs)")
                    if metric.thinking_count > 0:
                        think_pct = "n/a" if metric.thinking_percentage is None else f"{metric.thinking_percentage:.1f}"
                        ui.console.print(f"  Thinking: {metric.thinking_count:,} ({think_pct}% of convs)")
                    ui.console.print()

    except Exception:
        logger.warning("Analytics computation failed; summary will omit analytics", exc_info=True)


__all__ = ["print_summary_impl"]
