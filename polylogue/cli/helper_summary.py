"""CLI summary rendering helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from polylogue.cli.helper_support import load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.logging import get_logger
from polylogue.sync_bridge import run_coroutine_sync
from polylogue.ui.theme import provider_color

logger = get_logger(__name__)


def print_summary_impl(
    env: AppEnv,
    *,
    verbose: bool = False,
    latest_run_fn: Callable[[Any], Awaitable[Any]],
    format_sources_summary_fn: Callable[[Any], str],
    quick_health_summary_fn: Callable[[Any], str],
    get_health_fn: Callable[..., Any],
    get_provider_counts_fn: Callable[..., Awaitable[list[tuple[str, int]]]],
    list_provider_analytics_products_fn: Callable[..., Awaitable[list[Any]]],
) -> None:
    ui = env.ui
    config = load_effective_config(env)
    archive_stats = None

    last_run_data = run_coroutine_sync(latest_run_fn(env.backend))
    last_line = "Last run: none"
    if last_run_data:
        last_line = f"Last run: {last_run_data.run_id} ({last_run_data.timestamp})"

    try:
        archive_stats = run_coroutine_sync(env.repository.get_archive_stats())
    except Exception:
        logger.debug("Archive stats computation failed", exc_info=True)

    lines = [
        f"Archive: {config.archive_root}",
        f"Render: {config.render_root}",
        f"Sources: {format_sources_summary_fn(config.sources)}",
        last_line,
    ]
    if archive_stats is not None:
        embedding_line = (
            f"Embeddings: {archive_stats.embedded_conversations:,}/{archive_stats.total_conversations:,} convs, "
            f"{archive_stats.embedded_messages:,} msgs ({archive_stats.embedding_coverage:.1f}%)"
        )
        pending_embedding_conversations = getattr(
            archive_stats,
            "pending_embedding_conversations",
            0,
        )
        stale_embedding_messages = getattr(
            archive_stats,
            "stale_embedding_messages",
            0,
        )
        missing_embedding_provenance = getattr(
            archive_stats,
            "messages_missing_embedding_provenance",
            0,
        )
        if pending_embedding_conversations:
            embedding_line += f", pending {pending_embedding_conversations:,}"
        if stale_embedding_messages:
            embedding_line += f", stale {stale_embedding_messages:,}"
        if missing_embedding_provenance:
            embedding_line += f", missing provenance {missing_embedding_provenance:,}"
        lines.append(embedding_line)

    if verbose:
        report = get_health_fn(config)
        provenance = report.provenance
        source_val = getattr(provenance.source, "value", provenance.source) if hasattr(provenance, "source") else "live"
        health_header = f"Health (source={source_val})"
        lines.append(health_header)
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
        lines.append(f"Health: {quick_health_summary_fn(config.archive_root)}")

    ui.summary("Polylogue", lines)

    try:
        if verbose:
            metrics = run_coroutine_sync(list_provider_analytics_products_fn(services=env.services))
            counts: list[tuple[str, int]] = [(metric.provider_name, metric.conversation_count) for metric in metrics]
        else:
            counts = run_coroutine_sync(get_provider_counts_fn(services=env.services))
            metrics = []

        if counts:
            ui.console.print()
            total_convs = sum(count for _, count in counts)
            ui.console.print(f"[bold]Archive:[/bold] {total_convs:,} conversations")

            max_width = 30
            for provider_name, conv_count in counts:
                if total_convs > 0:
                    pct = (conv_count / total_convs) * 100
                    bar_len = int((conv_count / total_convs) * max_width)
                else:
                    pct = 0
                    bar_len = 0

                bar = "█" * bar_len
                color = provider_color(provider_name).hex

                name_padded = f"{provider_name}:".ljust(14)
                count_padded = f"{conv_count:,}".rjust(5)
                pct_padded = f"({pct:.0f}%)".rjust(5)

                ui.console.print(f"  {name_padded} {count_padded} {pct_padded}  │  [{color}]{bar}[/{color}]")

            if verbose and metrics:
                ui.console.print()
                ui.console.print("[bold]Deep Dive:[/bold]")
                for metric in metrics:
                    ui.console.print(f"[bold]{metric.provider_name}[/bold]")
                    ui.console.print(
                        f"  Messages: {metric.message_count:,} (avg {metric.avg_messages_per_conversation:.1f}/conv)"
                    )
                    ui.console.print(
                        f"  Words: {int(metric.avg_user_words)} user / {int(metric.avg_assistant_words)} asst (avg)"
                    )
                    if metric.tool_use_count > 0:
                        ui.console.print(
                            f"  Tool Use: {metric.tool_use_count:,} ({metric.tool_use_percentage:.1f}% of convs)"
                        )
                    if metric.thinking_count > 0:
                        ui.console.print(
                            f"  Thinking: {metric.thinking_count:,} ({metric.thinking_percentage:.1f}% of convs)"
                        )
                    ui.console.print()

    except Exception:
        logger.debug("Analytics computation failed", exc_info=True)


__all__ = ["print_summary_impl"]
