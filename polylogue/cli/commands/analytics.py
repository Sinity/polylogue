"""Analytics command for conversation insights."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from polylogue.analytics import compute_provider_comparison
from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError


@click.command("analytics")
@click.option("--provider-comparison", is_flag=True, help="Show provider comparison metrics")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--config", type=click.Path(path_type=Path), help="Path to config file")
@click.pass_obj
def analytics_command(
    env: AppEnv,
    provider_comparison: bool,
    json_output: bool,
    config: Path | None,
) -> None:
    """View analytics and insights about your conversation archive.

    Examples:
        polylogue analytics --provider-comparison
        polylogue analytics --provider-comparison --json
    """
    try:
        cfg = load_effective_config(config or env.config_path)
    except ConfigError as exc:
        fail("analytics", str(exc))

    # Default to provider comparison if no flags
    if not provider_comparison:
        provider_comparison = True

    if provider_comparison:
        metrics = compute_provider_comparison(cfg.archive_root)

        if json_output:
            output = [
                {
                    "provider": m.provider_name,
                    "conversations": m.conversation_count,
                    "messages": m.message_count,
                    "user_messages": m.user_message_count,
                    "assistant_messages": m.assistant_message_count,
                    "avg_messages_per_conversation": round(m.avg_messages_per_conversation, 1),
                    "avg_user_words": round(m.avg_user_words, 1),
                    "avg_assistant_words": round(m.avg_assistant_words, 1),
                    "tool_use_count": m.tool_use_count,
                    "thinking_count": m.thinking_count,
                    "tool_use_percentage": round(m.tool_use_percentage, 1),
                    "thinking_percentage": round(m.thinking_percentage, 1),
                }
                for m in metrics
            ]
            env.ui.console.print(json.dumps(output, indent=2))
        else:
            _display_provider_comparison(env.ui.console, metrics)


def _display_provider_comparison(console: Console, metrics: list) -> None:
    """Display provider comparison table with rich formatting."""
    table = Table(title="Provider Comparison", show_header=True, header_style="bold magenta")

    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Conversations", justify="right", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Avg Msgs/Conv", justify="right")
    table.add_column("Avg User Words", justify="right")
    table.add_column("Avg Asst Words", justify="right")
    table.add_column("Tool Use %", justify="right", style="yellow")
    table.add_column("Thinking %", justify="right", style="blue")

    for m in metrics:
        table.add_row(
            m.provider_name,
            str(m.conversation_count),
            str(m.message_count),
            f"{m.avg_messages_per_conversation:.1f}",
            f"{m.avg_user_words:.0f}",
            f"{m.avg_assistant_words:.0f}",
            f"{m.tool_use_percentage:.1f}%" if m.tool_use_count > 0 else "-",
            f"{m.thinking_percentage:.1f}%" if m.thinking_count > 0 else "-",
        )

    console.print(table)

    # Summary stats
    total_convs = sum(m.conversation_count for m in metrics)
    total_msgs = sum(m.message_count for m in metrics)
    console.print(f"\n[bold]Total:[/bold] {total_convs:,} conversations, {total_msgs:,} messages")
