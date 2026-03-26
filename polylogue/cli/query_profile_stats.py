"""Profile-backed grouped stats output helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.query_stats_structured import emit_structured_stats

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import ConversationSummary
    from polylogue.storage.repository import ConversationRepository


async def output_stats_by_profile_summaries(
    env: AppEnv,
    summaries: list[ConversationSummary],
    repo: ConversationRepository,
    dimension: str,
    *,
    output_format: str = "text",
    batch_size: int = 100,
) -> None:
    await output_stats_by_profile_ids(
        env,
        [str(summary.id) for summary in summaries],
        repo,
        dimension,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_profile_query(
    env: AppEnv,
    conversation_ids: list[str],
    repo: ConversationRepository,
    dimension: str,
    *,
    output_format: str = "text",
    batch_size: int = 100,
) -> None:
    await output_stats_by_profile_ids(
        env,
        conversation_ids,
        repo,
        dimension,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_profile_ids(
    env: AppEnv,
    conversation_ids: list[str],
    repo: ConversationRepository,
    dimension: str,
    *,
    output_format: str = "text",
    batch_size: int = 100,
) -> None:
    from collections import defaultdict

    from rich.table import Table

    if dimension not in {"project", "work-kind"}:
        raise ValueError(f"Unsupported profile stats dimension: {dimension}")
    if not conversation_ids:
        env.ui.console.print("No conversations matched.")
        return

    from polylogue.lib.session_profile import build_session_profile

    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"conversations": 0, "work_events": 0, "messages": 0}
    )
    matched_conversations = 0
    matched_work_events = 0
    matched_messages = 0

    for offset in range(0, len(conversation_ids), batch_size):
        batch_ids = conversation_ids[offset : offset + batch_size]
        profiles_by_id = await repo.get_session_profiles_batch(batch_ids)
        missing_ids = [conversation_id for conversation_id in batch_ids if conversation_id not in profiles_by_id]
        if missing_ids:
            for conversation in await repo.get_many(missing_ids):
                profiles_by_id[str(conversation.id)] = build_session_profile(conversation)

        for conversation_id in batch_ids:
            profile = profiles_by_id.get(conversation_id)
            if profile is None:
                groups["none"]["conversations"] += 1
                continue

            if dimension == "project":
                keys = profile.canonical_projects or ("none",)
            else:
                primary_kind = next(
                    (tag.split(":", 1)[1] for tag in profile.auto_tags if tag.startswith("kind:")),
                    None,
                )
                keys = (primary_kind or "none",)

            matched_conversations += 1
            matched_work_events += len(profile.work_events)
            matched_messages += profile.message_count

            for key in keys:
                groups[key]["conversations"] += 1
                groups[key]["work_events"] += len(profile.work_events)
                groups[key]["messages"] += profile.message_count

    rows = [
        {
            "group": key,
            "conversations": stats["conversations"],
            "work_events": stats["work_events"],
            "messages": stats["messages"],
        }
        for key, stats in sorted(groups.items(), key=lambda item: item[0])
    ]
    summary = {
        "group": "MATCHED",
        "conversations": matched_conversations,
        "work_events": matched_work_events,
        "messages": matched_messages,
    }
    multi_membership = dimension == "project"
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
        multi_membership=multi_membership,
    ):
        return

    env.ui.console.print(f"\nMatched: {matched_conversations} conversations (by {dimension})\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Msgs", justify="right")
    for row in rows:
        table.add_row(
            str(row["group"]),
            f"{row['conversations']:,}",
            f"{row['work_events']:,}",
            f"{row['messages']:,}",
        )
    table.add_section()
    table.add_row(
        "[bold]MATCHED[/]",
        f"[bold]{summary['conversations']:,}[/]",
        f"[bold]{summary['work_events']:,}[/]",
        f"[bold]{summary['messages']:,}[/]",
    )
    env.ui.console.print(table)
    if multi_membership:
        env.ui.console.print("Note: conversations may appear in multiple project groups.")


__all__ = [
    "output_stats_by_profile_ids",
    "output_stats_by_profile_query",
    "output_stats_by_profile_summaries",
]
