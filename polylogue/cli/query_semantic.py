"""Semantic action/tool/path slice helpers and grouped stats output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.cli.query_feedback import emit_no_results
from polylogue.cli.query_stats import emit_structured_stats

if TYPE_CHECKING:
    from polylogue.archive.action_event.action_events import ActionEvent
    from polylogue.archive.models import SessionSummary
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.semantic.facts import SessionSemanticFacts
    from polylogue.cli.shared.types import AppEnv


# ---------------------------------------------------------------------------
# Semantic slice (from query_semantic_slice.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticStatsSlice:
    referenced_path: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()

    @classmethod
    def from_selection(cls, selection: SessionQuerySpec | None) -> SemanticStatsSlice:
        if selection is None:
            return cls()
        return cls(
            referenced_path=selection.referenced_path,
            action_terms=selection.action_terms,
            excluded_action_terms=selection.excluded_action_terms,
            action_text_terms=selection.action_text_terms,
            tool_terms=selection.tool_terms,
            excluded_tool_terms=selection.excluded_tool_terms,
        )

    def has_filters(self) -> bool:
        return any(
            (
                self.referenced_path,
                self.action_terms,
                self.excluded_action_terms,
                self.action_text_terms,
                self.tool_terms,
                self.excluded_tool_terms,
            )
        )


def normalized_tool_name(action: ActionEvent) -> str:
    return action.normalized_tool_name


def referenced_path_matches_slice(action: ActionEvent, referenced_path: tuple[str, ...]) -> bool:
    if not referenced_path:
        return True
    affected_paths = tuple(path.lower().replace("\\", "/") for path in action.affected_paths)
    if not affected_paths:
        return False
    return any(any(term.lower().replace("\\", "/") in path for path in affected_paths) for term in referenced_path)


def action_matches_slice(action: ActionEvent, semantic_slice: SemanticStatsSlice) -> bool:
    if not referenced_path_matches_slice(action, semantic_slice.referenced_path):
        return False

    if "none" in semantic_slice.action_terms:
        return False
    required_action_terms = {term for term in semantic_slice.action_terms if term != "none"}
    if required_action_terms and action.kind.value not in required_action_terms:
        return False
    blocked_action_terms = {term for term in semantic_slice.excluded_action_terms if term != "none"}
    if action.kind.value in blocked_action_terms:
        return False
    if semantic_slice.action_text_terms:
        search_text = action.search_text.lower()
        if not all(term.lower() in search_text for term in semantic_slice.action_text_terms):
            return False

    tool_name = normalized_tool_name(action)
    if "none" in semantic_slice.tool_terms:
        return False
    required_tool_terms = {term for term in semantic_slice.tool_terms if term != "none"}
    if required_tool_terms and tool_name not in required_tool_terms:
        return False
    blocked_tool_terms = {term for term in semantic_slice.excluded_tool_terms if term != "none"}
    return tool_name not in blocked_tool_terms


def filtered_action_events(
    facts: SessionSemanticFacts,
    semantic_slice: SemanticStatsSlice,
) -> tuple[ActionEvent, ...]:
    if not semantic_slice.has_filters():
        return facts.action_events
    return tuple(action for action in facts.action_events if action_matches_slice(action, semantic_slice))


# ---------------------------------------------------------------------------
# Semantic stats (from query_semantic_stats.py)
# ---------------------------------------------------------------------------


async def output_stats_by_semantic_summaries(
    env: AppEnv,
    summaries: list[SessionSummary],
    dimension: str,
    *,
    selection: SessionQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 50,
) -> None:
    await output_stats_by_semantic_ids(
        env,
        [str(summary.id) for summary in summaries],
        dimension,
        selection=selection,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_semantic_query(
    env: AppEnv,
    session_ids: list[str],
    dimension: str,
    *,
    selection: SessionQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 50,
) -> None:
    await output_stats_by_semantic_ids(
        env,
        session_ids,
        dimension,
        selection=selection,
        output_format=output_format,
        batch_size=batch_size,
    )


async def output_stats_by_semantic_ids(
    env: AppEnv,
    session_ids: list[str],
    dimension: str,
    *,
    selection: SessionQuerySpec | None = None,
    output_format: str = "text",
    batch_size: int = 50,
) -> None:
    from collections import Counter, defaultdict

    from rich.table import Table

    if dimension not in {"action", "tool"}:
        raise ValueError(f"Unsupported semantic stats dimension: {dimension}")
    if not session_ids:
        emit_no_results(env, selection=selection, output_format=output_format)

    semantic_slice = SemanticStatsSlice.from_selection(selection)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"convs": 0, "facts": 0, "msgs": 0})
    matched_facts = 0
    matched_messages = 0
    key_func = (lambda action: action.kind.value) if dimension == "action" else normalized_tool_name

    poly = env.polylogue
    for offset in range(0, len(session_ids), batch_size):
        batch_ids = session_ids[offset : offset + batch_size]
        # Archives have no materialized action_events table; events are
        # derived on read from each session's tool blocks (the same derivation
        # the legacy materializer hashed into rows), so there is one code path.
        action_events_by_session = await poly.get_action_events_batch(batch_ids)
        session_actions = {
            session_id: tuple(
                action
                for action in action_events_by_session.get(session_id, ())
                if action_matches_slice(action, semantic_slice)
            )
            for session_id in batch_ids
        }

        for session_id in batch_ids:
            filtered_actions = session_actions.get(session_id, ())
            group_counts = Counter(key_func(action) for action in filtered_actions)
            if not group_counts:
                groups["none"]["convs"] += 1
                continue

            matched_facts += sum(group_counts.values())
            message_groups: dict[str, set[str]] = defaultdict(set)
            for action in filtered_actions:
                message_groups[key_func(action)].add(action.message_id)

            matched_messages += len({action.message_id for action in filtered_actions})
            for key, fact_count in group_counts.items():
                groups[key]["convs"] += 1
                groups[key]["facts"] += fact_count
                groups[key]["msgs"] += len(message_groups[key])

    rows = [
        {
            "group": key,
            "sessions": stats["convs"],
            "facts": stats["facts"],
            "messages": stats["msgs"],
        }
        for key, stats in sorted(groups.items())
    ]
    summary = {
        "group": "MATCHED",
        "sessions": len(session_ids),
        "facts": matched_facts,
        "messages": matched_messages,
    }
    if emit_structured_stats(
        output_format=output_format,
        dimension=dimension,
        rows=rows,
        summary=summary,
        multi_membership=True,
    ):
        return

    env.ui.console.print(f"\nMatched: {len(session_ids)} sessions (by {dimension})\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Group", style="bold", min_width=12)
    table.add_column("Convs", justify="right")
    table.add_column("Facts", justify="right")
    table.add_column("Msgs", justify="right")
    for row in rows:
        table.add_row(
            str(row["group"]),
            f"{row['sessions']:,}",
            f"{row['facts']:,}",
            f"{row['messages']:,}",
        )
    table.add_section()
    table.add_row(
        "[bold]MATCHED[/]",
        f"[bold]{summary['sessions']:,}[/]",
        f"[bold]{summary['facts']:,}[/]",
        f"[bold]{summary['messages']:,}[/]",
    )
    env.ui.console.print(table)
    env.ui.console.print(f"Note: sessions may appear in multiple {dimension} groups.")


__all__ = [
    "SemanticStatsSlice",
    "action_matches_slice",
    "filtered_action_events",
    "normalized_tool_name",
    "output_stats_by_semantic_ids",
    "output_stats_by_semantic_query",
    "output_stats_by_semantic_summaries",
    "referenced_path_matches_slice",
]
