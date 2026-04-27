from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.filter.filters import ConversationFilter
    from polylogue.lib.models import Conversation


def _pick_index(choice: str, total_results: int) -> int | None:
    if not choice:
        return 0
    try:
        idx = int(choice) - 1
    except ValueError:
        return None
    if 0 <= idx < total_results:
        return idx
    return None


async def pick_filter(f: ConversationFilter) -> Conversation | None:
    """Interactive picker for matching conversations.

    If running in a TTY, presents a menu to select from matches.
    Otherwise returns first match.
    """
    results = await f.list()
    if not results:
        return None

    if not sys.stdout.isatty():
        return results[0]

    print(f"\n{len(results)} matching conversations:\n")
    for i, conv in enumerate(results[:20], 1):
        title = conv.display_title[:50]
        date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else "unknown"
        print(f"  {i:2}. [{conv.provider}] {title} ({date})")

    if len(results) > 20:
        print(f"\n  ... and {len(results) - 20} more")

    try:
        choice = input("\nSelect number (or Enter for first): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    idx = _pick_index(choice, len(results))
    if idx is not None:
        return results[idx]

    return None


__all__ = ["_pick_index", "pick_filter"]
