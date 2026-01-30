#!/usr/bin/env python3
"""Analyze conversation costs across providers.

Aggregates costs from provider metadata, shows trends over time,
identifies most expensive conversations, and calculates token efficiency.

Usage:
    python examples/cost_analysis.py                    # Summary report
    python examples/cost_analysis.py --provider claude  # Filter by provider
    python examples/cost_analysis.py --json             # JSON output
    python examples/cost_analysis.py --since 2024-01-01 # Filter by date
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polylogue import Polylogue


def extract_cost(provider_meta: dict | None) -> float | None:
    """Extract cost from provider metadata."""
    if not provider_meta:
        return None

    # Different providers store cost differently
    # ChatGPT: typically no cost tracking
    # Claude: sometimes has usage.cost_usd
    # Claude Code: has cost metadata
    if isinstance(provider_meta, dict):
        if "cost_usd" in provider_meta:
            return float(provider_meta["cost_usd"])
        if "usage" in provider_meta and isinstance(provider_meta["usage"], dict):
            return float(provider_meta["usage"].get("cost_usd", 0))

    return None


def analyze_costs(provider_filter: str | None = None, since: datetime | None = None):
    """Analyze conversation costs."""
    with open_connection(None) as conn:
        query = """
            SELECT
                c.conversation_id,
                c.provider_name,
                c.title,
                c.created_at,
                c.updated_at,
                c.provider_meta,
                (SELECT COUNT(*) FROM messages WHERE conversation_id = c.conversation_id) as message_count,
                (SELECT SUM(LENGTH(text)) FROM messages WHERE conversation_id = c.conversation_id) as total_chars
            FROM conversations c
        """
        params = []

        if provider_filter:
            query += " WHERE c.provider_name = ?"
            params.append(provider_filter)

        query += " ORDER BY c.created_at"

        rows = conn.execute(query, params).fetchall()

    # Aggregate costs by provider
    provider_costs = defaultdict(list)
    total_cost = 0.0
    conversations_with_cost = 0

    for row in rows:
        # Parse provider_meta
        meta = None
        if row["provider_meta"]:
            try:
                meta = json.loads(row["provider_meta"])
            except json.JSONDecodeError:
                pass

        cost = extract_cost(meta)
        if cost is not None and cost > 0:
            provider_costs[row["provider_name"]].append({
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "cost": cost,
                "message_count": row["message_count"],
                "total_chars": row["total_chars"] or 0,
                "created_at": row["created_at"],
            })
            total_cost += cost
            conversations_with_cost += 1

    return {
        "total_cost": total_cost,
        "conversations_with_cost": conversations_with_cost,
        "total_conversations": len(rows),
        "provider_costs": dict(provider_costs),
    }


def print_report(analysis: dict):
    """Print human-readable cost report."""
    print(f"\n{'=' * 60}")
    print("CONVERSATION COST ANALYSIS")
    print(f"{'=' * 60}\n")

    print(f"Total conversations analyzed: {analysis['total_conversations']:,}")
    print(f"Conversations with cost data: {analysis['conversations_with_cost']:,}")
    print(f"Total cost: ${analysis['total_cost']:.2f}\n")

    if not analysis['conversations_with_cost']:
        print("No cost data found in provider metadata.")
        print("Cost tracking is typically available in Claude Code conversations.")
        return

    print(f"{'Provider':<15} {'Conversations':>15} {'Total Cost':>15} {'Avg Cost':>15}")
    print(f"{'-' * 60}")

    for provider, convs in sorted(analysis['provider_costs'].items()):
        count = len(convs)
        total = sum(c['cost'] for c in convs)
        avg = total / count if count > 0 else 0
        print(f"{provider:<15} {count:>15,} ${total:>14.2f} ${avg:>14.4f}")

    # Show most expensive conversations
    print(f"\n{'=' * 60}")
    print("TOP 10 MOST EXPENSIVE CONVERSATIONS")
    print(f"{'=' * 60}\n")

    all_convs = []
    for provider, convs in analysis['provider_costs'].items():
        for conv in convs:
            conv['provider'] = provider
            all_convs.append(conv)

    all_convs.sort(key=lambda x: x['cost'], reverse=True)

    print(f"{'Provider':<12} {'Cost':>8} {'Messages':>10} {'Title':<40}")
    print(f"{'-' * 72}")
    for conv in all_convs[:10]:
        title = conv['title'][:37] + "..." if len(conv['title']) > 40 else conv['title']
        print(f"{conv['provider']:<12} ${conv['cost']:>7.2f} {conv['message_count']:>10} {title:<40}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze conversation costs")
    parser.add_argument("--provider", help="Filter by provider (chatgpt, claude, etc.)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--since", help="Filter conversations since date (YYYY-MM-DD)")

    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)

    analysis = analyze_costs(provider_filter=args.provider, since=since)

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_report(analysis)


if __name__ == "__main__":
    main()
