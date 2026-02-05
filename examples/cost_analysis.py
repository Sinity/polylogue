#!/usr/bin/env python3
"""Analyze conversation costs across providers."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta

from polylogue import Polylogue


def extract_cost(message) -> float | None:
    """Extract cost_usd from message metadata."""
    if not message.provider_meta:
        return None

    # Direct cost field
    if "cost_usd" in message.provider_meta:
        cost = message.provider_meta["cost_usd"]
        if isinstance(cost, (int, float)):
            return float(cost)

    # Nested in usage object
    if "usage" in message.provider_meta and isinstance(message.provider_meta["usage"], dict):
        cost = message.provider_meta["usage"].get("cost_usd")
        if isinstance(cost, (int, float)):
            return float(cost)

    return None


def analyze_costs(archive: Polylogue, provider_filter: str | None = None, since: datetime | None = None):
    """Analyze conversation costs."""
    # Get all conversations
    conversations = list(archive.list_conversations(provider=provider_filter))

    # Filter by date if specified
    if since:
        conversations = [c for c in conversations if c.created_at and datetime.fromisoformat(c.created_at) >= since]

    # Aggregate costs by provider
    provider_costs = defaultdict(list)
    total_cost = 0.0
    conversations_with_cost = 0

    for conv in conversations:
        # Sum costs from all messages
        conv_cost = 0.0
        for msg in conv.messages:
            msg_cost = extract_cost(msg)
            if msg_cost is not None and msg_cost > 0:
                conv_cost += msg_cost

        if conv_cost > 0:
            provider_costs[conv.provider].append({
                "conversation_id": conv.id,
                "title": conv.title,
                "cost": conv_cost,
                "message_count": len(conv.messages),
                "total_chars": sum(len(m.text or "") for m in conv.messages),
                "created_at": conv.created_at,
            })
            total_cost += conv_cost
            conversations_with_cost += 1

    return {
        "total_cost": total_cost,
        "conversations_with_cost": conversations_with_cost,
        "total_conversations": len(conversations),
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

    print("Cost by Provider:")
    print(f"{'=' * 60}")

    for provider, convs in sorted(analysis['provider_costs'].items()):
        provider_cost = sum(c['cost'] for c in convs)
        avg_cost = provider_cost / len(convs)
        print(f"\n{provider.upper()}:")
        print(f"  Conversations: {len(convs):,}")
        print(f"  Total cost: ${provider_cost:.2f}")
        print(f"  Average cost: ${avg_cost:.4f} per conversation")

        # Top 5 most expensive conversations for this provider
        top_convs = sorted(convs, key=lambda x: x['cost'], reverse=True)[:5]
        if top_convs:
            print("\n  Top 5 most expensive:")
            for i, c in enumerate(top_convs, 1):
                title = (c['title'][:40] + '...') if len(c['title']) > 40 else c['title']
                cost_per_msg = c['cost'] / c['message_count'] if c['message_count'] > 0 else 0
                print(f"    {i}. ${c['cost']:.4f} - {title} ({c['message_count']} msgs, ${cost_per_msg:.6f}/msg)")

    print(f"\n{'=' * 60}")


def export_json(analysis: dict, output_path: str):
    """Export analysis to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nExported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze conversation costs")
    parser.add_argument("--provider", help="Filter by provider")
    parser.add_argument("--since-days", type=int, help="Only analyze conversations from last N days")
    parser.add_argument("--json", help="Export results to JSON file")
    args = parser.parse_args()

    archive = Polylogue()

    since = None
    if args.since_days:
        since = datetime.now() - timedelta(days=args.since_days)

    analysis = analyze_costs(archive, provider_filter=args.provider, since=since)
    print_report(analysis)

    if args.json:
        export_json(analysis, args.json)


if __name__ == "__main__":
    main()
