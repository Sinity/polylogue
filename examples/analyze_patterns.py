#!/usr/bin/env python3
"""Analyze conversation patterns and characteristics.

Identifies conversations with heavy tool use, thinking traces,
calculates substantive vs noise ratios, and generates provider
comparison reports.

Usage:
    python examples/analyze_patterns.py                 # Full report
    python examples/analyze_patterns.py --thinking      # Focus on thinking-heavy convos
    python examples/analyze_patterns.py --tools         # Focus on tool-heavy convos
    python examples/analyze_patterns.py --json          # JSON output
"""

from __future__ import annotations

import argparse
import json

from polylogue import Polylogue


def analyze_thinking_patterns(archive: Polylogue):
    """Find conversations with heavy thinking content."""
    thinking_convs = []

    all_convs = archive.list_conversations()

    for conv_ref in all_convs:
        conv = archive.get_conversation(conv_ref.id)

        # Count thinking blocks
        thinking_msgs = list(conv.iter_thinking())
        if len(thinking_msgs) > 0:
            thinking_ratio = len(thinking_msgs) / len(conv.messages)
            thinking_convs.append({
                "id": conv.id,
                "title": conv.title,
                "provider": conv.provider,
                "thinking_count": len(thinking_msgs),
                "total_messages": len(conv.messages),
                "thinking_ratio": thinking_ratio,
            })

    # Sort by thinking ratio
    thinking_convs.sort(key=lambda x: x["thinking_ratio"], reverse=True)

    return thinking_convs


def analyze_tool_use_patterns(archive: Polylogue):
    """Find conversations with heavy tool use."""
    tool_convs = []

    all_convs = archive.list_conversations()

    for conv_ref in all_convs:
        conv = archive.get_conversation(conv_ref.id)

        # Count tool use messages (check provider_meta for tool_use or role="tool")
        tool_msgs = [m for m in conv.messages if m.role == "tool" or (m.provider_meta and "tool" in str(m.provider_meta).lower())]

        if len(tool_msgs) > 0:
            tool_ratio = len(tool_msgs) / len(conv.messages)
            tool_convs.append({
                "id": conv.id,
                "title": conv.title,
                "provider": conv.provider,
                "tool_count": len(tool_msgs),
                "total_messages": len(conv.messages),
                "tool_ratio": tool_ratio,
            })

    # Sort by tool ratio
    tool_convs.sort(key=lambda x: x["tool_ratio"], reverse=True)

    return tool_convs


def analyze_substantive_ratios(archive: Polylogue):
    """Calculate substantive vs noise ratios per provider."""
    provider_stats = {}

    all_convs = archive.list_conversations()

    for conv_ref in all_convs:
        conv = archive.get_conversation(conv_ref.id)

        if conv.provider not in provider_stats:
            provider_stats[conv.provider] = {
                "total_messages": 0,
                "substantive_messages": 0,
                "conversations": 0,
            }

        provider_stats[conv.provider]["conversations"] += 1
        provider_stats[conv.provider]["total_messages"] += len(conv.messages)

        # Count substantive messages
        substantive = list(conv.iter_substantive())
        provider_stats[conv.provider]["substantive_messages"] += len(substantive)

    # Calculate ratios
    for _provider, stats in provider_stats.items():
        if stats["total_messages"] > 0:
            stats["substantive_ratio"] = stats["substantive_messages"] / stats["total_messages"]
        else:
            stats["substantive_ratio"] = 0.0

    return provider_stats


def print_report(thinking_convs, tool_convs, provider_stats):
    """Print human-readable pattern analysis report."""
    print(f"\n{'=' * 70}")
    print("CONVERSATION PATTERN ANALYSIS")
    print(f"{'=' * 70}\n")

    # Provider substantive ratios
    print("SUBSTANTIVE MESSAGE RATIOS BY PROVIDER:")
    print(f"{'-' * 70}")
    print(f"{'Provider':<15} {'Conversations':>15} {'Messages':>12} {'Substantive %':>15}")
    print(f"{'-' * 70}")

    for provider, stats in sorted(provider_stats.items()):
        ratio_pct = stats["substantive_ratio"] * 100
        print(f"{provider:<15} {stats['conversations']:>15,} {stats['total_messages']:>12,} {ratio_pct:>14.1f}%")

    # Thinking-heavy conversations
    if thinking_convs:
        print(f"\n{'=' * 70}")
        print("TOP 10 THINKING-HEAVY CONVERSATIONS:")
        print(f"{'=' * 70}\n")
        print(f"{'Provider':<12} {'Thinking %':>12} {'Count':>8} {'Title':<35}")
        print(f"{'-' * 70}")

        for conv in thinking_convs[:10]:
            ratio_pct = conv["thinking_ratio"] * 100
            title = conv["title"][:32] + "..." if len(conv["title"]) > 35 else conv["title"]
            print(f"{conv['provider']:<12} {ratio_pct:>11.1f}% {conv['thinking_count']:>8} {title:<35}")

    # Tool-heavy conversations
    if tool_convs:
        print(f"\n{'=' * 70}")
        print("TOP 10 TOOL-HEAVY CONVERSATIONS:")
        print(f"{'=' * 70}\n")
        print(f"{'Provider':<12} {'Tool %':>10} {'Count':>8} {'Title':<35}")
        print(f"{'-' * 70}")

        for conv in tool_convs[:10]:
            ratio_pct = conv["tool_ratio"] * 100
            title = conv["title"][:32] + "..." if len(conv["title"]) > 35 else conv["title"]
            print(f"{conv['provider']:<12} {ratio_pct:>9.1f}% {conv['tool_count']:>8} {title:<35}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze conversation patterns")
    parser.add_argument("--thinking", action="store_true", help="Focus on thinking-heavy conversations")
    parser.add_argument("--tools", action="store_true", help="Focus on tool-heavy conversations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Create archive instance
    archive = Polylogue()

    # Run analyses
    thinking_convs = analyze_thinking_patterns(archive)
    tool_convs = analyze_tool_use_patterns(archive)
    provider_stats = analyze_substantive_ratios(archive)

    if args.json:
        output = {
            "thinking_conversations": thinking_convs,
            "tool_conversations": tool_convs,
            "provider_stats": provider_stats,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(thinking_convs, tool_convs, provider_stats)


if __name__ == "__main__":
    main()
