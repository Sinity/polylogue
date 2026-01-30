#!/usr/bin/env python3
"""Extract knowledge from conversations: code snippets, insights, topics.

Builds a searchable knowledge base from substantive conversation content.

Usage:
    python examples/extract_knowledge.py --code           # Extract all code blocks
    python examples/extract_knowledge.py --thinking       # Extract thinking traces
    python examples/extract_knowledge.py --topic rust     # Find all Rust discussions
    python examples/extract_knowledge.py --export kb.md   # Export to markdown
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polylogue import Polylogue


def extract_code_blocks(archive: Polylogue):
    """Extract all code blocks from conversations."""
    code_blocks = []

    # Pattern for code blocks
    code_pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

    all_convs = archive.list_conversations()

    for conv_ref in all_convs:
        conv = archive.get_conversation(conv_ref.id)

        for msg in conv.messages:
            if not msg.text:
                continue

            # Find all code blocks
            matches = code_pattern.findall(msg.text)
            for language, code in matches:
                code_blocks.append({
                    "language": language or "unknown",
                    "code": code.strip(),
                    "conversation_id": conv.id,
                    "conversation_title": conv.title,
                    "provider": conv.provider,
                })

    return code_blocks


def extract_thinking_traces(archive: Polylogue, topic: str | None = None):
    """Extract thinking blocks, optionally filtered by topic."""
    thinking_traces = []

    all_convs = archive.list_conversations()

    for conv_ref in all_convs:
        conv = archive.get_conversation(conv_ref.id)

        thinking_msgs = list(conv.iter_thinking())

        for msg in thinking_msgs:
            if not msg.text:
                continue

            # If topic filter specified, check if it appears in thinking
            if topic and topic.lower() not in msg.text.lower():
                continue

            thinking_traces.append({
                "text": msg.text,
                "conversation_id": conv.id,
                "conversation_title": conv.title,
                "provider": conv.provider,
                "timestamp": msg.timestamp,
            })

    return thinking_traces


def find_topic_discussions(archive: Polylogue, topic: str):
    """Find all substantive discussions about a topic."""
    discussions = []

    all_convs = archive.list_conversations()

    for conv_ref in all_convs:
        conv = archive.get_conversation(conv_ref.id)

        # Check if topic appears in substantive messages
        substantive = list(conv.iter_substantive())
        topic_mentions = [m for m in substantive if topic.lower() in (m.text or "").lower()]

        if topic_mentions:
            discussions.append({
                "conversation_id": conv.id,
                "title": conv.title,
                "provider": conv.provider,
                "mention_count": len(topic_mentions),
                "total_substantive": len(substantive),
                "messages": [{"role": m.role, "text": m.text[:200]} for m in topic_mentions[:3]],  # First 3 mentions
            })

    discussions.sort(key=lambda x: x["mention_count"], reverse=True)
    return discussions


def export_to_markdown(code_blocks, thinking_traces, output_path: Path):
    """Export extracted knowledge to markdown file."""
    lines = ["# Extracted Knowledge Base\n"]

    # Code snippets section
    if code_blocks:
        lines.append("## Code Snippets\n")
        lang_groups = {}
        for block in code_blocks:
            lang_groups.setdefault(block["language"], []).append(block)

        for lang, blocks in sorted(lang_groups.items()):
            lines.append(f"### {lang.title()} ({len(blocks)} snippets)\n")
            for i, block in enumerate(blocks[:10]):  # Limit to 10 per language
                lines.append(f"**Source**: {block['conversation_title']}\n")
                lines.append(f"```{lang}")
                lines.append(block["code"])
                lines.append("```\n")

    # Thinking traces section
    if thinking_traces:
        lines.append("## Thinking Traces\n")
        for i, trace in enumerate(thinking_traces[:20]):  # Limit to 20
            lines.append(f"### Trace {i+1}: {trace['conversation_title']}\n")
            lines.append(f"**Provider**: {trace['provider']}  ")
            lines.append(f"**Timestamp**: {trace['timestamp']}\n")
            lines.append(trace["text"][:500])  # Truncate long traces
            lines.append("\n---\n")

    output_path.write_text("\n".join(lines))
    print(f"✓ Exported to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract knowledge from conversations")
    parser.add_argument("--code", action="store_true", help="Extract code blocks")
    parser.add_argument("--thinking", action="store_true", help="Extract thinking traces")
    parser.add_argument("--topic", help="Find discussions about a topic")
    parser.add_argument("--export", help="Export to markdown file")

    args = parser.parse_args()

    # Create archive instance
    archive = Polylogue()

    code_blocks = []
    thinking_traces = []

    if args.code or not (args.code or args.thinking or args.topic):
        print("Extracting code blocks...")
        code_blocks = extract_code_blocks(archive)
        print(f"  Found {len(code_blocks)} code blocks")

        # Show distribution
        langs = {}
        for block in code_blocks:
            langs[block["language"]] = langs.get(block["language"], 0) + 1

        print("\n  Language distribution:")
        for lang, count in sorted(langs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {lang}: {count}")

    if args.thinking or not (args.code or args.thinking or args.topic):
        print("\nExtracting thinking traces...")
        thinking_traces = extract_thinking_traces(archive, topic=args.topic if args.topic else None)
        print(f"  Found {len(thinking_traces)} thinking traces")

    if args.topic:
        print(f"\nSearching for topic: '{args.topic}'...")
        discussions = find_topic_discussions(archive, args.topic)
        print(f"  Found {len(discussions)} conversations mentioning '{args.topic}'")

        if discussions:
            print(f"\n  Top discussions:")
            for disc in discussions[:5]:
                print(f"    - {disc['title']} ({disc['mention_count']} mentions)")

    if args.export:
        export_path = Path(args.export)
        export_to_markdown(code_blocks, thinking_traces, export_path)


if __name__ == "__main__":
    main()
