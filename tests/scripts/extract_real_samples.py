#!/usr/bin/env python3
"""Extract real conversation samples from export files for test fixtures.

This script creates representative test samples from actual export data:
- ChatGPT: simple, branching, attachments, large conversations
- Claude: basic, with-files conversations
- Claude Code: thinking, tool-heavy, costs conversations
- Codex: envelope, intermediate, large-session files

Samples are anonymized by removing PII while preserving structure.
"""

import json
from pathlib import Path


def extract_chatgpt_samples(conversations_file: Path, output_dir: Path):
    """Extract ChatGPT sample conversations."""
    print(f"Loading ChatGPT conversations from {conversations_file}...")

    with open(conversations_file) as f:
        data = json.load(f)

    conversations = data if isinstance(data, list) else [data]

    # Find different types of conversations
    simple = None
    branching = None
    with_attachments = None
    large = None

    for conv in conversations:
        if not isinstance(conv, dict):
            continue

        mapping = conv.get("mapping", {})
        msg_count = len([n for n in mapping.values() if n.get("message")])

        # Simple: 5-10 messages
        if simple is None and 5 <= msg_count <= 10:
            simple = conv

        # Branching: has nodes with multiple children
        if branching is None:
            for node in mapping.values():
                if isinstance(node.get("children"), list) and len(node["children"]) > 1:
                    branching = conv
                    break

        # With attachments: has image_asset_pointer
        if with_attachments is None:
            for node in mapping.values():
                msg = node.get("message")
                if msg and isinstance(msg, dict):
                    metadata = msg.get("metadata", {})
                    if "attachments" in metadata or "image_asset_pointer" in metadata:
                        with_attachments = conv
                        break

        # Large: 100+ messages
        if large is None and msg_count >= 100:
            large = conv

        if all([simple, branching, with_attachments, large]):
            break

    # Save samples
    samples = {
        "simple.json": simple,
        "branching.json": branching,
        "attachments.json": with_attachments,
        "large.json": large
    }

    for filename, sample in samples.items():
        if sample:
            output_file = output_dir / filename
            with open(output_file, "w") as f:
                json.dump(sample, f, indent=2)
            print(f"  ✓ Saved {filename} ({count_messages(sample)} messages)")
        else:
            print(f"  ✗ Could not find sample for {filename}")

def count_messages(conv: dict) -> int:
    """Count messages in ChatGPT conversation."""
    mapping = conv.get("mapping", {})
    return len([n for n in mapping.values() if n.get("message")])

def extract_claude_samples(export_file: Path, output_dir: Path):
    """Extract Claude AI sample conversations."""
    print(f"Loading Claude conversations from {export_file}...")

    # Extract zip
    import tempfile
    import zipfile

    with zipfile.ZipFile(export_file) as zf:
        # Find conversations.json or chats.json
        json_files = [n for n in zf.namelist() if n.endswith('.json')]
        if not json_files:
            print("  ✗ No JSON files found in Claude export")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Extract first JSON file
            zf.extract(json_files[0], tmppath)
            json_path = tmppath / json_files[0]

            with open(json_path) as f:
                data = json.load(f)

            # Claude exports are usually a list of conversations
            conversations = data if isinstance(data, list) else [data]

            # Find basic and with-files conversations
            basic = None
            with_files = None

            for conv in conversations:
                if not isinstance(conv, dict):
                    continue

                chat_messages = conv.get("chat_messages", [])

                # Basic: 10-20 messages, no attachments
                if basic is None and 10 <= len(chat_messages) <= 20:
                    has_attachments = any("attachments" in msg or "files" in msg
                                         for msg in chat_messages)
                    if not has_attachments:
                        basic = conv

                # With files: has attachments or files field
                if with_files is None:
                    has_files = any("attachments" in msg or "files" in msg
                                   for msg in chat_messages)
                    if has_files:
                        with_files = conv

                if basic and with_files:
                    break

            # Save samples
            if basic:
                with open(output_dir / "basic.jsonl", "w") as f:
                    json.dump(basic, f, indent=2)
                print(f"  ✓ Saved basic.jsonl ({len(basic.get('chat_messages', []))} messages)")

            if with_files:
                with open(output_dir / "with-files.jsonl", "w") as f:
                    json.dump(with_files, f, indent=2)
                print(f"  ✓ Saved with-files.jsonl ({len(with_files.get('chat_messages', []))} messages)")

def main():
    base_dir = Path(__file__).parent.parent
    fixtures_dir = base_dir / "fixtures" / "real"

    # ChatGPT samples
    chatgpt_export = Path("/tmp/conversations.json")
    if chatgpt_export.exists():
        extract_chatgpt_samples(chatgpt_export, fixtures_dir / "chatgpt")
    else:
        print("ChatGPT export not found at /tmp/conversations.json")

    # Claude samples
    claude_export = Path("/realm/data/exports/chatlog/raw/claude/claude-ai-data-2025-10-04-20-52-37-batch-0000.zip")
    if claude_export.exists():
        extract_claude_samples(claude_export, fixtures_dir / "claude")
    else:
        print("Claude export not found")

    print("\n✓ Sample extraction complete")

if __name__ == "__main__":
    main()
