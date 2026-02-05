#!/usr/bin/env python3
"""Update golden reference files from current rendering output.

This script regenerates all golden reference files by:
1. Running the test suite to generate current output
2. Extracting rendered output from test fixtures
3. Saving to tests/golden/<test-case>/expected/ directories

Usage:
    python tests/scripts/update_golden.py                    # Update all
    python tests/scripts/update_golden.py chatgpt-simple     # Update specific case

When to use:
    - After intentionally changing rendering logic
    - When adding new golden test cases
    - After reviewing diffs and confirming changes are correct

WARNING: This overwrites golden files. Review changes with git diff before committing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from polylogue.rendering.core import ConversationFormatter
from tests.helpers import DbFactory


def create_golden_chatgpt_simple(golden_dir: Path, db_path: Path):
    """Create golden files for chatgpt-simple test case."""
    factory = DbFactory(db_path)

    # Create test conversation (same as in test)
    conv_id = "golden-chatgpt-simple"
    factory.create_conversation(
        id=conv_id,
        provider="chatgpt",
        title="Simple ChatGPT Conversation",
        messages=[
            {
                "id": "msg1",
                "role": "user",
                "text": "Hello, how are you?",
                "timestamp": "2024-01-01T12:00:00Z",
            },
            {
                "id": "msg2",
                "role": "assistant",
                "text": "I'm doing well, thank you for asking! How can I help you today?",
                "timestamp": "2024-01-01T12:00:05Z",
            },
            {
                "id": "msg3",
                "role": "user",
                "text": "Can you explain what markdown is?",
                "timestamp": "2024-01-01T12:00:15Z",
            },
            {
                "id": "msg4",
                "role": "assistant",
                "text": "Markdown is a lightweight markup language for creating formatted text using a plain-text editor.",
                "timestamp": "2024-01-01T12:00:20Z",
            },
        ],
    )

    # Render
    formatter = ConversationFormatter(Path("/tmp/archive"))
    formatted = formatter.format(conv_id)

    # Save golden file
    case_dir = golden_dir / "chatgpt-simple"
    case_dir.mkdir(parents=True, exist_ok=True)
    expected_dir = case_dir / "expected"
    expected_dir.mkdir(exist_ok=True)

    (expected_dir / "conversation.md").write_text(formatted.markdown_text)

    print(f"✓ Updated {case_dir / 'expected/conversation.md'}")


def create_golden_claude_thinking(golden_dir: Path, db_path: Path):
    """Create golden files for claude-thinking test case."""
    factory = DbFactory(db_path)

    conv_id = "golden-claude-thinking"
    factory.create_conversation(
        id=conv_id,
        provider="claude",
        title="Claude with Thinking",
        messages=[
            {"id": "msg1", "role": "user", "text": "What is 2+2?"},
            {
                "id": "msg2",
                "role": "assistant",
                "text": "<thinking>\nThis is a simple arithmetic question. 2+2 equals 4.\n</thinking>\n\nThe answer is 4.",
            },
        ],
    )

    formatter = ConversationFormatter(Path("/tmp/archive"))
    formatted = formatter.format(conv_id)

    case_dir = golden_dir / "claude-thinking"
    case_dir.mkdir(parents=True, exist_ok=True)
    expected_dir = case_dir / "expected"
    expected_dir.mkdir(exist_ok=True)

    (expected_dir / "conversation.md").write_text(formatted.markdown_text)

    print(f"✓ Updated {case_dir / 'expected/conversation.md'}")


def main():
    """Main entry point."""
    golden_dir = PROJECT_ROOT / "tests" / "golden"
    db_path = Path("/tmp/polylogue-golden-update.db")

    # Clean up old test DB
    if db_path.exists():
        db_path.unlink()

    print(f"Updating golden files in {golden_dir}")
    print()

    # Update each test case
    create_golden_chatgpt_simple(golden_dir, db_path)
    create_golden_claude_thinking(golden_dir, db_path)

    print()
    print("Golden files updated. Review with: git diff tests/golden/")
    print("If changes look correct, commit them.")


if __name__ == "__main__":
    main()
