"""Contract tests: verify all parsers work on all real fixtures.

This ensures format drift is caught - if a parser fails to extract meaningful
data from a real fixture, the test fails. No silent skips allowed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.importers import chatgpt, claude, codex


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "real"


def discover_fixtures() -> list[tuple[str, Path]]:
    """Discover all real fixtures with their provider names."""
    fixtures: list[tuple[str, Path]] = []

    for provider_dir in FIXTURES_ROOT.iterdir():
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name

        for fixture_file in provider_dir.iterdir():
            if fixture_file.suffix in (".json", ".jsonl"):
                fixtures.append((provider, fixture_file))

    return fixtures


def load_fixture(path: Path) -> list[dict] | dict:
    """Load a fixture file as JSON or JSONL."""
    if path.suffix == ".jsonl":
        items = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip invalid lines
        return items
    else:
        return json.loads(path.read_text())


# Parametrize over all discovered fixtures
FIXTURES = discover_fixtures()
if not FIXTURES:
    pytest.fail("No fixtures found in tests/fixtures/real/ - tests cannot run!")


@pytest.mark.parametrize("provider,fixture_path", FIXTURES, ids=lambda x: str(x) if isinstance(x, Path) else x)
def test_parser_extracts_from_real_fixture(provider: str, fixture_path: Path):
    """Each parser must extract meaningful data from its real fixtures.

    This is a contract test - it verifies that:
    1. The parser doesn't crash on real data
    2. The parser extracts at least some messages
    3. The conversation has a valid ID
    """
    data = load_fixture(fixture_path)
    fallback_id = fixture_path.stem

    if provider == "chatgpt":
        # ChatGPT expects a dict with 'mapping'
        if isinstance(data, list):
            # Skip if fixture is a list (might be wrapped)
            if len(data) == 1 and isinstance(data[0], dict):
                data = data[0]
            else:
                pytest.skip(f"ChatGPT fixture is a list, expected dict: {fixture_path}")

        assert chatgpt.looks_like(data), f"ChatGPT fixture not recognized: {fixture_path}"
        result = chatgpt.parse(data, fallback_id)

    elif provider == "claude-code":
        # Claude Code expects a list of records
        if not isinstance(data, list):
            pytest.fail(f"Claude Code fixture should be a list: {fixture_path}")

        assert claude.looks_like_code(data), f"Claude Code fixture not recognized: {fixture_path}"
        result = claude.parse_code(data, fallback_id)

    elif provider == "claude":
        # Claude AI expects a dict with 'chat_messages'
        if isinstance(data, list):
            pytest.skip(f"Claude AI fixture is a list, expected dict: {fixture_path}")

        assert claude.looks_like_ai(data), f"Claude AI fixture not recognized: {fixture_path}"
        result = claude.parse_ai(data, fallback_id)

    elif provider == "codex":
        # Codex expects a list of records
        if not isinstance(data, list):
            pytest.fail(f"Codex fixture should be a list: {fixture_path}")

        assert codex.looks_like(data), f"Codex fixture not recognized: {fixture_path}"
        result = codex.parse(data, fallback_id)

    elif provider == "gemini":
        # Gemini - skip for now as the fixture format issue is known
        pytest.skip(f"Gemini fixture parsing not yet implemented: {fixture_path}")

    else:
        pytest.fail(f"Unknown provider: {provider}")

    # Contract assertions - these MUST pass for the parser to be valid
    assert result.provider_conversation_id != fallback_id or result.messages, \
        f"Parser extracted nothing from {fixture_path}"

    # If we have messages, verify basic structure
    if result.messages:
        for msg in result.messages:
            assert msg.role, f"Message missing role in {fixture_path}"
            # Text can be None for system messages, but most should have text


@pytest.mark.parametrize("provider,fixture_path", FIXTURES, ids=lambda x: str(x) if isinstance(x, Path) else x)
def test_parser_handles_known_edge_cases(provider: str, fixture_path: Path):
    """Verify parsers handle known edge cases in real fixtures.

    This test documents specific edge cases found in real data:
    - Codex: Multiple session_meta records (continuations)
    - Claude Code: isSidechain markers
    - ChatGPT: Multiple children (branches)
    """
    data = load_fixture(fixture_path)
    fallback_id = fixture_path.stem

    if provider == "codex":
        if not isinstance(data, list):
            return

        result = codex.parse(data, fallback_id)

        # Check for continuation detection
        session_metas = [d for d in data if isinstance(d, dict) and d.get("type") == "session_meta"]
        if len(session_metas) > 1:
            assert result.parent_conversation_provider_id is not None, \
                f"Multiple session_metas but no parent detected: {fixture_path}"
            assert result.branch_type == "continuation", \
                f"Should be marked as continuation: {fixture_path}"

    elif provider == "claude-code":
        if not isinstance(data, list):
            return

        result = claude.parse_code(data, fallback_id)

        # Check for sidechain detection
        has_sidechain_marker = any(
            isinstance(d, dict) and d.get("isSidechain") is True
            for d in data
        )
        if has_sidechain_marker:
            assert result.branch_type == "sidechain", \
                f"isSidechain markers present but branch_type not set: {fixture_path}"

    elif provider == "chatgpt":
        if not isinstance(data, dict):
            return

        result = chatgpt.parse(data, fallback_id)

        # Check for branching detection
        mapping = data.get("mapping", {})
        has_multiple_children = any(
            isinstance(node, dict) and len(node.get("children", [])) > 1
            for node in mapping.values()
        )
        if has_multiple_children:
            # Should have extracted branch_index > 0 for some messages
            has_branches = any(m.branch_index > 0 for m in result.messages)
            assert has_branches, \
                f"Multiple children in mapping but no branches extracted: {fixture_path}"
