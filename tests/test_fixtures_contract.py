"""Contract tests: verify all parsers work on all real data.

These tests use the raw_conversations table as the source of truth.
Run `polylogue run --stage acquire` to populate it with real exports.

Control sample count via POLYLOGUE_TEST_SAMPLES:
- POLYLOGUE_TEST_SAMPLES=100 (default) - Fast CI
- POLYLOGUE_TEST_SAMPLES=0 - Exhaustive mode, ALL samples
"""

from __future__ import annotations

import json

import pytest

from polylogue.sources.parsers import chatgpt, claude, codex, drive
from polylogue.storage.store import RawConversationRecord


def parse_raw_content(sample: RawConversationRecord) -> tuple[list | dict, str]:
    """Parse raw_content bytes into JSON data."""
    content = sample.raw_content.decode("utf-8")
    provider = sample.provider_name

    # JSONL providers: parse all lines into a list
    if provider in ("claude-code", "codex", "gemini"):
        items = []
        for line in content.strip().split("\n"):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items, provider

    # JSON providers: parse as single object
    return json.loads(content), provider


class TestParserExtractsFromRealData:
    """Each parser must extract meaningful data from real exports."""

    def test_all_parsers_extract(self, raw_db_samples: list[RawConversationRecord]) -> None:
        """Every raw conversation can be parsed without crashing."""
        if not raw_db_samples:
            pytest.skip("No raw conversations (run: polylogue run --stage acquire)")

        failures = []

        for sample in raw_db_samples:
            try:
                data, provider = parse_raw_content(sample)
                fallback_id = sample.raw_id[:16]

                if provider == "chatgpt":
                    if not isinstance(data, dict):
                        continue  # Skip malformed
                    if not chatgpt.looks_like(data):
                        failures.append((sample.raw_id[:16], provider, "Not recognized as ChatGPT"))
                        continue
                    result = chatgpt.parse(data, fallback_id)

                elif provider == "claude-code":
                    if not isinstance(data, list):
                        continue
                    if not claude.looks_like_code(data):
                        failures.append((sample.raw_id[:16], provider, "Not recognized as Claude Code"))
                        continue
                    result = claude.parse_code(data, fallback_id)

                elif provider in ("claude", "claude-ai"):
                    if not isinstance(data, dict):
                        continue
                    if not claude.looks_like_ai(data):
                        failures.append((sample.raw_id[:16], provider, "Not recognized as Claude AI"))
                        continue
                    result = claude.parse_ai(data, fallback_id)

                elif provider == "codex":
                    if not isinstance(data, list):
                        continue
                    if not codex.looks_like(data):
                        failures.append((sample.raw_id[:16], provider, "Not recognized as Codex"))
                        continue
                    result = codex.parse(data, fallback_id)

                elif provider == "gemini":
                    if not isinstance(data, list) or not data:
                        continue
                    # Gemini JSONL: each line is a conversation dict with 'chunks'
                    item = data[0]
                    if not isinstance(item, dict) or "chunks" not in item:
                        continue
                    result = drive.parse_chunked_prompt("gemini", item, fallback_id)

                else:
                    continue  # Unknown provider

                # Verify extraction produced something
                # Claude Code sessions may contain only metadata records
                # (e.g. file-history-snapshot) with no chat messages — that's valid.
                if not result.messages and result.provider_conversation_id == fallback_id:
                    if provider != "claude-code":
                        failures.append((sample.raw_id[:16], provider, "No messages extracted"))

            except Exception as e:
                failures.append((sample.raw_id[:16], sample.provider_name, str(e)[:80]))

        if failures:
            msg = f"{len(failures)}/{len(raw_db_samples)} failed:\n"
            for raw_id, provider, error in failures[:10]:
                msg += f"  {provider}:{raw_id}: {error}\n"
            pytest.fail(msg)


class TestEdgeCaseHandling:
    """Verify parsers handle known edge cases in real data."""

    def test_codex_continuations(self, raw_db_samples: list[RawConversationRecord]) -> None:
        """Codex conversations with multiple session_meta are continuations."""
        if not raw_db_samples:
            pytest.skip("No raw conversations")

        codex_samples = [s for s in raw_db_samples if s.provider_name == "codex"]
        if not codex_samples:
            pytest.skip("No Codex samples")

        failures = []
        for sample in codex_samples:
            try:
                data, _ = parse_raw_content(sample)
                if not isinstance(data, list):
                    continue

                session_metas = [d for d in data if isinstance(d, dict) and d.get("type") == "session_meta"]
                if len(session_metas) > 1:
                    result = codex.parse(data, sample.raw_id[:16])
                    if result.parent_conversation_provider_id is None:
                        failures.append((sample.raw_id[:16], "No parent detected for continuation"))
                    elif result.branch_type != "continuation":
                        failures.append((sample.raw_id[:16], f"Wrong branch_type: {result.branch_type}"))
            except Exception as e:
                failures.append((sample.raw_id[:16], str(e)[:60]))

        if failures:
            pytest.fail(f"{len(failures)} continuation detection failures: {failures[:5]}")

    def test_claude_code_sidechains(self, raw_db_samples: list[RawConversationRecord]) -> None:
        """Claude Code conversations with isSidechain markers are detected."""
        if not raw_db_samples:
            pytest.skip("No raw conversations")

        cc_samples = [s for s in raw_db_samples if s.provider_name == "claude-code"]
        if not cc_samples:
            pytest.skip("No Claude Code samples")

        failures = []
        for sample in cc_samples:
            try:
                data, _ = parse_raw_content(sample)
                if not isinstance(data, list):
                    continue

                has_sidechain = any(
                    isinstance(d, dict) and d.get("isSidechain") is True
                    for d in data
                )
                if has_sidechain:
                    result = claude.parse_code(data, sample.raw_id[:16])
                    if result.branch_type != "sidechain":
                        failures.append((sample.raw_id[:16], f"Expected sidechain, got: {result.branch_type}"))
            except Exception as e:
                failures.append((sample.raw_id[:16], str(e)[:60]))

        if failures:
            pytest.fail(f"{len(failures)} sidechain detection failures: {failures[:5]}")

    def test_chatgpt_branching(self, raw_db_samples: list[RawConversationRecord]) -> None:
        """ChatGPT conversations with multiple children have branch_index > 0."""
        if not raw_db_samples:
            pytest.skip("No raw conversations")

        chatgpt_samples = [s for s in raw_db_samples if s.provider_name == "chatgpt"]
        if not chatgpt_samples:
            pytest.skip("No ChatGPT samples")

        failures = []
        for sample in chatgpt_samples:
            try:
                data, _ = parse_raw_content(sample)
                if not isinstance(data, dict):
                    continue

                mapping = data.get("mapping", {})
                has_multiple_children = any(
                    isinstance(node, dict) and len(node.get("children", [])) > 1
                    for node in mapping.values()
                )

                if has_multiple_children:
                    result = chatgpt.parse(data, sample.raw_id[:16])
                    has_branches = any(m.branch_index > 0 for m in result.messages)
                    if not has_branches:
                        failures.append((sample.raw_id[:16], "Multiple children but no branch_index > 0"))
            except Exception as e:
                failures.append((sample.raw_id[:16], str(e)[:60]))

        if failures:
            pytest.fail(f"{len(failures)} branching detection failures: {failures[:5]}")
