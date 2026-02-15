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

    def test_all_parsers_extract(self, raw_synthetic_samples: list[RawConversationRecord]) -> None:
        """Every raw conversation can be parsed without crashing."""

        failures = []
        unrecognized = 0  # Records that don't match expected provider format

        for sample in raw_synthetic_samples:
            try:
                data, provider = parse_raw_content(sample)
                fallback_id = sample.raw_id[:16]

                if provider == "chatgpt":
                    if not isinstance(data, dict):
                        continue  # Skip malformed
                    if not chatgpt.looks_like(data):
                        # ChatGPT exports include conversation stubs (deleted/empty)
                        # that lack 'mapping' — skip, don't fail.
                        unrecognized += 1
                        continue
                    result = chatgpt.parse(data, fallback_id)

                elif provider == "claude-code":
                    if not isinstance(data, list):
                        continue
                    if not claude.looks_like_code(data):
                        unrecognized += 1
                        continue
                    result = claude.parse_code(data, fallback_id)

                elif provider in ("claude", "claude-ai"):
                    if not isinstance(data, dict):
                        continue
                    if not claude.looks_like_ai(data):
                        unrecognized += 1
                        continue
                    result = claude.parse_ai(data, fallback_id)

                elif provider == "codex":
                    if not isinstance(data, list):
                        continue
                    if not codex.looks_like(data):
                        unrecognized += 1
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
                if not result.messages and result.provider_conversation_id == fallback_id and provider != "claude-code":
                    failures.append((sample.raw_id[:16], provider, "No messages extracted"))

            except Exception as e:
                failures.append((sample.raw_id[:16], sample.provider_name, str(e)[:80]))

        if unrecognized:
            import warnings
            warnings.warn(
                f"{unrecognized}/{len(raw_synthetic_samples)} records not recognized by looks_like "
                f"(conversation stubs or metadata — expected for ChatGPT exports)",
                stacklevel=2,
            )

        if failures:
            msg = f"{len(failures)}/{len(raw_synthetic_samples)} failed:\n"
            for raw_id, provider, error in failures[:10]:
                msg += f"  {provider}:{raw_id}: {error}\n"
            pytest.fail(msg)


class TestEdgeCaseHandling:
    """Verify parsers handle known edge cases in real data."""

    def test_codex_continuations(self, raw_synthetic_samples: list[RawConversationRecord]) -> None:
        """Codex conversations with multiple session_meta are continuations."""

        codex_samples = [s for s in raw_synthetic_samples if s.provider_name == "codex"]
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

    def test_claude_code_sidechains(self, raw_synthetic_samples: list[RawConversationRecord]) -> None:
        """Claude Code conversations with isSidechain markers are detected."""

        cc_samples = [s for s in raw_synthetic_samples if s.provider_name == "claude-code"]
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

    def test_chatgpt_branching(self, raw_synthetic_samples: list[RawConversationRecord]) -> None:
        """ChatGPT conversations with multiple children have branch_index > 0."""

        chatgpt_samples = [s for s in raw_synthetic_samples if s.provider_name == "chatgpt"]
        if not chatgpt_samples:
            pytest.skip("No ChatGPT samples")

        failures = []
        for sample in chatgpt_samples:
            try:
                data, _ = parse_raw_content(sample)
                if not isinstance(data, dict):
                    continue

                mapping = data.get("mapping", {})

                def _has_parseable_content(node_id: str, _mapping: dict = mapping) -> bool:
                    """Check if a mapping node has a message with non-empty text."""
                    n = _mapping.get(node_id, {})
                    if not isinstance(n, dict):
                        return False
                    m = n.get("message")
                    if not isinstance(m, dict):
                        return False
                    c = m.get("content")
                    if not isinstance(c, dict):
                        return False
                    parts = c.get("parts") or []
                    return any(isinstance(p, str) and p.strip() for p in parts)

                # Only count as branching when 2+ children have parseable content
                # (not empty-text system stubs alongside real responses)
                has_multiple_children = any(
                    isinstance(node, dict)
                    and sum(1 for cid in node.get("children", []) if _has_parseable_content(cid)) > 1
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
