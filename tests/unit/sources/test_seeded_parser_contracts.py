"""Seeded synthetic parser regressions.

The broad parse laws live in ``test_source_laws.py``. This file keeps one
seeded end-to-end extraction contract and the provider-specific edge cases that
are easiest to express against the seeded corpus.
"""

from __future__ import annotations

import json

import pytest

from polylogue.sources.parsers import chatgpt, claude, codex, drive
from polylogue.storage.store import RawConversationRecord


def parse_raw_content(sample: RawConversationRecord) -> tuple[list | dict, str]:
    content = sample.raw_content.decode("utf-8")
    provider = sample.provider_name
    if provider in ("claude-code", "codex", "gemini"):
        items = []
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items, provider
    return json.loads(content), provider


def _parse_sample(sample: RawConversationRecord):
    data, provider = parse_raw_content(sample)
    fallback_id = sample.raw_id[:16]

    if provider == "chatgpt":
        if not isinstance(data, dict) or not chatgpt.looks_like(data):
            return None
        return chatgpt.parse(data, fallback_id)
    if provider == "claude-code":
        if not isinstance(data, list) or not claude.looks_like_code(data):
            return None
        return claude.parse_code(data, fallback_id)
    if provider in ("claude", "claude-ai"):
        if not isinstance(data, dict) or not claude.looks_like_ai(data):
            return None
        return claude.parse_ai(data, fallback_id)
    if provider == "codex":
        if not isinstance(data, list) or not codex.looks_like(data):
            return None
        return codex.parse(data, fallback_id)
    if provider == "gemini":
        if not isinstance(data, list) or not data:
            return None
        item = data[0]
        if not isinstance(item, dict) or "chunks" not in item:
            return None
        return drive.parse_chunked_prompt("gemini", item, fallback_id)
    return None


def test_seeded_samples_parse_without_crashing(raw_synthetic_samples: list[RawConversationRecord]) -> None:
    failures: list[tuple[str, str, str]] = []
    recognized = 0

    for sample in raw_synthetic_samples:
        try:
            result = _parse_sample(sample)
            if result is None:
                continue
            recognized += 1
            if not result.messages and result.provider_conversation_id == sample.raw_id[:16] and sample.provider_name != "claude-code":
                failures.append((sample.raw_id[:16], sample.provider_name, "No messages extracted"))
        except Exception as exc:  # pragma: no cover - failure path is the assertion below
            failures.append((sample.raw_id[:16], sample.provider_name, str(exc)[:80]))

    assert recognized > 0
    if failures:
        summary = "\n".join(f"  {provider}:{raw_id}: {error}" for raw_id, provider, error in failures[:10])
        pytest.fail(f"{len(failures)}/{len(raw_synthetic_samples)} seeded samples failed:\n{summary}")


@pytest.mark.parametrize(
    ("provider", "predicate", "expected_desc"),
    [
        (
            "codex",
            lambda data, result: sum(1 for item in data if isinstance(item, dict) and item.get("type") == "session_meta") > 1
            and result.parent_conversation_provider_id is not None
            and result.branch_type == "continuation",
            "continuation",
        ),
        (
            "claude-code",
            lambda data, result: any(isinstance(item, dict) and item.get("isSidechain") is True for item in data)
            and result.branch_type == "sidechain",
            "sidechain",
        ),
        (
            "chatgpt",
            lambda data, result: any(
                isinstance(node, dict)
                and sum(
                    1
                    for child_id in node.get("children", [])
                    if isinstance(data.get("mapping", {}).get(child_id, {}), dict)
                    and isinstance(data.get("mapping", {}).get(child_id, {}).get("message"), dict)
                    and isinstance(data.get("mapping", {}).get(child_id, {}).get("message", {}).get("content"), dict)
                    and any(
                        isinstance(part, str) and part.strip()
                        for part in data.get("mapping", {}).get(child_id, {}).get("message", {}).get("content", {}).get("parts", [])
                    )
                ) > 1
                for node in data.get("mapping", {}).values()
            ) and any(message.branch_index > 0 for message in result.messages),
            "branching",
        ),
    ],
    ids=["codex", "claude-code", "chatgpt"],
)
def test_seeded_provider_edge_cases_hold(
    raw_synthetic_samples: list[RawConversationRecord],
    provider: str,
    predicate,
    expected_desc: str,
) -> None:
    provider_samples = [sample for sample in raw_synthetic_samples if sample.provider_name == provider]
    if not provider_samples:
        pytest.skip(f"No {provider} samples")

    checked = 0
    failures: list[str] = []
    for sample in provider_samples:
        parsed = _parse_sample(sample)
        if parsed is None:
            continue
        data, _ = parse_raw_content(sample)
        if predicate(data, parsed):
            checked += 1
        elif (
            (provider == "codex" and isinstance(data, list) and sum(1 for item in data if isinstance(item, dict) and item.get("type") == "session_meta") > 1)
            or (provider == "claude-code" and isinstance(data, list) and any(isinstance(item, dict) and item.get("isSidechain") is True for item in data))
        ):
            failures.append(sample.raw_id[:16])
        elif provider == "chatgpt" and isinstance(data, dict):
            mapping = data.get("mapping", {})
            has_multiple_children = any(
                isinstance(node, dict)
                and sum(
                    1
                    for child_id in node.get("children", [])
                    if isinstance(mapping.get(child_id, {}), dict)
                    and isinstance(mapping.get(child_id, {}).get("message"), dict)
                    and isinstance(mapping.get(child_id, {}).get("message", {}).get("content"), dict)
                    and any(
                        isinstance(part, str) and part.strip()
                        for part in mapping.get(child_id, {}).get("message", {}).get("content", {}).get("parts", [])
                    )
                ) > 1
                for node in mapping.values()
            )
            if has_multiple_children:
                failures.append(sample.raw_id[:16])

    assert checked or not failures
    if failures:
        pytest.fail(f"{provider} seeded {expected_desc} failures: {failures[:5]}")
