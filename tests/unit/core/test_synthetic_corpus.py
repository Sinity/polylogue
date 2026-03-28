"""Tests for synthetic corpus generation styles."""

from __future__ import annotations

import json

import pytest

from polylogue.schemas.synthetic import SyntheticCorpus


def _collect_chatgpt_parts(payload: dict) -> list[str]:
    mapping = payload.get("mapping", {})
    parts: list[str] = []
    if not isinstance(mapping, dict):
        return parts
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        raw_parts = content.get("parts")
        if not isinstance(raw_parts, list):
            continue
        for item in raw_parts:
            if isinstance(item, str):
                parts.append(item)
    return parts


def test_showcase_style_generates_human_readable_chatgpt_content() -> None:
    corpus = SyntheticCorpus.for_provider("chatgpt")
    [raw] = corpus.generate(
        count=1,
        messages_per_conversation=range(6, 7),
        seed=7,
        style="showcase",
    )
    payload = json.loads(raw)
    assert isinstance(payload.get("title"), str)
    assert not payload["title"].startswith("synthetic-")

    parts = _collect_chatgpt_parts(payload)
    assert parts
    assert any((" " in text and not text.startswith("synthetic-")) for text in parts)


def test_generate_rejects_unknown_style() -> None:
    corpus = SyntheticCorpus.for_provider("chatgpt")
    with pytest.raises(ValueError, match="Unknown synthetic style"):
        corpus.generate(style="unknown-style")
