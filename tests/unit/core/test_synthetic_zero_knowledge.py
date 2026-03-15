"""Zero-knowledge parse-roundtrip tests for the synthetic corpus system.

For every available provider, generates synthetic wire-format data and
verifies it can be parsed by the corresponding provider parser, producing
valid conversations with expected structure.
"""

from __future__ import annotations

import json

import pytest

from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources.source import parse_payload


# Schema providers use different canonical names than runtime parsers.
# The mapping is: claude-ai (schema) → claude (runtime/parser).
_SCHEMA_TO_RUNTIME_PROVIDER: dict[str, str] = {
    "chatgpt": "chatgpt",
    "claude-ai": "claude",
    "claude-code": "claude-code",
    "codex": "codex",
    "gemini": "gemini",
}


def _deserialize_for_parser(provider: str, raw: bytes) -> object:
    """Deserialize raw bytes into the format expected by parse_payload.

    JSON providers produce a single JSON object/dict.
    JSONL providers produce a list of parsed line dicts.
    """
    if provider in ("claude-code", "codex"):
        # JSONL: split lines, parse each as JSON
        lines = raw.decode("utf-8").strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]
    else:
        return json.loads(raw)


def _available_providers() -> list[str]:
    return SyntheticCorpus.available_providers()


# ---------------------------------------------------------------------------
# Core roundtrip: generate → parse for every provider
# ---------------------------------------------------------------------------


class TestSyntheticRoundtrip:
    """For every provider with a schema and wire format, synthetic data must
    be parseable by the real provider parser."""

    @pytest.mark.parametrize("provider", _available_providers())
    def test_roundtrip_produces_messages(self, provider: str) -> None:
        corpus = SyntheticCorpus.for_provider(provider)
        results = corpus.generate(count=3, seed=42, messages_per_conversation=range(4, 8))
        assert len(results) == 3

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)

        for i, raw in enumerate(results):
            payload = _deserialize_for_parser(provider, raw)
            conversations = parse_payload(runtime_provider, payload, f"synth-{provider}-{i}")
            assert len(conversations) >= 1, (
                f"Provider {provider}: parse_payload returned no conversations "
                f"for synthetic data (index {i})"
            )
            conv = conversations[0]
            assert len(conv.messages) > 0, (
                f"Provider {provider}: parsed conversation has no messages (index {i})"
            )

    @pytest.mark.parametrize("provider", _available_providers())
    def test_parsed_messages_have_roles(self, provider: str) -> None:
        corpus = SyntheticCorpus.for_provider(provider)
        [raw] = corpus.generate(count=1, seed=7, messages_per_conversation=range(5, 6))

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)
        payload = _deserialize_for_parser(provider, raw)
        conversations = parse_payload(runtime_provider, payload, f"synth-{provider}")
        assert conversations

        for msg in conversations[0].messages:
            assert msg.role is not None
            assert str(msg.role) != ""

    @pytest.mark.parametrize("provider", _available_providers())
    def test_parsed_messages_have_text_content(self, provider: str) -> None:
        corpus = SyntheticCorpus.for_provider(provider)
        [raw] = corpus.generate(count=1, seed=99, messages_per_conversation=range(4, 5))

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)
        payload = _deserialize_for_parser(provider, raw)
        conversations = parse_payload(runtime_provider, payload, f"synth-{provider}")
        assert conversations

        messages_with_text = [m for m in conversations[0].messages if m.text]
        assert len(messages_with_text) > 0, (
            f"Provider {provider}: no messages have text content"
        )


# ---------------------------------------------------------------------------
# Determinism: same seed → same output
# ---------------------------------------------------------------------------


class TestSyntheticDeterminism:
    @pytest.mark.parametrize("provider", _available_providers())
    def test_same_seed_same_output(self, provider: str) -> None:
        corpus = SyntheticCorpus.for_provider(provider)
        result_a = corpus.generate(count=3, seed=123)
        result_b = corpus.generate(count=3, seed=123)
        assert result_a == result_b

    @pytest.mark.parametrize("provider", _available_providers())
    def test_different_seeds_different_output(self, provider: str) -> None:
        corpus = SyntheticCorpus.for_provider(provider)
        result_a = corpus.generate(count=2, seed=1)
        result_b = corpus.generate(count=2, seed=2)
        # At least one conversation should differ
        assert result_a != result_b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSyntheticEdgeCases:
    @pytest.mark.parametrize("provider", _available_providers())
    def test_count_zero_produces_empty_list(self, provider: str) -> None:
        corpus = SyntheticCorpus.for_provider(provider)
        result = corpus.generate(count=0, seed=0)
        assert result == []

    @pytest.mark.parametrize("provider", _available_providers())
    def test_single_message_conversation(self, provider: str) -> None:
        """A conversation with exactly 1 message should still parse."""
        corpus = SyntheticCorpus.for_provider(provider)
        [raw] = corpus.generate(count=1, seed=55, messages_per_conversation=range(1, 2))

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)
        payload = _deserialize_for_parser(provider, raw)
        conversations = parse_payload(runtime_provider, payload, f"synth-{provider}")
        assert len(conversations) >= 1
        assert len(conversations[0].messages) >= 1


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------


class TestProviderAvailability:
    def test_available_providers_is_nonempty(self) -> None:
        providers = SyntheticCorpus.available_providers()
        assert len(providers) > 0

    def test_available_providers_are_known(self) -> None:
        known = {"chatgpt", "claude-ai", "claude-code", "codex", "gemini"}
        for provider in SyntheticCorpus.available_providers():
            assert provider in known, f"Unknown provider: {provider}"

    def test_for_provider_raises_on_unknown(self) -> None:
        with pytest.raises((FileNotFoundError, ValueError)):
            SyntheticCorpus.for_provider("nonexistent-provider-xyz")


# ---------------------------------------------------------------------------
# Showcase style roundtrip
# ---------------------------------------------------------------------------


class TestShowcaseStyleRoundtrip:
    @pytest.mark.parametrize("provider", _available_providers())
    def test_showcase_style_roundtrips(self, provider: str) -> None:
        """Showcase-style synthetic data should also parse correctly."""
        corpus = SyntheticCorpus.for_provider(provider)
        results = corpus.generate(
            count=2,
            seed=42,
            messages_per_conversation=range(4, 8),
            style="showcase",
        )
        assert len(results) == 2

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)

        for i, raw in enumerate(results):
            payload = _deserialize_for_parser(provider, raw)
            conversations = parse_payload(runtime_provider, payload, f"showcase-{provider}-{i}")
            assert len(conversations) >= 1
            assert len(conversations[0].messages) > 0
