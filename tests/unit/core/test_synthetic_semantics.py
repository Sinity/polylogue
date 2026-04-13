"""Tests for semantic value generation, wire formats, corpus generation, and roundtrips.

Verifies that SemanticValueGenerator produces role-appropriate values for
message_body, message_role, message_timestamp, and conversation_title
semantic roles, and that _text_for_role handles all known roles correctly.

Also validates wire format configuration, corpus generation contracts, and
round-trip parsing through provider parsers.

Consolidated from:
- test_synthetic_semantics.py
- test_synthetic_wire_formats.py
- test_synthetic_corpus.py
- test_synthetic_zero_knowledge.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import (
    PROVIDER_WIRE_FORMATS,
    SyntheticCorpus,
    WireFormat,
)
from polylogue.schemas.synthetic.semantic_values import (
    _ROLE_TEXTS,
    SemanticValueGenerator,
    _text_for_role,
)
from polylogue.schemas.synthetic.showcase import _SHOWCASE_THEMES
from polylogue.sources.dispatch import parse_payload

# ---------------------------------------------------------------------------
# _text_for_role
# ---------------------------------------------------------------------------


class TestTextForRole:
    """Verify _text_for_role returns valid text for all known roles."""

    @pytest.mark.parametrize("role", list(_ROLE_TEXTS.keys()))
    def test_known_roles_return_nonempty_text(self, role: str) -> None:
        rng = random.Random(42)
        text = _text_for_role(rng, role)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_unknown_role_falls_back_to_user_texts(self) -> None:
        rng = random.Random(42)
        text = _text_for_role(rng, "nonexistent_role")
        assert text in _ROLE_TEXTS["user"]

    def test_themed_user_turn_returns_theme_content(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "user", turn_index=0, theme=theme)
        assert text in theme.user_turns

    def test_themed_assistant_turn_returns_theme_content(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "assistant", turn_index=1, theme=theme)
        assert text in theme.assistant_turns

    def test_themed_model_role_uses_assistant_turns(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "model", turn_index=1, theme=theme)
        assert text in theme.assistant_turns

    def test_themed_human_role_uses_user_turns(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "human", turn_index=0, theme=theme)
        assert text in theme.user_turns

    def test_turn_index_cycles_through_theme_turns(self) -> None:
        """Theme turns should cycle when turn_index exceeds available turns."""
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        n_user_turns = len(theme.user_turns)
        # Turn index beyond the number of available turns should wrap
        text = _text_for_role(rng, "user", turn_index=n_user_turns * 2, theme=theme)
        assert text in theme.user_turns


# ---------------------------------------------------------------------------
# SemanticValueGenerator — construction and role cycling
# ---------------------------------------------------------------------------


class TestSemanticValueGeneratorBasics:
    def test_default_role_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        assert gen.role_cycle == ["user", "assistant"]
        assert gen.current_role == "user"

    def test_custom_role_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), role_cycle=["human", "model"])
        assert gen.current_role == "human"
        gen.advance_turn()
        assert gen.current_role == "model"
        gen.advance_turn()
        assert gen.current_role == "human"

    def test_turn_index_starts_at_zero(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        assert gen.turn_index == 0

    def test_advance_turn_increments(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        gen.advance_turn()
        assert gen.turn_index == 1
        gen.advance_turn()
        assert gen.turn_index == 2


class TestSyntheticConversationEnvelope:
    def test_chatgpt_tree_generation_produces_clean_conversation_id(self) -> None:
        corpus = SyntheticCorpus.for_provider("chatgpt")
        payload = json.loads(corpus.generate(count=1, seed=42, messages_per_conversation=range(3, 4))[0])

        conversation_id = payload.get("id")
        assert isinstance(conversation_id, str)
        assert conversation_id
        assert " " not in conversation_id

    def test_synthetic_corpus_from_spec_reuses_schema_selection(self) -> None:
        spec = CorpusSpec.for_provider(
            "chatgpt",
            package_version="default",
            count=2,
            messages_min=4,
            messages_max=4,
            seed=7,
        )

        corpus = SyntheticCorpus.from_spec(spec)
        batch = SyntheticCorpus.generate_batch_for_spec(spec)

        assert corpus.provider == "chatgpt"
        assert batch.report.provider == "chatgpt"
        assert batch.report.generated_count == 2

    def test_write_spec_artifacts_returns_written_paths(self, tmp_path) -> None:
        spec = CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=9,
        )

        written = SyntheticCorpus.write_spec_artifacts(spec, tmp_path, prefix="sample")

        assert len(written.files) == 1
        assert written.files[0].exists()
        assert written.batch.report.generated_count == 1


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_role
# ---------------------------------------------------------------------------


class TestSemanticRole:
    def test_generates_role_from_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "message_role"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == "user"

    def test_role_respects_observed_values(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {
            "x-polylogue-semantic-role": "message_role",
            "x-polylogue-values": ["user", "assistant", "system"],
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in {"user", "assistant", "system"}

    def test_role_uses_cycle_when_cycle_value_in_observed(self) -> None:
        gen = SemanticValueGenerator(
            random.Random(0),
            role_cycle=["human", "assistant"],
        )
        schema = {
            "x-polylogue-semantic-role": "message_role",
            "x-polylogue-values": ["human", "assistant"],
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == "human"  # current_role is "human"


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_body
# ---------------------------------------------------------------------------


class TestSemanticBody:
    def test_generates_nonempty_body_text(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "message_body"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert len(value) > 0

    def test_body_text_matches_current_role(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {"x-polylogue-semantic-role": "message_body"}

        # Turn 0 = user
        _, user_text = gen.try_generate(schema)
        assert user_text in _ROLE_TEXTS["user"]

        gen.advance_turn()
        # Turn 1 = assistant
        _, assistant_text = gen.try_generate(schema)
        assert assistant_text in _ROLE_TEXTS["assistant"]

    def test_body_with_theme_uses_themed_content(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        gen = SemanticValueGenerator(random.Random(0), theme=theme)
        schema = {"x-polylogue-semantic-role": "message_body"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in theme.user_turns  # turn 0 = user


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_timestamp
# ---------------------------------------------------------------------------


class TestSemanticTimestamp:
    def test_generates_epoch_by_default(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {"x-polylogue-semantic-role": "message_timestamp"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, float)
        assert value == 1700000000.0  # turn 0

    def test_sequential_timestamps_increase(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {"x-polylogue-semantic-role": "message_timestamp"}

        _, ts0 = gen.try_generate(schema)
        gen.advance_turn()
        _, ts1 = gen.try_generate(schema)
        gen.advance_turn()
        _, ts2 = gen.try_generate(schema)

        assert ts0 < ts1 < ts2

    def test_iso8601_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {
            "x-polylogue-semantic-role": "message_timestamp",
            "x-polylogue-format": "iso8601",
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(value)
        assert parsed.tzinfo is not None

    def test_unix_epoch_str_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {
            "x-polylogue-semantic-role": "message_timestamp",
            "x-polylogue-format": "unix-epoch-str",
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert float(value) == 1700000000.0

    def test_unix_epoch_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {
            "x-polylogue-semantic-role": "message_timestamp",
            "x-polylogue-format": "unix-epoch",
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — conversation_title
# ---------------------------------------------------------------------------


class TestSemanticTitle:
    def test_generates_title_string(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "conversation_title"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert len(value) > 0

    def test_themed_title_uses_theme(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        gen = SemanticValueGenerator(random.Random(0), theme=theme)
        schema = {"x-polylogue-semantic-role": "conversation_title"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == theme.title

    def test_title_with_observed_values(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {
            "x-polylogue-semantic-role": "conversation_title",
            "x-polylogue-values": ["Alpha", "Beta", "Gamma"],
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in {"Alpha", "Beta", "Gamma"}

    def test_title_without_theme_or_values_picks_showcase_theme(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {"x-polylogue-semantic-role": "conversation_title"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        known_titles = {t.title for t in _SHOWCASE_THEMES}
        assert value in known_titles


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — fallback for unknown/unhandled roles
# ---------------------------------------------------------------------------


class TestSemanticFallback:
    def test_no_semantic_role_returns_not_handled(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"type": "string"}
        handled, value = gen.try_generate(schema)
        assert handled is False
        assert value is None

    def test_unknown_semantic_role_returns_not_handled(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "completely_unknown_role"}
        handled, value = gen.try_generate(schema)
        assert handled is False
        assert value is None

    def test_message_container_role_returns_not_handled(self) -> None:
        """message_container is structural and defers to normal generation."""
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "message_container"}
        handled, value = gen.try_generate(schema)
        assert handled is False


# =============================================================================
# Merged from test_synthetic_wire_formats.py (2024-03-15)
# =============================================================================


class TestWireFormatShape:
    """Validate the wire format configuration data structure."""

    EXPECTED_PROVIDERS = {"chatgpt", "claude-code", "claude-ai", "codex", "gemini"}

    def test_all_expected_providers_have_entries(self):
        """Every known provider has a wire format config."""
        assert set(PROVIDER_WIRE_FORMATS.keys()) == self.EXPECTED_PROVIDERS

    @pytest.mark.parametrize("provider", sorted(EXPECTED_PROVIDERS))
    def test_encoding_is_valid(self, provider):
        """Each wire format has a valid encoding value."""
        wf = PROVIDER_WIRE_FORMATS[provider]
        assert isinstance(wf, WireFormat)
        assert wf.encoding in ("json", "jsonl"), f"{provider} has invalid encoding: {wf.encoding}"

    def test_json_providers_have_structure(self):
        """JSON-encoded providers have either tree or messages_path."""
        for name, wf in PROVIDER_WIRE_FORMATS.items():
            if wf.encoding == "json":
                has_structure = wf.tree is not None or wf.messages_path is not None
                assert has_structure, f"{name}: JSON provider needs tree or messages_path"

    def test_chatgpt_has_tree_with_container(self):
        """ChatGPT wire format uses tree structure with container_path."""
        wf = PROVIDER_WIRE_FORMATS["chatgpt"]
        assert wf.tree is not None
        assert wf.tree.container_path == "mapping"
        assert wf.tree.children_field == "children"

    def test_claude_code_has_tree_without_container(self):
        """Claude Code uses tree structure in JSONL (no container_path)."""
        wf = PROVIDER_WIRE_FORMATS["claude-code"]
        assert wf.encoding == "jsonl"
        assert wf.tree is not None
        assert wf.tree.container_path is None
        assert wf.tree.session_field == "sessionId"


class TestCorpusSeedDeterminism:
    """Corpus generation is deterministic with same seed."""

    def test_same_seed_same_output(self):
        """Two generations with the same seed produce identical bytes."""
        available = SyntheticCorpus.available_providers()
        if not available:
            pytest.skip("No schemas available")
        corpus = SyntheticCorpus.for_provider(available[0])
        a = corpus.generate(count=2, seed=99)
        b = corpus.generate(count=2, seed=99)
        assert a == b

    def test_different_seed_different_output(self):
        """Two generations with different seeds produce different bytes."""
        available = SyntheticCorpus.available_providers()
        if not available:
            pytest.skip("No schemas available")
        corpus = SyntheticCorpus.for_provider(available[0])
        a = corpus.generate(count=2, seed=1)
        b = corpus.generate(count=2, seed=2)
        assert a != b


class TestCorpusParseRoundtrip:
    """Generated corpus parses successfully through source iterators."""

    @pytest.mark.parametrize("provider", sorted(PROVIDER_WIRE_FORMATS.keys()))
    def test_generated_data_parses(self, provider, synthetic_source):
        """Synthetic data for each provider round-trips through parser."""
        from polylogue.sources import iter_source_conversations

        try:
            source = synthetic_source(provider, count=2, seed=42)
        except FileNotFoundError:
            pytest.skip(f"No schema available for {provider}")

        convos = list(iter_source_conversations(source))
        assert len(convos) > 0, f"No conversations parsed for {provider}"
        for conv in convos:
            assert len(conv.messages) > 0, f"Empty conversation for {provider}"


# =============================================================================
# Merged from test_synthetic_corpus.py (2024-03-15)
# =============================================================================


class TestSeedDeterminism:
    """Corpus generation is deterministic given the same seed."""

    @pytest.mark.parametrize("provider", sorted(SyntheticCorpus.available_providers() or ["chatgpt"]))
    def test_same_seed_produces_identical_output(self, provider):
        """Two generate() calls with same seed → identical byte output."""
        try:
            corpus = SyntheticCorpus.for_provider(provider)
        except FileNotFoundError:
            pytest.skip(f"No schema for {provider}")

        a = corpus.generate(count=3, seed=42)
        b = corpus.generate(count=3, seed=42)
        assert a == b, f"{provider}: same seed produced different output"

    def test_different_seeds_produce_different_output(self):
        """Different seeds produce different output."""
        available = SyntheticCorpus.available_providers()
        if not available:
            pytest.skip("No schemas available")

        corpus = SyntheticCorpus.for_provider(available[0])
        a = corpus.generate(count=2, seed=1)
        b = corpus.generate(count=2, seed=2)
        assert a != b


class TestMessageCountContract:
    """Generated conversations respect the message count range."""

    @pytest.mark.parametrize("provider", sorted(SyntheticCorpus.available_providers() or ["chatgpt"]))
    def test_generate_count_matches_requested(self, provider):
        """generate(count=N) returns exactly N items."""
        try:
            corpus = SyntheticCorpus.for_provider(provider)
        except FileNotFoundError:
            pytest.skip(f"No schema for {provider}")

        for count in (1, 3, 5):
            items = corpus.generate(count=count, seed=0)
            assert len(items) == count, f"{provider}: requested {count}, got {len(items)}"

    def test_generate_zero_returns_empty(self):
        """generate(count=0) returns an empty list."""
        available = SyntheticCorpus.available_providers()
        if not available:
            pytest.skip("No schemas available")

        corpus = SyntheticCorpus.for_provider(available[0])
        items = corpus.generate(count=0, seed=0)
        assert items == []

    def test_generate_batch_reports_provider_and_count(self):
        """generate_batch() exposes a typed report aligned with raw output."""
        available = SyntheticCorpus.available_providers()
        if not available:
            pytest.skip("No schemas available")

        corpus = SyntheticCorpus.for_provider(available[0])
        batch = corpus.generate_batch(count=2, seed=7, messages_per_conversation=range(4, 5))
        assert len(batch.artifacts) == 2
        assert batch.report.generated_count == 2
        assert batch.report.provider == corpus.provider
        assert batch.report.package_version == corpus.package_version
        assert batch.raw_items == [artifact.raw_bytes for artifact in batch.artifacts]


class TestParseRoundtrip:
    """Synthetic data round-trips through the actual provider parsers."""

    @pytest.mark.parametrize("provider", sorted(SyntheticCorpus.available_providers() or ["chatgpt"]))
    def test_synthetic_parses_to_conversations(self, provider, synthetic_source):
        """Synthetic corpus for each provider parses into valid conversations."""
        from polylogue.sources import iter_source_conversations

        try:
            source = synthetic_source(provider, count=2, seed=42)
        except FileNotFoundError:
            pytest.skip(f"No schema for {provider}")

        convos = list(iter_source_conversations(source))
        assert len(convos) > 0, f"No conversations parsed for {provider}"

        for conv in convos:
            assert len(conv.messages) > 0, f"Empty conversation for {provider}"
            # At least one message should have non-empty text
            assert any(m.text for m in conv.messages), f"No message text for {provider}"


# =============================================================================
# Merged from test_synthetic_zero_knowledge.py (2024-03-15)
# =============================================================================


# Schema providers and runtime parsers share the same canonical names.
_SCHEMA_TO_RUNTIME_PROVIDER: dict[str, str] = {
    "chatgpt": "chatgpt",
    "claude-ai": "claude-ai",
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
                f"Provider {provider}: parse_payload returned no conversations for synthetic data (index {i})"
            )
            conv = conversations[0]
            assert len(conv.messages) > 0, f"Provider {provider}: parsed conversation has no messages (index {i})"

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
        assert len(messages_with_text) > 0, f"Provider {provider}: no messages have text content"


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

    def test_for_provider_accepts_explicit_version_and_element(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.synthetic.core as synthetic_core

        fake_registry = MagicMock()
        fake_package = MagicMock(
            version="v2",
            default_element_kind="conversation_record_stream",
        )
        fake_package.element.side_effect = lambda element_kind: (
            {"element_kind": element_kind} if element_kind == "conversation_record_stream" else None
        )
        fake_schema = {"type": "object"}
        fake_registry.get_package.return_value = fake_package
        fake_registry.get_element_schema.return_value = fake_schema
        monkeypatch.setattr(synthetic_core, "SchemaRegistry", lambda: fake_registry)

        corpus = SyntheticCorpus.for_provider(
            "chatgpt",
            version="v2",
            element_kind="conversation_record_stream",
        )

        assert isinstance(corpus, SyntheticCorpus)
        fake_registry.get_package.assert_called_once_with("chatgpt", version="v2")
        fake_registry.get_element_schema.assert_called_once_with(
            "chatgpt",
            version="v2",
            element_kind="conversation_record_stream",
        )
        assert corpus.schema == fake_schema

    def test_for_provider_rejects_unknown_element_when_package_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.synthetic.core as synthetic_core

        fake_registry = MagicMock()
        fake_package = MagicMock(
            version="v2",
            default_element_kind="conversation_record_stream",
        )
        fake_package.element.return_value = None
        fake_registry.get_package.return_value = fake_package
        monkeypatch.setattr(synthetic_core, "SchemaRegistry", lambda: fake_registry)

        with pytest.raises(ValueError):
            SyntheticCorpus.for_provider("chatgpt", version="v2", element_kind="unknown_element")

        fake_registry.get_package.assert_called_once_with("chatgpt", version="v2")

    def test_for_provider_rejects_element_without_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.synthetic.core as synthetic_core

        fake_registry = MagicMock()
        fake_registry.get_package.return_value = None
        fake_schema = {"type": "object"}
        fake_registry.get_schema.return_value = fake_schema
        monkeypatch.setattr(synthetic_core, "SchemaRegistry", lambda: fake_registry)

        with pytest.raises(ValueError):
            SyntheticCorpus.for_provider("chatgpt", element_kind="conversation_record_stream")


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
