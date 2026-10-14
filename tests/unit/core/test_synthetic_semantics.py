"""Tests for semantic value generation, wire formats, corpus generation, and roundtrips.

Verifies that SemanticValueGenerator produces role-appropriate values for
message_body, message_role, message_timestamp, and session_title
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

import hashlib
import json
import random
import tracemalloc
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pytest

from polylogue.config import Source
from polylogue.core.json import JSONValue
from polylogue.scenarios import CorpusProfile, CorpusSpec
from polylogue.schemas.synthetic import (
    PROVIDER_WIRE_FORMATS,
    SyntheticCorpus,
    WireFormat,
)
from polylogue.schemas.synthetic.build_wire_formats import (
    _claude_code_content_fallback,
    _codex_content_fallback,
)
from polylogue.schemas.synthetic.demo_themes import _DEMO_THEMES
from polylogue.schemas.synthetic.models import SchemaRecord, SchemaValue, SyntheticSchemaSelection
from polylogue.schemas.synthetic.semantic_values import (
    _ROLE_TEXTS,
    SemanticValueGenerator,
    _text_for_role,
)
from polylogue.sources.dispatch import parse_payload
from tests.infra.qualified_schema_registry import QUALIFIED_SCHEMA_PROVIDERS
from tests.infra.schema_access import fail_missing_schema


def _schema(value: object) -> SchemaRecord:
    assert isinstance(value, dict)
    return {str(key): _schema_value(item) for key, item in value.items()}


def _schema_value(value: object) -> SchemaValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_schema_value(item) for item in value]
    assert isinstance(value, dict)
    return {str(key): _schema_value(item) for key, item in value.items()}


def _float_value(value: object) -> float:
    assert isinstance(value, float)
    return value


class SyntheticSourceFactory(Protocol):
    def __call__(
        self,
        provider: str,
        count: int = 1,
        messages_per_session: range = range(4, 12),
        seed: int = 42,
    ) -> Source: ...


class _FakePackage:
    def __init__(
        self,
        *,
        version: str,
        default_element_kind: str | None,
        available_elements: set[str],
    ) -> None:
        self.version = version
        self.default_element_kind = default_element_kind
        self._available_elements = available_elements

    def element(self, element_kind: str | None) -> object | None:
        if element_kind in self._available_elements:
            return {"element_kind": element_kind}
        return None


class _FakeRegistry:
    def __init__(
        self,
        *,
        package: _FakePackage | None,
        element_schema: SchemaRecord | None = None,
        schema: SchemaRecord | None = None,
    ) -> None:
        self._package = package
        self._element_schema = element_schema
        self._schema = schema
        self.package_calls: list[tuple[str, str]] = []
        self.element_schema_calls: list[tuple[str, str, str | None]] = []
        self.schema_calls: list[tuple[str, str]] = []

    def get_package(self, provider: str, version: str = "default") -> _FakePackage | None:
        self.package_calls.append((provider, version))
        return self._package

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> SchemaRecord | None:
        self.element_schema_calls.append((provider, version, element_kind))
        return self._element_schema

    def get_schema(self, provider: str, version: str = "default") -> SchemaRecord | None:
        self.schema_calls.append((provider, version))
        return self._schema

    def get_workload_profile(self, provider: str, version: str = "default") -> SchemaRecord | None:
        return None

    def list_providers(self) -> list[str]:
        return ["chatgpt"]


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
        theme = _DEMO_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "user", turn_index=0, theme=theme)
        assert text in theme.user_turns

    def test_themed_assistant_turn_returns_theme_content(self) -> None:
        theme = _DEMO_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "assistant", turn_index=1, theme=theme)
        assert text in theme.assistant_turns

    def test_themed_model_role_uses_assistant_turns(self) -> None:
        theme = _DEMO_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "model", turn_index=1, theme=theme)
        assert text in theme.assistant_turns

    def test_themed_human_role_uses_user_turns(self) -> None:
        theme = _DEMO_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "human", turn_index=0, theme=theme)
        assert text in theme.user_turns

    def test_turn_index_cycles_through_theme_turns(self) -> None:
        """Theme turns should cycle when turn_index exceeds available turns."""
        theme = _DEMO_THEMES[0]
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


class TestSyntheticSessionEnvelope:
    def test_chatgpt_tree_generation_produces_clean_session_id(
        self, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus("chatgpt")
        payload = json.loads(corpus.generate(count=1, seed=42, messages_per_session=range(3, 4))[0])

        session_id = payload.get("id")
        assert isinstance(session_id, str)
        assert session_id
        assert " " not in session_id

    def test_synthetic_corpus_from_spec_reuses_schema_selection(self, qualified_provider_defaults: None) -> None:
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

    def test_generation_plants_independent_wire_facts_before_ingest(self, qualified_provider_defaults: None) -> None:
        native_id = "c03-planted-target"
        spec = CorpusSpec.for_provider(
            "codex",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=71,
            style="tool-heavy",
            session_native_ids=(native_id,),
        )

        batch = SyntheticCorpus.generate_batch_for_spec(spec)
        facts = batch.artifacts[0].facts

        assert facts.provider == "codex"
        assert facts.native_session_id == native_id
        assert facts.expected_session_id == f"codex-session:{native_id}"
        assert facts.message_count == 4
        assert facts.tool_use_ids
        assert set(facts.paired_tool_ids) <= set(facts.tool_result_ids)
        assert facts.raw_sha256

    def test_write_spec_artifacts_returns_written_paths(
        self, tmp_path: Path, qualified_provider_defaults: None
    ) -> None:
        spec = CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=9,
        )

        written = SyntheticCorpus.write_spec_artifacts(spec, tmp_path, prefix="sample")

        assert len(written.files) == 1
        assert written.session_files[0].exists()
        assert written.report.generated_count == 1

    def test_write_specs_artifacts_avoids_same_source_name_collisions(self, tmp_path: Path) -> None:
        specs = (
            CorpusSpec(
                provider="chatgpt",
                package_version="v1",
                count=1,
                messages_min=4,
                messages_max=4,
                seed=9,
                profile=CorpusProfile(family_ids=("cluster-a",)),
            ),
            CorpusSpec(
                provider="chatgpt",
                package_version="v1",
                count=1,
                messages_min=4,
                messages_max=4,
                seed=10,
                profile=CorpusProfile(family_ids=("cluster-b",)),
            ),
        )

        written_batches = SyntheticCorpus.write_specs_artifacts(specs, tmp_path, prefix="sample")

        assert len(written_batches) == 2
        assert {path.name for batch in written_batches for path in batch.files} == {
            "sample-v1-cluster-a-00.json",
            "sample-v1-cluster-b-00.json",
        }


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_role
# ---------------------------------------------------------------------------


class TestSemanticRole:
    def test_generates_role_from_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = _schema({"x-polylogue-semantic-role": "message_role"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == "user"

    def test_role_respects_observed_values(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = _schema(
            {
                "x-polylogue-semantic-role": "message_role",
                "x-polylogue-values": ["user", "assistant", "system"],
            }
        )
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in {"user", "assistant", "system"}

    def test_role_uses_cycle_when_cycle_value_in_observed(self) -> None:
        gen = SemanticValueGenerator(
            random.Random(0),
            role_cycle=["human", "assistant"],
        )
        schema = _schema(
            {
                "x-polylogue-semantic-role": "message_role",
                "x-polylogue-values": ["human", "assistant"],
            }
        )
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == "human"  # current_role is "human"


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_body
# ---------------------------------------------------------------------------


class TestSemanticBody:
    def test_generates_nonempty_body_text(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = _schema({"x-polylogue-semantic-role": "message_body"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert len(value) > 0

    def test_body_text_matches_current_role(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = _schema({"x-polylogue-semantic-role": "message_body"})

        # Turn 0 = user
        _, user_text = gen.try_generate(schema)
        assert user_text in _ROLE_TEXTS["user"]

        gen.advance_turn()
        # Turn 1 = assistant
        _, assistant_text = gen.try_generate(schema)
        assert assistant_text in _ROLE_TEXTS["assistant"]

    def test_body_with_theme_uses_themed_content(self) -> None:
        theme = _DEMO_THEMES[0]
        gen = SemanticValueGenerator(random.Random(0), theme=theme)
        schema = _schema({"x-polylogue-semantic-role": "message_body"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in theme.user_turns  # turn 0 = user


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_timestamp
# ---------------------------------------------------------------------------


class TestSemanticTimestamp:
    def test_generates_epoch_by_default(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = _schema({"x-polylogue-semantic-role": "message_timestamp"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, float)
        assert value == 1700000000.0  # turn 0

    def test_sequential_timestamps_increase(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = _schema({"x-polylogue-semantic-role": "message_timestamp"})

        _, ts0 = gen.try_generate(schema)
        gen.advance_turn()
        _, ts1 = gen.try_generate(schema)
        gen.advance_turn()
        _, ts2 = gen.try_generate(schema)

        assert _float_value(ts0) < _float_value(ts1) < _float_value(ts2)

    def test_iso8601_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = _schema(
            {
                "x-polylogue-semantic-role": "message_timestamp",
                "x-polylogue-format": "iso8601",
            }
        )
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(value)
        assert parsed.tzinfo is not None

    def test_unix_epoch_str_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = _schema(
            {
                "x-polylogue-semantic-role": "message_timestamp",
                "x-polylogue-format": "unix-epoch-str",
            }
        )
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert float(value) == 1700000000.0

    def test_unix_epoch_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = _schema(
            {
                "x-polylogue-semantic-role": "message_timestamp",
                "x-polylogue-format": "unix-epoch",
            }
        )
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — session_title
# ---------------------------------------------------------------------------


class TestSemanticTitle:
    def test_generates_title_string(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = _schema({"x-polylogue-semantic-role": "session_title"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert len(value) > 0

    def test_themed_title_uses_theme(self) -> None:
        theme = _DEMO_THEMES[0]
        gen = SemanticValueGenerator(random.Random(0), theme=theme)
        schema = _schema({"x-polylogue-semantic-role": "session_title"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == theme.title

    def test_title_with_observed_values(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = _schema(
            {
                "x-polylogue-semantic-role": "session_title",
                "x-polylogue-values": ["Alpha", "Beta", "Gamma"],
            }
        )
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in {"Alpha", "Beta", "Gamma"}

    def test_title_without_theme_or_values_picks_demo_theme(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = _schema({"x-polylogue-semantic-role": "session_title"})
        handled, value = gen.try_generate(schema)
        assert handled is True
        known_titles = {t.title for t in _DEMO_THEMES}
        assert value in known_titles


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — fallback for unknown/unhandled roles
# ---------------------------------------------------------------------------


class TestSemanticFallback:
    def test_no_semantic_role_returns_not_handled(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = _schema({"type": "string"})
        handled, value = gen.try_generate(schema)
        assert handled is False
        assert value is None

    def test_unknown_semantic_role_returns_not_handled(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = _schema({"x-polylogue-semantic-role": "completely_unknown_role"})
        handled, value = gen.try_generate(schema)
        assert handled is False
        assert value is None

    def test_message_container_role_returns_not_handled(self) -> None:
        """message_container is structural and defers to normal generation."""
        gen = SemanticValueGenerator(random.Random(0))
        schema = _schema({"x-polylogue-semantic-role": "message_container"})
        handled, value = gen.try_generate(schema)
        assert handled is False


# =============================================================================
# Merged from test_synthetic_wire_formats.py (2024-03-15)
# =============================================================================


class TestWireFormatShape:
    """Validate the wire format configuration data structure."""

    EXPECTED_PROVIDERS = {"chatgpt", "claude-code", "claude-ai", "codex", "gemini"}

    def test_all_expected_providers_have_entries(self) -> None:
        """Every known provider has a wire format config."""
        assert set(PROVIDER_WIRE_FORMATS.keys()) == self.EXPECTED_PROVIDERS

    @pytest.mark.parametrize("provider", sorted(EXPECTED_PROVIDERS))
    def test_encoding_is_valid(self, provider: str) -> None:
        """Each wire format has a valid encoding value."""
        wf = PROVIDER_WIRE_FORMATS[provider]
        assert isinstance(wf, WireFormat)
        assert wf.encoding in ("json", "jsonl"), f"{provider} has invalid encoding: {wf.encoding}"

    def test_json_providers_have_structure(self) -> None:
        """JSON-encoded providers have either tree or messages_path."""
        for name, wf in PROVIDER_WIRE_FORMATS.items():
            if wf.encoding == "json":
                has_structure = wf.tree is not None or wf.messages_path is not None
                assert has_structure, f"{name}: JSON provider needs tree or messages_path"

    def test_chatgpt_has_tree_with_container(self) -> None:
        """ChatGPT wire format uses tree structure with container_path."""
        wf = PROVIDER_WIRE_FORMATS["chatgpt"]
        assert wf.tree is not None
        assert wf.tree.container_path == "mapping"
        assert wf.tree.children_field == "children"

    def test_claude_code_has_tree_without_container(self) -> None:
        """Claude Code uses tree structure in JSONL (no container_path)."""
        wf = PROVIDER_WIRE_FORMATS["claude-code"]
        assert wf.encoding == "jsonl"
        assert wf.tree is not None
        assert wf.tree.container_path is None
        assert wf.tree.session_field == "sessionId"

    @pytest.mark.parametrize("fallback", [_claude_code_content_fallback, _codex_content_fallback])
    def test_provider_fallback_can_emit_unknown_tool_result_shape(self, fallback: Callable[..., object]) -> None:
        """Fallbacks retain the no-verdict shape so the writer covers it."""
        for seed in range(100):
            result = fallback(random.Random(seed), "assistant", 0, None)
            if (
                isinstance(result, list)
                and result
                and isinstance(result[0], dict)
                and result[0].get("type") == "tool_result"
            ):
                assert "is_error" not in result[0]
                return
        pytest.fail("fallback did not produce a tool_result witness")

    def test_tool_heavy_batch_can_emit_unknown_tool_result_shape(self, qualified_provider_defaults: None) -> None:
        """Tool-heavy batches retain at least one result without a verdict."""
        spec = CorpusSpec.for_provider("codex", count=1, messages_min=4, messages_max=4, seed=71, style="tool-heavy")
        records = [json.loads(line) for line in SyntheticCorpus.generate_for_spec(spec)[0].splitlines()]
        result_blocks = [
            block
            for record in records
            for block in record.get("payload", {}).get("content", [])
            if block.get("type") == "tool_result"
        ]
        assert result_blocks
        assert any("is_error" not in block for block in result_blocks)


class TestCorpusSeedDeterminism:
    """Corpus generation is deterministic with same seed."""

    def test_same_seed_same_output(self, qualified_provider_corpus: Callable[[str], SyntheticCorpus]) -> None:
        """Two generations with the same seed produce identical bytes."""
        available = SyntheticCorpus.available_providers()
        if not available:
            fail_missing_schema("No schemas available")
        corpus = qualified_provider_corpus(available[0])
        a = corpus.generate(count=2, seed=99)
        b = corpus.generate(count=2, seed=99)
        assert a == b

    def test_different_seed_different_output(self, qualified_provider_corpus: Callable[[str], SyntheticCorpus]) -> None:
        """Two generations with different seeds produce different bytes."""
        available = SyntheticCorpus.available_providers()
        if not available:
            fail_missing_schema("No schemas available")
        corpus = qualified_provider_corpus(available[0])
        a = corpus.generate(count=2, seed=1)
        b = corpus.generate(count=2, seed=2)
        assert a != b


class TestCorpusParseRoundtrip:
    """Generated corpus parses successfully through source iterators."""

    @pytest.mark.parametrize("provider", sorted(PROVIDER_WIRE_FORMATS.keys()))
    def test_generated_data_parses(self, provider: str, synthetic_source: SyntheticSourceFactory) -> None:
        """Synthetic data for each provider round-trips through parser."""
        from polylogue.sources import iter_source_sessions

        try:
            source = synthetic_source(provider, count=2, seed=42)
        except FileNotFoundError:
            fail_missing_schema(f"No schema available for {provider}")

        convos = list(iter_source_sessions(source))
        assert len(convos) > 0, f"No sessions parsed for {provider}"
        for conv in convos:
            assert len(conv.messages) > 0, f"Empty session for {provider}"


# =============================================================================
# Merged from test_synthetic_corpus.py (2024-03-15)
# =============================================================================


class TestSeedDeterminism:
    """Corpus generation is deterministic given the same seed."""

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_same_seed_produces_identical_output(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        """Two generate() calls with same seed → identical byte output."""
        try:
            corpus = qualified_provider_corpus(provider)
        except FileNotFoundError:
            fail_missing_schema(f"No schema for {provider}")

        a = corpus.generate(count=3, seed=42)
        b = corpus.generate(count=3, seed=42)
        assert a == b, f"{provider}: same seed produced different output"


class TestMessageCountContract:
    """Generated sessions respect the message count range."""

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_generate_count_matches_requested(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        """generate(count=N) returns exactly N items."""
        try:
            corpus = qualified_provider_corpus(provider)
        except FileNotFoundError:
            fail_missing_schema(f"No schema for {provider}")

        for count in (1, 3, 5):
            items = corpus.generate(count=count, seed=0)
            assert len(items) == count, f"{provider}: requested {count}, got {len(items)}"

    def test_generate_zero_returns_empty(self, qualified_provider_corpus: Callable[[str], SyntheticCorpus]) -> None:
        """generate(count=0) returns an empty list."""
        available = SyntheticCorpus.available_providers()
        if not available:
            fail_missing_schema("No schemas available")

        corpus = qualified_provider_corpus(available[0])
        items = corpus.generate(count=0, seed=0)
        assert items == []

    def test_generate_batch_reports_provider_and_count(
        self, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        """generate_batch() exposes a typed report aligned with raw output."""
        available = SyntheticCorpus.available_providers()
        if not available:
            fail_missing_schema("No schemas available")

        corpus = qualified_provider_corpus(available[0])
        batch = corpus.generate_batch(count=2, seed=7, messages_per_session=range(4, 5))
        assert len(batch.artifacts) == 2
        assert batch.report.generated_count == 2
        assert batch.report.provider == corpus.provider
        assert batch.report.package_version == corpus.package_version
        assert batch.raw_items == [artifact.raw_bytes for artifact in batch.artifacts]


class TestParseRoundtrip:
    """Synthetic data round-trips through the actual provider parsers."""

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_synthetic_parses_to_sessions(self, provider: str, synthetic_source: SyntheticSourceFactory) -> None:
        """Synthetic corpus for each provider parses into valid sessions."""
        from polylogue.sources import iter_source_sessions

        try:
            source = synthetic_source(provider, count=2, seed=42)
        except FileNotFoundError:
            fail_missing_schema(f"No schema for {provider}")

        convos = list(iter_source_sessions(source))
        assert len(convos) > 0, f"No sessions parsed for {provider}"

        for conv in convos:
            assert len(conv.messages) > 0, f"Empty session for {provider}"
            # At least one message should have non-empty text
            assert any(m.text for m in conv.messages), f"No message text for {provider}"

    def test_antigravity_metadata_sidecar_artifact_still_refused_as_a_session(self) -> None:
        """Red twin for polylogue-ujitw: unblocking the real markdown-export
        route above must not accidentally make the synthetic generator's
        OTHER antigravity shape -- the generic-engine brain-metadata sidecar
        (artifactType/summary/updatedAt) -- pass as a session too. PR #3441
        made a real ``*.metadata.json`` file on disk classify as
        ``AGENT_SIDECAR_META``, never a session; this proves the synthetic
        generator's own sidecar-shaped output still hits that same refusal
        at the real acquisition-classification boundary, not just a parser
        smoke test.
        """
        from polylogue.archive.artifact_taxonomy import classify_artifact

        payload: dict[str, JSONValue] = {
            "artifactType": "brain-metadata",
            "summary": "Synthetic sidecar evidence",
            "updatedAt": "2026-01-01T00:00:00Z",
        }
        assert set(payload) >= {"artifactType", "summary", "updatedAt"}, (
            "fixture drifted off the brain-metadata sidecar shape this test targets"
        )

        artifact = classify_artifact(
            payload,
            provider="antigravity",
            source_path="/home/user/.antigravity/brain/artifact-1.metadata.json",
        )

        assert artifact.parse_as_session is False


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

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_roundtrip_produces_messages(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus(provider)
        results = corpus.generate(count=3, seed=42, messages_per_session=range(4, 8))
        assert len(results) == 3

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)

        for i, raw in enumerate(results):
            payload = _deserialize_for_parser(provider, raw)
            sessions = parse_payload(runtime_provider, payload, f"synth-{provider}-{i}")
            assert len(sessions) >= 1, (
                f"Provider {provider}: parse_payload returned no sessions for synthetic data (index {i})"
            )
            conv = sessions[0]
            assert len(conv.messages) > 0, f"Provider {provider}: parsed session has no messages (index {i})"

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_parsed_messages_have_roles(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus(provider)
        [raw] = corpus.generate(count=1, seed=7, messages_per_session=range(5, 6))

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)
        payload = _deserialize_for_parser(provider, raw)
        sessions = parse_payload(runtime_provider, payload, f"synth-{provider}")
        assert sessions

        for msg in sessions[0].messages:
            assert msg.role is not None
            assert str(msg.role) != ""

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_parsed_messages_have_text_content(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus(provider)
        [raw] = corpus.generate(count=1, seed=99, messages_per_session=range(4, 5))

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)
        payload = _deserialize_for_parser(provider, raw)
        sessions = parse_payload(runtime_provider, payload, f"synth-{provider}")
        assert sessions

        messages_with_text = [m for m in sessions[0].messages if m.text]
        assert len(messages_with_text) > 0, f"Provider {provider}: no messages have text content"


# ---------------------------------------------------------------------------
# Determinism: same seed → same output
# ---------------------------------------------------------------------------


class TestSyntheticDeterminism:
    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_same_seed_same_output(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus(provider)
        result_a = corpus.generate(count=3, seed=123)
        result_b = corpus.generate(count=3, seed=123)
        assert result_a == result_b

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_different_seeds_different_output(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus(provider)
        result_a = corpus.generate(count=2, seed=1)
        result_b = corpus.generate(count=2, seed=2)
        # At least one session should differ
        assert result_a != result_b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSyntheticEdgeCases:
    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_count_zero_produces_empty_list(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        corpus = qualified_provider_corpus(provider)
        result = corpus.generate(count=0, seed=0)
        assert result == []

    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_single_message_session(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        """A session with exactly 1 message should still parse."""
        corpus = qualified_provider_corpus(provider)
        [raw] = corpus.generate(count=1, seed=55, messages_per_session=range(1, 2))

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)
        payload = _deserialize_for_parser(provider, raw)
        sessions = parse_payload(runtime_provider, payload, f"synth-{provider}")
        assert len(sessions) >= 1
        assert len(sessions[0].messages) >= 1


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------


class TestProviderAvailability:
    def test_available_providers_is_nonempty(self) -> None:
        providers = SyntheticCorpus.available_providers()
        assert len(providers) > 0

    def test_available_providers_are_known(self) -> None:
        known = {"chatgpt", "claude-ai", "claude-code", "codex", "gemini", "antigravity"}
        for provider in SyntheticCorpus.available_providers():
            assert provider in known, f"Unknown provider: {provider}"

    def test_for_provider_raises_on_unknown(self) -> None:
        with pytest.raises((FileNotFoundError, ValueError)):
            SyntheticCorpus.for_provider("nonexistent-provider-xyz")

    def test_for_provider_accepts_explicit_version_and_element(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.synthetic.core as synthetic_core

        fake_package = _FakePackage(
            version="v2",
            default_element_kind="session_record_stream",
            available_elements={"session_record_stream"},
        )
        fake_schema = _schema({"type": "object"})
        fake_registry = _FakeRegistry(
            package=fake_package,
            element_schema=fake_schema,
        )
        monkeypatch.setattr(synthetic_core, "SchemaRegistry", lambda: fake_registry)

        corpus = SyntheticCorpus.for_provider(
            "chatgpt",
            version="v2",
            element_kind="session_record_stream",
        )

        assert isinstance(corpus, SyntheticCorpus)
        assert fake_registry.package_calls == [("chatgpt", "v2")]
        assert fake_registry.element_schema_calls == [("chatgpt", "v2", "session_record_stream")]
        assert corpus.schema == fake_schema

    def test_for_provider_rejects_unknown_element_when_package_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.synthetic.core as synthetic_core

        fake_package = _FakePackage(
            version="v2",
            default_element_kind="session_record_stream",
            available_elements=set(),
        )
        fake_registry = _FakeRegistry(package=fake_package)
        monkeypatch.setattr(synthetic_core, "SchemaRegistry", lambda: fake_registry)

        with pytest.raises(ValueError):
            SyntheticCorpus.for_provider("chatgpt", version="v2", element_kind="unknown_element")

        assert fake_registry.package_calls == [("chatgpt", "v2")]

    def test_for_provider_rejects_element_without_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.synthetic.core as synthetic_core

        fake_schema = _schema({"type": "object"})
        fake_registry = _FakeRegistry(package=None, schema=fake_schema)
        monkeypatch.setattr(synthetic_core, "SchemaRegistry", lambda: fake_registry)

        with pytest.raises(ValueError):
            SyntheticCorpus.for_provider("chatgpt", element_kind="session_record_stream")


# ---------------------------------------------------------------------------
# Demo style roundtrip
# ---------------------------------------------------------------------------


class TestDemoStyleRoundtrip:
    @pytest.mark.parametrize("provider", QUALIFIED_SCHEMA_PROVIDERS)
    def test_demo_style_roundtrips(
        self, provider: str, qualified_provider_corpus: Callable[[str], SyntheticCorpus]
    ) -> None:
        """Demo-style synthetic data should also parse correctly."""
        corpus = qualified_provider_corpus(provider)
        results = corpus.generate(
            count=2,
            seed=42,
            messages_per_session=range(4, 8),
            style="demo",
        )
        assert len(results) == 2

        runtime_provider = _SCHEMA_TO_RUNTIME_PROVIDER.get(provider, provider)

        for i, raw in enumerate(results):
            payload = _deserialize_for_parser(provider, raw)
            sessions = parse_payload(runtime_provider, payload, f"demo-{provider}-{i}")
            assert len(sessions) >= 1
            assert len(sessions[0].messages) > 0


def test_file_writer_streams_the_same_seeded_fixture_program(tmp_path: Path) -> None:
    """An output-sized buffer or a separate writer sampler turns this red."""
    length = 2_000_003
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "x-polylogue-observed-distribution": {
                    "string_length": {"count": 1, "histogram": [[20, 1]], "log_base": 2, "min": length, "max": length},
                },
            },
        },
    }
    selection = SyntheticSchemaSelection("chatgpt", "neutral", None, schema, WireFormat(encoding="json"))
    spec = CorpusSpec.for_provider("chatgpt", count=3, messages_min=1, messages_max=1, seed=97)
    tracemalloc.start()
    try:
        written = SyntheticCorpus.write_selection_artifacts(selection, spec, tmp_path, prefix="streamed")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024
    assert written.report.generated_count == 3
    fixture = SyntheticCorpus.generate_batch_for_selection(selection, spec)
    assert written.facts == tuple(artifact.facts for artifact in fixture.artifacts)
    for path, artifact in zip(written.session_files, fixture.artifacts, strict=True):
        raw = path.read_bytes()
        assert raw == artifact.raw_bytes
        assert len(json.loads(raw)["body"]) == length
        assert hashlib.sha256(raw).hexdigest() == artifact.facts.raw_sha256


def test_streamed_codex_lineage_retains_the_original_complete_prefix(tmp_path: Path) -> None:
    """Dropping an ancestor or copying only the latest tail turns this red."""
    import uuid
    from copy import deepcopy

    from polylogue.schemas.generation.workload_profiles import workload_profile_identity
    from polylogue.schemas.synthetic.authored_selection import authored_schema_selection

    # Explicit authored lineage operands borrow the same inferred neutral tool
    # program. An authored style must not replace a separately selected profile.
    authored = authored_schema_selection("codex", "tool-heavy")
    from polylogue.schemas.workload_profile_rows import WorkloadProfile

    assert isinstance(authored.workload_profile, WorkloadProfile)
    profile = WorkloadProfile(deepcopy(authored.workload_profile._fields), authored.workload_profile.depth_grammars)
    from polylogue.core.json import JSONDocument, json_document
    from polylogue.schemas.generation.models import _PackageAccumulator
    from polylogue.schemas.generation.observation_journal import ObservationJournal
    from polylogue.schemas.generation.workload_profiles import build_package_workload_profile
    from polylogue.schemas.observation import SchemaUnit

    native_ids = tuple(str(uuid.UUID(int=ordinal + 100, version=4)) for ordinal in range(4))
    own: list[JSONDocument] = [
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "neutral"}]},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "neutral"}],
            },
        },
    ]
    with ObservationJournal.create(root=tmp_path / "declared-lineage") as journal:
        previous: list[JSONDocument] = []
        for ordinal, identity in enumerate(native_ids):
            header: JSONDocument = {
                "type": "session_meta",
                "payload": {
                    "id": identity,
                    "timestamp": "2025-06-01T00:00:00Z",
                    **({"forked_from_id": native_ids[ordinal - 1]} if ordinal else {}),
                },
            }
            original_records = [header, *previous, *own]
            unit = journal.append_unit(
                SchemaUnit(
                    cluster_payload={},
                    schema_samples=original_records,
                    artifact_kind="session_record_stream",
                    raw_id=f"neutral-{ordinal}",
                    bundle_scope="declared-neutral",
                )
            )
            journal.assign_profile_family(unit, "lineage")
            previous = [*previous, *own]
        measured = build_package_workload_profile(
            provider="codex",
            version="neutral",
            package=_PackageAccumulator(
                provider="codex",
                anchor_family_id="lineage",
                anchor_kind="session_record_stream",
                memberships=journal.memberships(),
            ),
            element_schemas={"session_record_stream": authored.schema},
            privacy_policy="standard",
        )
    lineage = json_document(measured["session_lineage"])
    rows = lineage["header_classes"]
    assert isinstance(rows, list)
    root_class = next(
        index for index, row in enumerate(rows) if isinstance(row, dict) and row["outcome"] == "no_explicit_parent"
    )
    child_class = next(index for index, row in enumerate(rows) if isinstance(row, dict) and row["outcome"] == "matched")
    profile["session_lineage"] = lineage
    declarations: list[JSONValue] = [{"class": root_class}]
    for ordinal in range(1, 4):
        declarations.append(
            {
                "class": child_class,
                "prefix_point": [2 * ordinal + 3, 2 * ordinal + 1, 2 * ordinal + 2, 2 * ordinal, 2 * ordinal],
            }
        )
    profile["authored_session_cycle"] = declarations
    profile["profile_id"] = workload_profile_identity(profile)
    selection = SyntheticSchemaSelection(
        "codex", "neutral", "session_record_stream", authored.schema, authored.wire_format, workload_profile=profile
    )
    spec = CorpusSpec.for_provider(
        "codex",
        count=4,
        messages_min=2,
        messages_max=2,
        seed=71,
        style="default",
        session_native_ids=native_ids,
    )
    written = SyntheticCorpus.write_selection_artifacts(selection, spec, tmp_path, prefix="lineage")
    fixture = SyntheticCorpus.generate_batch_for_selection(selection, spec)
    assert written.report == fixture.report
    assert written.report.lineage_edge_count == 3
    assert written.facts == tuple(artifact.facts for artifact in fixture.artifacts)
    previous_messages: list[dict[str, object]] = []
    for ordinal, (path, artifact) in enumerate(zip(written.session_files, fixture.artifacts, strict=True)):
        raw = path.read_bytes()
        assert raw == artifact.raw_bytes
        records = [json.loads(line) for line in raw.splitlines()]
        assert records[0]["type"] == "session_meta"
        if ordinal:
            assert records[0]["payload"]["forked_from_id"] == spec.session_native_ids[ordinal - 1]
        messages = [record for record in records if record.get("payload", {}).get("type") == "message"]
        assert messages[: len(previous_messages)] == previous_messages
        assert len(messages) == (ordinal + 1) * 2
        assert artifact.message_count == artifact.facts.message_count == len(messages)
        tool_ids = tuple(
            block["id"]
            for record in messages
            for block in record["payload"]["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        )
        assert tool_ids
        assert artifact.facts.tool_use_ids == tool_ids
        previous_messages = messages

    # The lazy driver borrows inherited identities from the same physical
    # prefix owner. Only explicit fixture/facts collection builds tuples.
    lazy = SyntheticCorpus.from_selection(selection).iter_sessions(
        count=2, messages_per_session=range(2, 3), seed=71, session_native_ids=native_ids[:2]
    )
    try:
        original = next(lazy)
        for _chunk in original.chunks:
            pass
        for adjunct in original.adjuncts:
            for _chunk in adjunct.chunks:
                pass
        child = next(lazy)
        borrowed_calls = child.inherited_tool_ids[0]
        assert not isinstance(borrowed_calls, tuple | list)
        assert tuple(borrowed_calls) == written.facts[0].tool_use_ids
        for _chunk in child.chunks:
            pass
        for adjunct in child.adjuncts:
            for _chunk in adjunct.chunks:
                pass
    finally:
        lazy.close()
    with pytest.raises(RuntimeError, match="original prefix owner"):
        tuple(borrowed_calls)


def test_generated_arrays_retain_original_children_and_bound_serialization() -> None:
    import json
    import tracemalloc

    from polylogue.schemas.synthetic.array_value import walk_generated_values
    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph
    from polylogue.schemas.synthetic.runtime import _json_chunks

    owner = ForeignKeyGraph()
    sampled = owner.new_array()
    schema: SchemaRecord = {"type": "object", "properties": {"ordinal": {"type": "integer"}}}
    try:
        tracemalloc.start()
        for index in range(10_003):
            record = owner.new_record(schema)
            record["ordinal"] = index
            owner.append_array_item(sampled, record)
        for child in sampled:
            assert isinstance(child, GeneratedRecord)
            assert child.selected_schema == schema
            child["adapted"] = True
        # Encoding consumes one retained child at a time; fixture collection
        # below is deliberately separate from the measured streaming path.
        byte_count = sum(len(piece) for piece in _json_chunks(sampled, pretty=False))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 4 * 1024 * 1024
        assert byte_count > 300_000
        decoded = json.loads("".join(_json_chunks(sampled, pretty=False)))
        assert len(decoded) == 10_003
        assert decoded[0] == {"ordinal": 0, "adapted": True}
        assert decoded[-1] == {"ordinal": 10_002, "adapted": True}
        assert sum(isinstance(value, GeneratedRecord) for value in walk_generated_values(sampled)) == 10_003
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        owner.close()
    with pytest.raises(RuntimeError, match="original relation owner"):
        len(sampled)


def test_generated_array_depth_numeric_order_and_cycle_refusal() -> None:
    from polylogue.schemas.synthetic.array_value import walk_generated_values
    from polylogue.schemas.synthetic.models import GeneratedValue
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph
    from polylogue.schemas.synthetic.runtime import _json_chunks

    owner = ForeignKeyGraph()
    try:
        numbers = owner.new_array()
        for value in (2**61 + 1, 1.5, 2**61, -2, 1.0):
            owner.append_array_item(numbers, value)
        assert list(owner.sorted_numeric_array(numbers)) == [-2, 1.0, 1.5, 2**61, 2**61 + 1]
        replaced = owner.new_array()
        replaced.append({"original": True})
        borrowed = iter(replaced)
        stale = next(borrowed)
        replaced[0] = {"replacement": True}
        assert isinstance(stale, dict)
        stale["stale"] = True
        borrowed.close()
        assert replaced[0] == {"replacement": True}
        replaced.insert(0, {"header": True})
        assert list(replaced) == [{"header": True}, {"replacement": True}]
        nested: GeneratedValue = 7
        for _ in range(2_003):
            nested = {"child": nested}
        deep = owner.new_array()
        owner.append_array_item(deep, nested)
        pieces = list(_json_chunks(deep, pretty=False))
        assert pieces.count("{") == 2_003
        assert pieces.count("}") == 2_003
        assert "7" in pieces
        repeated = {"leaf": True}
        assert sum(isinstance(value, dict) for value in walk_generated_values([repeated, repeated])) == 2
        cyclic = owner.new_array()
        owner.append_array_item(cyclic, cyclic)
        with pytest.raises(ValueError, match="cyclic"):
            list(walk_generated_values(cyclic))
        ordinary: list[object] = []
        ordinary.append(ordinary)
        with pytest.raises(ValueError, match="cyclic"):
            list(walk_generated_values(ordinary))
    finally:
        owner.close()


@pytest.mark.parametrize("provider", ["claude-ai", "codex", "chatgpt"])
def test_record_containers_use_the_original_native_session_driver(tmp_path: Path, provider: str) -> None:
    import tracemalloc

    count = 6_003
    from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
    from polylogue.schemas.synthetic.authored_selection import authored_schema_selection

    if provider == "claude-ai":
        # This neutral container contract declares its actual native envelope
        # and message fields before draws; it is not an operator population.
        schema = generate_schema_from_samples(
            [
                {
                    "uuid": "native-records",
                    "chat_messages": [
                        {
                            "uuid": "00000000-0000-4000-8000-000000000001",
                            "sender": "human",
                            "text": "neutral message",
                        }
                    ],
                }
            ],
            provider=provider,
            max_stats_samples=0,
        )
        selection = SyntheticSchemaSelection(
            provider, "neutral-native-records", None, schema, PROVIDER_WIRE_FORMATS[provider]
        )
    else:
        selection = authored_schema_selection(
            provider, "demo" if provider == "chatgpt" else "tool-heavy", session_native_ids=("native-records",)
        )
    spec = CorpusSpec.for_provider(
        provider, count=1, messages_min=count, messages_max=count, seed=19, session_native_ids=("native-records",)
    )
    tracemalloc.start()
    try:
        written = SyntheticCorpus.write_selection_artifacts(selection, spec, tmp_path, prefix=provider)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024
    assert written.report.generated_count == 1
    assert written.facts[0].message_count == count
    raw = written.session_files[0].read_bytes()
    if provider == "claude-ai":
        document = json.loads(raw)
        assert len(document["chat_messages"]) == count
        assert document["uuid"] == "native-records"
        assert document["chat_messages"][-1]["sender"] == "human"
    elif provider == "chatgpt":
        document = json.loads(raw)
        nodes = document["mapping"]
        assert len(nodes) == count
        assert document["id"] == "native-records"
        assert document["current_node"] in nodes
        roots = 0
        for key, node in nodes.items():
            assert node["id"] == key
            if node["parent"] is None:
                roots += 1
            else:
                assert key in nodes[node["parent"]]["children"]
            for child in node["children"]:
                assert nodes[child]["parent"] == key
        assert roots == 1
    else:
        records = [json.loads(line) for line in raw.splitlines()]
        assert len(records) == count + 1
        assert records[0]["payload"]["id"] == "native-records"
        assert all(record["type"] == "response_item" for record in records[1:])
        assert records[-1]["payload"]["role"] == "user"


def test_dynamic_object_fanout_is_retained_and_streamed_on_the_original_owner() -> None:
    import random
    import tracemalloc

    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.schemas.synthetic.text_value import JoinedText, SampledText

    count = 10_003
    length = 4_099
    schema: SchemaRecord = {
        "type": "object",
        "additionalProperties": {
            "type": "string",
            "x-polylogue-observed-distribution": {
                "string_length": {"count": 1, "histogram": [[12, 1]], "log_base": 2, "min": length, "max": length},
            },
        },
        "x-polylogue-observed-distribution": {
            "object_fanout": {"count": 1, "histogram": [[13, 1]], "log_base": 2, "min": count, "max": count},
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "chatgpt")
    tracemalloc.start()
    try:
        data = corpus._generate_from_schema(schema, random.Random(31))
        assert isinstance(data, GeneratedRecord)
        assert len(data) == count
        final_value = data["key-10002"]
        assert isinstance(final_value, str | SampledText | JoinedText) and len(final_value) == length
        encoded = sum(len(chunk) for chunk in corpus.iter_serialized_chunks(data))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        corpus._relation_solver.close()
    assert peak < 4 * 1024 * 1024
    assert encoded > count * length
    with pytest.raises(RuntimeError, match="original relation owner"):
        len(data)


def test_lazy_session_driver_requires_consumption_and_closes_the_original_owner() -> None:
    from itertools import islice

    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.schemas.synthetic.text_value import SampledText

    length = 2_000_000_003
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "x-polylogue-observed-distribution": {
                    "string_length": {
                        "count": 1,
                        "histogram": [[30, 1]],
                        "log_base": 2,
                        "min": length,
                        "max": length,
                    }
                },
            }
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "test")
    tracemalloc.start()
    try:
        sessions = corpus.iter_sessions(seed=113)
        first = next(sessions)
        assert isinstance(first.data, GeneratedRecord)
        body = first.data["body"]
        assert isinstance(body, SampledText) and len(body) == length
        prefix = list(islice(first.chunks, 4))
        assert sum(map(len, prefix)) < 10_000
        competing = corpus.iter_sessions(count=1, seed=114)
        with pytest.raises(ValueError, match="already active"):
            next(competing)
        assert first.data["body"] == body
        with pytest.raises(ValueError, match="consume"):
            next(sessions)
        with pytest.raises(RuntimeError, match="original"):
            first.data["body"]
        assert next(first.chunks, None) is None
        # A fresh original scope is lawful after the failed partial consumer.
        retry = corpus.iter_sessions(count=1, seed=113)
        retried = next(retry)
        assert list(islice(retried.chunks, 4)) == prefix
        retry.close()
        assert isinstance(retried.data, GeneratedRecord)
        with pytest.raises(RuntimeError, match="original"):
            retried.data["body"]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


def test_lazy_session_driver_returns_actual_finite_completion_and_matches_fixture_bytes() -> None:
    from polylogue.schemas.synthetic.models import GeneratedRecord, SyntheticGenerationReport

    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "x-polylogue-values": ["neutral body"],
            }
        },
    }
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "test")
    sessions = corpus.iter_sessions(count=3, seed=127)
    raw_items: list[bytes] = []
    report: SyntheticGenerationReport | None = None
    while True:
        try:
            item = next(sessions)
        except StopIteration as completed:
            report = completed.value
            break
        assert item.index == len(raw_items)
        raw_items.append(b"".join(item.chunks))
    assert isinstance(report, SyntheticGenerationReport)
    assert report.requested_count == report.generated_count == 3
    assert all(json.loads(raw) == {"body": "neutral body"} for raw in raw_items)
    assert isinstance(item.data, GeneratedRecord)
    with pytest.raises(RuntimeError, match="original"):
        item.data["body"]
    assert corpus.generate_batch(count=3, seed=127).raw_items == raw_items


def test_original_cohort_prefix_cells_survive_scope_advance_without_repointing() -> None:
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    owner = ForeignKeyGraph()
    try:
        original = owner.new_record({})
        original["value"] = "first scope"
        connection = owner._connection
        root = owner.retain_session_prefix(
            "root", parent_prefix=None, inherited_messages=0, own_messages=2, record_count=3
        )
        owner.retain_prefix_chunk(
            root, 0, 0, b'{"role":"assistant","content":[{"type":"tool_use","id":"first-tool"}]}\n'
        )
        owner.retain_prefix_chunk(
            root, 1, 0, b'{"role":"assistant","content":[{"type":"tool_use","id":"second-tool"}]}\n'
        )
        owner.retain_prefix_tool(root, "call", 0, 0, "first-tool")
        owner.retain_prefix_tool(root, "call", 1, 0, "second-tool")
        owner.complete_session_prefix(root)
        owner.advance_scope()
        assert owner._connection is connection
        with pytest.raises(RuntimeError):
            original["value"]

        child = owner.retain_session_prefix(
            "child", parent_prefix=root, inherited_messages=1, own_messages=1, record_count=3
        )
        owner.retain_prefix_chunk(
            child, 0, 0, b'{"role":"assistant","content":[{"type":"tool_use","id":"child-tool"}]}\n'
        )
        owner.retain_prefix_tool(child, "call", 0, 0, "child-tool")
        with pytest.raises(ValueError):
            list(owner.iter_prefix_chunks(child, 2))
        owner.complete_session_prefix(child)
        owner.advance_scope()
        neighbour = owner.retain_session_prefix(
            "neighbour", parent_prefix=None, inherited_messages=0, own_messages=1, record_count=2
        )
        owner.retain_prefix_chunk(neighbour, 0, 0, b'{"role":"user","content":["neighbour"]}\n')
        owner.complete_session_prefix(neighbour)
        owner.advance_scope()
        assert owner._connection is connection
        assert (
            b"".join(owner.iter_prefix_chunks(child, 2))
            == b'{"role":"assistant","content":[{"type":"tool_use","id":"first-tool"}]}\n{"role":"assistant","content":[{"type":"tool_use","id":"child-tool"}]}\n'
        )
        assert tuple(owner.iter_prefix_tools(child, 2, "call")) == ("first-tool", "child-tool")
        assert b"second-tool" not in b"".join(owner.iter_prefix_chunks(child, 1))
        assert b"".join(owner.iter_prefix_chunks(neighbour, 1)) == b'{"role":"user","content":["neighbour"]}\n'
        with pytest.raises(ValueError):
            list(owner.iter_prefix_chunks(child, 3))
    finally:
        owner.close()
    with pytest.raises(RuntimeError):
        list(owner.iter_prefix_chunks(child, 1))


def test_prefix_identity_recipe_keeps_original_shape_and_history_without_hydration() -> None:
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph
    from polylogue.schemas.synthetic.text_value import SampledText

    owner = ForeignKeyGraph()
    identity = SampledText(2_000_000_000, "neutral")
    schema: SchemaRecord = {"type": "string"}
    try:
        prefix = owner.retain_session_prefix(
            "neutral-parent", parent_prefix=None, inherited_messages=0, own_messages=1, record_count=2
        )
        owner.retain_prefix_chunk(prefix, 0, 0, b"neutral original message\n")
        owner.retain_prefix_tool(prefix, "call", 0, 0, identity, schema=schema)
        owner.retain_prefix_tool(prefix, "result", 0, 1, identity, schema=schema, result_mode="null:inline")
        owner.complete_session_prefix(prefix)
        owner.advance_scope()
        tracemalloc.start()
        owner.restore_prefix_tool_context(prefix, 1)
        selected, selected_schema = owner.select_tool_identity(
            "both", random.Random(11), prior_result_mode="null:inline"
        )
        assert selected == identity and selected_schema == schema
        assert len(selected) == 2_000_000_000
        assert tuple(owner.iter_prefix_tools(prefix, 1, "call")) == (identity,)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 1024 * 1024
    finally:
        tracemalloc.stop()
        owner.close()
