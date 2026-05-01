"""Tests for semantic role annotation wiring in synthetic generation.

Verifies that:
1. Baseline schemas contain x-polylogue-semantic-role annotations
2. SemanticValueGenerator is initialized during generation
3. Semantic values from annotations take precedence over fixup fallbacks
4. Wire-format fixups still produce parseable output without annotations
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic.core import SyntheticCorpus
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator

_BUNDLED_REGISTRY = SchemaRegistry(storage_root=SCHEMA_DIR)

# Expected annotations per provider — all providers now consistently detect
# the same 5 roles after the semantic inference overhaul (2026-03-16).
EXPECTED_ANNOTATIONS: dict[str, list[str]] = {
    "chatgpt": ["conversation_title", "message_body", "message_container", "message_role", "message_timestamp"],
    "claude-ai": ["conversation_title", "message_body", "message_container", "message_role", "message_timestamp"],
    "claude-code": ["conversation_title", "message_body", "message_container", "message_role", "message_timestamp"],
    "codex": ["conversation_title", "message_body", "message_container", "message_role", "message_timestamp"],
    "gemini": ["conversation_title", "message_body", "message_container", "message_role", "message_timestamp"],
}


def _synthetic_source(factory: Callable[..., object], provider: str, *, count: int, seed: int) -> Source:
    source = factory(provider, count=count, seed=seed)
    assert isinstance(source, Source)
    return source


def _collect_semantic_roles(schema: JSONValue, *, roles: list[str] | None = None) -> list[str]:
    """Recursively collect all x-polylogue-semantic-role values from a schema."""
    if roles is None:
        roles = []
    if not isinstance(schema, dict):
        return roles
    role = schema.get("x-polylogue-semantic-role")
    if isinstance(role, str):
        roles.append(role)
    for value in schema.values():
        if isinstance(value, dict):
            _collect_semantic_roles(value, roles=roles)
        elif isinstance(value, list):
            for item in value:
                _collect_semantic_roles(item, roles=roles)
    return roles


def _bundled_schema(provider: str) -> JSONDocument:
    package = _BUNDLED_REGISTRY.get_package(provider, version="default")
    assert package is not None, f"No bundled package for provider {provider}"
    schema = _BUNDLED_REGISTRY.get_element_schema(
        provider,
        version=package.version,
        element_kind=package.default_element_kind,
    )
    assert schema is not None, f"No bundled schema for provider {provider}"
    return schema


def _bundled_schema_path(provider: str) -> Path:
    package = _BUNDLED_REGISTRY.get_package(provider, version="default")
    assert package is not None, f"No bundled package for provider {provider}"
    element = package.element(package.default_element_kind)
    assert element is not None and element.schema_file is not None
    return SCHEMA_DIR / provider / "versions" / package.version / "elements" / element.schema_file


# =============================================================================
# 1. Baseline schemas have semantic annotations
# =============================================================================


class TestBaselineSchemaAnnotations:
    """Verify baseline provider schemas contain semantic role annotations."""

    @pytest.mark.parametrize("provider", list(EXPECTED_ANNOTATIONS.keys()))
    def test_schema_has_expected_semantic_roles(self, provider: str) -> None:
        """Each provider schema contains the expected semantic role annotations."""
        schema_path = _bundled_schema_path(provider)
        assert schema_path.exists(), f"Schema not found: {schema_path}"

        schema = _bundled_schema(provider)

        found_roles = sorted(set(_collect_semantic_roles(schema)))
        expected = sorted(EXPECTED_ANNOTATIONS[provider])
        assert found_roles == expected, f"{provider} schema has roles {found_roles}, expected {expected}"

    @pytest.mark.parametrize("provider", list(EXPECTED_ANNOTATIONS.keys()))
    def test_annotation_injection_is_idempotent(self, provider: str) -> None:
        """Re-injecting annotations doesn't change the schema."""
        from devtools.inject_semantic_annotations import inject_annotations

        schema = _bundled_schema(provider)

        original = json.dumps(schema, sort_keys=True)
        inject_annotations(provider, schema)
        after = json.dumps(schema, sort_keys=True)
        assert original == after, f"Injection was not idempotent for {provider}"


# =============================================================================
# 2. SemanticValueGenerator is initialized during generation
# =============================================================================


class TestSemanticGeneratorActivation:
    """Verify SemanticValueGenerator is initialized in generation paths."""

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_semantic_gen_set_after_generate(self, provider: str) -> None:
        """After generate(), _semantic_gen attribute is set on the corpus."""
        corpus = SyntheticCorpus.for_provider(provider)
        corpus.generate(count=1, seed=42, messages_per_conversation=range(3, 5))

        semantic_gen = getattr(corpus, "_semantic_gen", None)
        assert semantic_gen is not None, f"_semantic_gen not set after generation for {provider}"
        assert isinstance(semantic_gen, SemanticValueGenerator)

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_semantic_gen_has_correct_role_cycle(self, provider: str) -> None:
        """SemanticValueGenerator uses the provider's role cycle."""
        corpus = SyntheticCorpus.for_provider(provider)
        corpus.generate(count=1, seed=42, messages_per_conversation=range(3, 5))

        expected_roles = corpus._role_cycle()
        semantic_gen = getattr(corpus, "_semantic_gen", None)
        assert isinstance(semantic_gen, SemanticValueGenerator)
        assert semantic_gen.role_cycle == expected_roles


# =============================================================================
# 3. Roundtrip with semantic generation active
# =============================================================================


class TestSemanticRoundtrip:
    """Verify roundtrip still works with semantic generation active."""

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_roundtrip_with_annotations(
        self,
        provider: str,
        synthetic_source: Callable[..., object],
    ) -> None:
        """Synthetic data with semantic annotations roundtrips through parsers."""
        from polylogue.sources import iter_source_conversations

        source = _synthetic_source(synthetic_source, provider, count=3, seed=42)
        convos = list(iter_source_conversations(source))
        assert convos, f"No conversations parsed for {provider}"

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_roundtrip_different_seed(
        self,
        provider: str,
        synthetic_source: Callable[..., object],
    ) -> None:
        """Roundtrip with a different seed to catch seed-dependent issues."""
        from polylogue.sources import iter_source_conversations

        source = _synthetic_source(synthetic_source, provider, count=2, seed=99)
        convos = list(iter_source_conversations(source))
        assert convos, f"No conversations parsed for {provider} (seed=99)"


# =============================================================================
# 4. Wire-format fixups work as fallback without annotations
# =============================================================================


class TestWireFormatFallback:
    """Verify wire-format fixups produce parseable output without annotations."""

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_generation_without_annotations_still_parses(self, provider: str, tmp_path: Path) -> None:
        """Stripping semantic annotations from schema still produces parseable output."""
        from polylogue.config import Source
        from polylogue.sources import iter_source_conversations

        # Build corpus with annotations stripped
        corpus = SyntheticCorpus.for_provider(provider)
        stripped = copy.deepcopy(corpus.schema)
        _strip_semantic_roles(stripped)
        corpus.schema = stripped

        # Generate data using the stripped corpus directly
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        raw_items = corpus.generate(count=2, seed=77, messages_per_conversation=range(4, 12))

        provider_dir = tmp_path / "synthetic" / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        for idx, raw_bytes in enumerate(raw_items):
            (provider_dir / f"synth-{idx:02d}{ext}").write_bytes(raw_bytes)

        source = Source(name=f"{provider}-test", path=provider_dir / f"synth-00{ext}")
        convos = list(iter_source_conversations(source))
        assert convos, f"No conversations parsed without annotations for {provider}"


def _strip_semantic_roles(schema: JSONValue) -> None:
    """Recursively remove all x-polylogue-semantic-role annotations."""
    if not isinstance(schema, dict):
        return
    schema.pop("x-polylogue-semantic-role", None)
    for value in schema.values():
        if isinstance(value, dict):
            _strip_semantic_roles(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strip_semantic_roles(item)
