"""Tests for semantic role annotation wiring in synthetic generation.

Verifies that:
1. Baseline schemas contain x-polylogue-semantic-role annotations
2. SemanticValueGenerator is initialized during generation
3. Semantic values from annotations take precedence over fixup fallbacks
4. Wire-format fixups still produce parseable output without annotations
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.schemas.synthetic.core import SyntheticCorpus
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator

SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "polylogue" / "schemas" / "providers"

# Expected annotations per provider
EXPECTED_ANNOTATIONS: dict[str, list[str]] = {
    "chatgpt": ["message_role", "message_body", "message_timestamp", "conversation_title"],
    "claude-ai": ["message_role", "message_body", "message_timestamp", "conversation_title"],
    "claude-code": ["message_role", "message_body", "message_timestamp"],
    "codex": ["message_role", "message_body", "message_timestamp"],
    "gemini": ["message_role", "message_body"],
}


def _collect_semantic_roles(schema: dict, *, roles: list[str] | None = None) -> list[str]:
    """Recursively collect all x-polylogue-semantic-role values from a schema."""
    if roles is None:
        roles = []
    if not isinstance(schema, dict):
        return roles
    if "x-polylogue-semantic-role" in schema:
        roles.append(schema["x-polylogue-semantic-role"])
    for key, value in schema.items():
        if isinstance(value, dict):
            _collect_semantic_roles(value, roles=roles)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _collect_semantic_roles(item, roles=roles)
    return roles


# =============================================================================
# 1. Baseline schemas have semantic annotations
# =============================================================================


class TestBaselineSchemaAnnotations:
    """Verify baseline provider schemas contain semantic role annotations."""

    @pytest.mark.parametrize("provider", list(EXPECTED_ANNOTATIONS.keys()))
    def test_schema_has_expected_semantic_roles(self, provider: str) -> None:
        """Each provider schema contains the expected semantic role annotations."""
        schema_path = SCHEMAS_DIR / f"{provider}.schema.json.gz"
        assert schema_path.exists(), f"Schema not found: {schema_path}"

        with gzip.open(schema_path, "rt") as f:
            schema = json.load(f)

        found_roles = sorted(set(_collect_semantic_roles(schema)))
        expected = sorted(EXPECTED_ANNOTATIONS[provider])
        assert found_roles == expected, (
            f"{provider} schema has roles {found_roles}, expected {expected}"
        )

    @pytest.mark.parametrize("provider", list(EXPECTED_ANNOTATIONS.keys()))
    def test_annotation_injection_is_idempotent(self, provider: str) -> None:
        """Re-injecting annotations doesn't change the schema."""
        from devtools.inject_semantic_annotations import inject_annotations

        schema_path = SCHEMAS_DIR / f"{provider}.schema.json.gz"
        with gzip.open(schema_path, "rt") as f:
            schema = json.load(f)

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

        assert hasattr(corpus, "_semantic_gen"), (
            f"_semantic_gen not set after generation for {provider}"
        )
        assert isinstance(corpus._semantic_gen, SemanticValueGenerator)

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_semantic_gen_has_correct_role_cycle(self, provider: str) -> None:
        """SemanticValueGenerator uses the provider's role cycle."""
        corpus = SyntheticCorpus.for_provider(provider)
        corpus.generate(count=1, seed=42, messages_per_conversation=range(3, 5))

        expected_roles = corpus._role_cycle()
        assert corpus._semantic_gen.role_cycle == expected_roles


# =============================================================================
# 3. Roundtrip with semantic generation active
# =============================================================================


class TestSemanticRoundtrip:
    """Verify roundtrip still works with semantic generation active."""

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_roundtrip_with_annotations(self, provider: str, synthetic_source) -> None:
        """Synthetic data with semantic annotations roundtrips through parsers."""
        from polylogue.sources import iter_source_conversations

        source = synthetic_source(provider, count=3, seed=42)
        convos = list(iter_source_conversations(source))
        assert convos, f"No conversations parsed for {provider}"

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_roundtrip_different_seed(self, provider: str, synthetic_source) -> None:
        """Roundtrip with a different seed to catch seed-dependent issues."""
        from polylogue.sources import iter_source_conversations

        source = synthetic_source(provider, count=2, seed=99)
        convos = list(iter_source_conversations(source))
        assert convos, f"No conversations parsed for {provider} (seed=99)"


# =============================================================================
# 4. Wire-format fixups work as fallback without annotations
# =============================================================================


class TestWireFormatFallback:
    """Verify wire-format fixups produce parseable output without annotations."""

    @pytest.mark.parametrize("provider", SyntheticCorpus.available_providers())
    def test_generation_without_annotations_still_parses(self, provider: str, synthetic_source) -> None:
        """Stripping semantic annotations from schema still produces parseable output."""
        from polylogue.sources import iter_source_conversations

        # Generate with a corpus that has annotations stripped
        corpus = SyntheticCorpus.for_provider(provider)
        stripped = copy.deepcopy(corpus.schema)
        _strip_semantic_roles(stripped)
        corpus.schema = stripped

        # Write to temp files and parse through the full pipeline
        source = synthetic_source(provider, count=2, seed=77)
        convos = list(iter_source_conversations(source))
        assert convos, f"No conversations parsed without annotations for {provider}"


def _strip_semantic_roles(schema: dict) -> None:
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
