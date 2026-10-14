"""Actual main-session populations drive shared-driver corpus selection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from devtools.fresh_build_bench.corpus import corpus_from_synthetic, verify_manifest
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.composition import SyntheticComposition
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload


def _composition(registry: SchemaRegistry) -> SyntheticComposition:
    return SyntheticComposition(
        SyntheticCorpus.from_selection(select_synthetic_schema(provider, registry_factory=lambda: registry))
        for provider in ("claude-code", "codex", "chatgpt")
    )


def test_default_weights_and_seeded_choices_use_actual_main_session_counts(
    qualified_schema_registry: SchemaRegistry,
) -> None:
    composition = _composition(qualified_schema_registry)
    # Independent authored input inventory: two native files each, one export.
    assert composition.populations == {"claude-code": 2, "codex": 2, "chatgpt": 1}
    assert composition.weights == {"claude-code": 0.4, "codex": 0.4, "chatgpt": 0.2}

    def choices(seed: int) -> list[tuple[str, int]]:
        return [(corpus.provider, draw_seed) for corpus, draw_seed in composition.iter_choices(count=4096, seed=seed)]

    first = choices(19)
    assert first == choices(19) and first != choices(20)
    observed = Counter(provider for provider, _ in first)
    assert set(observed) == set(composition.populations)
    assert sum(abs(observed[provider] / 4096 - weight) for provider, weight in composition.weights.items()) < 0.06


def test_composed_fresh_build_corpus_writes_same_driver_bytes_and_selection_receipt(
    qualified_schema_registry: SchemaRegistry,
) -> None:
    composition = _composition(qualified_schema_registry)
    expected: list[tuple[str, bytes]] = []
    for corpus, draw_seed in composition.iter_choices(count=24, seed=23):
        sessions = corpus.iter_sessions(count=1, seed=draw_seed)
        try:
            for session in sessions:
                expected.append((corpus.provider, b"".join(session.chunks)))
                assert not list(session.adjuncts)
        finally:
            sessions.close()
    with TemporaryDirectory(prefix="polylogue-composition-", dir="/realm/tmp/work") as temporary:
        root = Path(temporary) / "corpus"
        manifest = corpus_from_synthetic(root, corpus=composition, count=24, seed=23)
        verify_manifest(root, manifest)
        parameters = manifest["parameters"]
        assert parameters["provider_populations"] == composition.populations
        assert parameters["provider_weights"] == composition.weights
        assert parameters["provider_profile_ids"] == composition.profile_ids
        assert parameters["provider_selection_counts"] == Counter(provider for provider, _ in expected)
        actual = {(row[3], (root / row[0]).read_bytes()) for row in manifest["files"]}
        assert actual == set(expected) and manifest["file_count"] == len(expected) == 24
        for provider, raw in actual:
            payload = json.loads(raw) if provider == "chatgpt" else [json.loads(line) for line in raw.splitlines()]
            parsed = parse_payload(provider, payload, "neutral")
            assert parsed and sum(len(session.messages) for session in parsed) > 0
        assert all(not corpus._generation_state.driver_active for corpus in composition.corpora)


def test_composition_refuses_unmeasured_main_session_denominator(
    qualified_schema_registry: SchemaRegistry,
) -> None:
    corpus = _composition(qualified_schema_registry).corpora[0]
    profile = dict(corpus.workload_profile or {})
    provenance = profile.get("provenance")
    assert isinstance(provenance, dict)
    profile["provenance"] = {key: value for key, value in provenance.items() if key != "main_session_count"}
    corpus.workload_profile = profile
    with pytest.raises(ValueError, match="measured main-session counts"):
        SyntheticComposition([corpus])
