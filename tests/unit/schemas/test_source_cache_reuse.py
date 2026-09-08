"""Cache upgrades preserve evidence only when its semantics remain sufficient."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from polylogue.schemas import source_inference as source
from polylogue.schemas.field_stats import detection
from polylogue.schemas.source_recipe import SourceEvidenceRecipe


def write_source(root: Path, name: str, *, extra: dict[str, object] | None = None, width: int = 0) -> None:
    records = [
        {
            "type": "user",
            "sessionId": name,
            "message": {"role": "user", "content": "synthetic"},
            **(extra or {}),
            **({f"optional_field_{index}": index} if width else {}),
        }
        for index in range(width or 1)
    ]
    (root / f"{name}.jsonl").write_text("\n".join(json.dumps(record) for record in records))


def run(root: Path, cache: Path) -> source.SourceInferenceResult:
    return source.infer_sources((source.SchemaSourceInput("claude-code", root),), cache_path=cache, max_workers=1)


@pytest.fixture
def local_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Policy-change tests need the real collector to see each run's selected parameters.
    monkeypatch.setattr(source, "ProcessPoolExecutor", ThreadPoolExecutor)


def test_implementation_provenance_does_not_invalidate_semantically_unchanged_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_workers: None,
) -> None:
    """Tying the cache key to reporting/import changes repeats both source passes."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "session")
    monkeypatch.setattr(source, "_source_recipe_fingerprint", lambda: "a" * 64)
    cold = run(root, tmp_path / "cache.sqlite")
    monkeypatch.setattr(source, "_source_recipe_fingerprint", lambda: "b" * 64)
    warm = run(root, tmp_path / "cache.sqlite")
    assert warm.cache_phase_hits == {"structure": 1, "statistics": 1}
    assert warm.cache_misses == 0
    assert warm.evidence_by_element == cold.evidence_by_element
    assert warm.input_manifest_digest == cold.input_manifest_digest
    assert warm.recipe["implementation_fingerprint"] != cold.recipe["implementation_fingerprint"]


@pytest.mark.parametrize(
    ("recipe", "hits", "misses"),
    [
        (SourceEvidenceRecipe(statistics_revision=2), {"structure": 1}, {"statistics": 1}),
        (SourceEvidenceRecipe(structure_revision=2), {"statistics": 1}, {"structure": 1}),
        (SourceEvidenceRecipe(admission_revision=2), {}, {"structure": 1, "statistics": 1}),
    ],
)
def test_semantic_revision_invalidates_only_the_dependent_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_workers: None,
    recipe: SourceEvidenceRecipe,
    hits: dict[str, int],
    misses: dict[str, int],
) -> None:
    """A statistics change must neither reuse old counters nor discard structural evidence."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "session")
    cache = tmp_path / "cache.sqlite"
    cold = run(root, cache)
    monkeypatch.setattr(source, "SourceEvidenceRecipe", lambda: recipe)
    upgraded = run(root, cache)
    assert upgraded.cache_phase_hits == hits
    assert upgraded.cache_phase_misses == misses
    assert upgraded.evidence_by_element == cold.evidence_by_element


def test_key_limit_upgrade_recovers_collapsed_fields_and_reuses_other_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_workers: None,
) -> None:
    """A wildcard summary cannot supply the erased field names or separate field counters."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "wide", width=160)
    write_source(root, "narrow")
    cache = tmp_path / "cache.sqlite"
    monkeypatch.setattr(detection, "_HIGH_CARDINALITY_KEY_THRESHOLD", 128)
    old = run(root, cache)
    monkeypatch.setattr(detection, "_HIGH_CARDINALITY_KEY_THRESHOLD", 256)
    upgraded = run(root, cache)
    fresh = run(root, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_hits == {"structure": 1}
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 2}
    assert upgraded.evidence_by_element == fresh.evidence_by_element
    assert upgraded.evidence_by_element != old.evidence_by_element
    assert "optional_field_159" in json.dumps(upgraded.evidence_by_element)


def test_new_dynamic_path_only_reprocesses_sources_containing_that_path(
    tmp_path: Path,
    local_workers: None,
) -> None:
    """A global normalization-map key must not invalidate unrelated source contributions."""
    root = tmp_path / "inputs"
    root.mkdir()
    write_source(root, "existing")
    cache = tmp_path / "cache.sqlite"
    run(root, cache)
    write_source(root, "new", extra={"extra": {"question?": {"answer": "synthetic"}}})
    upgraded = run(root, cache)
    fresh = run(root, tmp_path / "fresh.sqlite")
    assert upgraded.cache_phase_hits == {"structure": 1, "statistics": 1}
    assert upgraded.cache_phase_misses == {"structure": 1, "statistics": 1}
    assert upgraded.evidence_by_element == fresh.evidence_by_element
