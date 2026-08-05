"""The inferred manifest's supported specs must reach real archive convergence."""

from __future__ import annotations

from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from tests.infra.inferred_corpus import (
    assert_inferred_corpus_convergence_handoff_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
)


def test_persisted_manifest_handoff_excludes_unsupported_catalog_entries() -> None:
    manifest = compile_inferred_corpus_manifest(registry=SchemaRegistry(storage_root=SCHEMA_DIR))
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)
    assert handoff.specs == manifest.supported_specs == ()
    assert manifest.unsupported_records
