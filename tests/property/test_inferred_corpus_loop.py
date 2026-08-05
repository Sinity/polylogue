"""The inferred manifest's supported specs must reach real archive convergence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from tests.infra.convergence_harness import (
    ConvergenceArchive,
    converge_convergence_archive,
    rich_convergence_pathology,
)
from tests.infra.inferred_corpus import (
    _SUPPORTED_SYNTHETIC_ANNOTATIONS,
    assert_inferred_corpus_convergence_handoff_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
)


def _spec_identity(spec: CorpusSpec) -> tuple[str, str, str | None]:
    return spec.provider, spec.package_version, spec.element_kind


def _generator_only_schema(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _generator_only_schema(child)
            for key, child in value.items()
            if not (isinstance(key, str) and key.startswith("x-") and key not in _SUPPORTED_SYNTHETIC_ANNOTATIONS)
        }
    if isinstance(value, list):
        return [_generator_only_schema(child) for child in value]
    return value


class _GeneratorOnlyRegistry:
    """Expose the generator-enforced schema view without catalog provenance."""

    def __init__(self, base: SchemaRegistry) -> None:
        self.base = base

    def list_providers(self) -> list[str]:
        return self.base.list_providers()

    def load_package_catalog(self, provider: str) -> object:
        return self.base.load_package_catalog(provider)

    def get_element_schema(self, provider: str, *, version: str = "default", element_kind: str | None = None) -> object:
        schema = self.base.get_element_schema(provider, version=version, element_kind=element_kind)
        return cast(object, _generator_only_schema(schema))

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)


@pytest.mark.asyncio
async def test_inferred_manifest_supported_specs_ingest_and_converge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = compile_inferred_corpus_manifest(
        registry=_GeneratorOnlyRegistry(SchemaRegistry(storage_root=SCHEMA_DIR))  # type: ignore[arg-type]
    )
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)
    assert handoff.specs

    source_root = tmp_path / "inferred-corpus"
    written = SyntheticCorpus.write_specs_artifacts(handoff.specs, source_root, prefix="inferred")
    expected_specs = {_spec_identity(spec) for spec in manifest.supported_specs}
    generated_specs = {
        (batch.batch.report.provider, batch.batch.report.package_version, batch.batch.report.element_kind)
        for batch in written
    }
    assert generated_specs == expected_specs
    assert len(written) == len(manifest.supported_specs)

    source_paths = tuple(path for batch in written for path in batch.files)
    sources = [
        Source(name=batch.batch.report.provider, path=path.relative_to(source_root))
        for batch in written
        for path in batch.files
    ]
    monkeypatch.chdir(source_root)
    result = await parse_sources_archive(tmp_path / "archive", sources, parse_workers=1)

    expected_sessions = sum(spec.count for spec in handoff.specs)
    assert result.parse_failures == 0
    assert len(result.processed_ids) == expected_sessions
    archive = ConvergenceArchive(
        root=tmp_path / "archive",
        pathology=rich_convergence_pathology(),
        source_paths=source_paths,
        session_ids=tuple(sorted(result.processed_ids)),
    )
    states = converge_convergence_archive(archive)

    assert set(states) == set(result.processed_ids)
    assert all(state.converged for state in states.values())
