"""Commit writes real package files while generation is replaced at its boundary."""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue.schemas.generation.models import GenerationResult
from polylogue.schemas.operator import commit as commit_module
from polylogue.schemas.operator.commit import commit_provider_schema
from polylogue.schemas.operator.models import SchemaCommitRequest
from polylogue.schemas.operator.receipt import (
    SCHEMA_INFERENCE_HANDOFF_FILENAME,
    build_schema_inference_receipt,
    load_schema_inference_receipt,
    write_schema_inference_receipt,
)
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.tooling_models import ClusterManifest
from tests.infra.inferred_corpus import compile_inferred_corpus_manifest

_PROVIDER = "chatgpt"


def _request(
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> SchemaCommitRequest:
    return SchemaCommitRequest(
        provider=_PROVIDER,
        output_dir=output_dir,
        db_path=output_dir.parent / "archive" / "index.db",
        full_corpus=True,
        dry_run=dry_run,
    )


def _bundle(
    *,
    version: str,
    schema: dict[str, Any],
    sample_count: int,
    element_kind: str = "session_document",
    provider: str = _PROVIDER,
) -> SimpleNamespace:
    package = SchemaVersionPackage(
        provider=provider,
        version=version,
        anchor_kind=element_kind,
        default_element_kind=element_kind,
        first_seen="2026-08-01T00:00:00+00:00",
        last_seen="2026-08-01T00:00:00+00:00",
        bundle_scope_count=1,
        sample_count=sample_count,
        elements=[
            SchemaElementManifest(
                element_kind=element_kind,
                schema_file=f"{element_kind}.schema.json.gz",
                sample_count=sample_count,
                artifact_count=sample_count,
            )
        ],
    )
    result = GenerationResult(
        provider=provider,
        sample_count=sample_count,
        schema=schema,
        error=None,
        versions=[version],
        default_version=version,
        package_count=1,
        cluster_count=1,
    )
    return SimpleNamespace(
        result=result,
        catalog=SchemaPackageCatalog(
            provider=provider,
            packages=[package],
            latest_version=version,
            default_version=version,
            recommended_version=version,
        ),
        package_schemas={version: {element_kind: schema}},
        manifest=ClusterManifest(provider=provider, clusters=[], artifact_counts={}),
    )


def _read_element_schema(output_dir: Path, version: str, element_kind: str = "session_document") -> dict[str, Any]:
    path = output_dir / _PROVIDER / "versions" / version / "elements" / f"{element_kind}.schema.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return cast("dict[str, Any]", json.load(handle))


def test_commit_threads_selected_archive_location_to_generation(tmp_path: Path) -> None:
    """The committing production route must not discard durable-root authority."""
    from polylogue.storage.archive_identity import ArchiveLocation

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    index = archive_root / "index.db"
    index.touch()
    location = ArchiveLocation.resolve(archive_root)
    observed: list[object] = []

    def generate(*_args: object, **kwargs: object) -> list[GenerationResult]:
        observed.append(kwargs.get("archive_location"))
        return [GenerationResult(provider=_PROVIDER, schema=None, sample_count=0, error="No samples found")]

    with patch("polylogue.schemas.operator.commit.generate_all_schemas", side_effect=generate):
        result = commit_provider_schema(
            SchemaCommitRequest(
                provider=_PROVIDER,
                output_dir=tmp_path / "providers",
                db_path=index,
                archive_location=location,
                full_corpus=True,
            )
        )

    assert not result.success
    assert observed == [location]


class TestCommitProviderSchemaWritesRealFiles:
    def test_provider_finishing_during_generation_remains_in_handoff(self, tmp_path: Path) -> None:
        """A receipt loaded before generation must not erase a later provider commit."""
        output_dir = tmp_path / "providers"

        def build(provider: str, **_kwargs: object) -> SimpleNamespace:
            if provider == "chatgpt":
                second = commit_provider_schema(
                    SchemaCommitRequest(
                        provider="claude-ai",
                        output_dir=output_dir,
                        db_path=tmp_path / "archive" / "index.db",
                        full_corpus=True,
                    )
                )
                assert second.success
            return _bundle(
                provider=provider,
                version="v1",
                schema={"type": "object", "properties": {"id": {"type": "string"}}},
                sample_count=1,
            )

        with patch("polylogue.schemas.generation.workflow._build_provider_bundle", side_effect=build):
            first = commit_provider_schema(_request(output_dir))

        assert first.success
        handoff = load_schema_inference_receipt(output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME)
        assert {item.provider for item in handoff.input_manifests} == {"chatgpt", "claude-ai"}
        assert {item.provider for item in handoff.packages} == {"chatgpt", "claude-ai"}

    def test_same_provider_publication_keeps_receipt_with_tree(self, tmp_path: Path) -> None:
        """Releasing the tree lock before the receipt lets an older receipt win."""
        output_dir = tmp_path / "providers"
        first_waiting = Event()
        release_first = Event()
        second_started = Event()
        original_write = write_schema_inference_receipt
        writes = 0

        def write(*args: Any, **kwargs: Any) -> Any:
            nonlocal writes
            writes += 1
            if writes == 1:
                first_waiting.set()
                assert release_first.wait(5)
            return original_write(*args, **kwargs)

        def build(*_args: Any, **kwargs: Any) -> SimpleNamespace:
            count = 1 if kwargs["source_inputs"][0].root.name == "first" else 2
            return _bundle(
                version="v1",
                schema={"type": "object", "properties": {"id": {"type": "string"}}},
                sample_count=count,
            )

        first_request = replace(_request(output_dir), source_inputs=(SchemaSourceInput(_PROVIDER, tmp_path / "first"),))
        second_request = replace(
            _request(output_dir), source_inputs=(SchemaSourceInput(_PROVIDER, tmp_path / "second"),)
        )

        def second_commit() -> Any:
            second_started.set()
            return commit_provider_schema(second_request)

        with (
            patch.object(commit_module, "build_provider_bundle_from_sources", side_effect=build),
            patch.object(commit_module, "write_schema_inference_receipt", side_effect=write),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            first = executor.submit(commit_provider_schema, first_request)
            try:
                assert first_waiting.wait(5)
                second = executor.submit(second_commit)
                assert second_started.wait(5)
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.2)
            finally:
                release_first.set()
            assert first.result(timeout=5).success
            assert second.result(timeout=5).success

        actual = load_schema_inference_receipt(output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME)
        expected = build_schema_inference_receipt(SchemaRegistry(storage_root=output_dir), provider=_PROVIDER)
        assert actual.packages == expected.packages

    def test_new_provider_writes_catalog_and_element_files(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        bundle = _bundle(version="v1", schema=schema, sample_count=5)

        with patch("polylogue.schemas.generation.workflow._build_provider_bundle", return_value=bundle):
            commit_result = commit_provider_schema(_request(output_dir))

        assert commit_result.success
        assert not commit_result.dry_run
        # Real files on disk, not merely "a function returned a result".
        assert (output_dir / _PROVIDER / "catalog.json").exists()
        on_disk = _read_element_schema(output_dir, "v1")
        assert on_disk["properties"]["id"]["type"] == "string"

        assert len(commit_result.versions) == 1
        version_report = commit_result.versions[0]
        assert version_report.version == "v1"
        assert version_report.status == "new"
        assert version_report.sample_count == 5
        assert not version_report.narrowed_paths
        assert "session_document.id" in version_report.added_paths
        assert commit_result.handoff is not None
        assert commit_result.handoff.input_manifests[0].digest is None
        assert commit_result.handoff.input_manifests[0].unavailable_reason == "input_manifest_unavailable"
        assert commit_result.handoff_path == output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME
        assert load_schema_inference_receipt(commit_result.handoff_path) == commit_result.handoff
        assert commit_result.handoff.packages[0].element_hashes[0].element_kind == "session_document"

    def test_commit_to_registry_to_campaign_manifest_is_a_real_route(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        bundle = _bundle(
            version="v1",
            schema={"type": "object", "properties": {"id": {"type": "string"}}},
            sample_count=5,
        )
        with patch("polylogue.schemas.generation.workflow._build_provider_bundle", return_value=bundle):
            result = commit_provider_schema(_request(output_dir))

        assert result.handoff is not None
        registry = SchemaRegistry(storage_root=output_dir)
        manifest = compile_inferred_corpus_manifest(
            registry=registry,
            providers=(_PROVIDER,),
            package_receipt=result.handoff.to_payload(),
            campaign_mode=True,
        )
        assert manifest.receipt_state == "package_receipt_attached"
        assert len(manifest.entries) == 1
        entry = manifest.entries[0]
        assert entry.spec is not None
        assert entry.generator_schema is not None
        assert entry.key.provider == "chatgpt"

        element_path = output_dir / _PROVIDER / "versions" / "v1" / "elements" / "session_document.schema.json.gz"
        mutated_schema = _read_element_schema(output_dir, "v1")
        mutated_schema["title"] = "mutation"
        with gzip.open(element_path, "wt", encoding="utf-8") as handle:
            json.dump(mutated_schema, handle)
        with pytest.raises(ValueError, match="package/version/element hashes"):
            compile_inferred_corpus_manifest(
                registry=SchemaRegistry(storage_root=output_dir),
                providers=(_PROVIDER,),
                package_receipt=result.handoff.to_payload(),
                campaign_mode=True,
            )

    def test_registry_construct_rejection_remains_an_explicit_unsupported_entry(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        bundle = _bundle(
            version="v1",
            schema={"type": "object", "properties": {"id": {"type": "string", "enum": ["x"]}}},
            sample_count=5,
        )
        with patch("polylogue.schemas.generation.workflow._build_provider_bundle", return_value=bundle):
            result = commit_provider_schema(_request(output_dir))

        assert result.handoff is not None
        with pytest.raises(ValueError, match="no executable synthetic corpus selection"):
            compile_inferred_corpus_manifest(
                registry=SchemaRegistry(storage_root=output_dir),
                providers=(_PROVIDER,),
                package_receipt=result.handoff.to_payload(),
                campaign_mode=True,
            )
        manifest = compile_inferred_corpus_manifest(
            registry=SchemaRegistry(storage_root=output_dir),
            providers=(_PROVIDER,),
            package_receipt=result.handoff.to_payload(),
            campaign_mode=False,
        )
        entry = manifest.entries[0]
        assert entry.spec is None
        assert entry.unsupported is not None
        assert entry.unsupported.reason == "unsupported_json_schema_construct"
        assert "enum" in entry.unsupported.details

    def test_regeneration_with_new_field_reports_changed_and_added(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        first_schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=first_schema, sample_count=5),
        ):
            commit_provider_schema(_request(output_dir))

        second_schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "newly_observed": {"type": "boolean"}},
        }
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=second_schema, sample_count=9),
        ):
            commit_result = commit_provider_schema(_request(output_dir))

        assert commit_result.success
        version_report = commit_result.versions[0]
        assert version_report.status == "changed"
        assert version_report.sample_count == 9
        assert not version_report.narrowed_paths
        assert "session_document.newly_observed" in version_report.added_paths

        # The field really is on disk, not just in the in-memory report.
        on_disk = _read_element_schema(output_dir, "v1")
        assert on_disk["properties"]["newly_observed"]["type"] == "boolean"
        assert on_disk["properties"]["id"]["type"] == "string"

    def test_identical_regeneration_reports_unchanged(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=schema, sample_count=5),
        ):
            commit_provider_schema(_request(output_dir))
            commit_result = commit_provider_schema(_request(output_dir))

        assert commit_result.versions[0].status == "unchanged"

    def test_thin_regeneration_window_cannot_narrow_committed_union(self, tmp_path: Path) -> None:
        """End-to-end proof that the real commit path inherits
        ``SchemaRegistry.replace_provider_packages``'s monotonic-merge safety
        net (the ov5r/polylogue-46kg incident class): a second, thinner
        generation window that would -- if written directly -- narrow a
        previously-observed type union instead leaves the union intact on
        disk, and the commit report correctly finds zero narrowed paths.
        """
        output_dir = tmp_path / "providers"
        wide_schema = {"type": "object", "properties": {"timestamp": {"type": ["string", "number"]}}}
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=wide_schema, sample_count=100),
        ):
            commit_provider_schema(_request(output_dir))

        thin_schema = {"type": "object", "properties": {"timestamp": {"type": "string"}}}
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=thin_schema, sample_count=3),
        ):
            commit_result = commit_provider_schema(_request(output_dir))

        assert not commit_result.narrowed
        assert not commit_result.versions[0].narrowed_paths
        on_disk = _read_element_schema(output_dir, "v1")
        assert set(on_disk["properties"]["timestamp"]["type"]) == {"string", "number"}

    def test_dry_run_does_not_touch_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=schema, sample_count=5),
        ):
            commit_provider_schema(_request(output_dir))

        catalog_before_bytes = (output_dir / _PROVIDER / "catalog.json").read_bytes()
        handoff_before_bytes = (output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME).read_bytes()

        second_schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "would_be_added": {"type": "boolean"}},
        }
        with patch(
            "polylogue.schemas.generation.workflow._build_provider_bundle",
            return_value=_bundle(version="v1", schema=second_schema, sample_count=9),
        ):
            commit_result = commit_provider_schema(_request(output_dir, dry_run=True))

        assert commit_result.dry_run
        assert commit_result.versions[0].status == "changed"
        assert "session_document.would_be_added" in commit_result.versions[0].added_paths
        # The real committed directory was never touched.
        assert (output_dir / _PROVIDER / "catalog.json").read_bytes() == catalog_before_bytes
        assert (output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME).read_bytes() == handoff_before_bytes
        on_disk = _read_element_schema(output_dir, "v1")
        assert "would_be_added" not in on_disk["properties"]

    def test_failed_generation_reports_no_success_and_no_versions(self, tmp_path: Path) -> None:
        # An unrecognized provider token fails inside `_build_provider_bundle`
        # itself (unknown-provider guard) with no DB access required -- no
        # mocking needed to exercise the failure path for real.
        output_dir = tmp_path / "providers"
        commit_result = commit_provider_schema(
            SchemaCommitRequest(
                provider="not-a-real-provider-k45pq",
                output_dir=output_dir,
                db_path=output_dir.parent / "archive" / "index.db",
                full_corpus=True,
            )
        )

        assert not commit_result.success
        assert not commit_result.versions
        assert not (output_dir / "not-a-real-provider-k45pq").exists()
