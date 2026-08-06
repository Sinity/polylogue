"""``commit_provider_schema`` -- the real, persisting full-corpus commit path.

``devtools lab schema generate`` (``generate_provider_schema``/``infer_schema``)
never writes to ``polylogue/schemas/providers/`` -- only
``generate_all_schemas`` does, and it had zero CLI wiring before
``polylogue.schemas.operator.commit`` (polylogue-k45pq). These tests prove the
new command actually changes files on disk, not merely that a function was
called: every assertion below reads back real gzip/JSON files written by
``SchemaRegistry.replace_provider_packages`` under a real ``tmp_path``, using
a fictional provider token so nothing here can read or write the repo's real
committed ``polylogue/schemas/providers/`` tree. The real bundled ``chatgpt``
wire format is used for campaign execution.

Only ``_build_provider_bundle`` (the sample-observation step) is mocked, the
same seam ``tests/unit/core/test_schema_generation.py`` uses for
``generate_all_schemas`` -- the persistence path under test
(``generate_all_schemas`` -> ``persist_generated_provider_bundle`` ->
``SchemaRegistry.replace_provider_packages``) runs for real.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue.maintenance.schema_inference_gate import (
    run_schema_inference_gate,
    schema_inference_gate_receipt_digest,
    schema_inference_hard_gate_evidence_digest,
)
from polylogue.schemas.generation.models import GenerationResult
from polylogue.schemas.operator.commit import commit_provider_schema
from polylogue.schemas.operator.models import SchemaCommitRequest
from polylogue.schemas.operator.receipt import SCHEMA_INFERENCE_HANDOFF_FILENAME, load_schema_inference_receipt
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.tooling_models import ClusterManifest
from tests.infra.frozen_clock import FrozenClock
from tests.infra.inferred_corpus import compile_inferred_corpus_manifest
from tests.unit.maintenance.test_schema_inference_gate import _seed_archive

_PROVIDER = "chatgpt"


def _gate_receipt(output_dir: Path) -> Path:
    path = output_dir.parent / "schema-inference-gate-receipt.json"
    archive_root = output_dir.parent / "archive"
    if path.exists():
        return path
    if not (archive_root / "source.db").exists():
        _seed_archive(archive_root)
    ground_truth = archive_root.parent / f"{archive_root.name}-codex-ground-truth"
    run_schema_inference_gate(
        archive_root,
        receipt_path=path,
        ground_truth_roots={"codex-session": (ground_truth,)},
    )
    return path


def _request(
    output_dir: Path,
    *,
    dry_run: bool = False,
    gate_path: Path | None = None,
) -> SchemaCommitRequest:
    return SchemaCommitRequest(
        provider=_PROVIDER,
        output_dir=output_dir,
        archive_root=output_dir.parent / "archive",
        db_path=output_dir.parent / "archive" / "index.db",
        full_corpus=True,
        schema_inference_gate_receipt_path=gate_path or _gate_receipt(output_dir),
        dry_run=dry_run,
    )


def _bundle(
    *,
    version: str,
    schema: dict[str, Any],
    sample_count: int,
    element_kind: str = "session_document",
) -> SimpleNamespace:
    package = SchemaVersionPackage(
        provider=_PROVIDER,
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
        provider=_PROVIDER,
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
            provider=_PROVIDER,
            packages=[package],
            latest_version=version,
            default_version=version,
            recommended_version=version,
        ),
        package_schemas={version: {element_kind: schema}},
        manifest=ClusterManifest(provider=_PROVIDER, clusters=[], artifact_counts={}),
    )


def _read_element_schema(output_dir: Path, version: str, element_kind: str = "session_document") -> dict[str, Any]:
    path = output_dir / _PROVIDER / "versions" / version / "elements" / f"{element_kind}.schema.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return cast("dict[str, Any]", json.load(handle))


@pytest.mark.frozen_clock_modules("polylogue.maintenance.schema_inference_gate")
class TestCommitProviderSchemaWritesRealFiles:
    @pytest.fixture(autouse=True)
    def _freeze_gate_clock(self, frozen_clock: FrozenClock) -> None:
        pass

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
        gate_payload = json.loads(_gate_receipt(output_dir).read_text(encoding="utf-8"))
        assert commit_result.handoff.gate_receipt_digest == schema_inference_gate_receipt_digest(gate_payload)
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
            gate_receipt_path=_gate_receipt(output_dir),
            archive_root=output_dir.parent / "archive",
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
                gate_receipt_path=_gate_receipt(output_dir),
                archive_root=output_dir.parent / "archive",
            )

    def test_commit_rejects_receipt_for_archive_a_when_generation_targets_archive_b(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        request = replace(
            _request(output_dir),
            db_path=tmp_path / "archive-b" / "index.db",
        )
        bundle = _bundle(
            version="v1",
            schema={"type": "object", "properties": {"id": {"type": "string"}}},
            sample_count=5,
        )

        with patch("polylogue.schemas.generation.workflow._build_provider_bundle", return_value=bundle):
            with pytest.raises(ValueError, match="db_path must identify the active index"):
                commit_provider_schema(request)

    def test_commit_requires_an_accepted_gate_receipt(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        with pytest.raises(ValueError, match="accepted schema-inference gate receipt"):
            commit_provider_schema(SchemaCommitRequest(provider=_PROVIDER, output_dir=output_dir, full_corpus=True))

    def test_commit_rejects_a_minimal_or_mutated_pass_payload(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        receipt_path = _gate_receipt(output_dir)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload.pop("full_blob_hash_verification")
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="authoritative fields"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

        receipt_path.unlink()
        payload = json.loads(_gate_receipt(output_dir).read_text(encoding="utf-8"))
        payload["full_blob_hash_verification"]["passed"] = False
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="full_blob_hash_verification PASS"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

        receipt_path.unlink()
        payload = json.loads(_gate_receipt(output_dir).read_text(encoding="utf-8"))
        payload["archive_identity_digest"] = "0" * 64
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="archive identity"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

        receipt_path.unlink()
        payload = json.loads(_gate_receipt(output_dir).read_text(encoding="utf-8"))
        payload["receipt_nonce"] = "forged"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="receipt nonce"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

        receipt_path.unlink()
        payload = json.loads(_gate_receipt(output_dir).read_text(encoding="utf-8"))
        payload["generated_at"] = "2020-01-01T00:00:00+00:00"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="stale or from the future"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

    def test_commit_rejects_recomputed_forgery_of_live_gate_evidence(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "providers"
        receipt_path = _gate_receipt(output_dir)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["query_results"]["zero-surviving-quarantine"]["count"] = 99
        payload["hard_gate_evidence_digest"] = schema_inference_hard_gate_evidence_digest(
            payload["query_results"], payload["full_blob_hash_verification"]
        )
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="hard-gate query results changed"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

        receipt_path.unlink()
        payload = json.loads(_gate_receipt(output_dir).read_text(encoding="utf-8"))
        payload["full_blob_hash_verification"]["before_snapshot"]["digest"] = "0" * 64
        payload["full_blob_hash_verification"]["after_snapshot"]["digest"] = "0" * 64
        payload["hard_gate_evidence_digest"] = schema_inference_hard_gate_evidence_digest(
            payload["query_results"], payload["full_blob_hash_verification"]
        )
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="hard-gate evidence digest"):
            commit_provider_schema(_request(output_dir, gate_path=receipt_path))

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
                gate_receipt_path=_gate_receipt(output_dir),
                archive_root=output_dir.parent / "archive",
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
                archive_root=output_dir.parent / "archive",
                db_path=output_dir.parent / "archive" / "index.db",
                full_corpus=True,
                schema_inference_gate_receipt_path=_gate_receipt(output_dir),
            )
        )

        assert not commit_result.success
        assert not commit_result.versions
        assert not (output_dir / "not-a-real-provider-k45pq").exists()
