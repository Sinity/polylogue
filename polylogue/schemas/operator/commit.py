"""Commit a real full-corpus schema generation into registered packages.

``generate_provider_schema``/``infer_schema`` (what ``devtools lab schema
generate`` calls) only ever return a ``GenerationResult`` -- they never call
``persist_generated_provider_bundle``, the function that actually writes
``polylogue/schemas/providers/<provider>/versions/...`` via
``SchemaRegistry.replace_provider_packages``. That write path exists only in
``generate_all_schemas``, which had zero CLI wiring before this module
(polylogue-k45pq): the documented "correct entry point" for regenerating
committed schema packages silently no-op'd on the committed files.

This module is the real, persisting entry point. It wraps
``generate_all_schemas`` (so writes go through the same monotonic-merge
protection ``SchemaRegistry.replace_provider_packages`` already enforces --
see ``tests/unit/schemas/test_promotion_monotonicity.py``) and adds a
before/after report: which package versions are new, changed, or unchanged,
and whether any previously-committed leaf type was lost or narrowed (using
``polylogue.schemas.type_narrowing``, extracted from the same test file's
ad hoc ``_types_by_path`` check so the production commit path and its test
coverage share one implementation).

Deliberately separate from ``promote_schema_cluster``
(``polylogue.schemas.operator.inference``): that function promotes a single
evidence *cluster* (from ``generate --cluster`` mode) into one registered
package version -- a narrow, single-version operation. This module performs
a full-corpus, potentially multi-version *replace* across every version
``generate_all_schemas`` produces for a provider. Different shapes; neither
supersedes the other.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from polylogue.core.json import JSONDocument
from polylogue.maintenance.schema_inference_gate import (
    validate_schema_inference_gate_receipt,
)
from polylogue.paths import archive_root as default_archive_root
from polylogue.schemas.generation.models import GenerationResult
from polylogue.schemas.generation.workflow import generate_all_schemas
from polylogue.schemas.operator.inference import privacy_config_from_payload
from polylogue.schemas.operator.models import SchemaCommitRequest, SchemaCommitResult, SchemaVersionCommitReport
from polylogue.schemas.operator.receipt import (
    SCHEMA_INFERENCE_HANDOFF_FILENAME,
    SchemaInferenceReceipt,
    build_schema_inference_receipt,
    load_schema_inference_receipt,
    write_schema_inference_receipt,
)
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.runtime_registry import canonical_schema_provider
from polylogue.schemas.type_narrowing import added_paths, narrowed_paths
from polylogue.storage.archive_identity import ArchiveLocation


def _element_schemas_by_kind(
    registry: SchemaRegistry, provider_token: str, version: str, element_kinds: tuple[str, ...]
) -> dict[str, JSONDocument | None]:
    return {
        kind: registry.get_element_schema(provider_token, version=version, element_kind=kind) for kind in element_kinds
    }


def _accepted_gate_receipt_digest(path: Path | None, *, archive_root: Path) -> str:
    if path is None:
        raise ValueError("schema commit requires an accepted schema-inference gate receipt path")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unable to read schema-inference gate receipt {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("schema-inference gate receipt must be a JSON object")
    return cast(
        str,
        validate_schema_inference_gate_receipt(
            cast(Mapping[str, object], payload),
            archive_root=archive_root,
        ),
    )


def _target_archive_location(request: SchemaCommitRequest) -> ArchiveLocation:
    configured_root = request.archive_root or default_archive_root()
    location = ArchiveLocation.resolve(configured_root)
    expected_db_path = location.active_index_path.resolve(strict=False)
    if request.db_path is not None and request.db_path.resolve(strict=False) != expected_db_path:
        raise ValueError(
            "schema commit db_path must identify the active index of the configured archive; "
            f"expected={expected_db_path}, actual={request.db_path.resolve(strict=False)}"
        )
    return location


def _commit_into(request: SchemaCommitRequest, output_dir: Path) -> SchemaCommitResult:
    provider_token = str(canonical_schema_provider(request.provider))
    output_dir = output_dir.absolute()
    handoff_path = output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME
    existing_handoff = load_schema_inference_receipt(handoff_path) if handoff_path.exists() else None
    archive_location = _target_archive_location(request)
    gate_receipt_digest = _accepted_gate_receipt_digest(
        request.schema_inference_gate_receipt_path,
        archive_root=archive_location.configured_root,
    )
    if existing_handoff is not None and existing_handoff.gate_receipt_digest != gate_receipt_digest:
        raise ValueError(
            "existing schema inference handoff was produced from a different gate receipt; "
            "regenerate the handoff from the accepted gate before committing"
        )

    registry_before = SchemaRegistry(storage_root=output_dir)
    # The bundled registry is a read fallback, not the prior state of this
    # commit's output directory. Compare against local persisted packages only.
    catalog_before = registry_before._load_local_catalog(provider_token)
    before_versions = {package.version for package in catalog_before.packages} if catalog_before is not None else set()
    before_schemas: dict[str, dict[str, JSONDocument | None]] = {}
    if catalog_before is not None:
        for package in catalog_before.packages:
            element_kinds = tuple(element.element_kind for element in package.elements)
            before_schemas[package.version] = _element_schemas_by_kind(
                registry_before, provider_token, package.version, element_kinds
            )

    generation_results = generate_all_schemas(
        output_dir,
        db_path=archive_location.active_index_path,
        providers=[request.provider],
        max_samples=request.max_samples,
        privacy_config=privacy_config_from_payload(request.privacy_config),
        full_corpus=request.full_corpus,
    )
    generation = (
        generation_results[0]
        if generation_results
        else GenerationResult(
            provider=request.provider, schema=None, sample_count=0, error="No generation result produced"
        )
    )

    version_reports: list[SchemaVersionCommitReport] = []
    handoff: SchemaInferenceReceipt | None = None
    registry_after: SchemaRegistry | None = None
    if generation.success:
        registry_after = SchemaRegistry(storage_root=output_dir)
        catalog_after = registry_after.load_package_catalog(provider_token)
        if catalog_after is not None:
            for package in catalog_after.packages:
                element_kinds = tuple(element.element_kind for element in package.elements)
                after_schemas = _element_schemas_by_kind(registry_after, provider_token, package.version, element_kinds)
                prior_schemas = before_schemas.get(package.version, {})

                version_narrowed: list[str] = []
                version_added: list[str] = []
                for element_kind, after_schema in after_schemas.items():
                    prior_schema = prior_schemas.get(element_kind)
                    version_narrowed.extend(
                        f"{element_kind}{path or ':$root'}" for path in narrowed_paths(prior_schema, after_schema)
                    )
                    version_added.extend(
                        f"{element_kind}{path or ':$root'}" for path in added_paths(prior_schema, after_schema)
                    )

                if package.version not in before_versions:
                    status = "new"
                elif not version_narrowed and not version_added:
                    # Structurally identical to the prior commit -- ignore
                    # incidental bookkeeping churn (e.g. a fresh
                    # x-polylogue-registered-at timestamp) that isn't a real
                    # type-level change.
                    status = "unchanged"
                else:
                    status = "changed"

                version_reports.append(
                    SchemaVersionCommitReport(
                        version=package.version,
                        status=status,
                        sample_count=package.sample_count,
                        narrowed_paths=tuple(version_narrowed),
                        added_paths=tuple(version_added),
                    )
                )

    if generation.success:
        if registry_after is None:
            raise AssertionError("successful schema generation did not produce a persisted registry")
        provider_handoff = build_schema_inference_receipt(
            registry_after,
            provider=provider_token,
            gate_receipt_digest=gate_receipt_digest,
        )
        handoff = existing_handoff.merged_with(provider_handoff) if existing_handoff is not None else provider_handoff
        write_schema_inference_receipt(handoff, handoff_path)

    return SchemaCommitResult(
        provider=request.provider,
        generation=generation,
        versions=tuple(version_reports),
        dry_run=request.dry_run,
        handoff=handoff,
        handoff_path=handoff_path if generation.success else None,
    )


def commit_provider_schema(request: SchemaCommitRequest) -> SchemaCommitResult:
    """Generate a provider's full schema for real and persist it, or preview it.

    With ``request.dry_run`` set, the generation runs against a scratch copy
    of the provider's current committed directory (so the real registry
    monotonic-merge/carry-forward behavior in
    ``SchemaRegistry.replace_provider_packages`` is exercised faithfully) and
    the scratch copy is discarded -- ``request.output_dir`` is never touched.
    """
    if not request.dry_run:
        return _commit_into(request, request.output_dir)

    provider_token = str(canonical_schema_provider(request.provider))
    with tempfile.TemporaryDirectory(prefix="polylogue-schema-commit-dry-run-") as tmp_name:
        staging_root = Path(tmp_name) / "providers"
        staging_root.mkdir(parents=True, exist_ok=True)
        committed_provider_dir = request.output_dir / provider_token
        if committed_provider_dir.exists():
            shutil.copytree(committed_provider_dir, staging_root / provider_token)
        handoff_path = request.output_dir / SCHEMA_INFERENCE_HANDOFF_FILENAME
        if handoff_path.exists():
            shutil.copy2(handoff_path, staging_root / SCHEMA_INFERENCE_HANDOFF_FILENAME)
        result = _commit_into(request, staging_root)
        return SchemaCommitResult(
            provider=result.provider,
            generation=result.generation,
            versions=result.versions,
            dry_run=True,
            handoff=result.handoff,
            handoff_path=None,
        )


__all__ = ["SchemaCommitRequest", "SchemaCommitResult", "SchemaVersionCommitReport", "commit_provider_schema"]
