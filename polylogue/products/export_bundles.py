"""Versioned archive-product export bundle contracts and writer."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from polylogue.config import Config
from polylogue.core.json import JSONDocument, dumps, require_json_document
from polylogue.products.archive import (
    ArchiveProductUnavailableError,
    DaySessionSummaryProduct,
    ProviderAnalyticsProduct,
    SessionEnrichmentProduct,
    SessionPhaseProduct,
    SessionProfileProduct,
    SessionTagRollupProduct,
    SessionWorkEventProduct,
    WeekSessionSummaryProduct,
    WorkThreadProduct,
)
from polylogue.products.archive_models import ARCHIVE_PRODUCT_CONTRACT_VERSION, ArchiveProductModel
from polylogue.products.readiness import ProductReadinessQuery, ProductReadinessReport
from polylogue.products.registry import PRODUCT_REGISTRY, ProductQueryError, ProductType, fetch_products_async
from polylogue.version import VERSION_INFO

ProductExportFormat = Literal["jsonl"]
PRODUCT_EXPORT_BUNDLE_VERSION = 1
DEFAULT_EXPORT_PRODUCTS: tuple[str, ...] = (
    "session_profiles",
    "session_enrichments",
    "session_work_events",
    "session_phases",
    "work_threads",
    "session_tag_rollups",
    "day_session_summaries",
    "week_session_summaries",
    "provider_analytics",
)
_PRODUCT_MODEL_BY_NAME: dict[str, type[ArchiveProductModel]] = {
    "session_profiles": SessionProfileProduct,
    "session_enrichments": SessionEnrichmentProduct,
    "session_work_events": SessionWorkEventProduct,
    "session_phases": SessionPhaseProduct,
    "work_threads": WorkThreadProduct,
    "session_tag_rollups": SessionTagRollupProduct,
    "day_session_summaries": DaySessionSummaryProduct,
    "week_session_summaries": WeekSessionSummaryProduct,
    "provider_analytics": ProviderAnalyticsProduct,
}
_PRODUCT_ALIASES = {
    **{name.replace("_", "-"): name for name in DEFAULT_EXPORT_PRODUCTS},
    **{
        product_type.resolved_cli_command_name: name
        for name, product_type in PRODUCT_REGISTRY.items()
        if name in DEFAULT_EXPORT_PRODUCTS
    },
}


class ProductExportBundleError(RuntimeError):
    """Raised when a product export bundle cannot be written."""


class ProductExportBundleRequest(ArchiveProductModel):
    output_path: Path
    products: tuple[str, ...] = ()
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    output_format: ProductExportFormat = "jsonl"
    overwrite: bool = False
    include_readme: bool = True


class ProductExportFileSummary(ArchiveProductModel):
    product_name: str
    file: str
    schema_file: str
    row_count: int = 0
    readiness_verdict: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class ProductExportBundleManifest(ArchiveProductModel):
    bundle_version: int = PRODUCT_EXPORT_BUNDLE_VERSION
    product_contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    generated_at: str
    polylogue_version: str
    git_revision: str | None = None
    git_dirty: bool = False
    archive_root: str
    database_path: str
    output_format: ProductExportFormat = "jsonl"
    query: dict[str, str | tuple[str, ...] | None]
    products: tuple[ProductExportFileSummary, ...] = ()
    warnings: tuple[str, ...] = ()


class ProductExportBundleResult(ArchiveProductModel):
    output_path: Path
    manifest_path: Path
    coverage_path: Path
    manifest: ProductExportBundleManifest


class ProductExportOperations(Protocol):
    async def get_product_readiness_report(
        self,
        query: ProductReadinessQuery | None = None,
    ) -> ProductReadinessReport: ...


def normalize_export_product_name(value: str) -> str:
    normalized = value.strip().replace("-", "_")
    if normalized in DEFAULT_EXPORT_PRODUCTS:
        return normalized
    alias = _PRODUCT_ALIASES.get(value.strip()) or _PRODUCT_ALIASES.get(value.strip().replace("_", "-"))
    if alias is not None:
        return alias
    raise ProductExportBundleError(f"Unknown export product: {value}")


def _selected_product_names(products: Sequence[str]) -> tuple[str, ...]:
    if not products:
        return DEFAULT_EXPORT_PRODUCTS
    selected: list[str] = []
    for product in products:
        name = normalize_export_product_name(product)
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def _product_path(product_name: str) -> str:
    return f"products/{product_name}.jsonl"


def _schema_path(product_name: str) -> str:
    return f"schemas/{product_name}.schema.json"


def _query_kwargs(
    product_type: ProductType, request: ProductExportBundleRequest
) -> tuple[dict[str, object], tuple[str, ...]]:
    query_model = product_type.query_model
    if query_model is None:
        return {}, (f"{product_type.name} has no query model and cannot be fetched",)
    fields = set(query_model.model_fields)
    kwargs: dict[str, object] = {}
    warnings: list[str] = []
    if "limit" in fields:
        kwargs["limit"] = None
    if "offset" in fields:
        kwargs["offset"] = 0
    for key, value in (("provider", request.provider), ("since", request.since), ("until", request.until)):
        if value is None:
            continue
        if key in fields:
            kwargs[key] = value
        else:
            warnings.append(f"{product_type.name} does not support {key} bounds")
    return kwargs, tuple(warnings)


def _json_schema_document(product_name: str) -> JSONDocument:
    model = _PRODUCT_MODEL_BY_NAME[product_name]
    schema = require_json_document(model.model_json_schema(), context=f"{product_name} JSON schema")
    return {
        "product_name": product_name,
        "model_name": model.__name__,
        "contract_version": ARCHIVE_PRODUCT_CONTRACT_VERSION,
        "schema": schema,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(dumps(payload) + "\n", encoding="utf-8")


def _write_product_jsonl(path: Path, items: Sequence[ArchiveProductModel]) -> None:
    lines = [item.model_dump_json(exclude_none=True) for item in items]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _write_readme(path: Path, manifest: ProductExportBundleManifest) -> None:
    lines = [
        "# Polylogue Product Export Bundle",
        "",
        f"- Generated: `{manifest.generated_at}`",
        f"- Polylogue: `{manifest.polylogue_version}`",
        f"- Product contract: `{manifest.product_contract_version}`",
        f"- Products: `{len(manifest.products)}`",
        "",
        "| Product | Rows | Readiness | File |",
        "| --- | ---: | --- | --- |",
    ]
    for product in manifest.products:
        lines.append(
            f"| `{product.product_name}` | {product.row_count} | `{product.readiness_verdict or '-'}` | `{product.file}` |"
        )
    if manifest.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_target(request: ProductExportBundleRequest) -> Path:
    target = request.output_path
    if target.exists() and not request.overwrite:
        raise ProductExportBundleError(f"Export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    tmp_target.mkdir(parents=False)
    (tmp_target / "products").mkdir()
    (tmp_target / "schemas").mkdir()
    return tmp_target


def _publish_target(tmp_target: Path, request: ProductExportBundleRequest) -> None:
    target = request.output_path
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    tmp_target.replace(target)


async def export_product_bundle(
    operations: ProductExportOperations,
    config: Config,
    request: ProductExportBundleRequest,
) -> ProductExportBundleResult:
    selected_products = _selected_product_names(request.products)
    readiness = await operations.get_product_readiness_report(
        ProductReadinessQuery(
            products=selected_products,
            provider=request.provider,
            since=request.since,
            until=request.until,
        )
    )
    readiness_by_name = {entry.product_name: entry for entry in readiness.products}
    tmp_target = _prepare_target(request)
    summaries: list[ProductExportFileSummary] = []
    bundle_warnings: list[str] = []
    try:
        for product_name in selected_products:
            product_type = PRODUCT_REGISTRY[product_name]
            product_file = _product_path(product_name)
            schema_file = _schema_path(product_name)
            kwargs, warnings = _query_kwargs(product_type, request)
            errors: list[str] = []
            items: list[ArchiveProductModel] = []
            try:
                items = await fetch_products_async(product_type, operations, **kwargs)
            except (ArchiveProductUnavailableError, ProductQueryError) as exc:
                errors.append(str(exc))
            _write_product_jsonl(tmp_target / product_file, items)
            _write_json(tmp_target / schema_file, _json_schema_document(product_name))
            readiness_entry = readiness_by_name.get(product_name)
            summaries.append(
                ProductExportFileSummary(
                    product_name=product_name,
                    file=product_file,
                    schema_file=schema_file,
                    row_count=len(items),
                    readiness_verdict=readiness_entry.verdict if readiness_entry is not None else None,
                    warnings=warnings,
                    errors=tuple(errors),
                )
            )
            bundle_warnings.extend(f"{product_name}: {warning}" for warning in warnings)
            bundle_warnings.extend(f"{product_name}: {error}" for error in errors)

        manifest = ProductExportBundleManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            polylogue_version=VERSION_INFO.full,
            git_revision=VERSION_INFO.commit,
            git_dirty=VERSION_INFO.dirty,
            archive_root=str(config.archive_root),
            database_path=str(config.db_path),
            output_format=request.output_format,
            query={
                "products": selected_products,
                "provider": request.provider,
                "since": request.since,
                "until": request.until,
            },
            products=tuple(summaries),
            warnings=tuple(bundle_warnings),
        )
        _write_json(tmp_target / "manifest.json", manifest.model_dump(mode="json"))
        _write_json(tmp_target / "coverage.json", readiness.model_dump(mode="json"))
        if request.include_readme:
            _write_readme(tmp_target / "README.md", manifest)
        _publish_target(tmp_target, request)
    except Exception:
        shutil.rmtree(tmp_target, ignore_errors=True)
        raise

    return ProductExportBundleResult(
        output_path=request.output_path,
        manifest_path=request.output_path / "manifest.json",
        coverage_path=request.output_path / "coverage.json",
        manifest=manifest,
    )


__all__ = [
    "DEFAULT_EXPORT_PRODUCTS",
    "PRODUCT_EXPORT_BUNDLE_VERSION",
    "ProductExportBundleError",
    "ProductExportBundleManifest",
    "ProductExportBundleRequest",
    "ProductExportBundleResult",
    "ProductExportFileSummary",
    "ProductExportFormat",
    "export_product_bundle",
    "normalize_export_product_name",
]
