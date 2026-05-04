"""Versioned archive-insight export bundle contracts and writer."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from polylogue.config import Config
from polylogue.core.json import JSONDocument, dumps, require_json_document
from polylogue.errors import PolylogueError
from polylogue.insights.archive import (
    ArchiveInsightUnavailableError,
    DaySessionSummaryInsight,
    ProviderAnalyticsInsight,
    SessionEnrichmentInsight,
    SessionPhaseInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
    SessionWorkEventInsight,
    WeekSessionSummaryInsight,
    WorkThreadInsight,
)
from polylogue.insights.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION, ArchiveInsightModel
from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport
from polylogue.insights.registry import INSIGHT_REGISTRY, InsightQueryError, InsightType, fetch_insights_async
from polylogue.version import VERSION_INFO

InsightExportFormat = Literal["jsonl"]
INSIGHT_EXPORT_BUNDLE_VERSION = 1
DEFAULT_EXPORT_INSIGHTS: tuple[str, ...] = (
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
_INSIGHT_MODEL_BY_NAME: dict[str, type[ArchiveInsightModel]] = {
    "session_profiles": SessionProfileInsight,
    "session_enrichments": SessionEnrichmentInsight,
    "session_work_events": SessionWorkEventInsight,
    "session_phases": SessionPhaseInsight,
    "work_threads": WorkThreadInsight,
    "session_tag_rollups": SessionTagRollupInsight,
    "day_session_summaries": DaySessionSummaryInsight,
    "week_session_summaries": WeekSessionSummaryInsight,
    "provider_analytics": ProviderAnalyticsInsight,
}
_INSIGHT_ALIASES = {
    **{name.replace("_", "-"): name for name in DEFAULT_EXPORT_INSIGHTS},
    **{
        insight_type.resolved_cli_command_name: name
        for name, insight_type in INSIGHT_REGISTRY.items()
        if name in DEFAULT_EXPORT_INSIGHTS
    },
}


class InsightExportBundleError(PolylogueError):
    """Raised when an insight export bundle cannot be written."""


class InsightExportBundleRequest(ArchiveInsightModel):
    output_path: Path
    insights: tuple[str, ...] = ()
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    output_format: InsightExportFormat = "jsonl"
    overwrite: bool = False
    include_readme: bool = True


class InsightExportFileSummary(ArchiveInsightModel):
    insight_name: str
    file: str
    schema_file: str
    row_count: int = 0
    readiness_verdict: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class InsightExportBundleManifest(ArchiveInsightModel):
    bundle_version: int = INSIGHT_EXPORT_BUNDLE_VERSION
    insight_contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    generated_at: str
    polylogue_version: str
    git_revision: str | None = None
    git_dirty: bool = False
    archive_root: str
    database_path: str
    output_format: InsightExportFormat = "jsonl"
    query: dict[str, str | tuple[str, ...] | None]
    insights: tuple[InsightExportFileSummary, ...] = ()
    warnings: tuple[str, ...] = ()


class InsightExportBundleResult(ArchiveInsightModel):
    output_path: Path
    manifest_path: Path
    coverage_path: Path
    manifest: InsightExportBundleManifest


class InsightExportOperations(Protocol):
    async def get_insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport: ...


def normalize_export_insight_name(value: str) -> str:
    normalized = value.strip().replace("-", "_")
    if normalized in DEFAULT_EXPORT_INSIGHTS:
        return normalized
    alias = _INSIGHT_ALIASES.get(value.strip()) or _INSIGHT_ALIASES.get(value.strip().replace("_", "-"))
    if alias is not None:
        return alias
    raise InsightExportBundleError(f"Unknown export insight: {value}")


def _selected_insight_names(insights: Sequence[str]) -> tuple[str, ...]:
    if not insights:
        return DEFAULT_EXPORT_INSIGHTS
    selected: list[str] = []
    for insight in insights:
        name = normalize_export_insight_name(insight)
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def _insight_path(insight_name: str) -> str:
    return f"insights/{insight_name}.jsonl"


def _schema_path(insight_name: str) -> str:
    return f"schemas/{insight_name}.schema.json"


def _query_kwargs(
    insight_type: InsightType, request: InsightExportBundleRequest
) -> tuple[dict[str, object], tuple[str, ...]]:
    query_model = insight_type.query_model
    if query_model is None:
        return {}, (f"{insight_type.name} has no query model and cannot be fetched",)
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
            warnings.append(f"{insight_type.name} does not support {key} bounds")
    return kwargs, tuple(warnings)


def _json_schema_document(insight_name: str) -> JSONDocument:
    model = _INSIGHT_MODEL_BY_NAME[insight_name]
    schema = require_json_document(model.model_json_schema(), context=f"{insight_name} JSON schema")
    return {
        "insight_name": insight_name,
        "model_name": model.__name__,
        "contract_version": ARCHIVE_INSIGHT_CONTRACT_VERSION,
        "schema": schema,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(dumps(payload) + "\n", encoding="utf-8")


def _write_insight_jsonl(path: Path, items: Sequence[ArchiveInsightModel]) -> None:
    lines = [item.model_dump_json(exclude_none=True) for item in items]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _write_readme(path: Path, manifest: InsightExportBundleManifest) -> None:
    lines = [
        "# Polylogue Insight Export Bundle",
        "",
        f"- Generated: `{manifest.generated_at}`",
        f"- Polylogue: `{manifest.polylogue_version}`",
        f"- Insight contract: `{manifest.insight_contract_version}`",
        f"- Insights: `{len(manifest.insights)}`",
        "",
        "| Insight | Rows | Readiness | File |",
        "| --- | ---: | --- | --- |",
    ]
    for insight in manifest.insights:
        lines.append(
            f"| `{insight.insight_name}` | {insight.row_count} | `{insight.readiness_verdict or '-'}` | `{insight.file}` |"
        )
    if manifest.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_target(request: InsightExportBundleRequest) -> Path:
    target = request.output_path
    if target.exists() and not request.overwrite:
        raise InsightExportBundleError(f"Export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    tmp_target.mkdir(parents=False)
    (tmp_target / "insights").mkdir()
    (tmp_target / "schemas").mkdir()
    return tmp_target


def _publish_target(tmp_target: Path, request: InsightExportBundleRequest) -> None:
    target = request.output_path
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    tmp_target.replace(target)


async def export_insight_bundle(
    operations: InsightExportOperations,
    config: Config,
    request: InsightExportBundleRequest,
) -> InsightExportBundleResult:
    selected_insights = _selected_insight_names(request.insights)
    readiness = await operations.get_insight_readiness_report(
        InsightReadinessQuery(
            insights=selected_insights,
            provider=request.provider,
            since=request.since,
            until=request.until,
        )
    )
    readiness_by_name = {entry.insight_name: entry for entry in readiness.insights}
    tmp_target = _prepare_target(request)
    summaries: list[InsightExportFileSummary] = []
    bundle_warnings: list[str] = []
    try:
        for insight_name in selected_insights:
            insight_type = INSIGHT_REGISTRY[insight_name]
            insight_file = _insight_path(insight_name)
            schema_file = _schema_path(insight_name)
            kwargs, warnings = _query_kwargs(insight_type, request)
            errors: list[str] = []
            items: list[ArchiveInsightModel] = []
            try:
                items = await fetch_insights_async(insight_type, operations, **kwargs)
            except (ArchiveInsightUnavailableError, InsightQueryError) as exc:
                errors.append(str(exc))
            _write_insight_jsonl(tmp_target / insight_file, items)
            _write_json(tmp_target / schema_file, _json_schema_document(insight_name))
            readiness_entry = readiness_by_name.get(insight_name)
            summaries.append(
                InsightExportFileSummary(
                    insight_name=insight_name,
                    file=insight_file,
                    schema_file=schema_file,
                    row_count=len(items),
                    readiness_verdict=readiness_entry.verdict if readiness_entry is not None else None,
                    warnings=warnings,
                    errors=tuple(errors),
                )
            )
            bundle_warnings.extend(f"{insight_name}: {warning}" for warning in warnings)
            bundle_warnings.extend(f"{insight_name}: {error}" for error in errors)

        manifest = InsightExportBundleManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            polylogue_version=VERSION_INFO.full,
            git_revision=VERSION_INFO.commit,
            git_dirty=VERSION_INFO.dirty,
            archive_root=str(config.archive_root),
            database_path=str(config.db_path),
            output_format=request.output_format,
            query={
                "insights": selected_insights,
                "provider": request.provider,
                "since": request.since,
                "until": request.until,
            },
            insights=tuple(summaries),
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

    return InsightExportBundleResult(
        output_path=request.output_path,
        manifest_path=request.output_path / "manifest.json",
        coverage_path=request.output_path / "coverage.json",
        manifest=manifest,
    )


__all__ = [
    "DEFAULT_EXPORT_INSIGHTS",
    "INSIGHT_EXPORT_BUNDLE_VERSION",
    "InsightExportBundleError",
    "InsightExportBundleManifest",
    "InsightExportBundleRequest",
    "InsightExportBundleResult",
    "InsightExportFileSummary",
    "InsightExportFormat",
    "export_insight_bundle",
    "normalize_export_insight_name",
]
