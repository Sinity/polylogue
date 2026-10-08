"""Versioned archive-insight export bundle contracts and writer."""

from __future__ import annotations

import shutil
import uuid
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.analysis.archive import (
    ArchiveCoverageInsight,
    ArchiveInsightUnavailableError,
    SessionProfileInsight,
    SessionTagRollupInsight,
    ThreadInsight,
)
from polylogue.analysis.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION, ArchiveInsightModel
from polylogue.analysis.export_bundle_contracts import (
    InsightExportBundleError,
    InsightExportBundleManifest,
    InsightExportBundleRequest,
    InsightExportBundleResult,
    InsightExportFileSummary,
)
from polylogue.analysis.insight_reads import iter_insight_rows
from polylogue.analysis.readiness import InsightReadinessQuery
from polylogue.analysis.registry import INSIGHT_REGISTRY, InsightQueryError, InsightType
from polylogue.core.json import JSONDocument, dumps, require_json_document
from polylogue.surfaces.outcome import decide_outcome

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

DEFAULT_EXPORT_INSIGHTS: tuple[str, ...] = (
    "session_profiles",
    "threads",
    "session_tag_rollups",
    "archive_coverage",
)
_INSIGHT_MODEL_BY_NAME: dict[str, type[ArchiveInsightModel]] = {
    "session_profiles": SessionProfileInsight,
    "threads": ThreadInsight,
    "session_tag_rollups": SessionTagRollupInsight,
    "archive_coverage": ArchiveCoverageInsight,
}
_INSIGHT_ALIASES = {
    **{name.replace("_", "-"): name for name in DEFAULT_EXPORT_INSIGHTS},
    **{
        insight_type.resolved_cli_command_name: name
        for name, insight_type in INSIGHT_REGISTRY.items()
        if name in DEFAULT_EXPORT_INSIGHTS
    },
}


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
    for key, value in (("origin", request.origin), ("since", request.since), ("until", request.until)):
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


def _write_insight_jsonl(
    path: Path, items: Generator[ArchiveInsightModel, None, None], checkpoint: Callable[[], None]
) -> int:
    count = 0
    with closing(items), path.open("w", encoding="utf-8") as stream:
        for item in items:
            checkpoint()
            stream.write(item.model_dump_json(exclude_none=True))
            stream.write("\n")
            count += 1
    return count


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
            f"| `{insight.insight_name}` | {insight.row_count} | `{insight.withheld_reason or '-'}` | `{insight.file}` |"
        )
    if manifest.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _withheld_reason(entry: object | None) -> str | None:
    """Why an insight's rows must not enter a bundle, or ``None`` to export.

    Divergence is the only export blocker. Rows flagged as heuristically weak
    (``degraded_count``) or merely incomplete are still exported: they are
    valid rows, just not the whole picture.
    """
    if entry is None:
        return None
    if not bool(getattr(entry, "table_present", True)):
        return "insight table is absent"
    if int(getattr(entry, "incompatible_count", 0) or 0):
        return "insight rows fail their schema contract"
    stale = int(getattr(entry, "stale_count", 0) or 0)
    orphan = int(getattr(entry, "orphan_count", 0) or 0)
    if stale or orphan:
        return f"insight rows diverge from their sources (stale={stale} orphan={orphan})"
    return None


def _prepare_target(request: InsightExportBundleRequest) -> Path:
    target = request.output_path
    if (target.exists() or target.is_symlink()) and not request.overwrite:
        raise InsightExportBundleError(f"Export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    tmp_target.mkdir(parents=False, mode=0o700)
    try:
        (tmp_target / "insights").mkdir()
        (tmp_target / "schemas").mkdir()
    except BaseException as primary:
        try:
            if tmp_target.exists():
                shutil.rmtree(tmp_target)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Insight export failed and staging cleanup failed", [primary, cleanup_error]
            ) from None
        raise
    return tmp_target


def _publish_target(tmp_target: Path, request: InsightExportBundleRequest) -> None:
    target = request.output_path
    if target.exists() or target.is_symlink():
        if not request.overwrite:
            raise InsightExportBundleError(f"Export target already exists: {target}")
        # Directory rename cannot replace a populated directory portably. Move
        # the old complete bundle aside first, install the staged bundle, then
        # restore the old name if installation fails. The sibling backup is
        # also the explicit recovery location if restoration itself fails.
        backup = target.parent / f".{target.name}.previous-{uuid.uuid4().hex}"
        while backup.exists() or backup.is_symlink():
            backup = target.parent / f".{target.name}.previous-{uuid.uuid4().hex}"
        target.replace(backup)
        try:
            tmp_target.replace(target)
        except BaseException as publication_error:
            try:
                backup.replace(target)
            except BaseException as restoration_error:
                raise BaseExceptionGroup(
                    f"Export installation failed; previous bundle is recoverable at {backup}",
                    [publication_error, restoration_error],
                ) from None
            raise
        try:
            _remove_export_target(backup)
        except BaseException as cleanup_error:
            raise InsightExportBundleError(
                f"New export is installed at {target}, but previous bundle cleanup failed at {backup}"
            ) from cleanup_error
        return
    tmp_target.replace(target)


def _remove_export_target(path: Path) -> None:
    """Remove a moved export target without following a symlink."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def export_insight_bundle(
    archive: ArchiveStore,
    request: InsightExportBundleRequest,
    *,
    checkpoint: Callable[[], None] = lambda: None,
) -> InsightExportBundleResult:
    selected_insights = _selected_insight_names(request.insights)
    checkpoint()
    readiness = archive.insight_readiness_report(
        InsightReadinessQuery(
            insights=selected_insights,
            origin=request.origin,
            since=request.since,
            until=request.until,
        )
    )
    readiness_by_name = {entry.insight_name: entry for entry in readiness.insights}
    tmp_target = _prepare_target(request)
    gaps: list[str] = [] if readiness.converged else ["insight_convergence_pending"]
    for entry in readiness.insights:
        if entry.diverged:
            gaps.append("insight_output_diverged")
        if entry.incomplete:
            gaps.append("insight_output_incomplete")
        if entry.degraded_count or entry.schema_contract_issues:
            gaps.append("insight_evidence_degraded")
    summaries: list[InsightExportFileSummary] = []
    bundle_warnings: list[str] = []
    try:
        for insight_name in selected_insights:
            insight_type = INSIGHT_REGISTRY[insight_name]
            insight_file = _insight_path(insight_name)
            schema_file = _schema_path(insight_name)
            kwargs, warnings = _query_kwargs(insight_type, request)
            errors: list[str] = []
            checkpoint()
            row_count = 0
            readiness_entry = readiness_by_name.get(insight_name)
            withheld_reason = _withheld_reason(readiness_entry)
            if withheld_reason is not None:
                # The rows do not reflect the sources they were built from: the
                # table is absent, its rows outlive their sessions, or it fails
                # its schema contract. Exporting them would bundle untrustworthy
                # data, so emit an empty file and record why.
                errors.append(f"{withheld_reason}; rows withheld from export")
            else:
                try:
                    assert insight_type.query_model is not None
                    query = insight_type.query_model(**kwargs)
                    row_count = _write_insight_jsonl(
                        tmp_target / insight_file, iter_insight_rows(archive, query, checkpoint=checkpoint), checkpoint
                    )
                except (ArchiveInsightUnavailableError, InsightQueryError) as exc:
                    errors.append(str(exc))
                    gaps.append("insight_export_read_failed")
            if errors:
                # A producer failure withholds the entire product, including any staged prefix.
                (tmp_target / insight_file).write_text("", encoding="utf-8")
                row_count = 0
            _write_json(tmp_target / schema_file, _json_schema_document(insight_name))
            summaries.append(
                InsightExportFileSummary(
                    insight_name=insight_name,
                    file=insight_file,
                    schema_file=schema_file,
                    row_count=row_count,
                    withheld_reason=withheld_reason,
                    warnings=warnings,
                    errors=tuple(errors),
                )
            )
            bundle_warnings.extend(f"{insight_name}: {warning}" for warning in warnings)
            bundle_warnings.extend(f"{insight_name}: {error}" for error in errors)

        from polylogue.version import VERSION_INFO

        manifest = InsightExportBundleManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            polylogue_version=VERSION_INFO.full,
            git_revision=VERSION_INFO.commit,
            git_dirty=VERSION_INFO.dirty,
            archive_root=str(archive.archive_root),
            database_path=str(archive.index_db_path),
            output_format=request.output_format,
            query={
                "insights": selected_insights,
                "origin": request.origin,
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
        checkpoint()
        _publish_target(tmp_target, request)
    except BaseException as primary:
        try:
            if tmp_target.exists():
                shutil.rmtree(tmp_target)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Insight export failed and staging cleanup failed", [primary, cleanup_error]
            ) from None
        raise

    return InsightExportBundleResult(
        output_path=request.output_path,
        manifest_path=request.output_path / "manifest.json",
        coverage_path=request.output_path / "coverage.json",
        manifest=manifest,
        outcome=decide_outcome(matched=sum(entry.row_count for entry in summaries), degraded=gaps),
    )


__all__ = ["DEFAULT_EXPORT_INSIGHTS", "export_insight_bundle", "normalize_export_insight_name"]
