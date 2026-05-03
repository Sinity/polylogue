"""Result collection, budget capture, and DB statistics for pipeline probes."""

from __future__ import annotations

from pathlib import Path

from devtools.pipeline_probe.request import BudgetReport, ProbeSummary, RawFanoutEntry
from polylogue.core.json import JSONDocument, JSONValue, json_document, require_json_document
from polylogue.scenarios import PipelineProbeRequest
from polylogue.storage.run_state import RunResult
from polylogue.storage.sqlite.connection import open_connection


def _db_row_counts(db_path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    with open_connection(db_path) as conn:
        for table in ("raw_conversations", "conversations", "messages", "content_blocks"):
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[f"{table}_count"] = int(row[0]) if row else 0
    return stats


def _db_raw_fanout(db_path: Path) -> list[RawFanoutEntry]:
    if not db_path.exists():
        return []
    with open_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                r.raw_id,
                COALESCE(r.payload_provider, r.provider_name) AS payload_provider,
                r.source_name,
                r.blob_size,
                r.parse_error,
                COUNT(DISTINCT c.conversation_id) AS conversation_count,
                COUNT(m.message_id) AS message_count
            FROM raw_conversations r
            LEFT JOIN conversations c ON c.raw_id = r.raw_id
            LEFT JOIN messages m ON m.conversation_id = c.conversation_id
            GROUP BY
                r.raw_id,
                COALESCE(r.payload_provider, r.provider_name),
                r.source_name,
                r.blob_size,
                r.parse_error
            ORDER BY r.blob_size DESC, r.raw_id ASC
            """
        ).fetchall()
    return [
        {
            "raw_id": str(row["raw_id"]),
            "payload_provider": row["payload_provider"],
            "source_name": row["source_name"],
            "blob_size_bytes": int(row["blob_size"]),
            "conversation_count": int(row["conversation_count"]),
            "message_count": int(row["message_count"]),
            "parse_error": row["parse_error"],
        }
        for row in rows
    ]


def _run_result_payload(result: RunResult) -> JSONDocument:
    return require_json_document(result.model_dump(mode="json"), context="pipeline probe result")


def _json_object_or_empty(value: object | None) -> JSONDocument:
    return json_document(value)


def _json_float_or_none(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _json_string_sequence(value: list[str] | None) -> list[JSONValue] | None:
    if value is None:
        return None
    result: list[JSONValue] = []
    result.extend(value)
    return result


def _observed_peak_rss_mb(metrics: JSONDocument) -> tuple[JSONValue, JSONValue, JSONValue]:
    peak_self = metrics.get("peak_rss_self_mb")
    peak_children = metrics.get("peak_rss_children_mb")
    peak_self_value = _json_float_or_none(peak_self)
    peak_children_value = _json_float_or_none(peak_children)
    if peak_self_value is None:
        return peak_self, peak_self, peak_children
    if peak_children_value is None:
        return peak_self, peak_self, peak_children
    return round(peak_self_value + peak_children_value, 1), peak_self, peak_children


def _build_budget_report(summary: ProbeSummary, request: PipelineProbeRequest) -> BudgetReport | None:
    if request.max_total_ms is None and request.max_peak_rss_mb is None:
        return None

    run_payload = _json_object_or_empty(summary.get("run_payload"))
    metrics = _json_object_or_empty(run_payload.get("metrics"))
    result_payload = _json_object_or_empty(summary.get("result"))
    observed_total_ms = metrics.get("total_duration_ms", result_payload.get("duration_ms"))
    observed_peak_rss_mb, observed_peak_rss_self_mb, observed_peak_rss_children_mb = _observed_peak_rss_mb(metrics)
    violations: list[str] = []

    if request.max_total_ms is not None:
        if observed_total_ms is None:
            violations.append("missing total runtime metric")
        elif (observed_total_ms_value := _json_float_or_none(observed_total_ms)) is None:
            violations.append("non-numeric total runtime metric")
        elif observed_total_ms_value > request.max_total_ms:
            violations.append(
                f"total runtime {observed_total_ms_value:.1f} ms exceeded budget {request.max_total_ms:.1f} ms"
            )

    if request.max_peak_rss_mb is not None:
        if observed_peak_rss_mb is None:
            violations.append("missing peak RSS metric")
        elif (observed_peak_rss_mb_value := _json_float_or_none(observed_peak_rss_mb)) is None:
            violations.append("non-numeric peak RSS metric")
        elif observed_peak_rss_mb_value > request.max_peak_rss_mb:
            violations.append(
                f"peak RSS {observed_peak_rss_mb_value:.1f} MiB exceeded budget {request.max_peak_rss_mb:.1f} MiB"
            )

    return {
        "ok": not violations,
        "max_total_ms": request.max_total_ms,
        "observed_total_ms": observed_total_ms,
        "max_peak_rss_mb": request.max_peak_rss_mb,
        "observed_peak_rss_mb": observed_peak_rss_mb,
        "observed_peak_rss_self_mb": observed_peak_rss_self_mb,
        "observed_peak_rss_children_mb": observed_peak_rss_children_mb,
        "violations": violations,
    }


__all__ = [
    "_build_budget_report",
    "_db_raw_fanout",
    "_db_row_counts",
    "_json_float_or_none",
    "_json_object_or_empty",
    "_json_string_sequence",
    "_observed_peak_rss_mb",
    "_run_result_payload",
]
