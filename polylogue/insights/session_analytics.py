"""Ad hoc cross-session analytics with no dedicated materialized-insight home.

These primitives originally lived inline in the MCP surface
(``mcp/server_insight_tools.py``) as one-off math, unreachable from the CLI
or the Python library facade. Moved here (polylogue-9e5.24) so any caller can
reach the identical algorithms: Pearson correlation between two numeric
session metrics, the metadata-similarity fallback for ``find_similar_sessions``,
and the per-key set-diff used by ``compare_sessions``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from polylogue.core.sources import source_name_to_origin
from polylogue.insights.archive import SessionProfileInsight

# Numeric session-profile metrics that ``pearson_session_correlation`` can
# correlate. Sourced from the evidence/inference payloads (#1691).
CORRELATABLE_SESSION_METRICS: frozenset[str] = frozenset(
    {
        "message_count",
        "word_count",
        "tool_use_count",
        "thinking_count",
        "engaged_duration_ms",
        "tool_active_duration_ms",
        "wall_duration_ms",
        "total_cost_usd",
        "total_duration_ms",
        "substantive_count",
    }
)

# The subset of a session-comparison row whose values are diffed across
# sessions in ``diff_session_comparison_rows``.
SESSION_COMPARISON_DIFF_KEYS: tuple[str, ...] = (
    "origin",
    "workflow_shape",
    "terminal_state",
    "message_count",
    "tool_call_count",
    "engaged_duration_ms",
    "tool_active_duration_ms",
    "word_count",
)


def ensure_known_session_metric(key: str, arg_name: str) -> None:
    """Raise ``ValueError`` unless ``key`` is a correlatable session metric."""
    if key not in CORRELATABLE_SESSION_METRICS:
        raise ValueError(f"Unknown {arg_name}: {key!r}")


def session_metric_value(profile: SessionProfileInsight, key: str) -> float | None:
    """Resolve one correlatable metric from a session profile's evidence/inference."""
    evidence = profile.evidence
    inference = profile.inference
    if key == "message_count":
        return float(evidence.message_count) if evidence else None
    if key == "word_count":
        return float(evidence.word_count) if evidence else None
    if key == "tool_use_count":
        return float(evidence.tool_use_count) if evidence else None
    if key == "thinking_count":
        return float(evidence.thinking_count) if evidence else None
    if key == "engaged_duration_ms":
        return float(inference.engaged_duration_ms) if inference else None
    if key == "tool_active_duration_ms":
        return float(evidence.tool_active_duration_ms) if evidence else None
    if key == "wall_duration_ms":
        return float(evidence.wall_duration_ms) if evidence else None
    if key == "total_cost_usd":
        return float(evidence.total_cost_usd) if evidence else None
    if key == "total_duration_ms":
        return float(evidence.total_duration_ms) if evidence else None
    if key == "substantive_count":
        return float(evidence.substantive_count) if evidence else None
    return None


def pearson_session_correlation(
    profiles: Sequence[SessionProfileInsight],
    *,
    metric_x: str,
    metric_y: str,
) -> dict[str, object]:
    """Pearson correlation coefficient between two numeric session metrics (#1691).

    Extracts ``(x, y)`` pairs from ``profiles`` (skipping sessions missing
    either metric) and returns the typed result payload with an
    interpretation string. Raises ``ValueError`` for an unknown metric name.
    """
    ensure_known_session_metric(metric_x, "metric_x")
    ensure_known_session_metric(metric_y, "metric_y")

    pairs: list[tuple[float, float]] = []
    for profile in profiles:
        x = session_metric_value(profile, metric_x)
        y = session_metric_value(profile, metric_y)
        if x is not None and y is not None:
            pairs.append((x, y))

    n = len(pairs)
    if n < 3:
        return {
            "metric_x": metric_x,
            "metric_y": metric_y,
            "pearson_r": None,
            "sample_count": n,
            "interpretation": "insufficient data (need at least 3 samples)",
        }

    sum_x = sum(p[0] for p in pairs)
    sum_y = sum(p[1] for p in pairs)
    sum_xy = sum(p[0] * p[1] for p in pairs)
    sum_x2 = sum(p[0] * p[0] for p in pairs)
    sum_y2 = sum(p[1] * p[1] for p in pairs)

    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if denominator == 0:
        return {
            "metric_x": metric_x,
            "metric_y": metric_y,
            "pearson_r": None,
            "sample_count": n,
            "interpretation": "constant metric — zero variance, correlation undefined",
        }

    r = (n * sum_xy - sum_x * sum_y) / denominator
    r = max(-1.0, min(1.0, r))

    if abs(r) >= 0.7:
        direction = "strong positive" if r > 0 else "strong negative"
    elif abs(r) >= 0.4:
        direction = "moderate positive" if r > 0 else "moderate negative"
    elif abs(r) >= 0.2:
        direction = "weak positive" if r > 0 else "weak negative"
    else:
        direction = "negligible"

    return {
        "metric_x": metric_x,
        "metric_y": metric_y,
        "pearson_r": round(r, 4),
        "sample_count": n,
        "interpretation": f"{direction} correlation (r={r:.3f})",
    }


def build_session_comparison_row(profile: SessionProfileInsight) -> dict[str, object]:
    """One session's side-by-side comparison row for ``compare_sessions`` (#1691)."""
    evidence = profile.evidence
    inference = profile.inference
    return {
        "id": profile.session_id,
        "origin": source_name_to_origin(profile.source_name),
        "title": profile.title,
        "workflow_shape": inference.workflow_shape if inference else "unknown",
        "terminal_state": inference.terminal_state if inference else "unknown",
        "message_count": evidence.message_count if evidence else 0,
        "tool_call_count": evidence.tool_use_count if evidence else 0,
        "engaged_duration_ms": inference.engaged_duration_ms if inference else 0,
        "tool_active_duration_ms": evidence.tool_active_duration_ms if evidence else 0,
        "word_count": evidence.word_count if evidence else 0,
        "tags": list(evidence.tags) if evidence else [],
        "auto_tags": list(inference.auto_tags) if inference else [],
    }


def diff_session_comparison_rows(rows: Sequence[dict[str, object]]) -> dict[str, list[object]]:
    """Per-key set-diff over comparison rows: keys with >1 distinct value (#1691)."""
    differences: dict[str, list[object]] = {}
    for key in SESSION_COMPARISON_DIFF_KEYS:
        vals = {row.get(key) for row in rows}
        if len(vals) > 1:
            differences[key] = sorted(vals, key=str)
    return differences


def compute_metadata_similarity_candidates(
    ref_profile: SessionProfileInsight,
    candidates: Sequence[SessionProfileInsight],
    *,
    exclude_session_id: str,
) -> list[dict[str, object]]:
    """Weighted metadata-similarity heuristic (#1691), the ``find_similar_sessions``
    fallback used when embeddings are unavailable or ``similarity_dimension="metadata"``.

    Scores each candidate against ``ref_profile`` on same workflow_shape (+3),
    same origin (+1), temporal proximity (+2 within 3 days, +1 within 14 days),
    and tag overlap (+1 per shared tag), with a ``similarity_reasons`` trail.
    Returns candidates with score > 0, sorted by descending score (uncapped --
    callers apply their own limit).
    """
    ref_evidence = ref_profile.evidence
    ref_inference = ref_profile.inference
    ref_shape = ref_inference.workflow_shape if ref_inference else None
    ref_source = ref_profile.source_name
    ref_origin = source_name_to_origin(ref_source)
    ref_date = ref_evidence.canonical_session_date if ref_evidence else None
    ref_tags = set(ref_evidence.tags) if ref_evidence else set()

    scored: list[tuple[int, dict[str, object]]] = []
    for profile in candidates:
        if profile.session_id == exclude_session_id:
            continue
        evidence = profile.evidence
        inference = profile.inference
        score = 0
        reasons: list[str] = []

        cand_shape = inference.workflow_shape if inference else None
        if ref_shape and cand_shape == ref_shape:
            score += 3
            reasons.append(f"same workflow_shape: {ref_shape}")

        cand_source = profile.source_name
        if ref_source and cand_source == ref_source:
            score += 1
            reasons.append(f"same origin: {ref_origin}")

        cand_date = evidence.canonical_session_date if evidence else None
        if ref_date and cand_date:
            try:
                ref_d = date.fromisoformat(ref_date)
                cand_d = date.fromisoformat(cand_date)
                delta = abs((ref_d - cand_d).days)
                if delta <= 3:
                    score += 2
                    reasons.append(f"within 3 days (delta={delta})")
                elif delta <= 14:
                    score += 1
                    reasons.append(f"within 14 days (delta={delta})")
            except (ValueError, TypeError):
                pass

        cand_tags = set(evidence.tags) if evidence else set()
        overlap = ref_tags & cand_tags
        if overlap:
            score += len(overlap)
            reasons.append(f"{len(overlap)} overlapping tags: {sorted(overlap)}")

        if score > 0:
            scored.append(
                (
                    -score,
                    {
                        "session_id": profile.session_id,
                        "title": profile.title,
                        "origin": source_name_to_origin(profile.source_name),
                        "workflow_shape": cand_shape or "unknown",
                        "terminal_state": inference.terminal_state if inference else "unknown",
                        "canonical_session_date": cand_date,
                        "similarity_score": score,
                        "similarity_reasons": reasons,
                    },
                )
            )

    scored.sort(key=lambda item: item[0])
    return [item for _, item in scored]


__all__ = [
    "CORRELATABLE_SESSION_METRICS",
    "SESSION_COMPARISON_DIFF_KEYS",
    "build_session_comparison_row",
    "compute_metadata_similarity_candidates",
    "diff_session_comparison_rows",
    "ensure_known_session_metric",
    "pearson_session_correlation",
    "session_metric_value",
]
