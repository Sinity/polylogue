"""A multi-model session's provider-reported dollars reach the cost readers.

The provider prices a session, not a model. ``session_model_usage.
provider_cost_usd`` is where every cost reader looks first
(``COALESCE(SUM(provider), SUM(catalog))``), so a session total that never
lands there is silently replaced by catalog-computed dollars.

Anti-vacuity: restoring a ``len(model_names) == 1`` guard around the provider
write in ``_seed_session_model_usage_rows`` makes
``test_multi_model_provider_total_reaches_cost_readers`` red -- the rollup
falls back to the catalog sum, which these fixtures deliberately price away
from the reported total.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.analysis.archive import CostRollupInsightQuery, UsageTimelineInsightQuery
from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_REPORTED_TOTAL = 12.5


def _multi_model_session(session_key: str) -> ParsedSession:
    """Two models, both catalog-priced, plus one provider-reported total."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_key,
        title="Two models, one provider total",
        created_at="2026-03-01T10:00:00+00:00",
        updated_at="2026-03-01T10:30:00+00:00",
        models_used=["gpt-5", "gpt-5-codex"],
        reported_cost_usd=_REPORTED_TOTAL,
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5",
                input_tokens=400_000,
                output_tokens=200_000,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first model")],
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                model_name="gpt-5-codex",
                input_tokens=100_000,
                output_tokens=50_000,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second model")],
            ),
        ],
    )


def _single_model_session(session_key: str, *, reported: float) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_key,
        title="One model, one provider total",
        created_at="2026-03-01T11:00:00+00:00",
        updated_at="2026-03-01T11:30:00+00:00",
        models_used=["gpt-5"],
        reported_cost_usd=reported,
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                model_name="gpt-5",
                input_tokens=10_000,
                output_tokens=5_000,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="only model")],
            )
        ],
    )


def _write(db_path: Path, *sessions: ParsedSession) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for session in sessions:
            write_parsed_session_to_archive(conn, session)
        conn.commit()
    finally:
        conn.close()


def _catalog_total(db_path: Path) -> float:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT SUM(catalog_cost_usd) FROM session_model_usage").fetchone()
    finally:
        conn.close()
    return float(row[0] or 0.0)


@pytest.mark.asyncio
async def test_multi_model_provider_total_reaches_cost_readers(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    _write(db_path, _multi_model_session("codex-multi-model-provider-total"))

    catalog_total = _catalog_total(db_path)
    assert catalog_total > 0.0, "fixture must be catalog-priced or the fallback is indistinguishable"
    assert catalog_total != pytest.approx(_REPORTED_TOTAL), "catalog sum must differ from the reported total"

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)

    rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery())
    assert sum(rollup.total_usd for rollup in rollups) == pytest.approx(_REPORTED_TOTAL)

    timeline = await archive.list_usage_timeline_insights(UsageTimelineInsightQuery())
    assert sum(entry.stored_cost_usd for entry in timeline) == pytest.approx(_REPORTED_TOTAL)


@pytest.mark.asyncio
async def test_mixed_cohort_counts_each_provider_total_once(
    cli_workspace: dict[str, Path],
) -> None:
    db_path = cli_workspace["db_path"]
    single_reported = 3.25
    _write(
        db_path,
        _multi_model_session("codex-cohort-multi"),
        _single_model_session("codex-cohort-single", reported=single_reported),
    )

    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)

    rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery())
    assert sum(rollup.total_usd for rollup in rollups) == pytest.approx(_REPORTED_TOTAL + single_reported)

    timeline = await archive.list_usage_timeline_insights(UsageTimelineInsightQuery())
    assert sum(entry.stored_cost_usd for entry in timeline) == pytest.approx(_REPORTED_TOTAL + single_reported)


def test_provider_total_is_apportioned_across_every_declared_model(
    cli_workspace: dict[str, Path],
) -> None:
    """No model row falls through to the catalog fallback while a total exists."""
    db_path = cli_workspace["db_path"]
    _write(db_path, _multi_model_session("codex-apportioned"))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT model_name, provider_cost_usd FROM session_model_usage ORDER BY model_name"
        ).fetchall()
    finally:
        conn.close()

    assert [row["model_name"] for row in rows] == ["gpt-5", "gpt-5-codex"]
    assert all(row["provider_cost_usd"] is not None for row in rows)
    assert sum(float(row["provider_cost_usd"]) for row in rows) == pytest.approx(_REPORTED_TOTAL)
    # Catalog share drives the split, so the heavier-priced model takes more.
    assert float(rows[0]["provider_cost_usd"]) > float(rows[1]["provider_cost_usd"])
