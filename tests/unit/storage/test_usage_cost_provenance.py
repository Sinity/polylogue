"""Usage observation, completeness and money stay independent through reads."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.derived.session.usage_rollup import reconcile_session_usage_rollup
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows
from polylogue.storage.usage import (
    _pricing_lane_reports,
    session_usage_costs_for_connection,
    session_usage_reconciliation_for_connection,
)
from tests.infra.index_writer import write_fixture_index_session


def _write(conn: sqlite3.Connection, session: ParsedSession) -> str:
    return write_fixture_index_session(
        conn, session, content_hash=str(session_content_hash(session)), prepared_rows=prepare_session_rows(session)
    )


@pytest.mark.parametrize("measurement", ["identity", "zero", "partial", "complete"])
@pytest.mark.parametrize("reported_money", [None, 0.0])
def test_observation_and_completeness_survive_reconciliation(
    tmp_path: Path, measurement: str, reported_money: float | None
) -> None:
    counters: dict[str, int | None] = dict.fromkeys(
        ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    )
    if measurement == "zero":
        counters = dict.fromkeys(counters, 0)
    elif measurement == "partial":
        counters["output_tokens"] = 100
    elif measurement == "complete":
        counters = dict.fromkeys(counters, 0)
        counters["output_tokens"] = 100
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="usage-provenance",
        models_used=["gemini-2.0-flash"],
        reported_cost_usd=reported_money,
        messages=[
            ParsedMessage(
                provider_message_id="assistant",
                role=Role.ASSISTANT,
                text="synthetic",
                model_name="gemini-2.0-flash",
                **counters,
            )
        ],
    )
    with closing(connect_measured(tmp_path / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = _write(conn, session)
        for _attempt in range(2):
            cost = session_usage_costs_for_connection(conn, [session_id])[session_id]
            reconciliation = session_usage_reconciliation_for_connection(conn, session_id=session_id)
            measured_complete = measurement in {"zero", "complete"}
            assert reconciliation.reconciled_tokens_evidence.value_state == (
                "known" if measured_complete else "unknown"
            )
            assert reconciliation.catalog_cost_evidence.value_state == ("known" if measured_complete else "unknown")
            if reported_money is not None:
                assert cost.total_usd == 0.0
                assert cost.exactness == "exact"
                assert cost.availability == "provider_money"
            elif measured_complete:
                assert cost.catalog_api_equivalent_usd == pytest.approx(0 if measurement == "zero" else 0.00004)
            else:
                assert cost.total_usd is None
                assert cost.availability == ("no_tokens" if measurement == "identity" else "unpriced")
            for logical in (False, True):
                (lane,) = _pricing_lane_reports(conn, None, logical=logical, observed_at="2026-10-09T00:00:00Z")
                assert lane.exact_total_tokens_evidence.value_state == ("known" if measured_complete else "unknown")
                assert lane.catalog_api_equivalent_evidence.value_state == ("known" if measured_complete else "unknown")
            reconcile_session_usage_rollup(conn, session_id)


@pytest.mark.parametrize("partial", [False, True])
def test_codex_optional_cache_write_keeps_core_measurement_contract(tmp_path: Path, partial: bool) -> None:
    usage = {"input_tokens": 100, "cached_input_tokens": 20}
    if not partial:
        usage["output_tokens"] = 0
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="native-lanes",
        messages=[],
        models_used=["gpt-4o"],
        session_events=[
            ParsedSessionEvent(event_type="token_count", payload={"model": "gpt-4o", "total_token_usage": usage})
        ],
    )
    with closing(connect_measured(tmp_path / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = _write(conn, session)
        cost = session_usage_costs_for_connection(conn, [session_id])[session_id]
        assert cost.provider_lanes_complete is not partial
        assert cost.input_tokens == 80
        assert cost.cache_read_tokens == 20
        assert (cost.total_usd is None) is partial


@pytest.mark.parametrize("reason_only_after", [False, True])
def test_terminal_metadata_does_not_invent_usage_observation(reason_only_after: bool) -> None:
    from polylogue.storage.usage import project_provider_usage_events

    metadata = {"session_id": "s", "model_name": "gpt-4o", "provider_event_type": "token_count", "position": 2}
    complete = {
        "session_id": "s",
        "model_name": "gpt-4o",
        "provider_event_type": "token_count",
        "position": 1,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_input_tokens": 0,
    }
    events = [complete, metadata] if reason_only_after else [metadata]
    (projection,) = project_provider_usage_events(events, origin="codex-session")
    assert projection.provider_usage_observed is reason_only_after
    assert projection.cost_usd == (0.0 if reason_only_after else None)


def test_observed_zero_keeps_known_tokens_without_a_catalog_price(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="unpriced-zero",
        models_used=["synthetic-unpriced-model"],
        messages=[
            ParsedMessage(
                provider_message_id="m",
                role=Role.ASSISTANT,
                text="synthetic",
                model_name="synthetic-unpriced-model",
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )
        ],
    )
    with closing(connect_measured(tmp_path / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        session_id = _write(conn, session)
        reconciliation = session_usage_reconciliation_for_connection(conn, session_id=session_id)
        assert reconciliation.reconciled_tokens_evidence.value_state == "known"
        assert reconciliation.reconciled_tokens_evidence.value == 0
        assert reconciliation.reconciled_cost_evidence.value_state == "unknown"
        assert session_usage_costs_for_connection(conn, [session_id])[session_id].availability == "unpriced"
