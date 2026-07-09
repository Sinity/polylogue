"""Cost-panel endpoint contracts for the reader (#1122).

``GET /api/sessions/{id}/cost`` returns a typed cost panel payload
shaped by ``polylogue/daemon/http.py:_cost_panel_payload``. Tests cover
the four confidence/availability states the panel must distinguish:

- exact (provider-reported) -> ``q-canonical``
- priced (catalog-estimated) -> ``q-estimated``
- partial (mixed) -> ``q-heuristic``
- unavailable / unconfigured -> ``q-unavailable``

The unknown-session case is a hard 404. The session-without-
session-cost-insight case is a 200 with an explicit unavailable shape
(empty panel never blank — AC#1122).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostEstimatePayload,
    CostUsagePayload,
)
from polylogue.daemon.http import (
    DaemonAPIHandler,
    DaemonAPIHTTPServer,
    _basis_dict,
    _confidence_tag,
    _cost_panel_payload,
    _empty_cost_payload,
    _usage_dict,
)
from polylogue.insights.archive import (
    ArchiveInsightProvenance,
    SessionCostInsight,
)

# ---------------------------------------------------------------------------
# In-process handler harness (mirrors test_daemon_http_security.py)
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str) -> DaemonAPIHandler:
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast(DaemonAPIHTTPServer, _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": "0"}
    handler.headers = cast(Message, _MockHeaders(headers))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _seed_minimum_archive(workspace_env: dict[str, Path]) -> str:
    """Seed the archive with one session + one message.

    Returns the archive session id (``origin:native_id``) the cost endpoint
    resolves through ``poly.get_session``.
    """
    from tests.infra.storage_records import SessionBuilder, db_setup

    builder = (
        SessionBuilder(db_setup(workspace_env), "cost-1")
        .provider("claude-code")
        .title("A conv")
        .add_message(message_id="m-cost-1", role="user", text="hello world")
    )
    builder.save()
    return builder.native_session_id()


# ---------------------------------------------------------------------------
# Pure helpers — confidence vocabulary + payload shapes
# ---------------------------------------------------------------------------


class TestConfidenceTagMapping:
    """``_confidence_tag`` maps status to the MK3 chip vocabulary (#1127)."""

    def test_exact_maps_to_q_canonical(self) -> None:
        assert _confidence_tag("exact") == "q-canonical"

    def test_priced_maps_to_q_estimated(self) -> None:
        assert _confidence_tag("priced") == "q-estimated"

    def test_partial_maps_to_q_heuristic(self) -> None:
        assert _confidence_tag("partial") == "q-heuristic"

    def test_unavailable_maps_to_q_unavailable(self) -> None:
        assert _confidence_tag("unavailable") == "q-unavailable"

    def test_unknown_status_falls_back_to_unavailable(self) -> None:
        # Defensive: any future status the panel does not recognise must
        # not pretend the data is canonical.
        assert _confidence_tag("totally-new") == "q-unavailable"


class TestCostPayloadShape:
    """``_cost_panel_payload`` projects a typed insight into the public shape."""

    def _make_insight(
        self,
        *,
        status: str,
        total_usd: float,
        basis: CostBasisPayload | None = None,
        confidence: float = 0.95,
    ) -> SessionCostInsight:
        estimate = CostEstimatePayload(
            source_name="claude-code",
            session_id="c1",
            model_name="claude-sonnet-4-6",
            normalized_model="claude-sonnet-4-6",
            status=status,  # type: ignore[arg-type]
            confidence=confidence,
            total_usd=total_usd,
            basis=basis or CostBasisPayload(provider_reported_usd=total_usd),
            usage=CostUsagePayload(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        return SessionCostInsight(
            session_id="c1",
            source_name="claude-code",
            estimate=estimate,
            provenance=ArchiveInsightProvenance(
                materializer_version=1,
                materialized_at="2026-05-17T00:00:00+00:00",
            ),
        )

    def test_exact_payload_emits_q_canonical(self) -> None:
        insight = self._make_insight(status="exact", total_usd=1.23)
        payload = _cost_panel_payload(insight)
        assert payload["origin"] == "claude-code-session"
        assert payload["status"] == "exact"
        assert payload["confidence_tag"] == "q-canonical"
        assert payload["total_usd"] == pytest.approx(1.23)
        basis = cast(dict[str, float], payload["basis"])
        assert basis["provider_reported_usd"] == pytest.approx(1.23)

    def test_priced_payload_emits_q_estimated(self) -> None:
        insight = self._make_insight(
            status="priced",
            total_usd=0.42,
            basis=CostBasisPayload(catalog_priced_usd=0.42, api_equivalent_usd=0.42),
            confidence=0.85,
        )
        payload = _cost_panel_payload(insight)
        assert payload["status"] == "priced"
        assert payload["confidence_tag"] == "q-estimated"
        basis = cast(dict[str, float], payload["basis"])
        assert basis["catalog_priced_usd"] == pytest.approx(0.42)
        assert basis["provider_reported_usd"] == 0.0

    def test_partial_payload_emits_q_heuristic(self) -> None:
        insight = self._make_insight(status="partial", total_usd=0.05, confidence=0.55)
        payload = _cost_panel_payload(insight)
        assert payload["confidence_tag"] == "q-heuristic"

    def test_unavailable_payload_emits_q_unavailable(self) -> None:
        insight = self._make_insight(status="unavailable", total_usd=0.0, confidence=0.0)
        payload = _cost_panel_payload(insight)
        assert payload["confidence_tag"] == "q-unavailable"

    def test_usage_and_basis_are_typed_dicts(self) -> None:
        insight = self._make_insight(status="exact", total_usd=1.0)
        payload = _cost_panel_payload(insight)
        usage = payload["usage"]
        basis = payload["basis"]
        assert isinstance(usage, dict)
        assert isinstance(basis, dict)
        assert set(usage.keys()) == {
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
        }
        assert set(basis.keys()) == {
            "provider_reported_usd",
            "api_equivalent_usd",
            "subscription_equivalent_usd",
            "catalog_priced_usd",
            "tool_surcharge_usd",
        }


class TestEmptyCostPayload:
    """``_empty_cost_payload`` is rendered when no session-cost insight exists."""

    def test_payload_is_explicit_unavailable_not_blank(self) -> None:
        payload = _empty_cost_payload("c-blank", "claude-code")
        assert payload["status"] == "unavailable"
        assert payload["confidence_tag"] == "q-unavailable"
        assert payload["total_usd"] == 0.0
        assert payload["missing_reasons"] == ["no_session_cost_insight"]
        # Basis split is present-but-zero so the panel never reaches for a
        # missing dict on the client.
        assert isinstance(payload["basis"], dict)
        assert payload["basis"]["provider_reported_usd"] == 0.0

    def test_origin_passes_through(self) -> None:
        payload = _empty_cost_payload("c-x", "codex-session")
        assert payload["origin"] == "codex-session"

    def test_helpers_round_trip_through_json(self) -> None:
        # Defensive: panel data ships over HTTP as JSON, so every field
        # must be JSON-serialisable.
        payload = _empty_cost_payload("c-x", "claude-code")
        json.dumps(payload)


# ---------------------------------------------------------------------------
# End-to-end endpoint contract
# ---------------------------------------------------------------------------


class TestCostEndpointDispatch:
    """``GET /api/sessions/{id}/cost`` routes to the cost handler."""

    def test_unknown_session_returns_404(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/sessions/does-not-exist/cost")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_not_called()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"

    def test_known_session_returns_typed_panel(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/sessions/{session_id}/cost")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["session_id"] == session_id
        # No tokens were materialised for the seed; the panel must still
        # surface the typed shape with an explicit confidence tag.
        assert payload["confidence_tag"] in {
            "q-canonical",
            "q-estimated",
            "q-heuristic",
            "q-unavailable",
        }
        assert "basis" in payload
        assert "usage" in payload
        assert "per_model_breakdown" in payload


# ---------------------------------------------------------------------------
# Pure helpers tested independently
# ---------------------------------------------------------------------------


def test_basis_dict_and_usage_dict_use_floats_and_ints() -> None:
    basis = CostBasisPayload(provider_reported_usd=1.5, catalog_priced_usd=2.5)
    usage = CostUsagePayload(input_tokens=10, output_tokens=20, total_tokens=30)
    assert _basis_dict(basis)["provider_reported_usd"] == pytest.approx(1.5)
    assert _basis_dict(basis)["catalog_priced_usd"] == pytest.approx(2.5)
    assert _usage_dict(usage)["input_tokens"] == 10
    assert _usage_dict(usage)["total_tokens"] == 30
