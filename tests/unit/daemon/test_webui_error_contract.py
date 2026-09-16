"""WebUI rendering must never show an unmeasured or failed state as a measured one.

Each test below pins one seam where a failure, a stall, an out-of-range value,
an unbounded payload, or a contract violation previously rendered as a positive
value, a measured zero, or an empty archive.
"""

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue.daemon.webui import (
    WebUIAssetBundle,
    WebUIAssetError,
    WebUIUnitContractError,
    _bounded_render_text,
    _format_timestamp,
    _message_rows,
    _render_card_field_value,
    _render_semantic_prose,
    _status_panel_payload,
    build_cost_payload,
    build_observability_payload,
)
from polylogue.rendering.semantic_cards import DEFAULT_PREVIEW_MAX_CHARS
from polylogue.surfaces.outcome import OutcomeEnvelope
from polylogue.surfaces.payloads import BlockQueryRowPayload, MessageQueryRowPayload, QueryUnitEnvelope


def _descriptor(name: str, *, readiness_exempt: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        display_name=name,
        json_key=name,
        fields=(SimpleNamespace(label="proof", accessor=lambda item: item.proof),),
        query_model=None,
        mcp_default_limit=1,
        readiness_exempt=readiness_exempt,
    )


_READY_STATUS = {"component_readiness": {"session_profiles": {"state": "ready"}}}


# ---------------------------------------------------------------- polylogue-7viwp


def test_one_stalled_descriptor_degrades_only_its_own_panel() -> None:
    """polylogue-7viwp: a descriptor that never returns must not starve its siblings.

    Anti-vacuity: the stalled fetch blocks forever, so a serial loop (or a
    removed ``asyncio.wait_for``) never reaches the healthy descriptor and the
    test hangs rather than passing. The sibling assertion is only reachable if
    the fetches are concurrent AND the stalled one is bounded and cancelled.
    """
    stalled = _descriptor("a-stalled")
    healthy = _descriptor("b-healthy")
    healthy_item = SimpleNamespace(proof="healthy", model_dump=lambda *, mode: {"proof": "healthy"})
    cancelled: list[str] = []

    async def fake_fetch(descriptor: object, _operations: object, **_kwargs: object) -> list[object]:
        if descriptor is stalled:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append("a-stalled")
                raise
            raise AssertionError("unreachable")
        return [healthy_item]

    with patch("polylogue.analysis.registry.fetch_insights_async", fake_fetch):
        payload = asyncio.run(
            build_observability_payload(
                object(),
                _READY_STATUS,
                registry={"a-stalled": stalled, "b-healthy": healthy},
                panel_timeout_s=0.05,
            )
        )

    panels = {str(p["name"]): p for p in cast(list[dict[str, object]], payload["insights"])}
    assert panels["b-healthy"]["state"] == "available"
    assert panels["b-healthy"]["items"], "the healthy sibling must still carry its rows"
    assert panels["a-stalled"]["state"] == "degraded"
    assert "panel budget" in str(panels["a-stalled"]["error"])
    assert cancelled == ["a-stalled"], "the timed-out fetch must be cancelled, not abandoned"


# ---------------------------------------------------------------- polylogue-0ye6q


def test_cost_panel_failure_with_empty_message_still_reports_degraded() -> None:
    """polylogue-0ye6q: degraded state is keyed on exception presence, not message text.

    Anti-vacuity: every fetch raises a bare ``RuntimeError()`` whose ``str()``
    is empty. Restoring ``if error and not items`` in the three cost renderers,
    or returning ``str(exc)`` unconditioned, makes each panel report itself as
    genuinely un-materialized and both assertions below go red.
    """

    async def fake_fetch(_descriptor: object, _operations: object, **_kwargs: object) -> list[object]:
        raise RuntimeError

    with patch("polylogue.analysis.registry.fetch_insights_async", fake_fetch):
        payload = asyncio.run(build_cost_payload(object()))

    from polylogue.daemon.webui import (
        _render_cost_rollups_section,
        _render_session_cost_drilldown,
        _render_usage_timeline_section,
    )

    for key in ("rollups", "timeline", "sessions"):
        section = cast(dict[str, object], payload[key])
        assert section["error"] is not None, f"{key} must carry the error's presence"
        assert str(section["error"]), f"{key} error text must never be empty"

    rollups = _render_cost_rollups_section(cast(Any, payload["rollups"]))
    timeline = _render_usage_timeline_section(cast(Any, payload["timeline"]))
    sessions = _render_session_cost_drilldown(cast(Any, payload["sessions"]))
    for markup in (rollups, timeline, sessions):
        assert "cost-degraded" in markup
        assert "materialized yet" not in markup


# ---------------------------------------------------------------- polylogue-jyzfs


def test_asset_body_oserror_becomes_the_structured_503(tmp_path: Path) -> None:
    """polylogue-jyzfs: an unreadable manifest asset yields the same 503 as an unreadable manifest.

    Anti-vacuity: ``read_bytes`` raises ``OSError`` for an asset that passes
    ``is_file()``. Removing the ``except OSError`` arm in ``read_asset`` lets it
    propagate past the handler's ``FileNotFoundError``/``WebUIAssetError`` arms,
    so ``_send_error`` is never called and the recorded list stays empty.
    """
    from polylogue.daemon.http import DaemonAPIHandler

    manifest = {
        "src/main.ts": {
            "isEntry": True,
            "name": "archive-overview",
            "file": "archive-overview-abcd1234.js",
        }
    }
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dist / "archive-overview-abcd1234.js").write_text("export {};", encoding="utf-8")

    bundle = WebUIAssetBundle(dist)
    with patch("pathlib.Path.read_bytes", side_effect=OSError("Input/output error")):
        with pytest.raises(WebUIAssetError):
            bundle.read_asset("archive-overview-abcd1234.js")

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.path = "/assets/archive-overview-abcd1234.js"
    handler.server = SimpleNamespace(webui_dist_root=dist)  # type: ignore[assignment]
    sent: list[tuple[HTTPStatus, str]] = []

    def record_error(_self: Any, status: HTTPStatus, code: str, detail: str | None = None) -> None:
        sent.append((status, code))

    with patch("pathlib.Path.read_bytes", side_effect=OSError("Input/output error")):
        with patch.object(DaemonAPIHandler, "_send_error", record_error):
            handler._serve_webui_asset("archive-overview-abcd1234.js")

    assert sent == [(HTTPStatus.SERVICE_UNAVAILABLE, "webui_assets_unavailable")]


# ---------------------------------------------------------------- polylogue-uwxir


def test_readiness_projection_separates_poisoned_rebuilding_and_missing() -> None:
    """polylogue-uwxir: poisoned and rebuilding map explicitly, not through the default.

    Anti-vacuity: drop either key from ``_COMPONENT_STATE_PROJECTION`` and that
    component falls to the ``unavailable`` default, collapsing severe corruption
    and active recovery into "no evidence" - the three-way distinction asserted
    below disappears.
    """
    payload = _status_panel_payload(
        {
            "component_readiness": {
                "a_poisoned": {"state": "poisoned", "reason": "raw frontier integrity failed"},
                "b_rebuilding": {"state": "rebuilding", "reason": "reindex in flight"},
                "c_missing": {"state": "missing", "reason": "never materialized"},
            }
        }
    )
    components = {str(c["name"]): c for c in cast(list[dict[str, object]], payload["components"])}
    assert components["a_poisoned"]["state"] == "degraded"
    assert components["a_poisoned"]["readiness_state"] == "poisoned"
    assert components["b_rebuilding"]["state"] == "refreshing"
    assert components["b_rebuilding"]["readiness_state"] == "rebuilding"
    assert components["c_missing"]["state"] == "unavailable"
    assert len({str(c["state"]) for c in components.values()}) == 3


# ---------------------------------------------------------------- polylogue-umca4


def test_out_of_range_timestamp_renders_a_typed_placeholder() -> None:
    """polylogue-umca4: a SQLite-valid but unrepresentable epoch never raises.

    Anti-vacuity: 9e18 ms is accepted by ``MessageQueryRowPayload`` and makes
    ``datetime.fromtimestamp`` raise ``year must be in 1..9999``. Remove the
    guard in ``_format_timestamp`` and this test raises ValueError instead of
    asserting; the placeholder is also asserted to be distinct from the
    "no timestamp recorded" answer so a blanket ``return None`` fails too.
    """
    assert _format_timestamp(None) is None

    result = _format_timestamp(9_000_000_000_000_000_000)
    assert result is not None
    iso, label = result
    assert iso is None, "an unrepresentable instant must not carry a datetime attribute"
    assert "out of range" in label

    from polylogue.daemon.webui import _render_message_row

    row = MessageQueryRowPayload(
        message_id="codex-session:s1:n:m1",
        session_id="codex-session:s1",
        origin="codex-session",
        role="user",
        message_type="text",
        position=0,
        word_count=1,
        text="hello",
        occurred_at_ms=9_000_000_000_000_000_000,
    )
    markup = _render_message_row(row)
    assert "out of range" in markup
    assert "<time" not in markup


# ---------------------------------------------------------------- polylogue-a03p0


def test_semantic_prose_and_card_fields_apply_a_counted_ceiling() -> None:
    """polylogue-a03p0: oversized untrusted text is truncated with a counted notice.

    Anti-vacuity: the payload is one character over
    ``DEFAULT_PREVIEW_MAX_CHARS``. Remove the ``_bounded_render_text`` call from
    either renderer and the full body reappears, the rendered length assertion
    fails, and the "1 characters elided" notice is absent.
    """
    overflow = 5
    text = "x" * (DEFAULT_PREVIEW_MAX_CHARS + overflow)
    bounded, omitted = _bounded_render_text(text)
    assert len(bounded) == DEFAULT_PREVIEW_MAX_CHARS
    assert omitted == overflow

    prose = _render_semantic_prose({"block_type": "text", "text": text})
    assert f'data-omitted-characters="{overflow}"' in prose
    assert f"{overflow:,} characters elided" in prose
    assert text not in prose

    field = _render_card_field_value(text)
    assert f'data-omitted-characters="{overflow}"' in field
    assert text not in field

    # A value at or below the ceiling is untouched, including the ref linking.
    assert _elided_free(_render_card_field_value("session:codex-session:s1"))


def _elided_free(markup: str) -> bool:
    return "elision-notice" not in markup and "/sessions/" in markup


# ---------------------------------------------------------------- polylogue-lbsh9


def test_unit_mismatched_row_is_a_typed_error_not_an_empty_panel() -> None:
    """polylogue-lbsh9: a row whose type contradicts the declared unit refuses loudly.

    Anti-vacuity: the envelope validly declares ``unit="message"`` while
    carrying a ``BlockQueryRowPayload``. Restore the silent ``isinstance``
    filter in ``_message_rows`` and this returns an empty tuple, which the page
    renders as "No indexed message activity" - the pytest.raises block goes red.
    """
    message_row = MessageQueryRowPayload(
        message_id="codex-session:s1:n:m1",
        session_id="codex-session:s1",
        origin="codex-session",
        role="user",
        message_type="text",
        position=0,
        word_count=1,
        text="hello",
    )
    block_row = BlockQueryRowPayload(
        block_id="codex-session:s1:n:m1:0",
        message_id="codex-session:s1:n:m1",
        session_id="codex-session:s1",
        origin="codex-session",
        block_type="text",
        position=0,
    )
    envelope = QueryUnitEnvelope.model_validate(
        {
            "unit": "message",
            "query": "messages where words >= 0",
            "items": [message_row.model_dump(mode="json"), block_row.model_dump(mode="json")],
            "total": 2,
            "limit": 6,
            "offset": 0,
            "outcome": OutcomeEnvelope(state="ok").model_dump(mode="json"),
        }
    )
    # The envelope model itself accepts the mismatch; the WebUI seam must not.
    assert isinstance(envelope.items[1], BlockQueryRowPayload)

    with pytest.raises(WebUIUnitContractError) as caught:
        _message_rows(envelope)
    assert "item 1" in str(caught.value)

    clean = QueryUnitEnvelope.model_validate(
        {
            "unit": "message",
            "query": "messages where words >= 0",
            "items": [message_row.model_dump(mode="json")],
            "total": 1,
            "limit": 6,
            "offset": 0,
            "outcome": OutcomeEnvelope(state="ok").model_dump(mode="json"),
        }
    )
    assert _message_rows(clean) == (clean.items[0],)
