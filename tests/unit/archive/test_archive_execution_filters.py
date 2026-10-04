"""Archive execution adapter filter coverage."""

from __future__ import annotations

import pytest

from polylogue.api.archive import _archive_query_kwargs
from polylogue.archive.query.expression import compile_expression
from polylogue.archive.query.filter_kwargs import plan_filter_kwargs
from polylogue.archive.query.spec import SessionQuerySpec


def test_archive_filter_kwargs_include_session_id() -> None:
    plan = compile_expression("id:abc123").to_plan()

    assert plan_filter_kwargs(plan)["session_id"] == "abc123"


def test_alternate_query_kwargs_preserve_canonical_structural_filters() -> None:
    spec = SessionQuerySpec.from_params(
        {
            "project": "project-ref",
            "conv_id": "codex-session:abc",
            "root": False,
            "exclude_text": ("secret",),
        },
        strict=True,
    )

    kwargs = _archive_query_kwargs(spec, default_limit=50)

    assert kwargs["project_refs"] == ("project-ref",)
    assert kwargs["session_id"] == "codex-session:abc"
    assert kwargs["root"] is False


def test_alternate_query_kwargs_resolve_implicit_root_filter() -> None:
    spec = SessionQuerySpec.from_params({}, strict=True)

    assert _archive_query_kwargs(spec, default_limit=50)["root"] is True


def test_attached_unit_gap_logs_requested_domains_without_field_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import json
    from unittest.mock import MagicMock

    from polylogue import logging as event_logging
    from polylogue.archive.query.archive_execution import _attach_units_to_domain
    from polylogue.archive.query.attached_units import AttachedUnitRows
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.builders import make_conv

    monkeypatch.setattr(
        "polylogue.archive.query.attached_units.fetch_attached_units",
        lambda *args, **kwargs: AttachedUnitRows(rows={}, gaps=("attached_unit_truncated",)),
    )
    stream = io.StringIO()
    try:
        event_logging.configure_events(stream=stream, fmt="json", level="warning")
        result = _attach_units_to_domain([make_conv(id="session")], MagicMock(spec=ArchiveStore), ("message", "action"))
        event_logging.flush_events(timeout_s=1)
    finally:
        event_logging.reset_events()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    gaps = [record for record in records if record["event"] == "archive.attached_units.truncated"]
    assert len(gaps) == 1
    assert gaps[0]["domain"] == "message,action"
    assert gaps[0]["outcome"] == "degraded"
    assert gaps[0]["reason"] == "attached_unit_truncated"
    assert gaps[0]["sessions"] == 1
    assert result[0].id == "session"
