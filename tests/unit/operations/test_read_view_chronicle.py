from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.operations.read_view_chronicle import _chronicle_edges, execute_chronicle_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def test_chronicle_edges_reads_composed_pages_and_counts_only_authored_dialogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        SimpleNamespace(
            id=f"message-{index}",
            role=role,
            message_type=message_type,
            material_origin=material_origin,
        )
        for index in range(14)
        for role, message_type, material_origin in [
            (
                "tool" if index == 2 else "user",
                "reasoning" if index == 4 else "message",
                "tool_authored" if index == 2 else "human_authored",
            )
        ]
    ]

    class PinnedArchive:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def has_prefix_lineage(self, session_id: str) -> bool:
            return True

        def read_session_page(self, session_id: str, *, limit: int, offset: int) -> SimpleNamespace:
            assert session_id == "origin:session"
            self.offsets.append(offset)
            return SimpleNamespace(messages=messages[offset : offset + 2], total_message_count=len(messages))

    archive = PinnedArchive()
    monkeypatch.setattr(
        "polylogue.operations.read_view_chronicle.archive_message_to_domain",
        lambda message, *, origin: message,
    )

    first, last, total = _chronicle_edges(
        cast(ArchiveStore, archive), "origin:session", 1, origin=Origin.from_string("codex")
    )

    assert [message.id for message in first] == ["message-0", "message-1", "message-3", "message-5", "message-6"]
    assert [message.id for message in last] == ["message-9", "message-10", "message-11", "message-12", "message-13"]
    assert total == 12
    assert archive.offsets == [0, 2, 4, 6, 8, 10, 12]


def test_chronicle_result_refuses_to_exceed_operation_wire_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.operations import read_view_chronicle

    monkeypatch.setattr("polylogue.archive.query.archive_execution._archive_summaries", lambda *args, **kwargs: [])
    monkeypatch.setattr(read_view_chronicle, "MAX_OPERATION_RESULT_BYTES", 1)
    monkeypatch.setattr(
        read_view_chronicle,
        "build_chronicle_projection_payload",
        lambda sessions, *, edge_limit: SimpleNamespace(model_dump=lambda mode: {"large": "payload"}),
    )

    with pytest.raises(ValueError, match="above the 1-byte operation result limit"):
        read_view_chronicle.execute_chronicle_read(
            {"session_id": None, "params": {}, "projection": {}},
            archive=Mock(archive_root="/tmp/archive"),
        )


def test_chronicle_operation_applies_exclude_text_before_offset_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.archive.query.archive_execution as archive_execution
    import polylogue.operations.read_view_chronicle as read_view_chronicle

    rows = [
        SimpleNamespace(session_id=f"codex-session:{index}", display_label=None, display_label_source=None)
        for index in (1, 2, 3)
    ]
    summaries = {
        row.session_id: SessionSummary(
            id=row.session_id,
            origin=Origin.from_string("codex-session"),
            updated_at=datetime(2026, 1, index, tzinfo=timezone.utc),
        )
        for index, row in enumerate(rows, start=1)
    }
    texts = {
        "codex-session:1": "keep first",
        "codex-session:2": "needle to exclude",
        "codex-session:3": "keep third",
    }
    archive = Mock(archive_root="/tmp/archive")
    monkeypatch.setattr(archive_execution, "_archive_summaries", lambda *args, **kwargs: rows)
    monkeypatch.setattr(read_view_chronicle, "archive_summary_to_domain", lambda row: summaries[row.session_id])
    monkeypatch.setattr(
        "polylogue.archive.hydration.archive_envelope_to_session",
        lambda envelope, **kwargs: SimpleNamespace(
            id=envelope.session_id,
            messages=[SimpleNamespace(text=texts[envelope.session_id])],
        ),
    )
    monkeypatch.setattr(read_view_chronicle, "_chronicle_edges", lambda *args, **kwargs: ([], [], 0))
    archive.read_session.side_effect = lambda session_id: SimpleNamespace(session_id=session_id)

    result = execute_chronicle_read(
        {"params": {"exclude_text": ["needle"], "sort": "date", "reverse": True, "offset": 1, "limit": 1}},
        archive=archive,
        vector_provider=None,
    )

    assert result["view"] == "chronicle"
    body = result["payload"]
    assert isinstance(body, dict)
    assert body["session_count"] == 1
    sessions = body["sessions"]
    assert isinstance(sessions, list)
    assert sessions[0]["session_id"] == "codex-session:3"
    assert archive.read_session.call_count == 3


def test_ranked_chronicle_count_sort_keeps_the_requested_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5695): clear the window for a ranked count sort
    and the semantic pool falls back to its default size, so ``offset=150``
    removes every candidate even when more matches exist."""
    import polylogue.archive.query.archive_execution as archive_execution
    from polylogue.archive.query.plan import SessionQueryPlan

    fetched: list[tuple[SessionQueryPlan, object]] = []

    def capture(plan: SessionQueryPlan, *_args: object, **kwargs: object) -> list[object]:
        fetched.append((plan, kwargs.get("complete")))
        return []

    monkeypatch.setattr(archive_execution, "_archive_summaries", capture)
    execute_chronicle_read(
        {"params": {"similar_text": "neutral probe", "sort": "messages", "offset": 150, "limit": 1}},
        archive=Mock(archive_root="/tmp/archive"),
        vector_provider=None,
    )

    ((plan, complete),) = fetched
    assert complete is False
    assert plan.offset == 150
    assert plan.limit == 1


@pytest.mark.parametrize(
    ("params", "scan"),
    [
        ({"sort": "messages"}, True),
        ({"sort": "tokens", "limit": 1}, True),
        ({"sort": "date"}, False),
        ({"sort": "messages", "similar_text": "neutral probe"}, False),
    ],
)
def test_a_complete_chronicle_count_sort_is_admitted_as_scan_work(params: dict[str, object], scan: bool) -> None:
    """Anti-vacuity (Codex P2, #5695): classify ``read.chronicle`` by its spec
    alone and a count-sorted page that hydrates every candidate runs under the
    interactive-read class and its two-second deadline."""
    from polylogue.operations.daemon_reads import read_is_archive_scan

    assert read_is_archive_scan("read.chronicle", {"params": params}) is scan
    assert read_is_archive_scan("cli.query", {"params": params}) is False
