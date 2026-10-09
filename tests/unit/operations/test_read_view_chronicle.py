from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Origin
from polylogue.operations.read_view_chronicle import ChronicleEdges, chronicle_edges, execute_chronicle_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.builders import make_conv, make_msg


def _serving(rows: list[Any]) -> Any:
    """A fake candidate fetch that streams to ``on_batch`` as the real one does."""

    def fetch(*_args: object, on_batch: Any = None, keep: Any = None, **_kwargs: object) -> list[Any]:
        selected = keep(rows) if keep is not None else rows
        if on_batch is None:
            return selected
        on_batch(selected)
        return []

    return fetch


def test_chronicle_edges_reads_composed_pages_and_counts_only_authored_dialogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        SimpleNamespace(
            id=f"message-{index}",
            text="authored prose",
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

        def check_operation_read(self) -> None:
            pass

        def read_session_page(self, session_id: str, *, limit: int, offset: int) -> SimpleNamespace:
            assert session_id == "origin:session"
            self.offsets.append(offset)
            return SimpleNamespace(
                messages=messages[offset : offset + 2],
                total_message_count=len(messages),
                lineage_complete=True,
                lineage_truncation_reason=None,
            )

    archive = PinnedArchive()
    monkeypatch.setattr(
        "polylogue.operations.read_view_chronicle.archive_message_to_domain",
        lambda message, *, origin: message,
    )

    edges = chronicle_edges(cast(ArchiveStore, archive), "origin:session", 1, origin=Origin.from_string("codex"))

    assert [message.id for message in edges.first] == ["message-0"]
    assert [message.id for message in edges.last] == ["message-13"]
    assert edges.total == 12
    assert archive.offsets == [0, 2, 4, 6, 8, 10, 12]


def test_chronicle_result_keeps_a_large_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.operations import read_view_chronicle

    value = "λ" * (9 * 1024 * 1024)
    monkeypatch.setattr("polylogue.archive.query.archive_execution._archive_summaries", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        read_view_chronicle,
        "build_chronicle_projection_payload",
        lambda sessions, *, edge_limit: SimpleNamespace(model_dump=lambda mode: {"large": value}),
    )
    result = read_view_chronicle.execute_chronicle_read(
        {"session_id": None, "params": {}, "projection": {}}, archive=Mock(archive_root="/tmp/archive")
    )
    assert result["payload"] == {"large": value}


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
    monkeypatch.setattr(archive_execution, "_archive_summaries", _serving(rows))
    monkeypatch.setattr(archive_execution, "archive_summary_to_domain", lambda row: summaries[row.session_id])
    monkeypatch.setattr(
        "polylogue.operations.read_view_selection.archive_summary_to_domain", lambda row: summaries[row.session_id]
    )
    monkeypatch.setattr(
        "polylogue.archive.query.archive_execution.archive_envelope_to_session",
        lambda envelope, **kwargs: SimpleNamespace(
            id=envelope.session_id,
            messages=[SimpleNamespace(text=texts[envelope.session_id])],
        ),
    )
    monkeypatch.setattr(
        read_view_chronicle, "chronicle_edges", lambda *args, **kwargs: ChronicleEdges([], [], 0, True, None)
    )
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


def test_ranked_chronicle_count_sort_uses_full_composed_order_before_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared ranked reader settles composed comparison keys before paging."""
    import polylogue.archive.query.archive_execution as archive_execution
    from polylogue.archive.query.plan import SessionQueryPlan

    fetched: list[tuple[SessionQueryPlan, object, object]] = []

    def capture(plan: SessionQueryPlan, *_args: object, **kwargs: object) -> list[object]:
        fetched.append((plan, kwargs.get("complete"), kwargs.get("full_sort")))
        return []

    monkeypatch.setattr(archive_execution, "_archive_summaries", capture)
    execute_chronicle_read(
        {"params": {"similar_text": "neutral probe", "sort": "messages", "offset": 150, "limit": 1}},
        archive=Mock(archive_root="/tmp/archive"),
        vector_provider=None,
    )

    ((plan, complete, full_sort),) = fetched
    assert complete is False
    assert full_sort is True
    assert plan.offset == 150
    assert plan.limit == 1


@pytest.mark.parametrize(
    ("params", "scan"),
    [
        ({"sort": "messages"}, True),
        ({"sort": "tokens", "limit": 1}, True),
        ({"sort": "date"}, False),
        ({"sort": "messages", "similar_text": "neutral probe"}, True),
    ],
)
def test_a_complete_chronicle_count_sort_is_admitted_as_scan_work(params: dict[str, object], scan: bool) -> None:
    """Anti-vacuity (Codex P2, #5695): classify ``read.chronicle`` by its spec
    alone and a count-sorted page that hydrates every candidate runs under the
    interactive-read class and its two-second deadline."""
    from polylogue.operations.daemon_reads import read_is_archive_scan

    assert read_is_archive_scan("read.chronicle", {"params": params}) is scan
    assert read_is_archive_scan("cli.query", {"params": params}) is False


@pytest.mark.parametrize("mode", ["count", "stats", "stats_by"])
def test_aggregate_selection_uses_scan_admission_even_with_small_page_limit(mode: str) -> None:
    from polylogue.operations.daemon_reads import read_is_archive_scan

    assert read_is_archive_scan("query.aggregate", {"mode": mode, "params": {"limit": 1}})


def test_a_chronicle_count_sort_hydrates_each_candidate_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A count-sorted chronicle page reads every candidate once, not twice.

    Anti-vacuity (Codex P2, #5695): hydrate for the content filters and then
    again for the composed order and ``read_session`` runs twice per session.
    """
    import polylogue.archive.query.archive_execution as archive_execution
    import polylogue.operations.read_view_chronicle as read_view_chronicle

    rows = [
        SimpleNamespace(session_id=f"codex-session:{index}", display_label=None, display_label_source=None)
        for index in range(5)
    ]
    summaries = {
        row.session_id: SessionSummary(
            id=row.session_id,
            origin=Origin.from_string("codex-session"),
            updated_at=datetime(2026, 1, index + 1, tzinfo=timezone.utc),
        )
        for index, row in enumerate(rows)
    }
    archive = Mock(archive_root="/tmp/archive")
    monkeypatch.setattr(archive_execution, "_archive_summaries", _serving(rows))
    monkeypatch.setattr(archive_execution, "archive_summary_to_domain", lambda row: summaries[row.session_id])
    monkeypatch.setattr(
        "polylogue.operations.read_view_selection.archive_summary_to_domain", lambda row: summaries[row.session_id]
    )
    monkeypatch.setattr(
        "polylogue.archive.hydration.archive_envelope_to_session",
        lambda envelope, **kwargs: make_conv(
            id=envelope.session_id,
            messages=[make_msg(id=f"m{i}", text="x") for i in range(int(envelope.session_id.rsplit(":", 1)[1]) + 1)],
        ),
    )
    monkeypatch.setattr(
        read_view_chronicle, "chronicle_edges", lambda *args, **kwargs: ChronicleEdges([], [], 0, True, None)
    )
    archive.read_session.side_effect = lambda session_id: SimpleNamespace(session_id=session_id)

    result = execute_chronicle_read({"params": {"sort": "messages", "limit": 1}}, archive=archive, vector_provider=None)

    body = result["payload"]
    assert isinstance(body, dict)
    sessions = body["sessions"]
    assert isinstance(sessions, list)
    assert sessions[0]["session_id"] == "codex-session:4"
    assert archive.read_session.call_count == len(rows)


def test_a_sampled_chronicle_count_sort_samples_every_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampled count-sorted chronicle draws from every qualified session.

    Anti-vacuity (Codex P2, #5695): keep only the top ``offset + limit`` before
    sampling and the sample can only ever return the single largest session.
    """
    import polylogue.archive.query.archive_execution as archive_execution
    import polylogue.operations.read_view_chronicle as read_view_chronicle
    from polylogue.archive.query.plan import SessionQueryPlan

    rows = [
        SimpleNamespace(session_id=f"codex-session:{index}", display_label=None, display_label_source=None)
        for index in range(5)
    ]
    summaries = {
        row.session_id: SessionSummary(
            id=row.session_id,
            origin=Origin.from_string("codex-session"),
            updated_at=datetime(2026, 1, index + 1, tzinfo=timezone.utc),
        )
        for index, row in enumerate(rows)
    }
    archive = Mock(archive_root="/tmp/archive")
    monkeypatch.setattr(archive_execution, "_archive_summaries", _serving(rows))
    monkeypatch.setattr(archive_execution, "archive_summary_to_domain", lambda row: summaries[row.session_id])
    monkeypatch.setattr(
        "polylogue.operations.read_view_selection.archive_summary_to_domain", lambda row: summaries[row.session_id]
    )
    monkeypatch.setattr(
        "polylogue.archive.hydration.archive_envelope_to_session",
        lambda envelope, **kwargs: make_conv(
            id=envelope.session_id,
            messages=[make_msg(id=f"m{i}", text="x") for i in range(int(envelope.session_id.rsplit(":", 1)[1]) + 1)],
        ),
    )
    monkeypatch.setattr(
        read_view_chronicle, "chronicle_edges", lambda *args, **kwargs: ChronicleEdges([], [], 0, True, None)
    )
    archive.read_session.side_effect = lambda session_id: SimpleNamespace(session_id=session_id)
    offered: list[int] = []
    original = SessionQueryPlan._finalize

    def finalize(self: SessionQueryPlan, items: list[Any]) -> list[Any]:
        offered.append(len(items))
        return original(self, items)

    monkeypatch.setattr(SessionQueryPlan, "_finalize", finalize)

    execute_chronicle_read(
        {"params": {"sort": "messages", "limit": 1, "sample": 1}}, archive=archive, vector_provider=None
    )

    # A reservoir of the sample's size: only one hydrated session is held.
    assert offered == [1]


def test_a_null_chronicle_limit_is_the_default_page() -> None:
    """``limit: null`` compiles to the five-session default, not an unbounded scan.

    Anti-vacuity (Codex P2, #5695): leave the null in place and the plan has
    ``limit=None``.
    """
    from polylogue.operations.read_view_chronicle import _chronicle_plan

    plan = _chronicle_plan({"params": {"sort": "messages", "limit": None}}, vector_provider=None)

    assert plan.limit == 5


def test_chronicle_scan_classification_does_not_impose_an_execution_deadline() -> None:
    """The scan keeps separate admission capacity while both read shapes stay unbounded."""
    from polylogue.operations.daemon_protocol import daemon_operation_spec
    from polylogue.operations.daemon_reads import read_is_archive_scan

    assert read_is_archive_scan("read.chronicle", {"params": {"sort": "messages"}})
    assert not read_is_archive_scan("read.chronicle", {"params": {"sort": "date"}})
    spec = daemon_operation_spec("read.chronicle")
    assert spec is not None and spec.deadline_s is None


@pytest.mark.parametrize("count", [20, 60, 300])
def test_physical_chronicle_keeps_the_actual_tail(tmp_path: Any, count: int) -> None:
    """The tail is the last authored rows, including beyond a storage page."""
    from datetime import UTC, timedelta

    from tests.infra.storage_records import SessionBuilder

    builder = SessionBuilder(tmp_path / "index.db", "physical-edges").provider("codex")
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(count):
        builder.add_message(
            message_id=f"edge-{index}",
            text=f"prose-{index}",
            timestamp=(instant + timedelta(seconds=index)).isoformat(),
            material_origin="human_authored",
        )
    builder.save()
    with ArchiveStore(tmp_path, read_only=True) as archive:
        result = execute_chronicle_read(
            {"session_id": builder.native_session_id(), "projection": {"edge_limit": 8}}, archive=archive
        )
    body = cast(dict[str, Any], result["payload"])["sessions"][0]
    assert [row["text"] for row in body["first_messages"]] == [f"prose-{index}" for index in range(8)]
    assert [row["text"] for row in body["last_messages"]] == [f"prose-{index}" for index in range(count - 8, count)]
    assert "empty_text_messages_omitted" not in body["caveats"]


def test_chronicle_pages_past_empty_authored_text(tmp_path: Any) -> None:
    from tests.infra.storage_records import SessionBuilder

    builder = SessionBuilder(tmp_path / "index.db", "empty-edges").provider("codex")
    for index in range(20):
        builder.add_message(message_id=f"empty-{index}", text="", material_origin="human_authored")
    for index in range(4):
        builder.add_message(message_id=f"prose-{index}", text=f"prose-{index}", material_origin="human_authored")
    builder.save()
    with ArchiveStore(tmp_path, read_only=True) as archive:
        result = execute_chronicle_read(
            {"session_id": builder.native_session_id(), "projection": {"edge_limit": 2}}, archive=archive
        )
    body = cast(dict[str, Any], result["payload"])["sessions"][0]
    assert [row["text"] for row in body["first_messages"]] == ["prose-0", "prose-1"]
    assert [row["text"] for row in body["last_messages"]] == ["prose-2", "prose-3"]
    assert body["total_matching_messages"] == 24
