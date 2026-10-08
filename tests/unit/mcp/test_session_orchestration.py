"""Structured evidence must reach API and MCP without inventing accounting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from polylogue import Polylogue
from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_archive_fixture_prepare
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


def _seed(root: Path, *, reset: bool = False) -> str:
    with ArchiveStore(root) as archive:
        parsed = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="orchestration-example",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    model_name="recorded-model",
                    timestamp="2026-01-01T10:00:00+00:00",
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Agent",
                            tool_id="launch-1",
                            tool_input={"model": "requested-model", "prompt": "private instructions"},
                        ),
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="exec_command",
                            tool_id="bead-1",
                            tool_input={"cmd": "bd show example-a12.3"},
                        ),
                    ],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.USER,
                    text="Agent named requested-model finished example-fake.1",
                ),
            ],
            session_events=[
                ParsedSessionEvent(
                    event_type="turn_context",
                    timestamp="2026-01-01T09:59:00+00:00",
                    payload={"model": "configured-model"},
                ),
                *[
                    ParsedSessionEvent(
                        event_type="token_count",
                        timestamp=f"2026-01-01T10:0{index}:00+00:00",
                        payload={
                            "source_index": index,
                            "total_token_usage": {"input_tokens": count, "output_tokens": 10},
                            "last_token_usage": {"input_tokens": 5},
                            "rate_limits": {"primary": {"used_percent": 25, "window_minutes": 300}},
                        },
                    )
                    for index, count in enumerate([100, 100, 20 if reset else 120], start=1)
                ],
                ParsedSessionEvent(
                    event_type="rate_limits",
                    timestamp="2026-01-01T10:01:00+00:00",
                    payload={
                        "source_index": 1,
                        "rate_limits": {"primary": {"used_percent": 25, "window_minutes": 300}},
                    },
                ),
                ParsedSessionEvent(event_type="collab_agent_spawn_end", payload={"status": "completed"}),
            ],
        )
        write_fixture_index_session(archive._conn, parsed, archive_root=archive.index_db_path.parent)
    return "codex-session:orchestration-example"


async def _seed_acquired(root: Path) -> tuple[str, str]:
    """Retain native Codex bytes and converge them through the resident owner."""
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    records: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": "orchestration-example"}},
        {"type": "turn_context", "timestamp": "2026-01-01T09:59:00Z", "payload": {"model": "configured-model"}},
        {
            "type": "response_item",
            "timestamp": "2026-01-01T10:00:00Z",
            "payload": {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "model": "recorded-model",
                "content": [{"type": "output_text", "text": "delegating work"}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-01T10:00:00Z",
            "payload": {
                "type": "function_call",
                "call_id": "launch-1",
                "name": "Agent",
                "arguments": json.dumps({"model": "requested-model", "prompt": "private instructions"}),
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-01T10:00:00Z",
            "payload": {
                "type": "function_call",
                "call_id": "bead-1",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "bd show example-a12.3"}),
            },
        },
    ]
    for index, count in enumerate((100, 100, 120), start=1):
        records.append(
            {
                "type": "event_msg",
                "timestamp": f"2026-01-01T10:0{index}:00Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {"input_tokens": count, "output_tokens": 10},
                        "last_token_usage": {"input_tokens": 5},
                    },
                    "rate_limits": {"primary": {"used_percent": 25, "window_minutes": 300}},
                },
            }
        )
    records.extend(
        [
            {"type": "event_msg", "payload": {"type": "collab_agent_spawn_end", "status": "completed"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m2",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Agent named requested-model finished example-fake.1"}],
                },
            },
        ]
    )
    source_path = root.parent / "orchestration.jsonl"
    captured = ("\n".join(json.dumps(record) for record in records) + "\n").encode()
    source_path.write_bytes(captured)

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                capture_mode=Provider.CODEX,
                payload=source_path.read_bytes(),
                source_path=str(source_path),
                canonical_source_path=str(source_path),
                acquired_at_ms=1_767_000_000_000,
                file_mtime_ms=1_767_000_000_000,
            )
            archive.commit()
            return raw_id

    raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
    session_id = "codex-session:orchestration-example"
    assert session_id in {identity for receipt in receipts for identity in receipt.changed_session_ids}
    return session_id, raw_id


@pytest.mark.asyncio
async def test_api_and_mcp_preserve_counter_and_native_evidence(tmp_path: Path) -> None:
    """Summing cumulative samples, guessing a model, or bypassing the owner fails."""
    from polylogue.mcp.server import build_server

    root = tmp_path / "archive"
    session_id, raw_id = await _seed_acquired(root)
    owner = Polylogue(archive_root=root)
    evidence = await owner.get_session_orchestration(session_id)
    assert evidence is not None
    payload = evidence.model_dump(mode="json")
    assert payload["usage"]["tokens"] == {"input_tokens": 120, "output_tokens": 10}
    assert len(payload["usage"]["observations"]) == 3
    assert payload["usage"]["quota_consumed_tokens"] is None
    assert payload["rate_limits"][0]["windows"]["primary"]["used_percent"] == 25
    # Event rows carry stable event identity; acquisition belongs to the
    # independently verified session watermark below, never a guessed Raw.
    assert payload["rate_limits"][0]["raw_id"] is None
    assert payload["rate_limits"][0]["event_id"]
    assert payload["rate_limits"][0]["source_index"] == 6
    assert {row["bead_id"] for row in payload["bead_mentions"]} == {"example-a12.3"}
    request = next(row for row in payload["launches"] if row["basis"] == "tool_request")
    assert request["requested_model"] == "requested-model"
    assert request["actual_model"] is None
    assert request["child_session_id"] is None
    assert request["block_id"]
    assert {row["basis"] for row in payload["model_segments"]} == {"recorded_message_model", "configured_turn_model"}
    assert payload["coverage"]["ingestion_watermark"]["raw_id"] == raw_id
    assert payload["coverage"]["observed_at"]
    assert payload["coverage"]["complete"] is False
    assert "native_spawn_child_identity_not_retained" in payload["gaps"]
    assert "private instructions" not in json.dumps(payload)
    assert "/private/example.jsonl" not in json.dumps(payload)

    server = cast(MCPServerUnderTest, build_server())
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=owner),
    ):
        result = json.loads(
            await invoke_surface_async(
                server._tool_manager._tools["get"].fn,
                ref=f"session:{session_id}",
                projection="orchestration",
            )
        )
        missing = json.loads(
            await invoke_surface_async(
                server._tool_manager._tools["get"].fn,
                ref="session:codex-session:missing",
                projection="orchestration",
            )
        )
    assert {key: value for key, value in result.items() if key != "outcome"} == {
        key: value for key, value in payload.items() if key != "outcome"
    }
    assert result["outcome"]["state"] == payload["outcome"]
    assert result["outcome"]["detail"]["gaps"] == payload["gaps"]
    assert result["outcome"]["reason"] in payload["gaps"]
    assert missing["code"] == "not_found"


def test_cli_orchestration_view_renders_the_payload_api_and_mcp_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``read --view orchestration`` serves the same evidence as ``get``.

    The CLI view dispatches the declared ``read.orchestration`` operation,
    which is executed here on a pinned archive instead of a daemon.
    Anti-vacuity: remove the ``orchestration`` read view (the CLI then has no
    handler for it) or build the operation's evidence from anything other than
    ``read_session_orchestration`` and the rendered file stops matching the
    API payload; an unknown session must still refuse by name.
    """
    from unittest.mock import MagicMock

    from polylogue.cli.operation_kernel import OperationRequest
    from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS, run_read_view
    from polylogue.cli.read_views import orchestration as orchestration_view
    from polylogue.cli.read_views.base import ReadViewInvocation
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.operations.daemon_protocol import validate_operation_result
    from polylogue.operations.daemon_reads import execute_read_operation

    root = tmp_path / "archive"
    session_id = _seed(root)
    dispatched: list[str] = []

    def pinned_dispatch(config: object, request: OperationRequest, **_: object) -> tuple[dict[str, object], None]:
        operation = request.operation
        dispatched.append(operation)
        with ArchiveStore.open_existing(root) as archive:
            result = execute_read_operation(operation, dict(request.payload), archive=archive, serving_identity="test")
        validate_operation_result(operation, result)
        return result, None

    monkeypatch.setattr(orchestration_view, "dispatch_read", pinned_dispatch)
    assert READ_VIEW_HANDLERS["orchestration"].session_policy == "required"
    env = MagicMock()
    env.config = SimpleNamespace(archive_root=root)
    out_path = tmp_path / "orchestration.json"
    run_read_view(
        env,
        RootModeRequest.from_params({"id": session_id}),
        ReadViewInvocation(
            view="orchestration",
            session_id=session_id,
            output_format=None,
            destination="file",
            out_path=str(out_path),
        ),
    )

    import asyncio

    evidence = asyncio.run(Polylogue(archive_root=root).get_session_orchestration(session_id))
    assert evidence is not None
    assert dispatched == ["read.orchestration"]
    assert json.loads(out_path.read_text()) == evidence.model_dump(mode="json")

    with ArchiveStore.open_existing(root) as archive, pytest.raises(KeyError, match="Session not found"):
        execute_read_operation(
            "read.orchestration",
            {"session_id": "codex-session:missing"},
            archive=archive,
            serving_identity="test",
        )


@pytest.mark.asyncio
async def test_reset_does_not_claim_a_session_total(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id = await run_archive_fixture_prepare(lambda: _seed(root, reset=True))
    evidence = await Polylogue(archive_root=root).get_session_orchestration(session_id)
    assert evidence is not None
    assert evidence.usage["tokens"] is None
    assert evidence.usage["latest_cumulative"] == {"input_tokens": 20, "output_tokens": 10}
    assert "cumulative_usage_counter_reset" in evidence.gaps


@pytest.mark.asyncio
async def test_absent_measurements_are_null_not_zero(tmp_path: Path) -> None:
    root = tmp_path / "archive"

    def prepare_archive_1() -> None:
        with ArchiveStore(root) as archive:
            write_fixture_index_session(
                archive._conn,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="empty-evidence",
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
                ),
                archive_root=archive.index_db_path.parent,
            )

    await run_archive_fixture_prepare(prepare_archive_1)
    evidence = await Polylogue(archive_root=root).get_session_orchestration("codex-session:empty-evidence")
    assert evidence is not None
    assert evidence.usage["tokens"] is None
    assert evidence.model_segments == []
    assert evidence.rate_limits == []
    assert evidence.outcome == "degraded"


def test_parser_retains_quota_windows_and_native_spawn_identity_through_storage(tmp_path: Path) -> None:
    """The real parser/write/read route must retain quota independently of usage lowering."""
    from polylogue.api.sync import SyncPolylogue
    from polylogue.sources.parsers.codex import parse

    root = tmp_path / "archive"
    parsed = parse(
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "start"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T10:01:00Z",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 10}},
                    "rate_limits": {"primary": {"used_percent": 0, "window_minutes": 300, "resets_in_seconds": 90}},
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T10:02:00Z",
                "payload": {
                    "type": "collab_agent_spawn_end",
                    "new_thread_id": "native-child",
                    "new_agent_nickname": "display-only",
                },
            },
        ],
        "native-evidence",
    )
    with ArchiveStore(root) as archive:
        write_fixture_index_session(archive._conn, parsed, archive_root=archive.index_db_path.parent)
    with SyncPolylogue(archive_root=root) as owner:
        evidence = owner.get_session_orchestration("codex-session:native-evidence")
    assert evidence is not None
    assert evidence.usage["tokens"] == {"input_tokens": 100, "output_tokens": 10}
    assert evidence.rate_limits[0]["windows"] == {
        "primary": {"used_percent": 0, "window_minutes": 300, "resets_in_seconds": 90}
    }
    spawn = next(row for row in evidence.launches if row["basis"] == "native_spawn_event")
    assert spawn["child_native_id"] == "native-child"
    assert spawn["child_session_id"] is None
    assert spawn["actual_model"] is None


def test_projection_uses_stored_edges_and_excludes_inherited_calls() -> None:
    """Inherited launch calls and human prose must not create extra children or task evidence."""
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.analysis.topology import SessionTopology, TopologyEdge, TopologyEdgeKind, TopologyNode
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId

    session = Session(
        id=SessionId("codex-session:parent"),
        origin=Origin.CODEX_SESSION,
        messages=MessageCollection(
            messages=[
                Message(
                    id="codex-session:ancestor:n:m1",
                    role=Role.ASSISTANT,
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Agent",
                            "tool_input": {"model": "inherited"},
                        }
                    ],
                ),
                Message(
                    id="codex-session:parent:n:m2",
                    role=Role.ASSISTANT,
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "exec_command",
                            "tool_input": {"cmd": "bd create --title example-title"},
                        }
                    ],
                ),
            ]
        ),
    )
    topology = SessionTopology(
        target_id=session.id,
        root_id=session.id,
        nodes=(TopologyNode(session_id=session.id), TopologyNode(session_id=SessionId("codex-session:child"))),
        edges=(
            TopologyEdge(
                parent_id=session.id, child_id=SessionId("codex-session:child"), kind=TopologyEdgeKind.SUBAGENT
            ),
        ),
    )
    evidence = build_session_orchestration(str(session.id), topology, messages=session.messages)
    assert [row["session_id"] for row in evidence.children] == ["codex-session:child"]
    assert evidence.launches == []
    assert evidence.bead_mentions == []
    assert evidence.coverage["message_count"] == 1


def test_topology_truncation_keeps_every_edge_endpoint() -> None:
    """The bounded topology is still a graph over only retained nodes.

    Anti-vacuity: the reported root-with-1,000-children shape used to retain
    1,000 edges while omitting the final child node; every emitted edge now
    has both endpoints in the emitted node set.
    """
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.analysis.topology import SessionTopology, TopologyEdge, TopologyEdgeKind, TopologyNode
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId

    root = SessionId("codex-session:large-root")
    children = [SessionId(f"codex-session:child-{index}") for index in range(1000)]
    topology = SessionTopology(
        target_id=root,
        root_id=root,
        nodes=(TopologyNode(session_id=root), *(TopologyNode(session_id=child) for child in children)),
        edges=tuple(TopologyEdge(parent_id=root, child_id=child, kind=TopologyEdgeKind.SUBAGENT) for child in children),
    )
    session = Session(id=root, origin=Origin.CODEX_SESSION, messages=MessageCollection(messages=[]))

    evidence = build_session_orchestration(str(session.id), topology, messages=session.messages)
    payload = evidence.topology
    assert payload is not None
    nodes = cast(list[dict[str, object]], payload["nodes"])
    edges = cast(list[dict[str, object]], payload["edges"])
    retained = {node["session_id"] for node in nodes}
    assert len(retained) == 1000
    assert len(edges) == 999
    assert all(edge["parent_id"] in retained and edge["child_id"] in retained for edge in edges)


def test_topology_truncation_keeps_unresolved_edges_of_retained_children() -> None:
    """An unresolved parent reference survives truncation with its retained child.

    Anti-vacuity: require a retained parent node for every edge and the
    unresolved edge (``parent_id`` is None) is dropped, hiding the unresolved
    relationship behind a generic observation-limit gap.
    """
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.analysis.topology import SessionTopology, TopologyEdge, TopologyEdgeKind, TopologyNode
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId

    root = SessionId("codex-session:wide-root")
    children = [SessionId(f"codex-session:wide-child-{index}") for index in range(1000)]
    unresolved = TopologyEdge(
        child_id=root,
        parent_id=None,
        dst_origin="codex-session",
        dst_native_id="missing-parent",
        kind=TopologyEdgeKind.CONTINUATION,
        resolved=False,
    )
    topology = SessionTopology(
        target_id=root,
        root_id=root,
        nodes=(TopologyNode(session_id=root), *(TopologyNode(session_id=child) for child in children)),
        edges=(
            unresolved,
            *(TopologyEdge(parent_id=root, child_id=child, kind=TopologyEdgeKind.SUBAGENT) for child in children),
        ),
    )
    session = Session(id=root, origin=Origin.CODEX_SESSION, messages=MessageCollection(messages=[]))

    payload = build_session_orchestration(str(session.id), topology, messages=session.messages).topology

    assert payload is not None
    edges = cast(list[dict[str, object]], payload["edges"])
    assert any(edge["child_id"] == root and edge["parent_id"] is None for edge in edges)
    retained = {node["session_id"] for node in cast(list[dict[str, object]], payload["nodes"])}
    assert all(edge["child_id"] in retained for edge in edges)


def test_unmeasured_token_lanes_are_a_distinct_bucket_from_measured_zero() -> None:
    """A nullable lane must neither crash the projection nor read as a measured zero.

    Anti-vacuity: ``Message``'s four token lanes default to ``None``, so a
    fixture whose lanes are all populated exercises nothing. This one mixes
    the three states that must stay apart -- measured-positive, measured-zero
    and unmeasured -- and fails if the projection orders ``None`` against an
    int (``TypeError``), folds ``None`` into ``0`` (the unmeasured message
    would vanish from ``messages_with_unmeasured_token_lanes``), or folds a
    measured ``0`` into the unmeasured bucket (the count would read 2).
    """
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId

    session = Session(
        id=SessionId("codex-session:lanes"),
        origin=Origin.CODEX_SESSION,
        messages=MessageCollection(
            messages=[
                Message(
                    id="codex-session:lanes:n:measured",
                    role=Role.ASSISTANT,
                    input_tokens=10,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                ),
                Message(
                    id="codex-session:lanes:n:partial",
                    role=Role.ASSISTANT,
                    input_tokens=5,
                    output_tokens=7,
                    cache_read_tokens=0,
                    cache_write_tokens=None,
                ),
            ]
        ),
    )

    evidence = build_session_orchestration(str(session.id), None, messages=session.messages)

    usage = evidence.usage
    assert usage["message_tokens_lower_bound"] == {"input_tokens": 15, "output_tokens": 7}
    assert usage["messages_with_positive_tokens"] == 2
    assert usage["messages_with_unmeasured_token_lanes"] == 1
    assert usage["unmeasured_token_lane_messages"] == {"cache_write_tokens": 1}
    assert usage["message_token_lane_scope"] == 2
    assert "message_token_lanes_unmeasured" in evidence.gaps


def test_all_measured_lanes_report_no_unmeasured_bucket() -> None:
    """The other direction: measured zeros must not manufacture an unmeasured gap."""
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session
    from polylogue.core.enums import Origin
    from polylogue.core.types import SessionId

    session = Session(
        id=SessionId("codex-session:measured"),
        origin=Origin.CODEX_SESSION,
        messages=MessageCollection(
            messages=[
                Message(
                    id="codex-session:measured:n:m1",
                    role=Role.ASSISTANT,
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                )
            ]
        ),
    )

    evidence = build_session_orchestration(str(session.id), None, messages=session.messages)

    assert evidence.usage["messages_with_unmeasured_token_lanes"] == 0
    assert evidence.usage["unmeasured_token_lane_messages"] is None
    assert "message_token_lanes_unmeasured" not in evidence.gaps


def test_truncated_orchestration_children_belong_to_retained_topology() -> None:
    """Keeping the pre-truncation child list returns a child with no retained node."""
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.analysis.topology import SessionTopology, TopologyEdge, TopologyEdgeKind, TopologyNode
    from polylogue.core.types import SessionId

    root = SessionId("codex-session:root")
    children = [SessionId(f"codex-session:child-{i:04d}") for i in range(1000)]
    topology = SessionTopology(
        target_id=root,
        root_id=root,
        nodes=(TopologyNode(session_id=root), *(TopologyNode(session_id=child) for child in children)),
        edges=tuple(TopologyEdge(parent_id=root, child_id=child, kind=TopologyEdgeKind.SUBAGENT) for child in children),
    )
    evidence = build_session_orchestration(str(root), topology)
    assert evidence.topology is not None
    nodes = cast(list[dict[str, object]], evidence.topology["nodes"])
    retained = {node["session_id"] for node in nodes}
    assert evidence.children
    assert {child["session_id"] for child in evidence.children} <= retained
    assert "observation_limit" in evidence.gaps


@pytest.mark.asyncio
async def test_orchestration_streams_own_records_without_hydrating_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded projection must not load the whole session to build itself.

    Anti-vacuity: route ``get_session_orchestration`` back through
    ``repository.get`` (the full lineage-composed transcript, attachments and
    events) and the patched hydrator fails the call. A page size of two forces
    every keyset stream across several pages, so a cursor that skips or
    repeats a row changes the counted sections.
    """
    import polylogue.operations.orchestration as orchestration_reads

    monkeypatch.setattr(orchestration_reads, "_PAGE_SIZE", 2)
    root = tmp_path / "archive"
    session_id = await run_archive_fixture_prepare(lambda: _seed(root))
    owner = Polylogue(archive_root=root)

    async def refuse_full_session(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("orchestration evidence hydrated the full session")

    monkeypatch.setattr(owner.repository, "get", refuse_full_session)
    evidence = await owner.get_session_orchestration(session_id)
    assert evidence is not None
    payload = evidence.model_dump(mode="json")
    assert payload["coverage"]["message_count"] == 2
    assert payload["coverage"]["event_count"] == 3
    assert payload["coverage"]["section_counts"] == {
        "children": 0,
        "launches": 2,
        "model_segments": 2,
        "bead_mentions": 1,
        "rate_limits": 1,
        "usage_observations": 3,
    }
    assert payload["usage"]["tokens"] == {"input_tokens": 120, "output_tokens": 10}
    assert len(payload["usage"]["observations"]) == 3
    assert {row["bead_id"] for row in payload["bead_mentions"]} == {"example-a12.3"}
    assert {row["basis"] for row in payload["launches"]} == {"tool_request", "native_spawn_event"}
    assert await owner.get_session_orchestration("codex-session:missing") is None


def test_streamed_sections_keep_full_counts_but_bounded_rows() -> None:
    """A long stream is counted in full while each section retains ``_LIMIT`` rows."""
    from collections.abc import Iterator

    from polylogue.analysis.orchestration_evidence import _LIMIT, build_session_orchestration
    from polylogue.archive.message.models import Message

    session_id = "codex-session:long"
    total = _LIMIT + 5

    def messages() -> Iterator[Message]:
        for index in range(total):
            yield Message(
                id=f"{session_id}:n:m{index}",
                role=Role.ASSISTANT,
                model_name=f"model-{index % 2}",
                blocks=[{"type": "tool_use", "tool_name": "Agent", "block_id": f"b{index}", "tool_input": {}}],
            )

    evidence = build_session_orchestration(session_id, None, messages=messages())
    section_counts = cast(dict[str, int], evidence.coverage["section_counts"])
    assert evidence.coverage["message_count"] == total
    assert section_counts["launches"] == total
    assert section_counts["model_segments"] == total
    assert len(evidence.launches) == _LIMIT
    assert len(evidence.model_segments) == _LIMIT
    assert set(cast(list[str], evidence.coverage["truncated_sections"])) == {"launches", "model_segments"}
    assert "observation_limit" in evidence.gaps
