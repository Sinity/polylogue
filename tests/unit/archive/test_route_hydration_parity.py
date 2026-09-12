"""One hydration owner, one set of contract masks (polylogue-blpir).

Four routes used to restate the same ``ArchiveSessionSummary -> SessionSummary``
mapping and two restated ``ArchiveMessageRow -> Message``, each populating a
different subset. These tests seed one provider-shaped session whose every
mapped message/block/attachment/topology/title/branch/repository field carries a
*nondefault* value, then compare what the API, archive-query, exact CLI, MCP and
HTTP routes report against the canonical domain object, restricted to each
route's declared mask.

Anti-vacuity: every parity assertion is driven by a declared mask, and the
mutation tests below delete one disposition at a time (``stop_reason``,
``is_active_leaf``, ``parent_message_id``, the block outcome reason,
``parent_id``, ``branch_type``, ``display_name``, ``title_source``) and require
a *production route* -- ``Polylogue.get_session``/``get_session_summary``, the
exact CLI read, the HTTP detail payload -- to report something different. A
test that only re-read the hydrator, or compared two default-valued payloads,
would stay green under those mutations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from polylogue.archive import hydration
from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.hydration import (
    ROW_DISPOSITIONS,
    archive_message_to_domain,
    archive_summary_to_domain,
    exposed_domain_names,
    unmapped_row_fields,
)
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.core.enums import (
    BlockType,
    MaterialOrigin,
    Provider,
    Role,
    ToolOutcome,
    ToolResultUnknownReason,
)
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore
from polylogue.storage.sqlite.archive_tiers.archive_query_reads import ArchiveMessageQueryRow
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveAttachmentRow,
    ArchiveBlockRow,
    ArchiveMessageRow,
    ArchiveSessionEnvelope,
)
from polylogue.surfaces.payloads import _MESSAGE_MASK, _SESSION_SUMMARY_MASK, MessageRenderEnvelope
from tests.infra.live_ingest import write_session_sync
from tests.infra.storage_records import db_setup

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import SessionSummary

_ROW_TYPES: tuple[type, ...] = (
    ArchiveBlockRow,
    ArchiveMessageRow,
    ArchiveAttachmentRow,
    ArchiveSessionSummary,
    ArchiveSessionEnvelope,
    ArchiveMessageQueryRow,
)

_PARENT_NATIVE_ID = "route-matrix-parent"
_CHILD_NATIVE_ID = "route-matrix-child"
_ORIGIN = "claude-code-session"
_PARENT_SESSION_ID = f"{_ORIGIN}:{_PARENT_NATIVE_ID}"


# ---------------------------------------------------------------------------
# Declaration coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row_type", _ROW_TYPES, ids=[row.__name__ for row in _ROW_TYPES])
def test_every_archive_row_field_declares_a_hydration_disposition(row_type: type) -> None:
    """Each semantic archive field is exposed, excluded or delegated by name."""
    undeclared, stale = unmapped_row_fields(row_type)
    assert undeclared == (), f"{row_type.__name__} fields with no hydration disposition: {undeclared}"
    assert stale == (), f"{row_type.__name__} dispositions naming fields that no longer exist: {stale}"


def test_an_undeclared_archive_row_field_fails_the_message_coverage_check() -> None:
    """Adding a field without a disposition must be a failure, not a silent drop."""
    widened = make_dataclass(
        "ArchiveMessageRow",
        [*((field.name, field.type) for field in fields(ArchiveMessageRow)), ("newly_added_semantics", "str | None")],
    )
    undeclared, stale = unmapped_row_fields(widened)
    assert undeclared == ("newly_added_semantics",)
    assert stale == ()


def test_every_exposed_disposition_names_a_real_domain_field() -> None:
    """An exposed disposition must reach a declared domain model field."""
    from polylogue.archive.attachment.models import Attachment
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session, SessionSummary

    targets = {
        "ArchiveBlockRow": None,  # domain blocks are open mappings, not a model
        "ArchiveMessageRow": Message,
        "ArchiveMessageQueryRow": Message,
        "ArchiveAttachmentRow": Attachment,
        "ArchiveSessionSummary": SessionSummary,
        "ArchiveSessionEnvelope": Session,
    }
    for row_name, dispositions in ROW_DISPOSITIONS.items():
        model = targets[row_name]
        if model is None:
            continue
        for field_name, disposition in dispositions.items():
            if disposition.kind != "exposed":
                assert disposition.reason, f"{row_name}.{field_name} must say why it is not exposed"
                continue
            assert disposition.domain_name in model.model_fields, (
                f"{row_name}.{field_name} maps to {disposition.domain_name!r}, which {model.__name__} does not declare"
            )


def test_retained_title_policy_is_a_named_input_not_a_forked_mapping() -> None:
    """The one semantic difference between session and summary title handling.

    A full session read suppresses a legacy ``TitleSource.PATH`` title (a
    structural fallback, not provider evidence) and lets the display-label
    projection speak; the summary keeps the stored row value beside
    ``display_label``. That difference is this single policy function -- it
    forks no other field. Red if the policy starts accepting ``path`` (or
    starts rejecting real provider evidence).
    """
    from polylogue.core.enums import TitleSource

    assert hydration.archive_provider_title("Real title", TitleSource.ORIGIN.value) == "Real title"
    assert hydration.archive_provider_title("Derived title", TitleSource.HEURISTIC.value) == "Derived title"
    assert hydration.archive_provider_title("/home/me/project", TitleSource.PATH.value) is None
    assert hydration.archive_provider_title("Untyped", None) is None


def test_one_display_text_policy_serves_every_archive_row_route() -> None:
    """No route may fork the flattened ``message.text`` formula.

    Both hydrators take the policy as a default argument and neither inlines a
    join, so a surface that wants different display text supplies a different
    policy instead of copying the semantic block mapping.
    """
    import inspect

    for hydrator in (hydration.archive_message_to_domain, hydration.archive_message_query_row_to_domain):
        default = inspect.signature(hydrator).parameters["display_text"].default
        assert default is hydration.archive_display_text

    blocks = (
        ArchiveBlockRow(block_id="m:0", message_id="m", block_type="text", text="first"),
        ArchiveBlockRow(block_id="m:1", message_id="m", block_type="thinking", text="second"),
        ArchiveBlockRow(block_id="m:2", message_id="m", block_type="text", text=None),
    )
    assert hydration.archive_display_text(blocks) == "first\n\nsecond"


# ---------------------------------------------------------------------------
# The seeded route matrix
# ---------------------------------------------------------------------------


def _parent_session() -> ParsedSession:
    from polylogue.core.enums import TitleSource

    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=_PARENT_NATIVE_ID,
        title="Route matrix parent",
        title_source=TitleSource.ORIGIN,
        messages=[
            ParsedMessage(
                provider_message_id="parent-m0",
                role=Role.USER,
                text="parent turn",
                timestamp="2026-03-01T09:00:00+00:00",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent turn")],
            )
        ],
    )


def _child_session() -> ParsedSession:
    """A child session with a nondefault value in every mapped family."""
    from polylogue.core.enums import BranchType, TitleSource

    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=_CHILD_NATIVE_ID,
        title="Route matrix child",
        title_source=TitleSource.HEURISTIC,
        title_ref="message:route-matrix-child:0",
        display_name="greedy-squishing-hamming",
        parent_session_provider_id=_PARENT_NATIVE_ID,
        branch_type=BranchType.SIDECHAIN,
        created_at="2026-03-01T10:00:00+00:00",
        updated_at="2026-03-01T10:30:00+00:00",
        working_directories=["/realm/project/polylogue"],
        git_branch="feature/route-matrix",
        git_repository_url="https://example.invalid/route-matrix",
        provider_project_ref="g-p-route-matrix",
        reported_cost_usd=1.25,
        attachments=[
            ParsedAttachment(
                provider_attachment_id="route-matrix-attachment",
                message_provider_id="child-m0",
                name="route-matrix.png",
                mime_type="image/png",
                size_bytes=1234,
                upload_origin="paste",
                direction="user_input",
                caption="a pasted screenshot",
                source_url="https://example.invalid/route-matrix.png",
            )
        ],
        messages=[
            ParsedMessage(
                provider_message_id="child-m0",
                role=Role.USER,
                text="please run the tool",
                timestamp="2026-03-01T10:00:00+00:00",
                position=0,
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="please run the tool")],
            ),
            ParsedMessage(
                provider_message_id="child-m1",
                role=Role.ASSISTANT,
                timestamp="2026-03-01T10:15:00+00:00",
                position=1,
                parent_message_provider_id="child-m0",
                is_active_path=True,
                is_active_leaf=False,
                duration_ms=4321,
                # The provider's terminal signal: the API message hydrator used
                # to drop this while the query hydrator carried it.
                stop_reason="tool_use",
                blocks=[
                    ParsedContentBlock(type=BlockType.THINKING, text="weighing the options"),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="shell",
                        tool_id="tool-route-matrix",
                        tool_input={"command": "ls -la"},
                    ),
                ],
            ),
            ParsedMessage(
                provider_message_id="child-m2",
                role=Role.USER,
                timestamp="2026-03-01T10:30:00+00:00",
                position=2,
                parent_message_provider_id="child-m1",
                is_active_path=True,
                is_active_leaf=True,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="tool-route-matrix",
                        text="total 0",
                        # The keystone structural outcome plus the deliberate
                        # unknown-outcome reason the API hydrator used to drop.
                        tool_outcome=ToolOutcome.UNKNOWN,
                        outcome_unknown_reason=ToolResultUnknownReason.NOT_REPORTED.value,
                    )
                ],
            ),
            # A second sibling at the same transcript position makes the
            # public row projection prove both creation order and active-path
            # selection.  The inactive sibling must remain visible so a
            # consumer can reconstruct the branch rather than receiving a
            # flattened mainline.
            ParsedMessage(
                provider_message_id="child-m1-variant",
                role=Role.ASSISTANT,
                timestamp="2026-03-01T10:16:00+00:00",
                position=1,
                variant_index=1,
                parent_message_provider_id="child-m0",
                is_active_path=False,
                is_active_leaf=False,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="the inactive sibling")],
            ),
        ],
    )


@dataclass(frozen=True)
class _Seeded:
    archive_root: Path
    session_id: str


def _seed(workspace_env: dict[str, Path]) -> _Seeded:
    db_path = db_setup(workspace_env)
    write_session_sync(db_path, _parent_session())
    session_id = write_session_sync(db_path, _child_session())
    return _Seeded(archive_root=workspace_env["archive_root"], session_id=session_id)


def _canonical_summary(seeded: _Seeded) -> tuple[ArchiveSessionSummary, SessionSummary]:
    with ArchiveStore(seeded.archive_root, initialize=False, read_only=True) as archive:
        row = archive.read_summary(seeded.session_id)
    return row, archive_summary_to_domain(row)


def _canonical_messages(seeded: _Seeded) -> tuple[tuple[ArchiveMessageRow, ...], list[Message]]:
    from polylogue.core.enums import Origin

    with ArchiveStore(seeded.archive_root, initialize=False, read_only=True) as archive:
        envelope = archive.read_session(seeded.session_id)
    rows = envelope.messages
    origin = Origin.from_string(envelope.origin)
    return rows, [archive_message_to_domain(row, origin=origin) for row in rows]


def _summary_facts(summary: object) -> dict[str, object]:
    """Read the declared summary-mask fields off any hydrated summary."""
    return {domain_name: getattr(summary, domain_name) for domain_name, _surface in _SESSION_SUMMARY_MASK}


def _topology_facts(summary: object) -> dict[str, object]:
    """The fields the three old summary mappers each dropped a different slice of."""
    return {
        "parent_id": str(getattr(summary, "parent_id", None) or "") or None,
        "branch_type": str(getattr(summary, "branch_type", None) or "") or None,
        "display_name": getattr(summary, "display_name", None),
        "title_source": str(getattr(summary, "title_source", None) or "") or None,
        "title_ref": getattr(summary, "title_ref", None),
        "session_kind": str(getattr(summary, "session_kind", None) or "") or None,
        "git_repository_url": getattr(summary, "git_repository_url", None),
        "provider_project_ref": getattr(summary, "provider_project_ref", None),
    }


def _message_facts(message: object) -> dict[str, object]:
    return {domain_name: getattr(message, domain_name) for domain_name, _surface in _MESSAGE_MASK}


def _attachment_facts(message: Message) -> list[dict[str, object]]:
    """The attachment fields the declaration exposes, plus its typed availability."""
    return [
        {name: getattr(attachment, name) for name in exposed_domain_names(hydration.ARCHIVE_ATTACHMENT_DISPOSITIONS)}
        for attachment in message.attachments
    ]


def _block_outcome_facts(message: Message) -> list[dict[str, object]]:
    return [
        {
            "id": block.get("id"),
            "type": block.get("type"),
            "tool_id": block.get("tool_id"),
            "tool_outcome": block.get("tool_outcome"),
            "tool_result_outcome_unknown_reason": block.get("tool_result_outcome_unknown_reason"),
            "tool_input": block.get("tool_input"),
        }
        for block in message.blocks
    ]


@pytest.mark.asyncio
async def test_seeded_session_carries_nondefault_values_in_every_mapped_family(
    workspace_env: dict[str, Path],
) -> None:
    """Guard the matrix itself: parity over default values would prove nothing."""
    seeded = _seed(workspace_env)
    row, summary = _canonical_summary(seeded)
    assert row.parent_id == _PARENT_SESSION_ID
    assert row.branch_type == "sidechain"
    assert row.display_name == "greedy-squishing-hamming"
    assert row.title_source == "heuristic"
    assert row.title_ref == "message:route-matrix-child:0"
    assert row.git_repository_url == "https://example.invalid/route-matrix"
    assert row.provider_project_ref == "g-p-route-matrix"
    assert summary.parent_id is not None
    assert summary.branch_type is not None

    _rows, messages = _canonical_messages(seeded)
    assistant = next(message for message in messages if message.stop_reason is not None)
    assert assistant.stop_reason == "tool_use"
    outcomes = [
        block
        for message in messages
        for block in _block_outcome_facts(message)
        if block["tool_result_outcome_unknown_reason"] is not None
    ]
    assert outcomes and outcomes[0]["tool_outcome"] == "unknown"

    attachments = [facts for message in messages for facts in _attachment_facts(message)]
    assert len(attachments) == 1
    assert attachments[0]["name"] == "route-matrix.png"
    assert attachments[0]["mime_type"] == "image/png"
    assert attachments[0]["caption"] == "a pasted screenshot"
    assert attachments[0]["direction"] == "user_input"
    # The typed physical-readability result owned by polylogue-hb9o6 reaches the
    # domain attachment; this fixture acquires no bytes, so it is an explicit
    # "not acquired" verdict rather than a missing field.
    assert attachments[0]["availability"] is not None


@pytest.mark.asyncio
async def test_summary_hydration_parity_across_api_query_cli_mcp_and_http(
    workspace_env: dict[str, Path],
) -> None:
    """Every summary route reports the canonical summary under its own mask."""
    from polylogue.api import Polylogue
    from polylogue.cli.read_views.standard import exact_read_summaries
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.config import Config
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.mcp.archive_support import archive_summary_payload

    seeded = _seed(workspace_env)
    row, canonical = _canonical_summary(seeded)
    expected_mask = _summary_facts(canonical)
    expected_topology = _topology_facts(canonical)

    # API: the public single-session summary read.
    api = Polylogue(archive_root=seeded.archive_root, db_path=seeded.archive_root / "index.db")
    api_summary = await api.get_session_summary(seeded.session_id)
    assert api_summary is not None
    assert _summary_facts(api_summary) == expected_mask
    assert _topology_facts(api_summary) == expected_topology

    # Archive-query execution: the route behind the CLI/MCP query grammar.
    # ``root`` unset resolves to top-level-only at the filter boundary
    # (polylogue-j8u2); the seeded child is parented, so the query route must be
    # asked for children explicitly or it legitimately returns nothing.
    plan = SessionQueryPlan(origins=(_ORIGIN,), limit=10, root=False)
    query_summaries = await SessionFilter(archive_root=seeded.archive_root, query_plan=plan).list_summaries()
    query_summary = next(summary for summary in query_summaries if str(summary.id) == seeded.session_id)
    assert _summary_facts(query_summary) == expected_mask
    assert _topology_facts(query_summary) == expected_topology

    # Exact CLI read: the ``--id`` path that never enumerates query rows.
    config = Config(
        archive_root=seeded.archive_root,
        db_path=seeded.archive_root / "index.db",
        render_root=seeded.archive_root / "render",
        sources=[],
    )
    cli_summaries = exact_read_summaries(config, RootModeRequest.from_params({"conv_id": _CHILD_NATIVE_ID}))
    assert cli_summaries is not None
    assert len(cli_summaries) == 1
    assert _summary_facts(cli_summaries[0]) == expected_mask
    assert _topology_facts(cli_summaries[0]) == expected_topology

    # MCP list/search rows: masked by the shared summary envelope builder, so
    # they carry the title provenance MCP get/resolve_ref already returned.
    mcp_payload = archive_summary_payload(row)
    assert canonical.title_source is not None
    assert mcp_payload.title_source == canonical.title_source.value
    assert mcp_payload.title_ref == canonical.title_ref
    assert mcp_payload.message_count == canonical.message_count

    # HTTP: the web reader's list shape and its summary-shaped detail route.
    handler = object.__new__(DaemonAPIHandler)
    http_list = handler._archive_summary_payload(row)
    assert http_list["id"] == str(canonical.id)
    assert http_list["message_count"] == canonical.message_count
    assert http_list["repo"] == canonical.git_repository_url
    http_detail = handler._do_archive_get_session_summary(seeded.archive_root, _CHILD_NATIVE_ID)
    assert isinstance(http_detail, dict)
    assert http_detail["parent_id"] == _PARENT_SESSION_ID
    assert http_detail["branch_type"] == "sidechain"
    assert http_detail["display_name"] == "greedy-squishing-hamming"
    assert http_detail["title_source"] == "heuristic"


@pytest.mark.asyncio
async def test_message_and_block_hydration_parity_across_api_query_mcp_and_http(
    workspace_env: dict[str, Path],
) -> None:
    """Every message route reports the canonical message under its own mask."""
    from polylogue.api import Polylogue
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.mcp.archive_support import archive_message_payload

    seeded = _seed(workspace_env)
    rows, canonical_messages = _canonical_messages(seeded)
    expected_mask = [_message_facts(message) for message in canonical_messages]
    expected_blocks = [_block_outcome_facts(message) for message in canonical_messages]
    expected_attachments = [_attachment_facts(message) for message in canonical_messages]

    # API: the public whole-session read.
    api = Polylogue(archive_root=seeded.archive_root, db_path=seeded.archive_root / "index.db")
    api_session = await api.get_session(seeded.session_id)
    assert api_session is not None
    api_messages = api_session.messages.to_list()
    assert [_message_facts(message) for message in api_messages] == expected_mask
    assert [_block_outcome_facts(message) for message in api_messages] == expected_blocks
    assert [_attachment_facts(message) for message in api_messages] == expected_attachments

    # Archive-query execution: the fluent filter's whole-session leg.
    # ``root`` unset resolves to top-level-only at the filter boundary
    # (polylogue-j8u2); the seeded child is parented, so the query route must be
    # asked for children explicitly or it legitimately returns nothing.
    plan = SessionQueryPlan(origins=(_ORIGIN,), limit=10, root=False)
    query_sessions = await SessionFilter(archive_root=seeded.archive_root, query_plan=plan).list()
    query_session = next(session for session in query_sessions if str(session.id) == seeded.session_id)
    query_messages = query_session.messages.to_list()
    assert [_message_facts(message) for message in query_messages] == expected_mask
    assert [_block_outcome_facts(message) for message in query_messages] == expected_blocks
    # The query route used to return no attachments at all: its message mapper
    # never hydrated them, so the same session had attachments through the API
    # and none through ``SessionFilter.list()``.
    assert [_attachment_facts(message) for message in query_messages] == expected_attachments

    # MCP rich content: structured block outcomes must survive the envelope.
    mcp_messages = [archive_message_payload(row, session_id=seeded.session_id) for row in rows]
    assert [payload.stop_reason for payload in mcp_messages] == [message.stop_reason for message in canonical_messages]
    assert [payload.parent_message_id for payload in mcp_messages] == [
        message.parent_id for message in canonical_messages
    ]
    assert [payload.variant_index for payload in mcp_messages] == [
        message.branch_index for message in canonical_messages
    ]
    assert [payload.is_active_path for payload in mcp_messages] == [
        message.is_active_path for message in canonical_messages
    ]
    assert [payload.is_active_leaf for payload in mcp_messages] == [
        message.is_active_leaf for message in canonical_messages
    ]
    mcp_blocks = [
        {
            "tool_outcome": block.get("tool_outcome"),
            "tool_result_outcome_unknown_reason": block.get("tool_result_outcome_unknown_reason"),
        }
        for payload in mcp_messages
        for block in payload.content_blocks
        if block.get("tool_id")
    ]
    assert {
        "tool_outcome": ToolOutcome.UNKNOWN.value,
        "tool_result_outcome_unknown_reason": ToolResultUnknownReason.NOT_REPORTED.value,
    } in mcp_blocks

    # HTTP detail: exposes the terminal signal it claims to render.
    handler = object.__new__(DaemonAPIHandler)
    http_messages = [handler._archive_message_payload(seeded.session_id, row) for row in rows]
    assert [payload["stop_reason"] for payload in http_messages] == [
        message.stop_reason for message in canonical_messages
    ]
    assert [payload["text"] for payload in http_messages] == [message.text or "" for message in canonical_messages]
    assert [payload["is_active_leaf"] for payload in http_messages] == [
        message.is_active_leaf for message in canonical_messages
    ]
    assert [payload["parent_message_id"] for payload in http_messages] == [
        message.parent_id for message in canonical_messages
    ]
    assert [payload["variant_index"] for payload in http_messages] == [
        message.branch_index for message in canonical_messages
    ]
    assert [payload["is_active_path"] for payload in http_messages] == [
        message.is_active_path for message in canonical_messages
    ]
    http_attachments = [
        att for payload in http_messages for att in cast("list[dict[str, object]]", payload["attachments"])
    ]
    assert len(http_attachments) == 1
    assert http_attachments[0]["name"] == "route-matrix.png"


@pytest.mark.asyncio
async def test_mcp_page_and_composed_message_routes_agree_on_text_and_outcomes(
    workspace_env: dict[str, Path],
) -> None:
    """MCP's two message routes read different SQL; they must not read differently.

    ``get_messages`` serves a bounded row projection and the composed read
    serves ``ArchiveStore.read_session``. Both now hydrate through the same
    owner, so flattened text and structured block outcomes match, and the
    bounded route names the richer operation for the fields it does not select.
    """
    from polylogue.mcp.archive_support import archive_message_page_payload, archive_message_payload

    seeded = _seed(workspace_env)
    rows, _canonical = _canonical_messages(seeded)
    composed = [archive_message_payload(row, session_id=seeded.session_id) for row in rows]

    with ArchiveStore(seeded.archive_root, initialize=False, read_only=True) as archive:
        # The seeded child names a parent but replays none of its prefix, so the
        # bounded row projection -- not the composed fallback -- serves this
        # page. Assert it rather than letting a fixture change silently reroute
        # the test to the route it is not trying to cover.
        assert archive.has_prefix_lineage(seeded.session_id) is False
        page = archive_message_page_payload(archive, seeded.session_id, limit=50, offset=0)
        query_rows = archive.query_session_messages((seeded.session_id,), limit=50, offset=0)
        from polylogue.archive.query.predicate import QueryFieldPredicate, QueryFieldRef

        predicate_rows = archive.query_messages(
            QueryFieldPredicate(field="session.id", values=(seeded.session_id,), op="=").with_field_ref(
                QueryFieldRef(scope="session", name="id", source_name="session.id")
            ),
            limit=50,
        )

    def _outcomes(messages: Sequence[MessageRenderEnvelope]) -> list[list[object]]:
        return [
            [block.get("tool_outcome"), block.get("tool_result_outcome_unknown_reason")]
            for message in messages
            for block in message.content_blocks
        ]

    assert [message.text for message in page.messages] == [message.text for message in composed]
    assert _outcomes(page.messages) == _outcomes(composed)
    assert [ToolOutcome.UNKNOWN.value, ToolResultUnknownReason.NOT_REPORTED.value] in _outcomes(page.messages)
    # The bounded projection carries topology because order/branch
    # reconstruction is part of the public message contract. It still names
    # the richer operation for the fields it intentionally does not select.
    assert [message.parent_message_id for message in page.messages] == [
        message.parent_message_id for message in composed
    ]
    assert [message.variant_index for message in page.messages] == [message.variant_index for message in composed]
    assert [message.is_active_path for message in page.messages] == [message.is_active_path for message in composed]
    assert [message.is_active_leaf for message in page.messages] == [message.is_active_leaf for message in composed]
    # The terminal query row payload is the CLI/API/HTTP query-unit adapter;
    # both bounded SQL entry points must carry the same topology fields.
    from polylogue.surfaces.payloads import MessageQueryRowPayload

    for query_rows_variant in (query_rows, predicate_rows):
        query_payloads = [MessageQueryRowPayload.from_row(row) for row in query_rows_variant]
        assert [payload.parent_message_id for payload in query_payloads] == [
            message.parent_message_id for message in page.messages
        ]
        assert [payload.variant_index for payload in query_payloads] == [
            message.variant_index for message in page.messages
        ]
        assert [payload.is_active_path for payload in query_payloads] == [
            message.is_active_path for message in page.messages
        ]
        assert [payload.is_active_leaf for payload in query_payloads] == [
            message.is_active_leaf for message in page.messages
        ]
    # A consumer using only the bounded public rows can rebuild the branch:
    # transcript coordinates determine sibling order, parent ids determine
    # edges, and active-path flags identify the accepted sibling.
    topology = sorted(
        (
            message.position,
            message.variant_index,
            message.parent_message_id,
            message.is_active_path,
            message.is_active_leaf,
        )
        for message in page.messages
    )
    assert topology == sorted(
        (
            message.position,
            message.variant_index,
            message.parent_message_id,
            message.is_active_path,
            message.is_active_leaf,
        )
        for message in composed
    )
    assert [message.variant_index for message in page.messages if message.position == 1] == [0, 1]
    assert [message.is_active_path for message in page.messages if message.position == 1] == [True, False]
    assert page.projection_note == hydration.MESSAGE_QUERY_ROW_RICHER_OPERATION
    assert "stop_reason" in hydration.MESSAGE_QUERY_ROW_UNPROJECTED
    assert "parent_id" not in hydration.MESSAGE_QUERY_ROW_UNPROJECTED
    assert "branch_index" not in hydration.MESSAGE_QUERY_ROW_UNPROJECTED
    assert "is_active_path" not in hydration.MESSAGE_QUERY_ROW_UNPROJECTED
    assert "is_active_leaf" not in hydration.MESSAGE_QUERY_ROW_UNPROJECTED


@pytest.mark.asyncio
async def test_dropping_topology_from_bounded_row_adapter_breaks_public_parity(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded SQL adapter is load-bearing, not a second default mapper."""
    from polylogue.mcp.archive_support import archive_message_page_payload

    seeded = _seed(workspace_env)
    with ArchiveStore(seeded.archive_root, initialize=False, read_only=True) as archive:
        expected = archive_message_page_payload(archive, seeded.session_id, limit=50, offset=0)

    monkeypatch.setattr(
        hydration,
        "ARCHIVE_MESSAGE_QUERY_ROW_DISPOSITIONS",
        _without(hydration.ARCHIVE_MESSAGE_QUERY_ROW_DISPOSITIONS, "parent_message_id"),
    )
    with ArchiveStore(seeded.archive_root, initialize=False, read_only=True) as archive:
        mutated = archive_message_page_payload(archive, seeded.session_id, limit=50, offset=0)

    assert [message.parent_message_id for message in mutated.messages] != [
        message.parent_message_id for message in expected.messages
    ]


# ---------------------------------------------------------------------------
# Controlled mutations: each must break the parity it is supposed to protect
# ---------------------------------------------------------------------------


def _without(dispositions: Any, field_name: str) -> dict[str, Any]:
    mutated = dict(dispositions)
    mutated[field_name] = hydration.delegated("mutant: dropped by the test")
    return mutated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dropped",
    ["stop_reason", "is_active_leaf", "parent_message_id"],
)
async def test_dropping_one_message_disposition_breaks_the_production_route(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    dropped: str,
) -> None:
    """A message field lost at the seam must change what a real route returns."""
    from polylogue.api import Polylogue

    seeded = _seed(workspace_env)
    api = Polylogue(archive_root=seeded.archive_root, db_path=seeded.archive_root / "index.db")
    before = await api.get_session(seeded.session_id)
    assert before is not None
    expected = [_message_facts(message) for message in before.messages.to_list()]

    monkeypatch.setattr(
        hydration,
        "ARCHIVE_MESSAGE_DISPOSITIONS",
        _without(hydration.ARCHIVE_MESSAGE_DISPOSITIONS, dropped),
    )
    after = await Polylogue(archive_root=seeded.archive_root, db_path=seeded.archive_root / "index.db").get_session(
        seeded.session_id
    )
    assert after is not None
    assert [_message_facts(message) for message in after.messages.to_list()] != expected


@pytest.mark.asyncio
async def test_dropping_the_block_outcome_disposition_breaks_the_http_detail_route(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The keystone tool outcome must not be droppable without a red route."""
    from polylogue.daemon.http import DaemonAPIHandler

    seeded = _seed(workspace_env)
    rows, canonical_messages = _canonical_messages(seeded)
    expected = [_block_outcome_facts(message) for message in canonical_messages]
    handler = object.__new__(DaemonAPIHandler)
    before_text = [handler._archive_message_payload(seeded.session_id, row)["text"] for row in rows]

    monkeypatch.setattr(
        hydration,
        "ARCHIVE_BLOCK_DISPOSITIONS",
        _without(hydration.ARCHIVE_BLOCK_DISPOSITIONS, "tool_result_outcome_unknown_reason"),
    )
    _rows, mutated_messages = _canonical_messages(seeded)
    assert [_block_outcome_facts(message) for message in mutated_messages] != expected
    # The mutation is confined to the structured outcome: the HTTP detail
    # route's flattened text is unchanged, so this test is not merely
    # observing "some payload differs".
    assert [handler._archive_message_payload(seeded.session_id, row)["text"] for row in rows] == before_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dropped",
    ["parent_id", "branch_type", "display_name", "title_source"],
)
async def test_dropping_one_summary_disposition_breaks_the_production_route(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    dropped: str,
) -> None:
    """Parent identity, branch type, display name and title provenance are load-bearing."""
    from polylogue.api import Polylogue

    seeded = _seed(workspace_env)
    api = Polylogue(archive_root=seeded.archive_root, db_path=seeded.archive_root / "index.db")
    before = await api.get_session_summary(seeded.session_id)
    assert before is not None
    expected = _topology_facts(before)

    monkeypatch.setattr(
        hydration,
        "ARCHIVE_SUMMARY_DISPOSITIONS",
        _without(hydration.ARCHIVE_SUMMARY_DISPOSITIONS, dropped),
    )
    after = await Polylogue(
        archive_root=seeded.archive_root, db_path=seeded.archive_root / "index.db"
    ).get_session_summary(seeded.session_id)
    assert after is not None
    assert _topology_facts(after) != expected

    # The exact CLI read is a separate production route over the same seam.
    from polylogue.cli.read_views.standard import exact_read_summaries
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.config import Config

    config = Config(
        archive_root=seeded.archive_root,
        db_path=seeded.archive_root / "index.db",
        render_root=seeded.archive_root / "render",
        sources=[],
    )
    cli_summaries = exact_read_summaries(config, RootModeRequest.from_params({"conv_id": _CHILD_NATIVE_ID}))
    assert cli_summaries is not None
    assert _topology_facts(cli_summaries[0]) != expected
