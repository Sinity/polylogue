"""Adversarial corpus for the query-contract differential.

Every session reaches SQLite through the live index-only parse/write seam
(``write_index_session``), so the derived read models the query
units expose -- the ``actions`` view, ``delegation_facts``, run and
observed-event projections, FTS -- are materialized the way ingest
materializes them, not seeded behind the writer.

The corpus shape is generated from :data:`~tests.infra.query_contract.CORPUS_SHAPE_ANCHORS`,
which quotes the same distribution keys the production archive-composition
profile emits, and it carries every member of
:data:`~tests.infra.query_contract.REQUIRED_PATHOLOGIES`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session
from tests.infra.query_contract import (
    CORPUS_SHAPE_ANCHORS_BY_DIMENSION,
    REQUIRED_PATHOLOGIES,
    PathologyName,
)

#: 2026-07-15, the day of the mandate incident this corpus keeps as a
#: permanent fixture: a coordinator that dispatched through both a
#: subagent-classified tool and ``tool:Workflow``, whose delegations and
#: workflow actions must stay distinguishable on every surface.
MANDATE_INCIDENT_MS = 1_784_073_600_000

CORPUS_ORIGIN = "claude-code-session"
CORPUS_REPO = "widget"
CORPUS_REPOSITORY_URL = "https://example.invalid/acme/widget"

#: Present in every session: the low-selectivity probe term.
COMMON_TERM = "mandate"
#: Present in exactly one block of one session: the high-selectivity probe.
RARE_TERM = "hapaxlegomenon"

COORDINATOR_ID = "coordinator-2026-07-15"
WORKFLOW_TOOL = "Workflow"
SUBAGENT_TOOL = "Task"


@dataclass(frozen=True, slots=True)
class QueryCorpus:
    """A materialized corpus plus the facts its laws quote."""

    archive_root: Path
    session_ids: tuple[str, ...]
    pathologies: Mapping[PathologyName, str]
    tagged_session_id: str

    def pathology_session(self, pathology: PathologyName) -> str:
        return self.pathologies[pathology]


def _text(value: str) -> ParsedContentBlock:
    return ParsedContentBlock(type=BlockType.TEXT, text=value)


def _tool_use(*, tool: str, tool_id: str, tool_input: dict[str, object], text: str | None = None) -> ParsedContentBlock:
    return ParsedContentBlock(
        type=BlockType.TOOL_USE,
        tool_name=tool,
        tool_id=tool_id,
        tool_input=tool_input,
        text=text or tool,
    )


def _tool_result(
    *,
    tool: str,
    tool_id: str,
    text: str,
    is_error: bool | None = None,
    exit_code: int | None = None,
) -> ParsedContentBlock:
    # This corpus creates a wire-shape result, not a parser assertion that the
    # prose means success.  Its ordinary synthetic records carry no structural
    # outcome; retain that fact explicitly.  The one error fixture below has
    # its own real marker and therefore must not receive an unknown reason.
    outcome_unknown_reason = (
        ToolResultUnknownReason.NOT_REPORTED.value if is_error is None and exit_code is None else None
    )
    return ParsedContentBlock(
        type=BlockType.TOOL_RESULT,
        tool_name=tool,
        tool_id=tool_id,
        text=text,
        is_error=is_error,
        exit_code=exit_code,
        outcome_unknown_reason=outcome_unknown_reason,
    )


def _message(
    *,
    message_id: str,
    role: Role,
    text: str,
    at_ms: int,
    blocks: list[ParsedContentBlock] | None = None,
    model: str | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=message_id,
        role=role,
        text=text,
        occurred_at_ms=at_ms,
        model_name=model,
        blocks=blocks if blocks is not None else [_text(text)],
    )


def _session(
    *,
    native_id: str,
    title: str,
    messages: list[ParsedMessage],
    parent: str | None = None,
    branch_type: BranchType | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=native_id,
        title=title,
        git_repository_url=CORPUS_REPOSITORY_URL,
        git_branch="main",
        working_directories=["/work/widget"],
        parent_session_provider_id=parent,
        branch_type=branch_type,
        messages=messages,
    )


def _large_payload_text() -> str:
    """A block at the declared payload tail, not an arbitrary blob."""

    anchor = CORPUS_SHAPE_ANCHORS_BY_DIMENSION["largest-block-bytes"]
    unit = f"{COMMON_TERM} payload line\n"
    return unit * (int(anchor.maximum) // len(unit))


def _mandate_incident_sessions() -> tuple[list[ParsedSession], dict[PathologyName, str]]:
    """The coordinator cohort: delegations, `tool:Workflow`, and result pathologies."""

    t0 = MANDATE_INCIDENT_MS
    subagent_prompt = f"review the {COMMON_TERM} diff"
    coordinator = _session(
        native_id=COORDINATOR_ID,
        title=f"Coordinator: the {COMMON_TERM} incident",
        messages=[
            _message(
                message_id="co-1",
                role=Role.USER,
                text=f"dispatch the {COMMON_TERM} review",
                at_ms=t0,
            ),
            _message(
                message_id="co-2",
                role=Role.ASSISTANT,
                text="dispatching",
                at_ms=t0 + 1_000,
                model="claude-opus-5",
                blocks=[
                    ParsedContentBlock(type=BlockType.THINKING, text=f"plan the {COMMON_TERM} dispatch"),
                    _tool_use(
                        tool=SUBAGENT_TOOL,
                        tool_id="dispatch-1",
                        tool_input={
                            "prompt": subagent_prompt,
                            "model": "claude-sonnet-5",
                            "subagent_type": "review",
                        },
                    ),
                    _tool_use(
                        tool=WORKFLOW_TOOL,
                        tool_id="workflow-1",
                        tool_input={"prompt": f"compile the {COMMON_TERM} obligations"},
                    ),
                ],
            ),
            # Duplicate tool result: the same tool id answered twice in one turn.
            _message(
                message_id="co-3",
                role=Role.USER,
                text="",
                at_ms=t0 + 2_000,
                blocks=[
                    _tool_result(tool=WORKFLOW_TOOL, tool_id="workflow-1", text="workflow finished"),
                    _tool_result(tool=WORKFLOW_TOOL, tool_id="workflow-1", text="workflow finished (replayed)"),
                ],
            ),
            _message(
                message_id="co-4",
                role=Role.ASSISTANT,
                text="editing",
                at_ms=t0 + 3_000,
                model="claude-opus-5",
                blocks=[
                    _tool_use(
                        tool="Edit",
                        tool_id="edit-1",
                        tool_input={"file_path": "polylogue/archive/query/plan.py"},
                    ),
                    # Missing tool result: no answer for this dispatch anywhere.
                    _tool_use(tool="Bash", tool_id="orphan-1", tool_input={"command": "sleep 1"}),
                ],
            ),
            _message(
                message_id="co-5",
                role=Role.USER,
                text="",
                at_ms=t0 + 4_000,
                blocks=[_tool_result(tool="Edit", tool_id="edit-1", text="applied")],
            ),
            _message(
                message_id="co-6",
                role=Role.ASSISTANT,
                text="verifying",
                at_ms=t0 + 5_000,
                model="claude-opus-5",
                blocks=[_tool_use(tool="Bash", tool_id="verify-1", tool_input={"command": "pytest -x"})],
            ),
            # Late tool result: answered many turns and hours after its dispatch.
            _message(
                message_id="co-7",
                role=Role.USER,
                text="",
                at_ms=t0 + 6_000,
                blocks=[_tool_result(tool=SUBAGENT_TOOL, tool_id="dispatch-1", text="subagent reported")],
            ),
            _message(
                message_id="co-8",
                role=Role.USER,
                text="",
                at_ms=t0 + 7_200_000,
                blocks=[
                    _tool_result(tool="Bash", tool_id="verify-1", text="1 failed", is_error=True, exit_code=1),
                ],
            ),
        ],
    )

    worker = _session(
        native_id="worker-review",
        title=f"Worker: {subagent_prompt}",
        parent=COORDINATOR_ID,
        branch_type=BranchType.SUBAGENT,
        messages=[
            _message(message_id="wr-1", role=Role.USER, text=subagent_prompt, at_ms=t0 + 1_500),
            _message(
                message_id="wr-2",
                role=Role.ASSISTANT,
                text=f"the {COMMON_TERM} diff is {RARE_TERM}",
                at_ms=t0 + 5_500,
                model="claude-sonnet-5",
            ),
        ],
    )

    pathologies: dict[PathologyName, str] = {
        "mandate-incident": COORDINATOR_ID,
        "duplicate-tool-result": COORDINATOR_ID,
        "missing-tool-result": COORDINATOR_ID,
        "late-tool-result": COORDINATOR_ID,
        "high-selectivity": "worker-review",
    }
    return [coordinator, worker], pathologies


def _lineage_sessions() -> tuple[list[ParsedSession], dict[PathologyName, str]]:
    """Wide (many children of one parent) and deep (a chain) lineage."""

    t0 = MANDATE_INCIDENT_MS + 100_000
    children_anchor = CORPUS_SHAPE_ANCHORS_BY_DIMENSION["children-per-parent"]
    fan_out = int(children_anchor.maximum)

    root = _session(
        native_id="wide-root",
        title=f"Wide root for the {COMMON_TERM} fan-out",
        messages=[
            _message(message_id="wd-1", role=Role.USER, text=f"fan out the {COMMON_TERM} work", at_ms=t0),
            _message(
                message_id="wd-2",
                role=Role.ASSISTANT,
                text="fanning out",
                at_ms=t0 + 500,
                model="claude-opus-5",
                blocks=[
                    _tool_use(
                        tool=SUBAGENT_TOOL,
                        tool_id=f"fan-{index}",
                        tool_input={"prompt": f"branch {index} of the {COMMON_TERM} work"},
                    )
                    for index in range(fan_out)
                ],
            ),
        ],
    )
    wide_children = [
        _session(
            native_id=f"wide-child-{index}",
            title=f"Wide child {index} of the {COMMON_TERM} fan-out",
            parent="wide-root",
            branch_type=BranchType.SUBAGENT,
            messages=[
                _message(
                    message_id=f"wc{index}-1",
                    role=Role.USER,
                    text=f"branch {index} of the {COMMON_TERM} work",
                    at_ms=t0 + 1_000 + index,
                ),
                _message(
                    message_id=f"wc{index}-2",
                    role=Role.ASSISTANT,
                    text=f"branch {index} of the {COMMON_TERM} work is done",
                    at_ms=t0 + 2_000 + index,
                    model="claude-sonnet-5",
                ),
            ],
        )
        for index in range(fan_out)
    ]

    deep: list[ParsedSession] = []
    parent_id: str | None = None
    for depth in range(3):
        native_id = f"deep-{depth}"
        deep.append(
            _session(
                native_id=native_id,
                title=f"Deep {depth} of the {COMMON_TERM} chain",
                parent=parent_id,
                branch_type=BranchType.CONTINUATION if parent_id is not None else None,
                messages=[
                    _message(
                        message_id=f"dp{depth}-1",
                        role=Role.USER,
                        text=f"continue the {COMMON_TERM} chain at depth {depth}",
                        at_ms=t0 + 10_000 + depth,
                    ),
                    _message(
                        message_id=f"dp{depth}-2",
                        role=Role.ASSISTANT,
                        text=f"depth {depth} of the {COMMON_TERM} chain",
                        at_ms=t0 + 11_000 + depth,
                        model="claude-opus-5",
                    ),
                ],
            )
        )
        parent_id = native_id

    pathologies: dict[PathologyName, str] = {
        "wide-lineage": "wide-root",
        "deep-lineage": "deep-2",
    }
    return [root, *wide_children, *deep], pathologies


def _large_payload_session() -> ParsedSession:
    t0 = MANDATE_INCIDENT_MS + 200_000
    return _session(
        native_id="large-payload",
        title=f"Large {COMMON_TERM} payload",
        messages=[
            _message(message_id="lp-1", role=Role.USER, text=f"attach the {COMMON_TERM} log", at_ms=t0),
            _message(
                message_id="lp-2",
                role=Role.ASSISTANT,
                text="attached",
                at_ms=t0 + 1_000,
                model="claude-opus-5",
                blocks=[_tool_use(tool="Read", tool_id="read-1", tool_input={"file_path": "/work/widget/big.log"})],
            ),
            _message(
                message_id="lp-3",
                role=Role.USER,
                text="",
                at_ms=t0 + 2_000,
                blocks=[_tool_result(tool="Read", tool_id="read-1", text=_large_payload_text())],
            ),
        ],
    )


def _growth_session(*, grown: bool) -> ParsedSession:
    """The active-growth member: the same session, appended after first ingest."""

    t0 = MANDATE_INCIDENT_MS + 300_000
    messages = [
        _message(message_id="gr-1", role=Role.USER, text=f"start the {COMMON_TERM} run", at_ms=t0),
        _message(
            message_id="gr-2",
            role=Role.ASSISTANT,
            text=f"the {COMMON_TERM} run started",
            at_ms=t0 + 1_000,
            model="claude-opus-5",
        ),
    ]
    if grown:
        messages.extend(
            [
                _message(message_id="gr-3", role=Role.USER, text=f"continue the {COMMON_TERM} run", at_ms=t0 + 2_000),
                _message(
                    message_id="gr-4",
                    role=Role.ASSISTANT,
                    text=f"the {COMMON_TERM} run continued",
                    at_ms=t0 + 3_000,
                    model="claude-opus-5",
                ),
            ]
        )
    return _session(native_id="active-growth", title=f"Active {COMMON_TERM} growth", messages=messages)


def build_query_corpus(archive_root: Path) -> QueryCorpus:
    """Materialize the corpus into ``archive_root`` through the write route."""

    incident_sessions, incident_pathologies = _mandate_incident_sessions()
    lineage_sessions, lineage_pathologies = _lineage_sessions()

    pathologies: dict[PathologyName, str] = {
        **incident_pathologies,
        **lineage_pathologies,
        "large-payload": "large-payload",
        "active-growth": "active-growth",
        "low-selectivity": COORDINATOR_ID,
    }
    missing = tuple(name for name in REQUIRED_PATHOLOGIES if name not in pathologies)
    if missing:
        raise AssertionError(f"query corpus does not carry required pathologies: {missing}")

    ordered: list[ParsedSession] = [
        *incident_sessions,
        *lineage_sessions,
        _large_payload_session(),
        _growth_session(grown=False),
    ]
    with ArchiveStore(archive_root) as archive:
        for session in ordered:
            write_index_session(archive, session)
        # Active growth: the same logical source re-ingested with a longer tail.
        write_index_session(archive, _growth_session(grown=True))

    session_ids = tuple(f"{CORPUS_ORIGIN}:{session.provider_session_id}" for session in ordered)
    tagged = f"{CORPUS_ORIGIN}:{COORDINATOR_ID}"
    return QueryCorpus(
        archive_root=archive_root,
        session_ids=session_ids,
        pathologies=pathologies,
        tagged_session_id=tagged,
    )


async def seed_user_tier(archive_root: Path, corpus: QueryCorpus) -> None:
    """Write the durable user-tier rows the ``assertion`` unit selects."""

    from polylogue.api import Polylogue

    archive = Polylogue(archive_root=archive_root)
    try:
        await archive.add_tag(corpus.tagged_session_id, COMMON_TERM)
        await archive.add_mark(corpus.tagged_session_id, "star")
    finally:
        await archive.close()


def build_query_corpus_sync(archive_root: Path) -> QueryCorpus:
    """Build the corpus and its user-tier overlay in one synchronous call."""

    corpus = build_query_corpus(archive_root)
    asyncio.run(seed_user_tier(archive_root, corpus))
    return corpus


__all__ = [
    "COMMON_TERM",
    "COORDINATOR_ID",
    "CORPUS_ORIGIN",
    "CORPUS_REPO",
    "MANDATE_INCIDENT_MS",
    "RARE_TERM",
    "SUBAGENT_TOOL",
    "WORKFLOW_TOOL",
    "QueryCorpus",
    "build_query_corpus",
    "build_query_corpus_sync",
    "seed_user_tier",
]
