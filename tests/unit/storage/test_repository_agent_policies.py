"""Async repository surface for ``session_agent_policies`` (Codex agent-policy facts).

The writer diverts Codex ``agent_policy`` events out of ``session_events``
into the dedicated ``session_agent_policies`` table because the evidence is
fully re-derivable there with no loss (see
``archive_tiers/write.py:_SESSION_EVENTS_REDUNDANT_TYPES``). Before
``SessionRepository.get_agent_policies``/``get_agent_policies_batch``
existed, nothing outside a sync test helper (``read_session_agent_policies``)
could read that table back -- real sandbox/approval/network-policy facts
landed durably with zero readers. These tests prove the async path actually
reaches the data the writer produced.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.live_ingest import ingest_session


def _session_with_policy(native_id: str, *, approval: str, sandbox: str, network: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title="Policy session",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="run it",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="run it")],
            ),
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="agent_policy",
                timestamp="2026-01-01T00:00:01+00:00",
                payload={
                    "approval_policy": approval,
                    "sandbox_policy": sandbox,
                    "network_policy": network,
                },
            ),
        ],
    )


async def test_get_agent_policies_reaches_writer_evidence(tmp_path: Path) -> None:
    """The production writer's ``session_agent_policies`` row must be reachable
    through the async repository, not just the sync test-only reader.
    """
    backend = SQLiteBackend(db_path=tmp_path / "agent-policies.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            _session_with_policy("policy-1", approval="never", sandbox="danger-full-access", network="true"),
            backend=backend,
        )

        policies = await repo.get_agent_policies(session_id)
    finally:
        await repo.close()

    assert len(policies) == 1
    policy = policies[0]
    assert policy.session_id == session_id
    assert policy.approval_policy == "never"
    assert policy.sandbox_policy == "danger-full-access"
    assert policy.network_policy == "true"


async def test_get_agent_policies_empty_for_session_without_policy_events(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "agent-policies-empty.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="no-policy",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )

        policies = await repo.get_agent_policies(session_id)
    finally:
        await repo.close()

    assert policies == []


async def test_get_agent_policies_batch_groups_by_session(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "agent-policies-batch.db")
    repo = SessionRepository(backend=backend)
    try:
        session_a = await ingest_session(
            _session_with_policy("policy-a", approval="untrusted", sandbox="read-only", network="false"),
            backend=backend,
        )
        session_b = await ingest_session(
            _session_with_policy("policy-b", approval="on-failure", sandbox="workspace-write", network="true"),
            backend=backend,
        )

        batch = await repo.get_agent_policies_batch([session_a, session_b])
    finally:
        await repo.close()

    assert {policy.approval_policy for policy in batch[session_a]} == {"untrusted"}
    assert {policy.approval_policy for policy in batch[session_b]} == {"on-failure"}
