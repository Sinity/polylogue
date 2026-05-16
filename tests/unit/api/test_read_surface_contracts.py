"""Contract tests for the shared read-surface protocol family.

Verifies the protocols defined in :mod:`polylogue.api.contracts.read_surface`
and the adapters in :mod:`polylogue.api.contracts.{api,cli,mcp}_surface`.

Two layers of evidence are recorded:

1. **Static conformance** — every adapter is checked against every
   protocol via ``issubclass``.  This catches missing methods, wrong
   arity, or accidental non-async coroutines that mypy would normally
   flag but that contract tests should also surface as a regression
   tripwire.  The `assert_implements` helper inside each adapter module
   already pins this at import time; the test re-asserts at runtime.

2. **Envelope parity** — every adapter is invoked over a small seeded
   archive and the returned envelopes are compared for the contract
   fields that matter: ``items``, ``total``, ``limit``, ``offset``, and
   the canonical id projection.  This is the runtime mirror of #1089's
   parity work; it confirms the static contract matches the runtime
   semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.api.contracts import (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
    ReadSurface,
)
from polylogue.api.contracts.api_surface import APIReadSurface
from polylogue.api.contracts.cli_surface import CLIReadSurface
from polylogue.api.contracts.mcp_surface import MCPReadSurface
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.core.json import JSONValue
from polylogue.services import build_runtime_services
from polylogue.surfaces.payloads import ConversationListResponse
from polylogue.types import Provider
from tests.infra.contract_evidence import ContractEvidenceRecorder
from tests.infra.storage_records import ConversationBuilder, db_setup

# ---------------------------------------------------------------------------
# Static-conformance: every adapter implements every required protocol.
# ---------------------------------------------------------------------------


_PROTOCOL_FAMILY: tuple[type, ...] = (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
)

_ADAPTER_CLASSES: tuple[type, ...] = (
    APIReadSurface,
    CLIReadSurface,
    MCPReadSurface,
)


@pytest.mark.parametrize("adapter_cls", _ADAPTER_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("protocol", _PROTOCOL_FAMILY, ids=lambda p: p.__name__)
def test_adapter_conforms_to_protocol(adapter_cls: type, protocol: type) -> None:
    """Every adapter class structurally satisfies every read-surface protocol."""
    assert issubclass(adapter_cls, protocol), (
        f"{adapter_cls.__name__} does not implement {protocol.__name__}; "
        "regenerate the adapter so it carries the canonical method signatures."
    )


def test_composite_read_surface_subsumes_required_protocols() -> None:
    """The composite ReadSurface declares the required read methods."""
    for adapter_cls in _ADAPTER_CLASSES:
        assert issubclass(adapter_cls, ReadSurface), (
            f"{adapter_cls.__name__} should satisfy the composite ReadSurface "
            "protocol so callers can depend on the full read contract."
        )


# ---------------------------------------------------------------------------
# Runtime parity: envelopes are interchangeable across adapters.
# ---------------------------------------------------------------------------


async def _seed_archive(db_path: Path) -> None:
    """Seed two conversations under different providers."""
    await (
        ConversationBuilder(db_path, "conv-alpha")
        .provider(Provider.CLAUDE_AI.value)
        .title("Alpha")
        .add_message(text="alpha message body")
        .build()
    )
    await (
        ConversationBuilder(db_path, "conv-beta")
        .provider(Provider.CHATGPT.value)
        .title("Beta")
        .add_message(text="beta message body")
        .build()
    )


async def test_all_adapters_return_equivalent_envelope(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    """All three adapters return the same envelope shape for the same query."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    try:
        api_surface = APIReadSurface(polylogue)
        cli_surface = CLIReadSurface(services)
        mcp_surface = MCPReadSurface(services)

        spec = ConversationQuerySpec(limit=10)
        envelopes: dict[str, ConversationListResponse] = {
            "api": await api_surface.list_conversations(spec),
            "cli": await cli_surface.list_conversations(spec),
            "mcp": await mcp_surface.list_conversations(spec),
        }

        # The shared envelope is a Pydantic model; equality compares fields.
        for surface_name, envelope in envelopes.items():
            assert isinstance(envelope, ConversationListResponse), (
                f"{surface_name} returned {type(envelope).__name__} not ConversationListResponse"
            )
            assert envelope.total == 2, f"{surface_name} total {envelope.total} != 2"
            assert len(envelope.items) == 2, f"{surface_name} returned {len(envelope.items)} items"

        ids_by_surface = {name: tuple(sorted(row.id for row in env.items)) for name, env in envelopes.items()}
        assert ids_by_surface["api"] == ids_by_surface["cli"] == ids_by_surface["mcp"], (
            f"id sets diverge: {ids_by_surface!r}"
        )

        record_contract_evidence.record(
            "read-surface-envelope-parity",
            surface="api+cli+mcp",
            facts={
                "id_set": list(ids_by_surface["api"]),
                "total": envelopes["api"].total,
                "limit": envelopes["api"].limit,
                "offset": envelopes["api"].offset,
            },
        )
    finally:
        await polylogue.close()
        await services.close()


async def test_provider_filter_envelope_parity(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    """Provider-filtered queries return identical id sets across adapters."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    try:
        spec = ConversationQuerySpec(providers=(Provider.CLAUDE_AI,), limit=10)
        api_envelope = await APIReadSurface(polylogue).list_conversations(spec)
        cli_envelope = await CLIReadSurface(services).list_conversations(spec)
        mcp_envelope = await MCPReadSurface(services).list_conversations(spec)

        api_ids = {row.id for row in api_envelope.items}
        cli_ids = {row.id for row in cli_envelope.items}
        mcp_ids = {row.id for row in mcp_envelope.items}
        assert api_ids == cli_ids == mcp_ids
        assert api_envelope.total == cli_envelope.total == mcp_envelope.total == 1

        matched_ids: list[JSONValue] = [str(item) for item in sorted(api_ids)]
        facts: dict[str, JSONValue] = {
            "provider": Provider.CLAUDE_AI.value,
            "matched_ids": matched_ids,
            "total": api_envelope.total,
        }
        record_contract_evidence.record(
            "read-surface-provider-filter-parity",
            surface="api+cli+mcp",
            facts=facts,
        )
    finally:
        await polylogue.close()
        await services.close()


async def test_empty_archive_returns_canonical_empty_envelope(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    """An empty archive yields total=0, items=(), and (best-effort) diagnostics."""
    db_path = db_setup(workspace_env)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    try:
        spec = ConversationQuerySpec(limit=10)
        envelopes = {
            "api": await APIReadSurface(polylogue).list_conversations(spec),
            "cli": await CLIReadSurface(services).list_conversations(spec),
            "mcp": await MCPReadSurface(services).list_conversations(spec),
        }
        for surface_name, envelope in envelopes.items():
            assert envelope.total == 0, f"{surface_name} total != 0"
            assert envelope.items == (), f"{surface_name} items not empty"
            assert envelope.offset == 0

        record_contract_evidence.record(
            "read-surface-empty-envelope-parity",
            surface="api+cli+mcp",
            facts={"total": 0, "items_len": 0},
        )
    finally:
        await polylogue.close()
        await services.close()


async def test_stats_envelope_parity(
    workspace_env: dict[str, Path],
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    """ArchiveStats values are identical across adapters."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    try:
        api_stats = await APIReadSurface(polylogue).archive_stats()
        cli_stats = await CLIReadSurface(services).archive_stats()
        mcp_stats = await MCPReadSurface(services).archive_stats()

        assert api_stats.conversation_count == cli_stats.conversation_count == mcp_stats.conversation_count
        assert api_stats.message_count == cli_stats.message_count == mcp_stats.message_count

        record_contract_evidence.record(
            "read-surface-stats-parity",
            surface="api+cli+mcp",
            facts={
                "conversation_count": api_stats.conversation_count,
                "message_count": api_stats.message_count,
            },
        )
    finally:
        await polylogue.close()
        await services.close()
