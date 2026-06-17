"""Contract tests for the shared write-surface protocol family.

Verifies the protocols defined in
:mod:`polylogue.api.contracts.write_surface` and the adapters in
:mod:`polylogue.api.contracts.{api,cli,mcp}_write_surface`.

Two layers of evidence are recorded:

1. **Static conformance** — every adapter is checked against every
   protocol via ``issubclass``.  This catches missing methods, wrong
   arity, or accidental non-async coroutines that mypy would normally
   flag but that contract tests should also surface as a regression
   tripwire.  The ``assert_implements`` helper inside each adapter
   module already pins this at import time; the test re-asserts at
   runtime.

2. **Envelope parity** — every adapter is invoked over a small seeded
   archive and the returned envelopes are compared for the contract
   fields that matter: ``operation_id``, ``status``, ``outcome``, and
   the typed envelope class itself.  This confirms the static contract
   matches the runtime semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.api.contracts import (
    IndexMaintenanceSurface,
    IngestSurface,
    MaintenanceSurface,
    SessionDeleteSurface,
    TagMutationSurface,
    WriteSurface,
)
from polylogue.api.contracts.api_write_surface import APIWriteSurface
from polylogue.api.contracts.cli_write_surface import CLIWriteSurface
from polylogue.api.contracts.mcp_write_surface import MCPWriteSurface
from polylogue.core.enums import Provider
from polylogue.maintenance.planner import BackfillOperation
from polylogue.operations.import_contracts import ImportOperation
from polylogue.services import build_runtime_services
from polylogue.surfaces.payloads import TagMutationResult
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder, db_setup

# ---------------------------------------------------------------------------
# Static-conformance: every adapter implements every required protocol.
# ---------------------------------------------------------------------------


_PROTOCOL_FAMILY: tuple[type, ...] = (
    IngestSurface,
    MaintenanceSurface,
    IndexMaintenanceSurface,
    TagMutationSurface,
    SessionDeleteSurface,
)

_ADAPTER_CLASSES: tuple[type, ...] = (
    APIWriteSurface,
    CLIWriteSurface,
    MCPWriteSurface,
)


@pytest.mark.parametrize("adapter_cls", _ADAPTER_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("protocol", _PROTOCOL_FAMILY, ids=lambda p: p.__name__)
def test_adapter_conforms_to_write_protocol(adapter_cls: type, protocol: type) -> None:
    """Every adapter class structurally satisfies every write-surface protocol."""
    assert issubclass(adapter_cls, protocol), (
        f"{adapter_cls.__name__} does not implement {protocol.__name__}; "
        "regenerate the adapter so it carries the canonical method signatures."
    )


def test_composite_write_surface_subsumes_required_protocols() -> None:
    """The composite WriteSurface declares the required write methods."""
    for adapter_cls in _ADAPTER_CLASSES:
        assert issubclass(adapter_cls, WriteSurface), (
            f"{adapter_cls.__name__} should satisfy the composite WriteSurface "
            "protocol so callers can depend on the full write contract."
        )


# ---------------------------------------------------------------------------
# Runtime parity: envelopes are interchangeable across adapters.
# ---------------------------------------------------------------------------


async def _seed_archive(db_path: Path) -> str:
    """Seed one session and return its archive session id."""
    await (
        SessionBuilder(db_path, "conv-alpha")
        .provider(Provider.CLAUDE_AI.value)
        .title("Alpha")
        .add_message(text="alpha message body")
        .build()
    )
    return native_session_id_for("claude-ai", "conv-alpha")


async def test_tag_mutation_envelope_parity(
    workspace_env: dict[str, Path],
) -> None:
    """``add_tag``/``remove_tag`` return the same TagMutationResult across adapters."""
    db_path = db_setup(workspace_env)
    conv_id = await _seed_archive(db_path)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    try:
        api_surface = APIWriteSurface(polylogue)
        cli_surface = CLIWriteSurface(services, polylogue=polylogue)
        mcp_surface = MCPWriteSurface(services, polylogue=polylogue)

        # First add via API; subsequent adds on the same tag should be no_op.
        api_added = await api_surface.add_tag(conv_id, "shared-tag")
        cli_added = await cli_surface.add_tag(conv_id, "shared-tag")
        mcp_added = await mcp_surface.add_tag(conv_id, "shared-tag")

        for surface_name, result in (
            ("api", api_added),
            ("cli", cli_added),
            ("mcp", mcp_added),
        ):
            assert isinstance(result, TagMutationResult), (
                f"{surface_name} returned {type(result).__name__}, not TagMutationResult"
            )
        assert api_added.outcome == "added", f"first add must be added, got {api_added.outcome!r}"
        assert cli_added.outcome == "no_op", "second add must be no_op (idempotent)"
        assert mcp_added.outcome == "no_op", "third add must be no_op (idempotent)"

        # Now remove from one surface; subsequent removes are not_present.
        api_removed = await api_surface.remove_tag(conv_id, "shared-tag")
        cli_removed = await cli_surface.remove_tag(conv_id, "shared-tag")
        mcp_removed = await mcp_surface.remove_tag(conv_id, "shared-tag")
        assert api_removed.outcome == "removed"
        assert cli_removed.outcome == "not_present"
        assert mcp_removed.outcome == "not_present"

    finally:
        await polylogue.close()
        await services.close()


async def test_maintenance_envelope_parity(
    workspace_env: dict[str, Path],
) -> None:
    """``run_maintenance`` dry-run returns a typed BackfillOperation across adapters."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    try:
        api_surface = APIWriteSurface(polylogue)
        cli_surface = CLIWriteSurface(services)
        mcp_surface = MCPWriteSurface(services)

        # Dry-run against an empty target tuple — the planner resolves to the
        # full target set or returns a FAILED operation when none resolve.
        # Either way the envelope shape is the contract under test.
        envelopes = {
            "api": await api_surface.run_maintenance((), dry_run=True),
            "cli": await cli_surface.run_maintenance((), dry_run=True),
            "mcp": await mcp_surface.run_maintenance((), dry_run=True),
        }

        for surface_name, envelope in envelopes.items():
            assert isinstance(envelope, BackfillOperation), (
                f"{surface_name} returned {type(envelope).__name__}, not BackfillOperation"
            )
            assert envelope.operation_id, f"{surface_name} envelope missing operation_id"
            assert envelope.status.value in {"completed", "failed", "pending", "running"}, (
                f"{surface_name} unexpected status {envelope.status.value!r}"
            )

    finally:
        await polylogue.close()
        await services.close()


async def test_ingest_envelope_parity_for_missing_path(
    workspace_env: dict[str, Path],
) -> None:
    """``ingest_path`` returns a typed ImportOperation across adapters.

    For a path that does not exist, every adapter must produce
    ``status="failed"`` rather than silently succeeding.  This is the
    truthfulness invariant called out in #861's acceptance criteria.
    """
    db_path = db_setup(workspace_env)

    archive_root = workspace_env["archive_root"]
    polylogue = Polylogue(archive_root=archive_root, db_path=db_path)
    services = build_runtime_services(db_path=db_path)
    missing = workspace_env["archive_root"] / "does-not-exist"
    try:
        api_surface = APIWriteSurface(polylogue)
        cli_surface = CLIWriteSurface(services)
        mcp_surface = MCPWriteSurface(services)

        envelopes = {
            "api": await api_surface.ingest_path(missing),
            "cli": await cli_surface.ingest_path(missing),
            "mcp": await mcp_surface.ingest_path(missing),
        }

        for surface_name, envelope in envelopes.items():
            assert isinstance(envelope, ImportOperation), (
                f"{surface_name} returned {type(envelope).__name__}, not ImportOperation"
            )
            assert envelope.status == "failed", (
                f"{surface_name} must report status=failed for missing path, got {envelope.status!r}"
            )
            assert envelope.error, f"{surface_name} status=failed envelope must populate error field"
            assert envelope.operation_id

    finally:
        await polylogue.close()
        await services.close()
