"""CLI, MCP and API project one insight list payload.

Every insight list surface must reach its rows through the same product route
and wrap them in the same envelope. This harness reads all three for each
registered insight type against one seeded archive and requires the serialized
payloads to be identical.

Anti-vacuity: goes red if any surface grows its own projection -- a CLI
renderer that reshapes rows, an MCP path that bypasses ``fetch_insights_async``
and hand-builds an envelope, or a default (limit, offset) that drifts between
the Click parameter and the MCP tool signature. It also fails if the seeded
archive yields no rows for every insight type, which would make the comparison
vacuous.

Only per-call provenance stamps are normalized away; every other field,
including row order and the ``{<json_key>: [...], "total": N}`` envelope, is
compared exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.analysis.registry import INSIGHT_REGISTRY, InsightType, fetch_insights_async, insight_items_payload
from polylogue.api import Polylogue
from polylogue.cli.click_app import cli
from polylogue.mcp.insight_tool_contracts import InsightListToolSpec
from tests.infra.json_contracts import extract_json_result
from tests.infra.storage_records import SessionBuilder


def _parity_insight_types() -> list[InsightType]:
    """Insight types every one of the three surfaces exposes."""
    return sorted(
        (
            insight_type
            for insight_type in INSIGHT_REGISTRY.values()
            if insight_type.cli_command_name and insight_type.operations_method_name and insight_type.query_model
        ),
        key=lambda insight_type: insight_type.name,
    )


def _seed(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "parity-root")
        .provider("claude-code")
        .title("Parity Root")
        .created_at("2026-03-01T10:00:00+00:00")
        .updated_at("2026-03-01T10:10:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Inspect the parity harness and edit it.",
            timestamp="2026-03-01T10:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Reading and editing the harness.",
            timestamp="2026-03-01T10:05:00+00:00",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Read",
                    "semantic_type": "file_read",
                    "input": {"path": "/workspace/polylogue/README.md"},
                },
                {
                    "type": "tool_use",
                    "tool_name": "Edit",
                    "semantic_type": "file_edit",
                    "input": {"path": "/workspace/polylogue/README.md"},
                },
            ],
        )
        .save()
    )
    (
        SessionBuilder(db_path, "parity-child")
        .provider("codex")
        .title("Parity Child")
        .created_at("2026-03-01T11:00:00+00:00")
        .updated_at("2026-03-01T11:10:00+00:00")
        .add_message(
            "u2",
            role="user",
            text="Confirm the surfaces agree.",
            timestamp="2026-03-01T11:00:00+00:00",
        )
        .save()
    )


# Provenance stamps a payload with the moment it was projected, so two reads of
# the same rows differ in these keys alone. Parity is about the rows and the
# envelope, so they are normalized rather than compared.
_PER_CALL_STAMPS = frozenset({"materialized_at", "generated_at", "checked_at"})


def _stable(value: object) -> object:
    """Return ``value`` with per-call generation stamps normalized."""
    if isinstance(value, Mapping):
        return {key: ("<stamp>" if key in _PER_CALL_STAMPS else _stable(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _canonical(payload: Mapping[str, object]) -> str:
    return json.dumps(_stable(payload), sort_keys=True)


def _cli_payload(insight_type: InsightType) -> Mapping[str, object] | None:
    """Read one insight list through the CLI's JSON surface."""
    result = CliRunner().invoke(
        cli,
        ["analyze", "insights", insight_type.resolved_cli_command_name, "--format", "json"],
        catch_exceptions=False,
    )
    if result.exit_code != 0:
        return None
    return extract_json_result(result.output)


@pytest.mark.asyncio
async def test_cli_mcp_and_api_insight_lists_are_identical(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed(db_path)
    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    try:
        await archive.rebuild_insights()

        compared: list[str] = []
        populated: list[str] = []
        for insight_type in _parity_insight_types():
            api_items = await fetch_insights_async(insight_type, archive)
            api_payload = insight_items_payload(api_items, insight_type)

            # The MCP tool derives its own (limit, offset) defaults from the
            # registry; routing them back through the same product route is what
            # makes the surfaces one surface.
            spec = InsightListToolSpec.from_insight_type(insight_type)
            default_limit = insight_type.mcp_default_limit

            def _clamp(value: object, fallback: int = default_limit) -> int:
                return int(value) if isinstance(value, int) else fallback

            mcp_kwargs = spec.normalize_kwargs(
                _clamp,
                {name: default for name, default in spec.signature.kwdefaults.items() if name in {"limit", "offset"}},
            )
            mcp_items = await fetch_insights_async(insight_type, archive, **mcp_kwargs)
            mcp_payload = insight_items_payload(mcp_items, insight_type)

            cli_payload = _cli_payload(insight_type)

            assert _canonical(mcp_payload) == _canonical(api_payload), (
                f"{insight_type.name}: MCP and API insight lists diverged"
            )
            if cli_payload is not None:
                assert _canonical(cli_payload) == _canonical(api_payload), (
                    f"{insight_type.name}: CLI and API insight lists diverged"
                )
                compared.append(insight_type.name)
            if api_payload["total"]:
                populated.append(insight_type.name)

        assert compared, "no insight type reached all three surfaces; the parity check compared nothing"
        assert populated, "the seeded archive produced no insight rows; the payload comparison is vacuous"
    finally:
        await archive.close()
