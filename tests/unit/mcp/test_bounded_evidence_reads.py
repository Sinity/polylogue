"""MCP pages select nested evidence in the canonical reader before hydration."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue import Polylogue
from polylogue.archive.context_models import ContextImage
from polylogue.config import resolve_runtime_config
from polylogue.mcp import server_support
from polylogue.services import RuntimeServices
from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async
from tests.infra.mcp_evidence_pages import seed_evidence_pages


@pytest.mark.asyncio
async def test_delegation_subtree_read_pages_all_nodes_and_rejects_changed_frame(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    root = tmp_path / "archive"
    session_id, identifiers = seed_evidence_pages(root)
    read = mcp_server._tool_manager._tools["read"].fn
    ref = f"delegation:subtree:{session_id}"
    with installed_runtime_services(root):
        first = json.loads(await invoke_surface_async(read, ref=ref, limit=5))
        assert first["payload"]["node_count"] == len(identifiers)
        assert len(first["payload"]["nodes"]) == len(first["object_refs"]) == 5
        seen = [item["session_id"] for item in first["payload"]["nodes"]]
        token = first["payload"]["continuation"]
        assert token is not None
        while token is not None:
            page = json.loads(await invoke_surface_async(read, ref=ref, continuation=token))
            assert page["payload"]["node_count"] == len(identifiers), page
            assert len(page["payload"]["nodes"]) <= 5
            seen.extend(item["session_id"] for item in page["payload"]["nodes"])
            token = page["payload"]["continuation"]
        assert seen == [identifiers[0], *sorted(identifiers[1:])]
        assert len(set(seen)) == len(identifiers)
        with sqlite3.connect(root / "index.db") as conn:
            conn.execute(
                "UPDATE delegation_facts SET mapping_state='quarantined' WHERE delegation_id='fixture-edge-000'"
            )
        refused = json.loads(await invoke_surface_async(read, ref=ref, continuation=first["payload"]["continuation"]))
        assert refused["code"] == "query_continuation_stale"


@pytest.mark.asyncio
async def test_context_receipt_pages_never_decode_images_and_keep_requested_limit(
    mcp_server: MCPServerUnderTest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    seed_evidence_pages(root, children=0, receipts=225)
    context = mcp_server._tool_manager._tools["context"].fn

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("receipt summaries must not decode context images")

    monkeypatch.setattr(ContextImage, "model_validate_json", forbidden)
    with installed_runtime_services(root):
        first = json.loads(await invoke_surface_async(context, intent="lookup", recipient_ref="agent:fixture", limit=5))
        second = json.loads(
            await invoke_surface_async(
                context, intent="lookup", recipient_ref="agent:fixture", limit=5, offset=first["next_offset"]
            )
        )
        assert first["total"] == second["total"] == 225
        assert len(first["items"]) == len(second["items"]) == 5
        assert first["next_offset"] == 5 and second["next_offset"] == 10
        assert not {item["snapshot_ref"] for item in first["items"]} & {
            item["snapshot_ref"] for item in second["items"]
        }
        assert all("context_image" not in item for item in first["items"])
        last = json.loads(
            await invoke_surface_async(context, intent="lookup", recipient_ref="agent:fixture", limit=5, offset=220)
        )
        assert last["next_offset"] is None and len(last["items"]) == 5
    page = await Polylogue(archive_root=root).list_context_deliveries(recipient_ref="agent:fixture", limit=225)
    assert len(page.items) == page.total == page.limit == 225
    assert page.next_offset is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["sinex", "archive"])
async def test_status_uses_injected_runtime_sinex_mode(
    mcp_server: MCPServerUnderTest, tmp_path: Path, scope: str
) -> None:
    root = tmp_path / "archive"
    seed_evidence_pages(root, children=0)
    runtime = resolve_runtime_config(
        environment={"HOME": str(tmp_path), "POLYLOGUE_SITE_CONFIG": ""},
        cli_overrides={"archive_root": str(root), "sinex_mode": "primary"},
    )
    services = RuntimeServices(runtime=runtime)
    with (
        patch.object(server_support, "_get_runtime_services", return_value=services),
        patch(
            "polylogue.config.load_polylogue_config", side_effect=AssertionError("status must use the injected config")
        ),
    ):
        result = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["status"].fn, scope=scope))
    section = "sinex" if scope == "sinex" else "sinex_publication"
    assert result[section]["mode"] == "primary"
    assert services.get_config().with_sources([]).sinex_mode == "primary"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["sinex", "archive"])
async def test_status_returns_invalid_config_mode_evidence_instead_of_tool_error(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
    scope: str,
) -> None:
    root = tmp_path / "archive"
    seed_evidence_pages(root, children=0)
    runtime = resolve_runtime_config(
        environment={"HOME": str(tmp_path), "POLYLOGUE_SITE_CONFIG": "", "POLYLOGUE_SINEX_MODE": "bogus"},
        cli_overrides={"archive_root": str(root)},
    )
    services = RuntimeServices(runtime=runtime)
    with patch.object(server_support, "_get_runtime_services", return_value=services):
        result = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["status"].fn, scope=scope))
    section = "sinex" if scope == "sinex" else "sinex_publication"
    assert result["scope"] == scope
    assert result[section]["mode"] == "bogus"
    assert result[section]["state"] == "unavailable"
    assert result[section]["code"] == "sinex_mode_unrecognized"
    assert "active_lag" not in result[section]
    assert "error" not in result
