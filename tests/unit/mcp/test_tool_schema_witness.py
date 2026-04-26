"""Regression test: the published MCP tool wire schema matches the committed witness.

Drift here would silently invalidate clients (CLI/agent integrations consume
the published JSON Schema). Re-running ``build_server`` and diffing against the
committed witness catches any change to a tool name, parameter set, required
flag, or type.

To accept legitimate changes, regenerate the witness data:

    python -c "
    import json
    from polylogue.mcp.server import build_server
    m = build_server()
    out = sorted(({'name': name, 'parameters': t.parameters} for name, t in m._tool_manager._tools.items()), key=lambda r: r['name'])
    with open('tests/data/witnesses/mcp-tool-schemas.json', 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True); f.write('\\n')
    "

See `#448 <https://github.com/Sinity/polylogue/issues/448>`_.
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.mcp.server import build_server
from polylogue.proof.witnesses import WITNESS_SCHEMA_VERSION, WitnessMetadata

WITNESS_PATH = Path(__file__).resolve().parents[3] / "tests" / "witnesses" / "mcp-tool-schemas.witness.json"
DATA_PATH = Path(__file__).resolve().parents[3] / "tests" / "data" / "witnesses" / "mcp-tool-schemas.json"


def _current_tool_catalog() -> list[dict[str, object]]:
    server = build_server()
    tools = server._tool_manager._tools
    return [{"name": name, "parameters": tools[name].parameters} for name in sorted(tools)]


def test_committed_witness_metadata_validates() -> None:
    metadata = WitnessMetadata.read(WITNESS_PATH)
    assert metadata.validation_errors() == ()
    assert metadata.schema_version == WITNESS_SCHEMA_VERSION
    assert metadata.committed is True


def test_mcp_tool_catalog_matches_witness() -> None:
    expected = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    actual = _current_tool_catalog()
    assert actual == expected, (
        "MCP tool wire schema drifted from committed witness. "
        "If the change is intentional, regenerate "
        "tests/data/witnesses/mcp-tool-schemas.json (see this test's docstring)."
    )


def test_mcp_tool_catalog_is_non_empty() -> None:
    catalog = _current_tool_catalog()
    assert catalog, "build_server registered no tools — registration regression?"
    for entry in catalog:
        params = entry["parameters"]
        assert isinstance(params, dict)
        assert params.get("type") == "object"
