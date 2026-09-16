"""CLI and MCP share one session-projection vocabulary (polylogue-mjupn).

MCP's ``get(ref, projection=X)`` carried four hand-written branches, each
re-spelling the projection name, the archive method and the payload key, with
nothing tying them to the registry-checked CLI ``read --view`` vocabulary.

Anti-vacuity: adding a projection name MCP serves that the shared read-view
vocabulary does not declare raises at import; renaming an archive method
without updating the table makes the method-existence assertion red.
"""

from __future__ import annotations

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID
from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
from polylogue.mcp.session_projections import SESSION_LIST_PROJECTIONS


def test_every_mcp_projection_is_a_declared_read_view() -> None:
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_PROFILE_BY_ID)
    assert set(SESSION_LIST_PROJECTIONS) <= set(READ_VIEW_HANDLER_METADATA)


def test_every_declared_method_exists_on_the_archive_facade() -> None:
    from polylogue.api.archive import Polylogue

    for projection in SESSION_LIST_PROJECTIONS.values():
        assert hasattr(Polylogue, projection.method), projection


def test_payload_keys_are_distinct_and_named_after_their_rows() -> None:
    keys = [projection.payload_key for projection in SESSION_LIST_PROJECTIONS.values()]
    assert len(set(keys)) == len(keys)
    assert SESSION_LIST_PROJECTIONS["file-edits"].payload_key == "file_edits"
