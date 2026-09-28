"""MCP privileged capabilities and the judge route's writer-ownership boundary."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from tests.infra.mcp import installed_runtime_services, invoke_surface


def test_mcp_declaration_inventory_names_every_privileged_capability() -> None:
    """A new privileged capability must receive an explicit writer-boundary review."""
    from polylogue.mcp.declarations.registry import MCP_TOOL_DECLARATIONS

    inventory = {
        capability: frozenset(
            declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.required_capability == capability
        )
        for capability in ("write", "judge", "maintenance")
    }
    assert inventory == {
        "write": frozenset({"write", "record_work_event", "emit_decision", "run"}),
        "judge": frozenset({"judge"}),
        "maintenance": frozenset({"maintenance"}),
    }
    assert {declaration.required_capability for declaration in MCP_TOOL_DECLARATIONS} == {
        None,
        "write",
        "judge",
        "maintenance",
    }


def test_judge_only_mcp_route_requires_the_daemon_and_leaves_user_db_untouched(
    workspace_env: dict[str, Path],
) -> None:
    """The real MCP judge handler sends its write to the daemon, never to user.db.

    With no ``polylogued run`` for the archive, the call must refuse with the
    typed ``daemon_required`` code and leave the candidate unjudged.
    Anti-vacuity: a judge route that wrote user.db in-process would change
    the candidate's status and the file bytes; reporting the refusal as the
    generic ``polylogue_error`` fails the code assertion.
    """
    from polylogue.core.enums import AssertionKind
    from polylogue.mcp.declarations.models import MCPCapabilities
    from polylogue.mcp.server import build_server
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection
    from tests.infra.storage_records import db_setup

    root = Path(workspace_env["archive_root"])
    db_setup(workspace_env)
    candidate_id = "candidate-mcp-daemon-boundary"
    user_db = root / "user.db"
    with closing(sqlite3.connect(user_db)) as conn:
        upsert_assertion(
            conn,
            assertion_id=candidate_id,
            target_ref="session:daemon-boundary",
            kind=AssertionKind.TRANSFORM_CANDIDATE,
            value={"candidate_kind": "decision"},
            body_text="MCP judge daemon boundary fixture.",
            evidence_refs=("session:daemon-boundary",),
            status="candidate",
            visibility="private",
            context_policy={"inject": False, "promotion_required": True},
            now_ms=1_700_000_000_000,
        )
        conn.commit()

    before = user_db.read_bytes()
    with installed_runtime_services(root):
        server = build_server(capabilities=MCPCapabilities(judge=True))
        judge = server._tool_manager._tools["judge"].fn
        refused = json.loads(invoke_surface(judge, candidate_ref=f"assertion:{candidate_id}", decision="defer"))

    assert refused["code"] == "daemon_required", refused
    assert refused["detail"] == "FacadeDaemonRequiredError"
    assert str(root) not in refused["message"], "the public message must not name the archive path"
    assert user_db.read_bytes() == before
    with open_readonly_connection(user_db) as conn:
        row = conn.execute("SELECT status FROM assertions WHERE assertion_id = ?", (candidate_id,)).fetchone()
    assert row is not None and row[0] == "candidate"
