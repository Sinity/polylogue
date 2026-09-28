"""Sidecar lookups stay inside the source tree that declared them.

A Gemini CLI snapshot's ``sessionId`` names a ``tool-outputs/`` subdirectory,
a value an untrusted export controls. Uncontained, an import could pull bytes
from anywhere the daemon can read into the archive as session evidence.

Anti-vacuity: revert the containment check and the escaping case below
resolves to the outside-the-tree target instead of being refused --
``escape_target`` appears in the resolved path.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.sources.live.gemini_tool_output_sidecars import resolve_tool_outputs_dir


def test_gemini_tool_outputs_dir_resolves_an_ordinary_session_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "project" / "chats" / "session-a.json"

    resolved = resolve_tool_outputs_dir(snapshot, "abc-123")

    assert resolved == tmp_path / "project" / "tool-outputs" / "session-abc-123"


def test_gemini_tool_outputs_dir_refuses_a_traversing_session_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "project" / "chats" / "session-a.json"

    assert resolve_tool_outputs_dir(snapshot, "hop/../../../../escape_target") is None
    assert resolve_tool_outputs_dir(snapshot, "a/b") is None
    assert resolve_tool_outputs_dir(snapshot, "/escape_target") is None
