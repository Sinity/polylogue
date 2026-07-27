"""Regression test for polylogue-6o9b.

The daemon's two session-detail fast paths must compute an identical
``message.text`` field for the same message:

- DB-backed: ``daemon/http.py:_do_get_session`` -> ``Polylogue.get_session()``
  -> ``api/archive.py:_archive_message_to_domain``.
- Archive-backed: ``daemon/http.py:_do_archive_get_session`` ->
  ``daemon/http.py:_archive_message_payload``.

Both used to independently reimplement the same "join every block's text"
formula; a message with mixed block types (TEXT + THINKING + TOOL_USE +
TOOL_RESULT) is exactly the shape most likely to expose a silent
re-divergence, since ``daemon/web_shell_reader.py``'s client-side rendering
heuristic dispatches its fold/thinking/tool treatment off this single
flattened string per message.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.daemon.http import DaemonAPIHandler
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, db_setup

_MIXED_BLOCKS = [
    {"type": "text", "text": "Hello prose"},
    {"type": "thinking", "text": "pondering the fix"},
    {"type": "tool_use", "tool_name": "shell", "tool_id": "t1"},
    {"type": "tool_result", "tool_id": "t1", "text": "command output"},
]


async def test_db_backed_and_archive_backed_message_text_agree_for_mixed_blocks(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "mixed-blocks")
        .provider("codex")
        .title("Mixed block types")
        .add_message(text="Hello prose", blocks=list(_MIXED_BLOCKS))
    )

    # DB-backed route: the exact call ``_do_get_session`` makes.
    session = await builder.build()
    assert session is not None
    session_messages = session.messages.to_list()
    assert len(session_messages) == 1
    db_backed_text = session_messages[0].text

    # Archive-backed route: the exact call ``_do_archive_get_session`` makes,
    # invoked directly on the handler (its body touches no other ``self``
    # state, so a bare, unconnected instance is sufficient).
    archive_root = db_path.parent
    with ArchiveStore(archive_root) as archive:
        resolved_id = archive.resolve_session_id(builder.native_session_id())
        envelope = archive.read_session(resolved_id)
    assert len(envelope.messages) == 1

    handler = object.__new__(DaemonAPIHandler)
    archive_payload = handler._archive_message_payload(resolved_id, envelope.messages[0])
    archive_backed_text = archive_payload["text"]

    assert db_backed_text == archive_backed_text

    # Sanity: prove this compared real mixed-block content, not two empty
    # strings agreeing by accident.
    assert db_backed_text is not None
    assert "Hello prose" in db_backed_text
    assert "pondering the fix" in db_backed_text
    assert "command output" in db_backed_text
