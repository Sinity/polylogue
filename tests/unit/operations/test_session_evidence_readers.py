"""The moved evidence views answer with the facade's own rows (polylogue-r3cuz).

``file-edits``, ``agent-policies`` and ``web-content`` moved from opening the
archive in this process through ``Polylogue`` onto the declared ``session.read``
evidence kinds.  Both CLI legs (direct and daemon) now run the *same*
operation, so a daemon/direct parity test cannot see a field the move dropped
or renamed -- both legs would drop it together.

These tests compare the operation's evidence body against the Python API
reader the view used before the move, field for field and in order.  That is
the comparison the move has to survive: the row vocabulary is the contract,
not merely "some rows came back".

Anti-vacuity: rename, drop or reorder any key in
``operations/session_evidence.py`` -- for instance projecting
``structured_patch_json`` instead of ``structured_patch``, or ordering file
edits by ``observed_at_ms`` instead of ``(message_id, tool_use_block_id)`` --
and the corresponding comparison fails.  The non-empty assertions keep two
empty relations from agreeing trivially.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedFileEdit, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session

_NATIVE_ID = "evidence-readers"
_SESSION_ID = f"claude-code-session:{_NATIVE_ID}"


def _seed(archive_root: Path) -> None:
    with ArchiveStore(archive_root) as archive_db:
        write_index_session(
            archive_db,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id=_NATIVE_ID,
                title="Evidence readers",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{index}",
                        role=Role.ASSISTANT,
                        position=index * 2,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Edit",
                                tool_id=f"edit-{index}",
                                tool_input={"file_path": f"/tmp/file-{index}.py"},
                            )
                        ],
                    )
                    for index in range(2)
                ]
                + [
                    ParsedMessage(
                        provider_message_id=f"r{index}",
                        role=Role.USER,
                        position=index * 2 + 1,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                outcome_unknown_reason="not_reported",
                                tool_id=f"edit-{index}",
                                text="applied",
                                file_edit=ParsedFileEdit(
                                    file_path=f"/tmp/file-{index}.py",
                                    structured_patch=[
                                        {
                                            "oldStart": 1,
                                            "oldLines": 1,
                                            "newStart": 1,
                                            "newLines": 2,
                                            "lines": [f"+line {index}"],
                                        }
                                    ],
                                    original_file=f"before {index}\n",
                                    old_string=f"old {index}",
                                    new_string=f"new {index}",
                                    replace_all=bool(index),
                                    user_modified=not index,
                                ),
                            )
                        ],
                    )
                    for index in range(2)
                ],
            ),
        )


@pytest.fixture
def seeded_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    run_off_event_loop(lambda: _seed(archive_root))
    return archive_root


async def test_file_edit_rows_match_the_facade_reader_they_replaced(seeded_root: Path) -> None:
    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import read_file_edits_page

    archive = Polylogue(archive_root=seeded_root)
    try:
        facade_rows = await archive.get_file_edits(_SESSION_ID)
    finally:
        await archive.close()

    with ArchiveStore.open_existing(seeded_root) as store:
        rows, total = read_file_edits_page(store, _SESSION_ID, limit=len(facade_rows or ()) or 1, offset=0)

    assert facade_rows, "the fixture must actually record file edits, or this comparison is vacuous"
    assert rows == facade_rows
    assert total == len(facade_rows or ())


async def test_agent_policy_rows_match_the_facade_reader_they_replaced(seeded_root: Path) -> None:
    """An empty relation still has to agree in shape, not merely in emptiness."""

    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import read_agent_policies_evidence

    archive = Polylogue(archive_root=seeded_root)
    try:
        facade_rows = await archive.get_agent_policies(_SESSION_ID)
    finally:
        await archive.close()

    with ArchiveStore.open_existing(seeded_root) as store:
        evidence = read_agent_policies_evidence(store, _SESSION_ID)

    assert facade_rows is not None
    assert evidence["agent_policies"] == facade_rows
    assert evidence["total"] == len(facade_rows)


async def test_web_content_rows_match_the_facade_reader_they_replaced(seeded_root: Path) -> None:
    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import read_web_content_constructs_page

    archive = Polylogue(archive_root=seeded_root)
    try:
        facade_rows = await archive.get_web_content_constructs(_SESSION_ID)
    finally:
        await archive.close()

    with ArchiveStore.open_existing(seeded_root) as store:
        rows, total = read_web_content_constructs_page(store, _SESSION_ID, limit=len(facade_rows or ()) or 1, offset=0)

    assert facade_rows is not None
    assert rows == facade_rows
    assert total == len(facade_rows or ())


def _seed_events(archive_root: Path, count: int) -> str:
    """One session carrying ``count`` timeline events, written the ordinary way."""
    from polylogue.sources.parsers.base import ParsedSessionEvent

    with ArchiveStore(archive_root) as archive_db:
        write_index_session(
            archive_db,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="ext-evidence-window",
                title="Windowed evidence",
                messages=[ParsedMessage(provider_message_id="m0", role=Role.ASSISTANT, position=0, text="body")],
                session_events=[
                    ParsedSessionEvent(event_type="world_state", payload={"n": index}) for index in range(count)
                ],
            ),
        )
    return "claude-code-session:ext-evidence-window"


def test_clipped_evidence_page_is_not_complete(tmp_path: Path) -> None:
    """A windowed evidence page separates truncated from finished in the body.

    This is why ``events`` could not be lowered as a whole-evidence read: it
    answered a ``--limit`` with the clipped count as its own ``total``, so
    nothing in the payload distinguished a cut body from a complete one, and
    ``SessionReadResult`` refuses a body that claims completeness it does not
    have.

    Anti-vacuity: return the page length as ``total`` from the reader (or drop
    ``complete`` from the body) and the first three assertions go green
    together while the page is still short -- which is exactly the shape this
    contract exists to make impossible.  A ``limit`` at or above the relation's
    own row count cannot witness it, so the window here is strictly smaller.
    """
    from polylogue.operations.daemon_reads import execute_read_operation

    root = tmp_path / "archive"
    session_id = _seed_events(root, 5)

    with ArchiveStore.open_existing(root) as archive:
        page = execute_read_operation(
            "session.read",
            {"ref": f"session:{session_id}", "kind": "events", "limit": 2, "offset": 0},
            archive=archive,
            serving_identity="test",
        )
        whole = execute_read_operation(
            "session.read",
            {"ref": f"session:{session_id}", "kind": "events", "limit": 5, "offset": 0},
            archive=archive,
            serving_identity="test",
        )

    assert page["total"] == 5, "total is the relation's own row count, not the page's"
    assert page["complete"] is False
    assert page["next_offset"] == 2
    assert page["continuation"], "a short page mints a continuation instead of claiming completeness"
    assert whole["complete"] is True
    assert whole["next_offset"] is None
    assert whole["continuation"] in (None, "")


def test_evidence_token_cannot_resume_messages(tmp_path: Path) -> None:
    """The two families refuse each other's tokens by name.

    ``transcript_window`` owns the message window's projection token; a
    windowed evidence relation mints its own.  Without that separation a token
    minted for "events 2..4" would be resumable by a reader that composes
    messages with it.

    Anti-vacuity: mint the evidence continuation in the message family's
    projection and this stops raising -- the transcript read accepts the token
    and pages messages under an events cursor.
    """
    from polylogue.archive.query.transaction import QueryContinuationInvalidError
    from polylogue.operations.daemon_reads import execute_read_operation

    root = tmp_path / "archive"
    session_id = _seed_events(root, 5)

    with ArchiveStore.open_existing(root) as archive:
        page = execute_read_operation(
            "session.read",
            {"ref": f"session:{session_id}", "kind": "events", "limit": 2, "offset": 0},
            archive=archive,
            serving_identity="test",
        )
        token = page["continuation"]
        assert token
        with pytest.raises((QueryContinuationInvalidError, ValueError)):
            execute_read_operation(
                "session.read",
                {"ref": f"session:{session_id}", "kind": "transcript", "continuation": token},
                archive=archive,
                serving_identity="test",
            )


def _seed_large_file_edits(archive_root: Path, *, rows: int, original_file_bytes: int) -> str:
    """One session whose file edits carry deliberately large ``original_file`` bodies.

    ``original_file`` is the pre-edit contents of the touched file, so it is
    the field that makes a file-edit row unbounded in production. The fixture
    is synthetic and deterministic; the size is the point, not the content.
    """
    payload = "x" * original_file_bytes
    with ArchiveStore(archive_root) as archive_db:
        write_index_session(
            archive_db,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="ext-large-file-edits",
                title="Large file edits",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{index}",
                        role=Role.ASSISTANT,
                        position=index * 2,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Edit",
                                tool_id=f"edit-{index}",
                                tool_input={"file_path": f"/tmp/big-{index}.py"},
                            )
                        ],
                    )
                    for index in range(rows)
                ]
                + [
                    ParsedMessage(
                        provider_message_id=f"r{index}",
                        role=Role.USER,
                        position=index * 2 + 1,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                outcome_unknown_reason="not_reported",
                                tool_id=f"edit-{index}",
                                text="applied",
                                file_edit=ParsedFileEdit(
                                    file_path=f"/tmp/big-{index}.py",
                                    structured_patch=[],
                                    original_file=payload,
                                    old_string="old",
                                    new_string="new",
                                ),
                            )
                        ],
                    )
                    for index in range(rows)
                ],
            ),
        )
    return "claude-code-session:ext-large-file-edits"


def _reassemble(windows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Independent consumer: every byte/field/row must arrive once and in order."""
    import base64
    import json

    from polylogue.operations.read_contracts import EvidenceWindowBody

    rows: list[dict[str, object]] = []
    pending: dict[str, bytearray] = {}
    current: dict[str, object] = {}
    for window in windows:
        EvidenceWindowBody.model_validate(window)
        before = len(rows)
        assert window["offset"] == before
        fragment = window.get("row_fragment")
        if fragment is None:
            assert not pending and not current
            rows.extend(window["rows"])
        else:
            assert fragment["row_offset"] == before
            assert not window["rows"]
            for part in fragment["fields"]:
                name = part["field"]
                assert name not in current, "a completed field was delivered twice"
                buffer = pending.setdefault(name, bytearray())
                assert len(buffer) == part["offset"], "a field skipped or repeated bytes"
                buffer.extend(base64.b64decode(part["data_base64"], validate=True))
                assert len(buffer) <= part["total_bytes"]
                if len(buffer) == part["total_bytes"]:
                    text = buffer.decode("utf-8")
                    current[name] = json.loads(text) if part["encoding"] == "json" else text
                    del pending[name]
            if fragment["complete"]:
                assert not pending
                rows.append(current)
                current = {}
        assert window["returned"] == len(rows) - before
        assert window["complete"] is (len(rows) == window["total"])
    assert not pending and not current
    assert windows[-1]["complete"] and windows[-1]["continuation"] is None
    return rows


@pytest.mark.parametrize("row_count", [1, 3])
def test_large_file_edits_are_losslessly_delivered(tmp_path: Path, row_count: int) -> None:
    """07.F049: both a multi-row overflow and one >8 MiB edit remain readable."""
    import json

    from polylogue.archive.query.transaction import QueryContinuation

    large_fixture_bytes = 8 * 1024 * 1024
    from polylogue.operations.daemon_reads import execute_read_operation
    from polylogue.operations.session_evidence import read_file_edits_page

    root = tmp_path / "archive"
    row_bytes = large_fixture_bytes // 2 if row_count == 3 else large_fixture_bytes + 4096
    session_id = _seed_large_file_edits(root, rows=row_count, original_file_bytes=row_bytes)
    windows: list[dict[str, Any]] = []
    token: str | None = None
    positions: list[tuple[int, int, int]] = []
    result_refs: set[str] = set()
    validity: set[tuple[int | None, int | None]] = set()
    with ArchiveStore.open_existing(root) as archive:
        expected, _ = read_file_edits_page(archive, session_id, limit=row_count, offset=0)
        while True:
            request: dict[str, object] = {"ref": f"session:{session_id}", "kind": "file-edits"}
            request.update({"continuation": token} if token is not None else {"limit": 1})
            result = execute_read_operation("session.read", request, archive=archive, serving_identity="test")
            assert len(json.dumps(result).encode("utf-8")) <= large_fixture_bytes
            window = cast("dict[str, Any]", result["evidence_window"])
            assert window["total"] == row_count
            windows.append(window)
            token = window["continuation"]
            if token is None:
                break
            decoded = QueryContinuation.decode(token)
            cursor = decoded.cursor or {"field": 0, "byte": 0}
            position = (decoded.request.offset, cast(int, cursor["field"]), cast(int, cursor["byte"]))
            assert not positions or position > positions[-1], "continuation made no progress"
            positions.append(position)
            result_refs.add(decoded.result_ref)
            validity.add((decoded.request.issued_at, decoded.request.expires_at))
    assert any(window.get("row_fragment") for window in windows)
    assert len(result_refs) == len(validity) == 1
    assert _reassemble(windows) == expected


def _seed_fragment_evidence(root: Path, kind: str, text: str) -> None:
    """Synthetic derived rows; no operator archive or acquisition is involved."""
    import json
    import sqlite3

    from polylogue.core.enums import WebConstructType

    _seed(root)
    with sqlite3.connect(root / "index.db") as conn:
        block, message = conn.execute(
            "SELECT tool_use_block_id, message_id FROM file_edits ORDER BY tool_use_block_id LIMIT 1"
        ).fetchone()
        if kind == "file-edits":
            conn.execute(
                "UPDATE file_edits SET original_file = ?, structured_patch_json = ?, "
                "old_string = '', new_string = ?, replace_all = 1, user_modified = 0 WHERE tool_use_block_id = ?",
                (text, json.dumps([{"lines": [text]}]), text, block),
            )
        else:
            conn.execute(
                "INSERT INTO web_content_constructs "
                "(session_id, message_id, block_id, position, provider, construct_type, title, text) "
                "VALUES (?, ?, ?, 0, 'claude', ?, '', ?)",
                (_SESSION_ID, message, block, next(iter(WebConstructType)).value, text),
            )


@pytest.mark.parametrize("kind", ["file-edits", "web-content"])
async def test_small_transport_budget_preserves_unicode_json_nulls_and_empty_fields(tmp_path: Path, kind: str) -> None:
    """The API's bounded owner can serve MCP-sized pages without a whole-row read."""
    import json

    from polylogue.api import Polylogue
    from polylogue.operations.session_evidence import SESSION_EVIDENCE_PAGE_READERS

    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed_fragment_evidence(root, kind, 'zażółć\x00🧪"\\\n' * 5000))
    with ArchiveStore.open_existing(root) as store:
        expected, _ = SESSION_EVIDENCE_PAGE_READERS[kind](store, _SESSION_ID, 100, 0)
    archive = Polylogue(archive_root=root)
    windows: list[dict[str, Any]] = []
    token: str | None = None
    try:
        while True:
            window = await archive.read_session_evidence_window(
                _SESSION_ID, kind, limit=50, continuation=token, max_bytes=20_904
            )
            assert window is not None
            assert len(json.dumps(window).encode("utf-8")) <= 20_904
            windows.append(window)
            following = cast("str | None", window["continuation"])
            assert following is None or following != token
            token = following
            if token is None:
                break
    finally:
        await archive.close()
    assert len(windows) > 2
    assert _reassemble(windows) == expected


@pytest.mark.parametrize("kind", ["file-edits", "web-content"])
def test_fragment_continuation_rejects_same_length_evidence_rewrite(tmp_path: Path, kind: str) -> None:
    """Changing only the evidence relation, not its session, invalidates the token."""
    import sqlite3

    from polylogue.archive.query.transaction import QueryContinuationStaleError
    from polylogue.operations.session_evidence import read_session_evidence_window

    root = tmp_path / "archive"
    _seed_fragment_evidence(root, kind, "a" * 40_000)
    with ArchiveStore.open_existing(root) as archive:
        first = read_session_evidence_window(
            archive, kind, ref=f"session:{_SESSION_ID}", limit=1, offset=0, continuation=None, max_bytes=10_000
        )
    assert first is not None and first["row_fragment"] and first["continuation"]
    table, column = ("file_edits", "original_file") if kind == "file-edits" else ("web_content_constructs", "text")
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(f"UPDATE {table} SET {column} = ? WHERE length({column}) = 40000", ("b" * 40_000,))
    with ArchiveStore.open_existing(root) as archive, pytest.raises(QueryContinuationStaleError):
        read_session_evidence_window(
            archive,
            kind,
            ref=f"session:{_SESSION_ID}",
            limit=1,
            offset=0,
            continuation=cast(str, first["continuation"]),
            max_bytes=10_000,
        )


@pytest.mark.parametrize(
    "cursor",
    [
        {"field": -1, "byte": 0},
        {"field": 999, "byte": 0},
        {"field": True, "byte": 0},
        {"field": 0, "byte": -1},
        {"field": 0, "byte": 2**62},
        {"unexpected": 1},
    ],
)
def test_fragment_cursor_rejects_invalid_coordinates(tmp_path: Path, cursor: dict[str, object]) -> None:
    from polylogue.archive.query.transaction import QueryContinuation, QueryContinuationInvalidError
    from polylogue.operations.session_evidence import read_session_evidence_window

    root = tmp_path / "archive"
    _seed_fragment_evidence(root, "file-edits", "a" * 40_000)
    with ArchiveStore.open_existing(root) as archive:
        first = read_session_evidence_window(
            archive,
            "file-edits",
            ref=f"session:{_SESSION_ID}",
            limit=1,
            offset=0,
            continuation=None,
            max_bytes=10_000,
        )
        assert first is not None
        decoded = QueryContinuation.decode(cast(str, first["continuation"]))
        malformed = QueryContinuation(decoded.request, decoded.result_ref, cursor=cursor).encode()
        with pytest.raises(QueryContinuationInvalidError):
            read_session_evidence_window(
                archive,
                "file-edits",
                ref=f"session:{_SESSION_ID}",
                limit=1,
                offset=0,
                continuation=malformed,
                max_bytes=10_000,
            )


def test_oversized_row_never_enters_the_full_row_mapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.operations import session_evidence

    root = tmp_path / "archive"
    _seed_fragment_evidence(root, "file-edits", "a" * 40_000)

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("materialized an oversized row before fragmenting it")

    monkeypatch.setattr(session_evidence, "read_file_edits_page", refuse)
    with ArchiveStore.open_existing(root) as archive:
        window = session_evidence.read_session_evidence_window(
            archive,
            "file-edits",
            ref=f"session:{_SESSION_ID}",
            limit=1,
            offset=0,
            continuation=None,
            max_bytes=10_000,
        )
    assert window is not None and window["row_fragment"]


async def test_oversized_web_construct_resumes_from_daemon_on_api(tmp_path: Path) -> None:
    """One >8 MiB web row crosses surfaces and byte budgets without restarting."""
    import json

    from polylogue.api import Polylogue

    large_fixture_bytes = 8 * 1024 * 1024
    from polylogue.operations.daemon_reads import execute_read_operation
    from polylogue.operations.session_evidence import SESSION_EVIDENCE_PAGE_READERS

    root = tmp_path / "archive"
    run_off_event_loop(lambda: _seed_fragment_evidence(root, "web-content", "x" * (large_fixture_bytes + 4096)))
    ref = f"session:{_SESSION_ID}"
    with ArchiveStore.open_existing(root) as store:
        expected, _ = SESSION_EVIDENCE_PAGE_READERS["web-content"](store, _SESSION_ID, 1, 0)
        result = execute_read_operation(
            "session.read",
            {"ref": ref, "kind": "web-content", "limit": 1},
            archive=store,
            serving_identity="test",
        )
    assert len(json.dumps(result).encode("utf-8")) <= large_fixture_bytes
    windows = [cast("dict[str, Any]", result["evidence_window"])]
    assert windows[0]["row_fragment"] is not None
    token = windows[0]["continuation"]
    archive = Polylogue(archive_root=root)
    try:
        while token is not None:
            window = await archive.read_session_evidence_window(
                ref,
                "web-content",
                continuation=token,
                max_bytes=512 * 1024,
            )
            assert window is not None
            assert len(json.dumps(window).encode("utf-8")) <= 512 * 1024
            following = window["continuation"]
            assert following is None or following != token
            windows.append(window)
            token = following
    finally:
        await archive.close()
    assert _reassemble(windows) == expected
