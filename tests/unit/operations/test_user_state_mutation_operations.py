"""The new user-state write operations really write, under daemon authority.

``mutation.session.mark`` and ``mutation.annotation.save`` replaced a facade
route that opened a writable ``ArchiveStore`` in the calling process. A
declaration and a handler that no one has driven end to end would be exactly the
failure the design warns about -- "a command with a declared operation it does
not use" -- so these run the real ``DaemonOperationRuntime`` over its real
socket and then read the durable rows back.

Anti-vacuity: make a handler return a receipt without calling its actuator, and
the read-back assertions go red; drop the durable-owner fallback from
``_resolve_session_target`` and the alias case goes red; let
``mutation.session.mark`` accept a message target it cannot resolve and the
idempotence assertions stop meaning what they say.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations
from tests.infra.storage_records import SessionBuilder

_SESSION_ID = "claude-ai-export:ext-conv-mark"


def _seed(archive_root: Path) -> None:
    initialize_active_archive_root(archive_root)
    (
        SessionBuilder(archive_root / "index.db", "conv-mark")
        .provider("claude-ai")
        .title("Markable session")
        .add_message("m0", role="user", text="hello alpha")
        .save()
    )


@pytest.fixture
def daemon_archive(tmp_path: Path) -> Iterator[tuple[DaemonOperationStack, Path]]:
    archive_root = (tmp_path / "archive").resolve()
    with running_daemon_operations(archive_root, seed_archive=_seed) as stack:
        yield stack, archive_root


def _run(stack: DaemonOperationStack, archive_root: Path, operation: str, payload: dict[str, object]) -> dict[str, Any]:
    """Send one operation and require an envelope back."""
    envelope = stack.client.operation(operation, payload, archive_root=str(archive_root))
    assert envelope is not None, f"{operation} returned no envelope"
    return envelope


def _marks(archive_root: Path) -> list[dict[str, str]]:
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        return list(archive.list_marks())


def test_session_mark_adds_and_removes_a_durable_mark(daemon_archive: tuple[DaemonOperationStack, Path]) -> None:
    """A star added through the operation is a real row, and removing it clears it."""
    stack, archive_root = daemon_archive

    added = _run(stack, archive_root, "mutation.session.mark", {"session_ids": [_SESSION_ID], "add_marks": ["star"]})

    assert added["outcome"] == "completed", added.get("error")
    assert [row["mark_type"] for row in _marks(archive_root)] == ["star"]

    removed = _run(
        stack, archive_root, "mutation.session.mark", {"session_ids": [_SESSION_ID], "remove_marks": ["star"]}
    )

    assert removed["outcome"] == "completed", removed
    assert _marks(archive_root) == []


def test_session_mark_is_idempotent(daemon_archive: tuple[DaemonOperationStack, Path]) -> None:
    """Re-adding an existing mark reports no effect rather than forking a row."""
    stack, archive_root = daemon_archive
    payload: dict[str, object] = {"session_ids": [_SESSION_ID], "add_marks": ["pin"]}

    _run(stack, archive_root, "mutation.session.mark", payload)
    repeated = _run(stack, archive_root, "mutation.session.mark", payload)

    assert repeated["result"]["effect"] == "no-effect", repeated
    assert len(_marks(archive_root)) == 1


def test_annotation_save_creates_then_updates_one_row(daemon_archive: tuple[DaemonOperationStack, Path]) -> None:
    """The same annotation id updates in place; a note is mutable, not append-only."""
    stack, archive_root = daemon_archive

    for text in ("first", "second"):
        _run(
            stack,
            archive_root,
            "mutation.annotation.save",
            {"annotation_id": "note-1", "session_id": _SESSION_ID, "note_text": text},
        )

    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        annotations = list(archive.list_annotations())
        stored = archive.get_annotation("note-1")

    assert len(annotations) == 1, annotations
    assert stored is not None
    assert stored["note_text"] == "second"
