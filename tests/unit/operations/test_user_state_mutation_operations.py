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
    envelope = stack.client.operation_to_completion(operation, payload, archive_root=str(archive_root))
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


def test_combined_mark_retains_one_authoritative_batch(daemon_archive: tuple[DaemonOperationStack, Path]) -> None:
    """Tags, marks, and notes are accepted together and report actual executions."""
    stack, root = daemon_archive
    envelope = _run(
        stack,
        root,
        "mutation.session.mark",
        {
            "selection": {"params": {"query": ["id:ext-conv-mark"]}, "mode": "single"},
            "tags": ["reviewed"],
            "add_marks": ["star"],
            "note_text": "retained note",
        },
    )
    assert envelope["outcome"] == "completed", envelope
    result = envelope["result"]
    assert result["reference"]["operation_name"] == "mutation.session.mark"
    assert result["reference"]["part_count"] == 3
    assert result["session_count"] == 1
    assert result["affected_count"] == 3
    assert result["completed_chunks"] == 3
    assert result["parts_total"] == 3
    assert result["not_attempted_count"] == 0
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        assert archive.read_summary(_SESSION_ID).tags == ("reviewed",)
        assert [row["mark_type"] for row in archive.list_marks()] == ["star"]
        assert [row["note_text"] for row in archive.list_annotations()] == ["retained note"]


def test_mark_missing_later_target_refuses_before_first_effect(
    daemon_archive: tuple[DaemonOperationStack, Path],
) -> None:
    stack, root = daemon_archive
    envelope = _run(
        stack,
        root,
        "mutation.session.mark",
        {
            "session_ids": [_SESSION_ID, "codex:missing-target"],
            "tags": ["must-not-commit"],
            "add_marks": ["star"],
        },
    )
    assert envelope["outcome"] in {"failed", "rejected"}, envelope
    assert envelope["error"]["code"] == "ValueError", envelope
    assert "codex:missing-target" in envelope["error"]["detail"], envelope
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        assert archive.read_summary(_SESSION_ID).tags == ()
        assert list(archive.list_marks()) == []


def test_first_mark_uses_bounded_resident_query(
    daemon_archive: tuple[DaemonOperationStack, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations import daemon_reads

    stack, root = daemon_archive
    original = daemon_reads.execute_read_operation
    limits: list[object] = []

    def observe(name: str, payload: dict[str, object], **kwargs: Any) -> dict[str, object]:
        if name == "cli.query":
            params = payload["params"]
            assert isinstance(params, dict)
            limits.append(params["limit"])
        return original(name, payload, **kwargs)

    monkeypatch.setattr(daemon_reads, "execute_read_operation", observe)
    envelope = _run(
        stack,
        root,
        "mutation.session.mark",
        {
            "selection": {"params": {"list_mode": True}, "mode": "first"},
            "add_marks": ["pin"],
        },
    )
    assert envelope["outcome"] == "completed", envelope
    assert limits == [1]
    assert [row["mark_type"] for row in _marks(root)] == ["pin"]


@pytest.mark.parametrize("operation", ["mutation.session.mark", "mutation.session.delete.preview"])
def test_selection_user_overlay_change_refuses_before_acceptance(
    daemon_archive: tuple[DaemonOperationStack, Path], monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from polylogue.operations import daemon_mutations

    stack, root = daemon_archive
    original = daemon_mutations._prepare_mutation_selection
    reached = False

    def select(*args: Any, **kwargs: Any) -> Any:
        nonlocal reached
        prepared = original(*args, **kwargs)

        def change_user_overlay() -> None:
            with ArchiveStore.open_existing(root, read_only=False) as archive:
                archive.add_user_tags((_SESSION_ID,), ("intervening-overlay",))

        stack.write_bridge.run_sync("test.selection.intervening-overlay", change_user_overlay)
        reached = True
        return prepared

    monkeypatch.setattr(daemon_mutations, "_prepare_mutation_selection", select)
    payload: dict[str, object] = {"selection": {"params": {"query": ["id:ext-conv-mark"]}, "mode": "single"}}
    if operation == "mutation.session.mark":
        payload["add_marks"] = ["star"]
    envelope = _run(stack, root, operation, payload)
    assert reached
    assert envelope["outcome"] == "failed", envelope
    assert envelope["error"]["code"] == "selection_frame_changed", envelope
    assert envelope["accepted_reference"] is None
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        assert archive.read_summary(_SESSION_ID).tags == ("intervening-overlay",)
        assert list(archive.list_marks()) == []
        assert archive.resolve_exact_session_ids((_SESSION_ID,)) == {_SESSION_ID: _SESSION_ID}


def test_mark_lifecycle_pages_preserve_whole_batch_outcome(tmp_path: Path) -> None:
    root = (tmp_path / "archive").resolve()

    def seed(archive_root: Path) -> None:
        initialize_active_archive_root(archive_root)
        for ordinal in range(45):
            (
                SessionBuilder(archive_root / "index.db", f"paged-{ordinal}")
                .provider("claude-ai")
                .add_message("m0", role="user", text="paged mark")
                .save()
            )

    with running_daemon_operations(root, seed_archive=seed) as stack:
        completed = _run(
            stack,
            root,
            "mutation.session.mark",
            {"selection": {"params": {}, "mode": "all"}, "add_marks": ["star"]},
        )
        assert completed["outcome"] == "completed", completed
        result = completed["result"]
        assert result["parts_total"] == 45
        assert len(result["parts"]) == 40
        assert result["next_parts_offset"] == 40
        assert result["affected_count"] == 45
        assert result["not_attempted_count"] == 0
        for operation in ("operation.status", "operation.await"):
            page = _run(
                stack,
                root,
                operation,
                {"request_id": result["reference"]["request_id"], "parts_offset": 40, "parts_limit": 40},
            )
            assert page["outcome"] == "completed", page
            tail = page["result"]
            assert tail["parts_total"] == 45
            assert len(tail["parts"]) == 5
            assert tail["next_parts_offset"] is None
            assert tail["affected_count"] == 45
            assert tail["not_attempted_count"] == 0
        assert len(_marks(root)) == 45
