"""Source-tier evidence for a cursor page costs one read, not one per path.

polylogue-s8x8s AC3 ("cursor commit ... opens no per-path connection").
``_source_tier_evidence_retained`` asks ``source.db`` two questions about
every ``tool-results/`` sidecar a pass committed. Each answer used to open its
own read-only connection inside the cursor-commit loop, so a cold build paid
two file opens, two schema loads and two single-row queries per sidecar.

These are cost assertions. The refusal semantics they must not disturb are
already covered end to end by ``test_live_sidecar_cursor_evidence.py``; what
is asserted here is how much work producing the same answers takes, which no
behavioural test can distinguish.

Anti-vacuity: drop the ``_pinned_source_tier_evidence`` wrapper from the
cursor-commit loop and ``test_cursor_page_opens_no_per_path_source_read`` goes
red with one per-path opener call per sidecar. Make the pin answer from
something other than the same rows and the equivalence test goes red.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore

_SESSION_ID = "6f0a1c2d-4e5b-4a7c-9d1e-2b3c4d5e6f70"


def _owner_lines(sidecar: Path) -> list[dict[str, object]]:
    pointer = f"<persisted-output>Output too large. Full output saved to: {sidecar}</persisted-output>"
    return [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": _SESSION_ID,
            "timestamp": "2026-07-20T10:00:00Z",
            "message": {"role": "user", "content": "run it"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": _SESSION_ID,
            "timestamp": "2026-07-20T10:00:01Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "sessionId": _SESSION_ID,
            "timestamp": "2026-07-20T10:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": pointer}],
            },
        },
    ]


def _build_tree(root: Path, *, sidecar_count: int) -> tuple[Path, list[Path]]:
    project = root / "-realm-project-cost"
    sidecar_dir = project / _SESSION_ID / "tool-results"
    sidecar_dir.mkdir(parents=True)
    sidecars = []
    for index in range(sidecar_count):
        sidecar = sidecar_dir / f"tool-{index:04d}.txt"
        sidecar.write_text(f"persisted tool output {index}\n" * 8, encoding="utf-8")
        sidecars.append(sidecar)
    owner = project / f"{_SESSION_ID}.jsonl"
    owner.write_text("\n".join(json.dumps(line) for line in _owner_lines(sidecars[0])) + "\n", encoding="utf-8")
    return owner, sidecars


def _make_processor(workspace_env: dict[str, Path], root: Path) -> tuple[Polylogue, CursorStore, LiveBatchProcessor]:
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "index.db")
    cursor = CursorStore(workspace_env["data_root"] / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root, suffixes=(".jsonl", ".txt")),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    return archive, cursor, processor


class _OpenerCounts:
    def __init__(self) -> None:
        self.latest_raw_fingerprint = 0
        self.history_sidecar = 0

    @property
    def total(self) -> int:
        return self.latest_raw_fingerprint + self.history_sidecar


def _count_per_path_openers(processor: LiveBatchProcessor, monkeypatch: pytest.MonkeyPatch) -> _OpenerCounts:
    """Count the two methods that open a fresh read-only ``source.db`` connection."""
    counts = _OpenerCounts()
    raw_original = processor._latest_archive_tiers_raw_fingerprint
    history_original = processor._history_sidecar_retained

    def counted_raw(path: Path) -> str | None:
        counts.latest_raw_fingerprint += 1
        return raw_original(path)

    def counted_history(path: Path) -> bool:
        counts.history_sidecar += 1
        return history_original(path)

    monkeypatch.setattr(processor, "_latest_archive_tiers_raw_fingerprint", counted_raw)
    monkeypatch.setattr(processor, "_history_sidecar_retained", counted_history)
    return counts


@pytest.mark.asyncio
async def test_cursor_page_opens_no_per_path_source_read(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page of sidecars resolves its evidence without a per-path connection."""
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    owner, sidecars = _build_tree(root, sidecar_count=16)
    archive, cursor, processor = _make_processor(workspace_env, root)
    try:
        counts = _count_per_path_openers(processor, monkeypatch)

        await processor.ingest_files([*sidecars, owner], emit_event=False)

        assert counts.total == 0, (
            f"cursor commit opened {counts.latest_raw_fingerprint} raw-fingerprint and "
            f"{counts.history_sidecar} history-sidecar connections for {len(sidecars)} sidecars"
        )
        # The cost saving is only a saving if the answers still advanced the
        # cursors: an evidence check that silently refuses is not cheaper.
        for sidecar in sidecars:
            record = cursor.get_record(sidecar)
            assert record is not None, sidecar
            assert record.byte_offset == sidecar.stat().st_size
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_pinned_evidence_agrees_with_the_per_path_read(
    workspace_env: dict[str, Path],
) -> None:
    """The pinned answer is the answer the single-path route gives.

    Both a retained sidecar and one source.db never received are checked, so
    a pin that answered ``True`` for everything -- the cheap way to make the
    cost test green -- is refuted here.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    owner, sidecars = _build_tree(root, sidecar_count=3)
    retained = sidecars[:2]
    unretained = sidecars[2]
    archive, _cursor, processor = _make_processor(workspace_env, root)
    try:
        await processor.ingest_files([*retained, owner], emit_event=False)

        unpinned = {
            str(path): (
                processor._latest_archive_tiers_raw_fingerprint(path),
                processor._history_sidecar_retained(path),
            )
            for path in (*retained, unretained)
        }
        with processor._pinned_source_tier_evidence([*retained, unretained]):
            pinned = {
                str(path): (processor._latest_raw_fingerprint(path), processor._history_sidecar_retained(path))
                for path in (*retained, unretained)
            }

        assert pinned == unpinned
        assert unpinned[str(unretained)] == (None, False)
        assert unpinned[str(retained[0])][0] is not None
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_unpinned_callers_still_read_source_db_themselves(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside the pinned page nothing is memoized: every call reads again.

    A pin that outlived its pass would answer later calls from rows written
    before it was taken. Leaving the page must restore the per-path read.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    owner, sidecars = _build_tree(root, sidecar_count=2)
    archive, _cursor, processor = _make_processor(workspace_env, root)
    try:
        await processor.ingest_files([*sidecars, owner], emit_event=False)
        assert processor._pinned_raw_fingerprints is None
        assert processor._pinned_history_sidecars is None

        counts = _count_per_path_openers(processor, monkeypatch)
        for sidecar in sidecars:
            processor._source_tier_evidence_retained(sidecar, raw_fingerprint="carried-raw-id")

        assert counts.latest_raw_fingerprint == len(sidecars)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_unreadable_source_tier_refuses_rather_than_answering(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read failure must never resolve to "no evidence".

    The un-pinned route raises too: ``_latest_archive_tiers_raw_fingerprint``
    resolves a broken read to ``None`` and ``_source_tier_evidence_retained``
    then asks ``_history_sidecar_retained``, whose connection is unguarded.
    The pin keeps that boundary rather than adding a softer one beside it.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    owner, sidecars = _build_tree(root, sidecar_count=2)
    archive, _cursor, processor = _make_processor(workspace_env, root)
    try:
        await processor.ingest_files([*sidecars, owner], emit_event=False)

        def refuse(*args: object, **kwargs: object) -> sqlite3.Connection:
            raise sqlite3.OperationalError("source tier unavailable")

        monkeypatch.setattr(sqlite3, "connect", refuse)
        with pytest.raises(sqlite3.OperationalError, match="source tier unavailable"):
            with processor._pinned_source_tier_evidence(sidecars):
                pass  # pragma: no cover - the pin raises before the body runs
        monkeypatch.undo()

        # Same refusal, same input, through the per-path route.
        monkeypatch.setattr(sqlite3, "connect", refuse)
        with pytest.raises(sqlite3.OperationalError, match="source tier unavailable"):
            processor._source_tier_evidence_retained(sidecars[0], raw_fingerprint="carried-raw-id")
    finally:
        monkeypatch.undo()
        await archive.close()


@pytest.mark.asyncio
async def test_a_source_tier_without_raw_sessions_pins_no_evidence(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An absent table is a fact read from the catalog, not a swallowed error."""
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    _owner, sidecars = _build_tree(root, sidecar_count=2)
    archive, _cursor, processor = _make_processor(workspace_env, root)
    try:
        empty_root = tmp_path / "empty-archive"
        empty_root.mkdir()
        sqlite3.connect(empty_root / "source.db").close()
        processor._archive_source_db_path = lambda: empty_root / "source.db"  # type: ignore[method-assign]

        with processor._pinned_source_tier_evidence(sidecars):
            pinned = processor._pinned_raw_fingerprints
            assert pinned == {str(path): None for path in sidecars}
            assert processor._pinned_history_sidecars == {str(path): False for path in sidecars}
    finally:
        await archive.close()
