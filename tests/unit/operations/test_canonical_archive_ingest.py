from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import polylogue.sources.source_parsing as source_parsing
import polylogue.sources.source_root_admission as source_root_admission
from polylogue.config import Source
from polylogue.maintenance.source_conservation import SourceConservationReport, audit_source_conservation
from polylogue.operations.canonical_archive_ingest import _ingest_selected_paths, ingest_one_shot_archive
from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.antigravity import AntigravitySessionSummary, parse_markdown_export
from polylogue.sources.parsers.base import ParsedSession, RawSessionData
from polylogue.storage.blob_store import BlobStore


def _install_antigravity_export_stub(monkeypatch: pytest.MonkeyPatch, roots: list[Path]) -> None:
    def export(
        source: Source,
        *,
        capture_raw: bool,
        blob_root: Path | None,
        blob_store: BlobStore | None,
        only_cascade_ids: frozenset[str] | None = None,
        excised: set[Path] | None = None,
        admit_path: Callable[[Path], bool] | None = None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        assert source.path is not None
        roots.append(source.path)
        for cascade_id in sorted(only_cascade_ids or ()):
            pb_path = source.path / "conversations" / f"{cascade_id}.pb"
            if admit_path is not None and not admit_path(pb_path):
                continue
            session = _synthetic_session(cascade_id)
            raw = source_parsing._antigravity_raw_snapshot(
                pb_path,
                source_sha256=sha256(pb_path.read_bytes()).hexdigest(),
                capture_raw=capture_raw,
                blob_root=blob_root,
                blob_store=blob_store,
            )
            yield raw, session

    def replay_export(
        root: Path,
        *,
        client: object | None = None,
        only_cascade_ids: frozenset[str] | None = None,
    ) -> Iterable[ParsedSession]:
        # Retained replay re-parses through the parser module's own export seam;
        # it must reproduce the same synthetic conversion as live intake.
        for cascade_id in sorted(only_cascade_ids or ()):
            yield _synthetic_session(cascade_id)

    monkeypatch.setattr(source_parsing, "iter_antigravity_language_server_sessions", export)
    monkeypatch.setattr(antigravity, "iter_language_server_exports", replay_export)


def _synthetic_session(cascade_id: str) -> ParsedSession:
    return parse_markdown_export(
        f"### User Input\n\nQuestion from {cascade_id}.\n\n### Planner Response\n\nAnswer.\n",
        AntigravitySessionSummary(cascade_id=cascade_id),
    )


def _source_conservation(archive_root: Path) -> SourceConservationReport:
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("ATTACH DATABASE ? AS idx_tier", (str(archive_root / "index.db"),))
        return audit_source_conservation(conn, archive_root=archive_root)


@pytest.mark.asyncio
async def test_canonical_ingest_resolves_relative_antigravity_conversation_path(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "antigravity"
    conversations = root / "conversations"
    conversations.mkdir(parents=True)
    pb_path = conversations / "cascade.pb"
    pb_path.write_bytes(b"synthetic trajectory")
    monkeypatch.chdir(conversations)
    parser_roots: list[Path] = []
    _install_antigravity_export_stub(monkeypatch, parser_roots)

    archive_root = one_shot_workspace_env["archive_root"]
    result = await ingest_one_shot_archive(
        archive_root,
        [Source(name="antigravity", path=Path("cascade.pb"))],
        parse_workers=1,
    )

    assert result.counts.get("sessions", 0) == 1
    assert parser_roots == [root.resolve()]
    with sqlite3.connect(archive_root / "source.db") as conn:
        [source_path] = conn.execute("SELECT source_path FROM raw_sessions").fetchone()
    assert Path(source_path) == pb_path.resolve()
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_canonical_ingest_traverses_directory_named_pb_as_antigravity_root(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "export.pb"
    conversations = root / "conversations"
    conversations.mkdir(parents=True)
    (conversations / "cascade.pb").write_bytes(b"synthetic trajectory")
    parser_roots: list[Path] = []
    _install_antigravity_export_stub(monkeypatch, parser_roots)

    archive_root = one_shot_workspace_env["archive_root"]
    result = await ingest_one_shot_archive(
        archive_root,
        [Source(name="antigravity", path=root)],
        parse_workers=1,
    )

    assert result.counts.get("sessions", 0) == 1
    assert parser_roots == [root.resolve()]
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_canonical_ingest_keeps_individual_antigravity_pb_ownership_exact(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "captures"
    conversations = root / "conversations"
    conversations.mkdir(parents=True)
    pb_path = conversations / "cascade.pb"
    pb_path.write_bytes(b"synthetic trajectory")
    codex_path = root / "session.jsonl"
    codex_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "session_meta", "payload": {"id": "sibling-codex", "timestamp": "2026-01-01T00:00:00Z"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "message-1",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parser_roots: list[Path] = []
    _install_antigravity_export_stub(monkeypatch, parser_roots)

    archive_root = one_shot_workspace_env["archive_root"]
    result = await ingest_one_shot_archive(
        archive_root,
        [
            Source(name="antigravity", path=pb_path),
            Source(name="codex", path=codex_path),
        ],
        parse_workers=1,
    )

    assert result.counts.get("sessions", 0) == 2
    assert parser_roots == [root.resolve()]
    with sqlite3.connect(archive_root / "source.db") as conn:
        recorded_paths = {Path(path) for (path,) in conn.execute("SELECT source_path FROM raw_sessions")}
    assert recorded_paths == {pb_path.resolve(), codex_path.resolve()}
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_canonical_ingest_keeps_sibling_provider_file_ownership_exact(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
) -> None:
    root = tmp_path / "capture-files"
    root.mkdir()
    codex_path = root / "codex.jsonl"
    codex_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "session_meta", "payload": {"id": "owned-codex", "timestamp": "2026-01-01T00:00:00Z"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "message-1",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Codex input"}],
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    claude_path = root / "claude.jsonl"
    claude_path.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "claude-user",
                "sessionId": "owned-claude",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "Claude input"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    archive_root = one_shot_workspace_env["archive_root"]
    result = await ingest_one_shot_archive(
        archive_root,
        [Source(name="codex", path=codex_path), Source(name="claude-code", path=claude_path)],
        parse_workers=1,
    )

    assert result.counts.get("sessions", 0) == 2
    with sqlite3.connect(archive_root / "source.db") as conn:
        recorded_paths = {Path(path) for (path,) in conn.execute("SELECT source_path FROM raw_sessions")}
    assert recorded_paths == {codex_path.resolve(), claude_path.resolve()}
    with sqlite3.connect(archive_root / "index.db") as conn:
        origins = {origin for (origin,) in conn.execute("SELECT origin FROM sessions")}
    assert origins == {"codex-session", "claude-code-session"}
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_canonical_ingest_prefers_declared_file_over_equal_depth_directory_root(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
) -> None:
    root = tmp_path / "capture-root"
    root.mkdir()
    codex_path = root / "session.jsonl"
    codex_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "session_meta", "payload": {"id": "explicit-codex", "timestamp": "2026-01-01T00:00:00Z"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "message-1",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Codex input"}],
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    archive_root = one_shot_workspace_env["archive_root"]
    result = await ingest_one_shot_archive(
        archive_root,
        [Source(name="antigravity", path=root), Source(name="codex", path=codex_path)],
        parse_workers=1,
    )

    assert result.counts.get("sessions", 0) == 1
    assert result.processed_ids == {"codex-session:explicit-codex"}
    with sqlite3.connect(archive_root / "index.db") as conn:
        [origin] = conn.execute("SELECT origin FROM sessions").fetchone()
    assert origin == "codex-session"
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_canonical_ingest_records_a_resolvable_path_for_relative_session_jsonl(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "external-source"
    source_dir.mkdir()
    source_path = source_dir / "session.jsonl"
    source_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "session_meta", "payload": {"id": "relative-source", "timestamp": "2026-01-01T00:00:00Z"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "message-1",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(source_dir)
    archive_root = one_shot_workspace_env["archive_root"]

    result = await ingest_one_shot_archive(
        archive_root,
        [Source(name="codex", path=Path("session.jsonl"))],
        parse_workers=1,
    )

    assert result.counts.get("sessions", 0) == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        [recorded_path] = conn.execute("SELECT source_path FROM raw_sessions").fetchone()
    assert Path(recorded_path) == source_path.resolve()
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_canonical_ingest_traverses_the_source_root_that_passed_admission(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_root = tmp_path / "original"
    replacement_root = tmp_path / "replacement"
    for root, cascade_id in ((original_root, "admitted"), (replacement_root, "retargeted")):
        conversations = root / "conversations"
        conversations.mkdir(parents=True)
        (conversations / f"{cascade_id}.pb").write_bytes(cascade_id.encode())
    source_link = tmp_path / "capture"
    source_link.symlink_to(original_root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    parser_roots: list[Path] = []
    _install_antigravity_export_stub(monkeypatch, parser_roots)

    original_admission = source_root_admission.refuse_non_capture_source_root
    checked_roots: list[Path] = []

    def admit_then_retarget(path: Path, *, destination: Path | None = None) -> None:
        original_admission(path, destination=destination)
        checked_roots.append(path)
        source_link.unlink()
        source_link.symlink_to(replacement_root, target_is_directory=True)

    monkeypatch.setattr(source_root_admission, "refuse_non_capture_source_root", admit_then_retarget)
    archive_root = one_shot_workspace_env["archive_root"]
    result = await ingest_one_shot_archive(
        archive_root,
        [Source(name="antigravity", path=Path("capture"))],
        parse_workers=1,
    )

    assert checked_roots == [original_root.resolve()]
    assert parser_roots == [original_root.resolve()]
    assert result.processed_ids == {"antigravity-session:admitted"}
    with sqlite3.connect(archive_root / "source.db") as conn:
        [source_path] = conn.execute("SELECT source_path FROM raw_sessions").fetchone()
    assert Path(source_path) == (original_root / "conversations" / "admitted.pb").resolve()
    assert _source_conservation(archive_root).term("source_missing").count == 0


@pytest.mark.asyncio
async def test_one_shot_ingest_reoffers_unattempted_paths_until_64_files_settle() -> None:
    paths = [Path(f"/synthetic/codex-{index:02d}.jsonl") for index in range(64)]
    offered_by_pass: list[list[Path]] = []

    async def ingest_pass(offered: list[Path]) -> SimpleNamespace:
        offered_by_pass.append(offered)
        succeeded = offered[:14]
        excluded = offered[14:]
        return SimpleNamespace(
            failed_file_count=0,
            deferred_file_count=0,
            succeeded_file_count=len(succeeded),
            succeeded_paths=tuple(succeeded),
            excluded_file_count=len(excluded),
            excluded_paths={str(path): "unattempted_time_budget" for path in excluded},
        )

    receipts = await _ingest_selected_paths(paths, ingest_pass)

    assert [len(batch) for batch in offered_by_pass] == [64, 50, 36, 22, 8]
    assert len(receipts) == 5
    assert offered_by_pass == [paths, paths[14:], paths[28:], paths[42:], paths[56:]]
    assert {path for batch in offered_by_pass for path in batch} == set(paths)


@pytest.mark.asyncio
async def test_one_shot_ingest_refuses_non_retryable_exclusions() -> None:
    path = Path("/synthetic/unsupported.jsonl")

    async def ingest_pass(offered: list[Path]) -> SimpleNamespace:
        return SimpleNamespace(
            failed_file_count=0,
            deferred_file_count=0,
            succeeded_file_count=0,
            succeeded_paths=(),
            excluded_file_count=1,
            excluded_paths={str(offered[0]): "unsupported_source"},
        )

    with pytest.raises(RuntimeError, match="non-retryable reason"):
        await _ingest_selected_paths([path], ingest_pass)


def _write_codex_session(path: Path, session_id: str, texts: tuple[str, ...]) -> Path:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-01-01T00:00:00Z"}}
    ]
    for index, text in enumerate(texts):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{session_id}-message-{index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": [{"type": "input_text" if index % 2 == 0 else "output_text", "text": text}],
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _normalized_material(archive_root: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        return {
            "sessions": sorted(
                conn.execute("SELECT session_id, origin, content_hash, message_count, raw_id FROM sessions")
            ),
            "messages": sorted(
                conn.execute("SELECT message_id, session_id, role, material_origin, content_hash FROM messages")
            ),
        }


@pytest.mark.asyncio
async def test_from_empty_and_incremental_one_shot_routes_write_identical_material(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One session writer serves a from-empty build and incremental offers alike.

    ``ingest_one_shot_archive`` and the daemon both hand files to
    ``LiveBatchProcessor``; the from-empty build must not differ from the same
    files arriving one at a time, and re-offering already-ingested files must
    change nothing.

    Anti-vacuity: give the one-shot route its own writer (for example a
    position-derived message id, or a batch path that skips the content-hash
    dedup) and the two archives' normalized rows diverge, or the re-offer
    reports changed sessions.
    """
    from polylogue.storage.blob_store import reset_blob_store

    sources_dir = tmp_path / "capture-files"
    sources_dir.mkdir()
    first = Source(
        name="codex",
        path=_write_codex_session(sources_dir / "first.jsonl", "diff-first", ("question one", "answer one")),
    )
    second = Source(
        name="codex",
        path=_write_codex_session(
            sources_dir / "second.jsonl", "diff-second", ("question two", "answer two", "follow-up")
        ),
    )

    from_empty_root = one_shot_workspace_env["archive_root"]
    from_empty = await ingest_one_shot_archive(from_empty_root, [first, second], parse_workers=1)
    assert from_empty.counts.get("sessions", 0) == 2

    incremental_root = tmp_path_factory.mktemp("one-shot-incremental")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(incremental_root))
    reset_blob_store()
    await ingest_one_shot_archive(incremental_root, [first], parse_workers=1)
    await ingest_one_shot_archive(incremental_root, [second], parse_workers=1)
    reoffer = await ingest_one_shot_archive(incremental_root, [first, second], parse_workers=1)

    assert reoffer.changed_counts.get("sessions", 0) == 0
    assert reoffer.processed_ids == set()
    material = _normalized_material(incremental_root)
    assert len(material["sessions"]) == 2
    assert len(material["messages"]) == 5
    assert material == _normalized_material(from_empty_root)


@pytest.mark.asyncio
async def test_one_shot_ingest_settles_a_source_that_produced_no_sessions() -> None:
    """A no-session exclusion is settled, not a refusal to retry or raise on.

    Anti-vacuity: once batch metrics report no-session files as excluded
    instead of succeeded, treating that reason like any other exclusion made
    a one-shot import of a valid but empty transcript raise.
    """
    paths = [Path("/synthetic/empty.jsonl"), Path("/synthetic/full.jsonl")]
    passes: list[list[Path]] = []

    async def ingest_pass(offered: list[Path]) -> SimpleNamespace:
        passes.append(offered)
        return SimpleNamespace(
            failed_file_count=0,
            deferred_file_count=0,
            succeeded_file_count=1,
            succeeded_paths=(offered[1],),
            excluded_file_count=1,
            excluded_paths={str(offered[0]): "no_sessions"},
        )

    receipts = await _ingest_selected_paths(paths, ingest_pass)

    assert len(receipts) == 1
    assert passes == [paths]


@pytest.mark.asyncio
async def test_one_shot_teardown_settles_the_writer_before_stopping_the_capture_stage(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled caller must not invalidate an admitted writer's carrier.

    An admitted writer may still consume the capture stage's prepared SQLite
    captures; stopping the stage first can discard them while the writer still
    owns the archive. Anti-vacuity: restoring a ``sqlite_capture_stage.shutdown()``
    -> coordinator-idle order reverses ``events``.
    """
    import polylogue.operations.canonical_archive_ingest as canonical
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage

    source_path = tmp_path / "external-source" / "session.jsonl"
    source_path.parent.mkdir()
    source_path.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "teardown", "timestamp": "2026-01-01T00:00:00Z"}}) + "\n",
        encoding="utf-8",
    )
    events: list[str] = []
    original_idle = canonical._wait_for_coordinator_idle
    original_shutdown = LiveSQLiteCaptureStage.shutdown

    async def wait_idle(coordinator: object) -> None:
        await original_idle(coordinator)  # type: ignore[arg-type]
        events.append("writer_settled")

    def shutdown(self: LiveSQLiteCaptureStage) -> None:
        events.append("stage_shutdown")
        original_shutdown(self)

    async def cancelled_ingest(self: LiveBatchProcessor, paths: object, **_kwargs: object) -> object:
        raise RuntimeError("caller cancelled after writer admission")

    monkeypatch.setattr(canonical, "_wait_for_coordinator_idle", wait_idle)
    monkeypatch.setattr(LiveSQLiteCaptureStage, "shutdown", shutdown)
    monkeypatch.setattr(LiveBatchProcessor, "ingest_files", cancelled_ingest)

    with pytest.raises(RuntimeError, match="caller cancelled"):
        await ingest_one_shot_archive(
            one_shot_workspace_env["archive_root"],
            [Source(name="codex", path=source_path)],
            parse_workers=1,
        )

    assert events == ["writer_settled", "stage_shutdown"]


@pytest.mark.asyncio
async def test_one_shot_teardown_stops_the_capture_stage_when_the_settle_wait_fails(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settle wait that raises (a second cancellation) still stops the stage.

    Anti-vacuity: with the teardown steps in sequence rather than nested
    ``finally`` blocks, the raising wait skips ``shutdown`` and the archive
    close, leaking the worker pool; ``events`` then lacks both.
    """
    import polylogue.operations.canonical_archive_ingest as canonical
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage

    source_path = tmp_path / "external-source" / "session.jsonl"
    source_path.parent.mkdir()
    source_path.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "teardown", "timestamp": "2026-01-01T00:00:00Z"}}) + "\n",
        encoding="utf-8",
    )
    events: list[str] = []
    original_shutdown = LiveSQLiteCaptureStage.shutdown
    from polylogue import Polylogue

    original_close = Polylogue.close

    async def failing_wait(coordinator: object) -> None:
        raise RuntimeError("settle wait interrupted")

    def shutdown(self: LiveSQLiteCaptureStage) -> None:
        events.append("stage_shutdown")
        original_shutdown(self)

    async def close(self: object) -> None:
        events.append("archive_closed")
        await original_close(self)  # type: ignore[arg-type]

    async def cancelled_ingest(self: LiveBatchProcessor, paths: object, **_kwargs: object) -> object:
        raise RuntimeError("caller cancelled after writer admission")

    monkeypatch.setattr(canonical, "_wait_for_coordinator_idle", failing_wait)
    monkeypatch.setattr(LiveSQLiteCaptureStage, "shutdown", shutdown)
    monkeypatch.setattr(Polylogue, "close", close)
    monkeypatch.setattr(LiveBatchProcessor, "ingest_files", cancelled_ingest)

    with pytest.raises(RuntimeError, match="settle wait interrupted"):
        await ingest_one_shot_archive(
            one_shot_workspace_env["archive_root"],
            [Source(name="codex", path=source_path)],
            parse_workers=1,
        )

    assert events == ["stage_shutdown", "archive_closed"]
