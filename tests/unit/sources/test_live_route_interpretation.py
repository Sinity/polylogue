"""Live intake publishes the retained-replay interpretation of the same bytes.

Each test drives the production live batch owner against a synthetic Codex
session and inspects the stored archive rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.parse_prefetch import LiveParseStage
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _codex_lines(native_id: str, messages: tuple[tuple[str, str], ...], *, meta: bool = True) -> bytes:
    rows: list[dict[str, object]] = (
        [{"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}] if meta else []
    )
    for message_id, text in messages:
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": message_id,
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _ingest(archive_root: Path, source: Path, *, parse_stage: LiveParseStage | None = None) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=source.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=parse_stage,
        read_snapshot=open_operation_read,
    )
    metrics = asyncio.run(processor.ingest_files([source], emit_event=False))
    assert metrics.failed_file_count == 0
    assert metrics.succeeded_file_count == 1


def _titles(archive_root: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return [(str(row[0]), row[1]) for row in conn.execute("SELECT title, title_source FROM sessions")]


def test_live_append_keeps_the_chain_title_winner(tmp_path: Path) -> None:
    """A tail append must not replace the title a full replay would store.

    Anti-vacuity: dropping the aggregate title carry in the tail-merge write
    (``apply_raw_revision_replay``) stores the appended chunk's own first
    prompt, ``"later prompt"``; dropping live enrichment stores the native id.
    """
    source = tmp_path / "sessions" / "chain.jsonl"
    source.parent.mkdir()
    first = _codex_lines("chain-session", (("m0", "opening prompt"),))
    source.write_bytes(first)
    archive_root = tmp_path / "archive"
    _ingest(archive_root, source)
    assert _titles(archive_root) == [("opening prompt", "heuristic")]

    source.write_bytes(first + _codex_lines("chain-session", (("m1", "later prompt"),), meta=False))
    _ingest(archive_root, source)

    assert _titles(archive_root) == [("opening prompt", "heuristic")]
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_prewarm_seals_existing_retained_members_outside_the_writer(tmp_path: Path) -> None:
    """A pending live path carries sealed carriers for its prior raws.

    Anti-vacuity: without ``prepare_live_retained_raws`` the stage hands the
    writer no retained member, and the writer must reparse the prior raw
    under its lease. The carrier is also bound to the raw's descriptor, so a
    changed descriptor stops ``current`` from accepting it.
    """
    source = tmp_path / "sessions" / "revised.jsonl"
    source.parent.mkdir()
    source.write_bytes(_codex_lines("revised-session", (("m0", "first draft"),)))
    archive_root = tmp_path / "archive"
    _ingest(archive_root, source)
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        [prior_raw_id] = [str(row[0]) for row in archive.source_connection.execute("SELECT raw_id FROM raw_sessions")]

    source.write_bytes(_codex_lines("revised-session", (("m0", "second draft"),)))
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        stage.warm_paths(
            [(str(source), Provider.CODEX, True)], archive_root=archive_root, read_snapshot=open_operation_read
        )
        prepared = stage.pop_path(str(source), blob_hash=hashlib.sha256(source.read_bytes()).hexdigest())
        assert prepared is not None and prepared.error is None and not prepared.deferred
        [current] = list(prepared.iter_sessions())
        assert current.title == "second draft"
        retained = stage.take_retained_path(str(source))
        assert set(retained) == {prior_raw_id}
        member = retained[prior_raw_id]
        with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
            assert member.current(archive)
        [prior] = list(member.artifact.session_sequence())
        assert prior.title == "first draft"
        member.discard()
        prepared.discard()
    finally:
        stage.shutdown()


def test_live_claude_code_intake_uses_retained_index_titles_parsed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live transcripts take the curated retained index title, as replay does.

    Anti-vacuity: without worker enrichment both sessions keep the heuristic
    first-prompt title; without the content-addressed parse memo the shared
    index is reparsed for every transcript (``parses`` grows with files).
    """
    import polylogue.sources.parsers.claude.index as claude_index
    import polylogue.sources.retained_assembly as retained_assembly
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    project = tmp_path / "live" / ".claude" / "projects" / "-synthetic-project"
    project.mkdir(parents=True)
    transcripts: list[Path] = []
    entries: list[dict[str, object]] = []
    for index in range(2):
        session_id = f"aaaaaaaa-1111-2222-3333-44444444444{index}"
        transcript = project / f"{session_id}.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": f"u{index}",
                    "sessionId": session_id,
                    "timestamp": "2026-07-20T10:00:00.000Z",
                    "message": {"role": "user", "content": f"prompt {index}"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        transcripts.append(transcript)
        entries.append({"sessionId": session_id, "fullPath": str(transcript), "summary": f"Curated {index}"})
    index_path = project / "sessions-index.json"
    index_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    parses = 0
    original_parse = claude_index.parse_sessions_index_bytes

    def counting_parse(payload: bytes) -> object:
        nonlocal parses
        parses += 1
        return original_parse(payload)

    monkeypatch.setattr(claude_index, "parse_sessions_index_bytes", counting_parse)
    retained_assembly._parsed_retained_cache.clear()
    watch = WatchSource(
        name="claude-code",
        root=project.parent,
        suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
    )

    def ingest(paths: list[Path]) -> None:
        processor = LiveBatchProcessor(
            Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
            (watch,),
            cursor=CursorStore(archive_root / "index.db"),
            parser_fingerprint=_PARSER_FINGERPRINT,
            read_snapshot=open_operation_read,
        )
        metrics = asyncio.run(processor.ingest_files(paths, emit_event=False))
        assert metrics.failed_file_count == 0

    ingest([index_path])
    ingest(transcripts)

    assert sorted(_titles(archive_root)) == [("Curated 0", "origin"), ("Curated 1", "origin")]
    assert parses == 1


def _claude_project(root: Path) -> tuple[Path, Path, Path]:
    project = root / ".claude" / "projects" / "-synthetic-project"
    project.mkdir(parents=True)
    session_id = "bbbbbbbb-1111-2222-3333-444444444440"
    transcript = project / f"{session_id}.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "u0",
                "sessionId": session_id,
                "timestamp": "2026-07-20T10:00:00.000Z",
                "message": {"role": "user", "content": "prompt 0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = project / "sessions-index.json"
    index_path.write_text(
        json.dumps({"entries": [{"sessionId": session_id, "fullPath": str(transcript), "summary": "Curated 0"}]}),
        encoding="utf-8",
    )
    return project, transcript, index_path


def _claude_ingest(
    archive_root: Path, project: Path, paths: list[Path], *, parse_stage: LiveParseStage | None = None
) -> None:
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    archive_root.mkdir(parents=True, exist_ok=True)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (
            WatchSource(
                name="claude-code",
                root=project.parent,
                suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
            ),
        ),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=parse_stage,
        read_snapshot=open_operation_read,
    )
    metrics = asyncio.run(processor.ingest_files(paths, emit_event=False))
    assert metrics.failed_file_count == 0


def _session_rows(archive_root: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return [tuple(row) for row in conn.execute("SELECT session_id, title, content_hash FROM sessions")]


def test_parse_stage_reenriches_when_a_sidecar_lands_in_the_same_pass(tmp_path: Path) -> None:
    """A carrier enriched before the pass admitted its sidecar is not published.

    The worker enriches during warm-up, before the writer admits the
    ``sessions-index.json`` arriving in the same pass. The writer recomputes
    the evidence digest against what it has admitted and re-enriches.

    Anti-vacuity: drop the write-time ``prepared_enrichment_dependency_state``
    check, or the evidence-first ordering of the pass
    (``_enrichment_evidence_first``, the transcript is offered first here), and
    the parse-stage route stores the heuristic ``"prompt 0"`` title with a
    different content hash than the route without the stage.
    """
    project, transcript, index_path = _claude_project(tmp_path / "live")

    plain_root = tmp_path / "plain"
    _claude_ingest(plain_root, project, [index_path, transcript])
    plain = _session_rows(plain_root)
    assert [row[1] for row in plain] == ["Curated 0"]

    staged_root = tmp_path / "staged"
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        # Discovery order: the UUID-named transcript sorts before the index.
        _claude_ingest(staged_root, project, [transcript, index_path], parse_stage=stage)
    finally:
        stage.shutdown()
    assert _session_rows(staged_root) == plain


def test_live_append_takes_the_latest_rename(tmp_path: Path) -> None:
    """A later rename in an appended tail wins, as the whole-file parse does.

    Anti-vacuity: break title-evidence ties toward the earlier chunk in
    ``merge_parsed_session_chunks`` and the stored title stays ``"Name A"``,
    disagreeing with a whole-file ingest of the same bytes.
    """
    session_id = "cccccccc-1111-2222-3333-444444444440"

    def record(kind: str, **fields: object) -> bytes:
        return json.dumps({"type": kind, "sessionId": session_id, **fields}).encode() + b"\n"

    first = record(
        "user",
        uuid="u0",
        timestamp="2026-07-20T10:00:00.000Z",
        message={"role": "user", "content": "opening prompt"},
    ) + record("custom-title", customTitle="Name A")
    tail = record(
        "user",
        uuid="u1",
        parentUuid="u0",
        timestamp="2026-07-20T10:05:00.000Z",
        message={"role": "user", "content": "later prompt"},
    ) + record("custom-title", customTitle="Name B")

    project = tmp_path / "live" / ".claude" / "projects" / "-rename-project"
    project.mkdir(parents=True)
    transcript = project / f"{session_id}.jsonl"
    transcript.write_bytes(first)
    appended_root = tmp_path / "appended"
    _claude_ingest(appended_root, project, [transcript])
    transcript.write_bytes(first + tail)
    _claude_ingest(appended_root, project, [transcript])

    whole_root = tmp_path / "whole"
    _claude_ingest(whole_root, project, [transcript])

    assert [row[1] for row in _session_rows(appended_root)] == ["Name B"]
    assert _session_rows(appended_root) == _session_rows(whole_root)


def test_broken_pool_restart_after_shutdown_creates_no_new_pool(tmp_path: Path) -> None:
    """A pool broken during shutdown is not replaced by a fresh one.

    Anti-vacuity: drop the ``_closing`` guard in
    ``_restart_broken_process_pool`` and a new executor replaces the stopped
    one, able to seal carriers after cleanup.
    """
    from concurrent.futures import ProcessPoolExecutor

    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards", use_processes=True)
    try:
        executor = stage._executor
        assert isinstance(executor, ProcessPoolExecutor)
        stage._closing = True
        stage._restart_broken_process_pool()
        assert stage._executor is executor
    finally:
        stage.shutdown()


def test_live_append_keeps_origin_provenance_for_an_equal_heuristic_title(tmp_path: Path) -> None:
    """Equal title text from weaker evidence does not replace provenance.

    The prefix names the session by rename; the appended tail's first prompt
    happens to read the same. Anti-vacuity: compare only the title text in
    the tail-merge carry (``apply_raw_revision_replay``) and the stored row
    takes the tail's heuristic ``title_source`` under the prefix's hash.
    """
    session_id = "dddddddd-1111-2222-3333-444444444440"

    def record(kind: str, **fields: object) -> bytes:
        return json.dumps({"type": kind, "sessionId": session_id, **fields}).encode() + b"\n"

    first = record(
        "user",
        uuid="u0",
        timestamp="2026-07-20T10:00:00.000Z",
        message={"role": "user", "content": "opening prompt"},
    ) + record("custom-title", customTitle="Shared name")
    tail = record(
        "user",
        uuid="u1",
        parentUuid="u0",
        timestamp="2026-07-20T10:05:00.000Z",
        message={"role": "user", "content": "Shared name"},
    )
    project = tmp_path / "live" / ".claude" / "projects" / "-provenance-project"
    project.mkdir(parents=True)
    transcript = project / f"{session_id}.jsonl"
    transcript.write_bytes(first)
    appended_root = tmp_path / "appended"
    _claude_ingest(appended_root, project, [transcript])
    transcript.write_bytes(first + tail)
    _claude_ingest(appended_root, project, [transcript])
    whole_root = tmp_path / "whole"
    _claude_ingest(whole_root, project, [transcript])

    def provenance(root: Path) -> list[tuple[object, ...]]:
        with sqlite3.connect(root / "index.db") as conn:
            return [tuple(row) for row in conn.execute("SELECT title, title_source, content_hash FROM sessions")]

    assert [row[:2] for row in provenance(appended_root)] == [("Shared name", "origin")]
    assert provenance(appended_root) == provenance(whole_root)


def test_writer_enrichment_resolves_an_unknown_acquisition_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``UNKNOWN`` raw enriches with its parsed provider's assembly.

    Anti-vacuity: drop the resolution in ``enrich_sessions_from_archive`` and
    the enricher is built for ``UNKNOWN``, which has no assembly spec.
    """
    from types import SimpleNamespace

    import polylogue.sources.revision_backfill as revision_backfill
    from polylogue.sources.parsers.base import ParsedSession

    seen: list[Provider] = []

    class RecordingEnricher:
        def __init__(self, provider: Provider, **_kwargs: object) -> None:
            seen.append(provider)

        def enrich_all(self, sessions: list[ParsedSession]) -> list[ParsedSession]:
            return sessions

    monkeypatch.setattr(revision_backfill, "RetainedSessionEnricher", RecordingEnricher)
    session = ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="resolved", messages=[])
    archive = SimpleNamespace(archive_root=Path("/nonexistent"), index_connection=None, source_connection=None)
    revision_backfill.enrich_sessions_from_archive(archive, Provider.UNKNOWN, "/nonexistent/x.jsonl", [session])
    assert seen == [Provider.CLAUDE_CODE]


def test_retained_prewarm_spends_one_deadline_across_members(tmp_path: Path) -> None:
    """A member that outlives the budget stops the prewarm; nothing waits again.

    Anti-vacuity: restore a per-member timeout and every later member is
    submitted and waited on in turn (``submitted`` grows to three).
    """
    from concurrent.futures import Future
    from types import SimpleNamespace

    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources.live.retained_prefetch import prepare_live_retained_raws

    blob = tmp_path / "blob"
    descriptors = {
        f"raw-{index}": (Provider.CODEX, f"{index:064x}", f"/src/{index}.jsonl", RawRevisionKind.FULL, 1)
        for index in range(3)
    }
    archive = SimpleNamespace(
        archive_root=tmp_path,
        source_db_path=tmp_path / "source.db",
        index_db_path=tmp_path / "index.db",
        raw_membership_raw_ids=lambda _key: set(descriptors),
        raw_membership_retired_full_revision_siblings=lambda _key: set(),
        convertible_full_revision_raw_ids=lambda _key: set(),
        raw_revision_head_raw_id=lambda _key: None,
        raw_revision_replay_plan=lambda _key: SimpleNamespace(accepted_raw_ids=()),
        raw_revision_descriptor=descriptors.__getitem__,
        raw_revision_file_mtime=lambda _raw_id: None,
    )
    blob.mkdir()
    submitted: list[str] = []

    class StalledExecutor:
        def submit(self, _fn: object, raw_id: str, *_args: object) -> Future[object]:
            submitted.append(raw_id)
            future: Future[object] = Future()
            future.set_running_or_notify_cancel()
            return future

    prepared = prepare_live_retained_raws(
        archive,
        logical_keys={"codex-session:x"},
        current_raw_id="current",
        directory=tmp_path / "retained",
        worker_executor=StalledExecutor(),  # type: ignore[arg-type]
        member_timeout_s=0.05,
    )
    assert prepared == {}
    assert submitted == ["raw-0"]
