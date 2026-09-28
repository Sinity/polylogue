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


def test_parse_stage_reenriches_when_a_sidecar_lands_in_the_same_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A carrier enriched before the pass admitted its sidecar is not published.

    The worker enriches during warm-up, before the writer admits the
    ``sessions-index.json`` arriving in the same pass. The writer recomputes
    the evidence digest against what it has admitted and re-enriches.

    Anti-vacuity: drop the write-time ``prepared_enrichment_dependency_state``
    check, or the evidence-first ordering of the source before it is split into
    progress groups (``_enrichment_evidence_first``; the transcript is offered
    first here, one file per group), and
    the parse-stage route stores the heuristic ``"prompt 0"`` title with a
    different content hash than the route without the stage.
    """
    project, transcript, index_path = _claude_project(tmp_path / "live")

    plain_root = tmp_path / "plain"
    _claude_ingest(plain_root, project, [index_path, transcript])
    plain = _session_rows(plain_root)
    assert [row[1] for row in plain] == ["Curated 0"]

    staged_root = tmp_path / "staged"
    # One file per progress group: the index must still lead, so evidence is
    # ordered across the whole source before the list is split.
    monkeypatch.setattr("polylogue.sources.live.batch_support._FULL_PARSE_PROGRESS_MAX_FILES", 1)
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


def test_unpublishable_retained_carrier_falls_back_to_the_writer_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A carrier the writer cannot publish is a prewarm miss, not a refusal.

    During a cold build the carrier may be enriched against the active index
    while the writer publishes into the candidate. Anti-vacuity: restore the
    ``PreparedSessionWriteRefusedError`` raise in ``_parse_raw_revision_chain``
    and this chain defers instead of replaying the member inline.
    """
    from types import SimpleNamespace

    from polylogue.sources.parsers.base import ParsedSession

    processor = LiveBatchProcessor(
        Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db"),
        (),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )
    replayed = ParsedSession(source_name=Provider.CODEX, provider_session_id="inline", messages=[])
    monkeypatch.setattr(processor, "_parse_retained_raw_sessions", lambda _archive, _raw_id: [replayed])
    stale = SimpleNamespace(current=lambda _archive: False)
    parsed = processor._parse_raw_revision_chain(
        SimpleNamespace(),
        SimpleNamespace(accepted_raw_ids=("older",)),
        retained_preparations_by_raw_id={"older": stale},  # type: ignore[dict-item]
    )
    assert parsed == {"older": replayed}


def _converge_to_fixpoint(archive_root: Path, root: Path) -> None:
    """Run the canonical raw-observation convergence until nothing is pending."""
    from polylogue.daemon.derivation import Outcome
    from polylogue.operations.raw_observation_derivation import converge_raw_observations

    report = None
    for _attempt in range(3):
        report = converge_raw_observations(archive_root, source_roots=(root,), limit=64)
    assert report is not None
    unsettled = [outcome for outcome in report.outcomes if outcome.outcome is not Outcome.DONE]
    assert not unsettled, unsettled


def _enrichment_bindings(archive_root: Path) -> dict[str, str]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return {str(row[0]): str(row[1]) for row in conn.execute("SELECT * FROM session_enrichment_bindings")}


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param((("index", "transcript"),), id="index-first-one-batch"),
        pytest.param((("transcript", "index"),), id="transcript-first-one-batch"),
        pytest.param((("transcript",), ("index",)), id="transcript-batch-then-index-batch"),
        pytest.param((("index",), ("transcript",)), id="index-batch-then-transcript-batch"),
    ],
)
def test_claude_code_rows_do_not_depend_on_evidence_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, groups: tuple[tuple[str, ...], ...]
) -> None:
    """The same bytes, in any admission order, converge to the same session row.

    The reference ingests the index before the transcript. The ordering hint
    (``_enrichment_evidence_first``) is disabled here, so a transcript written
    before its index is enriched without it; only the evidence binding makes
    the retained route re-derive it once the index is admitted.

    Anti-vacuity: make ``RawObservationDerivation._enrichment_evidence_moved``
    return ``False`` and the transcript-before-index orders keep the heuristic
    ``"prompt 0"`` title and a different content hash.
    """
    project, transcript, index_path = _claude_project(tmp_path / "live")
    by_name = {"index": index_path, "transcript": transcript}

    reference_root = tmp_path / "reference"
    _claude_ingest(reference_root, project, [index_path, transcript])
    _converge_to_fixpoint(reference_root, project.parent)
    reference = _session_rows(reference_root)
    assert [row[1] for row in reference] == ["Curated 0"]

    monkeypatch.setattr("polylogue.sources.live.batch._enrichment_evidence_first", lambda paths, _provider: paths)
    archive_root = tmp_path / "permuted"
    for group in groups:
        _claude_ingest(archive_root, project, [by_name[name] for name in group])
    _converge_to_fixpoint(archive_root, project.parent)
    assert _session_rows(archive_root) == reference
    assert set(_enrichment_bindings(archive_root)) == {str(row[0]) for row in reference}


def test_a_later_index_revision_re_derives_the_titled_session(tmp_path: Path) -> None:
    """An index rewritten after its transcript re-derives that session.

    The binding recorded with the first index no longer matches the retained
    evidence once the renamed index is admitted, so inspection re-derives the
    session through the retained route.

    Anti-vacuity: make ``_enrichment_evidence_moved`` return ``False`` and the
    stored title stays ``"Curated 0"`` after the rename.
    """
    project, transcript, index_path = _claude_project(tmp_path / "live")
    archive_root = tmp_path / "archive"
    _claude_ingest(archive_root, project, [index_path, transcript])
    _converge_to_fixpoint(archive_root, project.parent)
    before = _enrichment_bindings(archive_root)
    assert [row[1] for row in _session_rows(archive_root)] == ["Curated 0"]

    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["entries"][0]["summary"] = "Renamed later"
    index_path.write_text(json.dumps(document), encoding="utf-8")
    _claude_ingest(archive_root, project, [index_path])
    _converge_to_fixpoint(archive_root, project.parent)

    assert [row[1] for row in _session_rows(archive_root)] == ["Renamed later"]
    after = _enrichment_bindings(archive_root)
    assert set(after) == set(before)
    assert after != before


def test_session_index_dependents_are_paged_not_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An arrived session index queues a paged scan of its project.

    Anti-vacuity: listing the project's transcripts when the index arrives
    (the eager scan) never serves a transcript retained after that moment,
    and serves the whole history in one list regardless of the page bound.
    """
    from polylogue.operations import intake_adapters

    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT, source_path TEXT)")
        conn.execute("INSERT INTO raw_sessions VALUES ('index', '/p/proj/sessions-index.json')")
        conn.executemany(
            "INSERT INTO raw_sessions VALUES (?, ?)",
            [(f"t{n}", f"/p/proj/s{n}.jsonl") for n in range(3)] + [("other", "/p/proj2/s0.jsonl")],
        )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.connection_profile.open_readonly_connection",
        lambda path, **_kwargs: sqlite3.connect(path),
    )
    discovery = intake_adapters.RawMaterializationDiscovery(tmp_path)
    discovery._queue_evidence_dependents(["index"])
    with sqlite3.connect(source_db) as conn:
        conn.execute("INSERT INTO raw_sessions VALUES ('t3', '/p/proj/s3.jsonl')")

    class _Adapter:
        def inspect(self, _frame: object, keys: tuple[str, ...]) -> dict[str, str]:
            return dict.fromkeys(keys, "stale")

    pages = []
    while page := discovery._dependents_selected(None, _Adapter(), 2):
        pages.append(page)
    assert pages == [("t0", "t1"), ("t2", "t3")]


def test_session_index_dependents_rotate_with_the_fair_lanes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A project scan that owes work on every page never monopolizes discovery.

    Anti-vacuity (Codex P2, #5643): serve the dependent lane with strict
    priority and every call returns dependents while arrivals and the sweep,
    which also owe work, are never offered.
    """
    from types import SimpleNamespace

    from polylogue.operations import intake_adapters, raw_observation_derivation

    (tmp_path / "source.db").touch()
    frame = SimpleNamespace(archive_root=tmp_path, source_revision="r1", recipe_version=lambda _domain: "v1")
    monkeypatch.setattr(raw_observation_derivation, "raw_observation_frame", lambda _root: frame)
    monkeypatch.setattr(raw_observation_derivation, "make_raw_observation_derivation", lambda _root: object())
    discovery = intake_adapters.RawMaterializationDiscovery(tmp_path)
    monkeypatch.setattr(discovery, "_raw_frontier", lambda: 0)
    monkeypatch.setattr(discovery, "_with_costs", lambda selected: selected)
    monkeypatch.setattr(discovery, "_arrival_selected", lambda *_args: ("arrival",))
    monkeypatch.setattr(discovery, "_sweep_selected", lambda *_args: ("sweep",))
    monkeypatch.setattr(discovery, "_dependents_selected", lambda *_args: ("dependent",))
    discovery._evidence_projects.append(("/p/proj", "", -1))

    served = [discovery.discover_pending_raw_ids(8) for _ in range(6)]

    assert served.count(("dependent",)) == 2
    assert served.count(("arrival",)) == 2
    assert served.count(("sweep",)) == 2
