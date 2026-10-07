"""Live intake publishes the retained-replay interpretation of the same bytes.

Each test drives the production live batch owner against a synthetic Codex
session and inspects the stored archive rows.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
from polylogue.sources.parsers.hermes_identity import profile_key, qualified_session_id
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.blob_store import BlobStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.raw_owner_routes import (
    converge_pending_raws_with_owner,
    ingest_files_with_owners,
    inspect_raw_observations,
    run_ingest_files,
)


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


def _ingest(archive_root: Path, source: Path, *, provider: Provider = Provider.CODEX) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    bootstrap_archive_root(archive_root)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        # The production Hermes source watches its JSON session snapshots too.
        (WatchSource(name=provider.value, root=source.parent, layout=export_drop_layout((".json", ".jsonl"))),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )
    metrics = run_ingest_files(processor, [source], emit_event=False)
    assert metrics.failed_file_count == 0
    assert metrics.succeeded_file_count == 1


def _titles(archive_root: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return [(str(row[0]), row[1]) for row in conn.execute("SELECT title, title_source FROM sessions")]


def test_live_hermes_profile_receipts_replay_without_mutable_alias_or_original_files(tmp_path: Path) -> None:
    from polylogue.core.raw_failure_evidence import MissingProfileIdentityError
    from polylogue.sources.revision_backfill import _parse_one

    # A Hermes snapshot is recognized by its session signature fields.
    document = json.dumps(
        {"session_id": "shared", "platform": "cli", "messages": [{"role": "user", "content": "same input"}]}
    )
    first = tmp_path / "profile-a"
    second = tmp_path / "profile-b"
    for root in (first, second):
        (root / "sessions").mkdir(parents=True)
        (root / "sessions" / "session_shared.json").write_text(document)
    alias = tmp_path / "profile"
    alias.symlink_to(first, target_is_directory=True)
    source = alias / "sessions" / "session_shared.json"
    archive_root = tmp_path / "archive"
    _ingest(archive_root, source, provider=Provider.HERMES)
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    _ingest(archive_root, source, provider=Provider.HERMES)
    _ingest(archive_root, source, provider=Provider.HERMES)
    with sqlite3.connect(archive_root / "source.db") as connection:
        rows = connection.execute(
            "SELECT r.raw_id, r.blob_hash, r.source_path, r.canonical_source_path, p.profile_key "
            "FROM raw_sessions AS r JOIN raw_profile_identity_receipts AS p USING(raw_id) "
            "ORDER BY p.profile_key"
        ).fetchall()
    assert len(rows) == 2
    assert len({row[0] for row in rows}) == 2
    assert len({row[1] for row in rows}) == 1
    assert {row[4] for row in rows} == {profile_key(first), profile_key(second)}
    assert {row[3] for row in rows} == {
        str(first / "sessions" / "session_shared.json"),
        str(second / "sessions" / "session_shared.json"),
    }
    alias.unlink()
    for root in (first, second):
        (root / "sessions" / "session_shared.json").unlink()
    store = BlobStore(archive_root / "blob")
    replayed: set[str] = set()
    for _raw_id, blob_hash, coordinate, _physical, key in rows:
        payload = store.read_all(bytes(blob_hash).hex())
        replayed.update(
            session.provider_session_id
            for session in _parse_one(
                Provider.HERMES,
                payload,
                str(coordinate),
                profile_identity=str(key),
                archive_root=archive_root,
                sidecar_resolver=None,
            )
        )
        with pytest.raises(MissingProfileIdentityError):
            _parse_one(Provider.HERMES, payload, str(coordinate), archive_root=archive_root, sidecar_resolver=None)
    assert replayed == {
        qualified_session_id("shared", profile_key(first)),
        qualified_session_id("shared", profile_key(second)),
    }


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
    )

    def ingest(paths: list[Path]) -> None:
        bootstrap_archive_root(archive_root)
        processor = LiveBatchProcessor(
            Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
            (watch,),
            cursor=CursorStore(archive_root / "index.db"),
            parser_fingerprint=_PARSER_FINGERPRINT,
        )
        metrics = run_ingest_files(processor, paths, emit_event=False)
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


def _claude_ingest(archive_root: Path, project: Path, paths: list[Path]) -> None:
    asyncio.run(_claude_ingest_async(archive_root, project, paths))


async def _claude_ingest_async(archive_root: Path, project: Path, paths: list[Path]) -> None:

    archive_root.mkdir(parents=True, exist_ok=True)
    bootstrap_archive_root(archive_root)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (
            WatchSource(
                name="claude-code",
                root=project.parent,
            ),
        ),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )
    metrics = await ingest_files_with_owners(processor, paths, emit_event=False)
    assert metrics.failed_file_count == 0


def _claude_tool_result_project(root: Path) -> tuple[Path, Path, Path]:
    """Build one Claude Code transcript whose overflow sidecar may arrive later."""
    session_id = "eeeeeeee-1111-2222-3333-444444444440"
    project = root / ".claude" / "projects" / "-synthetic-project"
    transcript = project / f"{session_id}.jsonl"
    sidecar = project / session_id / "tool-results" / "toolu_synthetic.txt"
    pointer = (
        "<persisted-output>\n"
        f"Output too large. Full output saved to: {sidecar}\n\n"
        "Preview (first 2KB):\nshort synthetic preview\n"
        "</persisted-output>"
    )
    records = (
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": session_id,
            "timestamp": "2026-07-20T10:00:00Z",
            "message": {"role": "user", "content": "run it"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": session_id,
            "timestamp": "2026-07-20T10:00:01Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_synthetic", "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "sessionId": session_id,
            "timestamp": "2026-07-20T10:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_synthetic", "content": pointer}],
            },
        },
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return project, transcript, sidecar


def _write_claude_tool_result_sidecar(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((f"{marker} synthetic persisted output\n") * 400, encoding="utf-8")


def _tool_result_texts(archive_root: Path) -> list[str]:
    with sqlite3.connect(archive_root / "index.db") as connection:
        return [str(row[0]) for row in connection.execute("SELECT text FROM blocks WHERE block_type = 'tool_result'")]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_sidecar", [None, "older"])
async def test_live_claude_tool_sidecar_replays_its_owner_from_retained_bytes(
    tmp_path: Path, initial_sidecar: str | None
) -> None:
    """A later sidecar revision replaces the owner's preview or older output.

    The second watcher batch offers only the raw-only sidecar. The test removes
    its source file after acquisition and before retained replay, so success
    requires owner selection and parsing to use the retained Source rows and
    bytes held by the production writer route.
    """
    from tests.infra.raw_owner_routes import live_owner_set

    project, transcript, sidecar = _claude_tool_result_project(tmp_path / "live")
    archive_root = tmp_path / "archive"
    if initial_sidecar is not None:
        _write_claude_tool_result_sidecar(sidecar, initial_sidecar)
        await _claude_ingest_async(archive_root, project, [transcript, sidecar])
        assert _tool_result_texts(archive_root)[0].startswith(f"{initial_sidecar} synthetic")
    else:
        await _claude_ingest_async(archive_root, project, [transcript])
        assert "short synthetic preview" in _tool_result_texts(archive_root)[0]

    before = _session_rows(archive_root)
    _write_claude_tool_result_sidecar(sidecar, "newer")
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="claude-code", root=project.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )

    async def ingest_after_sidecar_capture() -> None:
        async with live_owner_set(archive_root) as owners:
            processor._sqlite_capture_stage = owners.stage
            processor._sync_runner = owners.coordinator.run_sync
            processor._append_runner = owners.raw_owner.ingest_append_plans
            processor._convergence_runner = owners.raw_owner.run_convergence_sync
            original_replay = owners.raw_owner.ingest_retained_raw_ids

            async def remove_source_then_replay(raw_ids: Sequence[str], **kwargs: Any) -> Any:
                sidecar.unlink()
                return await original_replay(raw_ids, **kwargs)

            processor._retained_runner = remove_source_then_replay
            metrics = await processor.ingest_files([sidecar], emit_event=False)
            assert metrics.failed_file_count == 0

    await ingest_after_sidecar_capture()

    assert not sidecar.exists(), "replay should have consumed the captured Source bytes"
    [text] = _tool_result_texts(archive_root)
    assert text.startswith("newer synthetic persisted output")
    after = _session_rows(archive_root)
    assert len(after) == 1, "a tool-results sidecar must not create another session"
    assert after[0][0] == before[0][0]
    assert after[0][2] != before[0][2], "the owner transcript must be republished with the new sidecar"


def _session_rows(archive_root: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return [tuple(row) for row in conn.execute("SELECT session_id, title, content_hash FROM sessions")]


def test_preparation_reenriches_when_a_sidecar_lands_in_the_same_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A carrier enriched before the pass admitted its sidecar is not published.

    Retained preparation may enrich before the writer admits the
    ``sessions-index.json`` arriving in the same pass. The writer recomputes
    the evidence digest against what it has admitted and re-enriches.

    Anti-vacuity: drop the write-time ``prepared_enrichment_dependency_state``
    check, or the evidence-first ordering of the source before it is split into
    progress groups (``_enrichment_evidence_first``; the transcript is offered
    first here, one file per group), and
    the transcript-first pass stores the heuristic ``"prompt 0"`` title with a
    different content hash than the index-first pass.
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
    # Discovery order: the UUID-named transcript sorts before the index.
    _claude_ingest(staged_root, project, [transcript, index_path])
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


def _session_usage(archive_root: Path) -> tuple[tuple[object, ...], list[tuple[object, ...]]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        header = conn.execute("SELECT reported_cost_usd, reported_duration_ms FROM sessions").fetchone()
        models = [
            tuple(row)
            for row in conn.execute(
                "SELECT model_name, provider_cost_usd, declared FROM session_model_usage ORDER BY model_name"
            )
        ]
    return tuple(header), models


def test_live_append_keeps_the_chain_cost_across_a_model_switch(tmp_path: Path) -> None:
    """An appended turn adds to the session's reported cost; it does not replace it.

    A $1 turn on model A is ingested, then a $2 turn on model B is appended.
    The whole-file ingest of the same bytes is the reference.

    Anti-vacuity: drop the chain header carry in ``apply_raw_revision_replay``'s
    tail write and the append stores the tail's own $2 and 200 ms, and model
    A's provider cost share is cleared.
    """
    session_id = "dddddddd-1111-2222-3333-444444444440"

    def record(kind: str, **fields: object) -> bytes:
        return json.dumps({"type": kind, "sessionId": session_id, **fields}).encode() + b"\n"

    def assistant_turn(uuid: str, parent: str, *, model: str, cost: float, duration: int, at: str) -> bytes:
        return record(
            "assistant",
            uuid=uuid,
            parentUuid=parent,
            timestamp=at,
            costUSD=cost,
            durationMs=duration,
            message={"role": "assistant", "model": model, "content": [{"type": "text", "text": f"answer {uuid}"}]},
        )

    first = record(
        "user",
        uuid="u0",
        timestamp="2026-07-20T10:00:00.000Z",
        message={"role": "user", "content": "opening prompt"},
    ) + assistant_turn("a0", "u0", model="synthetic-model-a", cost=1.0, duration=100, at="2026-07-20T10:01:00.000Z")
    tail = record(
        "user",
        uuid="u1",
        parentUuid="a0",
        timestamp="2026-07-20T10:05:00.000Z",
        message={"role": "user", "content": "later prompt"},
    ) + assistant_turn("a1", "u1", model="synthetic-model-b", cost=2.0, duration=200, at="2026-07-20T10:06:00.000Z")

    project = tmp_path / "live" / ".claude" / "projects" / "-cost-project"
    project.mkdir(parents=True)
    transcript = project / f"{session_id}.jsonl"
    transcript.write_bytes(first)
    appended_root = tmp_path / "appended"
    _claude_ingest(appended_root, project, [transcript])
    transcript.write_bytes(first + tail)
    _claude_ingest(appended_root, project, [transcript])

    whole_root = tmp_path / "whole"
    _claude_ingest(whole_root, project, [transcript])

    header, models = _session_usage(appended_root)
    assert header == (pytest.approx(3.0), 300)
    assert [(name, declared) for name, _cost, declared in models] == [
        ("synthetic-model-a", 1),
        ("synthetic-model-b", 1),
    ]
    costs = [cost for _name, cost, _declared in models]
    assert all(isinstance(cost, float) for cost in costs), "every declared model carries a cost share"
    assert sum(cast(float, cost) for cost in costs) == pytest.approx(3.0)
    assert (header, models) == _session_usage(whole_root)


def test_stage_shutdown_retains_shared_compute_owner(tmp_path: Path) -> None:
    """The live capture stage borrows the resident kernel; closing it keeps the kernel open."""
    from polylogue.core.compute import compute_adapter
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage

    adapter = compute_adapter()
    stage = LiveSQLiteCaptureStage(compute_adapter=adapter)
    stage.shutdown()
    assert adapter.submit(lambda: "still-open").future.result(timeout=2) == "still-open"


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

    Anti-vacuity: drop the provider resolution in
    ``iter_enriched_sessions_from_retained_read`` and the assembly is looked up
    for ``UNKNOWN``, which has no spec.
    """
    from types import SimpleNamespace
    from typing import Any

    import polylogue.sources.revision_backfill as revision_backfill
    from polylogue.sources import assembly
    from polylogue.sources.parsers.base import ParsedSession

    seen: list[Provider] = []

    def recording_spec(provider: Provider) -> None:
        seen.append(provider)

    monkeypatch.setattr(assembly, "get_assembly_spec", recording_spec)
    session = ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="resolved", messages=[])
    revision_backfill.enrich_sessions_from_retained_read(
        cast(Any, SimpleNamespace()), Provider.UNKNOWN, "/nonexistent/x.jsonl", [session], captured_zip_coordinate=None
    )
    assert seen == [Provider.CLAUDE_CODE]


def _converge_to_fixpoint(archive_root: Path) -> None:
    """Run the canonical raw-observation convergence until nothing is pending."""
    from polylogue.daemon.derivation import Outcome

    report = converge_pending_raws_with_owner(archive_root, limit=64, passes=3)
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
    _converge_to_fixpoint(reference_root)
    reference = _session_rows(reference_root)
    assert [row[1] for row in reference] == ["Curated 0"]

    monkeypatch.setattr("polylogue.sources.live.batch._enrichment_evidence_first", lambda paths, _provider: paths)
    archive_root = tmp_path / "permuted"
    for group in groups:
        _claude_ingest(archive_root, project, [by_name[name] for name in group])
    _converge_to_fixpoint(archive_root)
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
    _converge_to_fixpoint(archive_root)
    before = _enrichment_bindings(archive_root)
    assert [row[1] for row in _session_rows(archive_root)] == ["Curated 0"]

    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["entries"][0]["summary"] = "Renamed later"
    index_path.write_text(json.dumps(document), encoding="utf-8")
    _claude_ingest(archive_root, project, [index_path])
    _converge_to_fixpoint(archive_root)

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

        def terminal_decode_refusals(self, _keys: object) -> dict[str, object]:
            # No dependent is a terminal decode refusal in these paging laws.
            return {}

    pages = []
    while page := discovery._dependents_selected(None, _Adapter(), 2):
        pages.append(page)
    assert pages == [("t0", "t1"), ("t2", "t3")]


def test_a_revised_session_index_restarts_its_queued_project_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One scan per project, restarted when the project's index arrives again.

    Anti-vacuity: appending a second scan serves the project twice (and the
    queue grows with every index revision); keeping the first scan's cursor
    skips transcripts it served against the older index.
    """
    from polylogue.operations import intake_adapters

    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT, source_path TEXT)")
        conn.execute("INSERT INTO raw_sessions VALUES ('index', '/p/proj/sessions-index.json')")
        conn.executemany("INSERT INTO raw_sessions VALUES (?, ?)", [(f"t{n}", f"/p/proj/s{n}.jsonl") for n in range(3)])
    monkeypatch.setattr(
        "polylogue.storage.sqlite.connection_profile.open_readonly_connection",
        lambda path, **_kwargs: sqlite3.connect(path),
    )

    class _Adapter:
        def inspect(self, _frame: object, keys: tuple[str, ...]) -> dict[str, str]:
            return dict.fromkeys(keys, "stale")

        def terminal_decode_refusals(self, _keys: object) -> dict[str, object]:
            # No dependent is a terminal decode refusal in these paging laws.
            return {}

    discovery = intake_adapters.RawMaterializationDiscovery(tmp_path)
    discovery._queue_evidence_dependents(["index"])
    assert discovery._dependents_selected(None, _Adapter(), 2) == ("t0", "t1")
    discovery._queue_evidence_dependents(["index"])
    assert len(discovery._evidence_projects) == 1

    pages = []
    while page := discovery._dependents_selected(None, _Adapter(), 2):
        pages.append(page)
    assert pages == [("t0", "t1"), ("t2",)]


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
    monkeypatch.setattr(discovery, "_with_costs", lambda selected: tuple((key, 1) for key in selected))
    monkeypatch.setattr(discovery, "_arrival_selected", lambda *_args: ("arrival",))
    monkeypatch.setattr(discovery, "_sweep_selected", lambda *_args: ("sweep",))
    monkeypatch.setattr(discovery, "_dependents_selected", lambda *_args: ("dependent",))
    discovery._evidence_projects.append(("/p/proj", "", -1))

    served = [discovery.discover_pending_raw_ids(8) for _ in range(6)]

    assert served.count((("dependent", 1),)) == 2
    assert served.count((("arrival", 1),)) == 2
    assert served.count((("sweep", 1),)) == 2


def _evidence_db(path: Path, value: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS evidence (value TEXT)")
        conn.execute("DELETE FROM evidence")
        conn.execute("INSERT INTO evidence VALUES (?)", (value,))
        conn.commit()


def test_enrichment_frames_rebind_and_refuse_moved_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A preparation outliving a read frame gets a fresh frame, never a moved view.

    Anti-vacuity (Codex P1, #5682): hold one frame across the whole parse and
    a long preparation outlives its bound; rebind without comparing the
    enriched evidence and the sealed digest certifies evidence earlier
    sessions never read.
    """
    from polylogue.sources.revision_backfill import EnrichmentEvidenceMovedError, _RebindingEvidenceFrames
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    source, index = tmp_path / "source.db", tmp_path / "index.db"
    _evidence_db(source, "a")
    monkeypatch.setattr("polylogue.sources.revision_backfill._EVIDENCE_FRAME_REBIND_S", 0.0)
    frames = _RebindingEvidenceFrames(source_db_path=source, index_db_path=index)

    def digest() -> str:
        conn = frames.connections[ArchiveTier.SOURCE]
        assert conn is not None
        return str(conn.execute("SELECT value FROM evidence").fetchone()[0])

    try:
        first = frames.current(None)[ArchiveTier.SOURCE]
        rebound = frames.current(digest)[ArchiveTier.SOURCE]
        assert rebound is not first
        _evidence_db(source, "b")
        with pytest.raises(EnrichmentEvidenceMovedError):
            frames.current(digest)
    finally:
        frames.close()


def test_parsed_sidecars_of_another_origin_are_released(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enriching a Codex file releases the parsed Claude Code sidecars first.

    Anti-vacuity (Codex P1, #5643): keep one entry per kind for every origin
    and a worker holds both origins' indexes and histories at once.
    """
    from types import SimpleNamespace
    from typing import Any, cast

    from polylogue.sources import retained_assembly

    retained_assembly._parsed_retained_cache.clear()
    monkeypatch.setattr(retained_assembly, "_read", lambda _source_read, _artifact: b"payload")
    source_read = cast(Any, SimpleNamespace())
    store = cast(Any, SimpleNamespace(root="/archive/blob"))
    for kind, digest in (("claude_code.session_index", "a"), ("claude_code.history_paste_index", "b")):
        retained_assembly._read_parsed(
            source_read, store, cast(Any, SimpleNamespace(blob_hash=digest, source_path=kind)), kind, len
        )
    retained_assembly._read_parsed(
        source_read, store, cast(Any, SimpleNamespace(blob_hash="c", source_path="codex")), "codex.session_index", len
    )

    assert set(retained_assembly._parsed_retained_cache) == {"codex.session_index"}
    retained_assembly._parsed_retained_cache.clear()


def test_a_superseded_sibling_from_another_directory_reads_the_accepted_evidence(tmp_path: Path) -> None:
    """Inspection compares enrichment evidence from the accepted head's raw.

    Two observations of one Claude session sit in two project directories
    whose ``sessions-index.json`` files title it differently. The binding is
    written from the accepted head; the superseded sibling must be compared
    against that head's evidence, not its own directory's.

    Anti-vacuity (Codex P1, #5643): read evidence from the inspected raw's own
    ``source_path`` and the sibling is ``stale`` forever, so convergence never
    settles and this inspection is not all ``valid``.
    """

    projects = tmp_path / "live" / ".claude" / "projects"
    session_id = "bbbbbbbb-1111-2222-3333-444444444441"

    def record(index: int) -> dict[str, object]:
        return {
            "type": "user",
            "uuid": f"u{index}",
            "sessionId": session_id,
            "timestamp": f"2026-07-20T10:00:0{index}.000Z",
            "message": {"role": "user", "content": f"prompt {index}"},
        }

    def project(name: str, summary: str, count: int) -> tuple[Path, Path, Path]:
        directory = projects / name
        directory.mkdir(parents=True)
        transcript = directory / f"{session_id}.jsonl"
        transcript.write_text("".join(json.dumps(record(index)) + "\n" for index in range(count)), encoding="utf-8")
        index_path = directory / "sessions-index.json"
        index_path.write_text(
            json.dumps({"entries": [{"sessionId": session_id, "fullPath": str(transcript), "summary": summary}]}),
            encoding="utf-8",
        )
        return directory, transcript, index_path

    first, first_transcript, first_index = project("-first", "Title first", 1)
    second, second_transcript, second_index = project("-second", "Title second", 2)
    archive_root = tmp_path / "archive"
    _claude_ingest(archive_root, first, [first_index, first_transcript])
    _claude_ingest(archive_root, second, [second_index, second_transcript])
    _converge_to_fixpoint(archive_root)

    with sqlite3.connect(archive_root / "source.db") as conn:
        raws = [
            str(row[0])
            for row in conn.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path LIKE ?", (f"%{session_id}.jsonl",)
            )
        ]
    assert len(raws) == 2
    statuses = inspect_raw_observations(archive_root, raws)
    assert set(statuses.values()) == {"valid"}
