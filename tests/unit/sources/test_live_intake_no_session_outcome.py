"""A source file that yields no session is excluded, not admitted.

A transcript whose records carry no conversational evidence is acquired and
parsed, and its raw row records the typed terminal outcome. The intake
outcome for that file must say so: reporting ``ADMITTED`` counted a file that
produced nothing among the operator's admitted files (polylogue-xf8qp).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.logging import capture
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.source_layout import export_drop_layout
from tests.infra.raw_owner_routes import live_owner_set

_MAX_DEFERRED_PAGES = 20


async def _admit(
    archive_root: Path,
    source_root: Path,
    metrics_sink: list[Any] | None = None,
    *,
    source_name: str = "claude-code",
    suffixes: tuple[str, ...] = (".jsonl",),
) -> dict[str, Any]:
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    async with live_owner_set(archive_root) as owners:
        watcher = LiveWatcher(
            archive,
            (WatchSource(name=source_name, root=source_root, layout=export_drop_layout(suffixes)),),
            cursor=CursorStore(archive_root / "index.db"),
            **owners.watcher_kwargs(),
        )
        if metrics_sink is not None:
            ingest_files = watcher._ingest_files

            async def recording_ingest(*args: Any, **kwargs: Any) -> Any:
                metrics = await ingest_files(*args, **kwargs)
                metrics_sink.append(metrics)
                return metrics

            watcher._ingest_files = recording_ingest  # type: ignore[method-assign]
        try:
            outcomes: dict[str, Any] = {}
            for _ in range(_MAX_DEFERRED_PAGES):
                adapter = FileIntakeAdapter(
                    DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=watcher._sources),
                    watcher._sources[0],
                )
                outcomes = dict(await adapter.admit_page(await adapter.discover(limit=8)))
                if {result.outcome for result in outcomes.values()} != {AdmissionOutcome.DEFERRED}:
                    return outcomes
            raise AssertionError(f"the source stayed deferred for {_MAX_DEFERRED_PAGES} pages: {outcomes}")
        finally:
            watcher.stop()
            await archive.close()


@pytest.mark.asyncio
async def test_a_file_without_sessions_is_excluded_not_admitted(workspace_env: dict[str, Path]) -> None:
    """Anti-vacuity: the pre-fix route put the no-session raw into the
    succeeded set, so the outcome was ``ADMITTED`` with no session written."""
    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    source_path = source_root / "silent.jsonl"
    # A well-formed Claude Code record with no conversational content: it is
    # acquired and parsed, and yields no session with positive evidence.
    source_path.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": ""},
                "uuid": "u1",
                "sessionId": "silent",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batches: list[Any] = []
    outcomes = await _admit(archive_root, source_root, batches)

    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.EXCLUDED}, outcomes
    # The batch receipt agrees with the intake outcome (Codex): it once
    # counted the same file as a success with its bytes ingested.
    metrics = batches[-1]
    assert metrics.succeeded_file_count == 0 and not metrics.succeeded_paths
    assert metrics.excluded_file_count == 1
    assert metrics.excluded_paths == {str(source_path): "no_sessions"}
    assert metrics.ingested_bytes == 0
    # The durable attempt row carries the same typed refusal, not SUCCESS.
    from polylogue.core.enums import IngestOutcome

    with sqlite3.connect(archive_root / "ops.db") as ops:
        (outcome_code,) = ops.execute("SELECT outcome_code FROM ingest_attempts ORDER BY rowid DESC LIMIT 1").fetchone()
    assert outcome_code == IngestOutcome.UNSUPPORTED_SHAPE.value
    (result,) = outcomes.values()
    assert result.reason is not None and result.reason.startswith("no_sessions:")
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


#: A complete record that does not decode, between two good ones.
_MALFORMED_MIDDLE_RECORD = (
    json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
            "uuid": "u1",
            "sessionId": "corrupt",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ).encode()
    + b"\n{not json}\n"
    + json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "corrupt",
            "timestamp": "2026-01-01T00:00:01Z",
        }
    ).encode()
    + b"\n"
)


@pytest.mark.asyncio
async def test_a_corrupt_capture_is_excluded_as_corrupt_input(workspace_env: dict[str, Path]) -> None:
    """A capture live intake settles as ``terminal_corrupt_input`` is excluded, not admitted.

    The truncated capture that is no longer growing reaches the same settled
    exclusion through the batch route
    (``test_captured_incomplete_jsonl_is_rejected_after_source_disappears``).

    Anti-vacuity: the pre-fix route recorded the terminal evidence on the raw
    but left the path in the succeeded set, so the intake outcome was
    ``ADMITTED`` and the batch counted the file's bytes as ingested
    (polylogue-xf8qp); and a Claude Code capture with no semantic frontier
    refused its cursor write, so the page was ``RETRYABLE`` on every pass.
    """
    from polylogue.core.enums import IngestOutcome

    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    source_path = source_root / "corrupt.jsonl"
    source_path.write_bytes(_MALFORMED_MIDDLE_RECORD)

    batches: list[Any] = []
    outcomes = await _admit(archive_root, source_root, batches)

    metrics = batches[-1]
    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.EXCLUDED}, outcomes
    (result,) = outcomes.values()
    assert result.reason is not None and result.reason.startswith("corrupt_input:"), result
    assert metrics.succeeded_file_count == 0 and not metrics.succeeded_paths
    assert metrics.excluded_paths == {str(source_path): "corrupt_input"}
    assert metrics.ingested_bytes == 0
    with sqlite3.connect(archive_root / "ops.db") as ops:
        (outcome_code,) = ops.execute("SELECT outcome_code FROM ingest_attempts ORDER BY rowid DESC LIMIT 1").fetchone()
    assert outcome_code == IngestOutcome.CORRUPT_INPUT.value
    with sqlite3.connect(archive_root / "source.db") as conn:
        kinds = {str(row[0]) for row in conn.execute("SELECT artifact_kind FROM raw_artifacts")}
    assert "terminal_corrupt_input" in kinds
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize(
    "payload",
    [b"{", b'{"title": "cut", "mapping": {"n": {"id": "n", "message": '],
    ids=["opening-brace", "truncated-mapping"],
)
@pytest.mark.asyncio
async def test_an_undecodable_json_document_is_excluded_as_corrupt_input(
    workspace_env: dict[str, Path], payload: bytes
) -> None:
    """A known-provider JSON document that does not decode settles as corrupt input.

    Anti-vacuity: only a JSONL record's decode failure was terminal
    (polylogue-6r7wv). A JSON document's left the raw parse-failed with no
    terminal carrier and the page ``RETRYABLE``, so every pass re-read bytes
    that can never decode.
    """
    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "chatgpt-exports"
    source_root.mkdir(parents=True)
    source_path = source_root / "conversation.json"
    source_path.write_bytes(payload)

    batches: list[Any] = []
    outcomes = await _admit(archive_root, source_root, batches, source_name="chatgpt", suffixes=(".json",))

    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.EXCLUDED}, outcomes
    (result,) = outcomes.values()
    assert result.reason is not None and result.reason.startswith("corrupt_input:"), result
    assert batches[-1].excluded_paths == {str(source_path): "corrupt_input"}
    with sqlite3.connect(archive_root / "source.db") as conn:
        artifacts = conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchall()
    assert ("terminal_corrupt_input", 0) in artifacts, artifacts
    assert batches[-1].succeeded_file_count == 0
    assert batches[-1].ingested_bytes == 0
    with sqlite3.connect(archive_root / "ops.db") as conn:
        assert conn.execute("SELECT outcome_code FROM ingest_attempts ORDER BY rowid DESC LIMIT 1").fetchone() == (
            "corrupt_input",
        )
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.prepared_replay import current_fixture_parser_receipts

    with sqlite3.connect(archive_root / "source.db") as conn:
        raw_ids = [str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions")]
    assert raw_ids, "live refusal did not retain its accepted raw"
    # The live pass settles the refusal through its census phase in one pass.
    assert run_off_event_loop(lambda: current_fixture_parser_receipts(archive_root, raw_ids)) == (True,)
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT status FROM raw_authority_parser_census").fetchall() == [("complete",)]


def _claude_record(uuid: str, parent: str | None, role: str, text: str) -> bytes:
    content: object = text if role == "user" else [{"type": "text", "text": text}]
    return (
        json.dumps(
            {
                "type": role,
                "message": {"role": role, "content": content},
                "uuid": uuid,
                "parentUuid": parent,
                "sessionId": "partial",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ).encode()
        + b"\n"
    )


@pytest.mark.asyncio
async def test_a_stable_truncated_capture_is_admitted_as_a_typed_partial(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stable capture whose final record is truncated admits its complete records, visibly in part.

    The intake result is ``ADMITTED`` (the complete records are admitted, as
    rejecting them would drop valid input) and carries the typed partial:
    reason ``truncated_tail``, the complete-record count and the byte offset
    where the left-out tail begins. The batch metrics and the dispatcher's
    class report count it.

    Anti-vacuity: before polylogue-xf8qp the same capture was a plain
    ``ADMITTED`` with no partial and nothing in the batch counted the tail.
    """
    from polylogue.core.raw_failure_evidence import PARTIAL_TRUNCATED_TAIL, PartialAdmission

    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    source_path = source_root / "partial.jsonl"
    complete = _claude_record("u1", None, "user", "question") + _claude_record("a1", "u1", "assistant", "answer")
    payload = complete + b'{"type":"user","message":{"role":"user","cont'
    source_path.write_bytes(payload)

    from polylogue.sources.live.batch import LiveBatchProcessor

    writer_entry_prefix_counts: list[tuple[int, ...]] = []
    original_writer = LiveBatchProcessor._acquire_full_records_archive

    def observe_prepared_prefix_count(self: Any, records: list[Any], *args: Any, **kwargs: Any) -> Any:
        writer_entry_prefix_counts.append(
            tuple(record.complete_prefix_record_count for record in records if record.complete_prefix_size is not None)
        )
        return original_writer(self, records, *args, **kwargs)

    monkeypatch.setattr(LiveBatchProcessor, "_acquire_full_records_archive", observe_prepared_prefix_count)

    batches: list[Any] = []
    with capture() as events:
        outcomes = await _admit(archive_root, source_root, batches)

    (result,) = outcomes.values()
    assert result.outcome is AdmissionOutcome.ADMITTED, result
    expected = PartialAdmission(
        reason=PARTIAL_TRUNCATED_TAIL,
        complete_record_count=2,
        complete_prefix_bytes=len(complete),
        source_bytes=len(payload),
    )
    assert result.partial == expected
    assert writer_entry_prefix_counts == [(2,)], "acquisition did not seal the count before archive writer entry"
    metrics = batches[-1]
    assert metrics.partial_admission_paths == {str(source_path): expected}
    payload_fields = metrics.to_payload()
    assert payload_fields["partial_file_count"] == 1
    assert payload_fields["partial_reasons"] == {PARTIAL_TRUNCATED_TAIL: 1}
    assert payload_fields["partial_left_out_bytes"] == len(payload) - len(complete)
    assert payload_fields["ingested_bytes"] == len(complete)
    assert payload_fields["refused_bytes_by_reason"] == {PARTIAL_TRUNCATED_TAIL: len(payload) - len(complete)}
    (chunk,) = [event for event in events if event.get("event") == "live.ingest.chunk"]
    assert chunk["outcome"] == "degraded"
    assert chunk["partial_file_count"] == 1
    assert chunk["partial_left_out_bytes"] == len(payload) - len(complete)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
    with sqlite3.connect(archive_root / "ops.db") as conn:
        (outcome_code, evidence_ref) = conn.execute(
            "SELECT outcome_code, evidence_ref FROM ingest_attempts ORDER BY started_at_ms DESC, rowid DESC LIMIT 1"
        ).fetchone()
    assert outcome_code == "success"
    assert evidence_ref == "batch:partial_admission"
