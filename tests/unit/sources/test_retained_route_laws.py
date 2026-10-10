"""Laws of the retained raw-owner route whose predecessor tests retired with their mechanism.

The watcher parse-stage prefetch, the historical backfill engine and the eager
writers each carried tests for properties that still hold for the route that
replaced them: live acquisition hands acquired raws to
``RawObservationConvergenceOwner``, which prepares each retained raw on its
admitted compute worker (``RawObservationDerivation.compute``) and publishes
the sealed carrier through the writer. Every test here drives that owner and
names the production change that turns it red.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import tracemalloc
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider, ValidationMode
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.raw_owner_routes import cold_rebuilt_index, run_ingest_files

# ---------------------------------------------------------------------------
# Synthetic wire fixtures


def _chatgpt_conversation(
    conversation_id: str | None, *turns: tuple[str, str], title: str = "neutral"
) -> dict[str, object]:
    """One ChatGPT export conversation whose mapping is a single linear branch."""
    mapping: dict[str, dict[str, object]] = {}
    previous: str | None = None
    for index, (role, text) in enumerate(turns):
        node = f"n{index}"
        mapping[node] = {
            "id": node,
            "parent": previous,
            "children": [],
            "message": {
                "id": node,
                "author": {"role": role},
                "create_time": index + 1,
                "content": {"content_type": "text", "parts": [text]},
            },
        }
        if previous is not None:
            children = mapping[previous]["children"]
            assert isinstance(children, list)
            children.append(node)
        previous = node
    conversation: dict[str, object] = {
        "title": title,
        "create_time": 1,
        "update_time": len(turns),
        "current_node": previous,
        "mapping": mapping,
    }
    if conversation_id is not None:
        conversation["id"] = conversation_id
    return conversation


def _chatgpt_export(*conversations: dict[str, object]) -> bytes:
    return json.dumps(list(conversations)).encode()


def _codex_rollout(native_id: str, records: int, *, text_bytes: int = 200) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}
    ]
    for index in range(records):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": [
                        {
                            "type": "input_text" if index % 2 == 0 else "output_text",
                            "text": f"turn {index} " + "x" * text_bytes,
                        }
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _write_hermes_snapshot(path: Path, *, messages: int, text_bytes: int = 200) -> None:
    """A Hermes ``session.json`` snapshot written incrementally, never held whole."""
    with path.open("wb") as handle:
        handle.write(b'{"session_id":"large-hermes","platform":"cli","messages":[')
        for index in range(messages):
            if index:
                handle.write(b",")
            role = "user" if index % 2 == 0 else "assistant"
            handle.write(json.dumps({"role": role, "content": f"turn {index} " + "x" * text_bytes}).encode())
        handle.write(b"]}")


# ---------------------------------------------------------------------------
# Route drivers and observations


def _live_ingest(archive_root: Path, paths: Sequence[Path], *, source_name: str) -> Any:
    """One production live pass: acquisition, then retained publication by the raw owner."""
    archive_root.mkdir(parents=True, exist_ok=True)
    bootstrap_archive_root(archive_root)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name=source_name, root=paths[0].parent, layout=export_drop_layout((".json", ".jsonl"))),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
    )
    metrics = run_ingest_files(processor, list(paths), emit_event=False)
    assert metrics.failed_file_count == 0, metrics
    return metrics


def _rows(database: Path, sql: str, parameters: Sequence[object] = ()) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as conn:
        return [tuple(row) for row in conn.execute(sql, tuple(parameters))]


def _raw_ids_by_path(archive_root: Path) -> dict[str, list[str]]:
    by_path: dict[str, list[str]] = {}
    for raw_id, source_path in _rows(
        archive_root / "source.db", "SELECT raw_id, source_path FROM raw_sessions ORDER BY rowid"
    ):
        by_path.setdefault(str(source_path), []).append(str(raw_id))
    return by_path


def _acquire(
    archive_root: Path,
    provider: Provider,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
    *,
    native_id: str | None = None,
) -> str:
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=provider,
            payload=payload,
            source_path=source_path,
            canonical_source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            native_id=native_id,
        )


@contextmanager
def _traced_preparation(
    monkeypatch: pytest.MonkeyPatch, observe: Callable[[], None] | None = None
) -> Iterator[list[int]]:
    """Record the traced Python allocation peak of every retained preparation.

    Tracing covers ``RawObservationDerivation.compute`` only -- the canonical
    preparation on the admitted worker -- so acquisition and the test's own
    setup do not enter the measurement.
    """
    peaks: list[int] = []
    original = RawObservationDerivation.compute

    def traced(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> RawObservationReplacement:
        tracemalloc.start()
        try:
            if observe is not None:
                observe()
            return original(self, *args, **kwargs)
        finally:
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

    monkeypatch.setattr(RawObservationDerivation, "compute", traced)
    yield peaks


# ---------------------------------------------------------------------------
# Laws


@pytest.mark.slow
def test_large_hermes_snapshot_preparation_memory_does_not_track_session_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hermes snapshot larger than the decode budget prepares in bounded memory.

    The prepared message sink keeps a whole decoded session resident only
    while it fits half of ``DECODED_SESSION_BUDGET_BYTES``; a larger session
    streams from its sealed carrier. The budget is shrunk here so a modest
    synthetic snapshot is the "larger than the budget" case a giant one is in
    production. Twice the messages may not cost as much extra traced
    memory as the extra input bytes themselves.

    Anti-vacuity: lex the detection projection without
    ``LexemeAlignedReader`` and ``ijson.backends.python``'s buffer keeps every
    consumed chunk, so the peak grows about two bytes per input byte; any
    change that keeps a whole decoded session or document resident turns it
    red as well.
    """
    from polylogue.sources import prepared_message_sink

    monkeypatch.setattr(prepared_message_sink._DECODED_SESSIONS, "budget_bytes", 64 * 1024)
    measured: dict[str, tuple[int, int]] = {}
    with _traced_preparation(monkeypatch) as peaks:
        # Both snapshots fill every fixed 1 MiB read buffer the detection
        # projection stacks (decoder, prefix and lexeme-aligned readers) and
        # the sink's bounded row pages, so what remains to differ is memory
        # that tracks the session.
        for label, messages in (("small", 800), ("large", 1600)):
            prepared_message_sink._DECODED_SESSIONS.clear()
            prepared_message_sink._DECODED_SPOOLS.clear()
            sessions = tmp_path / label / "profile" / "sessions"
            sessions.mkdir(parents=True)
            source = sessions / "session_large.json"
            _write_hermes_snapshot(source, messages=messages, text_bytes=3000)
            assert source.stat().st_size > 2 * 1024 * 1024
            archive_root = tmp_path / label / "archive"
            peaks.clear()
            metrics = _live_ingest(archive_root, [source], source_name=Provider.HERMES.value)
            assert metrics.succeeded_file_count == 1
            assert metrics.ingested_message_count == messages
            assert peaks, "the retained preparation never ran"
            measured[label] = (source.stat().st_size, max(peaks))
    (small_bytes, small_peak), (large_bytes, large_peak) = measured["small"], measured["large"]
    assert large_peak - small_peak < large_bytes - small_bytes, measured


def test_identical_json_payloads_at_distinct_paths_keep_distinct_fallback_sessions(tmp_path: Path) -> None:
    """Equal bytes at two paths publish two path-bound sessions, never one shared parse.

    An export conversation without an ``id`` takes its fallback identity from
    the retained raw's own source path, so ``a.json`` and ``b.json`` holding
    identical bytes are two sessions, each bound to its own raw.

    Anti-vacuity: derive the retained JSON fallback id from anything shared by
    equal bytes (the blob hash) instead of the raw's source path in
    ``prepare_retained_jsonl_artifact`` and both raws publish one session.
    """
    sources = tmp_path / "chatgpt"
    sources.mkdir()
    payload = _chatgpt_export(_chatgpt_conversation(None, ("user", "same content"), title="same bytes"))
    paths = [sources / "a.json", sources / "b.json"]
    for path in paths:
        path.write_bytes(payload)
    archive_root = tmp_path / "archive"

    _live_ingest(archive_root, paths, source_name=Provider.CHATGPT.value)

    raws = _raw_ids_by_path(archive_root)
    assert sorted(raws) == sorted(str(path) for path in paths)
    assert all(len(ids) == 1 for ids in raws.values()), raws
    published = {
        str(native_id): str(raw_id)
        for native_id, raw_id in _rows(archive_root / "index.db", "SELECT native_id, raw_id FROM sessions")
    }
    assert published == {"a-0": raws[str(paths[0])][0], "b-0": raws[str(paths[1])][0]}, (published, raws)


def test_detected_origin_reaches_the_published_session(tmp_path: Path) -> None:
    """A document the inbox classifies keeps that origin from Source through the Index.

    The inbox source is provider-neutral; acquisition detects a Gemini CLI
    document. The retained raw, the published session and its accepted head
    all carry the detected origin, never the inbox's neutral one.

    Anti-vacuity: let the retained descriptor (``_raw_revision_descriptor_from_row``)
    hand preparation a provider other than the detected one -- for instance
    the ``gemini`` family token for ``gemini-cli`` -- and the document is no
    longer published under its detected origin.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    source = inbox / "session.json"
    source.write_text(
        json.dumps(
            {
                "sessionId": "gemini-prepared-json",
                "startTime": "2026-03-16T09:40:00.000Z",
                "lastUpdated": "2026-03-16T09:41:00.000Z",
                "kind": "chat",
                "messages": [
                    {"id": "u1", "timestamp": "2026-03-16T09:40:01.000Z", "type": "user", "content": ["hello"]},
                    {
                        "id": "a1",
                        "timestamp": "2026-03-16T09:40:02.000Z",
                        "type": "gemini",
                        "content": "world",
                        "model": "gemini-test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    archive_root = tmp_path / "archive"

    metrics = _live_ingest(archive_root, [source], source_name="inbox")

    assert metrics.ingested_session_count == 1
    ((raw_id, raw_origin, detected),) = _rows(
        archive_root / "source.db", "SELECT raw_id, origin, detected_provider FROM raw_sessions"
    )
    assert (raw_origin, detected) == ("gemini-cli-session", Provider.GEMINI_CLI.value)
    ((session_raw, session_origin, message_count),) = _rows(
        archive_root / "index.db", "SELECT raw_id, origin, message_count FROM sessions"
    )
    assert (session_raw, session_origin, message_count) == (raw_id, "gemini-cli-session", 2)
    ((logical_key, accepted_raw),) = _rows(
        archive_root / "index.db", "SELECT logical_source_key, accepted_raw_id FROM raw_revision_heads"
    )
    assert logical_key.startswith("gemini-cli-session:")
    assert accepted_raw == raw_id


def test_reacquired_bundle_binds_every_session_to_its_later_acquisition(tmp_path: Path) -> None:
    """Each conversation of a re-exported bundle is published from the later raw.

    One ``conversations.json`` carries two conversations; the re-export grows
    both. Preparation of the later raw must bind every session it writes --
    its row, its accepted head and its content -- to that raw, not to the
    earlier acquisition and not to one member's binding for both.

    Anti-vacuity: prepare a membership write under the cohort's first accepted
    raw instead of its last (``accepted_raw_ids[0]`` for ``[-1]`` in
    ``RawObservationDerivation._compute_prepared``) and the later pass does
    not publish the re-export from the later raw.
    """
    sources = tmp_path / "chatgpt"
    sources.mkdir()
    bundle = sources / "conversations.json"
    bundle.write_bytes(
        _chatgpt_export(
            _chatgpt_conversation("conv-one", ("user", "first")),
            _chatgpt_conversation("conv-two", ("user", "second")),
        )
    )
    archive_root = tmp_path / "archive"
    _live_ingest(archive_root, [bundle], source_name=Provider.CHATGPT.value)
    (earlier,) = _raw_ids_by_path(archive_root)[str(bundle)]

    bundle.write_bytes(
        _chatgpt_export(
            _chatgpt_conversation("conv-one", ("user", "first"), ("assistant", "reply one")),
            _chatgpt_conversation("conv-two", ("user", "second"), ("assistant", "reply two")),
        )
    )
    metrics = _live_ingest(archive_root, [bundle], source_name=Provider.CHATGPT.value)

    assert metrics.ingested_session_count == 2
    first, later = _raw_ids_by_path(archive_root)[str(bundle)]
    assert first == earlier and later != earlier
    assert _rows(archive_root / "index.db", "SELECT native_id, raw_id, message_count FROM sessions ORDER BY 1") == [
        ("conv-one", later, 2),
        ("conv-two", later, 2),
    ]
    assert _rows(
        archive_root / "index.db", "SELECT logical_source_key, accepted_raw_id FROM raw_revision_heads ORDER BY 1"
    ) == [("chatgpt-export:conv-one", later), ("chatgpt-export:conv-two", later)]
    replies = _rows(
        archive_root / "index.db",
        "SELECT b.text FROM blocks AS b JOIN messages AS m ON m.message_id = b.message_id "
        "WHERE m.role = 'assistant' ORDER BY m.session_id",
    )
    assert replies == [("reply one",), ("reply two",)]


def test_lowered_sink_identity_is_stable_from_preparation_through_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest preparation seals is the digest every publication of those bytes stores.

    Preparation seals the parser's session beside a separately lowered
    operand (tool outcomes derived) and a shard of prepared rows bound to the
    parse digest. Live publication, the accepted head and an independent
    re-preparation into a fresh generation must all agree on that digest and
    on the rows, and the lowered tool outcome must reach the stored blocks.

    Anti-vacuity: seal the parser's unlowered messages into the shard instead
    of the normalized operand (``append_session_to_shard``) and the live pass
    fails to publish; re-hash the lowered operand when sealing the shard
    instead of carrying the parse-bound digest and the stored digest no
    longer matches the one preparation sealed.
    """
    from polylogue.storage.sqlite.archive_tiers import write as archive_write

    sealed: dict[str, bytes] = {}
    original = archive_write.prepared_session_rows_from_shard

    def record_sealed(shard_path: Path, session_id: str) -> Any:
        rows = original(shard_path, session_id)
        sealed[session_id] = bytes(rows.session_content_hash)
        return rows

    monkeypatch.setattr(archive_write, "prepared_session_rows_from_shard", record_sealed)
    records: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": "lowered", "timestamp": "2026-07-19T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "lowered-m0",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the files"}],
            },
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call", "name": "shell", "arguments": '{"command":["ls"]}', "call_id": "c1"},
        },
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": "a.txt"}},
    ]
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    source = sessions / "lowered.jsonl"
    source.write_bytes(b"".join(json.dumps(record, sort_keys=True).encode() + b"\n" for record in records))
    archive_root = tmp_path / "archive"

    _live_ingest(archive_root, [source], source_name=Provider.CODEX.value)

    Published = tuple[str, bytes, bytes, list[tuple[Any, ...]], list[tuple[Any, ...]]]

    def published(index: Path) -> Published:
        ((session_id, content_hash),) = _rows(index, "SELECT session_id, content_hash FROM sessions")
        ((head_hash,),) = _rows(index, "SELECT accepted_content_hash FROM raw_revision_heads")
        messages = _rows(
            index,
            "SELECT message_id, position, role, message_type, material_origin, content_hash, fields_digest "
            "FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
        blocks = _rows(
            index,
            "SELECT block_id, block_type, tool_id, tool_outcome, text, content_hash "
            "FROM blocks WHERE session_id = ? ORDER BY block_id",
            (session_id,),
        )
        return str(session_id), bytes(content_hash), bytes(head_hash), messages, blocks

    live = published(archive_root / "index.db")
    session_id, live_hash, live_head, _messages, live_blocks = live
    assert sealed.get(session_id) == live_hash == live_head, (sealed, live_hash, live_head)
    outcomes = [row[3] for row in live_blocks if row[2] == "c1"]
    assert len(outcomes) == 2 and None not in outcomes, live_blocks

    sealed.clear()

    async def rebuild() -> Published:
        async with cold_rebuilt_index(archive_root) as index:
            return published(index)

    rebuilt = asyncio.run(rebuild())
    assert sealed.get(session_id) == live_hash
    assert rebuilt == live


def _contains_fault(failure: BaseException, message: str) -> bool:
    if isinstance(failure, BaseExceptionGroup):
        return any(_contains_fault(member, message) for member in failure.exceptions)
    return message in str(failure) or (failure.__cause__ is not None and _contains_fault(failure.__cause__, message))


@pytest.mark.asyncio
@pytest.mark.parametrize("identified", [False, True], ids=["identity-opaque", "native-identified"])
@pytest.mark.parametrize("failure_kind", ["runtime", "prepared_file"])
@pytest.mark.parametrize("fixed_page", [False, True])
async def test_one_raw_preparation_failure_does_not_block_its_replay_page_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identified: bool, failure_kind: str, fixed_page: bool
) -> None:
    """A retryable preparation failure in one raw leaves its page's other raws published.

    The live pass hands every acquired raw to one retained replay
    (``replay_retained_raw_ids``). A raw whose preparation raises is that
    raw's retryable outcome, returned with its error beside the receipts of
    the healthy raws offered before and after it, which still publish.
    ``identity-opaque`` raws carry no native id, so the replay censuses the
    page's envelopes together before any of them replays; ``native-identified``
    raws prepare one by one.

    Anti-vacuity: re-raise an untyped preparation failure in
    ``retained_replay_operation`` and ``page-after`` never prepares; drop the
    single-raw frame scope of the failed census and the opaque seeds' census
    keeps pulling ``broken`` in, so nothing publishes.
    """
    from polylogue.core.prepared_file import PreparedFileSeal
    from polylogue.logging import capture
    from polylogue.sources import revision_backfill

    private_file = tmp_path / "private-prepared-operand"
    private_file.write_bytes(b"neutral prepared bytes")
    seal = PreparedFileSeal.capture(private_file)
    private_file.unlink()
    private_file.write_bytes(b"replacement prepared bytes")

    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    acquired: dict[str, str] = {}
    for index, name in enumerate(("before", "broken", "after")):
        payload = _chatgpt_export(_chatgpt_conversation(f"page-{name}", ("user", f"{name} content")))
        acquired[name] = await run_archive_fixture_write(
            tmp_path,
            partial(
                _acquire,
                tmp_path,
                Provider.CHATGPT,
                payload,
                f"{name}.json",
                index + 1,
                native_id=f"page-{name}" if identified else None,
            ),
        )
    original = revision_backfill.prepare_retained_jsonl_artifact

    def failing(
        evidence_reader: Any,
        raw_id: str,
        *,
        directory: Path,
        allow_generic_object_alias: bool = False,
        validation_mode: ValidationMode = ValidationMode.ADVISORY,
        **preparation_options: Any,
    ) -> Any:
        if raw_id == acquired["broken"]:
            if failure_kind == "prepared_file":
                seal.verify(private_file, full=False)
            raise RuntimeError("synthetic preparation fault")
        return original(
            evidence_reader,
            raw_id,
            directory=directory,
            allow_generic_object_alias=allow_generic_object_alias,
            validation_mode=validation_mode,
            **preparation_options,
        )

    monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", failing)
    with capture() as events:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            outcome = await owner.replay_retained_raw_ids(
                tuple(acquired.values()),
                select_retained_raw_ids=(lambda _read: tuple(acquired.values())) if fixed_page else None,
            )
    retries = [e for e in events if e.get("event") == "storage.raw_observation.preparation_isolated"]
    assert retries
    assert {e["reason"] for e in retries} == {"page_scope_failed", "raw_preparation_failed"}
    for event in retries:
        assert event["outcome"] == "degraded" and event["phase"] == "source_preparation"
        assert event["productive_id"] in acquired.values()
        assert event["raws"] in {1, 2, 3}
        assert event["error_type"] == ("ValueError" if failure_kind == "prepared_file" else "RuntimeError"), retries
        if failure_kind == "prepared_file":
            assert event["operation"] == "PreparedFileSeal.verify"
        assert str(private_file) not in json.dumps(event) and "synthetic preparation fault" not in json.dumps(event)
    assert [failure.raw_id for failure in outcome.failures] == [acquired["broken"]], outcome
    expected_error = ValueError if failure_kind == "prepared_file" else RuntimeError
    assert isinstance(outcome.failures[0].error, expected_error), outcome.failures
    if failure_kind == "runtime":
        assert _contains_fault(outcome.failures[0].error, "synthetic preparation fault"), outcome.failures
    published = {str(native_id) for (native_id,) in _rows(tmp_path / "index.db", "SELECT native_id FROM sessions")}
    assert published == {"page-before", "page-after"}, published
    receipt_sessions = {sid for receipt in outcome.receipts for sid in receipt.written_session_ids}
    assert len(receipt_sessions) == 2, outcome.receipts
    with pytest.raises(expected_error):
        outcome.require_complete()


@pytest.mark.asyncio
async def test_page_isolation_does_not_reoffer_settled_independent_raws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late failed page sibling cannot make later seeds reparse settled outputs."""
    from polylogue.sources import revision_backfill

    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    acquired: dict[str, str] = {}
    for index, name in enumerate(("first", "second", "third")):
        acquired[name] = await run_archive_fixture_write(
            tmp_path,
            partial(
                _acquire,
                tmp_path,
                Provider.CHATGPT,
                _chatgpt_export(_chatgpt_conversation(f"page-{name}", ("user", name))),
                f"{name}.json",
                index + 1,
            ),
        )
    selected = tuple(sorted(acquired.values()))
    broken = selected[-1]
    original_prepare = revision_backfill.prepare_retained_jsonl_artifact
    original_publish = RawObservationDerivation.publish
    settled: set[str] = set()
    repeated: list[str] = []

    def prepare(evidence_reader: Any, raw_id: str, **kwargs: Any) -> Any:
        if raw_id in settled:
            repeated.append(raw_id)
        if raw_id == broken:
            raise RuntimeError("synthetic late preparation fault")
        return original_prepare(evidence_reader, raw_id, **kwargs)

    def publish(
        self: RawObservationDerivation, frame: Any, replacement: RawObservationReplacement, **kwargs: Any
    ) -> bool:
        published = original_publish(self, frame, replacement, **kwargs)
        if published:
            settled.update(replacement.raw_ids)
        return published

    monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", prepare)
    monkeypatch.setattr(RawObservationDerivation, "publish", publish)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        outcome = await owner.replay_retained_raw_ids(selected, select_retained_raw_ids=lambda _read: selected)
    assert [failure.raw_id for failure in outcome.failures] == [broken]
    assert {str(row[0]) for row in _rows(tmp_path / "index.db", "SELECT native_id FROM sessions")} == {
        f"page-{name}" for name, raw_id in acquired.items() if raw_id != broken
    }
    assert settled == set(selected[:-1])
    assert repeated == []


@pytest.mark.asyncio
async def test_settled_raw_remains_required_when_later_census_joins_its_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filtering page offers cannot remove a member required by current Source."""
    from polylogue.sources import revision_backfill

    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    conversations = (
        _chatgpt_conversation("shared-page", ("user", "neutral prefix")),
        _chatgpt_conversation("shared-page", ("user", "neutral prefix"), ("assistant", "neutral tail")),
        _chatgpt_conversation("broken-page", ("user", "neutral fault input")),
    )
    selected: list[str] = []
    for index, conversation in enumerate(conversations):
        selected.append(
            await run_archive_fixture_write(
                tmp_path,
                partial(
                    _acquire, tmp_path, Provider.CHATGPT, _chatgpt_export(conversation), f"part-{index}.json", index
                ),
            )
        )
    original_prepare = revision_backfill.prepare_retained_jsonl_artifact
    original_publish = RawObservationDerivation.publish
    published_units: list[tuple[str, ...]] = []

    def prepare(evidence_reader: Any, raw_id: str, **kwargs: Any) -> Any:
        if raw_id == selected[-1]:
            raise RuntimeError("synthetic independent preparation fault")
        return original_prepare(evidence_reader, raw_id, **kwargs)

    def publish(
        self: RawObservationDerivation, frame: Any, replacement: RawObservationReplacement, **kwargs: Any
    ) -> bool:
        published = original_publish(self, frame, replacement, **kwargs)
        if published:
            published_units.append(replacement.raw_ids)
        return published

    monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", prepare)
    monkeypatch.setattr(RawObservationDerivation, "publish", publish)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        outcome = await owner.replay_retained_raw_ids(
            tuple(selected), select_retained_raw_ids=lambda _read: tuple(selected)
        )
    assert [failure.raw_id for failure in outcome.failures] == [selected[-1]]
    assert published_units[0] == (selected[0],)
    assert any(set(unit) == set(selected[:2]) for unit in published_units[1:])
    assert _rows(tmp_path / "index.db", "SELECT native_id,message_count FROM sessions") == [("shared-page", 2)]
    assert _rows(tmp_path / "index.db", "SELECT text FROM blocks ORDER BY position") == [
        ("neutral prefix",),
        ("neutral tail",),
    ]


@pytest.mark.asyncio
async def test_raw_owner_keeps_one_preparation_in_flight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent admissions and a multi-raw replay hold one preparation at a time.

    The compute pool is wide enough to run four preparations at once; the
    owner's serialization and exclusive byte admission are what bound it.
    Within one retained replay, each raw's carrier is settled before the next
    raw prepares, so prepared trees never accumulate across a page.

    Anti-vacuity: drop the owner's ``_converge_lock`` and the concurrent
    ``converge_raw_id`` calls collide in exclusive byte admission and fail as
    saturated; leave each carrier open after publication and retire the
    page's carriers only after the replay loop, and the next preparation
    starts beside unsettled carriers.
    """
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    raw_ids: list[str] = []
    for index in range(6):
        payload = _chatgpt_export(_chatgpt_conversation(f"inflight-{index}", ("user", f"turn {index}")))
        raw_ids.append(
            await run_archive_fixture_write(
                tmp_path,
                partial(_acquire, tmp_path, Provider.CHATGPT, payload, f"inflight-{index}.json", index + 1),
            )
        )
    lock = threading.Lock()
    entered = threading.Condition(lock)
    state = {"computing": 0, "peak_computing": 0, "unsettled_at_entry": 0}
    unsettled: set[int] = set()
    original_compute = RawObservationDerivation.compute
    original_close = RawObservationReplacement.close

    def counted_compute(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> RawObservationReplacement:
        with entered:
            state["computing"] += 1
            state["peak_computing"] = max(state["peak_computing"], state["computing"])
            state["unsettled_at_entry"] = max(state["unsettled_at_entry"], len(unsettled))
            entered.notify_all()
            # Give an unbounded dispatcher the chance to start a sibling
            # preparation while this one is in flight; a bounded owner has
            # none to start, so the wait simply elapses.
            entered.wait_for(lambda: state["computing"] > 1, timeout=0.2)
        try:
            replacement = original_compute(self, *args, **kwargs)
        finally:
            with entered:
                state["computing"] -= 1
        with lock:
            unsettled.add(id(replacement))
        return replacement

    def counted_close(self: RawObservationReplacement) -> None:
        original_close(self)
        with lock:
            unsettled.discard(id(self))

    monkeypatch.setattr(RawObservationDerivation, "compute", counted_compute)
    monkeypatch.setattr(RawObservationReplacement, "close", counted_close)
    compute = BoundedComputeAdapter(max_workers=4, queue_units=8)
    try:
        async with prepared_live_convergence_owner(tmp_path, compute_adapter=compute) as owner:
            reports = await asyncio.gather(*(owner.converge_raw_id(raw_id) for raw_id in raw_ids[:3]))
            assert [(report.done, report.failed) for report in reports] == [(1, 0)] * 3
            results = (await owner.replay_retained_raw_ids(raw_ids[3:])).require_complete()
            assert len(results) == 3
    finally:
        await asyncio.to_thread(compute.shutdown, wait=True)
    assert state["peak_computing"] == 1, state
    assert state["unsettled_at_entry"] == 0, state
    assert len(_rows(tmp_path / "index.db", "SELECT 1 FROM sessions")) == 6


@pytest.mark.slow
@pytest.mark.asyncio
async def test_retained_replay_holds_the_declared_byte_envelope_for_a_raw_larger_than_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw many times the compute byte envelope replays inside that envelope.

    The envelope (``queue_bytes``) is the declared memory bound of daemon
    compute, standing in here for a multi-GiB raw against the production
    envelope. Fair intake replays a retained raw through ``converge_raw_id``,
    which declares the raw's full size. A raw larger than the whole envelope
    is never refused for its size: it is admitted exclusively and charged
    exactly the envelope. Preparation memory must not grow with that size:
    twice the bytes may not cost even half the extra bytes in extra traced
    memory.

    Anti-vacuity: charge the raw its declared size (drop the envelope cap in
    ``BoundedComputeAdapter.submit``) and admission refuses it as saturated;
    keep every Codex record resident regardless of
    ``_CODEX_REPLAY_MEMORY_BUDGET_BYTES`` (no spool in ``retain_records``) and
    the larger raw's preparation peak grows with it.
    """
    from polylogue.sources import prepared_message_sink
    from polylogue.sources.parsers import codex

    envelope = 64 * 1024
    # Every declared in-memory budget on this route is shrunk with the
    # envelope, so the synthetic raws are past all of them as a multi-GiB raw
    # is past the production values.
    monkeypatch.setattr(codex, "_CODEX_REPLAY_MEMORY_BUDGET_BYTES", envelope)
    monkeypatch.setattr(prepared_message_sink._DECODED_SESSIONS, "budget_bytes", envelope)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=envelope)
    snapshots: list[tuple[int, int, int]] = []

    def observe() -> None:
        snapshot = compute.snapshot()
        snapshots.append((snapshot.used_bytes, snapshot.exclusive_byte_units, snapshot.capacity_bytes))

    measured: dict[str, tuple[int, int]] = {}
    try:
        with _traced_preparation(monkeypatch, observe) as peaks:
            # The first preparation in a process also pays one-time import and
            # cache warm-up; it is measured but not compared. Both compared
            # raws exceed the route's fixed 1 MiB read buffers and its 512-row
            # sink pages, so what remains to differ is memory that tracks the
            # raw.
            for label, records in (("warm-up", 100), ("small", 1600), ("large", 3200)):
                root = tmp_path / label
                await run_archive_fixture_write(root, partial(bootstrap_archive_root, root))
                payload = _codex_rollout(f"envelope-{label}", records, text_bytes=700)
                assert len(payload) > envelope
                assert label == "warm-up" or len(payload) > 1024 * 1024
                raw_id = await run_archive_fixture_write(
                    root, partial(_acquire, root, Provider.CODEX, payload, f"envelope-{label}.jsonl", 1)
                )
                prepared_message_sink._DECODED_SESSIONS.clear()
                prepared_message_sink._DECODED_SPOOLS.clear()
                peaks.clear()
                snapshots.clear()
                async with prepared_live_convergence_owner(root, compute_adapter=compute) as owner:
                    report = await owner.converge_raw_id(raw_id)
                assert (report.done, report.failed, report.pending) == (1, 0, 0), report.outcomes
                assert _rows(root / "index.db", "SELECT message_count FROM sessions") == [(records,)]
                assert snapshots and all(snap == (envelope, 1, envelope) for snap in snapshots), snapshots
                measured[label] = (len(payload), max(peaks))
    finally:
        await asyncio.to_thread(compute.shutdown, wait=True)
    (small_bytes, small_peak), (large_bytes, large_peak) = measured["small"], measured["large"]
    assert large_peak - small_peak < (large_bytes - small_bytes) // 2, measured
