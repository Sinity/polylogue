"""Resident and batch Antigravity admission use the shared source route.

This file retains its historical command-path for the focused lane contract.
The old Antigravity-specific daemon loop is intentionally gone: the ordinary
live batch scheduler now invokes the same source-role and vendor-converter
route as batch acquisition.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers import antigravity
from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions, parse_one_source_path
from polylogue.sources.source_walk import _walk_source_paths


def test_source_role_contract_partitions_current_antigravity_items(tmp_path: Path) -> None:
    root = tmp_path / "antigravity"
    conversation = root / "conversations" / "cascade.pb"
    metadata = root / "brain" / "work" / "plan.md.metadata.json"
    document = root / "brain" / "work" / "plan.md"
    unknown = root / "settings" / "opaque.bin"

    expected = {
        conversation: (antigravity.AntigravitySourceRole.CONVERSATION_PROTOBUF, True),
        metadata: (antigravity.AntigravitySourceRole.METADATA_SIDECAR, False),
        document: (antigravity.AntigravitySourceRole.BRAIN_DOCUMENT, False),
        unknown: (antigravity.AntigravitySourceRole.UNKNOWN, False),
    }
    assert {
        path: (
            classification.role,
            classification.parse_as_session,
        )
        for path, classification in ((path, antigravity.classify_source_path(path)) for path in expected)
    } == expected


def test_shared_source_iterator_never_promotes_brain_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    (root / "conversations" / "cascade.pb").write_bytes(b"opaque")
    (root / "brain" / "work").mkdir(parents=True)
    (root / "brain" / "work" / "plan.md").write_text("# plan", encoding="utf-8")
    (root / "brain" / "work" / "plan.md.metadata.json").write_text("{}", encoding="utf-8")
    session = antigravity.parse_markdown_export(
        "### User Input\n\nhello",
        antigravity.AntigravitySessionSummary(cascade_id="cascade"),
    )

    def outcomes(*_args: object, **_kwargs: object) -> Iterator[antigravity.AntigravityExportOutcome]:
        yield antigravity.AntigravityExportOutcome(root / "conversations/cascade.pb", "cascade", session)

    monkeypatch.setattr(antigravity, "iter_language_server_export_results", outcomes)

    admitted = list(iter_antigravity_language_server_sessions(Source(name="antigravity", path=root)))

    assert [item[1].provider_session_id for item in admitted] == ["cascade"]
    assert all("metadata" not in item[1].provider_session_id for item in admitted)


def test_single_path_parser_uses_vendor_route_for_conversation_protobuf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "antigravity"
    conversation = root / "conversations" / "cascade.pb"
    conversation.parent.mkdir(parents=True)
    conversation.write_bytes(b"opaque protobuf")
    session = antigravity.parse_markdown_export(
        "### User Input\n\nhello",
        antigravity.AntigravitySessionSummary(cascade_id="cascade"),
    )
    calls: list[Path] = []

    def vendor_route(source: Source, **_kwargs: object) -> Iterator[tuple[object, object]]:
        assert source.path is not None
        calls.append(source.path)
        yield (None, session)

    monkeypatch.setattr(
        "polylogue.sources.source_parsing.iter_antigravity_language_server_sessions",
        vendor_route,
    )

    assert conversation in _walk_source_paths(root, provider=Provider.ANTIGRAVITY)

    admitted = list(
        parse_one_source_path(
            str(conversation),
            file_mtime=None,
            source_name="antigravity",
            sidecar_data={},
            capture_raw=False,
        )
    )

    assert calls == [root]
    assert [item[1].provider_session_id for item in admitted] == ["cascade"]


def test_poison_conversation_isolated_from_sibling_progress(tmp_path: Path) -> None:
    root = tmp_path / "antigravity"
    conversations = root / "conversations"
    conversations.mkdir(parents=True)
    for cascade_id in ("poison", "healthy"):
        (conversations / f"{cascade_id}.pb").write_bytes(cascade_id.encode())

    class Client:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

        def search_sessions(
            self, *, limit: int = 10000, query: str = ""
        ) -> list[antigravity.AntigravitySessionSummary]:
            return []

        def export_markdown(self, cascade_id: str) -> str:
            if cascade_id == "poison":
                return ""
            return "### User Input\n\nhealthy"

    outcomes = list(antigravity.iter_language_server_export_results(root, client=Client()))

    assert [outcome.cascade_id for outcome in outcomes] == ["healthy", "poison"]
    assert [outcome.cascade_id for outcome in outcomes if outcome.obtained] == ["healthy"]
    failed = next(outcome for outcome in outcomes if not outcome.obtained)
    assert failed.error is not None
    assert "partial" in failed.error or "empty" in failed.error


def test_common_live_batch_admits_conversation_through_vendor_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "antigravity"
    conversation = root / "conversations" / "cascade.pb"
    conversation.parent.mkdir(parents=True)
    conversation.write_bytes(b"opaque protobuf")

    class Client:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

        def search_sessions(
            self, *, limit: int = 10000, query: str = ""
        ) -> list[antigravity.AntigravitySessionSummary]:
            return []

        def export_markdown(self, cascade_id: str) -> str:
            assert cascade_id == "cascade"
            return "### User Input\n\nhello"

    monkeypatch.setattr(antigravity, "AntigravityLanguageServerClient", lambda _root: Client())
    index_db = tmp_path / "cursor.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="antigravity", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([conversation], source_name="antigravity")

    assert result.succeeded == [conversation], (result, caplog.text)
    assert result.failed == []


def test_failed_conversion_still_records_the_attempted_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conversion that raises must still capture the .pb revision it tried.

    Without the observation, the failed-cursor record keeps the old exclusion
    observation and the watcher re-runs the failing conversion every poll.

    Anti-vacuity: move the observation capture back after conversion and the
    failed path carries no captured observation here.
    """
    from polylogue.sources import source_parsing

    root = tmp_path / "antigravity"
    conversation = root / "conversations" / "cascade.pb"
    conversation.parent.mkdir(parents=True)
    conversation.write_bytes(b"opaque protobuf")

    def raising(*_args: object, **_kwargs: object) -> Iterator[object]:
        raise RuntimeError("language server unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(source_parsing, "iter_antigravity_language_server_sessions", raising)
    index_db = tmp_path / "cursor.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="antigravity", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    result = processor._ingest_full_paths_sync([conversation], source_name="antigravity")

    assert result.failed == [conversation]
    assert conversation in result.captured_file_observations


@pytest.mark.asyncio
async def test_common_live_batch_retries_a_failed_vendor_conversion(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from polylogue import Polylogue

    tmp_path = workspace_env["archive_root"]

    root = tmp_path / "antigravity"
    conversation = root / "conversations" / "cascade.pb"
    conversation.parent.mkdir(parents=True)
    conversation.write_bytes(b"opaque protobuf")

    class Client:
        attempts = 0

        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

        def search_sessions(
            self, *, limit: int = 10000, query: str = ""
        ) -> list[antigravity.AntigravitySessionSummary]:
            return []

        def export_markdown(self, cascade_id: str) -> str:
            self.attempts += 1
            if self.attempts == 1:
                raise antigravity.AntigravityExportError("transient conversion failure")
            return "### User Input\n\nhello"

    client = Client()
    monkeypatch.setattr(antigravity, "AntigravityLanguageServerClient", lambda _root: client)
    archive = Polylogue(archive_root=tmp_path, db_path=workspace_env["data_root"] / "index.db")
    index_db = tmp_path / "cursor.db"
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="antigravity", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    try:
        first = await processor.ingest_files([conversation], emit_event=False)
        failed_cursor = processor._cursor.get_record(conversation)

        assert first.failed_file_count == 1, (first, caplog.text)
        assert failed_cursor is not None
        assert failed_cursor.failure_count == 1
        assert failed_cursor.next_retry_at is not None

        second = await processor.ingest_files([conversation], emit_event=False)
        recovered_cursor = processor._cursor.get_record(conversation)

        assert client.attempts == 2
        assert second.succeeded_file_count == 1
        assert second.ingested_session_count == 1
        assert second.failed_file_count == 0
        assert recovered_cursor is not None
        assert recovered_cursor.failure_count == 0
        assert recovered_cursor.next_retry_at is None

    finally:
        await archive.close()


def _live_vendor_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_export: Any = None,
) -> tuple[LiveBatchProcessor, list[Path], list[str]]:
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    paths = [root / "conversations" / f"cascade-{number}.pb" for number in range(3)]
    for number, path in enumerate(paths):
        path.write_bytes(f"protobuf revision {number}".encode())
    exported: list[str] = []

    class Client:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def search_sessions(
            self, *, limit: int = 10000, query: str = ""
        ) -> list[antigravity.AntigravitySessionSummary]:
            return []

        def export_markdown(self, cascade_id: str) -> str:
            exported.append(cascade_id)
            if on_export is not None:
                on_export()
            return f"### User Input\n\nSynthetic conversation {cascade_id}"

    monkeypatch.setattr(antigravity, "AntigravityLanguageServerClient", lambda _root: Client())
    db_path = tmp_path / "cursor.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="antigravity", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test",
    )
    return processor, paths, exported


def test_vendor_conversion_cannot_publish_a_later_protobuf_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import sqlite3
    from hashlib import sha256

    from polylogue.sources import source_parsing
    from polylogue.sources.acquisition_boundary import capture_bound_path

    processor, paths, exported = _live_vendor_cohort(tmp_path, monkeypatch)
    converted_digest = sha256(paths[0].read_bytes()).hexdigest()
    original_capture = capture_bound_path

    def replace_between_conversion_and_capture(store: Any, path: Path, provider: Provider) -> tuple[str, int]:
        if path == paths[0]:
            path.write_bytes(b"different protobuf after the successful conversion")
        return original_capture(store, path, provider)

    monkeypatch.setattr(source_parsing, "capture_bound_path", replace_between_conversion_and_capture)
    result = processor._ingest_full_paths_sync(paths, source_name="antigravity")
    assert set(exported) == {path.stem for path in paths}, (exported, result, caplog.text)
    assert result.failed == [paths[0]]
    assert result.succeeded == paths[1:]
    assert paths[0] not in result.raw_fingerprints
    with sqlite3.connect(tmp_path / "source.db") as conn:
        retained = conn.execute("SELECT source_path, hex(blob_hash) FROM raw_sessions ORDER BY source_path").fetchall()
    assert retained == [(str(path), sha256(path.read_bytes()).hexdigest().upper()) for path in paths[1:]]
    assert converted_digest != sha256(paths[0].read_bytes()).hexdigest()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            (path.stem,) for path in paths[1:]
        ]


def test_vendor_cohort_checks_the_pass_budget_between_conversations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: Any,
) -> None:
    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED_TIME_BUDGET

    processor, paths, exported = _live_vendor_cohort(
        tmp_path,
        monkeypatch,
        on_export=lambda: frozen_clock.advance(2),
    )
    result = processor._ingest_full_paths_sync(
        paths,
        source_name="antigravity",
        max_pass_seconds=1,
        pass_started=frozen_clock.monotonic(),
    )
    assert exported == [paths[0].stem]
    assert result.failed == []
    assert result.time_budget_exceeded
    assert result.excluded == dict.fromkeys(paths[1:], REFUSED_UNATTEMPTED_TIME_BUDGET)
    assert paths[0] in result.succeeded or paths[0] in result.raw_deferred
    assert all(processor._cursor.get_record(path) is None for path in paths[1:])


def test_vendor_cohort_finishes_the_acquired_conversation_when_the_writer_budget_is_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: Any,
) -> None:
    """A conversation whose export outlasts the writer bound is published, not discarded.

    The rest stay unattempted backlog and the unit reports the spent hold.
    Anti-vacuity: raising at the next admission (the predecessor) discards
    the exported conversation, so a conversation that always outlasts the
    bound is re-exported and refused on every pass and never lands.
    """
    from polylogue.core.write_hold import enter_write_hold, exit_write_hold
    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED_TIME_BUDGET

    processor, paths, exported = _live_vendor_cohort(
        tmp_path,
        monkeypatch,
        on_export=lambda: frozen_clock.advance(31),
    )
    token = enter_write_hold("watcher.live_ingest.full", 30)
    try:
        result = processor._ingest_full_paths_sync(paths, source_name="antigravity")
    finally:
        exit_write_hold(token)
    assert exported == [paths[0].stem]
    assert result.write_hold_exhausted
    assert result.failed == []
    assert result.excluded == dict.fromkeys(paths[1:], REFUSED_UNATTEMPTED_TIME_BUDGET)
    assert paths[0] in result.succeeded or paths[0] in result.raw_deferred
    assert all(processor._cursor.get_record(path) is None for path in paths[1:])


def test_vendor_admission_refusal_happens_before_server_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processor, paths, exported = _live_vendor_cohort(tmp_path, monkeypatch)
    starts: list[Path] = []

    class RefusedClient:
        def start(self) -> None:
            starts.append(paths[0])
            raise AssertionError("unadmitted cohort started a vendor process")

        def close(self) -> None:
            pass

    monkeypatch.setattr(antigravity, "AntigravityLanguageServerClient", lambda _root: RefusedClient())
    assert (
        list(antigravity.iter_language_server_export_results(paths[0].parent.parent, admit_path=lambda _path: False))
        == []
    )
    assert starts == []
    assert exported == []


def test_an_excised_conversation_snapshot_is_reported_as_excised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused ``.pb`` snapshot is named to the caller as excised, not merely missing.

    Anti-vacuity (Codex P1, #5696): swallow the refusal without telling the
    caller and the live batch records a retryable failed cursor, retrying the
    forbidden file forever.
    """
    import polylogue.sources.source_parsing as source_parsing
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = tmp_path / "antigravity"
    conversation = root / "conversations" / "cascade.pb"
    conversation.parent.mkdir(parents=True)
    conversation.write_bytes(b"opaque")
    session = antigravity.parse_markdown_export(
        "### User Input\n\nhello", antigravity.AntigravitySessionSummary(cascade_id="cascade")
    )

    def outcomes(*_args: object, **_kwargs: object) -> Iterator[antigravity.AntigravityExportOutcome]:
        yield antigravity.AntigravityExportOutcome(conversation, "cascade", session)

    def refused(*_args: object, **_kwargs: object) -> object:
        raise ContentExcisedError(blob_hash=bytes(32), source_path=str(conversation))

    monkeypatch.setattr(antigravity, "iter_language_server_export_results", outcomes)
    monkeypatch.setattr(source_parsing, "_antigravity_raw_snapshot", refused)
    excised: set[Path] = set()

    admitted = list(
        iter_antigravity_language_server_sessions(
            Source(name="antigravity", path=root), capture_raw=True, excised=excised
        )
    )

    assert admitted == []
    assert excised == {conversation}
