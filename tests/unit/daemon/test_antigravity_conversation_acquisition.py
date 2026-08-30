"""Resident and batch Antigravity admission use the shared source route.

This file retains its historical command-path for the focused lane contract.
The old Antigravity-specific daemon loop is intentionally gone: the ordinary
live batch scheduler now invokes the same source-role and vendor-converter
route as batch acquisition.
"""

from __future__ import annotations

import asyncio
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

    assert result.succeeded == [conversation]
    assert result.failed == []


def test_common_live_batch_retries_a_failed_vendor_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

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
    initialize_active_archive_root(tmp_path)
    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    index_db = tmp_path / "cursor.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="antigravity", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    try:
        first = asyncio.run(processor.ingest_files([conversation], emit_event=False))
        failed_cursor = processor._cursor.get_record(conversation)

        assert first.failed_file_count == 1
        assert failed_cursor is not None
        assert failed_cursor.failure_count == 1
        assert failed_cursor.next_retry_at is not None

        second = asyncio.run(processor.ingest_files([conversation], emit_event=False))
        recovered_cursor = processor._cursor.get_record(conversation)
    finally:
        clear_degraded()

    assert second.failed_file_count == 0
    assert recovered_cursor is not None
    assert recovered_cursor.failure_count == 0
    assert recovered_cursor.next_retry_at is None
