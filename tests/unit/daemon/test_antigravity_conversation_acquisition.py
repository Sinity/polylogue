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
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers import antigravity
from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions


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
    assert "empty" in failed.error


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
