"""Every live cursor write carries the observed file's path authority.

The raw-frontier gate refuses any cursor row holding a byte offset without a
canonical source path ("ops cursor canonical path authority is unavailable"),
so one writer that drops it blocks every later source selection.

Anti-vacuity: drop ``authority=`` from any of the failed, refused or
full-retry invalidation writers (they wrote a byte offset with no canonical
path before) and its row reads back with ``canonical_source_path`` NULL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorPathAuthority, CursorStore
from polylogue.sources.parsers.hermes_identity import declares_profile_identity
from tests.infra.archive_templates import bootstrap_archive_root

_ROLLOUT = (
    b'{"type":"session_meta","payload":{"id":"authority","timestamp":"2026-06-02T00:00:00Z"}}\n'
    b'{"type":"response_item","payload":{"type":"message","id":"m0","role":"user",'
    b'"content":[{"type":"input_text","text":"zero"}]}}\n'
)


def _processor(root: Path) -> tuple[LiveBatchProcessor, CursorStore]:
    bootstrap_archive_root(root)
    cursor = CursorStore(root / "index.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=root, backend=SimpleNamespace(db_path=root / "index.db"))),
        (WatchSource(name="codex", root=root / "sessions"),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    return processor, cursor


def _unauthorized_offsets(root: Path) -> list[str]:
    with sqlite3.connect(root / "ops.db") as ops:
        return [
            str(row[0])
            for row in ops.execute(
                "SELECT source_path FROM ingest_cursor "
                "WHERE canonical_source_path IS NULL AND byte_offset IS NOT NULL AND excluded = 0"
            )
        ]


@pytest.mark.parametrize("writer", ["failed", "refused", "full_retry"])
def test_first_cursor_write_claims_the_observed_file_authority(tmp_path: Path, writer: str) -> None:
    processor, cursor = _processor(tmp_path)
    day = tmp_path / "sessions" / "2026" / "06" / "02"
    day.mkdir(parents=True)
    path = day / "rollout-authority.jsonl"
    path.write_bytes(_ROLLOUT)

    if writer == "failed":
        processor._record_failed_cursor(path)
    elif writer == "refused":
        processor._mark_refused_cursor(path, path.stat(), source_name="codex", reason="unsupported")
    else:
        processor._invalidate_cursor_for_full_retry(path, source_name="codex", stat=path.stat())

    record = cursor.get_record(path)
    assert record is not None
    observed = CursorPathAuthority.observe(path)
    assert observed.canonical_source_path == str(path.resolve())
    assert record.canonical_source_path == observed.canonical_source_path
    # A cursor records a profile key only where acquisition declares one
    # (``declares_profile_identity``); a Codex rollout declares none.
    assert not declares_profile_identity(Provider.CODEX)
    assert record.captured_profile_key is None
    assert _unauthorized_offsets(tmp_path) == []


def test_observed_authority_names_the_physical_file_behind_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.jsonl"
    target.write_bytes(_ROLLOUT)
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(target)

    assert CursorPathAuthority.observe(alias).canonical_source_path == str(target.resolve())
