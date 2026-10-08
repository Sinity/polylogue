"""Final retained publication admits its prepared hash and identities unchanged."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import polylogue.pipeline.ids as ids
import polylogue.sources.revision_backfill as replay
from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers import revision_governance, write
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


def _payload() -> bytes:
    lines = []
    for index, (role, text) in enumerate((("user", "hello"), ("assistant", "hi there"), ("user", "hello"))):
        lines.append(
            b'{"type":"%s","uuid":"u%d","sessionId":"carrier","parentUuid":%s,"cwd":"/neutral",'
            b'"timestamp":"2026-01-01T00:00:0%dZ","message":{"role":"%s","content":"%s"}}\n'
            % (
                role.encode(),
                index,
                b"null" if index == 0 else b'"u%d"' % (index - 1),
                index,
                role.encode(),
                text.encode(),
            )
        )
    return b"".join(lines)


@pytest.mark.asyncio
async def test_final_retained_writer_admits_original_prepared_identity_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=_payload(),
                source_path="session.jsonl",
                canonical_source_path="session.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)
    original = replay.apply_prepared_revision_replay
    carriers: list[tuple[str, bytes, tuple[tuple[str, int], ...]]] = []

    def fatal(name: str) -> Callable[..., NoReturn]:
        def refuse(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError(f"final writer rebuilt {name}")

        return refuse

    def publish(*args: Any, **kwargs: Any) -> Any:
        prepared = cast(Mapping[tuple[str, str], PreparedSessionWrite], kwargs["prepared_writes"])
        assert len(prepared) == 1
        [carrier] = prepared.values()
        identities = tuple(carrier.rows.content_identities)
        assert len(identities) == 3
        assert carrier.input_content_hash == carrier.rows.session_content_hash
        carriers.append((carrier.session_id, carrier.rows.session_content_hash, identities))
        with monkeypatch.context() as writer_patch:
            for module, name in (
                (ids, "session_content_hash"),
                (revision_governance, "session_content_hash"),
                (write, "message_content_identities"),
                (write, "disk_message_content_identities"),
                (write, "_build_message_rows"),
                (write, "_build_block_rows"),
            ):
                writer_patch.setattr(module, name, fatal(name))
            return original(*args, **kwargs)

    monkeypatch.setattr(replay, "apply_prepared_revision_replay", publish)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
    assert len(carriers) == 1
    session_id, content_hash, identities = carriers[0]
    assert {key for receipt in receipts for key in receipt.changed_session_ids} == {session_id}
    with closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as conn:
        stored_hash = conn.execute("SELECT content_hash FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]
        stored = conn.execute(
            "SELECT content_identity, content_occurrence FROM messages WHERE session_id=? ORDER BY position",
            (session_id,),
        ).fetchall()
    assert bytes(stored_hash) == content_hash
    assert [tuple(row) for row in stored] == list(identities)
