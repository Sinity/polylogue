"""A child whose parent publishes in the same replay unit lands as its tail, in one pass.

A fork and its parent are retained together, so one replay unit publishes
both, parent first. The child was prepared against an Index without the
parent: its prepared write expected no parent and stored its whole prefix.
Publishing it after the parent landed refused as moved lineage ("prepared
replay lineage evidence changed"), failing the whole unit on every pass.

Anti-vacuity: drop the in-unit lineage deferral and the pass fails both keys
with that refusal. Defer without re-preparing in the same pass and the child
is left pending.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

_PREFIX = ["parent question", "parent answer"]


def _rollout(session_id: str, texts: list[str], *, forked_from_id: str | None = None) -> bytes:
    meta: dict[str, object] = {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}
    if forked_from_id is not None:
        meta["forked_from_id"] = forked_from_id
    records: list[dict[str, object]] = [{"type": "session_meta", "payload": meta}]
    for position, text in enumerate(texts):
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"m{position}",
                    "role": "user" if position % 2 == 0 else "assistant",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records)


@pytest.mark.asyncio
async def test_parent_and_fork_retained_together_publish_parent_then_tail(tmp_path: Path) -> None:
    def acquire() -> tuple[str, ...]:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=_rollout(native_id, texts, forked_from_id=parent),
                    source_path=f"{native_id}.jsonl",
                    canonical_source_path=f"{native_id}.jsonl",
                    acquired_at_ms=1,
                    native_id=native_id,
                )
                # The parent sorts last, so only lineage order publishes it first.
                for native_id, texts, parent in (
                    ("zparent", _PREFIX, None),
                    ("afork", [*_PREFIX, "fork tail"], "zparent"),
                )
            )
            archive.commit()
        return raw_ids

    raw_ids = await run_archive_fixture_write(tmp_path, acquire)
    # Live intake publishes everything one batch acquired as one retained unit.
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = await owner.ingest_retained_raw_ids(raw_ids)

    written = {session_id for receipt in receipts for session_id in receipt.written_session_ids}
    assert written == {"codex-session:zparent", "codex-session:afork"}
    with sqlite3.connect(tmp_path / "index.db") as conn:
        parent = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = 'codex-session:afork'"
        ).fetchone()
        link = conn.execute(
            "SELECT resolved_dst_session_id, inheritance, branch_point_message_id FROM session_links "
            "WHERE src_session_id = 'codex-session:afork' AND resolved_dst_session_id IS NOT NULL"
        ).fetchone()
        child_messages = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'codex-session:afork'"
        ).fetchone()[0]
    assert parent == ("codex-session:zparent",)
    assert link is not None and link[0] == "codex-session:zparent"
    assert link[1] == "prefix-sharing" and link[2] is not None
    # Only the divergent tail is stored; the prefix is the parent's.
    assert child_messages == 1
