"""Membership-keyed sessions keep lineage-aware replay order.

One Claude Code transcript holds two sessions, so the census governs both
through memberships rather than the raw's revision key. The child claims the
parent through its fork-context record. The parent's id sorts last, so only
the lineage schedule puts it first.

Anti-vacuity: schedule only the byte-typed rebuild keys (the membership keys
then reach the schedule as an empty set), or rank representatives only by
``raw_sessions.logical_source_key`` (the child then claims no parent), and the
child is no longer scheduled after its parent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import polylogue.sources.revision_backfill as revision_backfill
from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_replay import replay_retained_components

_PARENT = "claude-code-session:zparent"
_CHILD = "claude-code-session:achild"


def _record(session: str, uuid: str, parent: str | None, role: str, text: str) -> dict[str, object]:
    content: object = text if role == "user" else [{"type": "text", "text": text}]
    return {
        "type": role,
        "message": {"role": role, "content": content},
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session,
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _bundle() -> bytes:
    records: list[dict[str, object]] = [
        _record("zparent", "p-u", None, "user", "question"),
        _record("zparent", "p-a", "p-u", "assistant", "answer"),
        {
            "type": "fork-context-ref",
            "sessionId": "achild",
            "parentSessionId": "zparent",
            "parentLastUuid": "p-a",
            "uuid": "c-ref",
            "timestamp": "2026-01-01T00:00:01Z",
        },
        _record("achild", "c-u", None, "user", "follow-up"),
        _record("achild", "c-a", "c-u", "assistant", "child answer"),
    ]
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def test_membership_keyed_child_replays_after_its_parent(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=_bundle(),
            source_path="projects/bundle/two-sessions.jsonl",
            canonical_source_path="projects/bundle/two-sessions.jsonl",
            acquired_at_ms=1,
        )
        archive.commit()

    original = revision_backfill._lineage_aware_replay_schedule
    schedules: list[Any] = []

    def capture(logical_keys: set[str], *args: Any, **kwargs: Any) -> Any:
        schedule = original(logical_keys, *args, **kwargs)
        schedules.append(schedule)
        return schedule

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(revision_backfill, "_lineage_aware_replay_schedule", capture)
        replay_retained_components(tmp_path)

    with ArchiveStore.open_existing(tmp_path) as archive:
        source = archive.source_connection
        assert source is not None
        members = {str(row[0]) for row in source.execute("SELECT logical_source_key FROM raw_session_memberships")}
    assert members == {_PARENT, _CHILD}

    replays = [schedule for schedule in schedules if {_PARENT, _CHILD} <= set(schedule.order)]
    assert replays, [schedule.order for schedule in schedules]
    order = list(replays[-1].order)
    assert order.index(_PARENT) < order.index(_CHILD)
    topology = dict(replays[-1].topology)
    assert topology[_PARENT] is revision_backfill.ReplayTopologyState.ROOT
    assert topology[_CHILD] is revision_backfill.ReplayTopologyState.DESCENDANT
