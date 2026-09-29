"""A child ingested before its parent does not leave duplicate prefix markers.

The child is stored whole while its parent is absent, so its accepted marker
carrier seals candidates for the replayed prefix under the child's message
ids. When the parent arrives, re-extraction hands those blocks to the parent,
whose own carrier seals the same markers under the parent's ids.
"""

from __future__ import annotations

import asyncio
import copy
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from polylogue.archive.session.branch_type import BranchType
from polylogue.config import Config
from polylogue.core.enums import BlockType, Origin, Provider, Role
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.services.ingest_batch import _core as ingest_batch_core
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.session.marker_domain import SessionMarkerDerivation
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.archive_templates import bootstrap_archive_root

_PREFIX_MARKER = "::note: shared prefix lesson"
_TAIL_MARKER = "::note: child tail lesson"


def _message(provider_id: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _payload(parsed: ParsedSession) -> SessionWritePayload:
    return SessionWritePayload(
        session_id=f"codex-session:{parsed.provider_session_id}",
        content_hash=str(session_content_hash(parsed)),
        parsed_session=parsed,
        message_count=len(parsed.messages),
    )


def _marker_rows(user_db: Path) -> dict[tuple[str, str], str]:
    with sqlite3.connect(user_db) as user:
        rows = user.execute(
            "SELECT body_text, target_ref, status FROM assertions WHERE author_kind = 'agent' ORDER BY target_ref"
        ).fetchall()
    return {(str(body), str(target)): str(status) for body, target, status in rows}


@pytest.mark.asyncio
async def test_late_parent_supersedes_the_childs_prefix_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: without carrier retirements the child's prefix marker stays a live candidate.

    Delivery would then hold two candidate assertions for the one inherited
    marker: one citing the child's deleted prefix row, one citing the parent.
    """
    bootstrap_archive_root(tmp_path)
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _message("c0", Role.USER, "hello", 0),
            _message("c1", Role.ASSISTANT, _PREFIX_MARKER, 1),
            _message("cx", Role.USER, "child diverges here", 2),
            _message("cy", Role.ASSISTANT, _TAIL_MARKER, 3),
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        messages=[
            _message("p0", Role.USER, "hello", 0),
            _message("p1", Role.ASSISTANT, _PREFIX_MARKER, 1),
            _message("p2", Role.USER, "parent continues alone", 2),
        ],
    )
    templates: dict[str, SessionWritePayload] = {}
    raw_ids: list[str] = []
    for name, parsed in (("child", child), ("parent", parent)):
        payload_bytes = f"late-parent-{name}".encode()
        BlobStore(tmp_path / "blob").write_from_bytes(payload_bytes)
        with sqlite3.connect(tmp_path / "source.db") as source:
            raw_id = write_source_raw_session(
                source,
                origin=Origin.CODEX_SESSION,
                source_path=f"late-parent-{name}.jsonl",
                source_index=0,
                payload=payload_bytes,
                acquired_at_ms=1,
            )
        raw_ids.append(raw_id)
        templates[raw_id] = _payload(parsed)

    def ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        payload = replace(copy.deepcopy(templates[record.raw_id]), raw_id=record.raw_id)
        return IngestRecordResult(
            raw_id=record.raw_id,
            payload_provider=Provider.CODEX.value,
            validation_status="passed",
            outcome_code="success",
            sessions=[payload],
        )

    monkeypatch.setattr(ingest_batch_core, "ingest_record", ingest)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(repository=repository, archive_root=tmp_path, config=config, ingest_workers=1)
    try:
        for raw_id in raw_ids:  # child first, then its parent
            await ingest_batch_core.process_ingest_batch(service, repository.backend, [raw_id], ParseResult(), None)
    finally:
        await repository.close()

    with sqlite3.connect(tmp_path / "index.db") as index:
        # The parent's arrival re-extracted the child to its divergent tail.
        assert index.execute(
            "SELECT inheritance FROM session_links WHERE src_session_id = 'codex-session:child'"
        ).fetchone() == ("prefix-sharing",)
        child_texts = {
            str(row[0]) for row in index.execute("SELECT text FROM blocks WHERE session_id = 'codex-session:child'")
        }
    assert _PREFIX_MARKER not in child_texts and _TAIL_MARKER in child_texts

    adapter = SessionMarkerDerivation(
        lambda: sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True),
        lambda: sqlite3.connect(f"file:{tmp_path / 'user.db'}?mode=ro", uri=True),
        lambda: sqlite3.connect(tmp_path / "user.db"),
    )
    frame = object()

    def deliver_all() -> int:
        # The consumer drives its own event loop, as the daemon's compute
        # worker thread does.
        delivered = 0
        while True:
            keys, _next = adapter.required_page(frame, cursor=None, limit=1)
            if not keys:
                return delivered
            assert adapter.publish(frame, adapter.compute(frame, keys[0])) is True
            delivered += 1

    delivered = await asyncio.to_thread(deliver_all)
    assert delivered == 2

    markers = _marker_rows(tmp_path / "user.db")
    live_prefix = [
        target
        for (body, target), status in markers.items()
        if body.endswith("shared prefix lesson") and status == "candidate"
    ]
    retired_prefix = [
        target
        for (body, target), status in markers.items()
        if body.endswith("shared prefix lesson") and status == "superseded"
    ]
    assert len(live_prefix) == 1 and "codex-session:parent" in live_prefix[0]
    assert len(retired_prefix) == 1 and "codex-session:child" in retired_prefix[0]
    assert [status for (body, _target), status in markers.items() if body.endswith("child tail lesson")] == [
        "candidate"
    ]


def test_retirement_delivered_before_the_child_carrier_still_holds(tmp_path: Path) -> None:
    """Anti-vacuity: a retirement that only supersedes existing rows is lost when the parent's carrier lands first.

    Source finalization does not order a batch's raws, and an interrupted
    child can finalize after its parent. The later child carrier must not
    lower the retired candidate live.
    """
    import aiosqlite

    from polylogue.markers import candidates_for_block
    from polylogue.markers.lowering import assertion_id_for_marker
    from polylogue.storage.accepted_marker_inputs import append_accepted_marker_input, prepare_accepted_marker_input
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    child_candidate = candidates_for_block("codex-session:child:c1", "codex-session:child:c1:0", _PREFIX_MARKER)[0]
    child_record = asdict(child_candidate)
    child_record["assertion_kind"] = child_candidate.assertion_kind.value if child_candidate.assertion_kind else None
    retired_id = assertion_id_for_marker(child_candidate)
    assert retired_id is not None
    parent_carrier = prepare_accepted_marker_input(
        "parent-raw",
        [{"session_id": "codex-session:parent", "candidates": [], "retired_assertions": [retired_id]}],
    )
    child_carrier = prepare_accepted_marker_input(
        "child-raw", [{"session_id": "codex-session:child", "candidates": [child_record]}]
    )

    async def append() -> None:
        async with aiosqlite.connect(tmp_path / "source.db") as conn:
            await append_accepted_marker_input(conn, parent_carrier)
            await append_accepted_marker_input(conn, child_carrier)
            await conn.commit()

    asyncio.run(append())
    adapter = SessionMarkerDerivation(
        lambda: sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True),
        lambda: sqlite3.connect(f"file:{tmp_path / 'user.db'}?mode=ro", uri=True),
        lambda: sqlite3.connect(tmp_path / "user.db"),
    )
    frame = object()
    for _ in range(2):
        keys, _next = adapter.required_page(frame, cursor=None, limit=1)
        assert adapter.publish(frame, adapter.compute(frame, keys[0])) is True

    with sqlite3.connect(tmp_path / "user.db") as user:
        live = user.execute(
            "SELECT COUNT(*) FROM assertions WHERE assertion_id = ? AND status = 'candidate'", (retired_id,)
        ).fetchone()
    assert live == (0,)
