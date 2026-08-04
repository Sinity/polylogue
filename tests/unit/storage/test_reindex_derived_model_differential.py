"""Derived-model differential survivor for full reindex and convergence routes.

Production dependencies exercised here:

* ``backfill_historical_revision_evidence`` is the incremental parsed-index
  route over retained raw bytes.
* ``repair_session_insights`` is the production convergence repair route.
* ``rebuild_index_from_source`` builds each owned inactive generation from
  the same durable source evidence.

The controls below mutate real temporary SQLite archives after a green route.
They prove this differential rejects omitted repair, a missing per-session
projection, stale FTS text with unchanged row count, and a stale profile with
unchanged row count.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.maintenance.replay import rebuild_index_from_source
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore, source_revision_snapshot
from polylogue.storage.repair import repair_session_insights
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.reindex_differential import (
    DerivedModelSnapshot,
    assert_derived_model_ready,
    assert_derived_models_equivalent,
    snapshot_derived_model,
)

_SESSION_ID = "chatgpt-export:derived-differential"
_CANONICAL_TOKEN = "reindexquartz"
_STALE_TOKEN = "stalequartz"
_SEARCH_QUERIES = (_CANONICAL_TOKEN, _STALE_TOKEN, "definitelyabsentderived")


def _config(root: Path, index_path: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=index_path)


def _chatgpt_payload() -> bytes:
    def node(
        native_id: str,
        role: str,
        text: str,
        *,
        parent: str | None,
        children: tuple[str, ...],
        created_at: int,
    ) -> dict[str, object]:
        return {
            "id": native_id,
            "parent": parent,
            "children": list(children),
            "message": {
                "id": native_id,
                "author": {"role": role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": created_at,
            },
        }

    rows = (
        node("root", "system", "", parent=None, children=("user",), created_at=1),
        node("user", "user", "inspect the derived-model route", parent="root", children=("assistant",), created_at=2),
        node(
            "assistant",
            "assistant",
            f"{_CANONICAL_TOKEN} canonical indexed answer",
            parent="user",
            children=(),
            created_at=3,
        ),
    )
    return json.dumps(
        [
            {
                "id": "derived-differential",
                "conversation_id": "derived-differential",
                "title": "Derived model differential",
                "create_time": 1_700_000_000,
                "update_time": 1_700_000_003,
                "current_node": "assistant",
                "mapping": {str(row["id"]): row for row in rows},
            }
        ],
        sort_keys=True,
    ).encode()


def _snapshot(root: Path, index_path: Path) -> DerivedModelSnapshot:
    return snapshot_derived_model(
        root,
        index_path,
        session_ids=(_SESSION_ID,),
        search_queries=_SEARCH_QUERIES,
    )


def _repair(root: Path, index_path: Path, generation: IndexGeneration | None = None) -> None:
    owned_generation = None if generation is None else (generation.generation_id, generation.owner_id)
    result = repair_session_insights(
        _config(root, index_path),
        dry_run=False,
        archive_root_override=root if generation is not None else None,
        owned_inactive_generation=owned_generation,
    )
    assert result.success is True
    assert result.detail == "Session insights ready"


def _rebuild_generation(
    root: Path,
    store: IndexGenerationStore,
    raw_id: str,
    *,
    owner_id: str,
) -> tuple[IndexGeneration, Path]:
    generation = store.create(owner_id=owner_id, source_snapshot=source_revision_snapshot(root))
    index_path = Path(generation.index_path)
    result = asyncio.run(
        rebuild_index_from_source(
            _config(index_path.parent, index_path),
            raw_ids=[raw_id],
            raw_batch_size=100,
            ingest_workers=1,
            materialize=True,
            progress_callback=None,
            owned_inactive_generation=(generation.generation_id, generation.owner_id),
        )
    )
    assert result["quarantined_raw_count"] == 0
    assert result["replayed_logical_source_count"] == 1
    assert store.load(generation.generation_id).state == "inactive"
    return generation, index_path


def _assert_control_rejected(expected: DerivedModelSnapshot, root: Path, index_path: Path) -> None:
    with pytest.raises(AssertionError):
        assert_derived_models_equivalent(expected, _snapshot(root, index_path))


def _replace_fts_text(index_path: Path) -> None:
    with sqlite3.connect(index_path) as conn:
        row = conn.execute(
            """
            SELECT rowid, block_id, message_id, session_id, block_type
            FROM blocks
            WHERE text LIKE ?
            """,
            (f"%{_CANONICAL_TOKEN}%",),
        ).fetchone()
        assert row is not None
        count = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (row[0],))
        conn.execute(
            """
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (*row, _STALE_TOKEN),
        )
        conn.commit()
        assert int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]) == count


def test_full_reindex_and_incremental_convergence_have_equal_derived_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two full generations and the incremental convergence route agree."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    initialize_active_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_chatgpt_payload(),
            source_path="derived-differential.json",
            acquired_at_ms=1,
        )
    incremental = backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    assert incremental.quarantined == 0
    assert incremental.replayed_logical_sources == 1
    incremental_index = tmp_path / "index.db"
    _repair(tmp_path, incremental_index)
    incremental_snapshot = _snapshot(tmp_path, incremental_index)
    assert_derived_model_ready(incremental_snapshot)

    store = IndexGenerationStore.for_archive_root(tmp_path)
    first_generation, first_index = _rebuild_generation(tmp_path, store, raw_id, owner_id="test-differential-full-a")

    # Mutation control: omitting the production repair call leaves the derived
    # model stale/partial even though raw replay completed successfully.
    _assert_control_rejected(incremental_snapshot, first_index.parent, first_index)
    _repair(first_index.parent, first_index, first_generation)
    first_snapshot = _snapshot(first_index.parent, first_index)
    assert_derived_model_ready(first_snapshot)

    second_generation, second_index = _rebuild_generation(tmp_path, store, raw_id, owner_id="test-differential-full-b")
    _repair(second_index.parent, second_index, second_generation)
    second_snapshot = _snapshot(second_index.parent, second_index)
    assert_derived_model_ready(second_snapshot)

    assert_derived_models_equivalent(incremental_snapshot, first_snapshot)
    assert_derived_models_equivalent(first_snapshot, second_snapshot)

    reference_index = tmp_path / "reference-index.db"
    shutil.copy2(second_index, reference_index)

    # Mutation control: one per-session table can disappear while the rest of
    # the materializer output remains present.
    with sqlite3.connect(second_index) as conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (_SESSION_ID,))
        conn.commit()
    _assert_control_rejected(second_snapshot, second_index.parent, second_index)
    shutil.copy2(reference_index, second_index)

    # Mutation control: stale FTS text retains the exact same number of rows.
    _replace_fts_text(second_index)
    _assert_control_rejected(second_snapshot, second_index.parent, second_index)
    shutil.copy2(reference_index, second_index)

    # Mutation control: a profile can be stale without changing any table's
    # row count, so the table and public-read comparisons must both matter.
    with sqlite3.connect(second_index) as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0])
        conn.execute("UPDATE session_profiles SET message_count = 999 WHERE session_id = ?", (_SESSION_ID,))
        conn.commit()
        assert int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]) == count
    _assert_control_rejected(second_snapshot, second_index.parent, second_index)
