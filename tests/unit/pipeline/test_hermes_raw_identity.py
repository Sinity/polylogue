"""Hermes raw acquisition identity contracts."""

from __future__ import annotations

import json
import shutil
import sqlite3
from hashlib import sha256
from pathlib import Path

from polylogue.config import Config, Source
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.pipeline.services.acquisition_records import make_raw_record
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.sources.parsers.base import RawSessionData
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


def _write_minimal_hermes_state(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,
                api_call_count INTEGER,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
                tool_calls TEXT,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions (
                id, source, model, model_config, parent_session_id, started_at, ended_at,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                reasoning_tokens, api_call_count, estimated_cost_usd, actual_cost_usd,
                cost_status, cost_source, pricing_version, billing_provider,
                billing_base_url, billing_mode, title
            ) VALUES (
                'shared-session', 'hermes', 'test-model', '{}', NULL, 1775000000.0, 1775000002.0,
                10, 20, 3, 4, 5, 2, 0.002, 0.0015,
                'estimated', 'litellm', '2026-07-10', 'openrouter',
                'https://openrouter.ai/api/v1', 'metered', 'Shared bytes'
            );
            INSERT INTO messages (
                session_id, role, content, timestamp, tool_calls, observed, active, compacted
            ) VALUES (
                'shared-session', 'user', 'same material', 1775000001.0, NULL, 0, 1, 0
            );
            """
        )


def test_non_hermes_acquisition_keeps_content_addressed_raw_identity(tmp_path: Path) -> None:
    payload = b'{"conversation_id":"unchanged"}'
    record = make_raw_record(
        RawSessionData(raw_bytes=payload, source_path="/imports/chatgpt.json"),
        "chatgpt",
        blob_root=tmp_path / "blob",
    )

    assert record.raw_id == sha256(payload).hexdigest()
    assert record.blob_hash is None


async def test_identical_hermes_profiles_persist_and_reprocess_independently(tmp_path: Path) -> None:
    first_db = tmp_path / "profile-a" / "state.db"
    second_db = tmp_path / "profile-b" / "state.db"
    _write_minimal_hermes_state(first_db)
    second_db.parent.mkdir(parents=True)
    shutil.copyfile(first_db, second_db)
    assert first_db.read_bytes() == second_db.read_bytes()

    backend = SQLiteBackend(db_path=tmp_path / "archive.db")
    sources = [
        Source(name="hermes", path=first_db),
        Source(name="hermes", path=second_db),
    ]
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=sources,
    )
    try:
        acquired = await AcquisitionService(backend=backend).acquire_sources(sources)
        assert acquired.acquired == 2
        assert len(set(acquired.raw_ids)) == 2

        records = await backend.get_raw_sessions_batch(acquired.raw_ids)
        assert len(records) == 2
        assert {record.source_path for record in records} == {str(first_db), str(second_db)}
        assert len({record.blob_hash for record in records}) == 1
        assert records[0].blob_hash is not None
        assert all(record.raw_id != record.blob_hash for record in records)
        assert list(BlobStore(tmp_path / "blob").iter_all()) == [records[0].blob_hash]

        parser = ParsingService(
            repository=SessionRepository(backend=backend),
            archive_root=tmp_path,
            config=config,
        )
        first_parse = await parser.parse_from_raw(raw_ids=acquired.raw_ids)
        assert first_parse.parse_failures == 0

        async with backend.connection() as conn:
            rows = list(
                await (
                    await conn.execute(
                        "SELECT session_id, raw_id FROM sessions WHERE origin = 'hermes-session' ORDER BY session_id"
                    )
                ).fetchall()
            )
        assert len(rows) == 2
        assert len({str(row["session_id"]) for row in rows}) == 2
        assert {str(row["raw_id"]) for row in rows} == set(acquired.raw_ids)
        session_ids = [str(row["session_id"]) for row in rows]

        async def durable_cost_payloads() -> list[dict[str, object]]:
            async with backend.connection() as conn:
                event_rows = list(
                    await (
                        await conn.execute(
                            """
                            SELECT payload_json
                            FROM session_provider_usage_events
                            WHERE provider_event_type = 'token_count'
                            ORDER BY session_id, position
                            """
                        )
                    ).fetchall()
                )
            return [json.loads(str(row["payload_json"])) for row in event_rows]

        expected_cost_provenance = {
            "estimated_cost_usd": 0.002,
            "actual_cost_usd": 0.0015,
            "cost_status": "estimated",
            "cost_source": "litellm",
            "pricing_version": "2026-07-10",
            "billing_provider": "openrouter",
            "billing_base_url": "https://openrouter.ai/api/v1",
            "billing_mode": "metered",
        }
        cost_payloads = await durable_cost_payloads()
        assert len(cost_payloads) == len(session_ids)
        assert all(
            {key: payload[key] for key in expected_cost_provenance} == expected_cost_provenance
            for payload in cost_payloads
        )

        async with backend.connection() as conn:
            await conn.execute("DELETE FROM sessions")
            await conn.commit()
            empty_row = await (await conn.execute("SELECT COUNT(*) FROM sessions")).fetchone()
        assert empty_row is not None
        assert int(empty_row[0]) == 0

        second_parse = await parser.parse_from_raw(raw_ids=acquired.raw_ids)
        assert second_parse.parse_failures == 0
        async with backend.connection() as conn:
            raw_row = await (await conn.execute("SELECT COUNT(*) FROM raw_sessions")).fetchone()
            session_row = await (await conn.execute("SELECT COUNT(*) FROM sessions")).fetchone()
        assert raw_row is not None
        assert session_row is not None
        raw_count = int(raw_row[0])
        session_count = int(session_row[0])
        assert raw_count == 2
        assert session_count == 2
        rebuilt_cost_payloads = await durable_cost_payloads()
        assert rebuilt_cost_payloads == cost_payloads
    finally:
        await backend.close()
