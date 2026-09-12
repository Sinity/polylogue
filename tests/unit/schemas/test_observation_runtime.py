"""Focused tests for schema-observation extraction helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.schemas.generation.evidence import SchemaEvidence
from polylogue.schemas.observation import ProviderConfig, extract_schema_units_from_payload
from polylogue.schemas.source_inference import _collect_candidate, _SourceCandidate


class TestExtractSchemaUnitsFromPayload:
    def test_record_granularity_compacts_and_profiles_samples(self) -> None:
        config = ProviderConfig(
            name=Provider.CLAUDE_CODE,
            description="Claude Code",
            sample_granularity="record",
            record_type_key="type",
            schema_sample_cap=8,
        )
        payload = [
            {"type": "session_meta", "id": "sess-1"},
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "x" * 2048}]},
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        ]

        units = extract_schema_units_from_payload(
            payload,
            source_name=Provider.CLAUDE_CODE,
            source_path="/tmp/session.jsonl",
            raw_id="raw-1",
            observed_at="2026-01-01T00:00:00Z",
            config=config,
        )

        assert len(units) == 1
        unit = units[0]
        assert unit.session_id == "raw-1"
        assert unit.bundle_scope == "session"
        assert any(token.startswith("bucket:") for token in unit.profile_tokens)
        content = unit.schema_samples[1]["content"]
        assert isinstance(content, list)
        first_block = content[0]
        assert isinstance(first_block, dict)
        text = first_block.get("text")
        assert isinstance(text, str)
        assert len(text) == 1024

    def test_document_granularity_emits_one_unit_per_document(self) -> None:
        config = ProviderConfig(
            name=Provider.CHATGPT,
            description="ChatGPT",
            sample_granularity="document",
        )
        payload = [
            {"id": "conv-1", "mapping": {"node-1": {"message": {"id": "m1"}}}},
            {"id": "conv-2", "mapping": {"node-9": {"message": {"author": {"role": "user"}}}}},
        ]

        units = extract_schema_units_from_payload(
            payload,
            source_name=Provider.CHATGPT,
            source_path="/tmp/sessions.json",
            raw_id="raw-docs",
            config=config,
        )

        assert len(units) == 2
        assert {unit.session_id for unit in units} == {"conv-1", "conv-2"}
        assert all(unit.artifact_kind == "session_document" for unit in units)

    def test_declared_sidecar_is_observed_without_session_admission(self) -> None:
        config = ProviderConfig(
            name=Provider.CLAUDE_CODE,
            description="Claude Code",
            sample_granularity="record",
            record_type_key="type",
        )
        payload = {"agent_id": "agent-1", "new_field": {"enabled": True}}
        units = extract_schema_units_from_payload(
            payload,
            source_name=Provider.CLAUDE_CODE,
            source_path="/tmp/subagents/run/agent-1.meta.json",
            raw_id="raw-sidecar",
            config=config,
        )
        assert len(units) == 1
        assert units[0].artifact_kind == "agent_sidecar_meta"
        assert units[0].schema_samples[0]["new_field"] == {"enabled": True}

    def test_declared_sidecar_reaches_source_evidence_through_normal_collection(self, tmp_path: Path) -> None:
        """A fact sidecar is not silently discarded by session admission."""
        path = tmp_path / "subagents" / "run" / "agent-1.meta.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"agent_id": "agent-1", "new_field": {"enabled": True}}), encoding="utf-8")

        collected = _collect_candidate(_SourceCandidate("claude-code", tmp_path, path, "sidecar-source"))

        assert collected.terminal.outcome == "included"
        assert len(collected.contributions) == 1
        evidence = SchemaEvidence.from_json(collected.contributions[0].evidence_by_element["agent_sidecar_meta"])
        assert "new_field" in json.dumps(evidence.structure)

    def test_database_schema_observation_includes_the_current_wal_shape(self, tmp_path: Path) -> None:
        """Observation must see committed schema changes that remain in SQLite's WAL."""
        path = tmp_path / "state_5.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT)")
            connection.execute(
                "CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT)"
            )
            connection.commit()
            connection.execute("ALTER TABLE threads ADD COLUMN observed_in_wal TEXT")
            connection.commit()

            collected = _collect_candidate(_SourceCandidate("codex", tmp_path, path, "codex-state"))
            assert collected.terminal.outcome == "included"
            assert len(collected.contributions) == 1
            evidence = SchemaEvidence.from_json(collected.contributions[0].evidence_by_element["database_schema"])
            assert "observed_in_wal" in json.dumps(evidence.structure)
        finally:
            connection.close()

    def test_codex_state_table_dispositions_and_unknown_tables_are_observable(self, tmp_path: Path) -> None:
        """Retention is per table, and a future table is never silently classified."""
        path = tmp_path / "state_5.sqlite"
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT);
                CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT);
                CREATE TABLE thread_artifacts (
                    id TEXT PRIMARY KEY, thread_id TEXT, artifact_type TEXT, identity_key TEXT, payload TEXT, created_at INTEGER
                );
                CREATE TABLE thread_dynamic_tools (
                    thread_id TEXT, position INTEGER, name TEXT, description TEXT, input_schema TEXT, defer_loading INTEGER, namespace TEXT
                );
                CREATE TABLE provider_extension (private_payload TEXT);
                """
            )

        collected = _collect_candidate(_SourceCandidate("codex", tmp_path, path, "codex-state"))
        assert collected.terminal.outcome == "included"
        evidence = SchemaEvidence.from_json(collected.contributions[0].evidence_by_element["database_schema"])
        encoded = json.dumps(evidence.structure)
        assert '"thread_artifacts"' in encoded
        assert '"retained-for-later-consumption"' in encoded
        assert '"provider_extension"' in encoded
        assert '"unrecognized"' in encoded
