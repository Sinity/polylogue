"""Direct tests for polylogue.schemas.sampling module.

Covers ProviderConfig, PROVIDERS dict, load_samples_from_db with missing DB,
and load_samples_from_sessions with JSON/JSONL fixtures.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.raw_payload import ReplayableRecordSamples
from polylogue.core.enums import Provider
from polylogue.schemas.generation.workflow import generate_provider_schema
from polylogue.schemas.observation import PROVIDERS, ProviderConfig
from polylogue.schemas.observation_runtime import _document_profile_tokens
from polylogue.schemas.sampling import iter_schema_units, load_samples_from_db, load_samples_from_sessions
from tests.infra.schema_access import schema_node

pytestmark = pytest.mark.uses_real_clock(
    "Reservoir sampling tests measure event clustering against a real epoch; production sampler uses the same wall clock by design."
)


def test_document_profile_tokens_collapse_record_identity_keys() -> None:
    # should_collapse_observed_keys() (schemas/field_stats/detection.py) now
    # also collapses a WHOLE map's keys once >=50% of them look dynamic --
    # deliberate, so a real chatgpt-shaped mapping (near-100% UUID-keyed
    # message nodes plus one literal root) collapses wholesale rather than
    # leaking a scattering of literal-looking survivors. A mostly-static map
    # with only an occasional identifier key must stay under that ratio to
    # exercise this test's actual subject: per-key collapse of just the
    # identity-shaped key, not the map-wide majority-collapse rule.
    tokens = _document_profile_tokens(
        {
            "mapping": {
                "2f5a7f5d-a809-469a-a79a-8f032618fa92": {"message": {}},
                "client-created-root": {"message": None},
                "conversation_meta": {"message": None},
                "message_root": {"message": None},
            }
        }
    )

    assert "child:mapping:*" in tokens
    assert "child:mapping:client-created-root" in tokens
    assert not any("2f5a7f5d" in token for token in tokens)


def _archive_index_db(tmp_path: Path) -> Path:
    """Initialize a split-file archive root and return its index.db path."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass
    return root / "index.db"


def _insert_raw_session(
    *,
    db_path: Path,
    origin: str,
    source_path: str,
    raw_content: bytes,
) -> str:
    """Write raw content to the blob store and register a raw_sessions row.

    Returns the generated ``raw_id``. The native ``raw_sessions`` row carries
    the 32-byte ``blob_hash`` digest (BLOB) and millisecond timestamps (#1743).
    """
    from polylogue.storage.blob_store import get_blob_store
    from polylogue.storage.sqlite.connection import open_connection

    blob_store = get_blob_store()
    hash_hex, blob_size = blob_store.write_from_bytes(raw_content)
    raw_id = f"raw-{hash_hex[:16]}"
    acquired_at_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (raw_id, origin, source_path, 0, bytes.fromhex(hash_hex), blob_size, acquired_at_ms),
        )
        conn.commit()
    return raw_id


class TestProviderConfig:
    def test_init_minimal(self) -> None:
        config = ProviderConfig(
            name=Provider.UNKNOWN,
            description="Test provider",
        )
        assert config.name is Provider.UNKNOWN
        assert config.description == "Test provider"
        assert config.db_source_name is None
        assert config.session_dir is None

    def test_init_with_db_provider(self) -> None:
        config = ProviderConfig(
            name=Provider.UNKNOWN,
            description="Test",
            db_source_name=Provider.CHATGPT,
        )
        assert config.db_source_name is Provider.CHATGPT

    def test_init_with_session_dir(self) -> None:
        path = Path.home() / ".test"
        config = ProviderConfig(
            name=Provider.UNKNOWN,
            description="Test",
            session_dir=path,
        )
        assert config.session_dir == path

    def test_defaults(self) -> None:
        config = ProviderConfig(name=Provider.UNKNOWN, description="Test")
        assert config.sample_granularity == "document"
        assert config.max_sessions is None
        assert config.record_type_key is None


class TestProvidersConfig:
    def test_all_configured_providers_present(self) -> None:
        expected = {
            "chatgpt",
            "claude-code",
            "claude-ai",
            "gemini",
            "gemini-cli",
            "hermes",
            "antigravity",
            "codex",
        }
        assert {provider.value for provider in PROVIDERS} == expected

    def test_each_has_name_and_description(self) -> None:
        for name, config in PROVIDERS.items():
            assert config.name is name
            assert isinstance(config.description, str)
            assert len(config.description) > 0

    def test_chatgpt_has_db_provider(self) -> None:
        assert PROVIDERS[Provider.CHATGPT].db_source_name is Provider.CHATGPT

    def test_claude_ai_db_name_is_canonical(self) -> None:
        # Claude AI rows are stored under the canonical provider token.
        assert PROVIDERS[Provider.CLAUDE_AI].db_source_name is Provider.CLAUDE_AI

    def test_claude_code_has_db_provider(self) -> None:
        assert PROVIDERS[Provider.CLAUDE_CODE].db_source_name is Provider.CLAUDE_CODE

    def test_gemini_has_db_provider(self) -> None:
        assert PROVIDERS[Provider.GEMINI].db_source_name is Provider.GEMINI

    def test_local_agent_sources_have_db_providers(self) -> None:
        assert PROVIDERS[Provider.GEMINI_CLI].db_source_name is Provider.GEMINI_CLI
        assert PROVIDERS[Provider.HERMES].db_source_name is Provider.HERMES
        assert PROVIDERS[Provider.ANTIGRAVITY].db_source_name is Provider.ANTIGRAVITY

    def test_codex_has_session_dir(self) -> None:
        assert PROVIDERS[Provider.CODEX].session_dir is not None

    def test_codex_record_type_key(self) -> None:
        assert PROVIDERS[Provider.CODEX].record_type_key == "type"

    def test_claude_code_record_type_key(self) -> None:
        assert PROVIDERS[Provider.CLAUDE_CODE].record_type_key == "type"

    def test_chatgpt_no_record_type_key(self) -> None:
        assert PROVIDERS[Provider.CHATGPT].record_type_key is None

    def test_document_granularity_for_chatgpt(self) -> None:
        assert PROVIDERS[Provider.CHATGPT].sample_granularity == "document"

    def test_record_granularity_for_claude_code(self) -> None:
        assert PROVIDERS[Provider.CLAUDE_CODE].sample_granularity == "record"


class TestLoadSamplesFromDb:
    def test_nonexistent_db_returns_empty(self, tmp_path: Path) -> None:
        result = load_samples_from_db("chatgpt", db_path=tmp_path / "nope.db")
        assert result == []

    def test_empty_db_returns_empty(self, tmp_path: Path) -> None:
        # Create a split-file archive with schema but no data
        db = _archive_index_db(tmp_path)
        result = load_samples_from_db("chatgpt", db_path=db)
        assert result == []

    def test_missing_source_db_tier_logs_warning_distinct_from_zero_rows(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """polylogue-es7b: a missing sibling ``source.db`` tier file previously
        yielded the exact same empty result as a genuine zero-row scan,
        silently. ``index.db`` exists here (unlike ``test_nonexistent_db_returns_empty``,
        whose outer ``db_path.exists()`` check in ``load_samples_from_db``
        short-circuits before ever reaching ``_iter_schema_units_from_db``),
        so this reaches the tier-file check inside
        ``polylogue.schemas.sampling_db._iter_schema_units_from_db`` and must
        warn instead of returning silently.
        """
        db = _archive_index_db(tmp_path)
        (db.parent / "source.db").unlink()
        assert db.exists()
        assert not (db.parent / "source.db").exists()

        with caplog.at_level(logging.WARNING, logger="polylogue.schemas.sampling_db"):
            result = load_samples_from_db("chatgpt", db_path=db)

        assert result == []
        assert any("source.db" in record.message and record.levelno == logging.WARNING for record in caplog.records), [
            record.message for record in caplog.records
        ]

    def test_nonexistent_db_with_default_path(self) -> None:
        # When db_path=None and default doesn't exist, should return []
        from unittest.mock import patch

        fake_path = Path("/nonexistent/fake/db.db")
        with patch("polylogue.schemas.sampling.index_db_path", return_value=fake_path):
            result = load_samples_from_db("chatgpt")
            assert result == []

    def test_claude_ai_reads_db_rows_stored_under_claude(self, tmp_path: Path) -> None:
        db = _archive_index_db(tmp_path)
        raw_content = json.dumps(
            [
                {
                    "uuid": "conv-1",
                    "name": "Session",
                    "summary": "Summary",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:05:00Z",
                    "account": {"uuid": "acct-1"},
                    "chat_messages": [],
                }
            ]
        ).encode("utf-8")

        _insert_raw_session(
            db_path=db,
            origin="claude-ai-export",
            source_path="/tmp/sessions.json",
            raw_content=raw_content,
        )

        result = load_samples_from_db("claude-ai", db_path=db)
        assert len(result) == 1
        assert result[0]["uuid"] == "conv-1"

    def test_record_provider_sampling_streams_without_full_envelope(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db = _archive_index_db(tmp_path)
        raw_content = "\n".join(
            json.dumps(record)
            for record in [
                {"type": "session_meta", "id": "sess-1"},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 2048}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            ]
        ).encode("utf-8")

        _insert_raw_session(
            db_path=db,
            origin="codex-session",
            source_path="/tmp/session.jsonl",
            raw_content=raw_content,
        )

        monkeypatch.setattr(
            "polylogue.schemas.sampling.build_raw_payload_envelope",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not decode full payload")),
        )

        result = load_samples_from_db("codex", db_path=db, max_samples=2)
        assert len(result) == 2
        assert {sample["type"] for sample in result} == {"session_meta", "message"}
        message_sample = next(sample for sample in result if sample["type"] == "message")
        content = message_sample.get("content")
        assert isinstance(content, list)
        first_block = schema_node(content[0]) if content else {}
        text = first_block.get("text")
        assert isinstance(text, str)
        assert len(text) == 1024

    def test_full_corpus_record_sampling_replays_every_record_without_materializing(
        self,
        tmp_path: Path,
    ) -> None:
        db = _archive_index_db(tmp_path)
        record_count = 1_024
        raw_content = "\n".join(
            json.dumps(
                {"type": "session_meta", "id": "sess-1"}
                if index == 0
                else {
                    "type": "message",
                    "role": "user" if index % 2 else "assistant",
                    "content": [{"type": "input_text", "text": f"message-{index}"}],
                }
            )
            for index in range(record_count)
        ).encode("utf-8")
        _insert_raw_session(
            db_path=db,
            origin="codex-session",
            source_path="/tmp/large-session.jsonl",
            raw_content=raw_content,
        )

        units = list(iter_schema_units("codex", db_path=db, full_corpus=True))

        assert len(units) == 1
        samples = units[0].schema_samples
        assert isinstance(samples, ReplayableRecordSamples)
        assert len(samples) == record_count
        assert samples[0]["type"] == "session_meta"
        assert samples[-1]["type"] == "message"

        generation = generate_provider_schema("codex", db_path=db, full_corpus=True)

        assert generation.success, generation.error
        assert generation.sample_count == record_count

    def test_document_provider_streams_compacted_values_into_envelope(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.schemas import sampling

        db = _archive_index_db(tmp_path)
        raw_content = json.dumps(
            {
                "title": "conversation",
                "mapping": {
                    "node-1": {
                        "id": "node-1",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["x" * 4096]},
                        },
                    }
                },
            }
        ).encode("utf-8")
        _insert_raw_session(
            db_path=db,
            origin="chatgpt-export",
            source_path="/tmp/conversation.json",
            raw_content=raw_content,
        )
        original_build = cast(Callable[..., object], sampling.build_raw_payload_envelope)
        observed_payloads: list[object] = []

        def _capture_payload(raw: object, **kwargs: object) -> object:
            observed_payloads.append(raw)
            return original_build(raw, **kwargs)

        monkeypatch.setattr(sampling, "build_raw_payload_envelope", _capture_payload)

        result = load_samples_from_db("chatgpt", db_path=db)

        assert len(result) == 1
        assert len(observed_payloads) == 1
        payload = schema_node(observed_payloads[0])
        mapping = schema_node(payload["mapping"])
        node = schema_node(mapping["node-1"])
        message = schema_node(node["message"])
        content = schema_node(message["content"])
        parts = content["parts"]
        assert isinstance(parts, list)
        assert parts == ["x" * 1024]
        assert "node-1" in mapping

    def test_schema_observation_records_included_and_decode_failed_raws(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db = _archive_index_db(tmp_path)
        good_raw_id = _insert_raw_session(
            db_path=db,
            origin="codex-session",
            source_path="/tmp/good.jsonl",
            raw_content=b'{"type":"session_meta","payload":{"id":"sess-1"}}\n',
        )
        bad_raw_id = _insert_raw_session(
            db_path=db,
            origin="codex-session",
            source_path="/tmp/bad.jsonl",
            raw_content=b"not-json\n",
        )
        monkeypatch.setattr(
            "polylogue.schemas.sampling.build_raw_payload_envelope",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not provider JSON")),
        )
        outcomes: list[dict[str, object]] = []

        units = list(
            iter_schema_units(
                "codex",
                db_path=db,
                full_corpus=True,
                terminal_recorder=lambda **outcome: outcomes.append(outcome),
            )
        )

        assert len(units) == 1
        assert {outcome["raw_id"]: outcome["status"] for outcome in outcomes} == {
            good_raw_id: "included",
            bad_raw_id: "decode_failed",
        }
        assert {outcome["raw_id"]: outcome["reason"] for outcome in outcomes} == {
            good_raw_id: "observed_schema_units",
            bad_raw_id: "payload_decode_failed",
        }

    def test_schema_observation_skips_duplicate_blob_content(
        self,
        tmp_path: Path,
    ) -> None:
        """Byte-identical raws (same blob_hash, distinct raw_id) are scanned once.

        Re-acquiring an unchanged file or a duplicate export legitimately
        produces two ``raw_sessions`` rows over the same content-addressed
        blob. Schema inference gets no new information from decoding that
        content twice, and on the live archive this duplication is
        substantial (13-38% of per-origin bytes, up to ~24GB for
        codex-session -- polylogue-el374): the second occurrence must be
        skipped without opening its raw content a second time.
        """
        from polylogue.storage.blob_store import get_blob_store
        from polylogue.storage.sqlite.connection import open_connection

        db = _archive_index_db(tmp_path)
        # A record-granularity codex session needs at least one message-shaped
        # record to be schema-eligible -- a lone ``session_meta`` line (as used
        # by other tests in this file that only assert on decode/terminal
        # bookkeeping) does not register any schema unit, which would make
        # ``iter_schema_units`` fall through to its real
        # ``~/.codex/sessions`` directory-scan fallback and defeat this test's
        # isolation.
        raw_content = (
            b"\n".join(
                [
                    b'{"type":"session_meta","id":"sess-1"}',
                    b'{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}',
                ]
            )
            + b"\n"
        )
        blob_store = get_blob_store()
        hash_hex, blob_size = blob_store.write_from_bytes(raw_content)
        acquired_at_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        first_raw_id = "raw-dup-first"
        second_raw_id = "raw-dup-second"
        with open_connection(db) as conn:
            for raw_id, source_path in (
                (first_raw_id, "/tmp/first.jsonl"),
                (second_raw_id, "/tmp/second.jsonl"),
            ):
                conn.execute(
                    """
                    INSERT INTO raw_sessions (
                        raw_id, origin, source_path, source_index,
                        blob_hash, blob_size, acquired_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (raw_id, "codex-session", source_path, 0, bytes.fromhex(hash_hex), blob_size, acquired_at_ms),
                )
            conn.commit()

        outcomes: list[dict[str, object]] = []
        units = list(
            iter_schema_units(
                "codex",
                db_path=db,
                full_corpus=True,
                terminal_recorder=lambda **outcome: outcomes.append(outcome),
            )
        )

        assert len(units) == 1
        assert units[0].raw_id == first_raw_id
        outcomes_by_raw_id = {outcome["raw_id"]: outcome for outcome in outcomes}
        assert outcomes_by_raw_id[first_raw_id]["status"] == "included"
        assert outcomes_by_raw_id[second_raw_id] == {
            "raw_id": second_raw_id,
            "status": "intentionally_excluded",
            "artifact_kind": None,
            "source_path": "/tmp/second.jsonl",
            "reason": "duplicate_blob_content",
        }

    def test_schema_observation_excludes_hermes_sqlite_evidence_before_json_decode(self, tmp_path: Path) -> None:
        db = _archive_index_db(tmp_path)
        raw_id = _insert_raw_session(
            db_path=db,
            origin="hermes-session",
            source_path="/tmp/hermes/verification_evidence.db",
            raw_content=b"SQLite format 3\x00\x10\x00\x02\x02",
        )
        outcomes: list[dict[str, object]] = []

        units = list(
            iter_schema_units(
                "hermes",
                db_path=db,
                full_corpus=True,
                terminal_recorder=lambda **outcome: outcomes.append(outcome),
            )
        )

        assert all(unit.raw_id != raw_id for unit in units)
        target_outcomes = [outcome for outcome in outcomes if outcome["raw_id"] == raw_id]
        assert target_outcomes == [
            {
                "raw_id": raw_id,
                "status": "intentionally_excluded",
                "artifact_kind": "metadata_document",
                "source_path": "/tmp/hermes/verification_evidence.db",
                "reason": "artifact_taxonomy:Hermes SQLite evidence sidecar",
            }
        ]


class TestLoadSamplesFromSessions:
    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        result = load_samples_from_sessions(tmp_path / "nope")
        assert result == []

    def test_reads_jsonl_files(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        records = [
            {"type": "session_meta", "id": "s1"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 2
        assert result[0]["type"] == "session_meta"
        assert result[1]["type"] == "message"

    def test_reads_json_session_documents(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        json_file = session_dir / "session-001.json"
        json_file.write_text(json.dumps({"sessionId": "s1", "messages": [{"type": "user", "content": "hi"}]}))

        result = load_samples_from_sessions(session_dir)
        assert result == [{"sessionId": "s1", "messages": [{"type": "user", "content": "hi"}]}]

    def test_multiple_jsonl_files(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        for i in range(3):
            jsonl_file = session_dir / f"session-{i:03d}.jsonl"
            records = [{"type": "message", "n": j} for j in range(2)]
            jsonl_file.write_text("\n".join(json.dumps(r) for r in records))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 6  # 3 files * 2 records each

    def test_max_samples_limits_output(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        records = [{"type": "message", "n": i} for i in range(100)]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records))

        result = load_samples_from_sessions(session_dir, max_samples=5)
        assert len(result) <= 5

    def test_malformed_jsonl_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        # Valid, invalid, valid on three lines
        jsonl_file.write_text('{"valid": true}\n{invalid json\n{"also_valid": true}\n')

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 2  # Only valid lines loaded

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        lines = [
            '{"a": 1}',
            "",  # empty
            '{"b": 2}',
            "   ",  # whitespace
            '{"c": 3}',
        ]
        jsonl_file.write_text("\n".join(lines))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 3

    def test_non_dict_json_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        lines = [
            '{"valid": true}',
            "[1, 2, 3]",  # array, not dict
            '"string"',  # string
            '{"also_valid": true}',
        ]
        jsonl_file.write_text("\n".join(lines))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 2

    def test_max_sessions_limits_files(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        # Create 5 session files
        for i in range(5):
            jsonl_file = session_dir / f"session-{i:03d}.jsonl"
            jsonl_file.write_text('{"type": "message"}\n')

        # Request only 2 sessions
        result = load_samples_from_sessions(session_dir, max_sessions=2)
        # Should have records from only 2 files
        assert len(result) <= 2

    def test_nested_session_dirs(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        subdir = session_dir / "subdir"
        subdir.mkdir()

        # File in root
        (session_dir / "session-001.jsonl").write_text('{"a": 1}\n')
        # File in subdir
        (subdir / "session-002.jsonl").write_text('{"b": 2}\n')

        result = load_samples_from_sessions(session_dir)
        # rglob should find both
        assert len(result) == 2

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        # Valid file
        (session_dir / "session-001.jsonl").write_text('{"valid": true}\n')
        # Unreadable file (by chmod)
        bad_file = session_dir / "session-002.jsonl"
        bad_file.write_text('{"also_valid": true}\n')
        bad_file.chmod(0o000)

        try:
            result = load_samples_from_sessions(session_dir)
            # Should have at least the readable one
            assert len(result) >= 1
        finally:
            # Restore permissions for cleanup
            bad_file.chmod(0o644)


class TestReplayableRecordSamplesBoundedSlice:
    """Regression coverage for polylogue-5svw0.

    ``ReplayableRecordSamples.__bool__`` (PR #3546) already answers
    emptiness with a bounded peek. Its ``__getitem__`` slice path, however,
    still resolved every slice via ``index.indices(len(self))`` -- forcing a
    full decode of the backing file even for a forward-bounded slice like
    ``samples[:5]``, the exact shape ``observation_runtime.py`` had to route
    around with ``islice`` + a warning comment. This verifies the bounded
    ``samples[:N]`` shape no longer decodes the whole file.
    """

    @staticmethod
    def _large_raw_content(record_count: int) -> str:
        return "\n".join(json.dumps({"type": "message", "n": index}) for index in range(record_count))

    def test_forward_bounded_slice_does_not_decode_full_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import polylogue.archive.raw_payload.sampling_extract as sampling_extract_module

        decode_calls = {"n": 0}
        original_loads = sampling_extract_module.loads  # type: ignore[attr-defined]

        def counting_loads(text: str) -> object:
            decode_calls["n"] += 1
            return original_loads(text)

        monkeypatch.setattr(sampling_extract_module, "loads", counting_loads)

        record_count = 20_000
        raw_content = self._large_raw_content(record_count)
        samples = ReplayableRecordSamples(raw_content)

        head = samples[:5]

        assert [item["n"] for item in head] == [0, 1, 2, 3, 4]
        # A bounded forward slice must not decode anywhere near the full
        # 20,000-record backing content -- only the first 5 lines.
        assert decode_calls["n"] <= 6, decode_calls["n"]

    def test_bool_still_does_not_decode_full_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import polylogue.archive.raw_payload.sampling_extract as sampling_extract_module

        decode_calls = {"n": 0}
        original_loads = sampling_extract_module.loads  # type: ignore[attr-defined]

        def counting_loads(text: str) -> object:
            decode_calls["n"] += 1
            return original_loads(text)

        monkeypatch.setattr(sampling_extract_module, "loads", counting_loads)

        record_count = 20_000
        raw_content = self._large_raw_content(record_count)
        samples = ReplayableRecordSamples(raw_content)

        assert bool(samples) is True
        assert decode_calls["n"] <= 2, decode_calls["n"]

    def test_negative_index_still_correct_via_full_scan(self) -> None:
        record_count = 25
        raw_content = self._large_raw_content(record_count)
        samples = ReplayableRecordSamples(raw_content)

        assert samples[-1]["n"] == record_count - 1

    def test_slice_with_negative_bound_still_correct(self) -> None:
        record_count = 25
        raw_content = self._large_raw_content(record_count)
        samples = ReplayableRecordSamples(raw_content)

        assert [item["n"] for item in samples[-3:]] == [22, 23, 24]
