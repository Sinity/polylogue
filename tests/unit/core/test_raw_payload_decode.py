from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.archive.raw_payload.decode import (
    JSONValue,
    _sample_jsonl_payload_with_detail,
    build_raw_payload_envelope,
    sample_jsonl_payload,
)


def _as_dict(sample: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(sample, dict)
    return sample


def _optional_int(value: JSONValue) -> int | None:
    assert value is None or isinstance(value, int)
    return value


def test_sample_jsonl_payload_accepts_lone_surrogates_via_stdlib_fallback(tmp_path: Path) -> None:
    path = tmp_path / "surrogate.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"text":"broken \\udce2 surrogate"}\n{"ok": 2}\n')

    samples, malformed = sample_jsonl_payload(path, max_samples=8, jsonl_dict_only=True)

    dict_samples = [_as_dict(sample) for sample in samples]

    assert malformed == 0
    assert [_optional_int(sample.get("ok")) for sample in dict_samples if "ok" in sample] == [1, 2]
    assert any(sample.get("text") == "broken \udce2 surrogate" for sample in dict_samples)


def test_raw_json_payload_preserves_utf8_encoded_lone_surrogates(tmp_path: Path) -> None:
    """Historical provider bytes may encode a lone UTF-16 surrogate directly."""
    path = tmp_path / "surrogate.json"
    path.write_bytes(b'{"text":"broken \xed\xa0\x80 provider value"}')

    envelope = build_raw_payload_envelope(
        path,
        source_path=str(path),
        fallback_provider="hermes",
    )

    payload = _as_dict(envelope.payload)
    assert payload["text"] == "broken \ud800 provider value"


def test_build_raw_payload_envelope_reports_first_bad_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\n{"broken": \n{"ok": 2}\n', encoding="utf-8")

    envelope = build_raw_payload_envelope(
        path,
        source_path=str(path),
        fallback_provider="codex",
        jsonl_dict_only=True,
    )

    assert envelope.malformed_jsonl_lines == 1
    assert envelope.malformed_jsonl_detail is not None
    assert "line 2" in envelope.malformed_jsonl_detail


def test_jsonl_sampling_can_stop_after_bounded_prefix(tmp_path: Path) -> None:
    path = tmp_path / "bounded.jsonl"
    path.write_text('{"ok": 1}\n{"ok": 2}\n{"broken": \n', encoding="utf-8")

    samples, malformed, detail = _sample_jsonl_payload_with_detail(
        path,
        max_samples=2,
        jsonl_dict_only=True,
        scan_full=False,
    )

    assert [_as_dict(sample)["ok"] for sample in samples] == [1, 2]
    assert malformed == 0
    assert detail is None


def _write_plain_sqlite_db(path: Path) -> None:
    """A genuine SQLite database with no Hermes state.db/verification_evidence.db shape.

    Stands in for the audit's concrete miscaptures (Codex ``state_5.sqlite``,
    or any other stray ``.db`` file) -- real SQLite magic bytes, but no
    dedicated, content-verified session parser claims it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE unrelated_thing (id INTEGER PRIMARY KEY, value TEXT);")
        conn.execute("INSERT INTO unrelated_thing (value) VALUES ('not a session')")
        conn.commit()


def test_build_raw_payload_envelope_refuses_unrecognized_sqlite_path(tmp_path: Path) -> None:
    """polylogue-hbtj2: a SQLite-shaped path with no dedicated parser must never be
    handed to a session parser -- refused before any JSON decode is attempted,
    classified as a binary-database artifact instead of a decode failure."""
    path = tmp_path / "state_5.sqlite"
    _write_plain_sqlite_db(path)

    envelope = build_raw_payload_envelope(
        path,
        source_path=str(path),
        fallback_provider="codex",
    )

    assert envelope.artifact.kind is ArtifactKind.BINARY_DATABASE
    assert envelope.artifact.parse_as_session is False
    assert envelope.artifact.schema_eligible is False
    assert isinstance(envelope.payload, dict)
    assert envelope.payload.get("polylogue_artifact") == "unrecognized_binary_artifact"
    assert envelope.payload.get("binary_format") == "sqlite"


def test_build_raw_payload_envelope_refuses_unrecognized_sqlite_bytes(tmp_path: Path) -> None:
    """Same refusal for in-memory bytes (the replay/backfill call shape)."""
    path = tmp_path / "state_5.sqlite"
    _write_plain_sqlite_db(path)
    raw_bytes = path.read_bytes()

    envelope = build_raw_payload_envelope(
        raw_bytes,
        source_path=str(path),
        fallback_provider="codex",
    )

    assert envelope.artifact.kind is ArtifactKind.BINARY_DATABASE
    assert envelope.artifact.parse_as_session is False


def test_build_raw_payload_envelope_still_recognizes_genuine_hermes_state_db(tmp_path: Path) -> None:
    """Regression guard: this bead tightens detection, it does not retire the
    existing, content-verified Hermes state.db session parser."""
    path = tmp_path / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, model_config TEXT,
                parent_session_id TEXT, started_at REAL, ended_at REAL, title TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_name TEXT, tool_calls TEXT,
                timestamp REAL NOT NULL, observed INTEGER DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0);
            """
        )
        conn.commit()

    envelope = build_raw_payload_envelope(
        path,
        source_path=str(path),
        fallback_provider="hermes",
    )

    assert envelope.artifact.kind is ArtifactKind.SESSION_DOCUMENT
    assert envelope.artifact.parse_as_session is True
