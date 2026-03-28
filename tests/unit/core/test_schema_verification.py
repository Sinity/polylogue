"""Tests for raw-corpus schema verification workflow."""

from __future__ import annotations

from datetime import datetime, timezone

from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.verification import verify_raw_corpus
from polylogue.storage.backends.connection import open_connection


def _insert_raw_record(
    *,
    db_path,
    raw_id: str,
    provider_name: str,
    payload_provider: str | None = None,
    source_name: str,
    source_path: str,
    raw_content: bytes,
) -> None:
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, payload_provider, source_name, source_path, source_index,
                raw_content, acquired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                provider_name,
                payload_provider,
                source_name,
                source_path,
                0,
                raw_content,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def test_verify_raw_corpus_reports_valid_synthetic_chatgpt(db_path, monkeypatch):
    from polylogue.schemas import ValidationResult

    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload, max_samples=16):
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample):
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.verification.SchemaValidator.for_provider",
        lambda _provider: _AlwaysValidValidator(),
    )

    corpus = SyntheticCorpus.for_provider("chatgpt")
    raw = corpus.generate(count=1, seed=42)[0]
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-1",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt.json",
        raw_content=raw,
    )

    report = verify_raw_corpus(db_path=db_path, providers=["chatgpt"], max_samples=16)
    stats = report.providers["chatgpt"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.valid_records == 1
    assert stats.invalid_records == 0
    assert stats.decode_errors == 0


def test_verify_raw_corpus_counts_missing_schema_as_skipped(db_path):
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-inbox-1",
        provider_name="inbox",
        source_name="inbox",
        source_path="/tmp/inbox.json",
        raw_content=b'{"hello":"world"}',
    )

    report = verify_raw_corpus(db_path=db_path, providers=["inbox"], max_samples=16)
    stats = report.providers["inbox"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.skipped_no_schema == 1
    assert stats.valid_records == 0
    assert stats.invalid_records == 0


def test_verify_raw_corpus_uses_persisted_payload_provider_for_filters(db_path, monkeypatch):
    from polylogue.schemas import ValidationResult

    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload, max_samples=16):
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample):
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.verification.SchemaValidator.for_provider",
        lambda _provider: _AlwaysValidValidator(),
    )

    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-generic-chatgpt",
        provider_name="inbox",
        payload_provider="chatgpt",
        source_name="inbox",
        source_path="/tmp/raw.json",
        raw_content=b'{"id":"one"}',
    )

    report = verify_raw_corpus(db_path=db_path, providers=["chatgpt"], max_samples=16)

    assert report.total_records == 1
    assert report.providers["chatgpt"].total_records == 1
    assert report.providers["chatgpt"].valid_records == 1


def test_verify_raw_corpus_counts_malformed_jsonl_as_decode_error(db_path):
    raw_id = "raw-codex-1"
    _insert_raw_record(
        db_path=db_path,
        raw_id=raw_id,
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/session.jsonl",
        raw_content=(
            b'{"type":"session_meta"}\n'
            b'not json at all\n'
            b'{"type":"response_item","payload":{"type":"message"}}'
        ),
    )

    report = verify_raw_corpus(db_path=db_path, providers=["codex"], max_samples=16)
    stats = report.providers["codex"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.decode_errors == 1
    assert stats.valid_records == 0
    assert stats.invalid_records == 0
    assert stats.quarantined_records == 0

    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT validation_status, validation_error, validation_mode, parse_error "
            "FROM raw_conversations WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None


def test_verify_raw_corpus_quarantine_malformed_updates_validation_state(db_path):
    raw_id = "raw-codex-q1"
    _insert_raw_record(
        db_path=db_path,
        raw_id=raw_id,
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/session-q1.jsonl",
        raw_content=(
            b'{"type":"session_meta"}\n'
            b'not json at all\n'
            b'{"type":"response_item","payload":{"type":"message"}}'
        ),
    )

    report = verify_raw_corpus(
        db_path=db_path,
        providers=["codex"],
        max_samples=16,
        quarantine_malformed=True,
    )
    stats = report.providers["codex"]

    assert report.total_records == 1
    assert stats.decode_errors == 1
    assert stats.quarantined_records == 1

    with open_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT validation_status, validation_error, validation_mode, validation_provider,
                   payload_provider,
                   validated_at, parse_error
            FROM raw_conversations
            WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "failed"
    assert isinstance(row[1], str) and "Malformed JSONL lines" in row[1]
    assert row[2] == "strict"
    assert row[3] == "codex"
    assert row[4] == "codex"
    assert row[5] is not None
    assert isinstance(row[6], str) and "Malformed JSONL lines" in row[6]


def test_verify_raw_corpus_honors_record_limit_and_offset(db_path, monkeypatch):
    from polylogue.schemas import ValidationResult

    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload, max_samples=16):
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample):
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.verification.SchemaValidator.for_provider",
        lambda _provider: _AlwaysValidValidator(),
    )

    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-1",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt-1.json",
        raw_content=b'{"id":"one"}',
    )
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-2",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt-2.json",
        raw_content=b'{"id":"two"}',
    )

    report = verify_raw_corpus(
        db_path=db_path,
        providers=["chatgpt"],
        max_samples=16,
        record_limit=1,
        record_offset=1,
    )
    stats = report.providers["chatgpt"]

    assert report.total_records == 1
    assert report.record_limit == 1
    assert report.record_offset == 1
    assert stats.total_records == 1
    assert stats.valid_records == 1
