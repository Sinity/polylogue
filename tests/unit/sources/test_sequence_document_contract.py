"""Complete document arrays preserve origin and every accepted document."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.decoder_json import JsonValue
from polylogue.sources.dispatch import (
    ForeignOriginContentError,
    detect_provider,
    detect_provider_from_stream_evidence,
    parse_payload,
)
from polylogue.storage.blob_store import BlobStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.retained_jsonl import retained_raw_fixture
from tests.infra.sequence_document_payloads import (
    claude_document,
    gemini_document,
    grok_document,
    sequence_document_cases,
)


@pytest.mark.parametrize(
    "label,provider,document", sequence_document_cases(), ids=lambda value: value if isinstance(value, str) else None
)
def test_complete_sequence_document_after_fragment_matches_stream_and_decoded_detection(
    label: str, provider: Provider, document: dict[str, JsonValue]
) -> None:
    del label
    payload = [{"mapping": {"fragment": {"message": None}}, "metadata": "fragment only"}, document]
    assert detect_provider(payload) is provider
    detected, _evidence = detect_provider_from_stream_evidence(io.BytesIO(json.dumps(payload).encode()))
    assert detected is provider


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_sequence_refuses_a_foreign_document_in_either_order(reverse: bool) -> None:
    payload = [claude_document("claude-one"), grok_document("grok-one")]
    if reverse:
        payload.reverse()
    with pytest.raises(ForeignOriginContentError) as refused:
        detect_provider(payload)
    assert {refused.value.expected, refused.value.found} == {Provider.CLAUDE_AI, Provider.GROK}
    with pytest.raises(ForeignOriginContentError):
        parse_payload(Provider.CLAUDE_AI, payload, "neutral")


@pytest.mark.parametrize("provider", [Provider.GEMINI_CLI, Provider.GROK])
def test_retained_complete_document_array_publishes_every_same_origin_document(
    tmp_path: Path, provider: Provider
) -> None:
    build = gemini_document if provider is Provider.GEMINI_CLI else grok_document
    documents = [build("first-document"), build("second-document")]
    ordinary = parse_payload(provider, documents, "neutral")
    expected_ids = sorted(session.provider_session_id for session in ordinary)
    assert len(expected_ids) == 2 and len(set(expected_ids)) == 2
    payload = json.dumps([{"metadata": "unrelated"}, *documents]).encode()
    bootstrap_archive_root(tmp_path)
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path, provider=Provider.UNKNOWN, blob_hash=digest, source_path="complete-array.json"
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest

    async def replay() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            results = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
            assert results and all(result.quarantined == 0 for result in results)

    asyncio.run(replay())
    with closing(sqlite3.connect(tmp_path / "index.db")) as index:
        rows = index.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall()
        assert rows == [(identity,) for identity in expected_ids]
        assert index.execute("SELECT COUNT(*) FROM messages").fetchone() == (2,)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("complete",)


@pytest.mark.parametrize(
    "record",
    [
        {"mapping": {"fragment": {"message": None}}, "metadata": "fragment only"},
        {"uuid": "empty-header", "chat_messages": []},
        {"metadata": "fact only"},
    ],
)
def test_sequence_fragment_and_header_only_records_do_not_claim_complete_documents(
    record: dict[str, JsonValue],
) -> None:
    payload = [record]
    assert detect_provider(payload) is None
    assert detect_provider_from_stream_evidence(io.BytesIO(json.dumps(payload).encode()))[0] is None


def test_disk_backed_document_array_lowering_keeps_one_live_document() -> None:
    import tracemalloc
    from collections.abc import Iterator

    from polylogue.sources.decoder_json import DecodedRecordSequence
    from polylogue.sources.dispatch import iter_parsed_payload
    from tests.infra.sequence_document_payloads import large_chatgpt_document

    count = 32
    text_size = 256 * 1024

    def records() -> Iterator[JsonValue]:
        for index in range(count):
            yield large_chatgpt_document(f"large-{index}", "x" * text_size)

    with closing(DecodedRecordSequence(records())) as tape:
        tracemalloc.start()
        try:
            observed = 0
            with closing(iter_parsed_payload(Provider.CHATGPT, tape, "neutral")) as sessions:
                for session in sessions:
                    assert session.provider_session_id == f"large-{observed}"
                    assert len(session.messages) == 1
                    text = session.messages[0].text
                    assert isinstance(text, str) and len(text) == text_size
                    observed += 1
                    del session
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert observed == count
    # The original decoded cohort is eight MiB. This controlled input can
    # retain one record's parser/model buffers, but cannot retain all specs.
    assert peak < count * text_size // 2
