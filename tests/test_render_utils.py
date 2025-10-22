import json

from datetime import datetime, timezone

from polylogue.importers.chatgpt import _inject_citations
from polylogue.render import _conversation_time_bounds, build_markdown_from_chunks


def test_inject_citations_formats_hidden_finance_reference():
    text = (
        "Close price \uE200cite\uE202turn1search6\uE201 "
        "current price \uE200cite\uE202turn0finance0\uE201"
    )
    metadata = {
        "citations": [
            {
                "metadata": {
                    "type": "webpage",
                    "title": "Example",
                    "url": "https://example.com",
                }
            }
        ],
        "content_references": [
            {
                "matched_text": "\uE200cite\uE202turn1search6\uE201",
                "safe_urls": ["https://example.com"],
                "type": "grouped_webpages",
            },
            {
                "matched_text": "\uE200cite\uE202turn0finance0\uE201",
                "safe_urls": [],
                "refs": ["hidden"],
                "type": "hidden",
            },
        ],
    }

    result = _inject_citations(text, metadata, {})
    assert "[^cite1]" in result
    assert "[^cite2]" in result
    assert "Live market quote (tool result)" in result


def test_conversation_time_bounds_and_metadata_roundtrip():
    chunks = [
        {"role": "user", "text": "Hi", "timestamp": 0},
        {"role": "model", "text": "Hello", "timestamp": "2024-01-01T00:00:00"},
        {"role": "model", "text": "Later", "timestamp": "2024-01-02T12:00:00Z"},
    ]
    first, last = _conversation_time_bounds(chunks)
    assert first == "1970-01-01T00:00:00Z"
    assert last == "2024-01-02T12:00:00Z"

    doc = build_markdown_from_chunks(
        chunks,
        per_chunk_links={},
        title="Test",
        source_file_id="conv",
        modified_time=None,
        created_time=None,
        collapse_threshold=25,
        run_settings=None,
        citations=None,
        source_mime=None,
        source_size=None,
        extra_yaml=None,
        attachments=None,
    )
    assert doc.metadata["firstMessageTime"] == "1970-01-01T00:00:00Z"
    assert doc.metadata["lastMessageTime"] == "2024-01-02T12:00:00Z"
    expected_duration = int(
        datetime.fromisoformat("2024-01-02T12:00:00+00:00").timestamp()
        - datetime.fromtimestamp(0, tz=timezone.utc).timestamp()
    )
    assert doc.metadata["conversationDurationSeconds"] == expected_duration
