import pytest

from polylogue.validation import SchemaError, ensure_chunked_prompt, ensure_gemini_payload


def test_ensure_chunked_prompt_accepts_valid_payload():
    payload = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}}
    chunks = ensure_chunked_prompt(payload, source="sample.json")
    assert chunks and chunks[0]["role"] == "user"


def test_ensure_chunked_prompt_rejects_missing_container():
    try:
        ensure_chunked_prompt({}, source="sample.json")
    except SchemaError as exc:
        assert "missing chunkedPrompt" in str(exc)
    else:  # pragma: no cover - defensive
        assert False, "expected SchemaError"


def test_ensure_chunked_prompt_rejects_empty_chunks():
    payload = {"chunkedPrompt": {"chunks": []}}
    try:
        ensure_chunked_prompt(payload, source="export.json")
    except SchemaError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover - defensive
        assert False, "expected SchemaError"


def test_ensure_gemini_payload_validates_metadata_and_attachments():
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "role": "user",
                    "text": "Upload",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "driveDocument": {"id": "file-1", "name": "doc.txt"},
                }
            ]
        },
        "metadata": {"createTime": "2024-01-01T00:00:00Z"},
        "attachments": [{"id": "file-2", "name": "img.png"}],
        "runSettings": {"model": "gemini-pro", "temperature": 0.25},
        "citations": ["https://example.com"],
    }
    chunks = ensure_gemini_payload(payload, source="drive.json")
    assert len(chunks) == 1
    assert chunks[0]["driveDocument"]["id"] == "file-1"


def test_ensure_gemini_payload_rejects_invalid_attachments():
    payload = {
        "chunkedPrompt": {"chunks": [{"role": "user", "text": "bad"}]},
        "attachments": [{"name": "no-id"}],
    }
    with pytest.raises(SchemaError):
        ensure_gemini_payload(payload, source="bad.json")
