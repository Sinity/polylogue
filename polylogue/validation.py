from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence


class SchemaError(ValueError):
    """Raised when provider payloads violate the expected structure."""


def ensure_chunked_prompt(payload: Any, *, source: str | None = None) -> List[Any]:
    """Ensure payload contains a chunkedPrompt.chunks list."""

    label = source or "payload"
    if not isinstance(payload, Mapping):
        raise SchemaError(f"{label}: expected an object with chunkedPrompt.chunks")

    container = payload.get("chunkedPrompt")
    if not isinstance(container, Mapping):
        raise SchemaError(f"{label}: missing chunkedPrompt object")

    chunks = container.get("chunks")
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise SchemaError(f"{label}: chunkedPrompt.chunks must be a list of messages")
    if not chunks:
        raise SchemaError(f"{label}: chunkedPrompt.chunks is empty")

    return list(chunks)


def _raise(label: str, detail: str) -> None:
    raise SchemaError(f"{label}: {detail}")


def _ensure_mapping(value: Any, label: str, path: str) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _raise(label, f"{path} must be an object")
    return value


def _validate_timestamp(value: Any, label: str, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _raise(label, f"{path} must be an ISO timestamp string")


def _validate_attachment_entry(entry: Mapping[str, Any], label: str, path: str) -> None:
    if not isinstance(entry, Mapping):
        _raise(label, f"{path} entries must be objects")
    doc_id = entry.get("id") or entry.get("driveId") or entry.get("fileId")
    if not isinstance(doc_id, str) or not doc_id.strip():
        _raise(label, f"{path} entries require an id/driveId/fileId")
    name = entry.get("name") or entry.get("filename")
    if name is not None and not isinstance(name, str):
        _raise(label, f"{path} entry names must be strings")


def _validate_attachment_list(entries: Any, label: str, path: str) -> None:
    if entries is None:
        return
    if not isinstance(entries, Sequence):
        _raise(label, f"{path} must be a list")
    for idx, entry in enumerate(entries):
        _validate_attachment_entry(entry, label, f"{path}[{idx}]")


def _validate_run_settings(run_settings: Any, label: str) -> None:
    if run_settings is None:
        return
    mapping = _ensure_mapping(run_settings, label, "runSettings")
    if mapping is None:
        return
    model = mapping.get("model")
    if model is not None and not isinstance(model, str):
        _raise(label, "runSettings.model must be a string")
    for numeric in ("temperature", "topP", "topK", "maxOutputTokens"):
        value = mapping.get(numeric)
        if value is not None and not isinstance(value, (int, float)):
            _raise(label, f"runSettings.{numeric} must be numeric")


def _validate_citations(citations: Any, label: str) -> None:
    if citations is None:
        return
    if not isinstance(citations, Sequence):
        _raise(label, "citations must be a list")
    for idx, citation in enumerate(citations):
        if isinstance(citation, str):
            continue
        if isinstance(citation, Mapping):
            uri = citation.get("uri") or citation.get("url")
            if uri is None or not isinstance(uri, str):
                _raise(label, f"citations[{idx}] must include a uri string")
            continue
        _raise(label, f"citations[{idx}] must be a string or object")


def _validate_chunk_enrichments(chunk: Mapping[str, Any], idx: int, label: str) -> None:
    timestamp = chunk.get("timestamp") or chunk.get("createTime") or chunk.get("createdTime")
    _validate_timestamp(timestamp, label, f"chunks[{idx}].timestamp")
    for key in ("driveDocument", "driveImage", "driveAttachment"):
        if key in chunk and chunk[key] is not None:
            doc = _ensure_mapping(chunk[key], label, f"chunks[{idx}].{key}")
            if doc is None:
                continue
            _validate_attachment_entry(doc, label, f"chunks[{idx}].{key}")
    if isinstance(chunk.get("attachments"), Sequence):
        for sub_idx, entry in enumerate(chunk["attachments"]):
            _validate_attachment_entry(entry, label, f"chunks[{idx}].attachments[{sub_idx}]")


def ensure_gemini_payload(payload: Any, *, source: str | None = None) -> List[Any]:
    """Validate Gemini/Drive exports including attachments and metadata."""

    label = source or "payload"
    chunks = ensure_chunked_prompt(payload, source=label)

    metadata = _ensure_mapping(payload.get("metadata"), label, "metadata")
    if metadata:
        for key in ("createTime", "createdTime", "modifiedTime", "updateTime"):
            _validate_timestamp(metadata.get(key), label, f"metadata.{key}")

    _validate_run_settings(payload.get("runSettings"), label)
    _validate_citations(payload.get("citations"), label)
    _validate_attachment_list(payload.get("attachments"), label, "attachments")

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, Mapping):
            _validate_chunk_enrichments(chunk, idx, label)

    return chunks


__all__ = ["SchemaError", "ensure_chunked_prompt", "ensure_gemini_payload"]
