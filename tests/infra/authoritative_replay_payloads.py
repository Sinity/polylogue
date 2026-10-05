"""Neutral actual Codex wire payloads for retained authoritative replay laws."""

import json


def codex_single_message_bytes(
    native_id: str, text: str, *, replayed_parent: str | None = None, tail_text: str | None = None
) -> bytes:
    metadata = {"id": native_id, "timestamp": "2026-01-01T00:00:00Z"}
    if replayed_parent is not None:
        metadata["forked_from_id"] = replayed_parent
    records: list[dict[str, object]] = [
        {"type": "session_meta", "payload": metadata},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "m0",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
    ]
    if tail_text is not None:
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": tail_text}],
                },
            }
        )
    return b"\n".join(json.dumps(record).encode() for record in records) + b"\n"
