"""Test helpers for generating large JSONL batches with controlled corruption.

Provides deterministic generators for:
- Valid provider JSONL records
- Single-line corruption (malformed JSON, bad UTF-8, truncation)
- Deterministic timestamp sequences
- Rerun assertion helpers
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from polylogue.lib.json import JSONDocument


def generate_valid_jsonl_record(
    index: int,
    *,
    provider: str = "codex",
    base_ts: float = 1700000000.0,
    session_id: str | None = None,
) -> JSONDocument:
    """Generate a single valid JSONL record for a given provider."""
    ts = base_ts + index * 60  # 1-minute intervals
    role = "user" if index % 2 == 0 else "assistant"

    if provider == "codex":
        return {
            "type": "message",
            "role": role,
            "id": str(uuid.UUID(int=index + 1, version=4)),
            "content": [
                {
                    "type": "input_text" if role == "user" else "output_text",
                    "text": f"Test message {index} from {role}",
                }
            ],
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        }

    if provider == "claude-code":
        return {
            "type": role,
            "uuid": str(uuid.UUID(int=index + 1, version=4)),
            "parentUuid": str(uuid.UUID(int=max(0, index), version=4)) if index > 0 else None,
            "sessionId": session_id or str(uuid.UUID(int=99, version=4)),
            "message": {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": f"Test message {index} from {role}",
                    }
                ],
            },
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        }

    raise ValueError(f"Unsupported provider: {provider}")


def generate_large_jsonl(
    count: int,
    *,
    provider: str = "codex",
    base_ts: float = 1700000000.0,
    session_id: str | None = None,
) -> list[str]:
    """Generate a list of JSONL lines (as strings).

    Returns list of JSON strings, one per record.
    """
    lines: list[str] = []
    for i in range(count):
        record = generate_valid_jsonl_record(
            i,
            provider=provider,
            base_ts=base_ts,
            session_id=session_id,
        )
        lines.append(json.dumps(record, separators=(",", ":")))
    return lines


def write_jsonl_file(
    path: Path,
    lines: list[str],
) -> None:
    """Write JSONL lines to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def corrupt_line_malformed_json(lines: list[str], index: int) -> list[str]:
    """Replace a line with malformed JSON (broken syntax)."""
    result = list(lines)
    result[index] = '{"type": "message", "broken'
    return result


def corrupt_line_truncated(lines: list[str], index: int) -> list[str]:
    """Truncate a line mid-value."""
    result = list(lines)
    original = result[index]
    result[index] = original[: len(original) // 2]
    return result


def corrupt_line_bad_utf8(lines: list[str], index: int) -> list[str]:
    """Replace a line with invalid UTF-8 bytes (as a string placeholder).

    Note: The actual bad UTF-8 must be written as bytes, not via this helper.
    This replaces the line with a marker that write_jsonl_bad_utf8 handles.
    """
    result = list(lines)
    result[index] = "__BAD_UTF8_MARKER__"
    return result


def write_jsonl_with_bad_utf8(
    path: Path,
    lines: list[str],
) -> None:
    """Write JSONL lines to a file, replacing BAD_UTF8_MARKER with actual bad bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for _i, line in enumerate(lines):
            if line == "__BAD_UTF8_MARKER__":
                f.write(b'{"type": "message", "data": "\xff\xfe\xfd"}\n')
            else:
                f.write(line.encode("utf-8") + b"\n")


def corrupt_line_wrong_envelope(lines: list[str], index: int) -> list[str]:
    """Replace a line with valid JSON but wrong provider envelope."""
    result = list(lines)
    result[index] = json.dumps(
        {
            "completely": "different",
            "structure": True,
            "no_type": "field",
        }
    )
    return result


def generate_timestamp_patterns() -> dict[str, list[JSONDocument]]:
    """Generate records with extreme timestamp patterns for chronology testing."""
    patterns: dict[str, list[JSONDocument]] = {}

    # 1970-adjacent
    patterns["epoch_near_zero"] = [_make_ts_record(i, ts=86400 + i * 3600) for i in range(5)]

    # 2038-adjacent (Unix Y2K38)
    patterns["y2038_adjacent"] = [_make_ts_record(i, ts=2147483647 - 5 * 3600 + i * 3600) for i in range(5)]

    # Far future but valid
    patterns["far_future"] = [_make_ts_record(i, ts=3000000000 + i * 3600) for i in range(5)]

    # Tomorrow
    import time

    now = time.time()
    patterns["tomorrow"] = [_make_ts_record(i, ts=now + 86400 + i * 60) for i in range(5)]

    # Mixed formats
    patterns["mixed_formats"] = [
        {
            "type": "message",
            "role": "user",
            "id": "m1",
            "timestamp": "2024-01-15T10:30:00+00:00",
            "content": [{"type": "input_text", "text": "ISO format"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "m2",
            "timestamp": 1705312200.0,
            "content": [{"type": "output_text", "text": "Epoch float"}],
        },
        {
            "type": "message",
            "role": "user",
            "id": "m3",
            "timestamp": "1705312260",
            "content": [{"type": "input_text", "text": "Epoch string"}],
        },
    ]

    # Missing timestamps alongside present ones
    patterns["missing_timestamps"] = [
        {
            "type": "message",
            "role": "user",
            "id": "m1",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "content": [{"type": "input_text", "text": "Has timestamp"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "m2",
            "content": [{"type": "output_text", "text": "No timestamp"}],
        },
        {
            "type": "message",
            "role": "user",
            "id": "m3",
            "timestamp": "2024-01-15T10:02:00+00:00",
            "content": [{"type": "input_text", "text": "Has timestamp again"}],
        },
    ]

    return patterns


def _make_ts_record(index: int, ts: float) -> JSONDocument:
    """Make a codex-shaped record with a specific timestamp."""
    role = "user" if index % 2 == 0 else "assistant"
    return {
        "type": "message",
        "role": role,
        "id": str(uuid.UUID(int=index + 1000, version=4)),
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "content": [
            {
                "type": "input_text" if role == "user" else "output_text",
                "text": f"Timestamp test {index}",
            }
        ],
    }
