"""Neutral independent JSONL inputs for ordinary retained-owner controls."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def acquire_independent_codex_raws(
    root: Path, count: int, *, messages: int = 1, text_size: int = 8, namespace: str = "independent"
) -> tuple[str, ...]:
    """Acquire identity-opaque bytes, leaving census and classification to Raw."""
    raws: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for number in range(count):
            native_id = f"{namespace}-{number}"
            records = [json.dumps({"type": "session_meta", "payload": {"id": native_id}})]
            records.extend(
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "id": f"m-{index}",
                            "role": "user",
                            "content": [{"type": "input_text", "text": str(index) + "x" * text_size}],
                        },
                    }
                )
                for index in range(messages)
            )
            path = f"{native_id}.jsonl"
            raws.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=("\n".join(records) + "\n").encode(),
                    source_path=path,
                    canonical_source_path=path,
                    acquired_at_ms=1,
                )
            )
    return tuple(raws)
