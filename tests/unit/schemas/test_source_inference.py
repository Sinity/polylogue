"""Declared-source schema inference regressions."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence
from polylogue.schemas.generation.workflow import generate_provider_schema_from_sources
from polylogue.schemas.source_inference import SchemaSourceInput, infer_sources


def test_declared_claude_jsonl_source_reaches_evidence_schema_emission(tmp_path: Path) -> None:
    """Anti-vacuity: bypassing source inference leaves the generated schema empty."""
    source_root = tmp_path / "claude"
    source_root.mkdir()
    (source_root / "session.jsonl").write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "synthetic-session",
                "version": "1.0.0",
                "message": {"role": "user", "content": "synthetic"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_provider_schema_from_sources(
        "claude-code",
        source_inputs=(SchemaSourceInput("claude-code", source_root),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
        privacy_config=None,
    )

    assert result.success
    assert result.sample_count > 0
    assert result.schema is not None
    assert result.schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert result.phase_receipt["source"]["source_terminal_outcomes"] == {"included": 1}


def test_declared_json_array_accepts_fractional_values_and_one_file_source(tmp_path: Path) -> None:
    """Anti-vacuity: Decimal values from the streaming decoder used to reject an otherwise valid export."""
    source = tmp_path / "session.json"
    source.write_text(
        json.dumps(
            [
                {
                    "type": "user",
                    "sessionId": "fractional-source",
                    "version": "1.2.3",
                    "timestamp": 1.25,
                    "message": {"role": "user", "content": "synthetic"},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = infer_sources(
        (SchemaSourceInput("claude-code", source),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
    )

    assert result.terminal_counts == {"included": 1}
    assert result.producer_version_counts == {"1.2.3": 1}


def test_identical_chatgpt_export_reacquisitions_count_one_native_source(tmp_path: Path) -> None:
    """Anti-vacuity: path-keyed duplicate collection doubles record denominators."""
    source_root = tmp_path / "chatgpt"
    source_root.mkdir()
    conversation = {
        "id": "synthetic-chatgpt-conversation",
        "conversation_id": "synthetic-chatgpt-conversation",
        "mapping": {
            "root": {"id": "root", "parent": None, "message": None},
            "user": {
                "id": "user",
                "parent": "root",
                "message": {
                    "id": "user",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "create_time": 1.25,
                },
            },
        },
    }
    for name in ("first.json", "reacquired.json"):
        (source_root / name).write_text(json.dumps([conversation]), encoding="utf-8")

    result = infer_sources(
        (SchemaSourceInput("chatgpt", source_root),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(row) for rows in result.evidence_by_element.values() for row in rows
    )

    assert result.terminal_counts == {"included": 1}
    assert evidence.current_source_count == 1
