"""Declared-source schema inference regressions."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.schemas.generation.workflow import generate_provider_schema_from_sources
from polylogue.schemas.source_inference import SchemaSourceInput


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
