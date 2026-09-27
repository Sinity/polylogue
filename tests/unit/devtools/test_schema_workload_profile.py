"""The workload-profile extractor publishes aggregates only."""

from __future__ import annotations

import json
from pathlib import Path

from devtools.schema_workload_profile import measure
from polylogue.schemas.synthetic.workload import generate_workload_corpus


def _records(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def test_measured_profile_holds_only_aggregates(tmp_path: Path) -> None:
    """Anti-vacuity: a profile that copies text, ids or paths from its sources fails."""
    corpus = generate_workload_corpus(seed=2, target_sessions=8, origins={"claude-code": 1.0})
    corpus.write(tmp_path)
    source_texts = {
        value
        for path in tmp_path.rglob("*.jsonl")
        for record in _records(path.read_bytes())
        for value in (record.get("sessionId"), record.get("uuid"), record.get("cwd"))
        if isinstance(value, str)
    }
    profile = measure("claude-code", tmp_path / "claude-code" / "projects", sample=50, tail=2, seed=1)
    rendered = json.dumps(profile)
    assert profile["streams"]
    assert not [text for text in source_texts if text in rendered]
