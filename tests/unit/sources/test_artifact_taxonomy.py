from __future__ import annotations

from polylogue.archive.artifact_taxonomy import ArtifactKind, classify_artifact
from polylogue.core.json import JSONValue


def test_relationship_index_jsonl_is_metadata_not_session_stream() -> None:
    records: list[JSONValue] = [
        {
            "session": f"conv-{index}",
            "parent": f"parent-{index}",
            "child": f"child-{index}",
            "type": "assistant",
            "timestamp": "2026-05-01T00:00:00.000Z",
        }
        for index in range(4)
    ]

    artifact = classify_artifact(
        records,
        provider="claude-code",
        source_path="/tmp/project/analysis/index/session_relationships.jsonl",
    )

    assert artifact.kind is ArtifactKind.METADATA_DOCUMENT
    assert artifact.parse_as_session is False
