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


def test_claude_workflow_artifacts_follow_origin_spec_path_rules() -> None:
    cases = {
        "/tmp/.claude/projects/x/workflows/wf-run.json": (ArtifactKind.WORKFLOW_RUN_SNAPSHOT, False),
        "/tmp/.claude/projects/x/subagents/workflows/wf-run/journal.jsonl": (ArtifactKind.WORKFLOW_JOURNAL, False),
        "/tmp/.claude/projects/x/subagents/agent-a.jsonl": (ArtifactKind.AGENT_TRANSCRIPT, True),
        "/tmp/.claude/projects/x/subagents/agent-a.meta.json": (ArtifactKind.AGENT_SIDECAR_META, False),
        "/tmp/.claude/projects/x/jobs/session-a/adopt.json": (ArtifactKind.ADOPT_MANIFEST, False),
    }

    for path, (kind, parse_as_session) in cases.items():
        artifact = classify_artifact({}, provider="claude-code", source_path=path)
        assert artifact.kind is kind
        assert artifact.parse_as_session is parse_as_session
