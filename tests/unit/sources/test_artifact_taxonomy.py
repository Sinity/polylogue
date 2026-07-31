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


def test_self_generated_analysis_index_is_not_a_session() -> None:
    """bd polylogue-omsw / polylogue-9ykn: a self-generated analysis index an
    agent wrote into its own Claude Code project directory (e.g. an index of
    prior conversation ids under ``analysis/problem_solutions/``) must never
    become a session, even though its per-line records carry a generic
    "type" key that ``looks_like_record_entry`` treats as recordish. The
    directory-path exclusion is what actually saves this case -- the payload
    shape alone (no "session"/"parent"/"child" keys) does not hit the
    existing ``_RELATIONSHIP_INDEX_KEYS`` metadata branch.
    """
    records: list[JSONValue] = [
        {"conversation": f"conv-{index}", "type": "unknown", "preview": "     1->use chrono::{...}"}
        for index in range(4)
    ]

    artifact = classify_artifact(
        records,
        provider="claude-code",
        source_path="/home/user/.claude/projects/x/analysis/problem_solutions/problems_index.jsonl",
    )

    assert artifact.kind is ArtifactKind.METADATA_DOCUMENT
    assert artifact.parse_as_session is False


def test_chatgpt_codex_cloud_task_classifies_as_session_document() -> None:
    """bd polylogue-2m2e: without this branch, a codex.json task record fails
    every generic session-document heuristic (no "mapping"/"messages" list)
    AND fails looks_metadataish_dict (its "turns" list is not scalarish), so
    it fell through to UNKNOWN/parse_as_session=False and was silently
    dropped before dispatch.py's chatgpt_codex_task lowering ever ran.
    """
    task: JSONValue = {
        "archived": False,
        "id": "task_e_abc123",
        "title": "Fix a bug",
        "turns": [
            {"id": "task_e_abc123~usertrn_1", "input_items": [], "role": "user"},
            {"id": "task_e_abc123~assttrn_1", "output_items": [], "role": "assistant"},
        ],
    }

    artifact = classify_artifact(task, provider="chatgpt")

    assert artifact.kind is ArtifactKind.SESSION_DOCUMENT
    assert artifact.parse_as_session is True


def test_chatgpt_library_files_entry_is_not_a_session() -> None:
    entry: JSONValue = {"file_id": "file_abc", "file_name": "notes.md", "mime_type": "text/markdown"}

    artifact = classify_artifact(entry, provider="chatgpt")

    assert artifact.parse_as_session is False


def test_claude_workflow_artifacts_follow_origin_spec_path_rules() -> None:
    cases = {
        "/tmp/.claude/projects/x/workflows/wf-run.json": (ArtifactKind.WORKFLOW_RUN_SNAPSHOT, False),
        "/tmp/.claude/projects/x/subagents/workflows/wf-run/journal.jsonl": (ArtifactKind.WORKFLOW_JOURNAL, False),
        "/tmp/.claude/projects/x/subagents/agent-a.jsonl": (ArtifactKind.AGENT_TRANSCRIPT, True),
        "/tmp/.claude/projects/x/subagents/agent-a.meta.json": (ArtifactKind.AGENT_SIDECAR_META, False),
        "/tmp/.claude/projects/x/jobs/session-a/adopt.json": (ArtifactKind.ADOPT_MANIFEST, False),
        "/tmp/.claude/projects/x/coordinator.jsonl": (ArtifactKind.COORDINATOR_SESSION_STREAM, True),
    }

    for path, (kind, parse_as_session) in cases.items():
        artifact = classify_artifact({}, provider="claude-code", source_path=path)
        assert artifact.kind is kind
        assert artifact.parse_as_session is parse_as_session
