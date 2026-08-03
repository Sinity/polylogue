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


def test_relationship_index_jsonl_conversation_field_is_metadata_not_session_stream() -> None:
    """Regression for polylogue-9ykn (gvgi): the REAL live-archive shape uses
    a ``"conversation"`` field, not ``"session"`` (the field name the sibling
    test above happens to use). Both must be refused -- pinning only the
    ``"session"`` spelling gave false confidence: with the real field name,
    ``looks_like_record_entry`` fell through to the generic ``"type"``-key
    match and misclassified this as a session-record stream before this fix
    (verified against a synthetic fixture matching the measured shape of
    ``conversation_relationships.jsonl``, the single largest contributor to
    the archive's empty-message rows: 96,748 of 101,765, ~95%).
    """
    records: list[JSONValue] = [
        {
            "conversation": f"conv-{index}",
            "parent": f"parent-{index}",
            "child": f"child-{index}",
            "type": "assistant" if index % 2 else "user",
            "timestamp": "2026-05-01T00:00:00.000Z",
        }
        for index in range(4)
    ]

    artifact = classify_artifact(
        records,
        provider="claude-code",
        source_path="/tmp/project/analysis/index/conversation_relationships.jsonl",
    )

    assert artifact.kind is ArtifactKind.METADATA_DOCUMENT
    assert artifact.parse_as_session is False


def test_analysis_signal_duplicate_messages_are_not_a_session() -> None:
    """bd polylogue-21qj: ``analysis/signal/high_value_messages.jsonl`` is a
    sinex-generated derivative index of "interesting" turns, copied verbatim
    out of a real conversation (per-line shape: ``file``/``timestamp``/
    ``type``/``content``, where ``type`` is the bare role word
    ``"assistant"``/``"user"``, not a genuine record envelope). Unlike
    ``conversation_relationships.jsonl``, its rows are real duplicated
    conversation text rather than structurally empty pointer rows -- but it
    must still never become its own ``claude-code-session``: the archive's
    live copy of this file materialized 8,763 duplicate messages (827,894
    words) that already exist verbatim in the session they were extracted
    from. This shape has no ``_TYPE_ENVELOPE_MARKERS``/``_RECORDISH_KEYS`` hit
    (``role`` is absent; the key is ``type``), so it falls through to the
    ``analysis/`` directory heuristic exactly like ``problems_index.jsonl``.
    """
    records: list[JSONValue] = [
        {
            "file": "bad69218-73bd-490a-869a-2b3a30bf421b.jsonl",
            "timestamp": "2025-06-13T17:40:52.056Z",
            "type": "assistant",
            "content": "Let me check the unified collector implementation for more context:",
        },
        {
            "file": "bad69218-73bd-490a-869a-2b3a30bf421b.jsonl",
            "timestamp": "2025-06-13T17:41:48.140Z",
            "type": "user",
            "content": "Search for ad-hoc solutions and pattern violations in the codebase.",
        },
    ]

    artifact = classify_artifact(
        records,
        provider="claude-code",
        source_path="/home/user/.claude/projects/x/analysis/signal/high_value_messages.jsonl",
    )

    assert artifact.kind is ArtifactKind.METADATA_DOCUMENT
    assert artifact.parse_as_session is False


def test_bare_tool_use_id_record_does_not_classify_as_session() -> None:
    """Regression for polylogue-9ykn (path rule landed for polylogue-omsw,
    see ``test_tool_result_sidecar_never_classifies_as_session_even_when_
    content_looks_like_one``): a record whose only distinguishing field is a
    ``tool_use_id``-shaped value (the real archive has 3 sessions whose
    native_id is a bare ``toolu_*`` tool-use id, evidence of a tool-result
    artifact acquired as an independent session rather than joined as a
    sidecar) must never become a session at a ``tool-results/`` path. Now
    refused by path (``TOOL_RESULT_SIDECAR``) rather than falling through
    content heuristics to UNKNOWN/METADATA_DOCUMENT -- the stronger,
    content-independent guarantee this whole family needed.
    """
    record: JSONValue = {
        "tool_use_id": "toolu_01AbCdEfGhIjKlMnOpQrStUv",
        "output": "some tool output text, not a conversation turn",
    }

    artifact = classify_artifact(
        record,
        provider="claude-code",
        source_path="/tmp/.claude/projects/x/tool-results/toolu_01AbCdEfGhIjKlMnOpQrStUv.json",
    )

    assert artifact.kind is ArtifactKind.TOOL_RESULT_SIDECAR
    assert artifact.parse_as_session is False


def test_agent_sidecar_meta_never_classifies_as_session() -> None:
    """Regression pin for polylogue-b508, re-verified as part of the
    polylogue-9ykn general classifier: an agent-*.meta.json sidecar must
    never become a session regardless of its content, only its path."""
    payload: JSONValue = {"agentId": "agent-deadbeef", "transcriptPath": "agent-deadbeef.jsonl"}

    artifact = classify_artifact(
        payload,
        provider="claude-code",
        source_path="/tmp/.claude/projects/x/subagents/agent-deadbeef.meta.json",
    )

    assert artifact.kind is ArtifactKind.AGENT_SIDECAR_META
    assert artifact.parse_as_session is False


def test_workflow_run_snapshot_never_classifies_as_session() -> None:
    """Regression pin: a workflows/wf_*.json run-snapshot record (real shape:
    top-level runId/taskId/script keys, no session/message envelope) must
    never become a session regardless of content, only its path."""
    payload: JSONValue = {"runId": "wf_54d4fb2e-841", "taskId": "wq88yulle", "script": "export const meta = {}"}

    artifact = classify_artifact(
        payload,
        provider="claude-code",
        source_path="/tmp/.claude/projects/x/workflows/wf_54d4fb2e-841.json",
    )

    assert artifact.kind is ArtifactKind.WORKFLOW_RUN_SNAPSHOT
    assert artifact.parse_as_session is False


def test_antigravity_brain_metadata_sidecar_never_schema_eligible_or_session() -> None:
    """Regression pin for polylogue-3m3de: a real ``brain/<uuid>/*.md.metadata
    .json`` sidecar (content shape verified against this machine's real
    ``~/.gemini/antigravity/brain`` corpus, 940 files, 0 leaks) must never
    become a session, AND must never be ``schema_eligible`` -- the latter is
    what keeps it out of schema-inference sampling
    (``schemas/sampling_db.py``'s ``_iter_schema_units_from_db`` already
    excludes any ``classify_artifact_path`` result with
    ``schema_eligible=False``, which this pins for antigravity specifically).

    polylogue-3m3de's live measurement (232 growing ``raw_sessions`` rows,
    zero ``.pb`` conversations acquired) is raw-tier acquisition volume --
    ``source.db`` durably retains every acquired file regardless of
    classification, by design -- not evidence that this classification gate
    is missing; this test locks in that the gate itself is intact. The
    unaddressed half is acquiring the real ``.pb`` conversations at all: the
    live daemon watcher only watches ``antigravity`` sources for
    ``.metadata.json`` (``sources/live/watcher.py``), never ``.pb`` -- a
    separate acquisition-wiring gap, not a classification one, and out of
    this fix's scope (would need the language-server RPC bridge wired into
    the live daemon, not merely a classification tightening).
    """
    real_metadata: JSONValue = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": (
            "Comprehensive audit results addressing: top-up/backfill elimination status, "
            "DB mode removal evaluation with architectural analysis, Provenance distinction "
            "justification, environment variable naming compliance check, per-crate "
            "documentation gaps, and entity resolution verification."
        ),
        "updatedAt": "2026-01-07T19:08:15.216541610Z",
    }

    artifact = classify_artifact(
        real_metadata,
        provider="antigravity",
        source_path="/tmp/.gemini/antigravity/brain/03c22aa3-8b7f-438d-baa8-d12567249cd9/comprehensive_audit.md.metadata.json",
    )

    assert artifact.kind is ArtifactKind.AGENT_SIDECAR_META
    assert artifact.parse_as_session is False
    assert artifact.schema_eligible is False


def test_tool_result_sidecar_never_classifies_as_session_even_when_content_looks_like_one() -> None:
    """Regression for polylogue-omsw: a ``tool-results/<name>`` sidecar must
    never become a session regardless of its content, only its path.

    Claude Code persists tool-call-overflow output verbatim to
    ``<session>/tool-results/<name>.<ext>`` (``sources/live/
    tool_result_sidecars.py`` joins it back to its owning ``tool_result``
    block by ``tool_use_id`` -- it is never independent conversation
    content). Content heuristics alone cannot refuse this family: a tool
    call's own output can coincidentally reproduce a genuine
    session-document shape byte-for-byte. This exact reproduction was found
    live (a real ``~/.claude/projects`` corpus scan, not a hypothetical): a
    ``tool-results/*.txt`` sidecar whose content was a real claude.ai
    export document -- some prior turn's tool call had fetched and dumped
    one -- classified as ``SESSION_DOCUMENT``/``parse_as_session=True``
    under the pre-fix content-only rules, because it has ``uuid``/
    ``title``/``messages`` exactly like a genuine claude-ai-export session.
    """
    real_export_document: JSONValue = {
        "uuid": "05c097b4-00f0-4233-a5e4-906f9b204ea3",
        "title": "Chat",
        "project": {"uuid": "3fef01aa-8a77-4acb-a9c4-cc73c8f1c7a3", "name": "Some Project"},
        "created_at": "2026-04-22T16:25:46.401856+00:00",
        "updated_at": "2026-04-23T17:10:40.966156+00:00",
        "messages": [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi there"},
        ],
    }

    # Sanity check the premise: this exact content, at a non-sidecar path,
    # really does classify as a session -- otherwise this test would pass
    # for the wrong reason.
    as_ordinary_document = classify_artifact(
        real_export_document,
        provider="claude-code",
        source_path="/tmp/.claude/projects/x/session/some_other_file.json",
    )
    assert as_ordinary_document.parse_as_session is True

    artifact = classify_artifact(
        real_export_document,
        provider="claude-code",
        source_path="/tmp/.claude/projects/x/session/tool-results/bvatzjyve.txt",
    )

    assert artifact.kind is ArtifactKind.TOOL_RESULT_SIDECAR
    assert artifact.parse_as_session is False
    assert artifact.schema_eligible is False


def test_tool_result_sidecar_hook_file_keeps_hook_event_classification() -> None:
    """``hook-*`` files under ``tool-results/`` are a distinct, already-
    tracked capture surface (raw hook stdout, polylogue-qqyg / #2781) with
    their own reliable content-shape detector -- the new tool-result-sidecar
    path rule must not shadow that classification."""
    hook_event: JSONValue = {
        "event_type": "PostToolUse",
        "session_id": "session-abc",
        "timestamp": "2026-08-01T00:00:00Z",
        "provider": "claude-code",
    }

    artifact = classify_artifact(
        hook_event,
        provider="claude-code",
        source_path="/tmp/.claude/projects/x/session/tool-results/hook-abc123.json",
    )

    assert artifact.kind is ArtifactKind.HOOK_EVENT
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
