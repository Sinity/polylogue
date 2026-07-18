"""Claude Code Workflow artifact admission and evidence-preserving parsing.

Workflow fact artifacts remain source-tier authorities rather than synthetic
sessions.  This module parses provider-native fields while retaining the source
path and line that support each fact.  Association and graph materialization
are separate: missing peers become coverage debt instead of filename guesses.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.origin_specs import artifact_rule_for_path

_DOCUMENT_FIELDS = frozenset(
    {
        "runId",
        "run_id",
        "workflowRunId",
        "workflow_run_id",
        "taskId",
        "task_id",
        "resumeFromRunId",
        "resume_from_run_id",
        "workflow",
        "workflowName",
        "workflow_name",
        "name",
        "scriptPath",
        "script_path",
        "scriptHash",
        "script_hash",
        "phases",
        "labels",
        "status",
        "phase",
        "progress",
        "model",
        "timing",
        "startedAt",
        "started_at",
        "completedAt",
        "completed_at",
        "tokens",
        "tools",
        "result",
        "results",
        "structuredResult",
        "structured_result",
        "finalResult",
        "final_result",
        "agentId",
        "agent_id",
        "sessionId",
        "session_id",
        "attempt",
        "attemptId",
        "attempt_id",
        "contentKey",
        "content_key",
        "callKey",
        "call_key",
        "transcriptPath",
        "transcript_path",
        "metaPath",
        "meta_path",
        "attributionAgent",
        "attribution_agent",
        "entrypoint",
        "adoptedSessionId",
        "adopted_session_id",
        "unresolved",
        "error",
    }
)
_JOURNAL_FIELDS = _DOCUMENT_FIELDS | frozenset({"type", "event", "key", "ordinal", "retryOf", "retry_of"})


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        if value := _string(payload.get(key)):
            return value
    return None


def _agent_id_from_path(source_path: str) -> str | None:
    name = Path(source_path).name
    if not name.startswith("agent-"):
        return None
    return name.removesuffix(".meta.json").removesuffix(".jsonl").removesuffix(".ndjson")


@dataclass(frozen=True, slots=True)
class ClaudeOrchestrationFact:
    """One provider-native fact plus its direct raw-artifact location."""

    kind: str
    source_path: str
    source_line: int | None
    run_id: str | None
    agent_id: str | None
    content_key: str | None
    payload: dict[str, object]

    @property
    def attempt_id(self) -> str | None:
        return _first_string(self.payload, "attemptId", "attempt_id", "attempt") or self.agent_id

    @property
    def transcript_path(self) -> str | None:
        return _first_string(self.payload, "transcriptPath", "transcript_path")

    @property
    def meta_path(self) -> str | None:
        return _first_string(self.payload, "metaPath", "meta_path")

    @property
    def is_result(self) -> bool:
        return any(
            key in self.payload
            for key in ("structuredResult", "structured_result", "result", "finalResult", "final_result")
        )


@dataclass(frozen=True, slots=True)
class ClaudeOrchestrationArtifact:
    kind: str
    source_path: str
    parse_policy: str
    facts: tuple[ClaudeOrchestrationFact, ...]
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class ClaudeOrchestrationCoverage:
    artifact_counts: dict[str, int]
    paired_agent_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    gaps: tuple[str, ...]


def parse_claude_orchestration_artifact(
    source_path: str,
    payload: bytes | str | object,
) -> ClaudeOrchestrationArtifact | None:
    """Parse one OriginSpec-declared Claude orchestration artifact.

    The source path is kept as the evidence locator.  The archive materializer
    attaches the retained raw revision ObjectRef; this parser never invents one.
    """

    rule = artifact_rule_for_path(Provider.CLAUDE_CODE, source_path)
    if rule is None:
        return None
    loaded = _decode(payload, jsonl=rule.kind == "workflow_journal")
    if rule.kind == "workflow_journal":
        records = loaded if isinstance(loaded, list) else []
        facts = tuple(
            _journal_fact(source_path, index, record)
            for index, record in enumerate(records, start=1)
            if isinstance(record, dict)
        )
    elif isinstance(loaded, dict):
        facts = (_document_fact(rule.kind, source_path, loaded),)
    else:
        facts = ()
    return ClaudeOrchestrationArtifact(rule.kind, source_path, rule.parse_policy, facts)


def inventory_claude_orchestration_artifacts(paths: Iterable[str | Path]) -> ClaudeOrchestrationCoverage:
    """Inventory declared members and report only evidence-backed gaps."""

    artifacts: list[tuple[str, str]] = []
    transcripts: set[str] = set()
    metas: set[str] = set()
    runs: set[str] = set()
    journals: set[str] = set()
    for candidate in paths:
        source_path = str(candidate)
        rule = artifact_rule_for_path(Provider.CLAUDE_CODE, source_path)
        if rule is None:
            continue
        artifacts.append((rule.kind, source_path))
        name = Path(source_path).name
        if rule.kind == "agent_transcript":
            if agent_id := _agent_id_from_path(source_path):
                transcripts.add(agent_id)
        elif rule.kind == "agent_sidecar_meta":
            if agent_id := _agent_id_from_path(source_path):
                metas.add(agent_id)
        elif rule.kind == "workflow_run_snapshot":
            runs.add(name.removesuffix(".json"))
        elif rule.kind == "workflow_journal":
            journals.add(Path(source_path).parent.name)
    gaps = [
        *(f"missing agent metadata for transcript {agent_id}" for agent_id in sorted(transcripts - metas)),
        *(f"missing agent transcript for metadata {agent_id}" for agent_id in sorted(metas - transcripts)),
        *(f"missing workflow run snapshot for journal {run_id}" for run_id in sorted(journals - runs)),
        *(f"missing workflow journal for run snapshot {run_id}" for run_id in sorted(runs - journals)),
    ]
    return ClaudeOrchestrationCoverage(
        artifact_counts=dict(sorted(Counter(kind for kind, _ in artifacts).items())),
        paired_agent_ids=tuple(sorted(transcripts & metas)),
        run_ids=tuple(sorted(runs | journals)),
        gaps=tuple(gaps),
    )


def _decode(payload: bytes | str | object, *, jsonl: bool) -> object:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        return payload
    if jsonl:
        return [json.loads(line) for line in payload.splitlines() if line.strip()]
    return json.loads(payload)


def _document_fact(kind: str, source_path: str, payload: Mapping[str, object]) -> ClaudeOrchestrationFact:
    fallback_run_id = Path(source_path).stem if kind == "workflow_run_snapshot" else None
    run_id = (
        _first_string(
            payload,
            "runId",
            "run_id",
            "workflowRunId",
            "workflow_run_id",
            "id",
        )
        or fallback_run_id
    )
    agent_id = _first_string(payload, "agentId", "agent_id", "sessionId", "session_id")
    if kind == "agent_sidecar_meta":
        agent_id = agent_id or _agent_id_from_path(source_path)
    content_key = _first_string(payload, "contentKey", "content_key", "callKey", "call_key", "key")
    retained = {key: value for key, value in payload.items() if key in _DOCUMENT_FIELDS}
    return ClaudeOrchestrationFact(kind, source_path, None, run_id, agent_id, content_key, retained)


def _journal_fact(source_path: str, line: int, payload: Mapping[str, object]) -> ClaudeOrchestrationFact:
    run_id = (
        _first_string(
            payload,
            "runId",
            "run_id",
            "workflowRunId",
            "workflow_run_id",
        )
        or Path(source_path).parent.name
    )
    agent_id = _first_string(payload, "agentId", "agent_id", "sessionId", "session_id")
    content_key = _first_string(payload, "contentKey", "content_key", "callKey", "call_key", "key")
    retained = {key: value for key, value in payload.items() if key in _JOURNAL_FIELDS}
    return ClaudeOrchestrationFact(
        "workflow_journal_entry",
        source_path,
        line,
        run_id,
        agent_id,
        content_key,
        retained,
    )


__all__ = [
    "ClaudeOrchestrationArtifact",
    "ClaudeOrchestrationCoverage",
    "ClaudeOrchestrationFact",
    "inventory_claude_orchestration_artifacts",
    "parse_claude_orchestration_artifact",
]
