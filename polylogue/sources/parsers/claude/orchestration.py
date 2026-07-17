"""Claude Code Workflow artifact admission and evidence-preserving parsing.

Workflow artifacts are not synthetic sessions.  This module parses their
provider-native fields into fact-shaped values with a source-path provenance
handle, while the generic work-evidence adapter decides how to materialize
them.  Missing counterpart artifacts are reported as gaps rather than paired
by filename wishful thinking.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.origin_specs import artifact_rule_for_path


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        if value := _string(payload.get(key)):
            return value
    return None


@dataclass(frozen=True, slots=True)
class ClaudeOrchestrationFact:
    """One provider-native fact plus direct raw-artifact provenance."""

    kind: str
    source_path: str
    source_line: int | None
    run_id: str | None
    agent_id: str | None
    content_key: str | None
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class ClaudeOrchestrationArtifact:
    kind: str
    source_path: str
    parse_policy: str
    facts: tuple[ClaudeOrchestrationFact, ...]


@dataclass(frozen=True, slots=True)
class ClaudeOrchestrationCoverage:
    artifact_counts: dict[str, int]
    paired_agent_ids: tuple[str, ...]
    gaps: tuple[str, ...]


def parse_claude_orchestration_artifact(
    source_path: str, payload: bytes | str | object
) -> ClaudeOrchestrationArtifact | None:
    """Parse one declared Claude orchestration artifact without inventing links.

    ``source_path`` itself is the raw-evidence handle.  Callers retain the raw
    revision and attach its raw id when these facts cross the work-graph
    boundary; this parser intentionally does not guess one.
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
    """Inventory declared workflow members and report only evidence-backed gaps."""

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
            transcripts.add(name.removesuffix(".jsonl").removesuffix(".ndjson"))
        elif rule.kind == "agent_sidecar_meta":
            metas.add(name.removesuffix(".meta.json"))
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
    run_id = _first_string(payload, "runId", "run_id", "workflowRunId", "id")
    agent_id = _first_string(payload, "agentId", "agent_id", "sessionId", "session_id")
    retained = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "runId",
            "run_id",
            "workflowRunId",
            "taskId",
            "task_id",
            "resumeFromRunId",
            "resume_from_run_id",
            "workflow",
            "workflowName",
            "name",
            "scriptPath",
            "scriptHash",
            "phases",
            "labels",
            "status",
            "model",
            "timing",
            "tokens",
            "tools",
            "result",
            "structuredResult",
            "agentId",
            "sessionId",
        }
    }
    return ClaudeOrchestrationFact(kind, source_path, None, run_id, agent_id, None, retained)


def _journal_fact(source_path: str, line: int, payload: Mapping[str, object]) -> ClaudeOrchestrationFact:
    run_id = _first_string(payload, "runId", "run_id", "workflowRunId") or Path(source_path).parent.name
    agent_id = _first_string(payload, "agentId", "agent_id", "sessionId", "session_id")
    content_key = _first_string(payload, "contentKey", "content_key", "callKey", "call_key", "key")
    retained = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "type",
            "event",
            "callKey",
            "call_key",
            "contentKey",
            "content_key",
            "attempt",
            "attemptId",
            "agentId",
            "model",
            "status",
            "phase",
            "progress",
            "timing",
            "tokens",
            "tools",
            "result",
            "structuredResult",
            "unresolved",
            "transcriptPath",
            "metaPath",
        }
    }
    return ClaudeOrchestrationFact("workflow_journal_entry", source_path, line, run_id, agent_id, content_key, retained)


__all__ = [
    "ClaudeOrchestrationArtifact",
    "ClaudeOrchestrationCoverage",
    "ClaudeOrchestrationFact",
    "inventory_claude_orchestration_artifacts",
    "parse_claude_orchestration_artifact",
]
