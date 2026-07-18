"""Synthetic continuity corpus compiler built on existing test-infra seams.

This module is fixture construction, not a second scenario framework.  It
loads planted constants from ``tests/data/continuity/catalog.json`` and emits
archive rows through :class:`ArchiveScenario` / :class:`SessionBuilder`, the
same builders used throughout the repository's integration tests.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioContentBlock,
    ScenarioMessage,
    native_session_id_for,
)
from tests.infra.storage_records import SessionBuilder

_DEFAULT_CATALOG = Path(__file__).parents[1] / "data" / "continuity" / "catalog.json"


@dataclass(frozen=True, slots=True)
class ContinuityFixtureSeed:
    """Identity and direct-storage census for one seeded continuity corpus."""

    archive_root: Path
    fixture_id: str
    coordinator_session_id: str
    direct_facts: JSONDocument


def continuity_catalog_path() -> Path:
    """Return the repository-owned independent oracle manifest."""

    return _DEFAULT_CATALOG


def load_continuity_catalog(path: Path | None = None) -> JSONDocument:
    """Load and minimally validate a continuity corpus/oracle manifest."""

    catalog_path = path or continuity_catalog_path()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = require_json_document(payload, context=f"continuity catalog {catalog_path}")
    if catalog.get("schema_version") != 2:
        raise ValueError("continuity catalog schema_version must be 2")
    if not isinstance(catalog.get("fixture_id"), str):
        raise ValueError("continuity catalog requires fixture_id")
    if not isinstance(catalog.get("corpus"), Mapping):
        raise ValueError("continuity catalog requires corpus object")
    if not isinstance(catalog.get("oracles"), Mapping):
        raise ValueError("continuity catalog requires oracles object")
    return dict(catalog)


def seed_continuity_archive(
    archive_root: Path,
    *,
    catalog: Mapping[str, JSONValue] | None = None,
) -> ContinuityFixtureSeed:
    """Compile the planted synthetic corpus into a fresh archive root.

    The constants are read from the independent manifest before any public
    query route is invoked.  Construction uses existing archive test builders;
    only the assertion and usage tables require their established direct seed
    primitives because those facts live outside parsed transcript rows.
    """

    fixture = dict(catalog or load_continuity_catalog())
    corpus = _mapping(fixture, "corpus")
    provider = _string(corpus, "provider")
    timestamp = _string(corpus, "timestamp")
    archive_root = archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"
    if db_path.exists() or (archive_root / "user.db").exists():
        raise ValueError(f"continuity archive root must be fresh: {archive_root}")

    _seed_text_scenario(
        db_path,
        provider=provider,
        timestamp=timestamp,
        data=_mapping(corpus, "resume"),
        title="Continuity resume",
    )
    _seed_action_scenario(
        db_path,
        provider=provider,
        timestamp=timestamp,
        data=_mapping(corpus, "forensic_debug"),
        title="Continuity forensic debug",
        tool_input={"path": _string(_mapping(corpus, "forensic_debug"), "path")},
    )
    _seed_text_scenario(
        db_path,
        provider=provider,
        timestamp=timestamp,
        data=_mapping(corpus, "prior_art"),
        title="Continuity prior art",
    )
    _seed_text_scenario(
        db_path,
        provider=provider,
        timestamp=timestamp,
        data={
            "session_key": _string(_mapping(corpus, "decision"), "session_key"),
            "message_key": _string(_mapping(corpus, "decision"), "message_key"),
            "text": "Decision evidence for cursor-bound-pages.",
        },
        title="Continuity decision evidence",
    )
    _seed_action_scenario(
        db_path,
        provider=provider,
        timestamp=timestamp,
        data=_mapping(corpus, "postmortem"),
        title="Continuity postmortem",
        tool_input={"command": "pytest -q"},
    )
    _seed_cost_usage(db_path, provider=provider, timestamp=timestamp, data=_mapping(corpus, "cost"))
    coordinator_id = _seed_parallel_incident(
        db_path,
        provider=provider,
        timestamp=timestamp,
        data=_mapping(corpus, "parallel_incident"),
    )
    _seed_decision_assertion(archive_root, provider=provider, data=_mapping(corpus, "decision"))

    direct_facts = validate_continuity_population(archive_root, fixture)
    return ContinuityFixtureSeed(
        archive_root=archive_root,
        fixture_id=_string(fixture, "fixture_id"),
        coordinator_session_id=coordinator_id,
        direct_facts=direct_facts,
    )


def validate_continuity_population(
    archive_root: Path,
    catalog: Mapping[str, JSONValue] | None = None,
) -> JSONDocument:
    """Read planted facts directly from SQLite, independently of MCP queries."""

    fixture = dict(catalog or load_continuity_catalog())
    corpus = _mapping(fixture, "corpus")
    provider = _string(corpus, "provider")
    incident = _mapping(corpus, "parallel_incident")
    run_ref = _string(incident, "run_ref")
    coordinator = native_session_id_for(provider, _string(incident, "coordinator_key"))
    db_path = archive_root / "index.db"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        child_rows = conn.execute(
            "SELECT session_id FROM sessions WHERE parent_session_id = ? ORDER BY session_id",
            (coordinator,),
        ).fetchall()
        child_ids = [str(row["session_id"]) for row in child_rows]
        marker_rows = conn.execute(
            """
            SELECT b.session_id, b.text
            FROM blocks AS b
            JOIN sessions AS s ON s.session_id = b.session_id
            WHERE s.parent_session_id = ? AND instr(COALESCE(b.text, ''), 'parallel-child') > 0
            ORDER BY b.session_id
            """,
            (coordinator,),
        ).fetchall()
        member_rows = [row for row in marker_rows if f"workflow_run:{run_ref}" in str(row["text"])]
        other_rows = [row for row in marker_rows if f"workflow_run:{run_ref}" not in str(row["text"])]
        invocation_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE session_id = ? AND instr(COALESCE(text, ''), ?) > 0",
                (coordinator, f"workflow-invocation:{run_ref}"),
            ).fetchone()[0]
        )
        final_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE session_id = ? AND instr(COALESCE(text, ''), ?) > 0",
                (coordinator, f"final-structured-result:{run_ref}"),
            ).fetchone()[0]
        )
        curriculum_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE session_id = ? AND instr(COALESCE(text, ''), ?) > 0",
                (coordinator, f"incident-curriculum:{run_ref}"),
            ).fetchone()[0]
        )
        usage = conn.execute(
            """
            SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            FROM session_model_usage
            WHERE session_id = ?
            """,
            (native_session_id_for(provider, _string(_mapping(corpus, "cost"), "session_key")),),
        ).fetchone()

    if len(child_ids) != _integer(incident, "member_children") + _integer(incident, "other_children"):
        raise AssertionError("planted coordinator child census does not match manifest")
    if len(member_rows) != _integer(incident, "member_children"):
        raise AssertionError("planted incident-run membership does not match manifest")
    if len(other_rows) != _integer(incident, "other_children"):
        raise AssertionError("planted non-run membership does not match manifest")
    if invocation_count != _integer(incident, "workflow_invocations"):
        raise AssertionError("planted workflow invocation count does not match manifest")
    if final_count != 1:
        raise AssertionError("planted final-result count must be one")
    curriculum = _curriculum_records(incident)
    if curriculum_count != len(curriculum):
        raise AssertionError("planted incident curriculum count does not match manifest")
    if usage is None:
        raise AssertionError("planted provider usage row is missing")

    input_tokens = int(usage["input_tokens"])
    output_tokens = int(usage["output_tokens"])
    cache_read_tokens = int(usage["cache_read_tokens"])
    cache_write_tokens = int(usage["cache_write_tokens"])
    return {
        "coordinator_session_id": coordinator,
        "coordinator_children": len(child_ids),
        "incident_members": len(member_rows),
        "other_children": len(other_rows),
        "workflow_invocations": invocation_count,
        "final_result_count": final_count,
        "incident_curriculum_cases": curriculum_count,
        "usage_input_tokens": input_tokens,
        "usage_output_tokens": output_tokens,
        "usage_cached_input_tokens": cache_read_tokens,
        "usage_cache_write_tokens": cache_write_tokens,
        "usage_total_tokens": input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
    }


def _seed_text_scenario(
    db_path: Path,
    *,
    provider: str,
    timestamp: str,
    data: Mapping[str, JSONValue],
    title: str,
) -> None:
    ArchiveScenario(
        name=_string(data, "session_key"),
        session_id=_string(data, "session_key"),
        provider=provider,
        title=title,
        created_at=timestamp,
        updated_at=timestamp,
        messages=(
            ScenarioMessage(
                role="assistant",
                text=_string(data, "text"),
                message_id=_string(data, "message_key"),
                timestamp=timestamp,
            ),
        ),
    ).seed(db_path)


def _seed_action_scenario(
    db_path: Path,
    *,
    provider: str,
    timestamp: str,
    data: Mapping[str, JSONValue],
    title: str,
    tool_input: JSONDocument,
) -> None:
    message_key = _string(data, "message_key")
    tool_name = _string(data, "tool")
    tool_id = f"{message_key}-call"
    output = _string(data, "output")
    ArchiveScenario(
        name=_string(data, "session_key"),
        session_id=_string(data, "session_key"),
        provider=provider,
        title=title,
        created_at=timestamp,
        updated_at=timestamp,
        messages=(
            ScenarioMessage(
                role="assistant",
                text=output,
                message_id=message_key,
                timestamp=timestamp,
                blocks=(
                    ScenarioContentBlock.tool_use(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_id=tool_id,
                    ),
                    ScenarioContentBlock.tool_result(
                        output,
                        tool_name=tool_name,
                        tool_id=tool_id,
                        is_error=True,
                        exit_code=_integer(data, "exit_code"),
                    ),
                ),
            ),
        ),
    ).seed(db_path)


def _seed_cost_usage(
    db_path: Path,
    *,
    provider: str,
    timestamp: str,
    data: Mapping[str, JSONValue],
) -> None:
    builder = (
        SessionBuilder(db_path, _string(data, "session_key"))
        .provider(provider)
        .title("Continuity cost audit")
        .created_at(timestamp)
        .updated_at(timestamp)
        .add_message(
            message_id="cost-evidence",
            role="assistant",
            text="Synthetic reported usage evidence.",
            timestamp=timestamp,
        )
    )
    builder.save()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO session_model_usage (
                session_id, model_name, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, message_count,
                cost_provenance
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 'origin_reported')
            """,
            (
                builder.native_session_id(),
                _string(data, "model"),
                _integer(data, "input_tokens"),
                _integer(data, "output_tokens"),
                _integer(data, "cached_input_tokens"),
                _integer(data, "cache_write_tokens"),
            ),
        )
        conn.commit()


def _seed_parallel_incident(
    db_path: Path,
    *,
    provider: str,
    timestamp: str,
    data: Mapping[str, JSONValue],
) -> str:
    coordinator_key = _string(data, "coordinator_key")
    run_ref = _string(data, "run_ref")
    coordinator = (
        SessionBuilder(db_path, coordinator_key)
        .provider(provider)
        .title("Synthetic parallel-agent coordinator")
        .created_at(timestamp)
        .updated_at(timestamp)
    )
    for index in range(1, _integer(data, "workflow_invocations") + 1):
        coordinator.add_message(
            message_id=f"workflow-invocation-{index:02d}",
            role="assistant",
            text=f"workflow-invocation:{run_ref} invocation:{index:02d}",
            timestamp=timestamp,
        )
    for record in _curriculum_records(data):
        case_id = _string(record, "case")
        fields = (
            f"incident-curriculum:{run_ref}",
            f"case:{case_id}",
            f"query_shape:{_string(record, 'query_shape')}",
            f"physical_size:{_string(record, 'physical_size')}",
            f"corpus_match:{_string(record, 'corpus_match')}",
            f"structure_discovery:{_string(record, 'structure_discovery')}",
            f"shipped_instruction:{_string(record, 'shipped_instruction')}",
            f"outcome:{_string(record, 'outcome')}",
        )
        coordinator.add_message(
            message_id=f"incident-curriculum-{case_id}",
            role="assistant",
            text=" ".join(fields),
            timestamp=timestamp,
        )
    coordinator.add_message(
        message_id="final-result",
        role="assistant",
        text=f"final-structured-result:{run_ref} outcome:complete",
        timestamp=timestamp,
    )
    coordinator.save()
    coordinator_native = coordinator.native_session_id()
    parent_native_id = f"ext-{coordinator_key}"

    member_count = _integer(data, "member_children")
    call_key_count = _integer(data, "call_keys")
    result_record_count = _integer(data, "result_records")
    completed_count = _integer(data, "completed_call_keys")
    unresolved_count = _integer(data, "unresolved_call_keys")
    if completed_count + unresolved_count != call_key_count:
        raise ValueError("incident completed + unresolved call keys must equal call key count")
    if result_record_count > member_count:
        raise ValueError("incident result records cannot exceed member children")

    for index in range(1, member_count + 1):
        call_number = ((index - 1) % call_key_count) + 1
        call_key = f"call-{call_number:02d}"
        tokens = [
            "parallel-child",
            f"workflow_run:{run_ref}",
            "attempt_transcript:yes",
            f"call_key:{call_key}",
        ]
        if index <= result_record_count:
            tokens.append("result_record:yes")
        if call_number <= completed_count:
            tokens.append(f"completed_key:{call_key}")
        else:
            tokens.append(f"unresolved_key:{call_key}")
        (
            SessionBuilder(db_path, f"continuity-incident-member-{index:03d}")
            .provider(provider)
            .title(f"Synthetic incident member {index:03d}")
            .created_at(timestamp)
            .updated_at(timestamp)
            .parent_session(parent_native_id)
            .branch_type("subagent")
            .add_message(
                message_id="attempt",
                role="assistant",
                text=" ".join(tokens),
                timestamp=timestamp,
            )
            .save()
        )

    for index in range(1, _integer(data, "other_children") + 1):
        (
            SessionBuilder(db_path, f"continuity-incident-other-{index:03d}")
            .provider(provider)
            .title(f"Synthetic non-run incident child {index:03d}")
            .created_at(timestamp)
            .updated_at(timestamp)
            .parent_session(parent_native_id)
            .branch_type("subagent")
            .add_message(
                message_id="attempt",
                role="assistant",
                text=(
                    f"parallel-child workflow_run:wf_other_{index:03d} "
                    f"attempt_transcript:yes call_key:other-{index:03d}"
                ),
                timestamp=timestamp,
            )
            .save()
        )
    return coordinator_native


def _seed_decision_assertion(
    archive_root: Path,
    *,
    provider: str,
    data: Mapping[str, JSONValue],
) -> None:
    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    target_session = native_session_id_for(provider, _string(data, "session_key"))
    message_ref = f"message:{target_session}:{_string(data, 'message_key')}"
    with sqlite3.connect(user_db) as conn:
        upsert_assertion(
            conn,
            assertion_id=_string(data, "assertion_id"),
            target_ref=f"session:{target_session}",
            kind=AssertionKind.DECISION,
            key="continuity-decision",
            body_text=_string(data, "body"),
            author_ref="user:continuity-fixture",
            author_kind="user",
            evidence_refs=[message_ref],
            status="active",
            visibility="public",
            now_ms=1_768_478_400_000,
        )
        conn.commit()


def _mapping(source: Mapping[str, JSONValue], key: str) -> Mapping[str, JSONValue]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"continuity catalog field {key!r} must be an object")
    return cast(Mapping[str, JSONValue], value)


def _string(source: Mapping[str, JSONValue], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise ValueError(f"continuity catalog field {key!r} must be a string")
    return value


def _integer(source: Mapping[str, JSONValue], key: str) -> int:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"continuity catalog field {key!r} must be an integer")
    return value


def _curriculum_records(source: Mapping[str, JSONValue]) -> tuple[Mapping[str, JSONValue], ...]:
    value = source.get("attempt_curriculum", [])
    if not isinstance(value, list):
        raise ValueError("continuity catalog attempt_curriculum must be a list")
    records: list[Mapping[str, JSONValue]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"continuity catalog attempt_curriculum[{index}] must be an object")
        records.append(cast(Mapping[str, JSONValue], item))
    return tuple(records)


__all__ = [
    "ContinuityFixtureSeed",
    "continuity_catalog_path",
    "load_continuity_catalog",
    "seed_continuity_archive",
    "validate_continuity_population",
]
