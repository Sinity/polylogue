"""Declared-source schema inference regressions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas import source_inference as source_inference_module
from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence
from polylogue.schemas.generation.workflow import generate_provider_schema_from_sources
from polylogue.schemas.source_inference import (
    SchemaSourceInput,
    SourceObservation,
    SourceRevision,
    _collect_candidate,
    _collect_payload_evidence,
    _SizedPayload,
    _SourceCandidate,
    _spooled_contributions,
    infer_sources,
)


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
    source_receipt = result.phase_receipt.get("source")
    assert isinstance(source_receipt, dict)
    assert source_receipt["source_terminal_outcomes"] == {"included": 1}


def test_declared_codex_database_observes_table_and_column_shape(tmp_path: Path) -> None:
    """Database structure is schema evidence even though rows are not sessions."""
    path = tmp_path / "state_5.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE threads (id TEXT, title TEXT, added_column INTEGER);"
            "CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT);"
        )
    candidate = _SourceCandidate("codex", tmp_path, path, "codex-state")
    collected = _collect_candidate(candidate)
    assert collected.terminal.outcome == "included"
    assert len(collected.contributions) == 1
    evidence = SchemaEvidence.from_json(collected.contributions[0].evidence_by_element["database_schema"])
    properties = evidence.structure["properties"]
    assert isinstance(properties, dict)
    tables = properties["tables"]
    assert isinstance(tables, dict)
    assert "added_column" in json.dumps(evidence.structure)
    assert "title" in json.dumps(evidence.structure)


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


def test_complete_jsonl_final_record_does_not_require_a_newline(tmp_path: Path) -> None:
    """Anti-vacuity: a newline-only check rejects a complete transcript accepted by the native decoder."""
    source = tmp_path / "session.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "unterminated-but-complete",
                "message": {"role": "user", "content": "synthetic"},
            }
        ),
        encoding="utf-8",
    )

    result = infer_sources(
        (SchemaSourceInput("claude-code", source),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
    )

    assert result.terminal_counts == {"included": 1}


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


def _chatgpt_conversation(session_id: str, *, updated: float, extra: str | None = None) -> dict[str, object]:
    conversation: dict[str, object] = {
        "id": session_id,
        "conversation_id": session_id,
        "update_time": updated,
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
    if extra is not None:
        conversation[extra] = True
    return conversation


def test_multi_session_export_keeps_independent_native_contributions(tmp_path: Path) -> None:
    """Anti-vacuity: treating an export file as one source undercounts two conversations."""
    source = tmp_path / "conversations.json"
    source.write_text(
        json.dumps([_chatgpt_conversation("first", updated=1), _chatgpt_conversation("second", updated=2)]),
        encoding="utf-8",
    )

    result = infer_sources(
        (SchemaSourceInput("chatgpt", source),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_document"])

    assert result.terminal_counts == {"included": 2}
    assert evidence.current_source_count == 2


def test_large_export_worker_spools_native_contributions_before_returning(tmp_path: Path) -> None:
    """Anti-vacuity: returning every native contribution retains one reduced payload per export record."""
    source = tmp_path / "conversations.json"
    source.write_text(
        json.dumps([_chatgpt_conversation(f"session-{index}", updated=float(index)) for index in range(96)]),
        encoding="utf-8",
    )
    candidate = _SourceCandidate("chatgpt", source, source, "synthetic-export")
    spool_path = tmp_path / "contributions.sqlite3"

    collected = _collect_candidate(candidate, spool_path=spool_path)

    assert collected.terminal.outcome == "included"
    assert collected.contributions == ()
    assert collected.spool_path == spool_path
    assert len(tuple(_spooled_contributions(spool_path))) == 96

    result = infer_sources(
        (SchemaSourceInput("chatgpt", source),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    assert result.terminal_counts == {"included": 96}
    assert len(result.evidence_by_element["session_document"]) == 1


def test_interrupted_export_never_publishes_a_complete_cache_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: a partial export cache hit would silently omit the interrupted native session."""
    root = tmp_path / "exports"
    root.mkdir()
    for name in ("first", "second"):
        (root / f"{name}.json").write_text(json.dumps([_chatgpt_conversation(name, updated=1)]), encoding="utf-8")
    cache_path = tmp_path / "source-cache.sqlite3"
    real_put = source_inference_module._put_contribution
    real_temporary_directory = tempfile.TemporaryDirectory
    created_spool_directories: list[Path] = []

    def track_temporary_directory(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory[str]:
        directory = real_temporary_directory(*args, **kwargs)
        created_spool_directories.append(Path(directory.name))
        return directory

    def interrupt_second(*args: Any, **kwargs: Any) -> None:
        candidate = args[1]
        assert isinstance(candidate, _SourceCandidate)
        if candidate.path.name == "second.json":
            raise RuntimeError("synthetic parent interruption")
        real_put(*args, **kwargs)

    monkeypatch.setattr(source_inference_module, "_put_contribution", interrupt_second)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", track_temporary_directory)
    with pytest.raises(RuntimeError, match="synthetic parent interruption"):
        infer_sources((SchemaSourceInput("chatgpt", root),), cache_path=cache_path, max_workers=1)
    assert created_spool_directories and all(not path.exists() for path in created_spool_directories)

    monkeypatch.setattr(source_inference_module, "_put_contribution", real_put)
    result = infer_sources((SchemaSourceInput("chatgpt", root),), cache_path=cache_path, max_workers=1)

    assert result.terminal_counts == {"included": 2}
    assert result.cache_hits >= 1
    assert result.cache_misses >= 1


def test_latest_nonprefix_revision_drives_fields_and_old_revision_only_adds_structure(tmp_path: Path) -> None:
    """Anti-vacuity: merging old exports into current statistics makes retired fields look current."""
    root = tmp_path / "exports"
    root.mkdir()
    (root / "old.json").write_text(
        json.dumps([_chatgpt_conversation("same", updated=1, extra="retired_field")]), encoding="utf-8"
    )
    (root / "latest.json").write_text(
        json.dumps([_chatgpt_conversation("same", updated=2, extra="current_field")]), encoding="utf-8"
    )

    result = infer_sources(
        (SchemaSourceInput("chatgpt", root),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_document"])

    assert evidence.current_source_count == 1
    assert evidence.historical_source_count == 1
    assert "$.current_field" in evidence.fields
    assert "$.retired_field" not in evidence.fields


def test_longer_jsonl_transcript_supersedes_its_strict_prefix(tmp_path: Path) -> None:
    """Anti-vacuity: restricting prefix checks to streams must retain transcript supersession."""
    root = tmp_path / "transcripts"
    root.mkdir()
    old_line = json.dumps(
        {"type": "user", "sessionId": "same", "message": {"role": "user", "content": "first"}},
        separators=(",", ":"),
    )
    old = f"{old_line}\n"
    for index in range(128):
        extension = json.dumps(
            {
                "type": "assistant",
                "sessionId": "same",
                "message": {"role": "assistant", "content": "second"},
                "stream_extension": index,
            },
            separators=(",", ":"),
        )
        longer = f"{old}{extension}\n"
        if hashlib.sha256(old.encode()).hexdigest() > hashlib.sha256(longer.encode()).hexdigest():
            break
    else:
        pytest.fail("could not produce a digest ordering that requires prefix selection")
    (root / "old.jsonl").write_text(old, encoding="utf-8")
    (root / "longer.jsonl").write_text(longer, encoding="utf-8")

    result = infer_sources(
        (SchemaSourceInput("claude-code", root),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_record_stream"]
    )

    assert evidence.current_source_count == 1
    assert "$.stream_extension" in evidence.fields


def test_source_route_folds_selected_contributions_before_returning(tmp_path: Path) -> None:
    """Anti-vacuity: materializing one row per source grows coordinator retention with source count."""
    root = tmp_path / "exports"
    root.mkdir()
    (root / "old.json").write_text(
        json.dumps([_chatgpt_conversation("same", updated=1, extra="retired_field")]), encoding="utf-8"
    )
    (root / "latest.json").write_text(
        json.dumps([_chatgpt_conversation("same", updated=2, extra="current_field")]), encoding="utf-8"
    )
    (root / "independent.json").write_text(
        json.dumps([_chatgpt_conversation("independent", updated=1, extra="independent_field")]), encoding="utf-8"
    )

    result = infer_sources(
        (SchemaSourceInput("chatgpt", root),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )

    rows = result.evidence_by_element["session_document"]
    assert len(rows) == 1
    evidence = SchemaEvidence.from_json(rows[0])
    assert evidence.current_source_count == 2
    assert evidence.historical_source_count == 1
    assert "$.current_field" in evidence.fields
    assert "$.retired_field" not in evidence.fields

    warm = infer_sources(
        (SchemaSourceInput("chatgpt", root),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    assert len(warm.evidence_by_element["session_document"]) == 1
    assert warm.cache_hits == 6


def test_zip_members_and_jsonl_header_share_the_declared_native_source(tmp_path: Path) -> None:
    """Anti-vacuity: routing non-header JSONL records by path inflates one session denominator."""
    import zipfile

    source = tmp_path / "sources.zip"
    header = {
        "type": "session_meta",
        "payload": {"id": "zip-codex", "timestamp": "2026-01-01T00:00:00Z", "cli_version": "v1.2.3"},
    }
    event = {
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    }
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("nested/session.jsonl", "\n".join((json.dumps(header), json.dumps(event))) + "\n")
        archive.writestr(
            "nested/second-session.jsonl",
            "\n".join(
                (
                    json.dumps({"type": "session_meta", "payload": {"id": "zip-codex-2", "cli_version": "v1.2.3"}}),
                    json.dumps(event),
                )
            )
            + "\n",
        )
    (tmp_path / "metadata.jsonl").write_text('{"type":"metadata"}\n', encoding="utf-8")

    result = infer_sources(
        (SchemaSourceInput("codex", tmp_path),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for rows in result.evidence_by_element.values() for item in rows
    )

    assert result.terminal_counts == {"included": 2, "unsupported": 1}
    assert evidence.current_source_count >= 1
    assert result.producer_version_counts == {"1.2.3": 1}
    provenance = result.provenance()
    assert provenance["source_terminal_outcomes"] == {"included": 2, "unsupported": 1}
    assert provenance["source_terminal_outcome_units"] == {
        "included": "native_source_revision",
        "unsupported": "physical_candidate",
    }
    assert provenance["source_candidate_terminal_outcomes"] == {"included": 1, "unsupported": 1}
    assert provenance["source_candidate_count"] == 2
    assert provenance["source_included_candidate_count"] == 1
    assert provenance["source_included_native_source_revision_count"] == 2


@pytest.mark.parametrize("with_spool", [False, True])
def test_zip_changed_after_member_collection_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_spool: bool
) -> None:
    """A ZIP must still have its original bytes after member reduction.

    Anti-vacuity: returning directly from the ZIP collector admits the stale
    contribution and can cache it under a revision that no longer exists.
    """
    import zipfile

    archive = tmp_path / "source.zip"
    replacement = tmp_path / "replacement.zip"
    records = "\n".join(
        (
            json.dumps(
                {
                    "type": "user",
                    "sessionId": "stable-session",
                    "version": "1.0.0",
                    "message": {"role": "user", "content": "synthetic"},
                }
            ),
            "",
        )
    )
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("session.jsonl", records)
    with zipfile.ZipFile(replacement, "w") as zip_file:
        zip_file.writestr("session.jsonl", records + "\n")
    candidate = _SourceCandidate("claude-code", tmp_path, archive, "synthetic-zip")
    spool_path = tmp_path / "contributions.sqlite" if with_spool else None

    stable = _collect_candidate(candidate, spool_path=spool_path)
    assert stable.terminal.outcome == "included"

    original_collect_zip = source_inference_module._collect_zip_candidate

    def replace_after_collection(
        collected_candidate: _SourceCandidate,
        revision: SourceRevision,
        *,
        dynamic_paths_by_element: dict[str, tuple[str, ...]],
        include_statistics: bool,
        metadata_only: bool,
        spool_path: Path | None,
    ) -> source_inference_module._CollectedCandidate:
        collected = original_collect_zip(
            collected_candidate,
            revision,
            dynamic_paths_by_element=dynamic_paths_by_element,
            include_statistics=include_statistics,
            metadata_only=metadata_only,
            spool_path=spool_path,
        )
        archive.write_bytes(replacement.read_bytes())
        return collected

    monkeypatch.setattr(source_inference_module, "_collect_zip_candidate", replace_after_collection)
    changed = _collect_candidate(candidate, spool_path=spool_path)

    assert changed.terminal.outcome == "changed_during_read"
    assert changed.contributions == ()
    assert changed.spool_path is None


def test_record_source_uses_codex_whole_stream_admission(tmp_path: Path) -> None:
    """Anti-vacuity: classifying each line drops valid non-message Codex records."""
    from polylogue.archive.artifact_taxonomy import classify_artifact
    from polylogue.core.enums import Provider

    records: tuple[JSONValue, ...] = (
        {"type": "session_meta", "payload": {"id": "admission", "cli_version": "1.2.3"}},
        {"type": "turn_context", "payload": {"cwd": "/repo"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": 1}}},
    )
    path = tmp_path / "rollout.jsonl"
    candidate = _SourceCandidate("codex", tmp_path, path, "synthetic-codex")
    revision = SourceRevision("codex", path, candidate.logical_source_id, "a" * 64, 0)

    assert classify_artifact(cast(JSONValue, records), provider=Provider.CODEX, source_path=path).schema_eligible
    contributions, record_count, _versions, _unrecognized = _collect_payload_evidence(
        candidate,
        revision,
        iter(records),
        dynamic_paths_by_element={},
    )

    assert record_count == len(records)
    assert [contribution.record_count for contribution in contributions] == [len(records)]

    mixed_generation: tuple[JSONValue, ...] = (
        *records[:3],
        {
            "type": "message",
            "id": "legacy-user",
            "role": "user",
            "content": [{"type": "input_text", "text": "legacy"}],
        },
    )
    assert not classify_artifact(
        cast(JSONValue, mixed_generation), provider=Provider.CODEX, source_path=path
    ).schema_eligible
    refused, refused_count, _versions, _unrecognized = _collect_payload_evidence(
        candidate,
        revision,
        iter(mixed_generation),
        dynamic_paths_by_element={},
    )
    assert refused == ()
    assert refused_count == 0


def test_record_source_keeps_claude_snapshot_after_stream_admission(tmp_path: Path) -> None:
    """Anti-vacuity: a snapshot line alone must not override its admitted transcript."""
    from polylogue.archive.artifact_taxonomy import classify_artifact
    from polylogue.core.enums import Provider

    path = tmp_path / ".claude" / "projects" / "project" / "claude-admission.jsonl"
    records: tuple[JSONValue, ...] = (
        {
            "type": "file-history-snapshot",
            "messageId": "u1",
            "snapshot": {"messageId": "u1", "trackedFileBackups": {}},
        },
        {"type": "user", "sessionId": "claude-admission", "uuid": "u1", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "sessionId": "claude-admission",
            "uuid": "u2",
            "parentUuid": "u1",
            "message": {"role": "assistant", "content": "hey"},
        },
    )
    candidate = _SourceCandidate("claude-code", path.parent, path, "synthetic-claude")
    revision = SourceRevision("claude-code", path, candidate.logical_source_id, "b" * 64, 0)

    assert classify_artifact(cast(JSONValue, records), provider=Provider.CLAUDE_CODE, source_path=path).schema_eligible
    assert not classify_artifact([records[0]], provider=Provider.CLAUDE_CODE, source_path=path).schema_eligible
    contributions, record_count, _versions, _unrecognized = _collect_payload_evidence(
        candidate,
        revision,
        iter(records),
        dynamic_paths_by_element={},
    )

    assert record_count == len(records)
    assert [contribution.record_count for contribution in contributions] == [len(records)]


def test_malformed_members_become_terminal_outcomes_without_aborting_inventory(tmp_path: Path) -> None:
    """Anti-vacuity: one broken ordinary source must not hide a valid sibling's evidence."""
    root = tmp_path / "sources"
    root.mkdir()
    (root / "valid.json").write_text(json.dumps([_chatgpt_conversation("valid", updated=1)]), encoding="utf-8")
    (root / "partial.jsonl").write_text(json.dumps(_chatgpt_conversation("partial", updated=1))[:-1], encoding="utf-8")
    (root / "invalid.zip").write_bytes(b"not a zip")

    result = infer_sources(
        (SchemaSourceInput("chatgpt", root),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )

    assert result.terminal_counts == {"decode_failed": 1, "included": 1, "partial_trailing_record": 1}
    assert result.terminal_reason_counts == {"invalid_zip": 1, "partial_trailing_record": 1}
    assert str(root) not in json.dumps(result.provenance())


def test_browser_capture_source_is_explicitly_excluded_with_aggregate_reason(tmp_path: Path) -> None:
    """Anti-vacuity: without the guard, a capture envelope becomes generic schema evidence."""
    source = tmp_path / "capture.json"
    source.write_text(json.dumps({"polylogue_capture_kind": "browser_llm_session"}), encoding="utf-8")
    progress: list[JSONDocument] = []

    result = infer_sources(
        (SchemaSourceInput("browser-capture", source),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
        progress=lambda _phase, payload: progress.append(payload),
    )

    assert result.evidence_by_element == {}
    assert result.terminal_counts == {"unsupported": 1}
    assert result.terminal_reason_counts == {"browser_capture_adapter_unavailable": 1}
    assert result.input_bytes == source.stat().st_size
    assert result.provenance()["source_terminal_reasons"] == {"browser_capture_adapter_unavailable": 1}
    assert progress[-1]["completed_candidates"] == progress[-1]["total_candidates"] == 1


def test_antigravity_non_json_inputs_are_counted_with_declared_terminal_reasons(tmp_path: Path) -> None:
    """Anti-vacuity: suffix-only inventory used to hide declared Antigravity inputs."""
    conversation = tmp_path / "conversations" / "session.pb"
    conversation.parent.mkdir()
    conversation.write_bytes(b"opaque-protobuf")
    brain = tmp_path / "brain" / "notes.md"
    brain.parent.mkdir()
    brain.write_text("sidecar", encoding="utf-8")

    result = infer_sources(
        (SchemaSourceInput("antigravity", tmp_path),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
    )

    assert result.evidence_by_element == {}
    assert result.terminal_counts == {"intentionally_excluded": 1, "unsupported": 1}
    assert result.terminal_reason_counts == {
        "antigravity_markdown_sidecar": 1,
        "antigravity_protobuf_adapter_unavailable": 1,
    }
    assert result.input_bytes == conversation.stat().st_size + brain.stat().st_size


def test_changed_source_between_evidence_passes_is_excluded_while_stable_peer_survives(tmp_path: Path) -> None:
    """Anti-vacuity: accepting first-pass evidence after a source rewrite mixes revisions."""
    stable = tmp_path / "stable.jsonl"
    changing = tmp_path / "changing.jsonl"

    def write_source(path: Path, session_id: str, content: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "user",
                    "sessionId": session_id,
                    "version": "1.2.3",
                    "message": {"role": "user", "content": content},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    write_source(stable, "stable", "stable source")
    write_source(changing, "changing", "first revision")
    mutated = False

    def mutate_after_preliminary(phase: str, _payload: JSONDocument) -> None:
        nonlocal mutated
        if phase == "reduce" and not mutated:
            write_source(changing, "changing", "second revision")
            mutated = True

    result = infer_sources(
        (SchemaSourceInput("claude-code", tmp_path),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
        progress=mutate_after_preliminary,
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_record_stream"]
    )

    assert mutated
    assert result.terminal_counts == {"changed_during_read": 1, "included": 1}
    assert result.terminal_reason_counts == {"changed_during_read": 1}
    assert evidence.current_source_count == 1


def test_source_chunking_matches_single_record_reduction_and_bounds_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: per-record merging makes a long source quadratic and hides chunk bounds."""
    from polylogue.schemas.generation import evidence as evidence_module

    candidate = _SourceCandidate("claude-code", tmp_path, tmp_path / "session.jsonl", "synthetic-source")
    revision = SourceRevision("claude-code", candidate.path, candidate.logical_source_id, "a" * 64, 0)
    payloads: tuple[JSONValue, ...] = tuple(
        {
            "type": "user" if index % 2 else "assistant",
            "sessionId": "chunked-session",
            "version": "1.2.3",
            "message": {
                "role": "user" if index % 3 else "assistant",
                "content": f"record-{index}",
                "metadata": {"even": index % 2 == 0} if index % 5 else {"multiple": index},
            },
        }
        for index in range(67)
    )
    single, single_records, single_versions, single_unrecognized = _collect_payload_evidence(
        candidate,
        revision,
        iter(payloads),
        dynamic_paths_by_element={},
        chunk_record_limit=1,
    )
    seen_chunk_sizes: list[int] = []
    real_collect = evidence_module.collect_source_evidence

    def collect_with_measurement(
        observation: SourceObservation,
        *,
        dynamic_paths: Collection[str] = (),
        is_current: bool | None = None,
        include_statistics: bool = True,
    ) -> SchemaEvidence:
        records = tuple(observation.records)
        seen_chunk_sizes.append(len(records))
        return real_collect(
            SourceObservation(
                logical_source_id=observation.logical_source_id,
                revision_sha256=observation.revision_sha256,
                subject=observation.subject,
                element_kind=observation.element_kind,
                records=records,
                is_current=observation.is_current,
            ),
            dynamic_paths=dynamic_paths,
            is_current=is_current,
            include_statistics=include_statistics,
        )

    monkeypatch.setattr(evidence_module, "collect_source_evidence", collect_with_measurement)
    chunked, chunked_records, chunked_versions, chunked_unrecognized = _collect_payload_evidence(
        candidate,
        revision,
        iter(payloads),
        dynamic_paths_by_element={},
        chunk_record_limit=7,
    )

    def assert_json_equivalent(left: object, right: object) -> None:
        if isinstance(left, float) and isinstance(right, float):
            assert left == pytest.approx(right, abs=1e-12)
        elif isinstance(left, Mapping) and isinstance(right, Mapping):
            assert left.keys() == right.keys()
            for key in left:
                assert_json_equivalent(left[key], right[key])
        elif isinstance(left, list) and isinstance(right, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_json_equivalent(left_item, right_item)
        else:
            assert left == right

    assert [(item.logical_source_id, item.revision_sha256, item.record_count) for item in chunked] == [
        (item.logical_source_id, item.revision_sha256, item.record_count) for item in single
    ]
    for single_item, chunked_item in zip(single, chunked, strict=True):
        assert_json_equivalent(single_item.evidence_by_element, chunked_item.evidence_by_element)
    assert (chunked_records, chunked_versions, chunked_unrecognized) == (
        single_records,
        single_versions,
        single_unrecognized,
    )
    from polylogue.schemas.generation.schema_builder import emit_schema_from_evidence
    from polylogue.schemas.observation import resolve_provider_config

    single_schema, _ = emit_schema_from_evidence(
        "claude-code",
        resolve_provider_config("claude-code"),
        SchemaEvidence.from_json(single[0].evidence_by_element["session_record_stream"]),
        privacy_config=None,
    )
    chunked_schema, _ = emit_schema_from_evidence(
        "claude-code",
        resolve_provider_config("claude-code"),
        SchemaEvidence.from_json(chunked[0].evidence_by_element["session_record_stream"]),
        privacy_config=None,
    )
    assert_json_equivalent(single_schema, chunked_schema)
    assert seen_chunk_sizes == [7] * 9 + [4]
    assert max(seen_chunk_sizes) == 7


def test_source_chunking_flushes_an_oversized_jsonl_record_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: a record-only chunk limit retains several oversized native JSONL lines together."""
    from polylogue.schemas.generation import evidence as evidence_module

    candidate = _SourceCandidate("claude-code", tmp_path, tmp_path / "session.jsonl", "synthetic-source")
    revision = SourceRevision("claude-code", candidate.path, candidate.logical_source_id, "b" * 64, 0)
    record: JSONValue = {
        "type": "user",
        "sessionId": "oversized-session",
        "message": {"role": "user", "content": "synthetic"},
    }
    seen_chunk_sizes: list[int] = []
    real_collect = evidence_module.collect_source_evidence

    def collect_with_measurement(
        observation: SourceObservation,
        *,
        dynamic_paths: Collection[str] = (),
        is_current: bool | None = None,
        include_statistics: bool = True,
    ) -> SchemaEvidence:
        records = tuple(observation.records)
        seen_chunk_sizes.append(len(records))
        return real_collect(
            SourceObservation(
                logical_source_id=observation.logical_source_id,
                revision_sha256=observation.revision_sha256,
                subject=observation.subject,
                element_kind=observation.element_kind,
                records=records,
                is_current=observation.is_current,
            ),
            dynamic_paths=dynamic_paths,
            is_current=is_current,
            include_statistics=include_statistics,
        )

    monkeypatch.setattr(source_inference_module, "_SOURCE_EVIDENCE_CHUNK_BYTE_LIMIT", 32)
    monkeypatch.setattr(evidence_module, "collect_source_evidence", collect_with_measurement)
    _collect_payload_evidence(
        candidate,
        revision,
        iter((_SizedPayload(record, 48), _SizedPayload(record, 48))),
        dynamic_paths_by_element={},
    )

    assert seen_chunk_sizes == [1, 1]


def test_source_route_measures_full_multiline_values_before_reduced_evidence(tmp_path: Path) -> None:
    """Anti-vacuity: compacting before collection caps source length and removes newline evidence."""
    from polylogue.schemas.observation_models import SCHEMA_SAMPLE_STRING_LIMIT

    content = "x" * (SCHEMA_SAMPLE_STRING_LIMIT + 17) + "\nsecond line"
    source = tmp_path / "long.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "long-source",
                "version": "1.2.3",
                "message": {"role": "user", "content": content},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = infer_sources(
        (SchemaSourceInput("claude-code", source),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_record_stream"]
    )
    stats = evidence.field_stats["$.message.content"]

    assert stats.string_length_distribution.maximum == len(content)
    assert stats.newline_distribution.maximum == 1


def test_progress_reports_source_aggregate_phases_without_source_paths(tmp_path: Path) -> None:
    """Anti-vacuity: long source runs need observable aggregate progress before completion."""
    source = tmp_path / "session.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "progress-source",
                "version": "1.2.3",
                "message": {"role": "user", "content": "synthetic"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events: list[tuple[str, JSONDocument]] = []

    result = infer_sources(
        (SchemaSourceInput("claude-code", source),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
        progress=lambda phase, payload: events.append((phase, payload)),
    )

    assert {phase for phase, _payload in events} >= {"inventory", "reduce", "revision_selection"}
    assert all(
        {"completed_candidates", "total_candidates", "input_bytes", "record_count"} <= payload.keys()
        for _, payload in events
    )
    assert all(str(source) not in json.dumps(payload) for _, payload in events)
    selection = next(payload for phase, payload in events if phase == "revision_selection")
    assert selection["record_count"] == 1
    assert selection["input_bytes"] == source.stat().st_size
    assert result.phase_timings_ms["revision_selection"] >= 0


def test_reduced_cache_rows_survive_later_interruption_and_are_private(tmp_path: Path) -> None:
    """Anti-vacuity: rolling back the enclosing scan used to discard completed source reductions."""
    from polylogue.schemas.source_cache import CachedContribution, SourceContributionCache

    cache_path = tmp_path / "source-cache.sqlite3"
    contribution = CachedContribution(
        cache_key="synthetic-key",
        evidence={"contributions": []},
        input_bytes=1,
        record_count=1,
        metadata={},
    )
    try:
        with SourceContributionCache(cache_path) as cache:
            cache.put(contribution)
            raise RuntimeError("interrupted after the completed contribution")
    except RuntimeError:
        pass

    with SourceContributionCache(cache_path) as cache:
        assert cache.get(contribution.cache_key) == contribution
    assert cache_path.stat().st_mode & 0o777 == 0o600


def test_source_inputs_reject_sample_limited_calls(tmp_path: Path) -> None:
    """Anti-vacuity: source inputs must never silently turn a full scan into a capped sample."""
    from polylogue.schemas.operator.inference import infer_schema
    from polylogue.schemas.operator.models import SchemaInferRequest

    with pytest.raises(ValueError, match="complete inputs"):
        infer_schema(
            SchemaInferRequest(
                provider="claude-code",
                db_path=tmp_path / "index.db",
                max_samples=2,
                source_inputs=(SchemaSourceInput("claude-code", tmp_path),),
            )
        )


def test_claude_subagent_files_with_one_parent_session_remain_independent(tmp_path: Path) -> None:
    """Anti-vacuity: grouping Claude subagents by the inherited parent ID loses one agent's evidence."""
    source_root = tmp_path / "claude"
    source_root.mkdir()
    for agent in ("agent-first", "agent-second"):
        (source_root / f"{agent}.jsonl").write_text(
            json.dumps(
                {
                    "type": "user",
                    "sessionId": "parent-session",
                    "version": "1.2.3",
                    "message": {"role": "user", "content": "shared subagent transcript"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    result = infer_sources(
        (SchemaSourceInput("claude-code", source_root),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for rows in result.evidence_by_element.values() for item in rows
    )

    assert result.terminal_counts == {"included": 2}
    assert evidence.current_source_count == 2
    warm = infer_sources(
        (SchemaSourceInput("claude-code", source_root),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
    )
    warm_evidence = merge_evidence(
        SchemaEvidence.from_json(item) for rows in warm.evidence_by_element.values() for item in rows
    )
    assert warm.cache_hits == 4
    assert warm_evidence.current_source_count == 2


@pytest.mark.parametrize(
    "source_key",
    [
        "src/private/config.py",
        r"src\private\config.py",
        "operator@example.invalid",
        "private-plan.md",
        ".env",
        "README",
    ],
)
def test_source_schema_hides_keys_in_small_content_maps(tmp_path: Path, source_key: str) -> None:
    """Anti-vacuity: retaining small map keys publishes source filenames and addresses."""
    source = tmp_path / "session.jsonl"
    records = [
        {"type": "user", "sessionId": "synthetic", "message": {"role": "user", "content": "hello"}},
        {
            "type": "file-history-snapshot",
            "messageId": "synthetic-message",
            "snapshot": {"trackedFileBackups": {source_key: {"version": 1}}},
        },
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    result = generate_provider_schema_from_sources(
        "claude-code",
        source_inputs=(SchemaSourceInput("claude-code", source),),
        cache_path=tmp_path / "cache.sqlite3",
        max_workers=1,
        privacy_config=None,
    )
    assert result.success
    assert result.schema is not None
    encoded = json.dumps(result.schema)
    assert json.dumps(source_key)[1:-1] not in encoded
    assert "trackedFileBackups" in encoded
    assert "additionalProperties" in encoded
    assert '"version"' in encoded


@pytest.mark.parametrize("generation", ["legacy", "envelope"])
def test_codex_schema_retains_wire_records_without_claiming_parser_support(tmp_path: Path, generation: str) -> None:
    """Requiring normalized semantics drops entire rollouts with tools or telemetry."""
    from polylogue.archive.artifact_taxonomy import classify_artifact
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.codex import is_supported_session_stream

    message: JSONDocument = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "synthetic"}],
    }
    records: list[JSONValue]
    if generation == "legacy":
        records = [
            {"id": "legacy", "timestamp": "2024-01-01T00:00:00Z"},
            {"record_type": "state"},
            message,
            {"type": "reasoning", "id": "reason", "summary": [], "encrypted_content": "synthetic"},
            {"type": "function_call", "call_id": "call", "name": "tool", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call", "output": "synthetic"},
        ]
    else:
        records = [
            {"type": "session_meta", "payload": {"id": "envelope"}},
            {"type": "response_item", "payload": message},
            {"type": "inter_agent_communication_metadata", "payload": {"trigger_turn": True}},
            {"type": "token_usage_record", "payload": {"usage": {"input_tokens": 10}}},
        ]
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    classification = classify_artifact(records, provider=Provider.CODEX, source_path=path)
    assert classification.schema_eligible
    assert not classification.parse_as_session
    assert not is_supported_session_stream(records)
    result = infer_sources((SchemaSourceInput("codex", path),), cache_path=tmp_path / "cache.sqlite", max_workers=1)
    assert result.terminal_counts == {"included": 1}
    assert result.record_count == len(records)
    assert sum(
        SchemaEvidence.from_json(row).current_record_count
        for rows in result.evidence_by_element.values()
        for row in rows
    ) == len(records)
    assert not classify_artifact(
        [*records, {"type": "invented_record"}], provider=Provider.CODEX, source_path=path
    ).schema_eligible


@pytest.mark.parametrize("turns", [False, True])
@pytest.mark.parametrize("document_suffix", [".json", ".jsonl"])
def test_gemini_checkpoint_stream_preserves_raw_records_and_document_cache(
    tmp_path: Path, turns: bool, document_suffix: str
) -> None:
    """Checkpoint records must be observed without document reconstruction or losing cached documents."""
    from polylogue.archive.artifact_taxonomy import classify_artifact
    from polylogue.core.enums import Provider
    from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources

    root = tmp_path / "sources"
    root.mkdir()
    document: JSONDocument = {
        "sessionId": "document-session",
        "projectHash": "synthetic-project",
        "kind": "main",
        "startTime": "2026-01-01T00:00:00Z",
        "lastUpdated": "2026-01-01T00:01:00Z",
        "messages": [{"id": "document-turn", "type": "user", "content": "synthetic"}],
    }
    (root / f"document{document_suffix}").write_text(json.dumps(document))
    inputs = (SchemaSourceInput("gemini-cli", root),)
    cache = tmp_path / "cache.sqlite"
    infer_sources(inputs, cache_path=cache, max_workers=1)
    records: list[JSONValue] = [
        {key: value for key, value in document.items() if key not in {"messages", "sessionId"}}
        | {"sessionId": "checkpoint-session"}
    ]
    if turns:
        records.extend(
            [
                {"id": "checkpoint-turn", "timestamp": "2026-01-01T00:00:01Z", "type": "user", "content": "synthetic"},
                {"$set": {"lastUpdated": "2026-01-01T00:02:00Z"}},
            ]
        )
    path = root / "checkpoint.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    candidate = _SourceCandidate("gemini-cli", root, path, "synthetic-source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    contributions, *_ = _collect_payload_evidence(
        candidate,
        SourceRevision("gemini-cli", path, "synthetic-source", digest, path.stat().st_size),
        records,
        dynamic_paths_by_element={},
    )
    assert len(contributions) == 1
    assert contributions[0].declared_updated_at == (
        1,
        "2026-01-01T00:02:00.000000+00:00" if turns else "2026-01-01T00:01:00.000000+00:00",
    )
    artifact = classify_artifact(records, provider=Provider.GEMINI_CLI, source_path=path)
    assert artifact.schema_eligible and not artifact.parse_as_session
    upgraded = infer_sources(inputs, cache_path=cache, max_workers=1)
    fresh = infer_sources(inputs, cache_path=tmp_path / "fresh.sqlite", max_workers=1)
    assert upgraded.cache_phase_hits == {"structure": 1, "statistics": 1}
    assert upgraded.evidence_by_element == fresh.evidence_by_element
    stream = merge_evidence(
        SchemaEvidence.from_json(item) for item in upgraded.evidence_by_element["session_record_stream"]
    )
    assert stream.current_source_count == 1
    assert stream.current_record_count == len(records)
    properties = stream.structure.get("properties")
    assert isinstance(properties, dict)
    assert "messages" not in properties
    if turns:
        assert "$set" in properties
    bundle = build_provider_bundle_from_sources(
        "gemini-cli",
        source_inputs=inputs,
        cache_path=cache,
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.sample_count == 1 + len(records)
    assert bundle.result.schema is not None
    assert bundle.result.schema["x-polylogue-sample-granularity"] == "document"
    stream_schema = next(iter(bundle.package_schemas.values()))["session_record_stream"]
    assert stream_schema["x-polylogue-sample-granularity"] == "record"


@pytest.mark.parametrize("invalid", [{"unrelated": True}, {"$set": "invalid"}])
def test_gemini_checkpoint_rejects_unrecognized_records(tmp_path: Path, invalid: JSONDocument) -> None:
    """A valid header must not admit arbitrary trailing data as a session record."""
    path = tmp_path / "checkpoint.jsonl"
    header = {"sessionId": "synthetic-session", "projectHash": "synthetic-project", "kind": "main"}
    path.write_text("\n".join(json.dumps(record) for record in (header, invalid)))
    result = infer_sources(
        (SchemaSourceInput("gemini-cli", path),), cache_path=tmp_path / "cache.sqlite", max_workers=1
    )
    assert not result.evidence_by_element
    assert result.terminal_reason_counts == {"no_schema_units": 1}


def test_explicit_file_symlink_reaches_source_evidence(tmp_path: Path) -> None:
    """Anti-vacuity: directory hardening must not discard an explicitly selected regular-file target."""
    target = tmp_path / "target.jsonl"
    target.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "linked-session",
                "version": "1.2.3",
                "message": {"role": "user", "content": "linked"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selected = tmp_path / "selected.jsonl"
    selected.symlink_to(target)

    result = infer_sources(
        (SchemaSourceInput("claude-code", selected),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )

    assert result.terminal_counts == {"included": 1}
    assert result.record_count == 1


def test_terminal_candidate_revision_changes_source_input_manifest_digest(tmp_path: Path) -> None:
    """Anti-vacuity: a failed declared file must remain bound into the private input manifest."""
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "valid-session",
                "version": "1.2.3",
                "message": {"role": "user", "content": "valid"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"broken": nope}\n', encoding="utf-8")
    cache_path = tmp_path.parent / f"{tmp_path.name}-source-cache.sqlite3"

    first = infer_sources((SchemaSourceInput("claude-code", tmp_path),), cache_path=cache_path, max_workers=1)
    broken.write_text('{"different": nope}\n', encoding="utf-8")
    second = infer_sources((SchemaSourceInput("claude-code", tmp_path),), cache_path=cache_path, max_workers=1)

    assert first.terminal_counts["included"] == second.terminal_counts["included"] == 1
    assert len(first.terminal_counts) == len(second.terminal_counts) == 2
    assert first.record_count == second.record_count == 1
    assert first.input_manifest_digest != second.input_manifest_digest


@pytest.mark.parametrize("mutation_phase", ["reduce", "statistics_plan"])
def test_changed_preliminary_source_cannot_normalize_a_stable_peer(tmp_path: Path, mutation_phase: str) -> None:
    """Anti-vacuity: a rejected source must not collapse the stable peer's ordinary object keys."""
    stable = tmp_path / "stable.jsonl"
    changing = tmp_path / "changing.jsonl"

    def record(session_id: str, metadata: dict[str, int]) -> str:
        return json.dumps(
            {
                "type": "user",
                "sessionId": session_id,
                "version": "1.2.3",
                "message": {"role": "user", "content": "synthetic", "metadata": metadata},
            }
        )

    stable.write_text(record("stable", {"keep": 1}) + "\n", encoding="utf-8")
    changing.write_text(
        "".join(record("changing", {f"key-{index:03d}": index}) + "\n" for index in range(256)), encoding="utf-8"
    )
    mutated = False

    def replace_after_structure(phase: str, _payload: JSONDocument) -> None:
        nonlocal mutated
        if phase == mutation_phase and not mutated:
            changing.write_text(record("changing", {"replacement": 1}) + "\n", encoding="utf-8")
            mutated = True

    result = infer_sources(
        (SchemaSourceInput("claude-code", tmp_path),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
        progress=replace_after_structure,
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_record_stream"]
    )

    assert mutated
    assert result.terminal_counts == {"changed_during_read": 1, "included": 1}
    assert "$.message.metadata" not in evidence.normalization_paths
    assert "$.message.metadata.keep" in evidence.fields


def test_identical_headerless_codex_captures_count_once(tmp_path: Path) -> None:
    """Anti-vacuity: path-derived fallback identities doubled byte-identical headerless captures."""
    record = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "same direct message"}],
    }
    payload = json.dumps(record) + "\n"
    (tmp_path / "first.jsonl").write_text(payload, encoding="utf-8")
    (tmp_path / "second.jsonl").write_text(payload, encoding="utf-8")

    result = infer_sources(
        (SchemaSourceInput("codex", tmp_path),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for item in result.evidence_by_element["session_record_stream"]
    )

    assert result.terminal_counts == {"included": 1}
    assert result.included_candidate_count == 2
    assert result.included_native_source_revision_count == 1
    assert evidence.current_source_count == 1
    assert evidence.current_record_count == 1
