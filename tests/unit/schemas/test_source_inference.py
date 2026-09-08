"""Declared-source schema inference regressions."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence
from polylogue.schemas.generation.workflow import generate_provider_schema_from_sources
from polylogue.schemas.source_inference import (
    SchemaSourceInput,
    SourceObservation,
    SourceRevision,
    _collect_payload_evidence,
    _SourceCandidate,
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

    result = infer_sources(
        (SchemaSourceInput("codex", source),), cache_path=tmp_path / "source-cache.sqlite3", max_workers=1
    )
    evidence = merge_evidence(
        SchemaEvidence.from_json(item) for rows in result.evidence_by_element.values() for item in rows
    )

    assert result.terminal_counts == {"included": 1}
    assert evidence.current_source_count >= 1
    assert result.producer_version_counts == {"1.2.3": 1}


def test_malformed_members_become_terminal_outcomes_without_aborting_inventory(tmp_path: Path) -> None:
    """Anti-vacuity: one broken ordinary source must not hide a valid sibling's evidence."""
    root = tmp_path / "sources"
    root.mkdir()
    (root / "valid.json").write_text(json.dumps([_chatgpt_conversation("valid", updated=1)]), encoding="utf-8")
    (root / "partial.jsonl").write_text('{"id":"partial"}', encoding="utf-8")
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

    infer_sources(
        (SchemaSourceInput("claude-code", source),),
        cache_path=tmp_path / "source-cache.sqlite3",
        max_workers=1,
        progress=lambda phase, payload: events.append((phase, payload)),
    )

    assert {phase for phase, _payload in events} >= {"inventory", "reduce"}
    assert all(
        {"completed_candidates", "total_candidates", "input_bytes", "record_count"} <= payload.keys()
        for _, payload in events
    )
    assert all(str(source) not in json.dumps(payload) for _, payload in events)


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
