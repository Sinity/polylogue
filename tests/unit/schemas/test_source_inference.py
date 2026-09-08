"""Declared-source schema inference regressions."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence
from polylogue.schemas.generation.workflow import generate_provider_schema_from_sources
from polylogue.schemas.source_inference import SchemaSourceInput, infer_sources


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
    assert result.phase_receipt["source"]["source_terminal_outcomes"] == {"included": 1}


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
    events: list[tuple[str, dict[str, object]]] = []

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
