"""Hypothesis strategies and helpers for pipeline service laws."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hypothesis import strategies as st


@dataclass(frozen=True)
class AcquisitionInputSpec:
    """One raw payload yielded by acquisition scanning."""

    payload_id: str
    provider_hint: str | None


@dataclass(frozen=True)
class ValidationCase:
    """One validation-stage contract case."""

    mode: str
    payload_kind: str
    invalid_sample_count: int
    malformed_jsonl_lines: int


@dataclass(frozen=True)
class ParseMergeEvent:
    """One ParseResult.merge_result event."""

    conversation_id: str
    result_counts: dict[str, int]
    content_changed: bool


@st.composite
def acquisition_input_batch_strategy(
    draw: st.DrawFn,
    *,
    min_items: int = 1,
    max_items: int = 6,
) -> tuple[AcquisitionInputSpec, ...]:
    """Generate acquisition inputs with duplicates and provider-fallback cases."""
    count = draw(st.integers(min_value=min_items, max_value=max_items))
    payload_ids = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
            min_size=count,
            max_size=count,
        )
    )
    provider_hints = draw(
        st.lists(
            st.sampled_from(("chatgpt", "codex", "claude-code", None)),
            min_size=count,
            max_size=count,
        )
    )
    return tuple(
        AcquisitionInputSpec(payload_id=payload_id, provider_hint=provider_hint)
        for payload_id, provider_hint in zip(payload_ids, provider_hints, strict=True)
    )


@st.composite
def validation_case_strategy(draw: st.DrawFn) -> ValidationCase:
    """Generate validation-mode cases spanning document and JSONL payloads."""
    payload_kind = draw(st.sampled_from(("document", "record")))
    mode = draw(st.sampled_from(("strict", "advisory")))
    invalid_sample_count = draw(st.integers(min_value=0, max_value=3))
    malformed_jsonl_lines = 0
    if payload_kind == "record":
        malformed_jsonl_lines = draw(st.integers(min_value=0, max_value=2))
    return ValidationCase(
        mode=mode,
        payload_kind=payload_kind,
        invalid_sample_count=invalid_sample_count,
        malformed_jsonl_lines=malformed_jsonl_lines,
    )


@st.composite
def parse_merge_events_strategy(
    draw: st.DrawFn,
    *,
    min_events: int = 1,
    max_events: int = 8,
) -> list[ParseMergeEvent]:
    """Generate merge-result event sequences with repeated conversation IDs."""
    count = draw(st.integers(min_value=min_events, max_value=max_events))
    ids = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=3, max_size=12),
            min_size=count,
            max_size=count,
        )
    )
    events: list[ParseMergeEvent] = []
    for conversation_id in ids:
        events.append(
            ParseMergeEvent(
                conversation_id=conversation_id,
                result_counts={
                    "conversations": draw(st.integers(min_value=0, max_value=1)),
                    "messages": draw(st.integers(min_value=0, max_value=5)),
                    "attachments": draw(st.integers(min_value=0, max_value=2)),
                    "skipped_conversations": draw(st.integers(min_value=0, max_value=2)),
                    "skipped_messages": draw(st.integers(min_value=0, max_value=5)),
                    "skipped_attachments": draw(st.integers(min_value=0, max_value=2)),
                },
                content_changed=draw(st.booleans()),
            )
        )
    return events


def build_acquisition_raw_bytes(spec: AcquisitionInputSpec) -> bytes:
    """Encode an acquisition input into the raw JSON bytes the stage hashes/stores."""
    return json.dumps({"id": spec.payload_id}).encode("utf-8")


def build_validation_payload(case: ValidationCase) -> tuple[bytes, str, str]:
    """Return raw bytes, provider name, and source path for a validation case."""
    if case.payload_kind == "record":
        lines: list[bytes] = [b'{"type":"session_meta"}', b'{"type":"response_item"}']
        lines.extend(b"not json" for _ in range(case.malformed_jsonl_lines))
        return b"\n".join(lines), "codex", "/tmp/session.jsonl"
    return b'{"id":"doc-1","mapping":{}}', "chatgpt", "/tmp/conversations.json"


def expected_validation_contract(case: ValidationCase) -> dict[str, Any]:
    """Return the expected persisted validation outcome for a generated case."""
    malformed_blocks = case.mode == "strict" and case.malformed_jsonl_lines > 0
    schema_invalid = case.invalid_sample_count > 0
    blocked = malformed_blocks or (case.mode == "strict" and schema_invalid)
    invalid_count = (
        1 if case.invalid_sample_count > 0 or (case.mode == "strict" and case.malformed_jsonl_lines > 0) else 0
    )
    return {
        "parseable": not blocked,
        "status": "failed" if blocked else "passed",
        "invalid_count": invalid_count,
        "mark_raw_parsed": malformed_blocks,
        "validation_samples_called": not malformed_blocks,
    }


def expected_parse_merge_totals(events: list[ParseMergeEvent]) -> dict[str, Any]:
    """Compute the aggregate ParseResult contract for a generated event list."""
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    changed_counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
    }
    processed_ids: set[str] = set()

    for event in events:
        ingest_changed = (
            event.result_counts["conversations"] + event.result_counts["messages"] + event.result_counts["attachments"]
        ) > 0
        if ingest_changed or event.content_changed:
            processed_ids.add(event.conversation_id)
        if event.content_changed:
            changed_counts["conversations"] += 1
        changed_counts["messages"] += event.result_counts["messages"]
        changed_counts["attachments"] += event.result_counts["attachments"]
        for key, value in event.result_counts.items():
            counts[key] += value

    return {
        "counts": counts,
        "changed_counts": changed_counts,
        "processed_ids": processed_ids,
    }
