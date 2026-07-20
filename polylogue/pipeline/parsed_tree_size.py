"""Cheap structural size estimation for parsed session trees.

Shared by the daemon parse-prefetch cache (polylogue-xb4i) and the
historical-backfill census spill's decoded layer: any component that retains
``ParsedSession`` trees in RAM budgets them by ESTIMATED TREE BYTES, never by
raw payload bytes (parsed trees inflate payload size by roughly 2-14x
depending on text density -- see the calibration data below).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from polylogue.sources.parsers.base import ParsedSession

# Calibration (measured 2026-07-20, see test_parse_prefetch.py for the exact
# reproducer): a manual deep-object-graph walk (sys.getsizeof over every
# reachable dict/list/model instance, the same technique pympler.asizeof
# uses, without adding a new dependency for one calibration script) against
# synthetic ParsedSession trees of increasing size gave:
#
#   messages=10,  blocks=20,   payload_chars=3_000     deep_bytes=41_165   (13.7x payload chars)
#   messages=100, blocks=200,  payload_chars=150_000   deep_bytes=368_795  (2.5x payload chars)
#   messages=500, blocks=1000, payload_chars=1_500_000 deep_bytes=2_521_995 (1.7x payload chars)
#
# A single per-char multiplier alone under-fits small/medium trees (fixed
# per-object overhead dominates there) and a single per-object constant alone
# under-fits text-heavy trees. A two-term linear fit (bytes_per_char * chars +
# object_overhead_bytes * object_count, least-squares against the three
# points above) recovers ~0.39 bytes/char and ~1290 bytes/object. This module
# rounds BOTH terms up for a deliberate safety margin (favor overestimating
# resident size, which biases toward eviction/reparse -- always correct --
# over underestimating, which risks the exact OOM this budget exists to
# prevent): 2 bytes/char and 1024 bytes/object, which lands within [0.9x,
# 1.8x] of the measured deep size across the three calibration points.
_ESTIMATOR_BYTES_PER_CHAR = 2
_ESTIMATOR_OBJECT_OVERHEAD_BYTES = 1024


def _text_len(value: str | None) -> int:
    return len(value) if value else 0


def _mapping_char_len(mapping: Mapping[str, object] | None) -> int:
    """Cheap, non-recursive-getsizeof approximation of a dict-like field's size.

    ``tool_input``/``metadata``/session-event ``payload`` fields are
    provider-controlled dicts, occasionally large (a tool call's full JSON
    args). A single ``repr()`` pass over each value is O(size) but touches
    every byte, exactly the kind of per-object deep walk this estimator is
    designed to avoid paying for the WHOLE tree -- so cap what a single
    mapping field is allowed to contribute by falling back to ``str`` length
    for string values (the overwhelmingly common case) and only ``repr``ing
    non-string values, which are typically short (numbers, bools, small
    nested structures).
    """
    if not mapping:
        return 0
    total = 0
    for key, value in mapping.items():
        total += len(str(key))
        total += len(value) if isinstance(value, str) else len(repr(value))
    return total


def estimate_parsed_tree_bytes(sessions: Sequence[ParsedSession]) -> int:
    """Cheap structural estimate of resident bytes for a parsed session tree.

    Deliberately NOT a recursive ``sys.getsizeof``/pympler-style deep walk --
    that is accurate but O(object graph size) with real per-call overhead,
    and this runs on ``warm()``'s hot path for every raw in a page (up to a
    couple thousand). Instead: a single linear pass sums text/content field
    lengths and counts model-instance nodes (sessions, messages, blocks,
    attachments, session events, web constructs), then applies two constants
    calibrated against a real deep-size measurement -- see the constants'
    docstring/comment above for the calibration data and measured ratio.
    """
    total_chars = 0
    object_count = 0
    total_inline_attachment_bytes = 0
    for session in sessions:
        object_count += 1
        total_chars += _text_len(session.title)
        total_chars += _text_len(session.instructions_text)
        total_chars += _text_len(session.created_at)
        total_chars += _text_len(session.updated_at)
        total_chars += _text_len(session.git_branch)
        total_chars += _text_len(session.git_repository_url)
        total_chars += _text_len(session.provider_project_ref)
        total_chars += _text_len(session.git_commit_hash)
        total_chars += sum(len(value) for value in session.working_directories)
        total_chars += sum(len(value) for value in session.models_used)
        total_chars += sum(len(value) for value in session.ingest_flags)

        for message in session.messages:
            object_count += 1
            object_count += len(message.paste_spans)
            total_chars += _text_len(message.text)
            total_chars += _text_len(message.timestamp)
            total_chars += _text_len(message.sender_name)
            total_chars += _text_len(message.recipient)
            total_chars += _text_len(message.user_context_text)
            total_chars += _text_len(message.model_name)
            total_chars += _text_len(message.model_effort)
            total_chars += _text_len(message.provider_message_id)
            total_chars += _text_len(message.parent_message_provider_id)

            for block in message.blocks:
                object_count += 1
                total_chars += _text_len(block.text)
                total_chars += _text_len(block.tool_name)
                total_chars += _text_len(block.tool_id)
                total_chars += _text_len(block.media_type)
                total_chars += _mapping_char_len(block.tool_input)
                total_chars += _mapping_char_len(block.metadata)
                for construct in block.web_constructs:
                    object_count += 1
                    total_chars += _text_len(construct.title)
                    total_chars += _text_len(construct.text)
                    total_chars += _text_len(construct.url)

        for attachment in session.attachments:
            object_count += 1
            total_chars += _text_len(attachment.name)
            total_chars += _text_len(attachment.path)
            total_chars += _text_len(attachment.mime_type)
            total_chars += _text_len(attachment.source_url)
            total_chars += _text_len(attachment.caption)
            if attachment.inline_bytes is not None:
                # Embedded attachment payloads stay resident in the decoded
                # session (and in the spill pickle) byte-for-byte -- count
                # them directly, no char multiplier.
                total_inline_attachment_bytes += len(attachment.inline_bytes)

        for event in session.session_events:
            object_count += 1
            total_chars += _text_len(event.event_type)
            total_chars += _mapping_char_len(event.payload)

    return (
        _ESTIMATOR_OBJECT_OVERHEAD_BYTES * object_count
        + _ESTIMATOR_BYTES_PER_CHAR * total_chars
        + total_inline_attachment_bytes
    )


def effective_physical_memory_bytes() -> int | None:
    """Physical RAM available to THIS process: host RAM capped by cgroup limit.

    Under a container/systemd cgroup memory limit, ``SC_PHYS_PAGES`` reports
    host RAM; sizing an adaptive cache from it starves a memory-limited
    process. Reads cgroup v2 ``memory.max`` (and legacy v1
    ``memory.limit_in_bytes``) and returns the minimum of host RAM and any
    finite limit; ``None`` when neither is knowable.
    """

    physical: int | None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical = pages * page_size if pages > 0 and page_size > 0 else None
    except (ValueError, OSError, AttributeError):
        physical = None
    limits = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw and raw != "max" and raw.isdigit():
            value = int(raw)
            # v1 reports an enormous sentinel when unlimited
            if 0 < value < (1 << 60):
                limits.append(value)
    if limits:
        smallest = min(limits)
        return smallest if physical is None else min(physical, smallest)
    return physical


__all__ = ["estimate_parsed_tree_bytes", "effective_physical_memory_bytes"]
