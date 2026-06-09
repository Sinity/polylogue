"""Side-by-side session comparison logic for the reader workspace.

This module owns the diff/alignment semantics for ``/api/compare`` (issue
#1124). The HTTP layer (``workspace_routes``) is a thin wrapper that loads two
session payloads from the existing reader detail loader and forwards them
here. Keeping the diff in its own module means UI surfaces (``web_shell``) and
tests can exercise alignment + metadata-diff behaviour without spinning up the
daemon.

The compare payload shape extends the original ``mode=compare`` envelope with:

* ``metadata_diff`` — per-field deltas across origin, model, timestamps,
  message_count, repo, tags. Each field carries ``left``/``right`` and a
  ``status`` of ``"equal"``, ``"changed"``, or ``"missing"``.
* ``pairs`` — each pair now carries ``diff_status`` (``"equal"``,
  ``"changed"``, ``"added"``, ``"removed"``) and ``role_match`` so the UI can
  highlight content drift.
* ``alignment`` — ``"anchor"`` when the two sides share message anchors
  (covers fork/branch case where lineage produced identical IDs), otherwise
  ``"sequential"``. Future work (#866 lineage) can plumb richer anchors in
  without changing the envelope.

The diff stays intentionally simple — it operates on the already-projected
reader message dicts, not on raw session models, so it is cheap to call
from the request handler and easy to unit-test with dicts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from polylogue.daemon.workspace_routes import (
    COMPARE_ALIGN_MODES,
    missing_session_target,
)

Message = Mapping[str, Any]
Payload = Mapping[str, Any]


METADATA_FIELDS: tuple[str, ...] = (
    "origin",
    "model",
    "created_at",
    "updated_at",
    "message_count",
    "word_count",
    "repo",
    "branch_type",
    "title",
    "session_id",
    "tags",
)


def _is_payload(value: object) -> bool:
    return isinstance(value, Mapping)


def _normalise_for_compare(value: object) -> object:
    """Normalise a metadata value so equality is meaningful across origins.

    Lists are compared as sets (tags) — order is irrelevant for equality but
    we keep the original ordering for display via ``left``/``right`` values.
    """
    if isinstance(value, list):
        return tuple(sorted(str(v) for v in value))
    return value


def build_metadata_diff(left: Payload, right: Payload) -> dict[str, dict[str, Any]]:
    """Compute per-field metadata deltas for the compare header.

    Both sides are reader detail payloads as returned by
    ``DaemonAPIHandler._do_get_session``. Each entry carries the raw
    left/right value for display plus a status:

    * ``equal`` — present on both sides and normalised values match
    * ``changed`` — present on both sides but values differ
    * ``missing`` — only one side has a value (the other is ``None``/absent)
    """
    diff: dict[str, dict[str, Any]] = {}
    for field in METADATA_FIELDS:
        l_val = left.get(field)
        r_val = right.get(field)
        l_present = l_val not in (None, "", [], {})
        r_present = r_val not in (None, "", [], {})
        if not l_present and not r_present:
            continue
        if not l_present or not r_present:
            status = "missing"
        elif _normalise_for_compare(l_val) == _normalise_for_compare(r_val):
            status = "equal"
        else:
            status = "changed"
        diff[field] = {"left": l_val, "right": r_val, "status": status}
    return diff


def _message_anchor(msg: Message) -> str | None:
    """Stable identity for a message used for alignment.

    Prefers the explicit reader ``anchor`` (e.g. ``"message-m-c1"``) so two
    forked sessions that retained the same message ID line up. Falls back
    to the raw ``id`` and finally to ``None`` for unidentifiable messages.
    """
    for key in ("anchor", "id"):
        val = msg.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _shared_anchors(left_msgs: Sequence[Message], right_msgs: Sequence[Message]) -> set[str]:
    left_anchors = {a for a in (_message_anchor(m) for m in left_msgs) if a is not None}
    right_anchors = {a for a in (_message_anchor(m) for m in right_msgs) if a is not None}
    return left_anchors & right_anchors


def _text_equal(left: Message | None, right: Message | None) -> bool:
    if left is None or right is None:
        return False
    return (left.get("text") or "") == (right.get("text") or "")


def _classify_pair(left: Message | None, right: Message | None) -> tuple[str, bool]:
    """Return ``(diff_status, role_match)`` for a pair.

    * ``"added"``    — only right side has a message
    * ``"removed"``  — only left side has a message
    * ``"equal"``    — both sides present and text matches
    * ``"changed"``  — both sides present, text differs

    ``role_match`` is only meaningful when both sides are present.
    """
    if left is None and right is not None:
        return "added", False
    if right is None and left is not None:
        return "removed", False
    if left is None and right is None:
        # Defensive — callers shouldn't emit empty pairs but stay safe.
        return "equal", True
    assert left is not None and right is not None  # for mypy narrowing
    role_match = (left.get("role") or "") == (right.get("role") or "")
    if _text_equal(left, right) and role_match:
        return "equal", True
    return "changed", role_match


def _pair_envelope(index: int, left: Message | None, right: Message | None) -> dict[str, Any]:
    diff_status, role_match = _classify_pair(left, right)
    # ``status`` is kept for backwards compatibility with the previous
    # ``"paired"``/``"unpaired"`` shape consumed by the existing UI.
    paired_status = "paired" if left is not None and right is not None else "unpaired"
    return {
        "index": index,
        "left": left,
        "right": right,
        "status": paired_status,
        "diff_status": diff_status,
        "role_match": role_match,
    }


def _anchor_aligned_pairs(
    left_msgs: Sequence[Message],
    right_msgs: Sequence[Message],
    shared: set[str],
) -> list[dict[str, Any]]:
    """Align two message sequences by shared anchors, preserving order.

    Walks both sides in parallel, emitting unpaired entries until the next
    shared anchor catches both walkers up to the same point. This handles the
    "common parent, then divergent suffix" case the issue describes for
    branch/fork compare.
    """
    pairs: list[dict[str, Any]] = []
    i = j = 0
    index = 0
    while i < len(left_msgs) or j < len(right_msgs):
        left_msg = left_msgs[i] if i < len(left_msgs) else None
        right_msg = right_msgs[j] if j < len(right_msgs) else None
        l_anchor = _message_anchor(left_msg) if left_msg is not None else None
        r_anchor = _message_anchor(right_msg) if right_msg is not None else None
        if left_msg is not None and right_msg is not None and l_anchor is not None and l_anchor == r_anchor:
            pairs.append(_pair_envelope(index, left_msg, right_msg))
            i += 1
            j += 1
        elif left_msg is not None and l_anchor in shared and r_anchor not in shared:
            # Right side has extra content before the next shared anchor.
            pairs.append(_pair_envelope(index, None, right_msg))
            j += 1
        elif right_msg is not None and r_anchor in shared and l_anchor not in shared:
            # Left side has extra content before the next shared anchor.
            pairs.append(_pair_envelope(index, left_msg, None))
            i += 1
        else:
            # Neither side anchored (or both diverged) — pair positionally.
            pairs.append(_pair_envelope(index, left_msg, right_msg))
            if left_msg is not None:
                i += 1
            if right_msg is not None:
                j += 1
        index += 1
    return pairs


def _sequential_pairs(left_msgs: Sequence[Message], right_msgs: Sequence[Message]) -> list[dict[str, Any]]:
    max_len = max(len(left_msgs), len(right_msgs))
    pairs: list[dict[str, Any]] = []
    for idx in range(max_len):
        left_msg = left_msgs[idx] if idx < len(left_msgs) else None
        right_msg = right_msgs[idx] if idx < len(right_msgs) else None
        pairs.append(_pair_envelope(idx, left_msg, right_msg))
    return pairs


def align_messages(left: Payload | None, right: Payload | None) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(alignment, pairs)`` for two reader session payloads.

    When at least one side is missing, falls back to whatever messages remain
    on the present side. The alignment string lets the UI show the operator
    whether anchors lined up or we degraded to positional pairing.
    """
    left_msgs = _extract_messages(left)
    right_msgs = _extract_messages(right)
    shared = _shared_anchors(left_msgs, right_msgs)
    if shared:
        return "anchor", _anchor_aligned_pairs(left_msgs, right_msgs, shared)
    return "sequential", _sequential_pairs(left_msgs, right_msgs)


def _extract_messages(payload: Payload | None) -> list[Message]:
    if not _is_payload(payload):
        return []
    raw = cast(Payload, payload).get("messages", [])
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, Mapping)]


def build_compare_envelope(
    left_payload: object,
    right_payload: object,
    left_id: str,
    right_id: str,
    align: str,
) -> dict[str, Any]:
    """Assemble the full ``/api/compare`` response envelope.

    Callers should validate ``align`` against :data:`COMPARE_ALIGN_MODES`
    before calling — this function trusts the value and just records it.
    """
    left_ok = _is_payload(left_payload)
    right_ok = _is_payload(right_payload)
    left = cast(Payload, left_payload) if left_ok else None
    right = cast(Payload, right_payload) if right_ok else None
    alignment, pairs = align_messages(left, right)
    metadata_diff = build_metadata_diff(left or {}, right or {}) if left_ok and right_ok else {}
    degraded_sides: list[str] = []
    if not left_ok:
        degraded_sides.append("left")
    if not right_ok:
        degraded_sides.append("right")
    return {
        "mode": "compare",
        "align": align,
        "alignment": alignment,
        "left": left if left_ok else missing_session_target(left_id),
        "right": right if right_ok else missing_session_target(right_id),
        "pairs": pairs,
        "metadata_diff": metadata_diff,
        "total": len(pairs),
        "degraded_count": len(degraded_sides),
        "degraded_sides": degraded_sides,
    }


__all__ = [
    "COMPARE_ALIGN_MODES",
    "METADATA_FIELDS",
    "align_messages",
    "build_compare_envelope",
    "build_metadata_diff",
]
