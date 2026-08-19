"""Read-only corpus acceptance measurements.

The measurements compare durable source evidence with the indexed read model
through the same archive tiers used by the production maintenance gate. Each
measure is reported separately so one unresolved corpus defect cannot become a
generic green result.

Revision counts are only approximately comparable across schema generations:
``raw_session_memberships.message_count`` is historical parser evidence while
the current index may represent former messages as ``session_events``. Event
reclassification is reported as an explanation, not silently treated as a
pass for a genuinely missing message.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any

DEFAULT_SAMPLE_LIMIT = 10

#: Origin token whose raw payloads the parse-boundary conservation census reads.
CHATGPT_CONSERVATION_ORIGIN = "chatgpt-export"

#: Keys inside a ChatGPT ``content`` object that describe the payload rather
#: than being payload. A node whose ``content`` carries only these is not
#: content-bearing, so the parser dropping it conserves nothing. ``language``
#: is the case this exists for: a ``code`` node with an empty ``text`` still
#: carries ``language: "python"``, and counting that as content would make the
#: census red for a node with nothing in it.
_CONTENT_DESCRIPTOR_KEYS = frozenset({"content_type", "language"})


def audit_absences(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Find logical documents backed by source evidence but absent from index."""
    present_ids = {str(row[0]) for row in index.execute("SELECT session_id FROM sessions")}
    by_document: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for origin, membership_provider_session_id, decision in source.execute(
        """
        SELECT r.origin, m.provider_session_id, COALESCE(m.decision, '<none>')
        FROM raw_session_memberships AS m
        JOIN raw_sessions AS r USING (raw_id)
        """
    ):
        by_document[(str(origin), str(membership_provider_session_id))].add(str(decision))

    unattributable_sample: list[str] = []
    unattributable = 0
    non_session_artifacts = 0
    for raw_id, origin, logical_source_key, native_id, status in source.execute(
        """
        SELECT r.raw_id, r.origin, r.logical_source_key, r.native_id,
               COALESCE(c.status, '')
        FROM raw_sessions AS r
        LEFT JOIN raw_membership_census AS c USING (raw_id)
        WHERE NOT EXISTS (
            SELECT 1 FROM raw_session_memberships AS m WHERE m.raw_id = r.raw_id
        )
        """
    ):
        if status == "non_session":
            non_session_artifacts += 1
            continue
        provider_session_id: str | None = None
        if logical_source_key and ":" in str(logical_source_key):
            provider_session_id = str(logical_source_key).split(":", 1)[1]
        elif native_id:
            provider_session_id = str(native_id)
        if provider_session_id:
            by_document[(str(origin), provider_session_id)].add("<byte-revision>")
        else:
            unattributable += 1
            if len(unattributable_sample) < sample_limit:
                unattributable_sample.append(str(raw_id))

    absent: collections.Counter[tuple[str, str]] = collections.Counter()
    documents_known_by_origin: collections.Counter[str] = collections.Counter()
    documents_present_by_origin: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)
    for (origin, provider_session_id), decisions in by_document.items():
        documents_known_by_origin[origin] += 1
        if f"{origin}:{provider_session_id}" in present_ids:
            documents_present_by_origin[origin] += 1
            continue
        if decisions == {"<byte-revision>"}:
            cause = "byte-revision-governed"
        elif decisions == {"ambiguous"}:
            cause = "ambiguous-only"
        elif "ambiguous" in decisions:
            cause = "mixed-ambiguous"
        else:
            cause = "settled-yet-absent"
        absent[(origin, cause)] += 1
        if len(samples[cause]) < sample_limit:
            samples[cause].append(f"{origin}:{provider_session_id}")

    return {
        "documents_known": len(by_document),
        "documents_present": len(by_document) - sum(absent.values()),
        "documents_known_by_origin": dict(sorted(documents_known_by_origin.items())),
        "documents_present_by_origin": dict(sorted(documents_present_by_origin.items())),
        "absent_total": sum(absent.values()),
        "raws_without_attributable_identity": unattributable,
        "membershipless_non_session_artifacts_excluded": non_session_artifacts,
        "absent_by_origin_cause": {
            f"{origin}/{cause}": count
            for (origin, cause), count in sorted(absent.items(), key=lambda item: (-item[1], item[0]))
        },
        "samples": dict(samples),
        "unattributable_sample": unattributable_sample,
    }


def audit_attachment_fidelity(index: sqlite3.Connection) -> dict[str, Any]:
    """Report attachment acquisition by origin, upload origin, and status.

    ``unavailable`` is terminal only when the reference retains structured
    provenance explaining where the unavailable bytes came from. An
    unprovenanced terminal status is indistinguishable from a blanket waiver
    and remains an actionable fidelity failure.
    """
    rows = index.execute(
        """
        SELECT s.origin,
               COALESCE(r.upload_origin, '<none>'),
               a.acquisition_status,
               CASE WHEN a.acquisition_status = 'unavailable'
                          AND NOT (
                              NULLIF(TRIM(COALESCE(r.upload_origin, '')), '') IS NOT NULL
                              OR NULLIF(TRIM(COALESCE(r.source_url, '')), '') IS NOT NULL
                              OR EXISTS (
                                  SELECT 1
                                  FROM attachment_native_ids AS n
                                  WHERE n.ref_id = r.ref_id
                              )
                          )
                    THEN 1 ELSE 0 END,
               COUNT(*)
        FROM attachments AS a
        JOIN attachment_refs AS r USING (attachment_id)
        JOIN sessions AS s USING (session_id)
        GROUP BY 1, 2, 3, 4
        """
    ).fetchall()
    breakdown: dict[str, int] = {}
    counts = collections.Counter[str]()
    unprovenanced_unavailable = 0
    for origin, upload_origin, status, unprovenanced, count in rows:
        amount = int(count)
        breakdown[f"{origin}/{upload_origin}/{status}"] = amount
        counts[str(status)] += amount
        if int(unprovenanced):
            unprovenanced_unavailable += amount
    return {
        "refs_acquired": counts["acquired"],
        "refs_unfetched": counts["unfetched"],
        "refs_unavailable": counts["unavailable"],
        "refs_unavailable_without_provenance": unprovenanced_unavailable,
        "refs_not_acquired": counts["unfetched"] + counts["unavailable"],
        "breakdown": breakdown,
    }


def audit_revision_fidelity(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Find indexed documents smaller than the best recorded revision."""
    best: dict[tuple[str, str], int] = {}
    for origin, provider_session_id, message_count in source.execute(
        """
        SELECT r.origin, m.provider_session_id, m.message_count
        FROM raw_session_memberships AS m
        JOIN raw_sessions AS r USING (raw_id)
        WHERE m.message_count IS NOT NULL
        """
    ):
        key = (str(origin), str(provider_session_id))
        best[key] = max(int(message_count), best.get(key, -1))

    messages = {
        str(row[0]): int(row[1]) for row in index.execute("SELECT session_id, COUNT(*) FROM messages GROUP BY 1")
    }
    events = {
        str(row[0]): int(row[1])
        for row in index.execute(
            """
            SELECT session_id, COUNT(*)
            FROM session_events
            WHERE source_message_id IS NOT NULL
               OR NULLIF(TRIM(COALESCE(source_message_provider_id, '')), '') IS NOT NULL
            GROUP BY 1
            """
        )
    }
    shortfalls: collections.Counter[str] = collections.Counter()
    explained: collections.Counter[str] = collections.Counter()
    worst: list[dict[str, Any]] = []
    for (origin, provider_session_id), best_count in best.items():
        session_id = f"{origin}:{provider_session_id}"
        have_messages = messages.get(session_id)
        if have_messages is None or have_messages >= best_count:
            continue
        have_events = events.get(session_id, 0)
        if have_messages + have_events >= best_count:
            explained[origin] += 1
            continue
        shortfalls[origin] += 1
        worst.append(
            {
                "session_id": session_id,
                "indexed_messages": have_messages,
                "indexed_events": have_events,
                "best_recorded_messages": best_count,
            }
        )
    worst.sort(key=lambda item: item["indexed_messages"] - item["best_recorded_messages"])
    return {
        "unexplained_shortfall": sum(shortfalls.values()),
        "explained_by_event_reclassification": sum(explained.values()),
        "unexplained_by_origin": dict(shortfalls.most_common()),
        "explained_by_origin": dict(explained.most_common()),
        "worst": worst[:sample_limit],
    }


def _payload_is_non_empty(value: object) -> bool:
    """Whether a JSON value carries anything a parser could materialize."""
    if value is None or value is True or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_payload_is_non_empty(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_payload_is_non_empty(item) for item in value)
    return True


def _node_content_type(content: Mapping[str, Any]) -> str:
    raw = content.get("content_type")
    return str(raw) if isinstance(raw, str) and raw else "<unset>"


def _content_bearing_nodes(mapping: Mapping[str, Any]) -> dict[str, str]:
    """Map provider message id -> content_type for every content-bearing node.

    Ground-truth side of the census: this reads the acquired ChatGPT bytes
    with no knowledge of which ``content_type`` values the parser happens to
    branch on. A node counts when it has an authored role and its ``content``
    object carries a non-empty payload field, which is exactly the set a
    lossless parse must account for. Deliberately *not* mirroring
    ``sources/parsers/chatgpt.py``'s branch table: a census whose universe is
    the parser's own vocabulary cannot see a content class the parser has
    never heard of, which is the whole class polylogue-xofj is about.
    """
    nodes: dict[str, str] = {}
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, Mapping):
            continue
        author = message.get("author")
        role = author.get("role") if isinstance(author, Mapping) else None
        if not isinstance(role, str) or not role:
            continue
        if not any(
            _payload_is_non_empty(value) for key, value in content.items() if key not in _CONTENT_DESCRIPTOR_KEYS
        ):
            continue
        provider_message_id = message.get("id") or node.get("id") or node_id
        nodes[str(provider_message_id)] = _node_content_type(content)
    return nodes


def _iter_chatgpt_conversations(payload: object) -> list[Mapping[str, Any]]:
    """Return every conversation document inside one acquired ChatGPT blob.

    A ChatGPT export is a bundle (top-level list, or an object with a
    ``conversations`` array); a browser capture or shared-page decode is a
    single conversation object. All three lower to the same per-conversation
    parse, so all three are enumerated here.
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping) and isinstance(item.get("mapping"), Mapping)]
    if not isinstance(payload, Mapping):
        return []
    if isinstance(payload.get("mapping"), Mapping):
        return [payload]
    nested = payload.get("conversations")
    if isinstance(nested, list):
        return _iter_chatgpt_conversations(nested)
    return []


def _conversation_id(conversation: Mapping[str, Any]) -> str | None:
    for key in ("id", "uuid", "conversation_id"):
        value = conversation.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def audit_chatgpt_content_conservation(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    read_blob: Callable[[str], bytes],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Census content-bearing ChatGPT raw nodes against materialized messages.

    The parse-boundary conservation measure polylogue-xofj's parent note asked
    for and no index-side check can supply: a ChatGPT node whose
    ``content_type`` the parser has no branch for, and whose text no fallback
    extracts, hits ``extract_messages_from_mapping``'s ``if not text and not
    content_blocks: continue`` and disappears leaving no row, no event, and no
    typed refusal anywhere downstream. Comparing indexed rows against other
    indexed rows cannot see it; only re-reading the acquired bytes can.

    Only the newest acquired revision of each conversation is measured. Older
    revisions are superseded by construction -- a branch the user deleted in
    ChatGPT is legitimately absent from the current index, and counting it
    would report supersession as a parser drop.

    Conversations with no indexed session at all are reported separately and
    excluded: that absence is ``corpus-absences``' finding, and double-counting
    it here would attribute a whole missing document to the parser.
    """
    indexed_sessions = {
        str(row[0])
        for row in index.execute("SELECT session_id FROM sessions WHERE origin = ?", (CHATGPT_CONSERVATION_ORIGIN,))
    }
    indexed_native_ids: dict[str, set[str]] = collections.defaultdict(set)
    for session_id, native_id in index.execute(
        """
        SELECT m.session_id, m.native_id
        FROM messages AS m
        JOIN sessions AS s USING (session_id)
        WHERE s.origin = ? AND m.native_id IS NOT NULL
        """,
        (CHATGPT_CONSERVATION_ORIGIN,),
    ):
        indexed_native_ids[str(session_id)].add(str(native_id))

    # Newest revision wins: ascending acquisition order, later raws overwrite.
    newest_nodes: dict[str, dict[str, str]] = {}
    raws_scanned = 0
    bytes_scanned = 0
    unreadable_raws: list[str] = []
    for raw_id, blob_hash in source.execute(
        """
        SELECT raw_id, blob_hash
        FROM raw_sessions
        WHERE origin = ? AND blob_hash IS NOT NULL
        ORDER BY COALESCE(acquired_at_ms, 0), raw_id
        """,
        (CHATGPT_CONSERVATION_ORIGIN,),
    ):
        try:
            blob = read_blob(bytes(blob_hash).hex())
            payload = json.loads(blob)
        except (OSError, ValueError, TypeError):
            if len(unreadable_raws) < sample_limit:
                unreadable_raws.append(str(raw_id))
            continue
        raws_scanned += 1
        bytes_scanned += len(blob)
        for conversation in _iter_chatgpt_conversations(payload):
            conversation_id = _conversation_id(conversation)
            mapping = conversation.get("mapping")
            if conversation_id is None or not isinstance(mapping, Mapping):
                continue
            newest_nodes[conversation_id] = _content_bearing_nodes(mapping)

    dropped_by_content_type: collections.Counter[str] = collections.Counter()
    conserved_by_content_type: collections.Counter[str] = collections.Counter()
    dropped_sample: list[dict[str, str]] = []
    documents_absent_from_index = 0
    documents_measured = 0
    documents_with_drops: set[str] = set()
    for conversation_id, nodes in newest_nodes.items():
        session_id = f"{CHATGPT_CONSERVATION_ORIGIN}:{conversation_id}"
        if session_id not in indexed_sessions:
            documents_absent_from_index += 1
            continue
        documents_measured += 1
        materialized = indexed_native_ids.get(session_id, set())
        for provider_message_id, content_type in nodes.items():
            if provider_message_id in materialized:
                conserved_by_content_type[content_type] += 1
                continue
            dropped_by_content_type[content_type] += 1
            documents_with_drops.add(session_id)
            if len(dropped_sample) < sample_limit:
                dropped_sample.append(
                    {
                        "session_id": session_id,
                        "provider_message_id": provider_message_id,
                        "content_type": content_type,
                    }
                )

    return {
        "raws_scanned": raws_scanned,
        "bytes_scanned": bytes_scanned,
        "unreadable_raw_sample": unreadable_raws,
        "documents_measured": documents_measured,
        "documents_absent_from_index": documents_absent_from_index,
        "documents_with_dropped_content": len(documents_with_drops),
        "content_units_conserved": sum(conserved_by_content_type.values()),
        "content_units_dropped": sum(dropped_by_content_type.values()),
        "conserved_by_content_type": dict(sorted(conserved_by_content_type.items())),
        "dropped_by_content_type": dict(dropped_by_content_type.most_common()),
        "dropped_sample": dropped_sample,
    }


__all__ = [
    "CHATGPT_CONSERVATION_ORIGIN",
    "DEFAULT_SAMPLE_LIMIT",
    "audit_absences",
    "audit_attachment_fidelity",
    "audit_chatgpt_content_conservation",
    "audit_revision_fidelity",
]
