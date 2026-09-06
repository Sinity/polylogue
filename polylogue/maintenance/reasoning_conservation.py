"""Per-origin reasoning conservation: acquired reasoning bytes to thinking blocks.

The forward universe is the acquired payload of every materialized coding-origin
raw, re-read from the blob store. Each reasoning witness inside it -- a Claude
Code ``thinking`` segment (text-bearing, signature-only, or empty) and a Codex
``reasoning`` response item (summary-bearing, content-bearing, or opaque) -- is
traced to the exact session, message and block that carries it, and lands in
exactly one typed term of the same ``source-conservation`` vocabulary
(:class:`~polylogue.maintenance.source_conservation.ConservationTerm`).

Witnesses are selected by structural predicates over the acquired bytes, never
by session identity, path or any personal identifier, and never by the parser
being checked: a shape the parser silently drops emits no row, no event and no
refusal, so only re-reading the bytes can see it. That is the exact class of
polylogue-vf9x, where a text-empty Claude thinking segment and every standalone
Codex reasoning record vanished while message volume rose.

Denominators and outcomes are per origin and per structural variant. A nonzero
global aggregate proves nothing here: an origin whose evidence was never read
is its own blocking term (``origin_evidence_absent``), and a truncated scan is
another (``scan_truncated``), so one origin's conserved witnesses can never
stand in for another origin's lost ones.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from polylogue.core.json import JSONDocument, json_document
from polylogue.maintenance.source_conservation import ConservationTerm
from polylogue.storage.introspection import column_exists, table_exists

DEFAULT_SAMPLE_LIMIT = 10

ORIGIN_CLAUDE_CODE = "claude-code-session"
ORIGIN_CODEX = "codex-session"

#: The origins whose acquired payloads carry reasoning material. Every one of
#: them must be measured for the report to be acceptable; see
#: ``origin_evidence_absent``.
REASONING_ORIGINS: tuple[str, ...] = (ORIGIN_CLAUDE_CODE, ORIGIN_CODEX)

# Structural variants, per origin. Each names one shape the acquired bytes can
# take and what conserving it means. A witness that matches none of them is
# ``unknown_shape`` and blocks.
VARIANT_TEXT = "text_bearing"
VARIANT_SIGNATURE_ONLY = "signature_only"
VARIANT_EMPTY = "empty_reasoning"
VARIANT_SUMMARY = "summary_bearing"
VARIANT_CONTENT = "content_bearing"
VARIANT_OPAQUE = "opaque_reasoning"
VARIANT_UNKNOWN = "unknown_shape"

DECLARED_VARIANTS: dict[str, tuple[str, ...]] = {
    ORIGIN_CLAUDE_CODE: (VARIANT_TEXT, VARIANT_SIGNATURE_ONLY, VARIANT_EMPTY),
    ORIGIN_CODEX: (VARIANT_SUMMARY, VARIANT_CONTENT, VARIANT_OPAQUE),
}

TERM_MATERIALIZED = "reasoning_materialized"
TERM_KIND_COLLAPSED = "reasoning_kind_collapsed"
TERM_MATERIAL_LOST = "reasoning_material_lost"
TERM_CARRIER_ABSENT = "reasoning_carrier_absent"
TERM_UNTRACEABLE = "reasoning_identity_untraceable"
TERM_UNKNOWN_SHAPE = "reasoning_unknown_shape"
TERM_SIGNATURE_UNSUPPORTED = "reasoning_signature_unsupported"
TERM_EVIDENCE_UNREADABLE = "reasoning_evidence_unreadable"
TERM_ORIGIN_EVIDENCE_ABSENT = "origin_evidence_absent"
TERM_SCAN_TRUNCATED = "scan_truncated"

_RULES: dict[str, str] = {
    TERM_MATERIALIZED: "the witness's material is carried by a thinking block of the traced message",
    TERM_KIND_COLLAPSED: (
        "the witness's material survives in the traced message under a block kind that is not thinking, "
        "so reasoning is indistinguishable from ordinary content on every read path"
    ),
    TERM_MATERIAL_LOST: "the traced carrier is materialized, yet no block of it carries the witness's material",
    TERM_CARRIER_ABSENT: (
        "the raw is materialized, yet the session or message the witness belongs to is absent "
        "from the index and from the carrier's inherited prefix"
    ),
    TERM_UNTRACEABLE: "the witness carries no provider identity to trace it by",
    TERM_UNKNOWN_SHAPE: "structurally a reasoning record, matching no declared variant of its origin",
    TERM_SIGNATURE_UNSUPPORTED: (
        "the index tier carries no blocks.signature column, so signature-only thinking has nowhere to land"
    ),
    TERM_EVIDENCE_UNREADABLE: "the acquired payload of a materialized raw is unreadable, so its witnesses are unknown",
    TERM_ORIGIN_EVIDENCE_ABSENT: (
        "a declared reasoning origin contributed no readable source evidence, so nothing about it was measured"
    ),
    TERM_SCAN_TRUNCATED: "the scan stopped at its bound, so the per-origin denominators are incomplete",
}

_BLOCKING: frozenset[str] = frozenset(
    {
        TERM_KIND_COLLAPSED,
        TERM_MATERIAL_LOST,
        TERM_CARRIER_ABSENT,
        TERM_UNTRACEABLE,
        TERM_UNKNOWN_SHAPE,
        TERM_SIGNATURE_UNSUPPORTED,
        TERM_ORIGIN_EVIDENCE_ABSENT,
        TERM_SCAN_TRUNCATED,
    }
)

_WARNING: frozenset[str] = frozenset({TERM_EVIDENCE_UNREADABLE})

_TERM_ORDER: tuple[str, ...] = (
    TERM_MATERIALIZED,
    TERM_KIND_COLLAPSED,
    TERM_MATERIAL_LOST,
    TERM_CARRIER_ABSENT,
    TERM_UNTRACEABLE,
    TERM_UNKNOWN_SHAPE,
    TERM_SIGNATURE_UNSUPPORTED,
    TERM_EVIDENCE_UNREADABLE,
    TERM_ORIGIN_EVIDENCE_ABSENT,
    TERM_SCAN_TRUNCATED,
)


@dataclass(frozen=True, slots=True)
class ReasoningWitness:
    """One reasoning unit read out of acquired bytes, before any archive lookup."""

    origin: str
    variant: str
    #: Provider message identity for origins that carry one; ``None`` when the
    #: origin addresses reasoning positionally (Codex).
    carrier_native_id: str | None
    ordinal: int
    material: tuple[str, ...] = ()
    signature: str | None = None

    def locator(self, raw_id: str) -> str:
        carrier = self.carrier_native_id or "<positional>"
        return f"{raw_id}:{carrier}:{self.ordinal}"


@dataclass(frozen=True, slots=True)
class OriginReasoningReport:
    """One origin's complete denominator and its outcome, on its own."""

    origin: str
    raws_selected: int
    raws_scanned: int
    raws_unreadable: int
    bytes_scanned: int
    witnesses: int
    witnesses_by_variant: dict[str, int]
    outcomes_by_term: dict[str, int]
    truncated: bool

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "raws_selected": self.raws_selected,
                "raws_scanned": self.raws_scanned,
                "raws_unreadable": self.raws_unreadable,
                "bytes_scanned": self.bytes_scanned,
                "witnesses": self.witnesses,
                "witnesses_by_variant": dict(sorted(self.witnesses_by_variant.items())),
                "outcomes_by_term": dict(sorted(self.outcomes_by_term.items())),
                "truncated": self.truncated,
            }
        )


@dataclass(frozen=True, slots=True)
class ReasoningConservationReport:
    """Per-origin reasoning denominators and the typed outcome of each witness."""

    origins: tuple[OriginReasoningReport, ...]
    terms: tuple[ConservationTerm, ...]

    @property
    def witness_total(self) -> int:
        return sum(entry.witnesses for entry in self.origins)

    @property
    def blocking_count(self) -> int:
        return sum(term.count for term in self.terms if term.blocking)

    @property
    def warning_count(self) -> int:
        return sum(term.count for term in self.terms if term.name in _WARNING)

    def origin(self, name: str) -> OriginReasoningReport:
        for entry in self.origins:
            if entry.origin == name:
                return entry
        raise KeyError(name)

    def term(self, name: str) -> ConservationTerm:
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(name)

    def summary(self) -> str:
        parts = [
            "; ".join(
                f"{entry.origin}: {entry.witnesses:,} witness(es) over {entry.raws_scanned:,} raw(s)"
                for entry in self.origins
            )
            or "no declared reasoning origin"
        ]
        for term in self.terms:
            if term.count and term.name != TERM_MATERIALIZED:
                parts.append(f"{term.name}={term.count:,}{'!' if term.blocking else ''}")
        return "; ".join(parts)

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "witness_total": self.witness_total,
                "blocking_count": self.blocking_count,
                "warning_count": self.warning_count,
                "origins": {entry.origin: entry.to_json() for entry in self.origins},
                "terms": {term.name: term.to_json() for term in self.terms},
            }
        )


# ---------------------------------------------------------------------------
# Source-side witness enumeration.
#
# Structural only: these predicates read the acquired bytes and know nothing
# about the parser, its exclusions or the current index. They must stay that
# way -- a denominator derived from the code under test proves nothing.
# ---------------------------------------------------------------------------


def _json_lines(payload: bytes) -> Iterator[dict[str, Any]]:
    for line in payload.splitlines():
        line = line.strip()
        if not line or not line.startswith(b"{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            yield record


def _recoverable_texts(value: object) -> tuple[str, ...]:
    """Non-empty text out of an OpenAI Responses-shaped ``summary``/``content``.

    Either a bare string or a list of ``{"text": ...}``-shaped items. Anything
    else carries no recoverable text, which is a genuine absence rather than a
    shape this cannot read.
    """
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, list):
        return ()
    texts: list[str] = []
    for item in value:
        text = item.get("text") if isinstance(item, dict) else item
        if isinstance(text, str) and text:
            texts.append(text)
    return tuple(texts)


def claude_code_witnesses(payload: bytes) -> Iterator[ReasoningWitness]:
    """Every ``thinking`` segment in an acquired Claude Code session payload."""
    for record in _json_lines(payload):
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        # Claude Code's message identity is the record-level ``uuid``; the
        # provider's own ``message.id`` is a second, weaker identity that the
        # archive does not key on.
        native_id = record.get("uuid")
        carrier = native_id if isinstance(native_id, str) and native_id else None
        ordinal = 0
        for segment in content:
            if not isinstance(segment, dict) or segment.get("type") != "thinking":
                continue
            text = segment.get("thinking")
            if not isinstance(text, str) or not text:
                text = segment.get("text") if isinstance(segment.get("text"), str) else ""
            signature = segment.get("signature")
            signature = signature if isinstance(signature, str) and signature else None
            variant: str
            material: tuple[str, ...]
            if text:
                variant, material = VARIANT_TEXT, (str(text),)
            elif signature is not None:
                variant, material = VARIANT_SIGNATURE_ONLY, ()
            elif set(segment) <= {"type", "thinking", "text", "signature"}:
                variant, material = VARIANT_EMPTY, ()
            else:
                variant, material = VARIANT_UNKNOWN, ()
            yield ReasoningWitness(
                origin=ORIGIN_CLAUDE_CODE,
                variant=variant,
                carrier_native_id=carrier,
                ordinal=ordinal,
                material=material,
                signature=signature,
            )
            ordinal += 1


def codex_witnesses(payload: bytes) -> Iterator[ReasoningWitness]:
    """Every standalone ``reasoning`` record in an acquired Codex rollout payload.

    Codex writes reasoning either as a bare record or wrapped in a
    ``response_item`` envelope; both are the same unit and both are counted.
    """
    ordinal = 0
    for record in _json_lines(payload):
        payload_object = record.get("payload")
        body = payload_object if isinstance(payload_object, dict) else record
        if body.get("type") != "reasoning":
            continue
        summary = _recoverable_texts(body.get("summary"))
        content = _recoverable_texts(body.get("content"))
        if summary:
            variant, material = VARIANT_SUMMARY, summary + content
        elif content:
            variant, material = VARIANT_CONTENT, content
        elif "summary" in body or "content" in body or "encrypted_content" in body:
            variant, material = VARIANT_OPAQUE, ()
        else:
            variant, material = VARIANT_UNKNOWN, ()
        yield ReasoningWitness(
            origin=ORIGIN_CODEX,
            variant=variant,
            carrier_native_id=None,
            ordinal=ordinal,
            material=material,
        )
        ordinal += 1


_ENUMERATORS: dict[str, Callable[[bytes], Iterator[ReasoningWitness]]] = {
    ORIGIN_CLAUDE_CODE: claude_code_witnesses,
    ORIGIN_CODEX: codex_witnesses,
}


# ---------------------------------------------------------------------------
# Index-side carriers.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Block:
    block_type: str
    text: str | None
    signature: str | None
    consumed: bool = False


@dataclass(slots=True)
class _Carrier:
    """The blocks reachable from one traced carrier, and what has been claimed."""

    session_id: str
    blocks: list[_Block] = field(default_factory=list)

    def claim_thinking(self, predicate: Callable[[_Block], bool]) -> bool:
        for block in self.blocks:
            if block.consumed or block.block_type not in ("thinking", "reasoning"):
                continue
            if predicate(block):
                block.consumed = True
                return True
        return False

    def claim_any(self, predicate: Callable[[_Block], bool]) -> bool:
        for block in self.blocks:
            if block.consumed:
                continue
            if predicate(block):
                block.consumed = True
                return True
        return False


def _session_ids_for_raw(index: sqlite3.Connection, raw_id: str) -> list[str]:
    return [str(row[0]) for row in index.execute("SELECT session_id FROM sessions WHERE raw_id = ?", (raw_id,))]


def _with_ancestors(index: sqlite3.Connection, session_ids: Sequence[str]) -> list[str]:
    """Include inherited prefixes: a child stores only its divergent tail.

    A witness replayed from the parent's prefix legitimately lives in the
    parent's rows, so the carrier pool is the session plus its ancestor chain.
    """
    seen: list[str] = []
    pending = list(session_ids)
    known: set[str] = set()
    while pending:
        session_id = pending.pop()
        if session_id in known:
            continue
        known.add(session_id)
        seen.append(session_id)
        row = index.execute("SELECT parent_session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is not None and row[0]:
            pending.append(str(row[0]))
    return seen


def _load_carriers(
    index: sqlite3.Connection, session_ids: Sequence[str], *, has_signature: bool
) -> tuple[dict[str, _Carrier], _Carrier]:
    """Return per-message carriers keyed by native id, plus the pooled carrier.

    The pooled carrier holds every block of the selected sessions and serves
    origins that address reasoning positionally.
    """
    by_native: dict[str, _Carrier] = {}
    pooled = _Carrier(session_id=session_ids[0] if session_ids else "")
    if not session_ids:
        return by_native, pooled
    placeholders = ",".join("?" for _ in session_ids)
    signature_column = "b.signature" if has_signature else "NULL"
    rows = index.execute(
        f"""
        SELECT m.native_id, m.session_id, b.block_type, b.text, {signature_column}
        FROM messages AS m
        JOIN blocks AS b ON b.message_id = m.message_id
        WHERE m.session_id IN ({placeholders})
        ORDER BY m.session_id, m.position, m.variant_index, b.position
        """,
        tuple(session_ids),
    ).fetchall()
    for native_id, session_id, block_type, text, signature in rows:
        block = _Block(
            block_type=str(block_type),
            text=None if text is None else str(text),
            signature=None if signature is None else str(signature),
        )
        pooled.blocks.append(block)
        if native_id is not None:
            carrier = by_native.setdefault(str(native_id), _Carrier(session_id=str(session_id)))
            carrier.blocks.append(block)
    return by_native, pooled


def _classify(
    witness: ReasoningWitness,
    carrier: _Carrier | None,
    *,
    has_signature: bool,
) -> str:
    if witness.variant == VARIANT_UNKNOWN:
        return TERM_UNKNOWN_SHAPE
    if carrier is None:
        return TERM_CARRIER_ABSENT
    if witness.material:
        exact = witness.origin == ORIGIN_CLAUDE_CODE
        for material in witness.material:

            def carries(block: _Block, material: str = material, exact: bool = exact) -> bool:
                if block.text is None:
                    return False
                return block.text == material if exact else material in block.text

            if carrier.claim_thinking(carries):
                continue
            if carrier.claim_any(carries):
                return TERM_KIND_COLLAPSED
            return TERM_MATERIAL_LOST
        return TERM_MATERIALIZED
    if witness.variant == VARIANT_SIGNATURE_ONLY:
        if not has_signature:
            return TERM_SIGNATURE_UNSUPPORTED
        signature = witness.signature
        if carrier.claim_thinking(lambda block: block.signature == signature):
            return TERM_MATERIALIZED
        return TERM_MATERIAL_LOST
    if carrier.claim_thinking(lambda block: block.text is None):
        return TERM_MATERIALIZED
    return TERM_MATERIAL_LOST


# ---------------------------------------------------------------------------
# The check.
# ---------------------------------------------------------------------------


def _materialized_raws(source: sqlite3.Connection, index: sqlite3.Connection, origin: str) -> list[tuple[str, str]]:
    """Materialized raws of one origin, newest revision of each logical source first.

    Restricting the cohort to raws the index actually materialized is what
    makes ``reasoning_carrier_absent`` mean something: a raw the archive
    deliberately excluded is source-conservation's term, not this one's, and
    counting it here would report a typed exclusion as reasoning loss.
    """
    indexed = {str(row[0]) for row in index.execute("SELECT raw_id FROM sessions WHERE raw_id IS NOT NULL")}
    rows = source.execute(
        """
        SELECT raw_id, lower(hex(blob_hash)),
               COALESCE(logical_source_key, source_path, native_id, raw_id) AS cohort,
               COALESCE(acquired_at_ms, 0)
        FROM raw_sessions
        WHERE origin = ? AND blob_hash IS NOT NULL
        ORDER BY cohort, COALESCE(acquired_at_ms, 0) DESC, raw_id DESC
        """,
        (origin,),
    ).fetchall()
    selected: list[tuple[str, str]] = []
    seen_cohorts: set[str] = set()
    for raw_id, blob_hash, cohort, _acquired in rows:
        if str(cohort) in seen_cohorts or str(raw_id) not in indexed:
            continue
        seen_cohorts.add(str(cohort))
        selected.append((str(raw_id), str(blob_hash)))
    return selected


def audit_reasoning_conservation(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    read_blob: Callable[[str], bytes],
    *,
    origins: Iterable[str] = REASONING_ORIGINS,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    max_raws_per_origin: int | None = None,
) -> ReasoningConservationReport:
    """Type every reasoning witness of every declared origin, per structural variant.

    ``max_raws_per_origin`` bounds an exploratory run. A bounded run that
    actually truncates reports ``scan_truncated`` and blocks, so a partial
    scan can never be read as acceptance.
    """
    has_signature = column_exists(index, "blocks", "signature")
    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    breakdowns: dict[str, dict[str, int]] = {}
    origin_reports: list[OriginReasoningReport] = []

    def record(term: str, key: str, locator: str) -> None:
        counts[term] = counts.get(term, 0) + 1
        bucket = samples.setdefault(term, [])
        if len(bucket) < sample_limit:
            bucket.append(locator)
        by = breakdowns.setdefault(term, {})
        by[key] = by.get(key, 0) + 1

    for origin in origins:
        enumerate_witnesses = _ENUMERATORS.get(origin)
        if enumerate_witnesses is None:
            continue
        selected = _materialized_raws(source, index, origin)
        scanned = 0
        unreadable = 0
        bytes_scanned = 0
        witnesses = 0
        by_variant: dict[str, int] = dict.fromkeys(DECLARED_VARIANTS.get(origin, ()), 0)
        by_term: dict[str, int] = {}
        truncated = max_raws_per_origin is not None and len(selected) > max_raws_per_origin
        cohort = selected[:max_raws_per_origin] if max_raws_per_origin is not None else selected
        for raw_id, blob_hash in cohort:
            try:
                payload = read_blob(blob_hash)
            except (OSError, ValueError):
                unreadable += 1
                record(TERM_EVIDENCE_UNREADABLE, origin, raw_id)
                by_term[TERM_EVIDENCE_UNREADABLE] = by_term.get(TERM_EVIDENCE_UNREADABLE, 0) + 1
                continue
            scanned += 1
            bytes_scanned += len(payload)
            session_ids = _with_ancestors(index, _session_ids_for_raw(index, raw_id))
            by_native, pooled = _load_carriers(index, session_ids, has_signature=has_signature)
            for witness in enumerate_witnesses(payload):
                witnesses += 1
                by_variant[witness.variant] = by_variant.get(witness.variant, 0) + 1
                if witness.origin == ORIGIN_CLAUDE_CODE and witness.carrier_native_id is None:
                    term = TERM_UNTRACEABLE
                elif witness.carrier_native_id is not None:
                    term = _classify(witness, by_native.get(witness.carrier_native_id), has_signature=has_signature)
                else:
                    term = _classify(witness, pooled if session_ids else None, has_signature=has_signature)
                record(term, f"{origin}:{witness.variant}", witness.locator(raw_id))
                by_term[term] = by_term.get(term, 0) + 1
        if truncated:
            record(TERM_SCAN_TRUNCATED, origin, f"{origin}:{len(cohort)}/{len(selected)}")
            by_term[TERM_SCAN_TRUNCATED] = by_term.get(TERM_SCAN_TRUNCATED, 0) + 1
        if not scanned:
            record(TERM_ORIGIN_EVIDENCE_ABSENT, origin, origin)
            by_term[TERM_ORIGIN_EVIDENCE_ABSENT] = by_term.get(TERM_ORIGIN_EVIDENCE_ABSENT, 0) + 1
        origin_reports.append(
            OriginReasoningReport(
                origin=origin,
                raws_selected=len(selected),
                raws_scanned=scanned,
                raws_unreadable=unreadable,
                bytes_scanned=bytes_scanned,
                witnesses=witnesses,
                witnesses_by_variant=by_variant,
                outcomes_by_term=by_term,
                truncated=truncated,
            )
        )

    terms = tuple(
        ConservationTerm(
            name=name,
            count=counts.get(name, 0),
            rule=_RULES[name],
            blocking=name in _BLOCKING,
            sample=tuple(samples.get(name, ())),
            breakdown=dict(sorted(breakdowns.get(name, {}).items())),
        )
        for name in _TERM_ORDER
    )
    return ReasoningConservationReport(origins=tuple(origin_reports), terms=terms)


def reasoning_populations_present(source: sqlite3.Connection) -> bool:
    """Whether the source tier can be asked about reasoning at all."""
    return table_exists(source, "raw_sessions")


__all__ = [
    "DECLARED_VARIANTS",
    "DEFAULT_SAMPLE_LIMIT",
    "ORIGIN_CLAUDE_CODE",
    "ORIGIN_CODEX",
    "REASONING_ORIGINS",
    "OriginReasoningReport",
    "ReasoningConservationReport",
    "ReasoningWitness",
    "audit_reasoning_conservation",
    "claude_code_witnesses",
    "codex_witnesses",
    "reasoning_populations_present",
]
