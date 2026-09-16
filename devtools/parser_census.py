"""``devtools bench parser-census``: a real-corpus oracle for the parser stack.

A parser change is normally proven on synthetic fixtures. Fixtures answer
"does the shape I thought of still parse"; they cannot answer "did this change
make 38,828 real Claude Code transcripts parse differently". The production
rebuild re-parses the whole corpus in one pass, so a regression that only
shows at scale is exactly the class of defect that is expensive to discover
during the build and cheap to discover here.

The census walks a **recorded denominator** -- by default the schema-source
frontier baseline (``polylogue.schemas.source_frontier``), which already names
every admitted member of every declared root with its byte count and content
digest -- parses each member through the ordinary dispatch route with **no
archive** (``detect_provider`` then ``parse_payload`` /
``parse_stream_payload``, the same entry the derivation compute uses), and
records one row per member: the input identity, the classified origin, the
typed outcome, session/message/block/action counts, the tool-outcome
histogram, the parsed sessions' content hashes, and one digest over that
projection.

Nothing here opens an archive tier. The archive root is used only to *locate*
the recorded frontier and the census directory; a skewed ``source.db`` cannot
refuse this run (polylogue-pv8xp), and the operator's source trees are read
and never written.

What a census **can** report, which is the whole point of keeping one:

* a parser that starts refusing a shape it used to accept -- the member's
  outcome moves from ``parsed`` to a failure outcome and the diff reports a
  new failure, per origin and per member;
* a sample set that silently narrows -- a member the previous census covered
  that the current denominator no longer names is ``member_absent``, an error.
  This is how ``tu1f``'s stale Gemini package survived for months;
* an origin whose members stop classifying -- ``origin_changed`` (including
  to ``unknown``) is an error;
* nondeterminism -- identical input bytes and an identical parser fingerprint
  producing a different digest is ``digest_changed_nondeterministic``, an
  error. When the parser fingerprint moved, the same digest change is an
  expected notice and the per-origin table says how many members it touched.

Members the direct JSON/JSONL route does not own (zip exports, SQLite
trajectory/state databases, taxonomy-refused paths) are censused at
``depth="explain"`` through the production ``explain_import_path`` route,
which owns those containers. Depth is part of the row, so a member that
quietly changes route is itself visible in the diff.

A member larger than the declared per-member byte bound is recorded as
``oversized`` rather than parsed -- every route here holds one whole member
and its parsed sessions in memory, and the corpus contains a 16 GB export
zip. Recorded, not omitted: the bound is in the census header, the refusal is
in the member's row, and a member that crosses the bound in either direction
moves its digest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial as partial_call
from io import BytesIO
from pathlib import Path
from typing import cast

CENSUS_SCHEMA = "polylogue.parser-census.v1"
CENSUS_DIFF_SCHEMA = "polylogue.parser-census-diff.v1"
CENSUS_DIR_ENV_VAR = "POLYLOGUE_PARSER_CENSUS_DIR"

#: Terminal outcome of censusing one member. Exhaustive: every member carries
#: exactly one, and ``unavailable`` is a real outcome rather than an omission.
OUTCOME_PARSED = "parsed"
OUTCOME_NO_SESSIONS = "no_sessions"
OUTCOME_NOT_SESSION_ARTIFACT = "not_session_artifact"
OUTCOME_DECODE_FAILURE = "decode_failure"
OUTCOME_PARSE_FAILURE = "parse_failure"
OUTCOME_READ_FAILURE = "read_failure"
OUTCOME_UNAVAILABLE = "unavailable"
#: The member is larger than the declared per-member bound, so it was not
#: parsed. A declared refusal, not a failure -- but it is recorded per member,
#: so a member that crosses the bound in either direction moves its digest and
#: the diff reports it.
OUTCOME_OVERSIZED = "oversized"

OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_PARSED,
        OUTCOME_NO_SESSIONS,
        OUTCOME_NOT_SESSION_ARTIFACT,
        OUTCOME_DECODE_FAILURE,
        OUTCOME_PARSE_FAILURE,
        OUTCOME_READ_FAILURE,
        OUTCOME_UNAVAILABLE,
        OUTCOME_OVERSIZED,
    }
)

#: Outcomes that mean this member produced no parsed sessions for a reason a
#: person has to own. ``not_session_artifact`` is a declared refusal and is
#: deliberately not here.
FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_NO_SESSIONS,
        OUTCOME_DECODE_FAILURE,
        OUTCOME_PARSE_FAILURE,
        OUTCOME_READ_FAILURE,
        OUTCOME_UNAVAILABLE,
    }
)

#: Suffixes the direct JSON/JSONL route owns. Everything else is censused
#: through ``explain_import_path``, which owns zip and SQLite containers.
_DIRECT_SUFFIXES: frozenset[str] = frozenset({".json", ".jsonl", ".ndjson"})

#: Per-member byte bound. Every route here reads the whole member into memory
#: and holds its parsed sessions, so an unbounded census is a memory hazard in
#: a shared pool: the corpus contains a 16 GB ChatGPT export zip and a 1.6 GB
#: Claude Code transcript. 23 of 47,537 members exceed this bound at the
#: 2026-09-15 frontier; each is recorded as ``oversized`` rather than omitted.
DEFAULT_MAX_MEMBER_BYTES = 256_000_000

DEPTH_FULL = "full"
DEPTH_EXPLAIN = "explain"

SEVERITY_ERROR = "error"
SEVERITY_NOTICE = "notice"


class ParserCensusError(RuntimeError):
    """The census could not resolve its denominator or its baseline."""


@dataclass(frozen=True, slots=True)
class DenominatorMember:
    """One member of the recorded denominator, before it is censused."""

    subject: str
    root: str
    relative: str
    byte_count: int
    sha256: str

    @property
    def key(self) -> str:
        return f"{self.subject}:{self.root}:{self.relative}"

    @property
    def path(self) -> Path:
        root = Path(self.root)
        return root if root.is_file() and self.relative == root.name else root / self.relative


@dataclass(frozen=True, slots=True)
class MemberCensus:
    """The censused result for one denominator member."""

    subject: str
    root: str
    relative: str
    byte_count: int
    sha256: str
    depth: str
    outcome: str
    origin: str
    reason: str | None = None
    artifact_kind: str | None = None
    parser_mode: str | None = None
    sessions: int = 0
    messages: int = 0
    blocks: int = 0
    actions: int = 0
    tool_outcomes: Mapping[str, int] = field(default_factory=dict)
    session_digests: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.subject}:{self.root}:{self.relative}"

    @property
    def failed(self) -> bool:
        return self.outcome in FAILURE_OUTCOMES

    @property
    def digest(self) -> str:
        """Digest over everything the parser decided about this member.

        Input identity (``sha256``/``byte_count``) is deliberately excluded:
        the diff compares it separately so a changed input and a changed
        parser verdict are never conflated.
        """
        from polylogue.core.hashing import hash_payload

        return hash_payload(
            {
                "depth": self.depth,
                "outcome": self.outcome,
                "origin": self.origin,
                "reason": self.reason,
                "artifact_kind": self.artifact_kind,
                "parser_mode": self.parser_mode,
                "sessions": self.sessions,
                "messages": self.messages,
                "blocks": self.blocks,
                "actions": self.actions,
                "tool_outcomes": dict(sorted(self.tool_outcomes.items())),
                "session_digests": list(self.session_digests),
            }
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "subject": self.subject,
            "root": self.root,
            "relative": self.relative,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "depth": self.depth,
            "outcome": self.outcome,
            "origin": self.origin,
            "sessions": self.sessions,
            "messages": self.messages,
            "blocks": self.blocks,
            "actions": self.actions,
            "digest": self.digest,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.artifact_kind is not None:
            payload["artifact_kind"] = self.artifact_kind
        if self.parser_mode is not None:
            payload["parser_mode"] = self.parser_mode
        if self.tool_outcomes:
            payload["tool_outcomes"] = dict(sorted(self.tool_outcomes.items()))
        if self.session_digests:
            payload["session_digests"] = list(self.session_digests)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MemberCensus:
        outcomes = payload.get("tool_outcomes")
        digests = payload.get("session_digests")
        return cls(
            subject=str(payload["subject"]),
            root=str(payload["root"]),
            relative=str(payload["relative"]),
            byte_count=int(cast(int, payload.get("byte_count", 0))),
            sha256=str(payload.get("sha256", "")),
            depth=str(payload.get("depth", DEPTH_FULL)),
            outcome=str(payload["outcome"]),
            origin=str(payload.get("origin", "unknown")),
            reason=None if payload.get("reason") is None else str(payload["reason"]),
            artifact_kind=None if payload.get("artifact_kind") is None else str(payload["artifact_kind"]),
            parser_mode=None if payload.get("parser_mode") is None else str(payload["parser_mode"]),
            sessions=int(cast(int, payload.get("sessions", 0))),
            messages=int(cast(int, payload.get("messages", 0))),
            blocks=int(cast(int, payload.get("blocks", 0))),
            actions=int(cast(int, payload.get("actions", 0))),
            tool_outcomes={str(k): int(v) for k, v in cast(Mapping[str, int], outcomes or {}).items()},
            session_digests=tuple(str(item) for item in cast(Sequence[object], digests or ())),
        )


@dataclass(frozen=True, slots=True)
class Census:
    """One complete census run over a recorded denominator."""

    parser_fingerprint: str
    denominator: Mapping[str, object]
    members: tuple[MemberCensus, ...]
    recorded_at: str
    duration_seconds: float = 0.0
    workers: int = 1
    #: A bounded (``--limit``) run. A partial census is never a baseline: the
    #: members it omits are not evidence that they disappeared.
    partial: bool = False
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES

    @property
    def totals(self) -> dict[str, int]:
        totals: dict[str, int] = dict.fromkeys(sorted(OUTCOMES), 0)
        for member in self.members:
            totals[member.outcome] = totals.get(member.outcome, 0) + 1
        totals["members"] = len(self.members)
        totals["sessions"] = sum(member.sessions for member in self.members)
        totals["messages"] = sum(member.messages for member in self.members)
        totals["blocks"] = sum(member.blocks for member in self.members)
        return totals

    def by_key(self) -> dict[str, MemberCensus]:
        return {member.key: member for member in self.members}

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": CENSUS_SCHEMA,
            "recorded_at": self.recorded_at,
            "parser_fingerprint": self.parser_fingerprint,
            "denominator": dict(self.denominator),
            "partial": self.partial,
            "max_member_bytes": self.max_member_bytes,
            "workers": self.workers,
            "duration_seconds": round(self.duration_seconds, 3),
            "totals": self.totals,
            "members": [member.to_payload() for member in sorted(self.members, key=lambda item: item.key)],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Census:
        schema = str(payload.get("schema", ""))
        if schema != CENSUS_SCHEMA:
            raise ParserCensusError(f"not a parser census document: schema={schema!r}")
        members = cast(Sequence[Mapping[str, object]], payload.get("members", ()))
        return cls(
            parser_fingerprint=str(payload.get("parser_fingerprint", "")),
            denominator=cast(Mapping[str, object], payload.get("denominator", {})),
            members=tuple(MemberCensus.from_payload(item) for item in members),
            recorded_at=str(payload.get("recorded_at", "")),
            duration_seconds=float(cast(float, payload.get("duration_seconds", 0.0))),
            workers=int(cast(int, payload.get("workers", 1))),
            partial=bool(payload.get("partial", False)),
            max_member_bytes=int(cast(int, payload.get("max_member_bytes", DEFAULT_MAX_MEMBER_BYTES))),
        )


@dataclass(frozen=True, slots=True)
class CensusFinding:
    """One difference between two censuses."""

    kind: str
    severity: str
    origin: str
    member: str
    detail: str

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "origin": self.origin,
            "member": self.member,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class OriginRow:
    """Per-origin rollup of one diff."""

    origin: str
    members: int = 0
    unchanged: int = 0
    changed_digest: int = 0
    new_failures: int = 0
    recovered_failures: int = 0
    added: int = 0
    removed: int = 0
    origin_changed: int = 0
    input_changed: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "members": self.members,
            "unchanged": self.unchanged,
            "changed_digest": self.changed_digest,
            "new_failures": self.new_failures,
            "recovered_failures": self.recovered_failures,
            "added": self.added,
            "removed": self.removed,
            "origin_changed": self.origin_changed,
            "input_changed": self.input_changed,
        }


@dataclass(frozen=True, slots=True)
class CensusDiff:
    """The comparison of a census against its predecessor."""

    findings: tuple[CensusFinding, ...]
    origins: tuple[OriginRow, ...]
    parser_fingerprint_before: str
    parser_fingerprint_after: str
    compared: bool
    skipped_reason: str | None = None

    @property
    def errors(self) -> tuple[CensusFinding, ...]:
        return tuple(item for item in self.findings if item.severity == SEVERITY_ERROR)

    @property
    def notices(self) -> tuple[CensusFinding, ...]:
        return tuple(item for item in self.findings if item.severity != SEVERITY_ERROR)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def parser_moved(self) -> bool:
        return self.parser_fingerprint_before != self.parser_fingerprint_after

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": CENSUS_DIFF_SCHEMA,
            "ok": self.ok,
            "compared": self.compared,
            "skipped_reason": self.skipped_reason,
            "parser_fingerprint_before": self.parser_fingerprint_before,
            "parser_fingerprint_after": self.parser_fingerprint_after,
            "parser_moved": self.parser_moved,
            "origins": [row.to_payload() for row in self.origins],
            "findings": [finding.to_payload() for finding in self.findings],
        }


# --------------------------------------------------------------------------
# Denominator
# --------------------------------------------------------------------------


def frontier_denominator(
    frontier_file: Path | None, subjects: Sequence[str] | None = None
) -> tuple[
    tuple[DenominatorMember, ...],
    dict[str, object],
]:
    """Read the recorded schema-source frontier baseline as the denominator.

    The *recorded* baseline is used rather than a fresh walk on purpose: it is
    a conserved artifact, so a census taken a week later is comparable member
    by member, and a member the frontier no longer records shows up here as a
    removal instead of vanishing quietly.
    """
    from polylogue.schemas.source_frontier import SchemaFrontierError, frontier_path, load_frontier

    resolved = frontier_path(frontier_file)
    try:
        frontier = load_frontier(frontier_file)
    except SchemaFrontierError as exc:
        raise ParserCensusError(f"frontier unavailable at {resolved}: {exc}") from exc
    if not frontier.baselines:
        raise ParserCensusError(
            f"frontier at {resolved} has no recorded baseline; run `devtools schema frontier --record` first"
        )
    wanted = set(subjects) if subjects else None
    members: list[DenominatorMember] = []
    for baseline in frontier.baselines:
        if wanted is not None and baseline.subject not in wanted:
            continue
        for member in baseline.members:
            members.append(
                DenominatorMember(
                    subject=baseline.subject,
                    root=baseline.root,
                    relative=member.relative,
                    byte_count=member.byte_count,
                    sha256=member.sha256,
                )
            )
    declaration: dict[str, object] = {
        "kind": "schema-source-frontier",
        "frontier": str(resolved),
        "baseline_digest": frontier.baseline_digest,
        "declaration_digest": frontier.declaration_digest,
        "recorded_at": frontier.recorded_at,
        "subjects": sorted({member.subject for member in members}),
        "member_count": len(members),
    }
    return tuple(sorted(members, key=lambda item: item.key)), declaration


def source_denominator(
    sources: Sequence[tuple[str, Path]],
) -> tuple[
    tuple[DenominatorMember, ...],
    dict[str, object],
]:
    """Build an ad-hoc denominator from ``SUBJECT=PATH`` pairs.

    This is the seeded/demo-corpus route: CI has no operator frontier, so the
    denominator is the declared fixture tree, hashed as it is walked.
    """
    from polylogue.core.hashing import hash_file

    members: list[DenominatorMember] = []
    for subject, root in sources:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise ParserCensusError(f"declared source root does not exist: {resolved}")
        paths = [resolved] if resolved.is_file() else sorted(p for p in resolved.rglob("*") if p.is_file())
        for path in paths:
            relative = path.name if resolved.is_file() else path.relative_to(resolved).as_posix()
            members.append(
                DenominatorMember(
                    subject=subject,
                    root=str(resolved),
                    relative=relative,
                    byte_count=path.stat().st_size,
                    sha256=hash_file(path),
                )
            )
    declaration: dict[str, object] = {
        "kind": "declared-sources",
        "sources": [{"subject": subject, "root": str(root.expanduser().resolve())} for subject, root in sources],
        "subjects": sorted({member.subject for member in members}),
        "member_count": len(members),
    }
    return tuple(sorted(members, key=lambda item: item.key)), declaration


# --------------------------------------------------------------------------
# Censusing one member
# --------------------------------------------------------------------------


def _origin_token(provider: object) -> str:
    from polylogue.core.enums import Provider
    from polylogue.core.sources import origin_from_provider

    if not isinstance(provider, Provider):
        return "unknown"
    return str(origin_from_provider(provider).value)


def _explain_member(member: DenominatorMember, path: Path) -> MemberCensus:
    """Census a container member through the production explain route."""
    from polylogue.sources.import_explain import explain_import_path

    payload = explain_import_path(path, source_name=member.subject, limit=1)
    entry = payload.entries[0] if payload.entries else None
    if entry is None:
        reason = payload.skipped[0].reason if payload.skipped else "no entry produced"
        return MemberCensus(
            subject=member.subject,
            root=member.root,
            relative=member.relative,
            byte_count=member.byte_count,
            sha256=member.sha256,
            depth=DEPTH_EXPLAIN,
            outcome=OUTCOME_NOT_SESSION_ARTIFACT,
            origin="unknown",
            reason=reason,
        )
    produced = entry.produced
    skipped_reason = entry.skipped[0].reason if entry.skipped else None
    if produced.sessions:
        outcome = OUTCOME_PARSED
    elif skipped_reason is not None and skipped_reason.startswith("parser failure"):
        outcome = OUTCOME_PARSE_FAILURE
    elif skipped_reason is not None and skipped_reason.startswith("decode failure"):
        outcome = OUTCOME_DECODE_FAILURE
    elif skipped_reason is not None and skipped_reason.startswith("read failure"):
        outcome = OUTCOME_READ_FAILURE
    elif skipped_reason is not None:
        outcome = OUTCOME_NOT_SESSION_ARTIFACT
    else:
        outcome = OUTCOME_NO_SESSIONS
    return MemberCensus(
        subject=member.subject,
        root=member.root,
        relative=member.relative,
        byte_count=member.byte_count,
        sha256=member.sha256,
        depth=DEPTH_EXPLAIN,
        outcome=outcome,
        origin=entry.detected_origin or "unknown",
        reason=skipped_reason,
        artifact_kind=entry.artifact_kind,
        parser_mode=entry.parser_mode,
        sessions=produced.sessions,
        messages=produced.messages,
        blocks=produced.blocks,
        actions=produced.actions,
        session_digests=tuple(sorted(produced.session_refs)),
    )


def census_member(member: DenominatorMember, *, max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES) -> MemberCensus:
    """Parse one denominator member with no archive and record what happened."""
    from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
    from polylogue.core.enums import Provider
    from polylogue.core.json import JSONValue
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources import dispatch
    from polylogue.sources.decoders import _decode_json_bytes, _iter_json_stream

    path = member.path
    provider_hint = Provider.from_string(member.subject)

    def refused(outcome: str, reason: str, *, origin: str | None = None, kind: str | None = None) -> MemberCensus:
        return MemberCensus(
            subject=member.subject,
            root=member.root,
            relative=member.relative,
            byte_count=member.byte_count,
            sha256=member.sha256,
            depth=DEPTH_FULL,
            outcome=outcome,
            origin=origin if origin is not None else _origin_token(provider_hint),
            reason=reason,
            artifact_kind=kind,
        )

    try:
        live_bytes = path.stat().st_size
    except OSError:
        return refused(OUTCOME_UNAVAILABLE, "denominator member is absent from its root")
    if max_member_bytes and live_bytes > max_member_bytes:
        return refused(
            OUTCOME_OVERSIZED,
            f"member is {live_bytes} bytes, over the declared {max_member_bytes}-byte census bound",
        )
    if path.suffix.lower() not in _DIRECT_SUFFIXES:
        try:
            return _explain_member(member, path)
        except Exception as exc:  # pragma: no cover - defensive: explain owns its own refusals
            return refused(OUTCOME_PARSE_FAILURE, f"explain failure: {type(exc).__name__}: {exc}")

    path_classification = classify_artifact_path(path, provider=provider_hint)
    if path_classification is not None and not path_classification.parse_as_session:
        return refused(
            OUTCOME_NOT_SESSION_ARTIFACT,
            path_classification.reason,
            kind=path_classification.kind.value,
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return refused(OUTCOME_READ_FAILURE, f"read failure: {type(exc).__name__}: {exc}")

    stream_name = path.name
    try:
        if dispatch.is_jsonl_source_path(stream_name):
            payload: JSONValue = cast(JSONValue, list(_iter_json_stream(BytesIO(raw_bytes), stream_name)))
        else:
            text = _decode_json_bytes(raw_bytes)
            if text is None:
                return refused(OUTCOME_DECODE_FAILURE, "decode failure: unsupported JSON encoding")
            payload = cast(JSONValue, json.loads(text))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return refused(OUTCOME_DECODE_FAILURE, f"decode failure: {type(exc).__name__}: {exc}")
    finally:
        del raw_bytes

    detected = dispatch.detect_provider(payload) or provider_hint
    origin = _origin_token(detected)
    artifact = path_classification or classify_artifact(payload, provider=detected, source_path=str(path))
    if not artifact.parse_as_session:
        return refused(OUTCOME_NOT_SESSION_ARTIFACT, artifact.reason, origin=origin, kind=artifact.kind.value)

    try:
        if dispatch.is_stream_record_provider(str(path), detected):
            stream_payloads = payload if isinstance(payload, list) else [payload]
            sessions = dispatch.parse_stream_payload(detected, stream_payloads, path.stem, source_path=str(path))
        else:
            sessions = dispatch.parse_payload(detected, payload, path.stem, source_path=str(path))
    except Exception as exc:
        return refused(
            OUTCOME_PARSE_FAILURE,
            f"parser failure: {type(exc).__name__}: {exc}",
            origin=origin,
            kind=artifact.kind.value,
        )

    messages = [message for session in sessions for message in session.messages]
    blocks = [block for message in messages for block in message.blocks]
    tool_outcomes: dict[str, int] = {}
    for block in blocks:
        outcome_value = getattr(block, "tool_outcome", None)
        if outcome_value is None:
            continue
        token = str(getattr(outcome_value, "value", outcome_value))
        tool_outcomes[token] = tool_outcomes.get(token, 0) + 1
    return MemberCensus(
        subject=member.subject,
        root=member.root,
        relative=member.relative,
        byte_count=member.byte_count,
        sha256=member.sha256,
        depth=DEPTH_FULL,
        outcome=OUTCOME_PARSED if sessions else OUTCOME_NO_SESSIONS,
        origin=origin,
        reason=None if sessions else "parser produced no sessions",
        artifact_kind=artifact.kind.value,
        parser_mode=None,
        sessions=len(sessions),
        messages=len(messages),
        blocks=len(blocks),
        actions=sum(1 for block in blocks if block.type.value == "tool_use"),
        tool_outcomes=tool_outcomes,
        session_digests=tuple(sorted(str(session_content_hash(session)) for session in sessions)),
    )


def _census_member_payload(max_member_bytes: int, member: DenominatorMember) -> dict[str, object]:
    """Process-pool entry point: dataclasses cross the boundary as payloads."""
    return census_member(member, max_member_bytes=max_member_bytes).to_payload()


def build_census(
    members: Sequence[DenominatorMember],
    denominator: Mapping[str, object],
    *,
    workers: int = 1,
    partial: bool = False,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> Census:
    """Census every member, in this process when ``workers`` is 1."""
    from polylogue.sources.origin_specs import lowering_fingerprint

    started = time.monotonic()
    results: list[MemberCensus] = []
    if workers <= 1:
        results = [census_member(member, max_member_bytes=max_member_bytes) for member in members]
    else:
        worker = partial_call(_census_member_payload, max_member_bytes)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for payload in pool.map(worker, members, chunksize=16):
                results.append(MemberCensus.from_payload(payload))
    return Census(
        parser_fingerprint=lowering_fingerprint(),
        denominator=dict(denominator),
        members=tuple(results),
        recorded_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=time.monotonic() - started,
        workers=max(1, workers),
        partial=partial,
        max_member_bytes=max_member_bytes,
    )


# --------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------


def diff_censuses(previous: Census | None, current: Census) -> CensusDiff:
    """Compare two censuses member by member."""
    if previous is None:
        return CensusDiff(
            findings=(),
            origins=(),
            parser_fingerprint_before="",
            parser_fingerprint_after=current.parser_fingerprint,
            compared=False,
            skipped_reason="no previous census",
        )
    if previous.partial or current.partial:
        return CensusDiff(
            findings=(),
            origins=(),
            parser_fingerprint_before=previous.parser_fingerprint,
            parser_fingerprint_after=current.parser_fingerprint,
            compared=False,
            skipped_reason="a bounded (--limit) census is not comparable evidence",
        )

    before = previous.by_key()
    after = current.by_key()
    parser_moved = previous.parser_fingerprint != current.parser_fingerprint
    findings: list[CensusFinding] = []
    rows: dict[str, dict[str, int]] = {}

    def row(origin: str) -> dict[str, int]:
        return rows.setdefault(
            origin,
            {
                "members": 0,
                "unchanged": 0,
                "changed_digest": 0,
                "new_failures": 0,
                "recovered_failures": 0,
                "added": 0,
                "removed": 0,
                "origin_changed": 0,
                "input_changed": 0,
            },
        )

    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if new is None:
            assert old is not None
            counters = row(old.origin)
            counters["removed"] += 1
            findings.append(
                CensusFinding(
                    kind="member_absent",
                    severity=SEVERITY_ERROR,
                    origin=old.origin,
                    member=key,
                    detail="the previous census covered this member and the current denominator does not name it",
                )
            )
            continue
        counters = row(new.origin)
        counters["members"] += 1
        if old is None:
            counters["added"] += 1
            findings.append(
                CensusFinding(
                    kind="member_added",
                    severity=SEVERITY_NOTICE,
                    origin=new.origin,
                    member=key,
                    detail=f"new denominator member, outcome {new.outcome}",
                )
            )
            continue

        if old.origin != new.origin:
            counters["origin_changed"] += 1
            findings.append(
                CensusFinding(
                    kind="origin_changed",
                    severity=SEVERITY_ERROR,
                    origin=new.origin,
                    member=key,
                    detail=f"classified {old.origin} before, {new.origin} now",
                )
            )
        if not old.failed and new.failed:
            counters["new_failures"] += 1
            findings.append(
                CensusFinding(
                    kind="parse_failure_new",
                    severity=SEVERITY_ERROR,
                    origin=new.origin,
                    member=key,
                    detail=f"{old.outcome} -> {new.outcome}: {new.reason or 'no reason recorded'}",
                )
            )
        elif old.failed and not new.failed:
            counters["recovered_failures"] += 1
            findings.append(
                CensusFinding(
                    kind="parse_failure_recovered",
                    severity=SEVERITY_NOTICE,
                    origin=new.origin,
                    member=key,
                    detail=f"{old.outcome} -> {new.outcome}",
                )
            )

        input_changed = old.sha256 != new.sha256 or old.byte_count != new.byte_count
        if input_changed:
            counters["input_changed"] += 1
            findings.append(
                CensusFinding(
                    kind="input_changed",
                    severity=SEVERITY_NOTICE,
                    origin=new.origin,
                    member=key,
                    detail="source bytes changed; parser differences for this member are not attributable",
                )
            )
        if old.digest == new.digest:
            counters["unchanged"] += 1
            continue
        counters["changed_digest"] += 1
        if input_changed:
            continue
        if parser_moved:
            findings.append(
                CensusFinding(
                    kind="digest_changed_parser_moved",
                    severity=SEVERITY_NOTICE,
                    origin=new.origin,
                    member=key,
                    detail=f"parse result changed under a moved parser fingerprint ({old.outcome} -> {new.outcome})",
                )
            )
            continue
        findings.append(
            CensusFinding(
                kind="digest_changed_nondeterministic",
                severity=SEVERITY_ERROR,
                origin=new.origin,
                member=key,
                detail="identical input bytes and an identical parser fingerprint produced a different parse",
            )
        )

    origins = tuple(OriginRow(origin=origin, **counters) for origin, counters in sorted(rows.items()))
    return CensusDiff(
        findings=tuple(findings),
        origins=origins,
        parser_fingerprint_before=previous.parser_fingerprint,
        parser_fingerprint_after=current.parser_fingerprint,
        compared=True,
    )


# --------------------------------------------------------------------------
# Census storage
# --------------------------------------------------------------------------


def census_dir(explicit: Path | None = None) -> Path:
    """Where censuses are kept.

    Precedence: an explicit path, ``POLYLOGUE_PARSER_CENSUS_DIR``, then a
    sibling of the archive root. It is deliberately *not* inside the archive:
    the census is derived evidence about the source corpus and must never be
    written into operator archive state.
    """
    if explicit is not None:
        return explicit.expanduser()
    raw = os.environ.get(CENSUS_DIR_ENV_VAR, "").strip()
    if raw:
        return Path(raw).expanduser()
    from polylogue.paths import archive_root

    root = archive_root()
    return root.parent / f"{root.name}-census"


def latest_census(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("census-*.json"))
    return candidates[-1] if candidates else None


def load_census(path: Path) -> Census:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParserCensusError(f"census unreadable at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParserCensusError(f"census at {path} is not an object")
    return Census.from_payload(cast(Mapping[str, object], payload))


def write_census(census: Census, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(census.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


def _parse_sources(values: Iterable[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for value in values:
        subject, _, raw = value.partition("=")
        if not subject or not raw:
            raise ParserCensusError(f"--source expects SUBJECT=PATH, got {value!r}")
        sources.append((subject, Path(raw)))
    return sources


def _default_workers() -> int:
    """Four by default: each worker holds one whole member plus its parsed
    sessions, so worker count is a memory decision before it is a speed one."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse every member of a recorded source denominator and diff the result against the last census."
    )
    parser.add_argument("--frontier", type=Path, default=None, help="Frontier document (default: archive state).")
    parser.add_argument("--subject", action="append", default=[], help="Limit the denominator to these subjects.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="SUBJECT=PATH",
        help="Census this tree instead of the frontier (the seeded/demo-corpus route).",
    )
    parser.add_argument("--census-dir", type=Path, default=None, help="Where censuses are kept.")
    parser.add_argument("--baseline", type=Path, default=None, help="Compare against this census (default: latest).")
    parser.add_argument("--no-baseline", action="store_true", help="Record without comparing.")
    parser.add_argument("--out", type=Path, default=None, help="Write the census here instead of the census dir.")
    parser.add_argument("--no-write", action="store_true", help="Do not persist the census.")
    parser.add_argument("--limit", type=int, default=None, help="Census at most N members (a partial census).")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="Parse worker processes.")
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_BYTES,
        help="Record members larger than this as `oversized` instead of parsing them (0 disables the bound).",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        if args.source:
            members, denominator = source_denominator(_parse_sources(args.source))
        else:
            members, denominator = frontier_denominator(args.frontier, tuple(args.subject) or None)
    except ParserCensusError as exc:
        print(f"parser-census: {exc}", file=sys.stderr)
        return 1

    partial = False
    if args.limit is not None and args.limit < len(members):
        members = members[: args.limit]
        partial = True

    census = build_census(
        members,
        denominator,
        workers=args.workers,
        partial=partial,
        max_member_bytes=args.max_member_bytes,
    )

    directory = census_dir(args.census_dir)
    previous: Census | None = None
    if not args.no_baseline:
        baseline_path = args.baseline or latest_census(directory)
        if baseline_path is not None:
            try:
                previous = load_census(baseline_path)
            except ParserCensusError as exc:
                print(f"parser-census: {exc}", file=sys.stderr)
                return 1

    diff = diff_censuses(previous, census)

    written: Path | None = None
    if not args.no_write:
        if args.out is not None:
            written = write_census(census, args.out)
        elif partial:
            print("parser-census: bounded census not persisted; pass --out to keep it", file=sys.stderr)
        else:
            stamp = census.recorded_at.replace(":", "").replace("-", "")
            written = write_census(census, directory / f"census-{stamp}.json")

    if args.as_json:
        print(
            json.dumps(
                {
                    "census": census.to_payload(),
                    "diff": diff.to_payload(),
                    "written": None if written is None else str(written),
                },
                sort_keys=True,
            )
        )
        return 0 if diff.ok else 1

    totals = census.totals
    print(f"parser-census: {'OK' if diff.ok else 'RED'}")
    print(f"  denominator={denominator.get('kind')} members={totals['members']:,} workers={census.workers}")
    print(f"  duration={census.duration_seconds:.1f}s partial={str(census.partial).lower()}")
    print(f"  parser_fingerprint={census.parser_fingerprint}")
    for outcome in sorted(OUTCOMES):
        if totals.get(outcome):
            print(f"  {outcome}: {totals[outcome]:,}")
    print(f"  sessions={totals['sessions']:,} messages={totals['messages']:,} blocks={totals['blocks']:,}")
    if written is not None:
        print(f"  census written to {written}")
    if not diff.compared:
        print(f"  diff skipped: {diff.skipped_reason}")
        return 0
    print(f"  parser fingerprint {'moved' if diff.parser_moved else 'unchanged'} since the previous census")
    for row in diff.origins:
        print(
            f"  {row.origin}: members={row.members:,} unchanged={row.unchanged:,} changed={row.changed_digest:,} "
            f"new_failures={row.new_failures:,} recovered={row.recovered_failures:,} "
            f"added={row.added:,} removed={row.removed:,}"
        )
    for finding in diff.notices[:50]:
        print(f"  notice {finding.kind}: {finding.member} -- {finding.detail}")
    for finding in diff.errors[:200]:
        print(f"  {finding.kind}: {finding.member} -- {finding.detail}", file=sys.stderr)
    if len(diff.errors) > 200:
        print(f"  ... {len(diff.errors) - 200:,} further errors omitted", file=sys.stderr)
    return 0 if diff.ok else 1


__all__ = [
    "CENSUS_DIFF_SCHEMA",
    "CENSUS_DIR_ENV_VAR",
    "CENSUS_SCHEMA",
    "DEFAULT_MAX_MEMBER_BYTES",
    "DEPTH_EXPLAIN",
    "DEPTH_FULL",
    "FAILURE_OUTCOMES",
    "OUTCOMES",
    "Census",
    "CensusDiff",
    "CensusFinding",
    "DenominatorMember",
    "MemberCensus",
    "OriginRow",
    "ParserCensusError",
    "build_census",
    "census_dir",
    "census_member",
    "diff_censuses",
    "frontier_denominator",
    "latest_census",
    "load_census",
    "main",
    "source_denominator",
    "write_census",
]


if __name__ == "__main__":
    raise SystemExit(main())
