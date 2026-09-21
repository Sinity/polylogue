"""One implementation of public-ref resolution, reachable without a surface.

``resolve_ref`` answers "what does this public ref name, and does it resolve
against the live archive?".  It was a ``Polylogue`` method, which made it
unreachable from a daemon handler -- substrate must not import a surface -- so
every route that needed it either constructed a facade in the daemon's own
process (``daemon/http.py`` did exactly that) or would have had to reimplement
it.  A second implementation is not merely duplication here: ``annotations
import`` admits or rejects every durable ``user.db`` candidate row on this
answer, so two implementations that drift change *which rows are admitted*,
with no failure at the point of divergence (polylogue-j5u2b).

The resolution is therefore expressed as a *plan* rather than as a call:

* :func:`plan_ref_resolution` classifies the ref without touching the archive
  and either decides the whole payload (malformed, oversized, pending
  substrate) or names the archive read that decides it -- its operation name,
  transaction arguments and projection included, so both callers frame the
  same transaction rather than inventing two;
* the facade runs that read through ``run_archive_read``;
* a daemon handler runs it against the reader it already pinned.

Neither owns the resolution.  ``resolve_ref_against_archive`` is the one-line
composition for a caller that already holds an open archive.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.core.enums import AssertionStatus
from polylogue.core.refs import (
    EvidenceRef,
    ObjectRef,
    parse_delegation_ancestry_object_id,
    parse_delegation_edge_object_id,
    parse_delegation_subtree_object_id,
    parse_public_ref,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.surfaces.operator_commands import is_shell_quote_canonical, quote_ref_argument

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.surfaces.payloads import PublicRefResolutionPayload

__all__ = [
    "REF_RESOLUTION_OPERATION",
    "BLOCK_ANCHOR_RESOLUTION_OPERATION",
    "RefResolutionPlan",
    "plan_ref_resolution",
    "resolve_ref_against_archive",
]

#: Transaction identity of the object/evidence-ref read.  Named here so the
#: facade and a daemon handler frame the *same* transaction: a read recorded
#: under two operation names is two reads as far as every receipt is concerned.
REF_RESOLUTION_OPERATION = "archive.resolve_ref"

#: Transaction identity of the block-anchor read.  A block anchor resolves
#: against drift-tolerant anchor state rather than by id lookup, so it is a
#: distinct read with its own projection rather than a branch inside the other.
BLOCK_ANCHOR_RESOLUTION_OPERATION = "archive.resolve_block_anchor"


@dataclass(frozen=True, slots=True)
class RefResolutionPlan:
    """How one ref is resolved: decided already, or by one named archive read.

    ``payload`` is set exactly when no archive read is needed -- the ref is
    malformed, oversized, or names substrate that does not exist yet.  In every
    other case ``read`` is the work, and ``operation``/``arguments``/
    ``projection``/``stable_order`` are the transaction identity the caller
    must frame it with.
    """

    ref: str
    payload: PublicRefResolutionPayload | None = None
    read: Callable[[ArchiveStore], PublicRefResolutionPayload] | None = None
    operation: str = REF_RESOLUTION_OPERATION
    arguments: dict[str, object] = field(default_factory=dict)
    projection: str = "ref-resolution"
    stable_order: str = "canonical"

    def __post_init__(self) -> None:
        if (self.payload is None) == (self.read is None):
            raise ValueError("a ref resolution plan is either already decided or one archive read")


def resolve_ref_against_archive(
    archive: ArchiveStore,
    ref: str,
    *,
    archive_root: Path | None = None,
) -> PublicRefResolutionPayload:
    """Resolve ``ref`` against an archive the caller already opened.

    This is the daemon-side entry point.  It runs the *same* plan the facade
    runs, so a handler cannot answer a ref differently from the Python API --
    which is what makes the durable annotation-admission decision single-valued
    (polylogue-j5u2b).
    """

    plan = plan_ref_resolution(ref, archive_root=archive_root or Path(archive.archive_root))
    if plan.payload is not None:
        return plan.payload
    assert plan.read is not None
    return plan.read(archive)


def plan_ref_resolution(ref: str, *, archive_root: Path) -> RefResolutionPlan:
    """Classify ``ref`` and name the archive read that resolves it, if any.

    Classification is deliberately archive-free: a malformed or oversized ref
    is refused before it can reach parsing or SQLite, and the refusal is the
    same one on every surface because there is only one classifier.
    """

    from polylogue.storage.block_anchor import InvalidBlockAnchorError, parse_block_anchor, resolve_block_anchor
    from polylogue.surfaces.payloads import PublicRefResolutionPayload

    root = archive_root

    invalid_unicode_ref = _invalid_unicode_ref_payload(ref)
    if invalid_unicode_ref is not None:
        return RefResolutionPlan(ref=ref, payload=cast("PublicRefResolutionPayload", invalid_unicode_ref))
    bounded_batch_ref = _oversized_annotation_batch_ref_payload(ref)
    if bounded_batch_ref is not None:
        # ``parse_public_ref`` deliberately falls back to EvidenceRef, whose
        # one-segment form accepts arbitrary colon-bearing session ids. Guard
        # batch-like malformed inputs before that fallback can misclassify
        # and reflect a multi-megabyte value through the session miss path.
        try:
            batch_candidate = ObjectRef.parse(ref)
        except ValueError:
            return RefResolutionPlan(ref=ref, payload=cast("PublicRefResolutionPayload", bounded_batch_ref))
        if batch_candidate.kind != "annotation-batch":
            return RefResolutionPlan(ref=ref, payload=cast("PublicRefResolutionPayload", bounded_batch_ref))
    try:
        block_anchor = parse_block_anchor(ref)
    except InvalidBlockAnchorError:
        block_anchor = None
    if block_anchor is not None:

        def read_anchor(archive: ArchiveStore) -> PublicRefResolutionPayload:
            resolution = resolve_block_anchor(archive._conn, block_anchor)
            resolved = resolution.state in {"ok", "drifted_position", "drifted_message"}
            object_refs = (
                (f"message:{resolution.resolved_message_id}",) if resolution.resolved_message_id is not None else ()
            )
            return PublicRefResolutionPayload(
                ref=ref,
                kind="block",
                resolved=resolved,
                payload_kind="block-anchor",
                payload={
                    "state": resolution.state,
                    "anchor": resolution.anchor.to_text(),
                    "resolved_message_id": resolution.resolved_message_id,
                    "resolved_position": resolution.resolved_position,
                    "candidates": [
                        {"message_id": message_id, "position": position}
                        for message_id, position in resolution.candidates
                    ],
                    "detail": resolution.detail,
                },
                object_refs=object_refs,
                caveats=() if resolved else (resolution.detail or f"block anchor state: {resolution.state}",),
            )

        return RefResolutionPlan(
            ref=ref,
            read=read_anchor,
            operation=BLOCK_ANCHOR_RESOLUTION_OPERATION,
            arguments={"ref": ref},
            projection="block-anchor-resolution",
        )
    try:
        parsed = parse_public_ref(ref)
    except ValueError as exc:
        if bounded_batch_ref is not None:
            return RefResolutionPlan(ref=ref, payload=cast("PublicRefResolutionPayload", bounded_batch_ref))
        return RefResolutionPlan(
            ref=ref, payload=cast("PublicRefResolutionPayload", _unresolved_ref_payload(ref, str(exc)))
        )

    if isinstance(parsed, EvidenceRef):
        evidence_ref: EvidenceRef | None = parsed
        object_ref = parsed.to_object_ref()
    else:
        evidence_ref = None
        object_ref = parsed
    normalized_ref = parsed.format()

    def read(archive: ArchiveStore) -> PublicRefResolutionPayload:
        if object_ref.kind == "session":
            return _resolve_session_object_ref(archive, ref, normalized_ref, object_ref, evidence_ref)
        if object_ref.kind == "message":
            return _resolve_message_object_ref(archive, ref, normalized_ref, object_ref, evidence_ref)
        if object_ref.kind == "block":
            return _resolve_block_object_ref(archive, ref, normalized_ref, object_ref, evidence_ref)
        if object_ref.kind == "action":
            return _resolve_block_object_ref(
                archive,
                ref,
                normalized_ref,
                ObjectRef(kind="block", object_id=object_ref.object_id, qualifiers=object_ref.qualifiers),
                evidence_ref,
            )
        if object_ref.kind == "assertion":
            return _resolve_assertion_object_ref(root, ref, normalized_ref, object_ref)
        if object_ref.kind == "finding":
            return _resolve_finding_object_ref(root, ref, normalized_ref, object_ref)
        if object_ref.kind == "annotation-batch":
            return _resolve_annotation_batch_object_ref(archive, ref, normalized_ref, object_ref)
        if object_ref.kind == "delegation":
            return _resolve_delegation_object_ref(archive, ref, normalized_ref, object_ref)
        if object_ref.kind in {"run", "observed-event", "context-snapshot"}:
            return _resolve_runtime_object_ref(archive, ref, normalized_ref, object_ref)
        if object_ref.kind in _PENDING_OBJECT_REF_KINDS:
            return cast("PublicRefResolutionPayload", _pending_ref_payload(ref, normalized_ref, object_ref.kind))
        return cast(
            "PublicRefResolutionPayload",
            _unresolved_ref_payload(
                ref,
                f"unsupported public ref kind for resolution: {object_ref.kind}",
                normalized_ref=normalized_ref,
                kind=object_ref.kind,
            ),
        )

    return RefResolutionPlan(
        ref=ref,
        read=read,
        operation=REF_RESOLUTION_OPERATION,
        arguments={"ref": normalized_ref},
        projection="ref-resolution",
    )


def _shell_safe_command(command: str) -> str:
    """Return ``command`` only if it is already shell-quote canonical.

    Printed guidance is copy-pasteable: an operator runs it verbatim, so an
    unquoted archive-derived ref carries provider-controlled syntax into the
    operator's shell. Building a command with :func:`_find_ref_command`
    satisfies this by construction, so a new call site that forgets to quote
    raises here instead of printing an injection.

    A literal placeholder such as ``<QUERY>`` is not canonical either (``<`` is
    a redirect): quote it, exactly as a real ref would be.
    """

    if not is_shell_quote_canonical(command):
        raise ValueError(
            "resolution command is not shell-quote canonical (quote archive-derived "
            f"refs with polylogue.surfaces.operator_commands.quote_ref_argument): {command!r}"
        )
    return command


def _find_ref_command(ref: str, *, id_prefixed: bool = True, tail: str = "") -> str:
    """Build a copy-pasteable find/read command with the ref quoted as one token.

    The complete argument -- ``id:`` prefix included -- is quoted together, and
    control characters are escaped to a visible form first, so no part of an
    archive-derived ref reaches the operator's shell or terminal as syntax.
    """

    argument = quote_ref_argument(str(ref), id_prefixed=id_prefixed)
    suffix = f" {tail}" if tail else ""
    return _shell_safe_command("polylogue find " + argument + " then read" + suffix)


def _resolution_action(label: str, command: str | None = None, href: str | None = None) -> Any:
    from polylogue.surfaces.payloads import RefResolutionActionPayload

    if command is not None:
        command = _shell_safe_command(command)
    return RefResolutionActionPayload(label=label, command=command, href=href)


def _unresolved_ref_payload(
    ref: str, message: str, *, normalized_ref: str | None = None, kind: str | None = None
) -> Any:
    from polylogue.surfaces.payloads import PublicRefResolutionPayload

    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind=kind,
        resolved=False,
        caveats=(message,),
    )


def _oversized_annotation_batch_ref_payload(ref: str) -> Any | None:
    """Return a bounded unresolved descriptor for an oversized batch-like ref."""

    if not ref.startswith("annotation-batch"):
        return None
    from polylogue.surfaces.payloads import (
        AnnotationBatchRefDigestPayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    try:
        descriptor = AnnotationBatchRefDigestPayload.from_oversized_ref(ref)
    except (UnicodeEncodeError, ValueError):
        return None
    return PublicRefResolutionPayload(
        ref=f"annotation-batch:sha256-{descriptor.original_ref_sha256}",
        normalized_ref=None,
        kind="annotation-batch",
        resolved=False,
        payload_kind="annotation-batch-ref-digest",
        payload=model_json_document(descriptor),
        caveats=("oversized annotation batch reference omitted from the public response",),
    )


def _invalid_unicode_ref_payload(ref: str) -> Any | None:
    """Fail closed before an invalid Python string reaches parsing or SQLite."""

    from polylogue.surfaces.payloads import (
        InvalidUnicodeRefDigestPayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    try:
        descriptor = InvalidUnicodeRefDigestPayload.from_invalid_ref(ref)
    except ValueError:
        return None
    batch_like = ref.startswith("annotation-batch")
    stable_prefix = "annotation-batch:invalid-unicode" if batch_like else "invalid-unicode-ref"
    return PublicRefResolutionPayload(
        ref=f"{stable_prefix}:sha256-{descriptor.original_ref_surrogatepass_sha256}",
        normalized_ref=None,
        kind="annotation-batch" if batch_like else None,
        resolved=False,
        payload_kind="invalid-unicode-ref-digest",
        payload=model_json_document(descriptor),
        caveats=("invalid Unicode public reference omitted from the response",),
    )


#: ObjectRefKind values registered ahead of their backing storage tier
#: (polylogue-rxdo analysis-provenance epic). ``resolve_ref`` returns a typed
#: ``PendingObjectRefPayload`` (reason=substrate-pending) for these instead of
#: attempting a lookup against tables that do not exist yet.
_PENDING_OBJECT_REF_KINDS: frozenset[str] = frozenset({"query", "query-run", "result-set", "cohort", "analysis"})


def _pending_ref_payload(ref: str, normalized_ref: str, kind: str) -> Any:
    from polylogue.surfaces.payloads import PendingObjectRefPayload, PublicRefResolutionPayload, model_json_document

    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind=kind,
        resolved=False,
        payload_kind="pending",
        payload=model_json_document(PendingObjectRefPayload(kind=kind)),
        caveats=(f"{kind} substrate is not implemented yet (reason=substrate-pending)",),
    )


def _resolve_session_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
    evidence_ref: EvidenceRef | None,
) -> PublicRefResolutionPayload:
    from polylogue.surfaces.payloads import (
        PublicRefResolutionPayload,
        model_json_document,
        session_summary_envelope_from_summary,
    )

    try:
        session_id = archive.resolve_session_id(object_ref.object_id)
    except KeyError:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "session not found", normalized_ref=normalized_ref, kind="session"),
        )
    summaries = archive.list_summaries(session_id=session_id, limit=1)
    if not summaries:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "session not found", normalized_ref=normalized_ref, kind="session"),
        )
    summary_payload = session_summary_envelope_from_summary(archive_summary_to_domain(summaries[0]))
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="session",
        resolved=True,
        payload_kind="session-summary",
        payload=model_json_document(summary_payload),
        title=summary_payload.title,
        summary=f"{summary_payload.message_count} messages",
        object_refs=(f"session:{session_id}",),
        evidence_refs=() if evidence_ref is None else (evidence_ref.format(),),
        actions=(_resolution_action("read", _find_ref_command(session_id, tail="--format json")),),
    )


def _resolve_message_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
    evidence_ref: EvidenceRef | None,
) -> PublicRefResolutionPayload:
    from polylogue.surfaces.payloads import (
        PublicRefResolutionPayload,
        message_render_envelope_from_domain,
        model_json_document,
    )

    row = archive._conn.execute(
        """
        SELECT m.session_id, m.message_id
        FROM messages m
        WHERE m.message_id = ?
           OR ('message:' || m.session_id || ':' || m.message_id) = ?
        LIMIT 1
        """,
        (object_ref.object_id, normalized_ref),
    ).fetchone()
    if row is None:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "message not found", normalized_ref=normalized_ref, kind="message"),
        )
    session_id = str(row["session_id"])
    message_id = str(row["message_id"])
    summary = archive.read_summary(session_id)
    session = archive_envelope_to_session(
        archive.read_session(session_id),
        display_label=summary.display_label,
        display_label_source=summary.display_label_source,
    )
    message = next((item for item in session.messages if str(item.id) == message_id), None)
    if message is None:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "message not found", normalized_ref=normalized_ref, kind="message"),
        )
    payload = message_render_envelope_from_domain(message, session_id=session_id)
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="message",
        resolved=True,
        payload_kind="message",
        payload=model_json_document(payload),
        title=session.display_title,
        summary=(message.text or "")[:240],
        object_refs=(f"session:{session_id}", f"message:{message_id}"),
        evidence_refs=() if evidence_ref is None else (evidence_ref.format(),),
        actions=(_resolution_action("read session", _find_ref_command(session_id, tail="--view messages")),),
    )


def _resolve_block_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
    evidence_ref: EvidenceRef | None,
) -> PublicRefResolutionPayload:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveBlockQueryRow
    from polylogue.surfaces.payloads import BlockQueryRowPayload, PublicRefResolutionPayload, model_json_document

    block_index: int | None = None
    if object_ref.qualifiers:
        try:
            block_index = int(object_ref.qualifiers[0])
        except ValueError:
            return cast(
                PublicRefResolutionPayload,
                _unresolved_ref_payload(
                    ref,
                    "block ref qualifier must be an integer",
                    normalized_ref=normalized_ref,
                    kind="block",
                ),
            )
    row = archive._conn.execute(
        """
        SELECT b.block_id, b.message_id, b.session_id, s.origin, s.title,
               b.block_type, b.position, b.text, b.tool_name, b.semantic_type,
               b.tool_command, b.tool_path
        FROM blocks b
        JOIN sessions s ON s.session_id = b.session_id
        WHERE b.block_id = ?
           OR (b.message_id = ? AND (? IS NOT NULL AND b.position = ?))
           OR ('block:' || b.block_id) = ?
        LIMIT 1
        """,
        (object_ref.object_id, object_ref.object_id, block_index, block_index, normalized_ref),
    ).fetchone()
    if row is None:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "block not found", normalized_ref=normalized_ref, kind="block"),
        )
    payload = BlockQueryRowPayload.from_row(
        ArchiveBlockQueryRow(
            block_id=str(row["block_id"]),
            message_id=str(row["message_id"]),
            session_id=str(row["session_id"]),
            origin=str(row["origin"]),
            title=str(row["title"]) if row["title"] is not None else None,
            block_type=str(row["block_type"]),
            position=int(row["position"]),
            text=str(row["text"]) if row["text"] is not None else None,
            tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
            semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
            tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
            tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
        )
    )
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="block",
        resolved=True,
        payload_kind="block",
        payload=model_json_document(payload),
        title=payload.title,
        summary=(payload.text or payload.tool_command or payload.tool_name or "")[:240],
        object_refs=(f"session:{payload.session_id}", f"message:{payload.message_id}", f"block:{payload.block_id}"),
        evidence_refs=() if evidence_ref is None else (evidence_ref.format(),),
        actions=(_resolution_action("read message", _find_ref_command(payload.session_id, tail="--view messages")),),
    )


def _resolve_assertion_object_ref(
    archive_root: Path,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
) -> PublicRefResolutionPayload:
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope
    from polylogue.surfaces.payloads import AssertionClaimPayload, PublicRefResolutionPayload, model_json_document

    user_db = archive_root / "user.db"
    if not user_db.exists():
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "assertion not found", normalized_ref=normalized_ref, kind="assertion"),
        )
    with closing(open_readonly_connection(user_db)) as conn:
        conn.row_factory = sqlite3.Row
        envelope = read_assertion_envelope(conn, object_ref.object_id)
    if envelope is None:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "assertion not found", normalized_ref=normalized_ref, kind="assertion"),
        )
    payload = AssertionClaimPayload.from_envelope(envelope)
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="assertion",
        resolved=True,
        payload_kind="assertion-claim",
        payload=model_json_document(payload),
        title=payload.key or payload.kind,
        summary=payload.body_text,
        object_refs=(normalized_ref, payload.target_ref),
        evidence_refs=payload.evidence_refs,
        actions=(
            _resolution_action("list assertion target", _find_ref_command(payload.target_ref, id_prefixed=False)),
        ),
    )


def _resolve_finding_object_ref(
    archive_root: Path,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
) -> PublicRefResolutionPayload:
    from polylogue.storage.sqlite.finding_provenance import compute_finding_provenance
    from polylogue.surfaces.payloads import (
        FindingEvidenceRefState,
        FindingProvenancePayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    user_db = archive_root / "user.db"
    if not user_db.exists():
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "finding not found", normalized_ref=normalized_ref, kind="finding"),
        )
    with closing(open_readonly_connection(user_db)) as conn:
        conn.row_factory = sqlite3.Row
        provenance = compute_finding_provenance(conn, object_ref.object_id)
        controls_document = _finding_controls_document(conn, object_ref.object_id)
    if provenance is None:
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(ref, "finding not found", normalized_ref=normalized_ref, kind="finding"),
        )
    payload = FindingProvenancePayload(
        assertion_id=provenance.assertion_id,
        claim_key=provenance.claim_key,
        target_ref=provenance.target_ref,
        finding_kind=provenance.finding_kind,
        query_ref=provenance.query_ref,
        result_set_ref=provenance.result_set_ref,
        baseline_ref=provenance.baseline_ref,
        current_ref=provenance.current_ref,
        detector_ref=provenance.detector_ref,
        status=AssertionStatus.from_string(provenance.status),
        evidence=tuple(
            FindingEvidenceRefState(ref=item.ref, resolvable=item.resolvable, reason=item.reason)
            for item in provenance.evidence
        ),
        staleness_verdict=provenance.staleness_verdict,
        created_at_ms=provenance.created_at_ms,
        updated_at_ms=provenance.updated_at_ms,
    )
    caveats: tuple[str, ...] = ()
    if provenance.staleness_verdict != "current":
        caveats = (f"finding evidence staleness verdict: {provenance.staleness_verdict}",)
    object_refs = tuple(
        dict.fromkeys(
            ref_value
            for ref_value in (
                normalized_ref,
                provenance.target_ref,
                provenance.query_ref,
                provenance.result_set_ref,
            )
            if ref_value
        )
    )
    payload_document = model_json_document(payload)
    if controls_document is not None:
        payload_document["controls"] = controls_document["controls"]
        payload_document["rank_tier"] = controls_document["rank_tier"]
        payload_document["downgraded"] = controls_document["downgraded"]
        if controls_document["downgraded"]:
            caveats = (*caveats, "claim downgraded: at least one bound negative control failed")

    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="finding",
        resolved=True,
        payload_kind="finding-provenance",
        payload=payload_document,
        title=provenance.claim_key or provenance.finding_kind or "finding",
        summary=f"{provenance.finding_kind or 'finding'} ({provenance.status})",
        object_refs=object_refs,
        evidence_refs=tuple(item.ref for item in provenance.evidence),
        caveats=caveats,
        actions=(
            _resolution_action("list target evidence", _find_ref_command(provenance.target_ref, id_prefixed=False)),
        ),
    )


def _finding_controls_document(conn: Any, assertion_id: str) -> dict[str, Any] | None:
    """Render claim-vs-control together when the finding declared controls (rxdo.9.7).

    Reuses :class:`~polylogue.analysis.judgment.controls.ClaimWithControls`
    (mutation-tested, previously constructed only by its own unit tests)
    rather than re-deriving the downgrade/rank-tier logic here.
    """
    from polylogue.analysis.judgment.controls import ClaimWithControls, ControlOutcome, NegativeControl
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope

    envelope = read_assertion_envelope(conn, assertion_id)
    if envelope is None or not isinstance(envelope.value, dict):
        return None
    raw_controls = envelope.value.get("controls")
    if not isinstance(raw_controls, list) or not raw_controls:
        return None
    outcomes = tuple(
        ControlOutcome(
            control=NegativeControl(
                control_kind=cast(Any, raw["control_kind"]),
                query_ref=cast(str, raw["query_ref"]),
                result_ref=cast(str, raw["result_ref"]),
                matching_variables=tuple(cast("Sequence[str]", raw.get("matching_variables", ()))),
                expected_null=cast(str, raw["expected_null"]),
                confounds_checked=tuple(cast("Sequence[str]", raw.get("confounds_checked", ()))),
            ),
            observed_null_held=bool(raw["observed_null_held"]),
        )
        for raw in raw_controls
        if isinstance(raw, dict)
    )
    claim = ClaimWithControls(claim_ref=f"assertion:{assertion_id}", controls=outcomes)
    return {
        "controls": [
            {
                "control_kind": outcome.control.control_kind,
                "query_ref": outcome.control.query_ref,
                "result_ref": outcome.control.result_ref,
                "expected_null": outcome.control.expected_null,
                "observed_null_held": outcome.observed_null_held,
            }
            for outcome in claim.controls
        ],
        "rank_tier": claim.rank_tier,
        "downgraded": claim.downgraded,
    }


def _resolve_annotation_batch_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
) -> PublicRefResolutionPayload:
    from polylogue.surfaces.payloads import (
        AnnotationBatchPayload,
        PublicRefResolutionPayload,
        RefResolutionActionPayload,
        model_json_document,
    )

    batch = archive.get_annotation_batch(object_ref.object_id)
    if batch is None:
        bounded = _oversized_annotation_batch_ref_payload(ref)
        if bounded is not None:
            return cast(PublicRefResolutionPayload, bounded)
        return cast(
            PublicRefResolutionPayload,
            _unresolved_ref_payload(
                ref,
                "annotation batch not found",
                normalized_ref=normalized_ref,
                kind="annotation-batch",
            ),
        )
    payload = AnnotationBatchPayload.from_batch(batch)
    scalar_ref_pairs = (
        (batch.batch_ref, payload.batch_ref),
        (batch.target_ref, payload.target_ref),
        (batch.source_result_ref, payload.source_result_ref),
        (batch.actor_ref, payload.actor_ref),
        (batch.model_ref, payload.model_ref),
        (batch.prompt_ref, payload.prompt_ref),
    )
    object_refs = tuple(dict.fromkeys(value for value, preview in scalar_ref_pairs if not preview.truncated))
    public_ref = normalized_ref
    public_normalized_ref: str | None = normalized_ref
    if payload.batch_ref.truncated:
        public_ref = f"annotation-batch:sha256-{payload.batch_ref.text_sha256}"
        public_normalized_ref = None
    actions: tuple[RefResolutionActionPayload, ...] = ()
    if not payload.target_ref.truncated:
        actions = (
            _resolution_action("read annotation target", _find_ref_command(batch.target_ref, id_prefixed=False)),
        )
    return PublicRefResolutionPayload(
        ref=public_ref,
        normalized_ref=public_normalized_ref,
        kind="annotation-batch",
        resolved=True,
        payload_kind="annotation-batch",
        payload=model_json_document(payload),
        title="annotation batch provenance",
        summary=(
            f"{payload.valid_count}/{payload.total_count} valid; "
            f"{payload.invalid_count} invalid; {payload.abstained_count} abstained"
        ),
        object_refs=object_refs,
        caveats=payload.truncation_caveats(),
        actions=actions,
    )


def _resolve_delegation_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
) -> PublicRefResolutionPayload:
    """Resolve a ``delegation:`` ref against the polylogue-y964
    `delegation_facts` relation. Two id shapes share one lookup: action-observed refs carry an
    ``instruction_tool_use_block_id`` verbatim; non-action refs carry the
    deterministic ``edge:<parent>::<child>`` relation identity (no
    parent-side dispatch action exists to key off for edge_only,
    quarantined, or authority-contradicted attempts). Missing, unresolved,
    edge_only, quarantined, and authority-contradicted states are returned
    as typed payloads, never silently guessed."""

    from polylogue.surfaces.payloads import (
        DELEGATION_STATE_CAVEATS,
        DelegationCardPayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    ancestry_session_id = parse_delegation_ancestry_object_id(object_ref.object_id)
    if ancestry_session_id is not None:
        return _resolve_delegation_ancestry_object_ref(archive, ref, normalized_ref, ancestry_session_id)
    subtree_session_id = parse_delegation_subtree_object_id(object_ref.object_id)
    if subtree_session_id is not None:
        return _resolve_delegation_subtree_object_ref(archive, ref, normalized_ref, subtree_session_id)

    edge_identity = parse_delegation_edge_object_id(object_ref.object_id)
    if edge_identity is not None:
        parent_session_id, child_session_id = edge_identity
        card = archive.get_delegation_card(parent_session_id=parent_session_id, child_session_id=child_session_id)
    else:
        card = archive.get_delegation_card(instruction_tool_use_block_id=object_ref.object_id)

    if card is None:
        return PublicRefResolutionPayload(
            ref=ref,
            normalized_ref=normalized_ref,
            kind="delegation",
            resolved=False,
            payload_kind="missing",
            caveats=("delegation attempt not found for this identity",),
        )

    payload = DelegationCardPayload.from_card(card)
    attempt = payload.attempt
    object_refs = [f"session:{attempt.parent_session_id}"]
    if attempt.child_session_id is not None:
        object_refs.append(f"session:{attempt.child_session_id}")
    if payload.run_ref is not None:
        object_refs.append(payload.run_ref)
    caveats: tuple[str, ...] = ()
    state_caveat = DELEGATION_STATE_CAVEATS.get(attempt.mapping_state)
    if state_caveat is not None:
        caveats = (state_caveat,)
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="delegation",
        resolved=True,
        payload_kind="delegation-card",
        payload=model_json_document(payload),
        title=f"delegation attempt ({attempt.mapping_state})",
        summary=(payload.instruction or "")[:240] or None,
        object_refs=tuple(object_refs),
        evidence_refs=payload.evidence_refs,
        caveats=caveats,
        actions=(_resolution_action("read parent session", _find_ref_command(attempt.parent_session_id)),),
    )


def _resolve_delegation_ancestry_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    session_id: str,
) -> PublicRefResolutionPayload:
    """Resolve a ``delegation:ancestry:<session_id>`` ref: the full
    root-to-node dispatch chain for ``session_id`` (polylogue-qsb4),
    depth-annotated, in one recursive-CTE call
    (``ArchiveStore.get_delegation_ancestry``)."""

    from polylogue.surfaces.payloads import (
        DelegationAncestryPayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    rows = archive.get_delegation_ancestry(session_id)
    payload = DelegationAncestryPayload.from_rows(session_id, rows)
    object_refs = tuple(f"session:{node.session_id}" for node in payload.nodes)
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="delegation",
        resolved=True,
        payload_kind="delegation-ancestry",
        payload=model_json_document(payload),
        title=f"delegation ancestry for {session_id} ({payload.max_depth} level(s) up)",
        summary=f"{len(payload.nodes)} node(s), root-to-node",
        object_refs=object_refs,
        evidence_refs=(),
        actions=(_resolution_action("read queried session", _find_ref_command(session_id)),),
    )


def _resolve_delegation_subtree_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    session_id: str,
) -> PublicRefResolutionPayload:
    """Resolve a ``delegation:subtree:<session_id>`` ref: the full
    dispatch subtree rooted at ``session_id`` (polylogue-qsb4),
    depth-annotated, in one recursive-CTE call
    (``ArchiveStore.get_delegation_subtree``)."""

    from polylogue.surfaces.payloads import (
        DelegationSubtreePayload,
        PublicRefResolutionPayload,
        model_json_document,
    )

    rows = archive.get_delegation_subtree(session_id)
    payload = DelegationSubtreePayload.from_rows(session_id, rows)
    object_refs = tuple(f"session:{node.session_id}" for node in payload.nodes)
    return PublicRefResolutionPayload(
        ref=ref,
        normalized_ref=normalized_ref,
        kind="delegation",
        resolved=True,
        payload_kind="delegation-subtree",
        payload=model_json_document(payload),
        title=f"delegation subtree rooted at {session_id} ({payload.node_count} node(s))",
        summary=f"{payload.max_depth} level(s) deep",
        object_refs=object_refs,
        evidence_refs=(),
        actions=(_resolution_action("read root session", _find_ref_command(session_id)),),
    )


def _resolve_runtime_object_ref(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
) -> PublicRefResolutionPayload:
    from polylogue.surfaces.payloads import PublicRefResolutionPayload

    summary_offset = 0
    while True:
        summaries = archive.list_summaries(limit=200, offset=summary_offset)
        if not summaries:
            break
        summary_offset += len(summaries)
        for summary in summaries:
            resolved = _resolve_runtime_object_ref_for_summary(archive, ref, normalized_ref, object_ref, summary)
            if resolved is not None:
                return resolved
        if len(summaries) < 200:
            break
    return cast(
        PublicRefResolutionPayload,
        _unresolved_ref_payload(
            ref, f"{object_ref.kind} not found", normalized_ref=normalized_ref, kind=object_ref.kind
        ),
    )


def _resolve_runtime_object_ref_for_summary(
    archive: Any,
    ref: str,
    normalized_ref: str,
    object_ref: ObjectRef,
    summary: Any,
) -> PublicRefResolutionPayload | None:
    from polylogue.analysis.transforms import compile_session_digest
    from polylogue.surfaces.payloads import (
        ContextSnapshotQueryRowPayload,
        ObservedEventQueryRowPayload,
        PublicRefResolutionPayload,
        RunQueryRowPayload,
        model_json_document,
    )

    session = archive_envelope_to_session(
        archive.read_session(str(summary.session_id)),
        display_label=summary.display_label,
        display_label_source=summary.display_label_source,
    )
    digest = compile_session_digest(session)
    if object_ref.kind == "run":
        for run in digest.run_projection.runs:
            if run.run_ref.format() != normalized_ref:
                continue
            run_payload = RunQueryRowPayload(
                run_ref=run.run_ref.format(),
                session_id=str(summary.session_id),
                origin=str(summary.origin),
                title=summary.display_label or summary.title,
                native_session_id=run.native_session_id,
                native_parent_session_id=run.native_parent_session_id,
                parent_run_ref=run.parent_run_ref.format() if run.parent_run_ref else None,
                agent_ref=run.agent_ref.format() if run.agent_ref else None,
                lineage_refs=tuple(lineage.format() for lineage in run.lineage_refs),
                provider_origin=run.provider_origin,
                harness=run.harness,
                role=run.role,
                cwd=run.cwd,
                git_branch=run.git_branch,
                status=run.status,
                confidence=run.confidence,
                transcript_ref=run.transcript_ref.format() if run.transcript_ref else None,
                evidence_refs=tuple(evidence.format() for evidence in run.evidence_refs),
                context_snapshot_ref=run.context_snapshot_ref.format() if run.context_snapshot_ref else None,
            )
            return PublicRefResolutionPayload(
                ref=ref,
                normalized_ref=normalized_ref,
                kind="run",
                resolved=True,
                payload_kind="run",
                payload=model_json_document(run_payload),
                title=summary.display_label or summary.title,
                summary=f"{run_payload.role} {run_payload.status}",
                object_refs=(f"session:{summary.session_id}", normalized_ref),
                evidence_refs=run_payload.evidence_refs,
            )
    if object_ref.kind == "observed-event":
        for event in digest.run_projection.events:
            if event.event_ref.format() != normalized_ref:
                continue
            event_payload = ObservedEventQueryRowPayload(
                event_ref=event.event_ref.format(),
                session_id=str(summary.session_id),
                origin=str(summary.origin),
                title=summary.display_label or summary.title,
                kind=event.kind,
                summary=event.summary,
                delivery_state=event.delivery_state,
                subject_ref=event.subject_ref.format() if event.subject_ref else None,
                object_refs=tuple(item.format() for item in event.object_refs),
                evidence_refs=tuple(item.format() for item in event.evidence_refs),
            )
            return PublicRefResolutionPayload(
                ref=ref,
                normalized_ref=normalized_ref,
                kind="observed-event",
                resolved=True,
                payload_kind="observed-event",
                payload=model_json_document(event_payload),
                title=summary.display_label or summary.title,
                summary=event_payload.summary,
                object_refs=(f"session:{summary.session_id}", normalized_ref, *event_payload.object_refs),
                evidence_refs=event_payload.evidence_refs,
            )
    if object_ref.kind == "context-snapshot":
        for snapshot in digest.run_projection.context_snapshots:
            if snapshot.snapshot_ref.format() != normalized_ref:
                continue
            snapshot_payload = ContextSnapshotQueryRowPayload(
                snapshot_ref=snapshot.snapshot_ref.format(),
                session_id=str(summary.session_id),
                origin=str(summary.origin),
                title=summary.display_label or summary.title,
                run_ref=snapshot.run_ref.format(),
                boundary=snapshot.boundary,
                inheritance_mode=snapshot.inheritance_mode,
                segment_refs=tuple(item.format() for item in snapshot.segment_refs),
                evidence_refs=tuple(item.format() for item in snapshot.evidence_refs),
                metadata=dict(snapshot.metadata),
            )
            return PublicRefResolutionPayload(
                ref=ref,
                normalized_ref=normalized_ref,
                kind="context-snapshot",
                resolved=True,
                payload_kind="context-snapshot",
                payload=model_json_document(snapshot_payload),
                title=summary.display_label or summary.title,
                summary=f"{snapshot_payload.boundary} ({snapshot_payload.inheritance_mode})",
                object_refs=(
                    f"session:{summary.session_id}",
                    normalized_ref,
                    snapshot_payload.run_ref,
                    *snapshot_payload.segment_refs,
                ),
                evidence_refs=snapshot_payload.evidence_refs,
            )
    return None
