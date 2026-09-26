"""Direct durable writer products behind the embedded facade."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.config import Config
from polylogue.config import active_archive_root as _active_archive_root
from polylogue.context.compiler import ContextImage
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.refs import ObjectRef, normalize_object_ref_text, parse_public_ref
from polylogue.operations.archive_mutation import require_archive_write_authority as _require_archive_write_authority
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import ArchiveContextDeliveryEnvelope
from polylogue.storage.sqlite.connection_profile import open_connection


def _archive_record_context_delivery(
    config: Config,
    *,
    image: ContextImage,
    boundary: str,
    recipient_ref: str,
    delivered_by_ref: str,
    run_ref: str | None,
    inheritance_mode: str,
) -> ArchiveContextDeliveryEnvelope:
    """Persist one exact delivery receipt for a compiled context image.

    This is the delivery boundary the storage layer (fs1.11/PR #2703) was
    built for but that no surface called: compilation alone is not evidence.
    Exact retries are idempotent and any drift in the immutable delivery
    identity is rejected -- both enforced by ``write_context_delivery``, not
    reimplemented here.
    """

    from polylogue.context.compiler import context_snapshot_record_from_image
    from polylogue.storage.sqlite.archive_tiers.context_delivery_write import write_context_delivery

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        raise ValueError("context-delivery user tier is not initialized")
    _require_archive_write_authority(config, "api.context_delivery")
    record = context_snapshot_record_from_image(
        image, boundary=boundary, run_ref=run_ref, inheritance_mode=inheritance_mode
    )
    try:
        conn = open_connection(user_db)
        conn.row_factory = sqlite3.Row
        try:
            envelope = write_context_delivery(
                conn,
                image=image,
                record=record,
                recipient_ref=recipient_ref,
                delivered_by_ref=delivered_by_ref,
            )
            conn.commit()
            return envelope
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to record context delivery: {exc}") from exc


def _archive_judge_assertion_candidate(
    config: Config,
    *,
    candidate_ref: str,
    decision: str,
    reason: str | None = None,
    actor_ref: str = "user:local",
    inject: bool = False,
    replacement_kind: str | None = None,
    replacement_body_text: str | None = None,
    replacement_value: object | None = None,
) -> Any:
    """Write an assertion-candidate judgment to ``user.db``."""

    from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidate

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        raise ValueError("assertion user tier is not initialized")
    _require_archive_write_authority(config, "api.judge_assertion_candidate")
    try:
        conn = open_connection(user_db)
        conn.row_factory = sqlite3.Row
        try:
            result = judge_assertion_candidate(
                conn,
                candidate_ref=candidate_ref,
                decision=decision,
                reason=reason,
                actor_ref=actor_ref,
                inject=inject,
                replacement_kind=replacement_kind,
                replacement_body_text=replacement_body_text,
                replacement_value=replacement_value,
            )
            conn.commit()
            return result
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to judge assertion candidate: {exc}") from exc


def _archive_capture_assertion_candidate(
    config: Config,
    *,
    body_text: str,
    kind: AssertionKind,
    refs: Sequence[str] = (),
    scope_refs: Sequence[str] = (),
    cwd: Path | None = None,
    author_ref: str = "user:local",
    author_kind: str = "user",
    idempotency_key: str | None = None,
    ttl_seconds: int | None = None,
) -> Any:
    """Write one terminal-captured assertion through the user-tier gate.

    ``ttl_seconds``, when given, stamps ``staleness={"expires_at_ms": ...}``
    on the written row (polylogue-37t.1): the admission read
    (:func:`~polylogue.storage.sqlite.archive_tiers.user_write.list_assertion_claims`)
    excludes expired claims from the preamble compiler and every other
    ``ASSERTION_CLAIM_KINDS`` consumer once ``expires_at_ms`` elapses, with no
    new assertion status introduced.
    """

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope, upsert_assertion

    normalized_body = body_text.strip()
    if not normalized_body:
        raise ValueError("note text cannot be empty")
    normalized_author_ref = normalize_object_ref_text(author_ref)
    normalized_author_kind = author_kind.strip().lower()
    if not normalized_author_kind:
        raise ValueError("author_kind cannot be empty")
    if ttl_seconds is not None and ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    normalized_idempotency_key = None if idempotency_key is None else idempotency_key.strip()
    if idempotency_key is not None and not normalized_idempotency_key:
        raise ValueError("idempotency_key cannot be empty")
    if normalized_idempotency_key is not None and len(normalized_idempotency_key) > 240:
        raise ValueError("idempotency_key exceeds 240 characters")

    if normalized_idempotency_key is None:
        assertion_id = f"assertion-terminal-note:{uuid.uuid4()}"
    else:
        identity = hashlib.sha256(
            f"{normalized_author_ref}\0{normalized_idempotency_key}".encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        assertion_id = f"assertion-terminal-note:{identity}"
    resolved_refs: list[str] = []
    _require_archive_write_authority(config, "api.capture_assertion_candidate")
    with ArchiveStore.open_existing(_active_archive_root(config), read_only=False) as archive:
        for ref in refs:
            if ref == "last":
                resolved_cwd = (cwd or Path.cwd()).resolve()
                repo_root = next(
                    (candidate for candidate in (resolved_cwd, *resolved_cwd.parents) if (candidate / ".git").exists()),
                    resolved_cwd,
                )
                summaries = archive.list_summaries(cwd_prefix=str(repo_root), limit=1)
                if not summaries:
                    raise ValueError("--ref last found no archived session for the current repository/cwd")
                session_ref = f"session:{summaries[0].session_id}"
                resolved_refs.append(session_ref)
                continue
            parsed = ObjectRef.parse(ref)
            if parsed.kind != "session":
                raise ValueError("--ref must be a session:<id> ref or 'last'")
            try:
                session_id = archive.resolve_session_id(parsed.object_id)
            except KeyError:
                raise ValueError(f"session ref not found: {parsed.object_id}") from None
            resolved_refs.append(f"session:{session_id}")

        normalized_scope_refs = [parse_public_ref(ref).format() for ref in scope_refs]
        target_ref = resolved_refs[0] if resolved_refs else f"assertion:{assertion_id}"
        user_db = archive.user_db_path

    fingerprint_document = {
        "author_kind": normalized_author_kind,
        "author_ref": normalized_author_ref,
        "body_text": normalized_body,
        "evidence_refs": list(dict.fromkeys((*resolved_refs, *normalized_scope_refs))),
        "kind": kind.value,
        "scope_refs": normalized_scope_refs,
        "target_ref": target_ref,
    }
    capture_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    try:
        _require_archive_write_authority(config, "api.capture_assertion_candidate")
        conn = open_connection(user_db)
        conn.row_factory = sqlite3.Row
        try:
            # The key lookup and first write share one reservation. Without
            # this, two changed captures racing on the same key could both
            # observe absence and the later writer would overwrite history.
            conn.execute("BEGIN IMMEDIATE")
            existing = read_assertion_envelope(conn, assertion_id)
            if existing is not None:
                existing_value = existing.value if isinstance(existing.value, dict) else {}
                existing_scope_refs = existing_value.get("scope_refs")
                existing_document = {
                    "author_kind": existing.author_kind,
                    "author_ref": existing.author_ref,
                    "body_text": existing.body_text,
                    "evidence_refs": existing.evidence_refs,
                    "kind": existing.kind.value,
                    "scope_refs": existing_scope_refs if isinstance(existing_scope_refs, list) else [],
                    "target_ref": existing.target_ref,
                }
                existing_fingerprint = hashlib.sha256(
                    json.dumps(
                        existing_document,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8", errors="surrogatepass")
                ).hexdigest()
                if existing_fingerprint == capture_fingerprint:
                    conn.commit()
                    return existing
                raise ValueError("idempotency_key conflicts with a different assertion candidate capture")
            capture_now_ms = int(datetime.now(UTC).timestamp() * 1000)
            staleness = None if ttl_seconds is None else {"expires_at_ms": capture_now_ms + ttl_seconds * 1000}
            envelope = upsert_assertion(
                conn,
                assertion_id=assertion_id,
                target_ref=target_ref,
                scope_ref=normalized_scope_refs[0] if normalized_scope_refs else None,
                kind=kind,
                key="terminal-note",
                value={
                    "capture_surface": "terminal",
                    "scope_refs": normalized_scope_refs,
                    "unanchored": not bool(resolved_refs),
                },
                body_text=normalized_body,
                author_ref=normalized_author_ref,
                author_kind=normalized_author_kind,
                evidence_refs=tuple(dict.fromkeys((*resolved_refs, *normalized_scope_refs))),
                status=AssertionStatus.CANDIDATE,
                staleness=staleness,
                context_policy={"inject": False, "promotion_required": True},
                now_ms=capture_now_ms,
            )
            conn.commit()
            return envelope
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to capture assertion candidate: {exc}") from exc


def _archive_judge_assertion_candidates(
    config: Config,
    *,
    items: Sequence[Any],
) -> Any:
    """Write an independently-recoverable bulk candidate judgment batch."""

    from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidates

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        raise ValueError("assertion user tier is not initialized")
    _require_archive_write_authority(config, "api.judge_assertion_candidates")
    try:
        conn = open_connection(user_db)
        conn.row_factory = sqlite3.Row
        try:
            result = judge_assertion_candidates(conn, items)
            conn.commit()
            return result
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to judge assertion candidates: {exc}") from exc


def _archive_record_comparative_judgment(
    config: Config,
    judgment: Any,
    *,
    author_kind: str,
) -> Any:
    """Write one comparative judgment (rxdo.9.11/9.6/9.7/9.12) as an assertion row.

    Mirrors :func:`_archive_judge_assertion_candidates`'s connection
    lifecycle. This is the first production caller of
    :func:`~polylogue.storage.sqlite.archive_tiers.user_write.upsert_comparative_judgment_assertion`
    -- the storage/read functions were fully built and tested but never
    invoked outside ``tests/unit/storage/`` before the ``judge compare`` /
    ``judge calibration`` CLI commands.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_comparative_judgment_assertion

    user_db = _active_archive_root(config) / "user.db"
    _require_archive_write_authority(config, "api.record_comparative_judgment")
    initialize_archive_database(user_db, ArchiveTier.USER)
    try:
        conn = open_connection(user_db)
        conn.row_factory = sqlite3.Row
        try:
            envelope = upsert_comparative_judgment_assertion(conn, judgment, author_kind=author_kind)
            conn.commit()
            return envelope
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to record comparative judgment: {exc}") from exc


def record_manual_continuation_product(config: Config, child_session_id: str, parent_session_id: str) -> None:
    child = str(child_session_id).strip()
    parent = str(parent_session_id).strip()
    if not child or not parent or ":" not in child or ":" not in parent:
        raise ValueError("manual continuation requires origin-prefixed child and parent session ids")
    parent_origin, parent_native = parent.split(":", 1)
    root = _active_archive_root(config)
    _require_archive_write_authority(config, "api.record_manual_continuation")
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    index = open_connection(root / "index.db")
    try:
        if index.execute("SELECT 1 FROM sessions WHERE session_id = ?", (child,)).fetchone() is None:
            raise ValueError("manual continuation child session does not exist")
        if index.execute("SELECT 1 FROM sessions WHERE session_id = ?", (parent,)).fetchone() is None:
            raise ValueError("manual continuation parent session does not exist")
        index.execute(
            # ``status`` is an exceptional marker (``TopologyEdgeStatus``:
            # repaired / quarantined / authority-contradicted), not the
            # ordinary resolved state -- resolvedness is carried by
            # ``resolved_dst_session_id IS NOT NULL``. Writing 'resolved'
            # here failed the column's generated CHECK, so this route
            # raised IntegrityError on every call (polylogue-pkst).
            """INSERT OR REPLACE INTO session_links
               (src_session_id, dst_origin, dst_native_id, link_type, inheritance,
                resolved_dst_session_id, method, confidence, evidence_json, observed_at_ms)
               VALUES (?, ?, ?, 'continuation', 'spawned-fresh', ?,
                       'manual-continuation', 1.0, '[]', ?)""",
            (child, parent_origin, parent_native, parent, now_ms),
        )
        index.commit()
    finally:
        index.close()

    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    user = open_connection(root / "user.db")
    try:
        upsert_assertion(
            user,
            assertion_id="handoff:" + hashlib.sha256(f"{child}\0{parent}".encode()).hexdigest()[:32],
            target_ref=f"session:{child}",
            kind=AssertionKind.HANDOFF,
            body_text=f"Continuation from session {parent}.",
            # ``author_ref`` is an ObjectRef: ``service`` is not a
            # declared kind, so this raised before the assertion landed
            # (polylogue-pkst). ``actor:`` is the kind the other
            # automated writers use (``actor:judgment-automation``).
            author_ref="actor:polylogue",
            author_kind="service",
            evidence_refs=[f"session:{parent}", f"session:{child}"],
            status=AssertionStatus.CANDIDATE,
            context_policy={"inject": False, "promotion_required": True},
            now_ms=now_ms,
        )
        user.commit()
    finally:
        user.close()


def record_context_ledger_product(config: Config, admission: Any) -> None:
    """Persist scheduler admission in the disposable operations tier."""
    from polylogue.context.scheduler import record_context_ledger
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db = _active_archive_root(config) / "ops.db"
    _require_archive_write_authority(config, "api.context_injection_ledger")
    if not ops_db.exists():
        initialize_archive_database(ops_db, ArchiveTier.OPS)
    ops_conn = open_connection(ops_db)
    try:
        record_context_ledger(ops_conn, admission, observed_at_ms=0)
    finally:
        ops_conn.close()


def _daemon_config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root, sources=[])


def _daemon_writer_result(request: Any, value: object, *, affected_count: int = 1) -> dict[str, object]:
    from polylogue.operations.facade_mutations import _to_wire

    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if affected_count else "no-effect",
        "affected_count": affected_count,
        "result": {"value": _to_wire(value)},
    }


def facade_record_work_event(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    from polylogue.operations.facade_mutations import record_work_event_product

    del audit, snapshot
    payload = request.payload
    result = record_work_event_product(
        _daemon_config(context.archive_root),
        str(payload["session_id"]),
        event_id=str(payload["event_id"]),
        event_type=str(payload["event_type"]),
        summary=str(payload["summary"]),
        payload=payload.get("payload"),
        timestamp=payload.get("timestamp"),
    )
    return _daemon_writer_result(request, result)


def facade_record_manual_continuation(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del audit, snapshot
    payload = request.payload
    record_manual_continuation_product(
        _daemon_config(context.archive_root), str(payload["child_session_id"]), str(payload["parent_session_id"])
    )
    return _daemon_writer_result(request, {})


def facade_record_context_delivery(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del audit, snapshot
    payload = request.payload
    envelope = _archive_record_context_delivery(
        _daemon_config(context.archive_root),
        image=ContextImage.model_validate(payload["image"]),
        boundary=str(payload["boundary"]),
        recipient_ref=str(payload["recipient_ref"]),
        delivered_by_ref=str(payload["delivered_by_ref"]),
        run_ref=payload.get("run_ref"),
        inheritance_mode=str(payload["inheritance_mode"]),
    )
    from dataclasses import fields

    metadata = {
        field.name: getattr(envelope, field.name) for field in fields(envelope) if field.name != "context_image"
    }
    return _daemon_writer_result(request, metadata, affected_count=int(envelope.outcome == "recorded"))


def facade_judge_assertion_candidate(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    from polylogue.surfaces.payloads import AssertionJudgmentResultPayload

    del audit, snapshot
    payload = request.payload
    envelope = _archive_judge_assertion_candidate(
        _daemon_config(context.archive_root),
        candidate_ref=str(payload["candidate_ref"]),
        decision=str(payload["decision"]),
        reason=payload.get("reason"),
        actor_ref=str(payload.get("actor_ref") or "user:local"),
        inject=bool(payload.get("inject", False)),
        replacement_kind=payload.get("replacement_kind"),
        replacement_body_text=payload.get("replacement_body_text"),
        replacement_value=payload.get("replacement_value"),
    )
    value = AssertionJudgmentResultPayload.from_envelope(envelope)
    return _daemon_writer_result(request, value, affected_count=int(value.outcome == "applied"))


def facade_record_comparative_judgment(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    from polylogue.operations.judgment_wire import comparative_judgment_from_wire_form

    del audit, snapshot
    payload = request.payload
    judgment = comparative_judgment_from_wire_form(payload["judgment"])
    envelope = _archive_record_comparative_judgment(
        _daemon_config(context.archive_root), judgment, author_kind=str(payload.get("author_kind") or "user")
    )
    return _daemon_writer_result(request, envelope)


def facade_context_ledger(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    from types import SimpleNamespace

    from polylogue.context.scheduler import ContextLedgerRow

    del audit, snapshot
    payload = request.payload
    rows = tuple(ContextLedgerRow(**row) for row in payload["ledger_rows"])
    record_context_ledger_product(
        _daemon_config(context.archive_root), SimpleNamespace(build_ref=payload["build_ref"], ledger=rows)
    )
    return _daemon_writer_result(request, {})
