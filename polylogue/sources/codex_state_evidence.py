"""Terminal receipts for retained Codex state exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from polylogue.core.enums import Provider
from polylogue.logging import emit
from polylogue.sources import codex_state_projection
from polylogue.sources.parsers import codex_state
from polylogue.storage.materials import PreparedMaterial, admit_material, link_material

if TYPE_CHECKING:
    from polylogue.sources.prepared_jsonl import PreparedJsonl


CODEX_STATE_CENSUS_DETAIL = "retained Codex state evidence applied"


def codex_material_coordinate(source_path: str, thread_id: str, kind: str, item_id: str) -> tuple[str, str]:
    # A Codex install is the scope of state-database observations.  Names in
    # different installs are not competing revisions: R3 requires each to
    # remain independently readable.  The thread is part of the logical
    # coordinate too; it prevents a provider reusing a goal id from joining
    # two unrelated sessions.
    source_scope = str(Path(source_path).parent)
    source_uri = (
        f"codex://state/{kind}/{quote(thread_id, safe='')}/{quote(item_id, safe='')}"
        f"?scope={quote(source_scope, safe='')}"
    )
    referrer_ref = f"codex-session:{thread_id}"
    return source_uri, referrer_ref


def _upsert_codex_material(
    archive: Any,
    *,
    raw_id: str,
    prepared: PreparedMaterial,
    observed_at_ms: int,
) -> None:
    """Retain one generated Codex record through the shared material route."""
    conn = archive.source_connection
    source_uri, referrer_ref = prepared.source_uri, prepared.referrer_ref
    previous = conn.execute(
        "SELECT m.material_id FROM material_evidence_links AS l "
        "JOIN material_observations AS m USING(material_id) "
        "WHERE l.evidence_ref = ? AND l.relation = 'refers_to' "
        "AND m.source_uri = ? AND m.acquisition_state != 'superseded' "
        "ORDER BY m.created_at_ms DESC, m.material_id DESC LIMIT 1",
        (referrer_ref, source_uri),
    ).fetchone()
    material = admit_material(
        conn,
        prepared=prepared,
        observed_at_ms=observed_at_ms,
        supersedes_material_id=str(previous[0]) if previous is not None else None,
        commit=False,
    )
    if previous is not None and str(previous[0]) != material.material_id:
        conn.execute(
            "UPDATE material_observations SET acquisition_state = 'superseded' WHERE material_id = ?",
            (str(previous[0]),),
        )
    link_material(
        conn,
        material.material_id,
        raw_id,
        relation="acquired_from",
        authority="provider",
        observed_at_ms=observed_at_ms,
        source_diagnostic=f"Codex state retained export {raw_id}",
        commit=False,
    )
    link_material(
        conn,
        material.material_id,
        referrer_ref,
        relation="refers_to",
        authority="provider",
        observed_at_ms=observed_at_ms,
        source_diagnostic="Codex provider-generated state associated with its thread",
        commit=False,
    )


@dataclass(frozen=True, slots=True)
class CodexStateMaterializationReceipt:
    """Typed completion receipt for one paged state materialization."""

    kind: str
    rows_available: int
    rows_materialized: int
    #: Compatibility counters. Complete projection leaves both at zero.
    rows_declined_row_cap: int
    rows_declined_byte_cap: int
    clipped_item_ids: tuple[str, ...]
    bytes_materialized: int
    row_cap: int
    text_char_cap: int
    aggregate_byte_cap: int
    rows_skipped_invalid: int = 0

    @property
    def rows_declined(self) -> int:
        return self.rows_declined_row_cap + self.rows_declined_byte_cap

    @property
    def bounded(self) -> bool:
        """Whether projection skipped any source content."""
        return bool(self.rows_declined or self.clipped_item_ids or self.rows_skipped_invalid)

    def as_detail(self) -> str:
        """One-line durable summary for the membership census detail."""
        if self.rows_skipped_invalid:
            return (
                f"partial {self.kind} materialization: "
                f"{self.rows_materialized}/{self.rows_available} rows, "
                f"{self.rows_skipped_invalid} invalid row(s) skipped"
            )
        if not self.bounded:
            return (
                f"complete {self.kind} materialization: "
                f"{self.rows_materialized}/{self.rows_available} rows, "
                f"{self.bytes_materialized} encoded bytes in bounded parts"
            )
        return (
            f"bounded {self.kind} materialization: "
            f"{self.rows_materialized}/{self.rows_available} rows materialized, "
            f"{self.rows_declined_row_cap} declined by row cap {self.row_cap}, "
            f"{self.rows_declined_byte_cap} declined by aggregate byte cap "
            f"{self.aggregate_byte_cap}, "
            f"{len(self.clipped_item_ids)} payload(s) clipped at "
            f"{self.text_char_cap} chars"
        )


def _encode_state_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def materialize_codex_state_content(
    archive: Any,
    raw_id: str,
    *,
    prepared_state: PreparedJsonl,
    source_path: str,
    state_kind: str,
    acquired_at_ms: int,
    row_limit: int = codex_state.CODEX_STATE_PAGE_ROWS,
    aggregate_byte_limit: int = codex_state.CODEX_STATE_MAX_AGGREGATE_BYTES,
) -> CodexStateMaterializationReceipt | None:
    """Project the complete retained export through bounded row and text pages."""
    if state_kind == "goals":
        kind = "goal"
    elif state_kind == "memories":
        kind = "memory"
    else:
        return None
    rows_available = 0
    materialized = 0
    total_bytes = 0
    window_bytes = 0
    if prepared_state.codex_state_kind != state_kind:
        raise ValueError("prepared state material kind changed")
    for _thread_id, _item_id, part_kind, byte_size, prepared in prepared_state.iter_codex_state_material():
        if part_kind == "invalid":
            rows_available += 1
            continue
        if part_kind == "record" and materialized and materialized % row_limit == 0:
            archive.commit()
            window_bytes = 0
        if window_bytes + byte_size > aggregate_byte_limit:
            archive.commit()
            window_bytes = 0
        if prepared is None:
            raise ValueError("prepared state material is missing its sealed carrier")
        _upsert_codex_material(
            archive,
            raw_id=raw_id,
            prepared=prepared,
            observed_at_ms=acquired_at_ms,
        )
        if part_kind == "record":
            materialized += 1
            rows_available += 1
        total_bytes += byte_size
        window_bytes += byte_size

    receipt = CodexStateMaterializationReceipt(
        kind=kind,
        rows_available=rows_available,
        rows_materialized=materialized,
        rows_declined_row_cap=0,
        rows_declined_byte_cap=0,
        clipped_item_ids=(),
        bytes_materialized=total_bytes,
        row_cap=int(row_limit),
        text_char_cap=prepared_state.codex_state_text_chars,
        aggregate_byte_cap=int(aggregate_byte_limit),
        rows_skipped_invalid=max(0, rows_available - materialized),
    )
    if receipt.bounded:
        emit(
            "sources.codex_state.materialization_truncated",
            outcome="degraded",
            reason="invalid_state_rows" if receipt.rows_skipped_invalid else "declared_cap_reached",
            raw_id=raw_id,
            kind=receipt.kind,
            rows_available=receipt.rows_available,
            rows_materialized=receipt.rows_materialized,
            rows_declined_row_cap=receipt.rows_declined_row_cap,
            rows_declined_byte_cap=receipt.rows_declined_byte_cap,
            rows_skipped_invalid=receipt.rows_skipped_invalid,
            payloads_clipped=len(receipt.clipped_item_ids),
            row_cap=receipt.row_cap,
            text_char_cap=receipt.text_char_cap,
            aggregate_byte_cap=receipt.aggregate_byte_cap,
        )
    return receipt


def record_codex_state_snapshot_terminal(
    archive: Any,
    raw_id: str,
    *,
    prepared_state: PreparedJsonl,
    state_kind: str,
    source_path: str,
    acquired_at_ms: int,
    censused_at_ms: int,
    blob_hash: str | None = None,
) -> None:
    """Finalize one admitted Codex state export as terminal non-session evidence.

    A state export has no byte frontier and never yields a session, so the
    cursor-authority gate can only account for it through a terminal
    source-tier receipt: the ``non_session`` membership census plus a
    finalized parse state. Live ingest and retained-raw replay both end here
    so a raw admitted by either route satisfies the same gate.

    A ``thread_state`` export also recomputes the index-tier thread-state
    projection, which is why both routes pass through one function.
    """
    from polylogue.storage.raw_authority import raw_authority_parser_fingerprint

    receipt: CodexStateMaterializationReceipt | None = None
    if state_kind in {"goals", "memories"}:
        receipt = materialize_codex_state_content(
            archive,
            raw_id,
            prepared_state=prepared_state,
            source_path=source_path,
            state_kind=state_kind,
            acquired_at_ms=acquired_at_ms,
        )
    if state_kind == codex_state_projection.THREAD_STATE_KIND:
        snapshot = prepared_state.codex_state_snapshot
        if snapshot is None or prepared_state.codex_state_kind != state_kind:
            raise ValueError("prepared thread state snapshot is absent")
        prepared_state.verify_files(full=False)
        codex_state_projection.apply_prepared_state_snapshot(
            archive,
            raw_id,
            snapshot=snapshot,
            blob_hash=blob_hash or archive.raw_revision_descriptor(raw_id)[1],
            observed_at_ms=acquired_at_ms,
            source_path=source_path,
        )
    archive.replace_raw_membership_census(
        raw_id,
        [],
        parser_fingerprint=raw_authority_parser_fingerprint(),
        censused_at_ms=censused_at_ms,
        detail=(
            f"{CODEX_STATE_CENSUS_DETAIL}; {receipt.as_detail()}" if receipt is not None else CODEX_STATE_CENSUS_DETAIL
        ),
        retire_full_revision_governance=True,
        revision_authority=None,
    )
    archive.mark_raw_parse_succeeded(raw_id, provider=Provider.CODEX)


__all__ = [
    "CODEX_STATE_CENSUS_DETAIL",
    "CodexStateMaterializationReceipt",
    "materialize_codex_state_content",
    "record_codex_state_snapshot_terminal",
]
