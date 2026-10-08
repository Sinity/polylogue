"""Terminal receipts for retained Codex state exports."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.core.work_progress import advance_work_progress, reports_work_progress, stable_productive_identity
from polylogue.logging import emit
from polylogue.sources import codex_state_projection
from polylogue.sources.parsers import codex_state
from polylogue.storage.materials import (
    MaterialSourceProducer,
    PreparedMaterial,
    _admit_material,
    _link_material,
    _supersede_material,
)

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


__all__ = [
    "CODEX_STATE_CENSUS_DETAIL",
    "CodexStateMaterializationReceipt",
]


def _upsert_codex_material_source(
    producer: MaterialSourceProducer,
    *,
    raw_id: str,
    prepared: PreparedMaterial,
    observed_at_ms: int,
) -> None:
    """Retain one generated Codex record through the shared material route."""
    source_uri, referrer_ref = prepared.source_uri, prepared.referrer_ref
    with producer.material_previous_rows(referrer_ref, source_uri) as rows:
        previous = rows.fetchone()
    material = _admit_material(
        producer,
        prepared=prepared,
        observed_at_ms=observed_at_ms,
        supersedes_material_id=str(previous[0]) if previous is not None else None,
    )
    if previous is not None and str(previous[0]) != material.material_id:
        _supersede_material(producer, str(previous[0]))
    _link_material(
        producer,
        material.material_id,
        raw_id,
        relation="acquired_from",
        authority="provider",
        observed_at_ms=observed_at_ms,
        source_diagnostic=f"Codex state retained export {raw_id}",
    )
    _link_material(
        producer,
        material.material_id,
        referrer_ref,
        relation="refers_to",
        authority="provider",
        observed_at_ms=observed_at_ms,
        source_diagnostic="Codex provider-generated state associated with its thread",
    )


def _codex_state_productive_identity(
    producer: MaterialSourceProducer,
    raw_id: str,
    *,
    prepared_state: PreparedJsonl,
    source_path: str,
    state_kind: str,
    acquired_at_ms: int,
    settle_page: Callable[[], None],
    row_limit: int = codex_state.CODEX_STATE_PAGE_ROWS,
    aggregate_byte_limit: int = codex_state.CODEX_STATE_MAX_AGGREGATE_BYTES,
) -> str:
    del producer, settle_page
    return stable_productive_identity(
        (
            "codex-state-materialization",
            raw_id,
            prepared_state.blob_hash,
            source_path,
            state_kind,
            acquired_at_ms,
            row_limit,
            aggregate_byte_limit,
        )
    )


@reports_work_progress("codex-state-materialization", productive_identity=_codex_state_productive_identity)
def _materialize_codex_state_content(
    producer: MaterialSourceProducer,
    raw_id: str,
    *,
    prepared_state: PreparedJsonl,
    source_path: str,
    state_kind: str,
    acquired_at_ms: int,
    settle_page: Callable[[], None],
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
        check_compute_cancelled()
        advance_work_progress(messages=1, bytes=byte_size)
        if part_kind == "invalid":
            rows_available += 1
            continue
        if part_kind == "record" and materialized and materialized % row_limit == 0:
            settle_page()
            window_bytes = 0
        if window_bytes + byte_size > aggregate_byte_limit:
            settle_page()
            window_bytes = 0
        if prepared is None:
            raise ValueError("prepared state material is missing its sealed carrier")
        _upsert_codex_material_source(
            producer,
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


def _codex_state_terminal_detail(receipt: CodexStateMaterializationReceipt | None) -> str:
    return f"{CODEX_STATE_CENSUS_DETAIL}; {receipt.as_detail()}" if receipt is not None else CODEX_STATE_CENSUS_DETAIL


def prepare_codex_state_source_terminal(
    seal: PreparedIndexMutation,
    raw_id: str,
    *,
    prepared_state: PreparedJsonl,
    state_kind: str,
    source_path: str,
    acquired_at_ms: int,
    censused_at_ms: int,
    source_read: PreparedSessionSourceRead,
) -> CodexStateMaterializationReceipt | None:
    """Prepare only canonical Source material and terminal evidence.

    The parent has published the artifact's Blob pages and owns this same
    original read window/Source phase. The sealed thread-state snapshot stays
    on its existing Index projection owner; Source-only evidence does not
    acquire an Index capability through this terminal receipt.
    """
    from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        _PreparedSourceProducer,
        _raw_parse_success_state,
        prepare_raw_state_update,
        replace_raw_membership_census,
    )

    check_compute_cancelled()
    if state_kind not in codex_state.IN_SCOPE_KINDS or prepared_state.codex_state_kind != state_kind:
        raise ValueError("prepared Codex state kind differs from its terminal evidence")
    prepared_state.verify_files(full=False)
    if state_kind == codex_state_projection.THREAD_STATE_KIND:
        receipt_at_ms, receipt_order = source_read.raw_revision_observation_order(raw_id)
        prepared_state.prepare_thread_projection(
            seal,
            source_read=source_read,
            raw_id=raw_id,
            blob_hash=source_read.raw_revision_descriptor(raw_id)[1],
            observed_at_ms=receipt_at_ms,
            observation_order=receipt_order,
            source_path=source_path,
        )
    receipt = None
    if state_kind in {"goals", "memories"}:
        receipt = _materialize_codex_state_content(
            _PreparedSourceProducer(seal),
            raw_id,
            prepared_state=prepared_state,
            source_path=source_path,
            state_kind=state_kind,
            acquired_at_ms=acquired_at_ms,
            settle_page=check_compute_cancelled,
        )
    replace_raw_membership_census(
        seal,
        raw_id,
        [],
        parser_fingerprint=raw_authority_parser_fingerprint(),
        censused_at_ms=censused_at_ms,
        detail=_codex_state_terminal_detail(receipt),
        retire_full_revision_governance=True,
        revision_authority=None,
    )
    prepare_raw_state_update(seal, raw_id, state=_raw_parse_success_state(Provider.CODEX))
    check_compute_cancelled()
    return receipt


if TYPE_CHECKING:
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
