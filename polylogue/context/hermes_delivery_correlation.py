"""Correlate Hermes ``context_injected`` lifecycle events with delivery receipts (fs1.11).

fs1.11 shipped a minimal read-only Hermes recall leg (PR #2792:
``get_context_delivery`` API/MCP lookup over the existing
``context_deliveries`` ledger) and deferred the scheduler/authorization flow
to ``polylogue-37t.11``. This module adds the piece the 2026-07-10 Nous
follow-up refinement asked for and #2792 did not yet have a producer for:
correlating each delivery with the Hermes snapshot revision and the durable
``context_injected`` event fs1.7's spool now carries.

Design choice, consistent with "extend the existing ContextSnapshotRecord/
delivery ledger rather than creating another manifest" (fs1.11 design): this
is a **read-side join**, not a new write path or schema. A Hermes producer
that actually injects a compiled context pack into a live turn emits a
``context_injected`` lifecycle event (via ``sources.hooks``, provider=
``hermes``) whose payload carries the ``snapshot_ref`` it received — an id,
not a duplicated transcript, so it satisfies the same payload-hygiene rule
every other lifecycle event does. This module resolves that id against the
existing ``context_deliveries`` receipt, exactly like ``actions`` is a VIEW
joining ``tool_use``/``tool_result`` blocks by id rather than a new stored
relation.

Token budget and rendered-token estimate are already persisted by
``context.compiler.context_snapshot_record_from_image`` (``metadata["max_tokens"]``/
``metadata["token_estimate"]``) — this module surfaces them via the
correlation result rather than duplicating that write.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import Origin
from polylogue.sources.parsers.hermes_lifecycle import CONTEXT_INJECTED
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import (
    ArchiveContextDeliveryEnvelope,
    read_context_delivery,
)
from polylogue.storage.sqlite.archive_tiers.source_write import list_hook_events


@dataclass(frozen=True, slots=True)
class HermesContextDeliveryCorrelation:
    """One ``context_injected`` event paired with its delivery receipt, if any."""

    hermes_session_native_id: str
    context_injected_event_id: str
    observed_at_ms: int
    snapshot_ref: str | None
    receipt: ArchiveContextDeliveryEnvelope | None
    token_budget: str | None
    rendered_token_estimate: str | None
    rendered_bytes_sha256: str | None
    available: bool
    caveats: tuple[str, ...]


def correlate_hermes_context_deliveries(
    source_conn: sqlite3.Connection,
    user_conn: sqlite3.Connection,
    *,
    hermes_session_native_id: str,
) -> tuple[HermesContextDeliveryCorrelation, ...]:
    """Correlate every drained ``context_injected`` event for a Hermes session.

    ``source_conn`` reads the durable spool (``raw_hook_events``, source.db);
    ``user_conn`` reads the delivery ledger (``context_deliveries``, user.db)
    -- the two durable tiers this correlation bridges. A ``context_injected``
    event with no resolvable receipt is not an error: it renders an explicit
    ``available=False`` row with a caveat (AC: "timeout or archive outage
    records an explicit unavailable state without blocking Hermes") rather
    than being silently skipped or raising.
    """
    events = list_hook_events(source_conn, origin=Origin.HERMES_SESSION, session_native_id=hermes_session_native_id)
    correlations: list[HermesContextDeliveryCorrelation] = []
    for event in events:
        if event.event_type != CONTEXT_INJECTED:
            continue
        # ``event.payload`` is the full spooled envelope (event_id/event_type/
        # session_id/timestamp/provider/payload/observed_at_ms, see
        # sources.hooks._persist_record); the producer's own event body is
        # nested one level deeper under its own "payload" key.
        inner_payload = event.payload.get("payload")
        snapshot_ref = inner_payload.get("snapshot_ref") if isinstance(inner_payload, dict) else None
        snapshot_ref_text = snapshot_ref if isinstance(snapshot_ref, str) and snapshot_ref else None
        receipt: ArchiveContextDeliveryEnvelope | None = None
        caveats: list[str] = []
        if snapshot_ref_text is None:
            caveats.append("context_injected event carries no snapshot_ref; cannot resolve a delivery receipt.")
        else:
            try:
                receipt = read_context_delivery(user_conn, snapshot_ref_text)
            except ValueError as exc:
                caveats.append(f"snapshot_ref failed validation: {exc}")
            if receipt is None and not caveats:
                caveats.append(
                    "no context_deliveries receipt found for this snapshot_ref "
                    "(archive outage, or the delivery write has not committed yet)."
                )
        correlations.append(
            HermesContextDeliveryCorrelation(
                hermes_session_native_id=hermes_session_native_id,
                context_injected_event_id=event.hook_event_id,
                observed_at_ms=event.observed_at_ms,
                snapshot_ref=snapshot_ref_text,
                receipt=receipt,
                token_budget=receipt.metadata.get("max_tokens") if receipt else None,
                rendered_token_estimate=receipt.metadata.get("token_estimate") if receipt else None,
                rendered_bytes_sha256=receipt.context_image_sha256 if receipt else None,
                available=receipt is not None,
                caveats=tuple(caveats),
            )
        )
    return tuple(correlations)


__all__ = [
    "HermesContextDeliveryCorrelation",
    "correlate_hermes_context_deliveries",
]
