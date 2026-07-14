"""Sinex-backed evidence mode: durable publication obligation and transport.

Polylogue-side implementation of polylogue-303r.2 ("Publish Sinex materials
and anchored observations with durable retry"). This package owns:

- the durable **publication obligation** ledger (``obligations.py``), a
  ``source.db`` table recording that a normalized-session material revision
  (see ``polylogue.material_protocol.v1``) must reach Sinex before backed-mode
  local projections may treat that revision as authoritative;
- the transport contract (``transport.py``) a real Sinex producer must
  satisfy, modeled on Sinex's documented ``DurableEmissionReceipt``
  (sinex-r6d.11) and ``RawEnvelopeSettlement`` (sinex-r6d.12) primitives, plus
  a contract-faithful in-process reference transport for local operation and
  tests;
- the orchestration service (``service.py``) that stages obligations and
  drains them against a transport;
- a best-effort adapter (``material_adapter.py``) from live archive
  ``Session`` reads to the ``SessionMaterial`` input the v1 encoder expects.

Scope note (binding, see ``docs/sinex-interop.md``): as of this package's
introduction, Sinex's own consumer primitives for this exact contract
(sinex-4j2.1.1, layered on sinex-r6d.11 which is itself still open upstream)
are not yet landed. This package is therefore a real, fully-tested Polylogue-
side producer against a documented contract, wired to a local reference
transport by default. Pointing ``polylogue.toml``'s ``[sinex]`` at a live
Sinex JetStream endpoint is cross-repo follow-up work, not something this
package can complete unilaterally. Standalone (``mode = "off"``) is and
remains the default and permanently supported mode (operator directive,
``docs/sinex-interop.md``).
"""

from __future__ import annotations

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    PublicationReceipt,
    ReceiptState,
)
from polylogue.sinex.obligations import get_obligation, list_obligations, record_obligation
from polylogue.sinex.service import DrainSummary, PublicationService
from polylogue.sinex.transport import LocalReferenceTransport, NullTransport, SinexTransport

__all__ = [
    "DrainSummary",
    "LocalReferenceTransport",
    "NullTransport",
    "ObligationStatus",
    "PublicationMode",
    "PublicationObligation",
    "PublicationReceipt",
    "PublicationService",
    "ReceiptState",
    "SinexTransport",
    "get_obligation",
    "list_obligations",
    "record_obligation",
]
