"""Durable Sinex publication outbox and daemon convergence integration.

In ``mirror`` and ``primary`` modes, production ingest encodes each accepted
normalized session revision and stages its exact manifest/segment bytes in the
same ``source.db`` transaction that marks the source raw accepted.  The daemon
then drains that source-tier outbox through an injected
:class:`~polylogue.sinex.transport.SinexTransport`, persists every receipt,
and resumes from durable payload bytes after restart.

``primary`` mode places the publication stage before derived local convergence
and blocks only the affected objects until their newest accepted revision has
an allowed durable receipt.  ``mirror`` retains exact lag and failure history
without blocking later projection stages.  ``off`` mode constructs no
publication service, encodes no material payload, writes no obligation, and
performs no transport work.

``SinexHTTPTransport`` publishes through the authenticated native revision RPC.
It stages separate acquired originals from retained Source CAS, not provider
URIs. ``LocalReferenceTransport`` remains an in-process test double.
"""

from __future__ import annotations

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationObligation,
    PublicationOriginal,
    PublicationPayload,
    PublicationReceipt,
    PublicationStatus,
    ReceiptState,
)
from polylogue.sinex.network_transport import SinexHTTPTransport
from polylogue.sinex.obligations import (
    PublicationPayloadConflictError,
    PublicationPayloadInvalidError,
    get_obligation,
    list_obligations,
    load_payload,
    record_obligation,
    stage_payload,
)
from polylogue.sinex.service import DrainSummary, PublicationService, publication_status_payload
from polylogue.sinex.transport import (
    LocalReferenceTransport,
    NullTransport,
    SinexTransport,
    SinexTransportUnavailableError,
    TransportPayloadConflictError,
    clear_configured_transport_factory,
    register_configured_transport_factory,
    resolve_configured_transport,
)

__all__ = [
    "DrainSummary",
    "LocalReferenceTransport",
    "NullTransport",
    "ObligationStatus",
    "PublicationMode",
    "PublicationObligation",
    "PublicationOriginal",
    "PublicationPayload",
    "SinexHTTPTransport",
    "PublicationPayloadConflictError",
    "PublicationPayloadInvalidError",
    "PublicationReceipt",
    "PublicationService",
    "publication_status_payload",
    "PublicationStatus",
    "ReceiptState",
    "SinexTransport",
    "SinexTransportUnavailableError",
    "TransportPayloadConflictError",
    "clear_configured_transport_factory",
    "get_obligation",
    "list_obligations",
    "load_payload",
    "record_obligation",
    "register_configured_transport_factory",
    "resolve_configured_transport",
    "stage_payload",
]
