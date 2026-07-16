"""The Sinex transport contract, plus a null and a local reference transport.

``SinexTransport`` is the interface a real Sinex producer (staging exact
material bytes, publishing anchored observations through JetStream, and
reconciling r6d.11 durable receipts / r6d.12 aggregate raw-envelope
settlement) must satisfy. No implementation in this module talks to a live
Sinex instance: as of this package landing, the counterpart Sinex-side
primitives this contract targets (sinex-4j2.1.1, layered on sinex-r6d.11
which is itself still open upstream) are not yet consumable from Python. See
the package docstring in ``polylogue/sinex/__init__.py`` for the full note.

:class:`LocalReferenceTransport` is a contract-faithful in-process double: it
enforces the same idempotency-by-request_id and receipt-state vocabulary a
real transport must honor, and it is intentionally injectable with fault
points (crash-before-receipt, rejection, partial settlement) so obligation
durability can be tested without a live Sinex. It is explicitly not, and does
not claim to be, live Sinex JetStream transport.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol, runtime_checkable

from polylogue.sinex.models import PublicationReceipt, ReceiptState


@runtime_checkable
class SinexTransport(Protocol):
    """What a Sinex producer must expose to the publication service.

    ``publish_revision`` must be idempotent by ``request_id``: calling it
    twice with the same ``request_id`` and the same bytes must not create a
    second durable Sinex-side record, and must return a receipt consistent
    with the first successful attempt (design: "same-revision retry is
    idempotent; changed revision preserves history").
    """

    async def publish_revision(
        self,
        *,
        request_id: str,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
    ) -> PublicationReceipt: ...


class SinexTransportUnavailableError(RuntimeError):
    """Configured mirror/primary mode has no injected transport."""


TransportFactory = Callable[[], SinexTransport]
_TRANSPORT_FACTORY_LOCK = RLock()
_CONFIGURED_TRANSPORT_FACTORY: TransportFactory | None = None


def register_configured_transport_factory(factory: TransportFactory) -> None:
    """Register the deployment-owned transport composition hook.

    Polylogue intentionally does not infer endpoint credentials from ad-hoc
    environment variables.  Deployment composition registers one factory
    before the daemon builds its default convergence stages.
    """
    if not callable(factory):
        raise TypeError("Sinex transport factory must be callable")
    global _CONFIGURED_TRANSPORT_FACTORY
    with _TRANSPORT_FACTORY_LOCK:
        _CONFIGURED_TRANSPORT_FACTORY = factory


def clear_configured_transport_factory() -> None:
    """Clear process-local transport composition (primarily test isolation)."""
    global _CONFIGURED_TRANSPORT_FACTORY
    with _TRANSPORT_FACTORY_LOCK:
        _CONFIGURED_TRANSPORT_FACTORY = None


def resolve_configured_transport() -> SinexTransport:
    """Construct the registered transport or fail backed-mode startup loudly."""
    with _TRANSPORT_FACTORY_LOCK:
        factory = _CONFIGURED_TRANSPORT_FACTORY
    if factory is None:
        raise SinexTransportUnavailableError(
            "mirror/primary mode requires deployment composition to register a Sinex transport factory"
        )
    transport = factory()
    if not isinstance(transport, SinexTransport):
        raise TypeError("configured Sinex transport does not satisfy SinexTransport")
    return transport


class TransportPayloadConflictError(RuntimeError):
    """A request id was reused with bytes different from its first attempt."""


class TransportUsedInOffModeError(RuntimeError):
    """Raised when a transport is invoked despite ``PublicationMode.OFF``.

    Off mode performs zero transport work by design (polylogue-303r design
    section) -- this exception makes an accidental call a loud failure
    instead of a silent, unobserved side effect.
    """


class NullTransport:
    """The off-mode transport: any call is a bug, not a graceful no-op.

    Off mode's "zero transport work" guarantee is enforced by never wiring a
    transport at all when mode is off; this class exists so a caller who
    accidentally wires it anyway fails loudly and immediately, in tests and
    in production, rather than degrading to a silent skip.
    """

    async def publish_revision(
        self,
        *,
        request_id: str,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
    ) -> PublicationReceipt:
        raise TransportUsedInOffModeError(
            f"NullTransport.publish_revision called for request_id={request_id!r}; "
            "off mode must never reach a transport"
        )


@dataclass
class _RecordedCall:
    request_id: str
    manifest_bytes: bytes
    segment_bytes: dict[str, bytes]


@dataclass
class LocalReferenceTransport:
    """In-process contract double: real idempotency semantics, no network.

    Fault injection: pass ``fault_fn`` to control the outcome per call. It
    receives ``(request_id, attempt_number)`` and returns a
    :class:`ReceiptState` to force (or ``None`` to use the default
    ``PERSISTED_CONFIRMED`` success path). This is how tests exercise crash-
    before-receipt, rejection, and durable-debt scenarios without a live
    Sinex instance.
    """

    fault_fn: Callable[[str, int], ReceiptState | None] | None = None
    _receipts_by_request_id: dict[str, PublicationReceipt] = field(default_factory=dict)
    _attempt_counts: dict[str, int] = field(default_factory=dict)
    _payload_digests: dict[str, str] = field(default_factory=dict)
    calls: list[_RecordedCall] = field(default_factory=list)

    async def publish_revision(
        self,
        *,
        request_id: str,
        manifest_bytes: bytes,
        segment_bytes: Mapping[str, bytes],
    ) -> PublicationReceipt:
        digest = hashlib.sha256(manifest_bytes)
        for name, payload in sorted(segment_bytes.items()):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload)
        payload_digest = digest.hexdigest()
        existing_digest = self._payload_digests.get(request_id)
        if existing_digest is not None and existing_digest != payload_digest:
            raise TransportPayloadConflictError(f"request_id={request_id!r} was reused with different exact bytes")
        self._payload_digests.setdefault(request_id, payload_digest)

        # Idempotency: a request_id that already reached a durable outcome
        # returns that SAME receipt rather than doing (or recording) another
        # publish. This is the property real Sinex transport must have too.
        existing = self._receipts_by_request_id.get(request_id)
        if existing is not None and existing.state.unlocks_progress():
            return existing

        self._attempt_counts[request_id] = self._attempt_counts.get(request_id, 0) + 1
        attempt_number = self._attempt_counts[request_id]
        self.calls.append(_RecordedCall(request_id, manifest_bytes, dict(segment_bytes)))

        forced = self.fault_fn(request_id, attempt_number) if self.fault_fn is not None else None
        state = forced if forced is not None else ReceiptState.PERSISTED_CONFIRMED
        receipt = PublicationReceipt(request_id=request_id, state=state, detail=f"attempt={attempt_number}")
        if state.unlocks_progress():
            self._receipts_by_request_id[request_id] = receipt
        return receipt

    def call_count(self, request_id: str | None = None) -> int:
        """Total publish attempts, or attempts for one ``request_id``."""
        if request_id is None:
            return len(self.calls)
        return sum(1 for call in self.calls if call.request_id == request_id)


__all__ = [
    "LocalReferenceTransport",
    "NullTransport",
    "SinexTransport",
    "SinexTransportUnavailableError",
    "TransportFactory",
    "TransportPayloadConflictError",
    "TransportUsedInOffModeError",
    "clear_configured_transport_factory",
    "register_configured_transport_factory",
    "resolve_configured_transport",
]
