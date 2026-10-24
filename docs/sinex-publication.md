# Publishing bytes to Sinex

The [Sinex caller](../crates/sinex/src/producer_bytes.rs) stages byte-bearing
publications under the shared protocol's private producer spool. It sends an
explicit class and a reservation for the producer and final frame before the
receiver can install bytes in public CAS. The receiver independently copies and
verifies the private roles under its durable ownership. Polylogue's current
archive publication uses `Open`; sealed byte publication refuses before reading
or staging plaintext. Receivers without the versioned reservation response refuse.

This caller uses same-host private placement. Native qualification runs the
publisher and receiver under the same UID. Cross-UID spool access and Nix
integration remain separate work (`sxnext-9zw.14.2.4.4`); remote byte upload
belongs to the shared protocol's Source drivers.

[Store](../crates/store/src/sinex_rows.rs) exports source blobs as anonymous,
hash-verified plain snapshots. Publication never writes public Sinex CAS directly.
Oversized JSON frames use a private envelope owned by the same final publication.
A large material owns both its body and its exact typed JSON envelope;
no extra material is created just to carry the envelope.

Before reservation, the shared spool freezes the exact typed request and its
UUIDv7 material ID. Retry loads this request before minting another ID or reading
the source. New reserve attempts have fresh names; the final frame and descriptor
stay fixed. A completed reservation returns the original final acknowledgment,
which permits retry after receiver role cleanup without opening the source again.
Original envelope serialization preserves event record hashes; only the material
descriptor commitment canonicalizes object order.
Only the checked final acknowledgment permits sender artifact cleanup. Refusal,
cancellation or cleanup failure leaves attributable retry state. Missing or corrupt
frozen bytes remain an explicit error; changed provider bytes cannot replace them.

The frozen artifact lives under `incoming/producer-requests`; private roles live
under `incoming/producers`. Both paths are scoped by the shared protocol's
producer/final-frame key. They are durable retry state, so preserve them while a
publication is incomplete. Each publish pass streams only this producer's frozen
artifacts and asks batched `have` which final frames completed. It cleans checked
successes without resending other pending frames. Public CAS availability remains
independently checked by the ordinary verified restore path.

[Caller controls](../crates/sinex/src/producer_bytes/tests.rs) exercise reservation,
cancellation, exact identity reuse, private envelopes and missing evidence through
real NATS. The [native restore proof](../crates/sinex/src/tests.rs) uses the paired
Sinex receiver and schema tools. Its historical machine control uses actual old8
Writer data with a consistent SQLite snapshot and the current reserved publisher;
it does not establish compatibility of old8's unreserved primary transport.
