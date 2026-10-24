# Machine publication evidence

Adding a hostname must not change an acknowledged observation unit. Archive9
records each new alias's local commit cutoff; it publishes the original `id`,
`host` and `first_seen` fields only when that alias is eligible for the unit.
The cutoff is ordering metadata, not event time.

An upgrade preserves every current legacy alias at cutoff zero. It cannot
reconstruct a pre9 batch that had already changed after acknowledgment.
Publication therefore checks held observations for machines with several
baseline-zero aliases before publishing new units. It binds the actual held
frame's acknowledged EventId to its confirmed descriptor, folds authorized
replacements through the existing restore Catalog, verifies selected batch
hashes and compares machine dependencies with the current read snapshot.
Intentionally superseded before-images are not fetched.

A conflict names the held frame and verified batch. Missing acknowledged
descriptors or unavailable batch evidence remain errors. The comparison writes
neither archive rows nor acknowledgments. Preserve the local archive and any
unpublished facts together with the original acknowledgment, confirmed
descriptor and verified batch evidence. Establish the authoritative complete
view before ordinary restore; its local-fact and dependency-conflict refusals
still apply. There is no automatic reconciliation or separate repair command.

The first check folds the confirmed stream's metadata into the existing
disk-backed Catalog and reads only selected candidate observation batches,
not raw source blobs. The connection caches successful historical comparisons
bound to archive/producer identity, all relevant alias fields and cutoffs,
local restore/excision state, and candidate acknowledgments and selected batch
proofs. Unchanged polls and newly held unrelated frames do not refold that
history. Newly held candidate observations, relevant state changes and a new
connection require another fold. Changed candidate acknowledgments may repeat
only their bounded frame lookups to bind that new proof. Failed checks are not cached.

A cached historical comparison does not establish current batch-file
availability. Payload availability is checked when the proof is read again;
physical retirement retains its separate confirmation.

The implementation owners are `store::sinex_rows` for snapshot, selection and
verified row comparison, and `polylogue-sinex` for frame acknowledgments and
confirmed metadata. Native controls in `crates/sinex/src/tests/historical_machine.rs`
use the retained actual Archive8 frontend named by `POLYLOGUE_OLD8`; they run
inside the existing ignored excision publication control when that variable is
set. Its ordinary cache controls also run without the old frontend. The Store
admission/counter-floor controls remain in
`crates/store/src/sinex_rows/tests/machine_alias/history.rs`.
