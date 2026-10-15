# Provider state and convergence steps

`state::{claude,codex}` adapts the existing provider interpretation cores to
store-owned, pinned SQLite snapshots. It preserves typed storage/admission
failures through fallible provider callbacks and admits cumulative lookup and
encoded-delta traffic. Provider interpretation and serialization finish before
the writer transaction starts.

Each committed step retains its pending effects under its next predecessor
step number, including steps that do not advance a raw byte offset. Evidence
hashing, Claude contribution layout, Codex result assembly and canonical
publication remain explicit obligations. A provider reaching EOF does not clear
them. Required Claude body references come from the same pinned state basis;
the adapter performs no filesystem lookup.

`converge::PendingInstall` performs one provider preparation transaction per
owner turn: stage one admitted member, seal the complete delta, apply an
admitted member batch, or publish the scalar checkpoint. It rejects completion
claims. `recover_turn` discards an interrupted unsealed delta before reparse,
or finishes a sealed delta without interpreting against partly applied state.
The durable barrier blocks new source snapshots while existing pinned readers
retain their coherent predecessor view.

The tests use actual retained synthetic inputs, prepared writes, reopen and
owned-process exit. Store peer operations between turns establish a component
seam; they do not qualify a daemon or public-client end-to-end outcome.

Encoded-byte allowances are logical admission limits. Whole cells, source
record decoding, decoded objects, effects buffers, allocator/RSS use, SQLite
page work and arbitrarily large output fanout are not fully resource-qualified.
Canonical Message/body selection, indexed contribution folding, version/hash
finalization, search statistics isolation, fairness and actual public-client
read/write/restart behavior remain unfinished. No live collection or product
query is registered by this crate.

The model's `version_evidence::Framer` streams the owning source-evidence recipe
in admitted byte chunks. Its digest is distinct from normalized payload content
integrity. Providers still own exact membership/order/canonicalization; lost
ephemeral hash progress replays from zero. This prerequisite does not qualify
append-local hash updates or canonical publication.


`source_turn` computes one window from the existing provider cores. Its linear
commit guard releases framing continuation only after the actual prepared-store
publication receipt matches. `owner` connects Codex imports to one Writer actor,
one capture worker and one compute worker, with bounded metadata requests.
Restart enumerates retained imports and resumes or discards durable prepared
state before interpreting again. Peer requests can run between installation
turns; provider EOF and waiting tails keep canonical finalization pending.
Current scheduling drains one input before the next. Neither this owner nor
its progress endpoint publishes canonical Messages or certifies fair per-input
rounds, arbitrary record capacity, physical IO/RSS or search completion.
