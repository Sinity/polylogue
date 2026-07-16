---
created: 2026-07-16
purpose: Decide write conflicts, leases, multi-file publication, and resumability for L07-L10
status: recommended-decision
project: polylogue
---

# Concurrent writes, publication, and resume

## Decision

Use the narrowest correctness primitive that matches the invariant:

- atomic SQL expressions for commutative counters and monotonic state;
- compare-and-swap with explicit `MutationConflict` for operator-visible
  replace/judgment state;
- `BEGIN IMMEDIATE` for invariants spanning several rows in one SQLite file;
- durable reservation and effect receipts for filesystem or cross-tier work;
- the daemon write coordinator for in-process scheduling only, never as the
  sole cross-process correctness mechanism.

Do not promise cross-file ACID. Operations spanning `source.db`, `index.db`,
`user.db`, `ops.db`, or blob paths are idempotent receipt-backed sagas with an
observable intermediate state and deterministic recovery.

## Mutation result contract

Every non-append user or control mutation accepts:

- logical target and archive identity;
- expected version/generation or explicit unconditional-create intent;
- idempotency key;
- actor/capability identity;
- requested new value and evidence/judgment refs.

It returns `applied`, `already_applied`, `conflict`, `blocked`, or `failed`.
`conflict` includes expected and current generation/digest without overwriting
either writer. A blind last-writer-wins update is permitted only for a field
whose product contract explicitly declares it replaceable and non-judgmental.

For `upsert_assertion`, read/check/write must therefore be one conditional SQL
statement or one immediate transaction. The current separated `SELECT` then
`INSERT` shape is not an allowed synchronization boundary.

## Reservation lifecycle

A reservation has stable `reservation_id`, owner/attempt id, resource identity,
created generation, last observed heartbeat, expected effect, and one terminal
state:

- `committed` — the durable reference/effect exists and matches the receipt;
- `released` — the attempt explicitly abandoned before effect;
- `superseded` — a newer proven attempt owns the logical work;
- `reconcile_required` — owner is absent or evidence is contradictory;
- `corrupt` — receipt/resource integrity failed.

Only the owner or a reconciler with writer exclusion and terminal proof may
release a reservation. Wall-clock age may make it eligible for inspection; age
alone never proves an active slow writer dead.

## Blob and multi-tier publication saga

The durable sequence is:

1. allocate a reservation/attempt in `source.db` before publishing a final blob;
2. write and hash a private temporary blob;
3. commit the source raw/blob reference and publication receipt under the same
   source-tier write boundary where possible;
4. atomically publish or verify the content-addressed final path;
5. commit any index reference with the same effect identity;
6. mark the receipt terminal only after every required durable reference is
   observed;
7. allow cleanup only when no durable reference exists, the attempt is proven
   terminal/abandoned, no live lease owns it, and writer exclusion closes the
   observe/delete race.

If a crash occurs between steps, restart continues from the receipt. It never
repeats an already-proven effect and never deletes bytes merely because the
index projection is late.

`user.db` mutations do not join this saga transactionally with rebuildable
index work. User truth commits first; index consequences become explicit
convergence debt keyed to the user mutation receipt.

## Checkpoints and exactly-once effects

A checkpoint means “all effects through this input identity are durably
receipted”, not “the loop reached ordinal N”. Advance it with a conditional
write tied to the effect receipt. On resume:

- re-read the receipt for the next input identity;
- skip only an effect whose identity and result integrity match;
- retry pending/unknown effects idempotently;
- never infer completion from a cursor ahead of receipts;
- expose duplicate, gap, and incomparable cursor debt explicitly.

This applies to ingestion, convergence debt, embedding windows, backfill queues,
and rebuild generations even when their domain-specific states differ.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Coarse global process lock | Easy reasoning inside one daemon | Fails across processes/crashes and destroys safe concurrency |
| Last writer wins | Simple UX | Reverts operator judgment and hides lost updates |
| Optimistic CAS for every write | Uniform | Needlessly burdens commutative counters and multi-row invariants |
| SQLite transaction across attached databases as universal ACID | Appears atomic | Filesystem blobs and independently managed tiers still escape; durability/restore semantics are misleading |
| TTL/age-expire every lease | Automatic cleanup | Can delete live slow work after pause or pressure; clock is not owner-death proof |
| At-least-once loop with downstream dedup | Simple workers | Dedup keys are frequently incomplete and cursor/effect disagreement remains invisible |

## Failpoints and proof

Production code should expose deterministic, test-only-injected barriers at
real transaction/publication boundaries, not sleeps. Required tests cover:

- two writers forced through the old `SELECT`/write gap;
- CAS conflict and atomic counter success under permutations;
- crash after every saga step, then repeated restart to fixed point;
- GC interleaving before/after reservation, reference, publication, and release;
- slow live writer older than the reconciliation threshold is retained;
- checkpoint ahead/behind/duplicate/incomparable states;
- restoration of the historical `41ow`, `qug2`, `y337`, `0puw`, and `qs0a`
  failure shapes.

Primary source seams: `storage/sqlite/archive_tiers/user_write.py`,
`source_write.py`, `ops_write.py`, `storage/blob_publication.py`, daemon write
coordination, and the existing publication/revision receipt types.
