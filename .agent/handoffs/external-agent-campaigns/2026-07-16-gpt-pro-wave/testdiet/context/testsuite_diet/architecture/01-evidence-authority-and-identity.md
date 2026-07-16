---
created: 2026-07-16
purpose: Decide raw evidence authority, reconciliation, and identity contracts for L01-L03
status: recommended-decision
project: polylogue
---

# Evidence authority and identity

## Decision

Adopt one `RawAuthorityReconciler` as the only component allowed to choose or
repair the authoritative raw revision for a logical acquisition. It consumes
immutable acquisition/revision evidence, emits a complete typed plan, applies
only proof-authorized transitions with compare-and-swap revalidation, and
records an immutable receipt and postflight.

Byte/content proof is authoritative. Timestamps, path order, acquisition order,
and apparent richness are diagnostic evidence only. When byte-semantic evidence
cannot establish one safe result, the state is blocked for reacquisition or
judgment; no “newest wins” fallback is legal.

## Identities that must not collapse

| Identity | Meaning | Required fields |
| --- | --- | --- |
| `archive_id` | One logical five-tier archive and blob namespace | durable random/stable id in the archive manifest; identical across tier pointers |
| `acquisition_id` | One observed acquisition attempt/source object | origin/source identity, source-native locator or stable artifact ref, attempt/generation |
| `raw_revision_id` | One immutable acquired byte revision | content digest plus canonical revision envelope identity |
| `logical_material_id` | The provider/source-native object revisions refer to | origin plus native identity under `OriginSpec`; never guessed from a path alone |
| `content_hash` | Semantic normalized session payload used for ingest idempotency | existing NFC-normalized inclusion contract; excludes user metadata |
| `blob_hash` | Exact stored raw bytes | SHA-256 over bytes; no semantic normalization |

The same bytes may occur in several acquisitions. Several raw revisions may
normalize to one semantic content hash. Those facts permit deduplication but do
not erase provenance or acquisition receipts.

## Authority state machine

Every reconciled frontier member ends in exactly one of:

- `proven_current` — the chosen complete revision is byte-proven and all known
  applicable appends are applied in an exact predecessor chain;
- `safe_rekey` — byte-identical/equivalent evidence can be rebound without
  changing semantic history;
- `duplicate_alias` — retained provenance points to already-authoritative bytes;
- `superseded` — a revision is provably an ancestor or dominated duplicate;
- `reacquire_required` — evidence claims material that is missing or corrupt;
- `conflicting_authority` — two incompatible candidates remain plausible;
- `unresolved_provenance` — identity or predecessor evidence is insufficient;
- `corrupt` — digest, length, envelope, or receipt proof fails.

Only the first four are automatic convergence states. The remaining states
must remain visible debt. Operator judgment may select an outcome only through
a durable assertion that names the exact conflicting revision identities and
the evidence available at judgment time.

## Selection rules

1. Validate envelopes and byte hashes before classification.
2. Partition by `logical_material_id`; never compare unrelated native objects.
3. A full revision is an ancestor of another only when byte/semantic proof
   establishes it, not because it is older or smaller.
4. An append applies only when its exact predecessor/baseline identity and
   offset/chain proof match the selected history.
5. A unique proven full plus an unambiguous append chain yields current history.
6. Equal acquisition generations or incompatible proven fulls are conflicts,
   not arbitrary tie-breaks.
7. Reconciliation is a fixed point: reordering equivalent evidence or replaying
   the same plan changes neither authority nor receipts.
8. Before apply, re-read the frontier and head generation. Any change invalidates
   the plan and returns a typed conflict; it does not partially improvise.

This preserves the useful parts of `RawRevisionEnvelope`,
`ApplicationDecision`, `FullSnapshotFoldAuthorization`, and the immutable
revision-application receipts already present in the source while eliminating
parallel repair authorities in `storage/repair.py`.

## Plan/apply protocol

The reconciler has four explicit phases:

1. **Observe** — capture archive id, frontier digest, head generation, every
   candidate envelope, missing byte obligation, and current debt.
2. **Plan** — classify every candidate and produce exact row/blob/head effects,
   blockers, resource bounds, rollback target, and expected postflight.
3. **Authorize/apply** — automatic only for proof-safe states; destructive live
   repair requires an operator-authorized digest of the exact plan. Apply uses
   generation predicates and existing publication receipts.
4. **Postflight** — independently re-read byte/reference/frontier state, record
   terminal or residual debt counts, and prove a second plan is empty or
   blocked for the same stated reason.

Read-only preflight and receipt generation are always safe. The explicit live
authorization gate retained by `polylogue-yla8` is the sole operator choice in
this design.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Newest acquisition wins | Simple and fast | Clock/path/order metadata cannot prove authority; silently loses divergent evidence |
| Richest/largest payload wins | Often repairs partial captures | “Richness” is provider- and parser-dependent and may combine incompatible histories |
| Keep every revision and pick at read time | Preserves bytes | Makes every reader an authority engine and permits surface disagreement |
| Manual judgment for every mismatch | Safest against automation mistakes | Unnecessarily blocks provable prefix, duplicate, and rekey cases; does not scale |
| Current replay planner plus independent repair actuators | Minimal code change | Parallel policy paths can disagree and cannot produce one complete fixed-point proof |

## Migration and implementation seams

- Keep all raw bytes and receipts during migration; first run the reconciler in
  report-only mode and compare it to existing repair outputs.
- Move existing deterministic classification into the reconciler rather than
  rewriting it wholesale.
- Make `storage/repair.py` repair entry points delegate to plans; remove direct
  chooser logic only after equivalence and historical incident proofs.
- Persist judgment as durable user evidence, not a mutable repair flag.
- Do not combine this work with provider-to-origin public-surface retirement.

## Required proof

- permutations and duplicate insertions of equivalent evidence converge to the
  same head and terminal classification;
- incompatible fulls remain blocked under timestamp/path/generation changes;
- append predecessor/offset mutations fail;
- apply interrupted at every durable boundary resumes to the same result;
- a second plan is empty or reproduces the same explicit blocker;
- the historical `lkrc`/`yla8` broken-head and cursor-ahead witnesses are
  retained, with a temporary mutation that restores an arbitrary-newest choice.

Primary evidence: `polylogue-lkrc`, `polylogue-yla8`, `polylogue-25vy`,
`polylogue-rgh2`; `polylogue/archive/revision_authority.py`,
`polylogue/archive/revision_replay.py`,
`polylogue/storage/sqlite/archive_tiers/revision_application.py`, and
`polylogue/storage/repair.py`.
