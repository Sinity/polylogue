Title: "One monotonic embedding freshness invariant: DerivationKey + a single stale predicate for all four selectors (wmsc)"

Result ZIP: `misc-02-embedding-freshness-r01.zip`

## Mission

Implement bead `polylogue-wmsc` (P1 — read its full record). Embedding
freshness currently has COMPETING authorities: only one of four real
selection callers compares `message_embeddings_meta.content_hash` with
current message content; backlog, manual embed, and preflight bypass that
check. A prior success-write race was fixed by checking model identity,
but `mark_session_embedding_error` still unconditionally clears
`needs_reindex` for a non-retryable old attempt and can clobber a newer
config-change mark. Consequence: silently stale vectors that "look
converged" — the exact class of quiet dishonesty the archive exists to
prevent.

The decided design (bead is authority):

1. **DerivationKey value type**: first consumer of a small storage-neutral
   `polylogue/storage/derivation_identity.py` — a typed value/protocol
   carrying subject reference/grain, exact source identity, complete
   computational recipe identity, and output contract. Attempt generation,
   producer/resource data, eligibility/privacy, and result hash stay
   SEPARATE. Explicit anti-goal: no universal derivation table, scheduler,
   or lifecycle (the Diet architecture decision `05-derived-freshness.md`
   in the snapshot's testdiet context ratifies this: one protocol,
   domain-specific ledgers). Note: bead `polylogue-1xc.12` (FTS identity
   ledger) will consume the same value shape — keep it genuinely
   storage-neutral, and coordinate field naming with that bead's design.
2. **For embeddings**: source identity = current embeddable content
   (canonicalization rules coordinate with `polylogue-303r.7` — read it);
   recipe identity = model/provider/dimensions/canonicalization version.
3. **One indexed stale predicate**: per-source convergence, bulk backlog,
   manual embed, and preflight ALL use the same predicate and reconcile on
   the same snapshot semantics (bead AC #2). Kill the three bypasses.
4. **Error-path monotonicity**: `mark_session_embedding_error` (and any
   sibling status writers — grep the embeddings storage module) must not
   clear a newer staleness mark for an older attempt: guard writes by
   generation/recipe comparison so freshness marks are monotonic.
5. Tests: each of the four callers against a stale-content fixture (all
   four must select it), the config-change-vs-old-error clobber race
   (deterministic interleaving, not sleeps), recipe-change staleness
   (model swap ⇒ everything stale), and a mutation-named test per bypass
   removed. embeddings.db is rebuildable — if the meta schema needs a
   column, edit canonical DDL + bump its version (rebuild route, no
   migration chain).

## Constraints

- The daemon embedding catch-up loop and `embedding_preflight`/status
  surfaces consume these predicates — update their projections and list
  every touched consumer.
- Do not touch vector storage format or the Voyage integration; this is
  freshness/selection semantics only.
- Live-archive verification is the integrator's: ship the read-only audit
  query that counts stale-by-new-predicate vs stale-by-old-predicates so
  the drift this bug caused becomes a number.

## Deliverable emphasis

HANDOFF.md: DerivationKey type spec (spelled fully — 1xc.12 will import
this design), the four-caller predicate unification diff map, error-path
monotonicity mechanism, schema/version changes if any, consumer updates,
and the live audit query with expected interpretation.
