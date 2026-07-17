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


---

## Context and authority

You are a long-running ChatGPT Pro engineering worker. A recent Polylogue
project-state archive will be attached. Retrieve and inspect it broadly; do not
assume attachment bytes consume your active prompt context. The attached
snapshot is the code authority. This prompt defines your mission. Repository
instructions and complete relevant Beads records define constraints and intent;
later Beads notes may supersede older descriptions. Current source wins when a
stale plan names paths or APIs that no longer exist.

Start by reporting the snapshot commit/branch/dirty-patch identity you found and
the source, tests, Beads, and history you inspected. Follow dependencies beyond
the obvious files when they affect the production route. Do not invent an API,
test helper, product contract, or parallel framework to make the task easy.

## Working contract

- Produce the largest internally coherent implementation draft that fits the
  mission. Prefer one real end-to-end behavior over disconnected scaffolding.
- Preserve Polylogue's substrate-first architecture and existing typed
  interfaces. Small production seams are allowed only when real production
  behavior needs observation or control.
- Write concrete production changes and real-route tests. A test must name the
  production dependency it exercises and the representative implementation
  mutation/removal that should make it fail.
- Do not delete existing tests or helpers. Identify proposed dominated
  deletions separately for independent local certification.
- Use your container and run meaningful self-contained checks when possible.
  Never claim access to the operator's live daemon, browser, archive, secrets,
  NixOS deployment, or current worktree. Mark those checks `unverified`.
- If the full scope is unsafe, complete the strongest coherent subset and make
  the remaining decisions and exact continuation steps explicit. Do not return
  placeholders, ellipses, pseudocode presented as code, or a generic plan in
  place of implementation.

## Deliverable

Create the exact `Result ZIP` named near the top of this prompt under
`/mnt/data/`. Do not include the supplied repository/project-state archive or
other copied inputs in the result. The finished ZIP must be attached to the
conversation through a working, user-clickable download link. Work left only
in an internal shell directory, temporary notebook, scattered sandbox files,
or prose is not delivered.

The ZIP must contain:

- `HANDOFF.md`: mission, snapshot identity, inspected evidence, mechanism,
  decisions, changed files, acceptance matrix, apply order, risks, and exact
  verification performed/remaining;
- `PATCH.diff`: one apply-ready unified diff against the named snapshot;
- `TESTS.md`: test design, production dependencies, anti-vacuity mutation,
  commands, and honest execution results;
- `EVIDENCE.md`: relevant source/Bead/history findings and any contradictions;
- `FILES/`: complete replacements only where they materially disambiguate the
  patch; omit it when unnecessary.

Before answering, reopen the ZIP, list and validate its members, compute its
SHA-256 and byte size, and confirm that `PATCH.diff` has no placeholders or
copied source snapshot. Your final chat response must begin with a substantive
operator-readable report of what you did and why. It must also state important
limitations, missing or unverified work, and how much additional value another
iteration could plausibly add—distinguishing a small repair from a substantial
second pass. Then report verification and risks and give a prominent working
link to the exact `/mnt/data/` ZIP. A bare download receipt is not acceptable.

## Continuation protocol

Do not perform a separate adversarial review unless the user explicitly asks
for one. If the user asks to **iterate** or **continue**, preserve valid prior
work, perform the highest-value remaining implementation/research pass, and
publish a new cohesive package revision with the same complete structure—not a
loose supplemental patch. Explain exactly what changed, what improved, what
still remains, and whether another iteration is likely to pay off.

If the user explicitly asks for an **adversarial review**, attack your prior
result against the original mission and current attached authority: search for
unsupported claims, invented or stale APIs, missing call sites, composition
failures, unsafe assumptions, vacuous tests, patch/apply defects, incomplete
acceptance criteria, and evidence that would falsify the design. Preserve work
that survives. Then repair every legitimate finding you can, regenerate the
entire cohesive package as the next revision, and report findings, repairs,
remaining disputes, and the value of another adversarial/implementation pass.
