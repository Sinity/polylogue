Title: "Order-independent lineage writes and branch-point safety: close the three production-shaped state-machine failures (866e)"

Result ZIP: `lin-01-lineage-law-r01.zip`

## Mission

Implement bead `polylogue-866e` (P0, in_progress — read its FULL record and
notes; PR #2922 already landed canonical sibling ordering and
missing-branch-cut safety, so verify current master state first and build
the REMAINING law). Stateful property testing found three production-shaped
failures on clean master: (1) repeated full replacement with sibling
variants plus child-before-parent ingestion can retain OLDER sibling text
as primary; (2) deleting a referenced parent branch point can leave a child
whose modeled prefix exceeds the surviving parent and CRASH composition;
(3) the general invariant — equivalent lineage histories must converge to
identical composed reads under ANY arrival/replacement order — does not
hold universally.

The write-path authority map (bead design, verify line numbers against the
snapshot): `polylogue/storage/sqlite/archive_tiers/write.py` — full-replace
normalization ~235–455, batch link normalization ~1858–1889,
composed-signature/prefix alignment ~3167–3454, stale branch-point repair
~3478–3594. Link resolution:
`storage/sqlite/queries/session_links.py` ~175–341. Independent read
oracle: `storage/sqlite/queries/message_query_reads.py` ~74+.

Deliver:

1. **Reproduce first**: run the saved Hypothesis failure classes
   (`POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1` against
   `tests/property/test_write_path_state_machine.py`) on the snapshot
   baseline; commit each class as a named deterministic transition fixture
   with the three oracles the AC demands (physical-row, link-row,
   composed-read).
2. **The law**: make lineage write transitions order-independent —
   equivalent histories (same final logical content, any arrival order:
   child-first, parent-replaced-later, sibling-variant races) converge to
   identical composed reads; transitions atomic and rollback-safe; missing/
   dangling relations remain TYPED, readable states (LineageCompleteness
   vocabulary — the Diet architecture decision
   `02-lineage-composition-and-snapshots.md` in the snapshot's testdiet
   context is the ratified contract: canonical divergent tails + typed
   edges, one deferred read snapshot, incomplete lineage explicit, never
   fabricated content).
3. **Branch-point safety**: parent deletion/replacement paths guarantee a
   child's modeled prefix never exceeds surviving parent content — degrade
   to explicit truncated-lineage state instead of crashing composition.
   Remember the load-bearing constraint: `branch_point_message_id` is
   deliberately NOT a FK (full-replace deletes+reinserts parent messages;
   SET NULL would destroy the splice) — preserve that.
4. Tests: the state machine extended with the adversarial transitions
   (sibling variant replacement, parent-delete-then-reingest, quarantined
   cycles), each named class asserting through all three oracles; plus the
   composed-read equivalence property over permuted histories.

## Constraints

- Content-hash idempotency must survive: equal material stays idempotent;
  your normalization cannot change import identity semantics
  (`pipeline/ids.py`, `core/hashing.py`).
- index.db is rebuildable: if a fix requires reinterpreting stored rows,
  the canonical DDL + rebuild is the route, never in-place migration.
- This is the hottest storage hotspot: keep the diff surgical; every
  changed write-path function needs its named mutation-killing test.

## Deliverable emphasis

HANDOFF.md: per-failure-class root cause + fix mechanism, the equivalence
law statement as implemented, oracle design, what PR #2922 had already
covered vs what you added, composition-degradation semantics table, and
residual risks for the integrator's live-archive validation.


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
