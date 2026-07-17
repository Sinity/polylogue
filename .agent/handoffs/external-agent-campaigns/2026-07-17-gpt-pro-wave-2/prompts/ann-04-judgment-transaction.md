Title: "Finish the canonical judgment transaction: evidence disclosure, queue health, TOCTOU-safe writes, one operator lifecycle"

Result ZIP: `ann-04-judgment-transaction-r01.zip`

## Mission

Complete beads `polylogue-37t.12` (P1) + `polylogue-41ow` (P1 bug) + prepare
`polylogue-mrxt`'s canary. PR #2791 already landed the core: candidate
lifecycle transition authority, bulk SAVEPOINT semantics, explicit injection
authorization, immutable retry/conflict behavior, MCP review capability, and
the root `polylogue judge` workflow. Treat that merged lifecycle as the SOLE
transaction authority — audit and extend, never reimplement (37t.12 design
is explicit). Read all three beads fully in `.beads/issues.jsonl`.

Work items:

1. **41ow first** (it corrupts trust in everything else):
   `polylogue/storage/sqlite/archive_tiers/user_write.py::upsert_assertion`
   has a reproduced TOCTOU race that silently reverts operator judgments
   when an automated candidate update interleaves. Decided fix: one
   BEGIN IMMEDIATE preserve/write transaction — operator judgments must
   never be silently reverted. Also reconcile the `tilk` finding (upsert_*
   identity semantics inconsistent: content-hash append vs stable-identity
   update — classify each upsert_ helper and document/align its identity
   rule). Regression test = the reproduced interleaving (the bead cites the
   reproduction; find it in dogfood-2 investigation notes under
   `.agent/scratch/dogfood-2/investigations/` if present in the snapshot).
2. **37t.12 residuals**: evidence disclosure (a judgment presents the
   evidence refs it judged — surface them through the judge workflow and
   MCP review capability); queue health (pending-candidate counts/ages as a
   queryable product, wired into status); retire the duplicate `mark
   candidates` public workflow after porting any still-useful presentation
   onto root `polylogue judge` (parity first, then removal — bead design
   names `cli/query_verbs.py` as the duplicate's home and
   `click_command_registration.py` as the canonical registration).
3. **The mrxt canary, prepared not faked**: mrxt requires a GENUINE
   operator-authored action through the production route — you cannot
   perform it (no live archive). Deliver the exact operator script: the
   commands, the expected durable effects at each step (candidate row →
   verdict → assertion + evidence ref + judgment receipt + context policy →
   surface visibility), and the verification queries — so the operator
   executes the canary in minutes and any friction becomes child beads.
4. Tests: 41ow interleaving regression; lifecycle invariants from 37t.12's
   AC (machine candidates stay candidate/inject=false; default accept
   promotes one active inject=false assertion; explicit authenticated
   review may inject; reject/defer do not promote; exact retries
   idempotent; changed decisions conflict); bulk semantics
   (valid/idempotent/malformed/conflicting refs).

## Constraints

- user.db is durable-irreplaceable: no schema changes unless additive
  numbered migration + backup-manifest rules demand it — prefer none.
- The judgment lifecycle is the authority the annotation program (parallel
  jobs ann-01/02/03) builds on — keep interfaces stable and document them.

## Deliverable emphasis

HANDOFF.md: what PR #2791 already guaranteed vs what you added (exact),
the 41ow fix mechanism + proof, upsert identity-semantics table, duplicate-
workflow retirement diff, the operator canary script, and the queue-health
query surface.


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
