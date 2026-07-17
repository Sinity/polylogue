Title: "Lower Codex functions.exec child operations into typed, provenance-linked actions (j2zz)"

Result ZIP: `lin-04-codex-delegations-r01.zip`

## Mission

Implement bead `polylogue-j2zz` (P1 — read its full record). Modern Codex
embeds typed operations INSIDE `functions.exec` JavaScript envelopes. Live
evidence from the newest 100-session sample: every session had nested tool
calls; 14,004 envelopes held child operations; 19,180 results yielded ZERO
structured paths or outcomes even though 1,444 result texts contained
`exit_code`. Polylogue currently retains only the outer exec/shell
semantics — so for modern Codex work, the actions relation (the archive's
core forensic value: what tools ran, what failed) is nearly blind. This
also silently biases the claim-vs-evidence analysis (Codex structured
failures undercounted).

Build the lowering (bead design is authority):

1. **Typed child registry**: lower `functions.exec` children —
   `exec_command`, `apply_patch`, `write_stdin`, `update_plan`, `wait`,
   `web`, `image`, MCP calls, and unknown shapes — into provenance-linked
   child actions while RETAINING the outer call as transport (evidence
   preserved, interpretation added; never destroy the envelope).
2. **Structural promotion only**: promote only structural result fields
   (exit codes, paths, byte counts) — outcome fields are structural or
   `unknown`, never regex-guessed from prose (the tool_result_is_error
   discipline). Commands and patches expose normalized command strings and
   touched paths.
3. **Ordering + pairing**: preserve ordering and repeated calls; child
   use/result pairing is deterministic (Nth-use↔Nth-result within
   envelope), continuations pair without inventing recovery; malformed and
   unknown children retain raw evidence with typed unknown state.
4. **Feed the bounded relation**: children land in the `action_pairs`
   derived relation that `polylogue-z9gh.2` landed (PR #3018 — read the
   current `action_pairs` schema/rebuild path in
   `storage/sqlite/archive_tiers/` and the actions compatibility view).
   Parent/child linkage: child actions carry the transport action's
   identity as provenance so delegation queries can distinguish outer
   transport from inner operations.
5. **Fixtures + live-sample report**: fixtures lowering single and
   multiple children (bead AC enumerates the cases); plus a snapshot-
   runnable census script reporting child/parent counts and structured-
   outcome coverage over any Codex session corpus — the integrator runs it
   on the live archive to produce the before/after coverage numbers
   (14,004 envelopes → N typed children, 0 → M structured outcomes).

## Constraints

- Parser layer changes live in `polylogue/sources/parsers/` (codex parser)
  + normalization; keep detection tightness order intact
  (`sources/dispatch.py`).
- Reparse semantics: this is semantic-reparse-affecting for Codex sessions
  — content hash includes blocks, so lowering that CHANGES block structure
  changes import identity. Read `pipeline/ids.py` hashing rules and state
  clearly in HANDOFF whether your lowering alters content hashes (and thus
  triggers re-import of Codex sessions on next ingest) — if yes, that is
  acceptable but must be explicit, with the expected re-ingest cost.
- Index-tier effects (new action rows) are rebuild-route only; no durable
  tier changes.

## Deliverable emphasis

HANDOFF.md: registry design, per-child-type field promotion table,
pairing/ordering semantics, content-hash impact statement, census-script
usage, and the exact claim-vs-evidence interaction (which undercounts this
fixes — coordinate wording with the parallel demo lane's economy work).


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
