Title: "Stop mis-parenting Task-subagent self-compactions: prefix-membership test before parent assignment (4ts.3)"

Result ZIP: `lin-03-subagent-compaction-r01.zip`

## Mission

Implement bead `polylogue-4ts.3` (P1 — read its full record; root cause is
code-confirmed). The Claude Code parser's `agent-acompact-*` prefix
classifier assigns `parent = main-session` UNCONDITIONALLY, but ~39 of 187
such files in the live corpus are Task-SUBAGENT self-compactions (<90%
content overlap with the main session; 9 at 0% overlap). Consequence: the
parser asserts the wrong parent, and lineage composition prepends the WRONG
transcript — an agent reading the composed session sees a main-session
prefix that never belonged to it. This corrupts exactly the continuity
evidence the archive exists to preserve.

Fix (bead design is authority):

1. In the Claude parser's compaction classification (find it via the
   `agent-acompact` handling in `polylogue/sources/parsers/` —
   claude-code streaming/session parser), BEFORE assigning the main
   session as parent: test prefix content/UUID membership against the main
   session (or detect a fresh task-prompt head). On mismatch: treat as a
   fresh subagent — sidechain topology, NO inherited prefix.
2. Regression fixtures for both cases (bead AC): (a) a TRUE main-session
   `agent-acompact-*` whose prefix genuinely belongs to the main session →
   parent assignment + prefix-sharing inheritance preserved exactly as
   today; (b) a Task-subagent self-compaction (build from the 0%-overlap
   shape) → sidechain topology, spawned-fresh inheritance, no prefix.
   Derive fixture structure from real shapes but fully synthetic content
   (public repo).
3. **Reclassification path for existing data**: index.db is rebuildable —
   the fix takes effect on reparse. Deliver the verification query the
   integrator runs post-rebuild (count of agent-acompact sessions by
   parent-assignment class before/after; the ~39 misparented files should
   flip) and note that `session_links` rows re-resolve on save (writer
   stores divergent tails; `resolve_session_links_for_session` runs per
   save — confirm the reparse route re-evaluates them).
4. Sibling-route check (the repo's recurring failure shape is
   fixed-one-path-missed-the-twin): confirm whether the memory-bounded
   streaming path for multi-GiB Claude Code JSONL shares the classifier or
   duplicates it — if duplicated, fix both and add the divergence test.

## Constraints

- `Role`/`MaterialOrigin` semantics untouched; this is topology/parentage
  only.
- No schema changes; `session_links` vocabulary (prefix-sharing vs
  spawned-fresh, TopologyEdgeStatus) already expresses both outcomes.
- Detection must be shape/structural (content/UUID membership), never
  filename-only (detection tightness discipline in
  `sources/dispatch.py`).

## Deliverable emphasis

HANDOFF.md: classifier change mechanism, membership-test cost note (it
must not blow up streaming-path memory), fixture derivations, the
post-rebuild verification queries with expected count movement, and the
sibling-route audit result.


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
