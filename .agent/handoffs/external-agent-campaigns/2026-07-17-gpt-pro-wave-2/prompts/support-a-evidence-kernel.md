# Lane A support — first-party analysis-evidence kernel and synthetic demo proof

You are producing an implementation package for Polylogue, a local archive for
AI sessions. Work from the attached fresh Chisel archive; it is authoritative
over any prior branch/PR memory. Your package will be integrated by a local
lane that separately has access to the private live archive.

## Mission

Implement the reusable first-party representation needed for an analysis to be
an archive object, not a bespoke report: a content-addressed analysis
definition, an immutable analysis-run receipt bound to existing query result
sets/evaluation receipts and archive generations, and materialized findings
which cite that evidence. Use the existing `user.db` assertion and
query-object substrate; do not invent a parallel report registry or make an
unjustified durable schema migration.

Build a privacy-safe synthetic fixture and a real-route test that proves:

1. a definition has stable identity;
2. a run binds the exact inputs, evaluator/world, and privacy/retention stamp;
3. a finding cannot be read as evidence-free prose; and
4. re-running with changed corpus/evaluator inputs creates a distinguishable
   new receipt rather than overwriting history.

Trace the current query-object, evaluation-receipt, assertion, and public
claims code before choosing names. Fit the smallest extension into its native
layer and expose a typed read path only where the product has a real consumer.

## Boundary

Do not regenerate a private claim-vs-evidence report, inspect a live archive,
or alter classifier semantics. The paired local lane owns calibration,
interpretation, and live evidence. Your deliverable must be independently
valuable for any future analysis, not hard-coded to one metric.

## Required package

Return `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`. Include exact
file paths, an acceptance matrix, migration classification, and tests that
exercise production storage/read code—not a toy stand-in. Do not claim tests
were run. If the snapshot proves a necessary prerequisite already exists or a
durable migration is unavoidable, report that precisely rather than fabricating
a second substrate.


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
