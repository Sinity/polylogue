Title: "Seven continuity replay scenarios with independent known-answer oracles (the t8t black-box program)"

Result ZIP: `lin-02-continuity-scenarios-r01.zip`

## Mission

Bead `polylogue-t8t` (P0, read its complete record in the snapshot's
`.beads/issues.jsonl`) specifies seven black-box continuity jobs that a cold
agent must be able to complete against the archive: resume, forensic lookup,
prior art, decisions, postmortem, cost audit, and self-inspection — plus the
parallel-agent incident replay. The design is execution-grade; the code does
not exist. An earlier external attempt was REJECTED for implementing only two
of seven jobs and introducing a competing scenario seam — do not repeat that:
find and reuse the repo's existing scenario/fixture machinery
(`polylogue/scenarios/`, `tests/infra/`, demo corpus seeding) and Beads notes
about where scenario declarations belong.

Implement the full declaration set + oracle machinery:

1. Seven scenario declarations, each: the operator-language question, the
   fixture archive state it runs against, the expected answer expressed as
   INDEPENDENT planted facts (from fixture construction/source evidence —
   never computed via the production query route under test), and the
   allowed query surfaces.
2. A known-answer oracle harness that executes a scenario through the real
   public route (CLI/API/MCP-shaped entry), compares against planted facts,
   and reports per-scenario pass/fail with evidence refs — designed so the
   same scenarios can later run against the live archive as the z9gh.7
   terminal gate (parameterize the corpus, don't hardcode fixtures).
3. Fixtures for all seven jobs, including the parallel-agent incident shape
   (coordinator + worker children where the corrected known-answer population
   is 129 coordinator children: 91 in the specific run, 38 not — mirror that
   structure synthetically).
4. Tests proving oracle independence: mutate the production query route
   (e.g. drop a filter) and show the scenario FAILS; mutate a planted fact
   and show the mismatch is reported with the right diagnosis.

## Constraints

- No new scenario framework if an existing seam serves; extend
  `scenarios`/test-infra idioms. State explicitly which seam you chose and
  why it is not a competing parallel abstraction.
- Scenario fixtures are synthetic/sanitized only.

## Deliverable emphasis

HANDOFF.md: scenario inventory table (job → question → oracle facts →
route), harness entry points, how the z9gh.7 live replay would invoke these
against the real archive, exactly which t8t acceptance criteria are
satisfied vs remaining, and proposed follow-up beads.


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
