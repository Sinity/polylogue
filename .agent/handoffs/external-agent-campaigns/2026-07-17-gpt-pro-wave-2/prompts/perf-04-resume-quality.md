Title: "Resume-candidate quality: fix dead-path Jaccard anti-selection and build the ranking evaluation harness (v1vo)"

Result ZIP: `perf-04-resume-quality-r01.zip`

## Mission

Implement bead `polylogue-v1vo` (P2, read its full record — it carries a
measured live finding from 2026-07-17): `find_resume_candidates`
(`polylogue/insights/resume.py`, `file_overlap` scoring ~line 652) computes
exact-path Jaccard between the caller's current `recent_files` and each
candidate session's historical `file_paths_touched`, weighted 0.25 in the
resume ranking that feeds the SessionStart context preamble and MCP resume
tools. Measured against the live corpus (4,264 polylogue-repo sessions,
27,785 file-touch occurrences): repo layout refactors kill exact-path
overlap, so the ranking ANTI-SELECTS exactly the refactor-ancestor sessions
an operator most wants to resume — sessions whose files have since moved
score zero.

Fix per the bead's decided design (scoring-time, profiles stay honest):

1. In `_profile_paths`/file_overlap scoring: (a) partition candidate paths
   into resolvable vs dead against the resume repo root (existence +
   file-to-package correction); (b) match dead paths by parent-directory
   prefix against the caller's recent_files directories (59% recovery,
   already measured — reproduce the measurement); (c) drop
   still-unresolvable paths from the Jaccard union so they stop deflating
   scores; (d) emit an `overlap_basis` breakdown (exact/dir/dead-excluded)
   so the ranking is explainable in resume output.
2. **Ranking evaluation harness** (the durable half): a small offline
   evaluator that scores ranking quality over synthetic-and-sampled
   scenarios — given a "current work" fixture and a candidate pool with
   known-correct resume targets (construct from lineage: the true parent/
   sibling sessions of a fork family are ground truth), report
   hit@1/hit@3/MRR before vs after the fix. This harness is what makes
   every FUTURE ranking change reviewable instead of vibes.
3. Tests from the bead's AC: 100%-dead-path session with shared directories
   gets nonzero file_overlap; two candidates identical except extra dead
   paths score equally; existing exact-overlap behavior unchanged (no
   regression on resolvable paths); overlap_basis rendered in the resume
   brief surfaces (`get_resume_brief` MCP tool + CLI `continue` path — find
   both consumers and update their projections).

## Constraints

- Scoring change only — do NOT rewrite stored session profiles (evidence
  stays as-captured; adaptation happens at query time). No schema changes.
- The resume surface feeds agent context (SessionStart hook) — keep output
  shapes backward-stable except the additive overlap_basis field; list
  every consumer touched (grep get_resume_brief / find_resume_candidates).
- Weight retuning (the 0.25) is OUT of scope unless the evaluation harness
  shows an obviously dominated setting — if so, propose, don't silently
  change.

## Deliverable emphasis

HANDOFF.md: the fix mechanism, measured before/after ranking metrics from
the harness (on synthetic + snapshot-derived fixtures; label live-corpus
numbers unverified), consumer-surface updates, and harness usage docs for
future ranking work.


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
