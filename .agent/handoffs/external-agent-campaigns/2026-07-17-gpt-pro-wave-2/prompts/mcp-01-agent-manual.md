Title: "Six-tool-era standing agent manual: regenerate the cold-start manual and continuity recipes for the replaced MCP surface"

Result ZIP: `mcp-01-agent-manual-r01.zip`

## Mission

Polylogue's MCP surface is being cut over from 103 tools to a six-tool
transaction surface (default read role: `query`, `read`, `get`, `explain`,
`context`, `status`; continuation as input/result-ref protocol; write/judge/
run/operate role-gated; URI resources for stable objects; MCP prompts for
workflows). That cutover is a LOCAL lane running in parallel (bead
`polylogue-t46.8` + children — read them). Separately, an external package
("beads-06", covering beads `polylogue-3gd.2`/`3gd.3`) already delivered an
executable agent-manual + installation kit against the 103-tool surface:
project-owned cold-start manual, typed recipe/capability declarations,
role-scoped MCP manifests, native installation for Claude Code / Codex /
Gemini CLI / Hermes, and verification lanes. Check whether its content is in
your snapshot under `.agent/handoffs/external-agent-campaigns/`
(`beads-06` results) or already merged into source; adjudicate and REUSE its
architecture — do not design a competing manual system.

Your job: produce the six-tool-era manual CONTENT and generation pipeline so
that when the cutover lands, agents get a correct, cache-stable, installable
manual with zero "fetch the manual first" turns:

1. **Manual content, generated from declarations**: normal invocation of
   each of the six tools with realistic argument examples; the continuation
   protocol (how to resume a truncated result — exact request shape);
   result-limit semantics and recovery; result-ref citation discipline;
   role ladder (what write/judge/run/operate add and how confirm gates
   work); URI resource addressing; source coverage (the 10 Origin tokens
   with one-line meaning each); continuity workflows (resume a session,
   forensic lookup, prior-art search, cost audit) as step-by-step recipes
   using ONLY the six tools. Every example must be derivable/compilable
   against declarations — follow the beads-06 pattern of verification lanes
   that resolve generated routes against production declarations.
2. **Query-language teaching**: a compact DSL section whose every example
   round-trips the real parser (`archive/query/expression.py`) — extract
   valid examples from repo tests; include the strict-command-floor rule and
   the three intent signals (find keyword / quoted expression / field
   syntax).
3. **Install/delivery**: reconcile with beads-06's SessionStart/native
   installation mechanisms — what changes for the six-tool era, what is
   untouched; per-client (Claude Code, Codex, Gemini, Hermes) deltas.
4. Tests: manual-compilation lane (examples resolve against declarations;
   parser round-trip for DSL examples), staleness check (manual regenerates
   deterministically; drift fails a `--check`).

## Constraints

- If the six-tool declarations are not yet in your snapshot, write the
  manual against the contract stated above and the t46.8.1 declaration
  models (`polylogue/mcp/declarations/` if present), parameterizing tool
  schemas so integration is mechanical; state exactly what must be
  re-generated post-cutover.
- The manual is public-repo content: no private paths/data in examples.

## Deliverable emphasis

HANDOFF.md: adjudication verdict on beads-06 reuse (what you kept/changed),
manual generation pipeline, the complete manual itself (rendered), per-client
install deltas, post-cutover regeneration checklist, and which parts are
blocked on cutover-final schemas.


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
