Title: "WebUI v2 generated client contracts: one typed client from the daemon's declared read/operation surface"

Result ZIP: `webui-08-client-contracts-r01.zip`

## Mission

WebUI v2 verticals must not hand-write fetch calls against informally-known
JSON shapes — that recreates the marshaling triplication the repo is
actively killing. Build the generated-client layer: TypeScript types +
a thin typed client generated from the daemon's declared surface, so a
server-side contract change breaks the web build instead of production.

Ground truth to read first:

- `devtools render openapi` and its output (find the generated OpenAPI/
  schema artifacts in the snapshot — `devtools/` + `docs/` + `schemas/`);
  also `render cli-output-schemas` machinery, which shows how the repo
  already generates typed output contracts from Python models.
- The declaration direction: beads `polylogue-o21.1` (DeclarationSpec
  kernel: declare-once → generate registration/schemas/inventories/docs)
  and `polylogue-t46.8.1` (MCP declaration registry pilot at
  `polylogue/mcp/declarations/`). A parallel local lane is cutting MCP to a
  six-tool declared surface; the daemon HTTP JSON surface is expected to
  converge on the same declared read contract. Your generator should
  consume whatever declaration/schema artifact exists in the snapshot and
  be trivially retargetable when the declaration kernel lands.
- `polylogue/archive/query/transaction.py`: QueryResultPage/continuation/
  result-ref vocabulary — the paging envelope every read returns.

Deliver:

1. A generator (Node or Python script, run at build time, committed output):
   OpenAPI/declared-schema → TypeScript types + typed client functions with
   the continuation envelope modeled once (generic `Page<T>` with cursor,
   coverage exact/qualified, result refs). Deterministic output, diffable,
   with a `--check` mode for CI drift detection (mirror the repo's
   `render all --check` idiom).
2. Runtime client: same-origin fetch wrapper with typed errors (map the
   daemon's error envelope), deadline/abort support, and a continuation
   iterator utility (`for await (const page of client.query(...))`) that the
   verticals adopt.
3. Contract tests: golden-file tests pinning generated output for a fixture
   schema; a drift test that fails when the daemon schema and committed
   client diverge; unit tests for the continuation iterator including a
   truncation/qualified-total page.
4. Adoption notes for webui-02…06 (which hand-written calls each replaces).

## Constraints

- Do not invent a parallel schema source: consume the repo's generated
  artifacts; where a route lacks schema coverage, add the smallest schema
  declaration server-side following existing `render` machinery, and list
  every such addition.
- Zero runtime dependencies beyond what webui-01's scaffold establishes.

## Deliverable emphasis

HANDOFF.md: generator design, exact schema sources consumed, the Page/
continuation type contract (spelled fully), drift-check wiring for CI,
per-vertical adoption map, and how this retargets onto the o21.1/t46.8.1
declaration kernel when it lands.


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
