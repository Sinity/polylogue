Title: "WebUI v2 vertical: insights, named-source freshness, and status panels with evidence-state honesty"

Result ZIP: `webui-05-insights-status-r01.zip`

## Mission

Build the observability vertical: what an operator sees when they ask "what
does the archive know, how fresh is it, and is the system healthy?" —
rendered honestly (the archive's core discipline: absence, staleness, and
unknown are first-class states, never zeros or blanks).

Ground truth to read first:

- `polylogue/insights/registry.py` — `INSIGHT_REGISTRY` is descriptor-driven:
  each InsightType declares field accessors, query model, operations method,
  CLI/MCP metadata, JSON key, readiness/export behavior. Your panels must be
  GENERATED from this registry (one loop over descriptors), not hand-built
  per insight — a new insight type should appear in the web UI with zero
  web-code changes.
- Named-source freshness (bead `polylogue-1xc.13`, merged PR #2924): exact
  per-source projection joining filesystem state, cursor/exclusion, accepted
  authority, application evidence, index high-water, FTS, and insight debt —
  五 stages from unseen→searchable, with excluded/cursor-ahead/broken-head
  degradation. Find its status/MCP surface and render it as the freshness
  panel (per-origin lanes, stage badges, degradation reasons).
- Status: bead `polylogue-20d.17` (P1, open) defines the target
  StatusComponentSpec/StatusSnapshot protocol (budgeted per-component
  snapshots with fresh/stale/refreshing/timed_out/unavailable/degraded +
  last-good evidence + ages). A parallel job (perf-01) drafts that backend;
  DESIGN YOUR PANEL TO THAT PROTOCOL (state the interface assumption), with
  a fallback adapter over today's status JSON.

Deliver:

1. Insights browser: registry-generated panel list; each insight renders its
   plaintext/JSON fields with provenance (materializer version, evidence
   refs where the model carries them) and its readiness state.
2. Freshness panel: per-named-source stage ladder with exact counts, ages,
   and degradation states; excluded/cursor-ahead render as attention states.
3. Status panel: component grid honoring the snapshot protocol states —
   a timed-out component shows its last-good value + age, never blocks
   healthy siblings (mirror 20d.17's AC in the UI contract).
4. SSR + islands; Vitest tests including one asserting that adding a fake
   descriptor to a test registry makes a panel appear (the zero-web-change
   regression test); Python route tests for the JSON contracts.

## Constraints

- No client-side aggregation/reinterpretation; surfaces project.
- Read `docs/daemon.md` for convergence semantics so degraded states use the
  system's real vocabulary (pending debt ≠ failure — `false_means_pending`).
- Zero CDN; sanitized fixtures.

## Deliverable emphasis

HANDOFF.md: the registry→panel generation mechanism, freshness/status JSON
contracts consumed (exact fields), the 20d.17 interface assumption spelled as
a typed protocol, fallback-adapter notes, superseded web_shell files list,
and what perf-01's backend must provide for zero-rework integration.


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
