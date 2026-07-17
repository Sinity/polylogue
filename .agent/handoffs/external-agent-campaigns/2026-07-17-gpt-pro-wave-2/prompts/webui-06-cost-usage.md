Title: "WebUI v2 vertical: cost/usage explorer that keeps reported, derived, priced, and estimated lanes distinct"

Result ZIP: `webui-06-cost-usage-r01.zip`

## Mission

Build the cost/usage explorer for WebUI v2 (scaffold interface per webui-01;
state assumptions if its result is absent). Polylogue's cost model is
deliberately multi-lane and the UI must NOT collapse lanes into one number —
that collapse is a named anti-goal (`docs/cost-model.md`, read it fully).

Ground truth and traps (all verifiable in the snapshot):

- Authoritative token/usage rows: `session_model_usage` and provider-usage
  events. Bead `polylogue-r7p6`: `session_profiles` token columns undercount
  Codex 1000× — never source from profiles.
- Codex `input` INCLUDES cached tokens (~96% of it) and `output` includes
  reasoning; disjoint-lane billing or you double-count (this bug once caused
  7.69× cost inflation — commit `3938bc6c2` history).
- Claude subscription semantics: cache reads effectively free on Max/Pro;
  `cost_usd` is API-list-equivalent and overstates subscription spend — the
  UI should offer both "API-equivalent" and "subscription-credit" views
  where the data model exposes them.
- Bead `polylogue-f2qv.6` (open): profiles/costs not yet reconciled to exact
  provider usage — your UI must render the reconciliation state rather than
  pretending agreement.

Deliver:

1. Overview: spend/tokens by time bucket × origin × model, with lane
   selector (reported $ / catalog-priced $ / estimated) and an explicit lane
   legend explaining what each lane means (copy semantics from
   docs/cost-model.md, don't paraphrase loosely).
2. Cache economics view: cache-read share per model/origin over time, cache
   hit ratio, "what cached-token accounting would look like if billed naively"
   toggle — this is the view an agentic-cost-obsessed reader inspects first;
   make it exact and beautiful.
3. Session drill-down: per-session usage panel (API calls, token lanes,
   models used, per-model rows) linking into the webui-02 read page.
4. Honest absence: sessions/origins lacking usage evidence render as
   "no usage evidence", never $0. Aggregates must state coverage (N of M
   sessions carry usage rows).
5. SSR first page + islands; server-computed aggregates ONLY (client never
   sums rows — coverage/qualification must come from the server); Vitest +
   Python route tests including a coverage-qualification regression test.

## Constraints

- Continuation-paged JSON per the shared QueryTransaction vocabulary
  (`archive/query/transaction.py`).
- If an aggregate endpoint you need is missing server-side, add the smallest
  daemon route over existing operations/repository methods — no new SQL in
  the web layer; check `api/archive.py` cost/usage verbs first.
- Zero CDN; sanitized fixtures.

## Deliverable emphasis

HANDOFF.md: lane semantics table as implemented, JSON contracts, coverage/
qualification mechanism, screenshots-in-words of the cache-economics view,
f2qv.6 interaction notes (what changes when it lands), superseded web_shell
files, decisions the integrator could overturn.


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
