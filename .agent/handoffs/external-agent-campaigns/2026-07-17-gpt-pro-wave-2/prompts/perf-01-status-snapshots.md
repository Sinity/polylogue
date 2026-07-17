Title: "Budgeted per-component status snapshots: make every status surface sub-second with honest staleness (20d.17)"

Result ZIP: `perf-01-status-snapshots-r01.zip`

## Mission

Implement bead `polylogue-20d.17` (P1 — read its full record). Live
evidence: `polylogued status` produced NO result within 15 seconds while the
daemon heartbeat and DB descriptors were healthy; coordination status reads
measured 2.6–16.6 s. Root cause: request paths synchronously combine
millisecond facts with multi-second probes (raw census, debt, embedding,
Beads, process, archive, handoff). Output byte-bounding does not make status
interactive; the fix is the snapshot protocol the bead designs:

1. **One `StatusComponentSpec`/`StatusSnapshot` protocol** reused by
   daemon/archive status AND agent-coordination status. Each component
   declares: collector, dependencies, cost/detail class, deadline, refresh
   trigger or source fingerprint, staleness policy, privacy class, and
   projection fields.
2. **Off-request scheduler**: refreshes components independently (in the
   daemon's background loops — read `daemon/cli.py` periodic-loop idioms),
   retains last-good evidence, records state ∈ {fresh, stale, refreshing,
   timed_out, unavailable, degraded} with observed/start/finish timestamps
   and evidence refs.
3. **Request paths serve snapshots only**: CLI `polylogue status` /
   `polylogued status`, MCP status/readiness tools, HTTP status routes, and
   coordination envelopes SELECT components + detail class; no request path
   synchronously rebuilds the rich whole (bead AC #1). A stalled component
   returns its explicit state + age + last-good + deadline + detail ref and
   cannot delay healthy components (AC #2).
4. **Direct-mode (no daemon) story**: CLI status without a running daemon
   computes a bounded cheap subset and labels the expensive components
   `unavailable (daemon not running)` — honest, instant.
5. Tests: deterministic — inject slow/hanging collectors via the component
   spec and assert isolation, staleness marking, last-good retention, and
   sub-deadline response with the frozen-clock fixture
   (`tests/infra/frozen_clock.py`; the clock-hygiene lint rejects direct
   `time.time` in tests). Include one CLI-level test asserting the status
   command renders a mixed fresh/stale/timed_out grid correctly.

## Constraints

- Two known implementations to CONVERGE, not triplicate: `daemon/status.py`
  (~2.7k LOC) and `cli/commands/status.py` (~2.6k LOC) duplicate gathering
  (bead `polylogue-703` documents the intended one-assembly direction) —
  your protocol should be the shared core both consume; do the minimal
  convergence this requires and list the rest as 703 follow-up.
- The daemon stays sole writer; snapshot persistence (if any) belongs in
  ops.db (disposable tier — bootstrap ALTER acceptable).
- Coordinate: WebUI job webui-05 renders these snapshot states; keep the
  JSON projection field names self-describing and stable.

## Deliverable emphasis

HANDOFF.md: the protocol types (spelled fully), component inventory table
(every current status fact → component, cost class, deadline, refresh
trigger), scheduler wiring, before/after latency reasoning, 703-residual
list, and the JSON contract webui-05 consumes.


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
