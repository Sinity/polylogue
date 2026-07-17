Title: "Hot/thick daemon: keep-warm read services, bounded catch-up latency, and a measured memory envelope"

Result ZIP: `perf-02-hot-daemon-r01.zip`

## Mission

The operator wants Polylogue to feel instant: the daemon (`polylogued`) is
always running, so reads should never pay cold-start, cache-cold, or
catch-up-storm costs. Today: status assembly has been observed at 15+ seconds
(`polylogue-20d.17`), coordination reads at 2.6–16.6 s, full-ingest catch-up
latency and WAL shape are untracked (`20d.6`), and the daemon catch-up memory
envelope is unmeasured/unbounded (`ng9m`). Read those Beads plus `20d` epic
context, `daemon/cli.py`, `daemon/convergence*.py`, `daemon/http.py`,
`daemon/status.py`, and the interruptible-read/QueryTransaction seams in
`archive/query/`.

Produce an implementation draft for the thick-daemon architecture:

1. **Keep-warm read plane**: persistent read connections with page-cache
   affinity (mmap/cache_size pragmas chosen deliberately), pre-warmed
   statement caches for the hot read families (list/read/search/status),
   and an explicit warm-up pass at daemon start + after index generation
   switches. Justify every pragma against SQLite documentation semantics.
2. **Snappy status**: integrate with (do not duplicate) the per-component
   budgeted snapshot direction of `20d.17` — a component scheduler with
   fingerprints, last-good values, deadlines, and staleness marks, so
   `polylogue status` and the HTTP status route serve sub-second from
   snapshots while gathering happens in the background.
3. **Catch-up discipline**: bound ingest catch-up bursts (batch sizes, WAL
   checkpoint cadence, backpressure between watcher and converger) so a
   returning daemon neither starves interactive reads nor balloons RSS;
   define the measured memory envelope and the enforcement mechanism
   (`ng9m`) with degradation instead of OOM.
4. **CLI fast path — interface only**: a parallel job (perf-03) OWNS the
   CLI's daemon-first read path. Your part: make the daemon side ready for
   it — a cheap health/warmth probe endpoint (what perf-03's detection
   calls) and the guarantee that hot read families stay warm. Specify the
   probe contract in HANDOFF.md; do not implement CLI-side changes.
5. Tests: latency assertions on the snapshot path (fixture-scale), catch-up
   backpressure unit tests with the frozen clock, and a memory-envelope test
   using deterministic work counters rather than wall-clock sleeps.

## Constraints

- The daemon remains the sole writer; nothing here may add a second writer
  or a cache that can serve archive-contradicting data without a staleness
  mark (evidence-honesty rule).
- Respect the convergence architecture (check/execute stages, debt); do not
  replace it — bound and schedule it.
- Mark all live-host measurements `unverified`; provide the measurement
  commands for the integrator to run locally.

## Deliverable emphasis

HANDOFF.md: architecture summary, exact files changed/added, pragma/limit
table with rationale, the status-snapshot integration contract with 20d.17,
measurement plan (commands + expected envelopes), and staged integration
order (what can merge independently).


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
