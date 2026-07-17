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
