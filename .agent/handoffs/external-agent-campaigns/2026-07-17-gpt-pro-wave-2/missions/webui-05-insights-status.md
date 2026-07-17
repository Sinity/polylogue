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
