# Fork 07 — Polylogue web reliability and truthful degraded states

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-0hqs` and `polylogue-bby.1`. This is the launch-blocking reliability lane.

## Mission

Prevent the web reader from hanging or presenting partial/stale data as an ordinary successful response. The product must return a bounded truthful result under slow archive queries.

## Owned scope

Own daemon HTTP query execution, request concurrency/timeout controls, cancellation and cleanup, response envelopes for degraded/timeout states, and the corresponding web UX. Avoid README/demo-fixture edits.

## Requirements

- bound concurrent archive-query execution;
- impose per-request timeouts with a clear server-side ceiling;
- ensure timed-out work does not continue consuming the shared executor indefinitely;
- return a typed degraded response carrying reason, elapsed time, partiality, and relevant readiness/frontier state;
- distinguish timeout, unavailable projection, stale index, invalid query, and internal failure;
- render slow/loading, partial, unavailable, and retry states visibly in the web reader;
- preserve request IDs and audit/ref information;
- do not silently retry a non-idempotent operation;
- prevent one pathological query from starving other requests.

## Test design

Use deterministic fault injection rather than wall-clock sleeps where possible. Cover:

- saturation of the worker bound;
- one timeout while a second cheap query succeeds;
- cancellation/cleanup;
- late worker completion not mutating an already-final response;
- browser rendering of each degraded state;
- no raw exception or private path leakage;
- metrics or reflection evidence for the timeout.

## Launch gate

Define a narrow public demo-route SLO and test it on the deterministic archive. Keep real-archive timings as field observations, not universal promises.

## Deliverables

Produce the patch, focused load/fault tests, a response-envelope example, screenshots of degraded UX, measured deterministic timings, and any residual risk requiring architectural follow-up.
