---
created: 2026-07-16
purpose: Decide query admission, cancellation, work bounds, result addressability, and cleanup for L14-L17
status: recommended-decision
project: polylogue
---

# Query cancellation and bounds

## Decision

Every query surface executes through one immutable `QueryExecutionContext` and
one owned read lifecycle. Admission is bounded and fair; SQLite work runs on a
dedicated read-only connection in a worker thread with a progress handler and
interrupt support; cancellation/deadline/disconnect converge on the same state
machine; large logical results page or spool losslessly.

Reject only syntactically unsafe or unauthorized work. Do not impose semantic
row caps or metadata-only refusals to make the transport appear bounded.

## Query execution context

The context contains:

- stable query/run identity and canonical plan digest;
- actor/role and disclosure policy;
- monotonic deadline and cancellation event;
- workload class, admission weight, and resource budget;
- source/archive snapshot identity;
- owned reader, temp/spool, cursor, and continuation references;
- progress counters over logical selected/scanned/emitted work;
- terminal state and cleanup receipt.

It is created at the outermost CLI/API/MCP/HTTP request boundary and passed to
lowering/execution/readers. No adapter invents a second timeout or background
task that outlives it anonymously.

## Admission

`QueryAdmissionController` provides bounded queues and weighted fairness across
interactive, background/export, and maintenance classes. FIFO holds within a
class. Weights and concurrency are tuned from workload-profile evidence and
recorded in config; they do not change query meaning.

Overload returns a typed `admission_deferred`/`busy` result with retry timing or
an owned resumable query ref. It must not create untracked work before
admission. Health/status reads use a reserved small lane and component snapshots,
so incident queries cannot starve them.

## Interruptible SQLite read

`InterruptibleSQLiteRead`:

1. creates a dedicated read-only connection inside its worker thread;
2. applies the canonical connection profile and begins one read snapshot;
3. installs a progress handler that checks deadline/cancellation/work budget;
4. permits the coordinator to call `connection.interrupt()` for disconnect or
   explicit cancel;
5. translates SQLite interruption into the common typed terminal state;
6. rolls back/closes and releases every cursor/temp resource in `finally`.

Do not interrupt a shared active connection or the writer. Threads are preferred
to subprocesses for SQLite query execution because they preserve connection and
snapshot ownership without serializing results. CPU-heavy post-processing may
use a bounded process worker later under the same execution context.

## Input and work bounds

- Before Lark transformation, run a streaming token/depth/length scan with a
  parser-complexity budget; excessive nesting fails with a typed syntax/resource
  error before recursive allocation.
- Lowering must expose estimated/declared unit, pushdown, projection, ordering,
  and continuation behavior.
- Runtime counters distinguish rows scanned, logical units selected, relation
  expansions, bytes spooled, and rows emitted.
- Limits constrain collection work only when the query explicitly requests a
  top-k/sample/page semantic. A renderer byte cap cannot silently become a
  query row cap.

## Result lifecycle

Small results return inline. Larger results become an explicitly owned
`QueryResultRef` keyed by query run and snapshot, with schema, completeness,
page cursor, byte/row counts, expiry/retention policy, and cleanup state.

Paging is stable over the captured snapshot and distinguishes exhaustive page,
top-k, sample, aggregate, bounded context, and recursive graph results. A client
disconnect cancels computation unless the request explicitly asked for a
resumable result; resumable work has a durable/owned receipt and bounded spool.

## Terminal states

`completed`, `cancelled_by_client`, `deadline_exceeded`, `work_budget_exceeded`,
`admission_deferred`, `syntax_budget_exceeded`, `failed`, and `abandoned` are
distinct. Each records partial work and cleanup outcome. A partial page is never
labelled complete.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Async wrapper with timeout only | Small change | SQLite continues after coroutine cancellation and holds resources |
| One shared read connection | Fewer connections | Unsafe cancellation/snapshot ownership and head-of-line blocking |
| Process per query | Strong isolation | High serialization/startup cost; unnecessary for SQLite interruption |
| Hard maximum rows | Simple memory bound | Changes semantics and hides exhaustive-result obligations |
| Renderer truncation | Easy transport compliance | Work is still unbounded and result completeness becomes dishonest |
| Reject all structurally broad queries | Protects resources | Prevents valid bounded/spooled analysis; admission and cancellation solve the actual problem |
| Separate cancellation logic in MCP/HTTP | Adapter-local implementation | Disconnect behavior and cleanup drift across surfaces |

## Migration sequence

1. Introduce context and interruptible reader under one canonical query route.
2. Thread HTTP/MCP disconnect and CLI signals into the context.
3. Add admission and progress counters, then set budgets from workload tiers.
4. Add result refs/page-spool lifecycle before retiring capped convenience paths.
5. Move all query/read/get surfaces to the common transaction during the MCP
   verb-algebra rewrite; preserve current aliases until equivalence proof.

## Required proof

- deterministic cancellation during parse, SQLite scan, relation expansion,
  rendering, and spool write releases all resources;
- repeated concurrent incident-scale calls keep status responsive and return to
  steady RSS/PSS/temp/reader counts;
- equivalent queries conserve membership and snapshot identity;
- irrelevant archive growth does not increase selected-work counters beyond the
  declared index/path complexity;
- nested-input mutation fails before parser recursion;
- transport page boundaries concatenate to the exact logical result;
- temporary removal of progress handler/interrupt/cleanup or restoration of a
  semantic cap fails the survivor laws.

Primary evidence: `polylogue-z9gh.1`, `polylogue-u0dm`, `polylogue-rsad`,
`polylogue-20d.17`, `polylogue-t46.8`; query expression/lowering/evaluator,
SQLite connection profile, and surface adapters.
