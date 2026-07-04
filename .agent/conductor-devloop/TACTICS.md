# Polylogue Devloop Tactics

## If A Command Runs Long

Do not idle, and do not pretend contention is progress. While a
daemon/import/test is running, pick one foreground lane from
`.agent/scripts/devloop-ahead`:

- adjacent source audit;
- artifact/demo update;
- backlog/radar sharpening;
- subagent/audit prompt for a non-overlapping read-heavy lane;
- next verification command preparation;
- velocity/meta friction capture.

The output must be concrete: patch candidate, refreshed artifact, ranked next
slice, narrow audit prompt, proof command, or process-health check. Avoid starting another
heavy command against the same checkout/archive while the current proof is
pending.

## Greedy Batch Default

Batch toward full bead closure by default. The normal loop is:

1. audit the bead acceptance criteria;
2. gather evidence for all remaining criteria;
3. edit the coherent shared substrate in one batch;
4. run focused proof once for the batch;
5. generate one live/demo artifact that exercises the whole claim;
6. publish one PR for the complete bead.

Avoid turning each small helper, renderer field, construct declaration, or
artifact refresh into its own PR. A small PR is appropriate only when it closes
a named bead or phase, unblocks other active work, isolates genuine risk, or
keeps a truly large bead reviewable. Otherwise, keep working on the same branch
until the acceptance matrix is meaningful.

Phase splits are a fallback, not the default. Use them only when the full bead
would be materially harder to review, would mix different risk/deployment
timing, would block another active lane if held, or would force unrelated proof
families into one verification pass. If the remaining work is the same substrate
and the same proof artifact, keep batching.

When tempted to publish a partial slice, ask:

- Would this PR let the bead close, or would it just make the next agent read
  another PR to understand the same feature?
- Can the remaining acceptance criteria be implemented and verified with the
  same test/live-artifact pass?
- Is the split about risk/reviewability, or just because the current diff is
  already green?
- Would finishing the whole bead now be clearer than creating another
  integration boundary?

If the answer is "already green," keep batching.

## If Convergence Is Slow

Check, in order:

1. Is the daemon using the intended archive root?
2. Are there two roots or two daemons?
3. Is schema version current?
4. Which stage dominates logs?
5. Is the slow stage a real dataset cost or an avoidable global operation?
6. Is the run using optimized env where still relevant?

The current known suspicious shape is `full.index.attachments` dominating
catch-up chunks, sometimes ~50-80s per chunk. Treat that as a performance bug
candidate until disproved.

## If A Surface Looks Weird

Ask whether it is a function pretending to be a view. Recovery, export,
read-format flags, and demo commands should usually be projections over
query/read/render primitives, not separate ontologies.

## If A Demo Is Useful

Preserve the readable artifact and command trail under `.agent/demos` unless an
explicit task names another destination.
The artifact should state what it proves, what data it uses, how to rerun or
inspect it, and what caveats remain.

## If Scratch Gets Messy

Move durable research to `.agent/scratch/research/`, generated process/runtime
evidence to `.agent/task-history/`, superseded compact archaeology to
`.agent/archive/`, and current operating context to `.agent/conductor-devloop/`.
Run `.agent/scripts/devloop-sync` after material updates.
