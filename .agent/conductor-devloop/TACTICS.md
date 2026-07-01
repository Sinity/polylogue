# Polylogue Devloop Tactics

## If A Command Runs Long

Do not idle. While a daemon/import/test is running, do one useful foreground
task: inspect adjacent call sites, update a demo manifest, write the proof
claim, summarize the latest artifact, check archive-root drift, or improve the
scaffold.

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
