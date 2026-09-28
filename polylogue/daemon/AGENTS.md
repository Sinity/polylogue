# Daemon

## Code Review Rules

- Interruption: for each multi-step state change, flag what a cancel,
  deadline, kill, or restart between steps leaves behind: a stage marked
  skipped or converged that never ran, a cursor advanced before its commit, a
  session committed before its cursor, a resumed candidate treated as fresh.
- Flag a timeout that abandons and restarts work that is making progress; it
  is a livelock. Safe path: bound waiting by progress or cancellation.
- Flag a live-archive mutation outside the single writer route without a
  declared authority (`declared_unguarded_write`) (P1).
