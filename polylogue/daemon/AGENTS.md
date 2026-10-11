# Daemon

## Code Review Rules

- Flag a timeout that abandons and restarts work that is making progress; it
  is a livelock. Safe path: bound waiting by progress or cancellation.
- Flag a live-archive mutation outside the single writer route that does not
  hold the archive write lock, unless it writes only a file the process
  created itself, such as an inactive index generation (P1).
