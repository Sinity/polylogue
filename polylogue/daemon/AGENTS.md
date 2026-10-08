# Daemon

## Code Review Rules

- Flag a timeout that abandons and restarts work that is making progress; it
  is a livelock. Safe path: bound waiting by progress or cancellation.
- Flag a live-archive mutation outside the single writer route without a
  archive-bound custody or exact owned offline destination authority (P1).
