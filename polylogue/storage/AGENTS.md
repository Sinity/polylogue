# Storage

## Code Review Rules

- Do not ask for a migration or a version bump on a derived tier (`index`,
  `embeddings`, `ops`); its schema identity moves and the daemon reconverges
  it.
- Flag an enum-generated `CHECK (col IN ...)` in DDL. Safe path:
  `require_vocabulary` at the write boundary.
- Flag a writable open or commit on a live archive that bypasses the daemon
  writer route without `declared_unguarded_write` (P1).
- Flag rebuildable state (`index.db`, `ops.db`) used as the authority for a
  durable mutation (P1).
