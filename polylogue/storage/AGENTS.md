# Storage

## Code Review Rules

- Do not ask for a migration or a version bump on `index.db` or `ops.db`;
  their schema identity moves and the daemon reconverges them.
- `embeddings.db` has no schema identity. Flag an embeddings DDL change that
  leaves `EMBEDDINGS_SCHEMA_VERSION` unchanged, and treat deleting or
  replacing the tier as destructive: its vectors are purchased again.
- Flag an enum-generated `CHECK (col IN ...)` in durable DDL (`source`,
  `user`, `audit`). Safe path: `require_vocabulary` at the write boundary.
  Derived tiers may carry such checks.
- Flag a writable open or commit on a live archive that bypasses the daemon
  writer route without archive-bound custody or exact owned offline destination
  authority (P1).
- Flag rebuildable state (`index.db`, `ops.db`) used as the authority for a
  durable mutation (P1).
