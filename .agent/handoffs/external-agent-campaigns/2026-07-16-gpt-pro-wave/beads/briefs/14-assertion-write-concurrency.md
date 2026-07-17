Title: "[beads 14] Assertion-write concurrency repair"

Job ID: `beads-14`
Result ZIP: `beads-14-assertion-write-concurrency-r01.zip`

Repair the reproduced `polylogue-41ow` operator-judgment TOCTOU race in
`user_write.py`. Concurrent assertion upserts can silently revert an accepted
operator judgment; this is durable user-tier correctness, not a cosmetic retry
problem. Trace every read-modify-write path and preserve immutable assertion,
authorization, provenance, and revision semantics.

Implement one atomic, typed production route with a deterministic concurrency
fixture that demonstrates the old lost update and proves the repaired state
contains the intended successor(s). Do not solve it with timing sleeps,
best-effort retries, broad SQLite locking that breaks ordinary writes, or a
test-only serialization wrapper. Include migration/compatibility reasoning if
the durable schema contract changes.
