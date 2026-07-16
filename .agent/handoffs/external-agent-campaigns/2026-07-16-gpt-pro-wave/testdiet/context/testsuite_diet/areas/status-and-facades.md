---
created: 2026-07-16
purpose: Execution packet for status truth and thin facade test consolidation
status: needs-detailed-inventory
project: polylogue
---

# Status and facades

CLI status, daemon status, and CLI-command status total about 4.9k nonblank
lines, with mocks/patches in 79 of 202 tests. The local payload suites did not
prevent the same archive state being described inconsistently across surfaces.

## Target test shape

Define a compact state-transition table over real archive/daemon facts: empty,
ingesting, pending convergence, partially failed, stale, repaired, healthy, and
unavailable. Materialize each state once, then assert that every surface
projects the same authority with only documented presentation differences. Add
transition laws for retry, repair, restart, and stale-state clearance.

`tests/unit/api/test_facade_contracts.py` (~5.4k lines) should retain split-tier
authority, corruption, absence, and mutation ownership. Its discovery guard,
typed-signature sweep, repeated empty shapes, and same-name forwarding families
are candidates for compression or deletion. Strict typing owns static
signatures; one generic adapter contract plus real-substrate policy cases owns
intentional forwarding.

Before estimating precisely, inventory each test by state, surface, real
production dependency, and unique failure branch. Record overlaps with query
tests and MCP rewrite tests.
